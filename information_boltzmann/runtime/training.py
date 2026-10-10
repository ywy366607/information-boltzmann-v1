"""Quiet, differentiable chunk execution and fixed-shape CUDA training capture."""
from __future__ import annotations

from dataclasses import replace
import gc
import math

import torch
from torch.nn import functional as F

from ..core.plastic_ports import PlasticBelief, PlasticMediumPorts3D
from ..core.state_checkpoint import checkpoint_state, checkpoint_state_vjp
from ..core.temporal_probes import TemporalProbeState


def training_event(model, current, token, table, prepared, event_duration, substeps, diagnostics,
                   activation_checkpointing=False, read_feature=True, schedule_box=None):
    """Tensor event shared by eager and compiled execution, with detached audit outputs."""
    duration = model.event_time(current, event_duration)
    written, full_write = model.assimilate(current, token, token_features=table,
        diagnostics=diagnostics)
    if schedule_box is not None and model.solver_max_step is not None:
        from ..core.intrinsic_time import EvolutionSchedule
        if 'schedule' not in schedule_box and getattr(model, 'fixed_schedule', None) is not None:
            schedule_box['schedule'] = model.fixed_schedule      # fixed clock: planned once, no device read
        if 'schedule' not in schedule_box:
            schedule_box['schedule'] = EvolutionSchedule.for_duration(duration,
                solver_max_step=model.effective_solver_max_step, observer_max_step=model.observer_max_step,
                max_steps=model.max_evolution_steps)
        outgoing, full_evolution, motion = model.advance_interval(written, duration,
            substeps=substeps, prepared=prepared, diagnostics=diagnostics, return_motion=True,
            activation_checkpointing=activation_checkpointing, schedule=schedule_box['schedule'])
    else:
        outgoing, full_evolution, motion = model.advance(written, duration,
            substeps=substeps, prepared=prepared, diagnostics=diagnostics, return_motion=True,
            activation_checkpointing=activation_checkpointing)
    # Writer auxiliary credit needs the complete recurrent trajectory, but its
    # unused expression feature cannot affect a later writer or event clock.
    # Temporal/RHS sampling remains part of advance(), even when this is omitted.
    feature = (model.read(outgoing, decode=False, prepared=prepared,
                          diagnostics=False, motion=motion)[0] if read_feature else None)
    write = {'_write_free_energy': full_write['_write_free_energy']}
    for key in ('port_nll', 'write_action_kl'):
        if key in full_write:
            write[key] = full_write[key].detach()
    evolution = {}
    if diagnostics:
        write.update({key: full_write[key].detach() for key in ('incident_energy', 'reflected_energy')})
        keys = ('bath_out_energy', 'response_source_work', 'transport_energy_change',
                'transport_spatial_energy_change', 'collision_energy_change',
                'collision_spatial_energy_change', 'bath_energy_change', 'bath_spatial_energy_change',
                'junction_energy_change', 'junction_spatial_energy_change')
        evolution = {key: full_evolution[key].detach() for key in keys if key in full_evolution}
    return written, outgoing, write, evolution, feature


_compiled_training_event = None


def compiled_training_event(*args):
    global _compiled_training_event
    if args[0].solver_max_step is not None:
        raise ValueError('Adaptive interval planning uses eager orchestration; fuse medium kernels instead')
    if _compiled_training_event is None:
        import torch._inductor.config as config
        config.compile_threads = 1
        config.use_static_cuda_launcher = False
        _compiled_training_event = torch.compile(training_event, fullgraph=True, dynamic=False)
    return _compiled_training_event(*args)


class _WriterAuxiliaryGradient(torch.autograd.Function):
    """First-order block VJP whose auxiliary gradients stay on CPU until used."""

    @staticmethod
    def forward(ctx, value, cpu_gradients, *parameters):
        ctx.cpu_gradients = cpu_gradients
        ctx.parameter_specs = tuple((p.device, p.dtype) for p in parameters)
        return value.clone()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, scale):
        gradients = []
        for gradient, (device, dtype) in zip(ctx.cpu_gradients, ctx.parameter_specs):
            # Do not keep another persistent CUDA copy throughout the main
            # forward. Each returned tensor is consumed by normal accumulation.
            # copy=True also keeps repeated/scaled CPU backwards immutable.
            local = None if gradient is None else gradient.to(device=device, dtype=dtype, copy=True)
            gradients.append(None if local is None else local.mul_(scale))
        return (None, None, *gradients)


def _writer_auxiliary_gradients(model, ids, initial, event_duration, substeps,
                                activation_checkpointing, compile_event, parameters,
                                checkpoint_granularity='nested', read_feature=False):
    """Replay the complete chunk, differentiate once, then release its graph.

    The replay keeps every recurrent state and interval derivative. It does not
    commit belief, health observations, posterior evidence or RNG advancement.
    This costs one additional complete port-only forward/backward, without a
    decoder pass, instead of retaining a second checkpoint recomputation graph
    alongside the task graph. Only detached writer gradients leave this helper.
    """
    devices = [initial.medium.field.device.index] if initial.medium.field.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        table = F.normalize(model.source.embedding.weight, dim=-1)
        # This objective updates the writer block only. Coefficients remain
        # numerically identical constants while state cotangents still traverse
        # every physical interval. A trainable embedding keeps its table graph.
        with torch.no_grad():
            prepared = model.medium.prepare_evolution()
        belief, objectives = initial, []
        replay_parameters = parameters

        def event_step(current, token, shared_table, coefficients, physical_duration, schedule_box):
            operation = compiled_training_event if compile_event else training_event
            return operation(model, current, token, shared_table, coefficients, physical_duration,
                             substeps, False,
                             activation_checkpointing and checkpoint_granularity == 'nested',
                             read_feature, **({} if compile_event else {'schedule_box': schedule_box}))

        for token in ids.unbind(1):
            schedule_box = {}
            def scheduled_step(current, token, shared_table, coefficients, physical_duration,
                               box=schedule_box):
                return event_step(current, token, shared_table, coefficients, physical_duration, box)
            if activation_checkpointing:
                result = checkpoint_state_vjp(scheduled_step, belief, token, table, prepared,
                                              event_duration, parameters=replay_parameters)
            else:
                result = scheduled_step(belief, token, table, prepared, event_duration)
            belief = result[1]
            objectives.append(result[2]['_write_free_energy'])
            del result
        objective = torch.stack(objectives).mean()
        # Auxiliary credit ends at the writer losses. Unused final read/state
        # outputs otherwise keep side-branch checkpoint holders alive while
        # backward fills their recomputation caches across the whole window.
        del belief, objectives
        if not objective.requires_grad:
            return tuple(None for _ in parameters)
        # Normal learner chunks enter at detached BPTT boundaries, so no graph
        # needs retention. Preserve an explicitly supplied live prefix, if any,
        # because the later main forward must still differentiate that prefix.
        shared_prefix = any(value.grad_fn is not None for value in belief_tensors(initial))
        shared_prefix |= isinstance(event_duration, torch.Tensor) and event_duration.grad_fn is not None
        gradients = torch.autograd.grad(objective, parameters,
            retain_graph=shared_prefix, allow_unused=True)
        return tuple(None if gradient is None else gradient.detach().to('cpu', copy=True)
                     for gradient in gradients)


def quiet_training_chunk(model: PlasticMediumPorts3D, ids: torch.Tensor,
                         targets: torch.Tensor, belief: PlasticBelief, *,
                         event_duration: float | torch.Tensor, substeps: int = 1,
                         health_capture=None, return_token_nll: bool = False,
                         return_loss_components: bool = False,
                         activation_checkpointing: bool = False, compile_event: bool = False,
                         checkpoint_granularity: str = 'nested'):
    """The same CE plus W4 objective, reusing parameter graphs within a chunk.

    Structural candidates route exact writer credit through a separate complete
    replay or two cotangent lanes on one shared primal. Both retain the entire
    declared BPTT chunk and physical intervals, with the same parameter scope.
    """
    if ids.ndim != 2 or ids.shape != targets.shape or ids.shape[1] < 1:
        raise ValueError('Nonempty matching [B,L] observations and targets required')
    if checkpoint_granularity not in ('nested', 'event'):
        raise ValueError('Checkpoint granularity must be nested or event')
    posterior = model.medium.structural_posterior
    if posterior is not None and ids.shape[0] != 1:
        raise ValueError('Structural evidence requires one persistent individual, B=1')
    if posterior is not None and not posterior.window_is_active():
        raise ValueError('Learner must begin one structural evidence window before execution')
    local_parameters, local_grads = (), None
    shared_credit = False
    if posterior is not None and torch.is_grad_enabled():
        local_parameters = tuple(p for name, p in model.learning_named_parameters()
                                 if name.startswith(('source.', 'write_agent.')))
        if local_parameters:
            # A shared primal trajectory has two independent recurrent
            # cotangent lanes. Live external prefixes retain the older general
            # replay contract; production enters at detached BPTT boundaries.
            shared_credit = (getattr(model, 'shared_credit_execution', False)
                and activation_checkpointing and checkpoint_granularity == 'event'
                and not any(x.grad_fn is not None for x in belief_tensors(belief))
                and not (isinstance(event_duration, torch.Tensor)
                         and event_duration.requires_grad))
            if not shared_credit:
                local_grads = _writer_auxiliary_gradients(model, ids, belief, event_duration,
                    substeps, activation_checkpointing, compile_event, local_parameters,
                    checkpoint_granularity)
            # Drop any Python checkpoint cycles before constructing the main
            # graph; allocator blocks can then be reused without empty_cache().
            gc.collect()
    table = F.normalize(model.source.embedding.weight, dim=-1)
    if getattr(model, 'deferred_writer_credit', False) and not shared_credit:
        raise ValueError('Deferred port credit requires a shared-credit detached event checkpoint window')
    prepared = model.medium.prepare_evolution()
    replay_parameters = tuple(p for _, p in model.learning_named_parameters()
                              if p.requires_grad) if posterior is not None else ()
    features, objectives, port_nlls, policy_kls = [], [], [], []
    auxiliary_state, auxiliary_objectives = belief.detach(), []
    for index in range(ids.shape[1]):
        if health_capture is not None and hasattr(health_capture, 'select'):
            health_capture.select(index)
        incoming = belief
        schedule_box = {}
        def event_step(current, token, shared_table, coefficients, physical_duration,
                       box=schedule_box):
            operation = compiled_training_event if compile_event else training_event
            return operation(model, current, token, shared_table, coefficients, physical_duration,
                             substeps, health_capture is not None,
                             activation_checkpointing and checkpoint_granularity == 'nested',
                             **({} if compile_event else {'schedule_box': box}))
        if activation_checkpointing:
            arguments = (belief, ids[:, index], table, prepared, event_duration)
            if shared_credit:
                from ..core.shared_credit import checkpoint_shared_credit
                result, (auxiliary_state, auxiliary_objective) = checkpoint_shared_credit(
                    event_step, belief, auxiliary_state, *arguments[1:],
                    parameters=replay_parameters, auxiliary_parameters=local_parameters,
                    shared_auxiliary_tensors=(table,),
                    history_offload=model.credit_history_offload)
                written, belief, info, evolution_info, feature = result
                auxiliary_objectives.append(auxiliary_objective)
                del result
            elif posterior is not None:
                written, belief, info, evolution_info, feature = checkpoint_state_vjp(
                    event_step, *arguments, parameters=replay_parameters)
            else:
                written, belief, info, evolution_info, feature = checkpoint_state(
                    event_step, *arguments)
        else:
            written, belief, info, evolution_info, feature = event_step(
                belief, ids[:, index], table, prepared, event_duration)
        objectives.append(info['_write_free_energy'])
        if 'port_nll' in info:
            port_nlls.append(info['port_nll'])
        if 'write_action_kl' in info:
            policy_kls.append(info['write_action_kl'])
        if health_capture is not None:
            health_capture.record(model, incoming, written, belief, info, evolution_info, feature)
        features.append(feature)
    expression = torch.stack(features, 1)
    logits = model.decode(expression)
    if health_capture is not None:
        health_capture.record_decode(model, expression, logits)
    token_nll = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                               reduction='none').reshape_as(targets)
    nll = token_nll.mean()
    port_objective = torch.stack(objectives).mean()
    main_objective = nll
    structure_kl = structure_maintenance = None
    if posterior is not None:
        structure_maintenance = posterior.maintenance(prepared.structural_allocation,
                                                       1 / math.prod(model.medium.shape))
        structure_kl = posterior.kl_divergence()
        # The active learner scales each complete chunk objective by count/L.
        # Use L (not the chunk length) so total structural evidence is counted once.
        window_events = getattr(model, '_structural_window_events', ids.shape[1])
        main_objective = posterior.objective(nll, structure_maintenance, window_events)
        if shared_credit:
            auxiliary = torch.stack(auxiliary_objectives).mean()
        elif local_grads is not None and port_objective.requires_grad:
            # Exact block objective: auxiliary credit trains its local writer.
            # It supplies no duplicate likelihood evidence to q or the medium.
            auxiliary = _WriterAuxiliaryGradient.apply(port_objective.detach(),
                                                        local_grads, *local_parameters)
        else:
            auxiliary = port_objective.detach()
    else:
        auxiliary = port_objective
    joint = main_objective + auxiliary
    result = (joint, belief, nll)
    if return_token_nll:
        result = (*result, token_nll)
    if return_loss_components:
        components = {'task_nll': nll.detach(), 'port_objective': port_objective.detach(),
                      'joint_objective': joint.detach()}
        if len(port_nlls) == ids.shape[1]:
            components['port_nll'] = torch.stack(port_nlls).mean().detach()
        if len(policy_kls) == ids.shape[1]:
            components['write_action_kl'] = torch.stack(policy_kls).mean().detach()
        if posterior is not None:
            components.update(structural_kl=structure_kl.detach(),
                              structural_maintenance=structure_maintenance.detach(),
                              structural_objective=main_objective.detach())
        result = (*result, components)
    return result


def belief_tensors(belief: PlasticBelief):
    """Every persistent state component, including clocks and receptor gates."""
    medium = belief.medium
    temporal = (() if belief.temporal is None else
                (belief.temporal.value, belief.temporal.elapsed))
    return tuple(x for x in (medium.field, *medium.flux, medium.elapsed,
                             medium.conduction, medium.receptors, medium.transmission, belief.precision)
                 if x is not None) + temporal


def clone_belief(belief: PlasticBelief):
    medium = belief.medium
    clone = lambda x: None if x is None else x.detach().clone()
    return PlasticBelief(replace(medium, field=clone(medium.field),
                                 flux=tuple(clone(x) for x in medium.flux),
                                 elapsed=clone(medium.elapsed),
                                 conduction=clone(medium.conduction),
                                 receptors=clone(medium.receptors),
                                 transmission=clone(medium.transmission)), clone(belief.precision),
                         None if belief.temporal is None else TemporalProbeState(
                             clone(belief.temporal.value), clone(belief.temporal.elapsed)))


class CapturedPlasticChunk:
    """Forward AND backward graph with persistent input/output state buffers.

    Grads are allocated during side-stream warmup, then kept at stable addresses.
    Capture therefore records accumulation into existing grads on every replay.
    Call zero_grad() once per optimizer update, never between its BPTT chunks.
    Model parameters retain their addresses and are read afresh on replay, so
    optimizer updates require no recapture. This is a training graph, distinct
    from the runtime's inference-only CUDA graph.
    """

    def __init__(self, model: PlasticMediumPorts3D, ids: torch.Tensor,
                 targets: torch.Tensor, belief: PlasticBelief, *,
                 event_duration: float, substeps: int = 1, loss_scale: float = 1.0,
                 health_capture=None, activation_checkpointing: bool = False, compile_event: bool = False,
                 checkpoint_granularity: str = 'nested'):
        posterior = model.medium.structural_posterior
        if model.solver_max_step is not None and model.intrinsic_time is not None:
            raise ValueError('An adaptive clock plans its interval on the host; capture needs a fixed clock')
        if posterior is not None and compile_event:
            raise ValueError('Structural evidence windows cannot use the compiled whole-event path')
        if not belief.medium.field.is_cuda:
            raise ValueError('CUDA training capture requires CUDA state')
        if not math.isfinite(event_duration) or event_duration <= 0 or substeps < 1:
            raise ValueError('Positive duration and solver resolution required')
        if not math.isfinite(loss_scale) or loss_scale <= 0:
            raise ValueError('Positive finite loss scaling required')
        self.model = model
        self.health_capture = health_capture
        self.checkpoint_granularity = checkpoint_granularity
        self.ids, self.targets = ids.clone(), targets.clone()
        self.input = clone_belief(belief)
        # Keep the physical clock in FP64; the dynamics casts locally to field
        # precision, just like the eager float-duration path.
        self.duration = torch.tensor(event_duration, device=belief.medium.field.device,
                                     dtype=torch.float64)

        def backward():
            loss, output, nll, token_nll, components = quiet_training_chunk(
                model, self.ids, self.targets, self.input,
                event_duration=self.duration, substeps=substeps, return_token_nll=True,
                return_loss_components=True,
                health_capture=health_capture, activation_checkpointing=activation_checkpointing, compile_event=compile_event,
                checkpoint_granularity=checkpoint_granularity)
            (loss * loss_scale).backward()
            return loss, output, nll, token_nll, components

        if model.solver_max_step is not None:
            # Fixed clock: plan the solver/observer schedule once on the host, never inside the graph.
            from ..core.intrinsic_time import EvolutionSchedule
            model.fixed_schedule = EvolutionSchedule.for_duration(
                event_duration, solver_max_step=model.effective_solver_max_step,
                observer_max_step=model.observer_max_step, max_steps=model.max_evolution_steps)
        if posterior is not None:
            # One structural window with a static sampling buffer; real windows rewrite the buffers in place.
            if bool(posterior.window_active):
                raise ValueError('Capture must be built before any structural window begins')
            posterior.begin_window(torch.zeros_like(posterior.mean))
            model._structural_window_events = int(round(ids.shape[1] / loss_scale))

        current = torch.cuda.current_stream()
        warm = torch.cuda.Stream()
        warm.wait_stream(current)
        with torch.cuda.stream(warm):
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                backward()
        current.wait_stream(warm)
        torch.cuda.synchronize()
        self.active_parameters = tuple(p for p in model.parameters() if p.grad is not None)
        self.zero_grad()
        torch.cuda.empty_cache()
        self.graph = torch.cuda.CUDAGraph()
        if posterior is not None:
            posterior.capture_window_active = True
        try:
            with torch.cuda.graph(self.graph):
                self.loss, self.output, self.nll, self.token_nll, self.loss_components = backward()
        finally:
            if posterior is not None:
                posterior.capture_window_active = False
        self.zero_grad()
        if posterior is not None:
            with torch.no_grad():
                posterior.window_active.fill_(False)
                posterior.window_evidence_recorded.fill_(False)

    def zero_grad(self):
        for parameter in self.active_parameters:
            if parameter.grad is None:
                raise RuntimeError('Captured gradient storage was removed; recapture is required')
            parameter.grad.zero_()

    def backward(self, ids: torch.Tensor, targets: torch.Tensor, belief: PlasticBelief):
        if ids.shape != self.ids.shape or targets.shape != self.targets.shape:
            raise ValueError('Captured chunk requires fixed token shape')
        with torch.no_grad():
            self.ids.copy_(ids)
            self.targets.copy_(targets)
            incoming, static = belief_tensors(belief), belief_tensors(self.input)
            if len(incoming) != len(static):
                raise ValueError('Persistent state structure changed')
            for destination, source in zip(static, incoming):
                destination.copy_(source)
        self.graph.replay()
        # Own the returned values, rather than aliasing the next replay's buffers.
        return self.loss.detach().clone(), clone_belief(self.output), self.nll.detach().clone()
