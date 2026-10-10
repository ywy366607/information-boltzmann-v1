"""One live BPTT learner, including context changes and checkpoint continuity."""
from collections import deque
import math

import torch

from .training import CapturedPlasticChunk, quiet_training_chunk
from .medium_health import ChunkHealthCapture
from .optimizer_storage import optimizer_state_on_host
from information_boltzmann.core.gradient_norms import stable_clip_grad_norm_


class ActiveMediumTrainer:
    """Score each target before learning it; retain every physical state variable.

    Tape truncation changes credit, never forward-state persistence. Short phase
    boundaries use the same eager objective and preserve optimizer cadence.
    """

    def __init__(self, model, optimizer, belief, *, carry_token, event_duration,
                 chunk_tokens=32, tokens_per_update=32, substeps=1, captured=None, health=None, activation_checkpointing=False, compile_event=False,
                 optimizer_state_offload=False, checkpoint_granularity='nested'):
        if min(chunk_tokens, tokens_per_update, substeps) < 1:
            raise ValueError('Positive BPTT and optimizer cadence required')
        self.model, self.optimizer, self.belief = model, optimizer, belief.detach()
        posterior = getattr(getattr(model, 'medium', None), 'structural_posterior', None)
        if posterior is not None and posterior.capacity_growth is not None:
            owned = {id(p) for group in optimizer.param_groups for p in group['params']}
            if id(posterior.mean) in owned or id(posterior.log_std) in owned:
                raise ValueError('Capacity growth owns mean and freezes log_std; Adam must exclude both')
            if compile_event:      # a captured chunk is fine: propose/apply/commit stay on the host between replays
                raise ValueError('Capacity transactions cannot use the compiled whole-event path')
        self.carry_token = int(carry_token)
        self.event_duration, self.substeps = event_duration, substeps
        self.chunk_tokens, self.tokens_per_update = chunk_tokens, tokens_per_update
        self.compile_event = compile_event
        self.activation_checkpointing = activation_checkpointing
        if checkpoint_granularity not in ('nested', 'event'):
            raise ValueError('Checkpoint granularity must be nested or event')
        if captured is not None and checkpoint_granularity != getattr(captured, 'checkpoint_granularity', 'nested'):
            raise ValueError('Captured execution owns its checkpoint policy')
        self.checkpoint_granularity = checkpoint_granularity
        if optimizer_state_offload and captured is not None:
            raise ValueError('Optimizer state staging requires eager execution')
        self.optimizer_state_offload = optimizer_state_offload
        self.captured = captured
        self.health = health
        # Scale of the riverbed sampling noise drawn per structural window; 0 evaluates credit at the posterior mean.
        self.structure_noise = 1.0
        self.structural_gravity = 0.0      # linear e-folding rate per window of the installed-capacity self-gravity; 0 = off
        self.structural_gravity_screening = 0.0     # kappa^2 (lattice Laplacian units): finite-range force
        self.structural_gravity_diffusion = 0.0     # D: weak diffusion; with the screening it selects the structure scale
        self.channel_growth = 0.0          # unified riverbed: share relaxation rate per window toward the through-flow target; 0 = off
        self.channel_mu = 2.0              # Tero feedback exponent of the channel target
        self.freeze_structure = False      # control runs: the task no longer moves the installed riverbed (OU still applies)
        self.skip_unstable_structure = False   # opt-in: skip (count) a window whose structural step has no represented descent
        self.skipped_structure_windows = 0
        self._channel_flow = None          # EMA of the through-flow snapshots (not checkpointed; rebuilt after a resume)
        self.health_capture = None if health is None else ChunkHealthCapture(model, chunk_tokens)
        if captured is not None and health is not None:
            self.health_capture = captured.health_capture
            if self.health_capture is None:
                raise ValueError('Training graph must capture fourth-pillar observations')
        self.pending = self.events = self.optimizer_updates = 0
        self.recent = deque(maxlen=128)
        self.phase_totals = {}
        # New ledgers begin at the first actually observed chunk. Legacy saves
        # have no component history; never reconstruct it from cumulative CE.
        self.loss_component_sums = {}
        self.loss_component_events = {}
        self.last_loss_components = {}
        self._parameter_norm_cache = {}
        self._structural_window_start = None
        self.optimizer.zero_grad(set_to_none=False)

    def _record_loss_components(self, loss, nll, count, components=None):
        """Record unscaled chunk means without changing the learning objective."""
        values = {'task_nll': nll, 'port_objective': loss.detach() - nll.detach(),
                  'joint_objective': loss}
        if components is not None:
            values.update(components)
        names = list(values)
        scalars = torch.stack([
            torch.as_tensor(values[name], device=loss.device, dtype=loss.dtype).detach()
            for name in names]).cpu().tolist()
        if any(not math.isfinite(value) for value in scalars):
            raise FloatingPointError('Nonfinite likelihood component')
        self.last_loss_components = dict(zip(names, scalars))
        for name, value in self.last_loss_components.items():
            self.loss_component_sums[name] = self.loss_component_sums.get(name, 0.) + count * value
            self.loss_component_events[name] = self.loss_component_events.get(name, 0) + count

    def _health_update_snapshot(self):
        """Only Adam/SGD participants with gradients can change in this step.

        A zero gradient still participates: momentum and weight decay can update
        that parameter. None gradients, frozen and unoptimized parameters need
        no full-size copy; their values still count in the parameter norm.
        """
        optimized = {id(p) for group in self.optimizer.param_groups for p in group['params']}
        posterior = getattr(getattr(self.model, 'medium', None), 'structural_posterior', None)
        if posterior is not None and posterior.capacity_growth is not None:
            optimized.add(id(posterior.mean))
        return [(p, p.detach().clone()) for p in self.model.parameters()
                if p.requires_grad and p.grad is not None and id(p) in optimized]

    @torch.no_grad()
    def _health_update_norms(self, before):
        """Keep the historical full-model denominator, caching unchanged norms."""
        parameters = list(self.model.parameters())
        norms = []
        for parameter in parameters:
            # Ordinary optimizer steps and load/copy operations bump _version.
            # Storage/device changes also invalidate a cached immutable term.
            version = (parameter._version, parameter.data_ptr(), parameter.shape,
                       parameter.dtype, parameter.device)
            cached = self._parameter_norm_cache.get(id(parameter))
            if cached is None or cached[0] != version:
                norm = torch.linalg.vector_norm(parameter).double().square()
                self._parameter_norm_cache[id(parameter)] = (version, norm)
            norms.append(self._parameter_norm_cache[id(parameter)][1])
        parameter_norm = torch.stack(norms).sum().sqrt()
        changes = []
        for parameter, old in before:
            # Keep the actual-update metric, but bound its difference workspace
            # to4MiB instead of allocating another148MiB vocabulary matrix.
            current, previous = parameter.reshape(-1), old.reshape(-1)
            terms = [torch.linalg.vector_norm(current[left:left + 1048576]
                                              - previous[left:left + 1048576]).double().square()
                     for left in range(0, current.numel(), 1048576)]
            changes.append(torch.stack(terms).sum())
        update_norm = (torch.stack(changes).sum().sqrt() if changes
                       else parameter_norm.new_zeros(()))
        return update_norm, parameter_norm

    def consume(self, targets, *, phase, prior_nll):
        targets = targets.reshape(-1)
        phases = [phase] * len(targets) if isinstance(phase, str) else list(phase)
        if len(phases) != len(targets):
            raise ValueError('Every scored target requires its actual phase')
        result = []
        cursor = 0
        while cursor < len(targets):
            count = min(self.chunk_tokens, self.tokens_per_update - self.pending,
                        len(targets) - cursor)
            scored = targets[cursor:cursor + count]
            observed = torch.cat((scored.new_tensor([self.carry_token]), scored[:-1]))[None]
            posterior = getattr(getattr(self.model, 'medium', None), 'structural_posterior', None)
            if posterior is not None:
                if self.compile_event:
                    raise ValueError('Structural evidence windows use eager chunk orchestration')
                if not bool(posterior.window_active):
                    if self.pending:
                        raise ValueError('Pending gradients require the saved structural sample')
                    posterior.begin_window(self.structure_noise * torch.randn_like(posterior.mean))
                    self._structural_window_start = float(self.belief.medium.elapsed.item())
                self.model._structural_window_events = self.tokens_per_update
            if self.captured is not None and count == self.chunk_tokens:
                loss, evolved, nll = self.captured.backward(observed, scored[None], self.belief)
                token_nll = self.captured.token_nll.detach().clone()
                components = getattr(self.captured, 'loss_components', None)
            else:
                # Adam moments are idle throughout physics and credit replay.
                # Restore them before clipping, diagnostics and the actual step.
                with optimizer_state_on_host(self.optimizer, enabled=self.optimizer_state_offload,
                                             release_cached_memory=self.optimizer_state_offload):
                    chunk = quiet_training_chunk(
                        self.model, observed, scored[None], self.belief,
                        event_duration=self.event_duration, substeps=self.substeps,
                        return_token_nll=True, return_loss_components=True,
                        health_capture=self.health_capture,
                        activation_checkpointing=self.activation_checkpointing, compile_event=self.compile_event,
                        checkpoint_granularity=self.checkpoint_granularity)
                    loss, evolved, nll, token_nll = chunk[:4]
                    components = chunk[4] if len(chunk) > 4 else None
                    # Outgoing values persist across the existing BPTT boundary.
                    # The loss retains all declared event and interval credit.
                    evolved = evolved.detach()
                    del chunk
                    (loss * (count / self.tokens_per_update)).backward()
                    if getattr(self.model, 'deferred_writer_credit', False):
                        self.model.flush_deferred_credit()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite joint likelihood')
            self._record_loss_components(loss, nll, count, components)
            values = token_nll.detach().cpu().reshape(-1).tolist()
            labels = scored.detach().cpu().tolist()
            reference = [float(prior_nll[token]) for token in labels]
            if any(not math.isfinite(value) for value in values):
                raise FloatingPointError('Nonfinite pre-update score')
            # Commit scores and full state before parameters can learn targets.
            self.belief = evolved.detach()
            observations = (iter(self.health_capture.observations(count))
                            if self.health is not None else None)
            previous = self.carry_token
            for target, nll, prior, event_phase in zip(
                    labels, values, reference, phases[cursor:cursor + count]):
                if self.health is not None:
                    self.health.record_event(nll, novel=event_phase in ('train_first_pass', 'fresh_B'),
                                             phase=event_phase, observation=next(observations))
                row = (previous, target, nll, prior, event_phase)
                self.recent.append(row)
                result.append(row)
                previous = target
                totals = self.phase_totals.setdefault(event_phase,
                    {'events': 0, 'nll_sum': 0., 'prior_sum': 0.})
                totals['events'] += 1
                totals['nll_sum'] += nll
                totals['prior_sum'] += prior
            self.carry_token = previous
            self.events += count
            self.pending += count
            if self.pending == self.tokens_per_update:
                growth = None if posterior is None else posterior.capacity_growth
                if growth is None:
                    self.last_gradient_norm = float(stable_clip_grad_norm_(
                        self.model.parameters(), 1.0, error_if_nonfinite=True))
                before = self._health_update_snapshot() if self.health is not None else None
                proposal = None
                if posterior is not None:
                    if self._structural_window_start is None:
                        raise ValueError('Missing physical start of the structural evidence window')
                    with torch.no_grad():
                        prepared = self.model.medium.prepare_evolution()
                        maintenance = posterior.maintenance(prepared.structural_allocation,
                                                             1 / math.prod(self.model.medium.shape))
                    duration = float(self.belief.medium.elapsed.item()) - self._structural_window_start
                    posterior.record_window_evidence(self.tokens_per_update, duration, maintenance)
                    del prepared, maintenance
                    if growth is not None and not self.freeze_structure:
                        basis = self.model.medium.material.basis(self.model.medium.coordinates)
                        try:
                            proposal = growth.propose(posterior, basis,
                                                      1 / math.prod(self.model.medium.shape))
                        except FloatingPointError:
                            if not self.skip_unstable_structure:
                                raise
                            # A heavy-tailed credit window with no represented descent inside the KL allowance:
                            # skip this window's structural step instead of ending the individual.
                            proposal = None
                            self.skipped_structure_windows += 1
                        del basis
                    if growth is not None and proposal is None:
                        ordinary = [p for group in self.optimizer.param_groups for p in group['params']]
                        self.last_gradient_norm = float(stable_clip_grad_norm_(ordinary, 1.0, error_if_nonfinite=True))
                    if proposal is not None:
                        # Variance is frozen under this rule. Validate the
                        # candidate OU/dual transition before Adam can mutate
                        # any ordinary parameter. Fatal optimizer failures
                        # still resume from a completed checkpoint.
                        posterior.validate_window_commit(
                            posterior_mean=proposal['mean'],
                            dual_learning_rate=getattr(self.model, 'structure_dual_learning_rate', None))
                        # Independent geometric trust region owns structural
                        # credit; the ordinary clip owns only Adam's parameters.
                        ordinary = [p for group in self.optimizer.param_groups for p in group['params']]
                        norm = float(stable_clip_grad_norm_(ordinary, 1.0, error_if_nonfinite=True))
                        self.last_gradient_norm = math.hypot(norm, proposal['gradient_norm'])
                if self.optimizer_state_offload and self.belief.medium.field.is_cuda:
                    # Restoring moments and taking the health snapshot can
                    # leave fragmented replay blocks. Adam's largest temporary
                    # must get one contiguous vocabulary-sized allocation.
                    torch.cuda.empty_cache()
                self.optimizer.step()
                if posterior is not None:
                    if growth is not None and proposal is not None:
                        growth.apply(posterior, proposal)
                    posterior.commit_window(dual_learning_rate=getattr(
                        self.model, 'structure_dual_learning_rate', None))
                    if self.structural_gravity:
                        self.model.medium.structural_gravity_step(self.structural_gravity, self.structural_gravity_screening,
                                                                   self.structural_gravity_diffusion)
                    if self.channel_growth:
                        # One snapshot of the through-flow per window, time-averaged across windows (EMA 0.9).
                        flow = self.model.medium.through_flow(self.belief.medium)
                        self._channel_flow = flow if self._channel_flow is None else 0.9 * self._channel_flow + 0.1 * flow
                        self.model.medium.channel_growth_step(self._channel_flow, self.channel_growth, self.channel_mu)
                    self._structural_window_start = None
                if before is not None:
                    update_norm, parameter_norm = self._health_update_norms(before)
                    self.health.record_update(self.last_gradient_norm, float(update_norm), float(parameter_norm))
                    del before
                self.optimizer.zero_grad(set_to_none=False)
                if growth is not None:
                    # This parameter is deliberately absent from Adam.
                    if self.captured is None:
                        posterior.mean.grad = None
                    else:                       # a captured graph accumulates into this fixed address
                        posterior.mean.grad.zero_()
                    del proposal
                self.optimizer_updates += 1
                self.pending = 0
                if self.captured is None:
                    # Every eager chunk differentiates the same decoder CE.
                    # Its completed zero buffer carries no pending credit and
                    # is recreated by the next backward before Adam can run.
                    # Other parameters retain their None/zero distinction;
                    # captured execution retains every fixed gradient address.
                    decoder = getattr(self.model, 'decoder', None)
                    weight = getattr(decoder, 'weight', None)
                    if weight is not None:
                        weight.grad = None
            cursor += count
            # Completed window graphs must not remain live while the next
            # window stages optimizer state and returns idle allocator segments.
            del loss, nll, token_nll, components
        return result

    def summary(self):
        elapsed = getattr(getattr(self.belief, 'medium', None), 'elapsed', None)
        result = {'events': self.events, 'optimizer_updates': self.optimizer_updates,
                'physical_elapsed': None if elapsed is None else float(elapsed.mean().detach()),
                'pending_gradient_events': self.pending,
                'loss_components': {**{name: value / self.loss_component_events[name]
                                       for name, value in self.loss_component_sums.items()},
                                    'component_events': dict(self.loss_component_events)},
                'last_loss_components': dict(self.last_loss_components),
                'health': None if self.health is None else self.health.summary(),
                'phases': {phase: {**values,
                    'nll': values['nll_sum'] / values['events'],
                    'fixed_unigram_nll': values['prior_sum'] / values['events'],
                    'gain': (values['prior_sum'] - values['nll_sum']) / values['events']}
                    for phase, values in self.phase_totals.items()}}
        posterior = getattr(getattr(self.model, 'medium', None), 'structural_posterior', None)
        if posterior is not None and posterior.capacity_growth is not None:
            result['capacity_growth'] = posterior.capacity_growth.summary()
        return result

    def state_dict(self):
        if getattr(self.model, 'deferred_writer_credit', False):
            self.model.deferred_credit.assert_flushed()
        return {'health': None if self.health is None else self.health.state_dict(),
                'version': 1, 'events': self.events, 'optimizer_updates': self.optimizer_updates,
                'pending': self.pending, 'carry_token': self.carry_token,
                'structural_window_start': self._structural_window_start,
                'chunk_tokens': self.chunk_tokens, 'tokens_per_update': self.tokens_per_update,
                'recent': list(self.recent), 'phase_totals': self.phase_totals,
                'loss_component_sums': dict(self.loss_component_sums),
                'loss_component_events': dict(self.loss_component_events),
                'last_loss_components': dict(self.last_loss_components),
                'pending_gradients': {n: None if p.grad is None else p.grad.detach().cpu().clone()
                                      for n, p in self.model.named_parameters()}}

    def load_state_dict(self, saved):
        if (saved['version'] != 1 or saved['chunk_tokens'] != self.chunk_tokens or
                saved['tokens_per_update'] != self.tokens_per_update or
                not 0 <= saved['pending'] < self.tokens_per_update):
            raise ValueError('Resume changes learning cadence')
        if self.captured is not None:
            names = {id(p): name for name, p in self.model.named_parameters()}
            if any(saved['pending_gradients'][names[id(p)]] is None
                   for p in self.captured.active_parameters):
                raise ValueError('Captured storage cannot restore None-gradient semantics; use eager restore')
        if self.health is not None:
            if saved.get('health') is None:
                raise ValueError('Missing fourth-pillar continuation state')
            self.health.load_state_dict(saved['health'])
        for key in ('events', 'optimizer_updates', 'pending', 'carry_token', 'phase_totals'):
            setattr(self, key, saved[key])
        self.loss_component_sums = dict(saved.get('loss_component_sums', {}))
        self.loss_component_events = dict(saved.get('loss_component_events', {}))
        self.last_loss_components = dict(saved.get('last_loss_components', {}))
        self._structural_window_start = saved.get('structural_window_start')
        self._parameter_norm_cache.clear()
        self.recent = deque(saved['recent'], maxlen=128)
        for name, parameter in self.model.named_parameters():
            gradient = saved['pending_gradients'].get(name)
            # The decoder participates in every eager CE window. At a completed
            # update its saved zero gradient carries no pending credit. Let the
            # first backward own that large buffer instead of allocating both a
            # restored vocabulary gradient and the compiled backward's output.
            # Captured graphs retain fixed addresses; nonzero/pending credit is
            # restored exactly through the normal path below.
            if (self.captured is None and self.pending == 0 and name == 'decoder.weight'
                    and gradient is not None and torch.count_nonzero(gradient).item() == 0):
                parameter.grad = None
                continue
            if gradient is None:
                # None skips AdamW momentum/decay, while a zero tensor does not.
                parameter.grad = None
            elif parameter.grad is None:
                if (self.captured is not None and
                        any(parameter is p for p in self.captured.active_parameters)):
                    raise ValueError('Captured gradient storage changed')
                parameter.grad = gradient.to(parameter).clone()
            else:
                parameter.grad.copy_(gradient.to(parameter))
