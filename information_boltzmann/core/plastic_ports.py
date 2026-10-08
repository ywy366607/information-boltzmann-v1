"""Existing W4/read agents attached to the independent plastic-wave medium.

This is a runnable causal observation model, not a new training result. Its full
belief includes edge responses: an f-only trainer/checkpoint cannot continue it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os

import torch
from torch import nn
from torch.nn import functional as F

from .plastic_medium import EvolutionCoefficients, MediumState, PlasticMedium3D
from .readout_probes import PredictivePhysicalReadAgent
from .state_checkpoint import checkpoint_state
from .temporal_probes import TemporalProbeReadout, TemporalProbeState, sample_compact_probes
from .torus3d import FullRankTorusWrite, PredictiveImpedanceWriteAgent


@dataclass(frozen=True)
class PlasticBelief:
    medium: MediumState
    precision: torch.Tensor
    temporal: TemporalProbeState | None = None

    def detach(self) -> "PlasticBelief":
        return PlasticBelief(self.medium.detach(), self.precision.detach(),
                             None if self.temporal is None else self.temporal.detach())


class PlasticMediumPorts3D(nn.Module):
    """One causal graph: W4 write -> local medium -> learned physical read.

    ``assimilate``, ``advance``, and ``read`` are separate operations. Fast state
    continues through observations and parameter updates. Training uses actual
    targets only in the likelihood, never in the pre-target prediction path.
    """

    def __init__(self, vocab_size: int = 50257, shape=(8, 8, 4), channels: int = 128,
                 material_width: int = 8, hidden: int = 64, collision_layers: int = 2,
                 port_modes: int = 8, heads: int = 4, queries: int = 4,
                 adaptive_conduction: bool = True, plasticity_time_reference: float = 1.0,
                 bath_type: str = 'quadratic', response_time_reference: float = 1.0,
                 voltage_reference: float = 1.0, capacitance_reference: float = 1.0,
                 write_exchange: str = 'contact_mode', port_scope: str = 'compact',
                 write_port_radius=None, read_port_radius=None,
                 activity_adaptation: bool = False, short_term_plasticity: bool = False,
                 medium_execution: str = 'native', port_execution: str = 'native',
                 pre_decoder_norm: bool = True, read_mode: str = 'instantaneous',
                 temporal_rates=None, temporal_frequencies=None, anisotropic_transport: bool = False,
                 transport_capacity_budget: float | None = None,
                 material_reference_shape=(8, 8, 4), temporal_time_reference: float = 1.0,
                 intrinsic_time_reference: float | None = None,
                 intrinsic_max_duration: float | None = None,
                 solver_max_step: float | None = None, observer_max_step: float | None = None,
                 max_evolution_steps: int = 4096,
                 write_aperture_budget: float | None = None,
                 read_aperture_budget: float | None = None,
                 structure_options: dict | None = None,
                 read_key_execution: str = 'dense'):
        super().__init__()
        if port_execution not in ('native', 'fused'):
            raise ValueError('port_execution must be native or fused')
        self.port_execution = port_execution
        self.read_key_execution = read_key_execution
        if port_scope not in ('global', 'compact'):
            raise ValueError('port_scope must be global or compact')
        self.port_scope = port_scope
        if read_mode not in ('instantaneous', 'dynamic', 'temporal'):
            raise ValueError('read_mode must be instantaneous, dynamic or temporal')
        if read_mode in ('dynamic', 'temporal') and port_scope != 'compact':
            raise ValueError('Dynamic read requires finite compact ports')
        self.read_mode = read_mode
        self.read_time_reference = float(response_time_reference)
        if not math.isfinite(self.read_time_reference) or self.read_time_reference <= 0:
            raise ValueError('Read physical time reference must be finite and positive')
        self.activity_adaptation = bool(activity_adaptation)
        self.short_term_plasticity = bool(short_term_plasticity)
        self.solver_max_step = solver_max_step
        self.observer_max_step = observer_max_step
        self.max_evolution_steps = int(max_evolution_steps)
        if (solver_max_step is None) != (observer_max_step is None):
            raise ValueError('Declare both solver and observer sampling steps')
        if solver_max_step is not None and (not math.isfinite(solver_max_step)
                or solver_max_step <= 0 or not math.isfinite(observer_max_step)
                or observer_max_step <= 0 or max_evolution_steps < 1):
            raise ValueError('Positive finite numerical steps and an execution budget required')
        if intrinsic_time_reference is not None and solver_max_step is None:
            raise ValueError('Intrinsic time requires separately declared solver/observer resolution')
        if intrinsic_time_reference is not None and port_scope != 'compact':
            raise ValueError('Intrinsic time reads only declared compact apertures')
        self.intrinsic_time = None
        if intrinsic_time_reference is not None:
            from .intrinsic_time import IntrinsicTimePolicy
            self.intrinsic_time = IntrinsicTimePolicy(intrinsic_time_reference,
                max_duration=intrinsic_max_duration)
        if activity_adaptation and (bath_type != 'conductance' or port_scope != 'compact'):
            raise ValueError('Activity adaptation requires conductance response and compact ports')
        self.architecture = ("PlasticMedium3D-W4-learned-read-adaptive-v2"
                             if adaptive_conduction else "PlasticMedium3D-W4-learned-read-v1")
        if bath_type == 'conductance':
            self.architecture = 'PlasticMedium3D-W4-learned-read-conductance-v3'
        self.medium = PlasticMedium3D(tuple(shape), channels, material_width,
                                      hidden, collision_layers,
                                      adaptive_conduction=adaptive_conduction,
                                      plasticity_time_reference=plasticity_time_reference,
                                      bath_type=bath_type, response_time_reference=response_time_reference,
                                      voltage_reference=voltage_reference,
                                      capacitance_reference=capacitance_reference,
                                      activity_adaptation=activity_adaptation,
                                      short_term_plasticity=short_term_plasticity,
                                      execution_backend=medium_execution,
                                      anisotropic_transport=anisotropic_transport,
                                      transport_capacity_budget=transport_capacity_budget,
                                      material_reference_shape=material_reference_shape,
                                      structure_options=structure_options)
        if anisotropic_transport:
            self.architecture += '-tensor-transport'
        self.source = FullRankTorusWrite(vocab_size, tuple(shape), channels,
                                        write_type="w2_impedance", relative_address=True)
        self.write_agent = PredictiveImpedanceWriteAgent(
            channels, vocab_size, port_modes, exchange=write_exchange,
            local_shape=tuple(shape) if port_scope == 'compact' else None,
            port_radius=write_port_radius, activity_adaptation=activity_adaptation,
            aperture_budget=write_aperture_budget)
        if write_exchange == 'contact_mode':
            self.architecture += '-contact-port-v4'
        # The new wave collision protects channel mean rather than D3Q8 moments.
        w = self.medium.mean_reflector
        basis = torch.eye(channels, dtype=w.dtype) - 2 * w[:, None] * w[None, :]
        self.readout = PredictivePhysicalReadAgent(
            tuple(shape), channels, basis[:, 1:], heads=heads, queries=queries,
            aperture_type="compact_probes" if port_scope == 'compact' else "learned_probes",
            port_radius=read_port_radius, dynamic=read_mode in ('dynamic', 'temporal'),
            aperture_budget=read_aperture_budget, coordinate_reflector=w,
            compact_key_execution=read_key_execution)
        self.temporal_readout = None
        if read_mode == 'temporal':
            if temporal_rates is None or temporal_frequencies is None:
                raise ValueError('Temporal read requires explicit model-time rates/frequencies')
            self.temporal_readout = TemporalProbeReadout(
                heads * queries, channels, temporal_rates, temporal_frequencies,
                time_reference=temporal_time_reference)
        elif temporal_rates is not None or temporal_frequencies is not None:
            raise ValueError('Temporal parameters require temporal read mode')
        if port_scope == 'compact':
            self.architecture += '-compact-ports-v5'
        if activity_adaptation:
            self.architecture += '-activity-feedback-v6'
        if short_term_plasticity:
            self.architecture += '-stp-v7'
        self.decoder = nn.Linear(channels, vocab_size)
        nn.init.normal_(self.decoder.weight, std=0.02)
        nn.init.zeros_(self.decoder.bias)
        # Normalize the expression interface, never the persistent physical field.
        # Keep decoder keys intact for historical checkpoints. Legacy continuation
        # explicitly requests False; new learning branches use the normalized head.
        self.pre_decoder_norm = bool(pre_decoder_norm)
        self.read_norm = nn.RMSNorm(channels) if pre_decoder_norm else nn.Identity()
        if pre_decoder_norm:
            self.architecture += '-pre-decoder-rms-v8'
        if read_mode in ('dynamic', 'temporal'):
            self.architecture += '-local-dynamic-read-v9'
        if read_mode == 'temporal':
            self.architecture += '-causal-temporal-probes-v10'
        if self.intrinsic_time is not None:
            self.architecture += '-intrinsic-interval-v11'
        if structure_options is not None:
            self.architecture += '-structural-posterior-v12'

    def learning_named_parameters(self):
        """Parameters on this graph, excluding unused legacy writer machinery."""
        for name, parameter in self.named_parameters():
            if name.startswith('source.') and name not in (
                    'source.embedding.weight', 'source.channel_scale'):
                continue
            if parameter.requires_grad:
                yield name, parameter

    def event_time(self, belief: PlasticBelief, event_duration):
        """Choose time from pre-observation local history, before the new input.

        The arriving token and its target cannot broadcast information through
        a global clock. The physical state still responds to the arriving token
        during every subsequent solver step.
        """
        if self.intrinsic_time is None:
            return event_duration
        if belief.medium.field.shape[0] != 1:
            raise ValueError('Intrinsic event clock currently describes one persistent individual')
        field = sample_compact_probes(self.readout, belief.medium.field)
        flux = torch.stack([sample_compact_probes(self.readout, q)
                            for q in belief.medium.flux], -1)
        energy = field.square().mean((1, 2))
        moving = flux.square().mean((1, 2, 3))
        def scalar_measure(value):
            return torch.einsum('pn,bn->bp', self.readout.weights().to(value),
                                value.flatten(1)).mean(1)
        inhibition = (torch.zeros_like(energy) if belief.medium.receptors is None else
            scalar_measure(belief.medium.receptors[..., 1, :].mean(-1)))
        resource = (torch.ones_like(energy) if belief.medium.transmission is None else
            scalar_measure(belief.medium.transmission[..., 0].mean(-1)))
        features = torch.stack((torch.log1p(energy), torch.log1p(moving), inhibition, resource), -1)
        return self.intrinsic_time(features).squeeze(0)

    def decode(self, feature: torch.Tensor) -> torch.Tensor:
        """One shared expression head for native, chunked and captured execution."""
        return self.decoder(self.read_norm(feature))

    def initial_belief(self, batch_size: int = 1, *, device=None, dtype=None) -> PlasticBelief:
        if self.medium.structural_posterior is not None and batch_size != 1:
            raise ValueError('A structural evidence window belongs to one persistent individual')
        state = self.medium.initial_state(batch_size, device=device, dtype=dtype)
        precision = self.write_agent.initial_precision(
            batch_size, device=state.field.device, dtype=state.field.dtype)
        temporal = (None if self.temporal_readout is None
                    else self.temporal_readout.bank.initial_state(batch_size))
        return PlasticBelief(state, precision, temporal)

    def assimilate(self, belief: PlasticBelief, observed_ids: torch.Tensor, *,
                   token_features: torch.Tensor | None = None,
                   diagnostics: bool = True, training_terms: bool = True) -> tuple[PlasticBelief, dict]:
        """Fuse quiet port arithmetic and its backward without changing the policy."""
        operation = self._quiet_operation('assimilate', belief, diagnostics)
        return operation(belief, observed_ids, token_features=token_features,
                         diagnostics=diagnostics, training_terms=training_terms)

    def _quiet_operation(self, name, belief, diagnostics):
        native = getattr(self, 'native_' + name)
        if (self.port_execution != 'fused'
                or not belief.medium.field.is_cuda
                or belief.medium.field.dtype != torch.float32
                or torch.compiler.is_compiling()):
            return native
        # Conditional full-state response uses forward AD, not the production
        # reverse-mode tape. Its dual tensors need the native JVP and must not
        # create extra fullgraph compiler variants at evaluation boundaries.
        medium = belief.medium
        values = (medium.field, *medium.flux, medium.elapsed, medium.conduction,
                  medium.receptors, medium.transmission, belief.precision,
                  None if belief.temporal is None else belief.temporal.value,
                  None if belief.temporal is None else belief.temporal.elapsed)
        if any(torch.autograd.forward_ad.unpack_dual(value).tangent is not None
               for value in values if value is not None):
            return native
        key = '_compiled_' + name + ('_diagnostic' if diagnostics else '')
        if not hasattr(self, key):
            if os.name == 'nt':
                import torch._inductor.config as config
                config.use_static_cuda_launcher = False
                config.compile_threads = 1
            setattr(self, key, torch.compile(native, fullgraph=True, dynamic=False))
        return getattr(self, key)

    def native_assimilate(self, belief: PlasticBelief, observed_ids: torch.Tensor, *,
                          token_features: torch.Tensor | None = None,
                          diagnostics: bool = True, training_terms: bool = True) -> tuple[PlasticBelief, dict]:
        field, precision, _, info = self.write_agent(
            self.source, belief.medium.field, observed_ids, belief.precision,
            token_features=token_features, port_activity=self.write_port_activity(belief),
            return_diag=diagnostics, training_terms=training_terms)
        if diagnostics and self.port_scope == 'compact':
            info.update(self.port_diagnostics(field))
        return PlasticBelief(belief.medium.with_field(field), precision, belief.temporal), info

    def port_diagnostics(self, field):
        from .local_ports import port_overlap_diagnostics
        return port_overlap_diagnostics(self.write_agent.local_ports, self.readout, field)

    def write_port_activity(self, belief):
        if not self.activity_adaptation:
            return None
        inhibition = belief.medium.receptors[..., 1, :].flatten(1, 3)
        return self.write_agent.local_ports.observe(inhibition).mean(-1)

    @torch.no_grad()
    def port_snapshot(self, belief):
        """Current policy use as well as geometry, without advancing the belief.

        Write gate is the pre-observation policy for the next arriving event;
        read attention is the actual current read. This is monitoring only.
        """
        field, precision = belief.medium.field, belief.precision
        info = self.port_diagnostics(field)
        local_mean = self.write_agent.local_ports.observe(field.flatten(1, 3)).mean(1)
        features = torch.cat((F.rms_norm(local_mean, (self.medium.channels,)),
                              precision.clamp_min(torch.finfo(field.dtype).eps).log()), -1)
        gate = self.write_agent.chart_policy(features, self.write_port_activity(belief))
        _, read = self.read(belief, decode=False, diagnostics=True)
        if self.temporal_readout is not None:
            info.update({key: value for key, value in read.items()
                         if key.startswith('temporal_')})
        attention = read['read_attention_weights']
        eps = torch.finfo(field.dtype).eps
        entropy = -(attention * attention.clamp_min(eps).log()).sum(-1)
        active_write = gate @ self.write_agent.local_ports.weights()
        normalized_read = F.normalize(attention.flatten(1, 2), dim=-1)
        pair = normalized_read @ normalized_read.transpose(-1, -2)
        count = self.readout.heads * self.readout.queries
        info.update({
            'write_bank_gate_from_current_belief': gate,
            'write_effective_port_count': (-(gate * gate.clamp_min(eps).log()).sum(-1)).exp().mean(),
            'read_entropy_per_head': entropy.mean((0, 2)),
            'read_attention_per_head': attention.mean((0, 2)),
            'read_max_weight_per_head': attention.amax((0, 2, 3)),
            'read_attention_pair_overlap': ((pair.sum((1, 2)) - count) / max(1, count * (count - 1))).mean(),
            'active_write_read_overlap': (normalized_read * F.normalize(active_write, dim=-1)[:, None]).sum(-1).mean(),
        })
        if self.activity_adaptation:
            response = self.medium.conductance_response
            activity = response.processing_activity(field, belief.medium.flux).squeeze(-1)
            inhibition = response.inhibition_history(belief.medium.receptors)
            moving_energy = sum(x.square().sum(-1) for x in belief.medium.flux).flatten(1)
            eps = torch.finfo(field.dtype).eps
            info.update({
                'processing_activity_per_node': activity,
                'inhibition_history_per_node': inhibition,
                'write_port_inhibition': self.write_port_activity(belief),
                'activity_sensitivity': self.write_agent.log_activity_sensitivity.exp(),
                'flow_effective_node_count': (moving_energy.sum(-1).square() /
                                              moving_energy.square().sum(-1).clamp_min(eps)).mean(),
                'inhibition_spatial_std': inhibition.flatten(1).std(-1, correction=0).mean(),
            })
        return info

    def advance(self, belief: PlasticBelief, duration: float | torch.Tensor, *,
                substeps: int = 1, prepared: EvolutionCoefficients | None = None,
                diagnostics: bool = True, return_motion: bool = False,
                activation_checkpointing: bool = False):
        if self.solver_max_step is not None:
            return self.advance_interval(belief, duration, substeps=substeps,
                prepared=prepared, diagnostics=diagnostics, return_motion=return_motion,
                activation_checkpointing=activation_checkpointing)
        state, info = self.medium.advance(belief.medium, duration, substeps=substeps,
                                         prepared=prepared, diagnostics=diagnostics)
        motion = (self.read_time_reference * self.medium.field_rhs(state, prepared=prepared)
                  if return_motion and self.read_mode in ('dynamic', 'temporal') else None)
        outgoing = self.complete_advance(belief, state, duration, prepared=prepared, motion=motion)
        return (outgoing, info, motion) if return_motion else (outgoing, info)

    def advance_interval(self, belief, duration, *, substeps=1, prepared=None,
                         diagnostics=True, schedule=None, return_motion=False,
                         activation_checkpointing=False):
        """Evolve and sample an interval, retaining its whole differentiable path.

        Integer quadrature counts depend only on a detached duration. Physical
        interval tensors keep their gradients. A saved schedule can be replayed
        by activation checkpointing without making another timing decision.
        """
        from .intrinsic_time import EvolutionSchedule
        if self.solver_max_step is None or self.observer_max_step is None:
            raise ValueError('Interval execution requires explicit numerical resolution')
        schedule = (EvolutionSchedule.for_duration(duration,
            solver_max_step=self.solver_max_step, observer_max_step=self.observer_max_step,
            max_steps=self.max_evolution_steps) if schedule is None else schedule)
        duration = torch.as_tensor(duration, device=belief.medium.field.device,
                                   dtype=belief.medium.elapsed.dtype)
        interval = schedule.interval_duration(duration)
        if schedule.observer_count * max(substeps, schedule.solver_substeps) > self.max_evolution_steps:
            raise ValueError('Actual requested substeps exceed the explicit evolution budget')
        prepared = self.medium.prepare_evolution() if prepared is None else prepared
        info, totals, motion = {}, {}, None
        def interval_step(current, sample_motion):
            state, part = self.medium.advance(current.medium, interval,
                substeps=max(substeps, schedule.solver_substeps), prepared=prepared,
                diagnostics=diagnostics, activation_checkpointing=activation_checkpointing)
            motion = None
            if sample_motion:
                def measure_motion(value):
                    return self.read_time_reference * self.medium.field_rhs(value, prepared=prepared)
                motion = (checkpoint_state(measure_motion, state)
                          if activation_checkpointing and torch.is_grad_enabled()
                          else measure_motion(state))
            outgoing = self.complete_advance(current, state, interval,
                                              prepared=prepared, motion=motion)
            return outgoing, part, motion
        for index in range(schedule.observer_count):
            sample_motion = self.temporal_readout is not None or (return_motion
                and index == schedule.observer_count - 1 and self.read_mode == 'dynamic')
            if activation_checkpointing and torch.is_grad_enabled():
                # Pass the per-interval decision as an argument; replay never
                # observes a mutated loop index or truncates the physical path.
                belief, part, motion = checkpoint_state(interval_step, belief, sample_motion)
            else:
                belief, part, motion = interval_step(belief, sample_motion)
            if diagnostics:
                for key, value in part.items():
                    if key.endswith('_change') or key in (
                            'bath_out_energy', 'response_source_work', 'response_joule_heat',
                            'response_energy_residual'):
                        totals[key] = totals.get(key, 0) + value
                if 'energy_before' not in info:
                    info['energy_before'] = part['energy_before']
                info.update({key: value for key, value in part.items()
                             if key not in totals and key != 'energy_before'})
        info.update(totals)
        if diagnostics:
            info.update(intrinsic_duration=torch.as_tensor(duration).detach(),
                        observer_samples=schedule.observer_count,
                        solver_steps=schedule.observer_count * max(substeps, schedule.solver_substeps))
        return (belief, info, motion) if return_motion else (belief, info)

    def complete_advance(self, belief, state, duration, *, prepared=None, motion=None):
        """Commit one physical interval and its already-observed endpoint sample.

        Held-endpoint quadrature is causal at emission and matches the accepted
        filter audit. Calls to read/assimilate never integrate observer history.
        Sampling cadence is the advance-call cadence, separately from physical
        solver substeps. Deployment must train at its declared sampling cadence.
        """
        history = belief.temporal
        if self.temporal_readout is not None:
            if history is None:
                raise ValueError('Temporal continuation requires its saved probe history')
            torch._assert_async(torch.isclose(history.elapsed, belief.medium.elapsed,
                                             atol=1e-10, rtol=1e-10).all(),
                                'Temporal history clock differs from physical time')
            field = sample_compact_probes(self.readout, state.field)
            if motion is None:
                motion = self.read_time_reference * self.medium.field_rhs(state, prepared=prepared)
            signal = torch.cat((field, sample_compact_probes(self.readout, motion)), -1)
            history = self.temporal_readout.bank(signal, history, duration)
            torch._assert_async(torch.isclose(history.elapsed, state.elapsed,
                                             atol=1e-10, rtol=1e-10).all(),
                                'Temporal and physical advancement durations differ')
        elif history is not None:
            raise ValueError('Temporal state requires a temporal-read model')
        return PlasticBelief(state, belief.precision, history)

    def read(self, belief: PlasticBelief, *, decode: bool = True,
             diagnostics: bool = False,
             prepared: EvolutionCoefficients | None = None, motion=None) -> tuple[torch.Tensor, dict | None]:
        # Monitoring may request a fresh RHS without prepared coefficients.
        # Preparing a structural window is a Python transaction; keep it outside
        # the compiled port. Training already supplies the measured endpoint RHS.
        operation = (self.native_read if motion is None and self.read_mode in ('dynamic', 'temporal')
                     else self._quiet_operation('read', belief, diagnostics))
        return operation(belief, decode=decode, diagnostics=diagnostics, prepared=prepared, motion=motion)

    def native_read(self, belief: PlasticBelief, *, decode: bool = True,
                    diagnostics: bool = False,
                    prepared: EvolutionCoefficients | None = None, motion=None) -> tuple[torch.Tensor, dict | None]:
        if motion is None and self.read_mode in ('dynamic', 'temporal'):
            motion = self.read_time_reference * self.medium.field_rhs(belief.medium, prepared=prepared)
        feature, info = self.readout(belief.medium.field, belief.precision,
                                     return_diag=diagnostics, motion=motion)
        if self.temporal_readout is not None:
            if belief.temporal is None:
                raise ValueError('Temporal read requires persistent history')
            torch._assert_async(torch.isclose(belief.temporal.elapsed, belief.medium.elapsed,
                                             atol=1e-10, rtol=1e-10).all(),
                                'Temporal read clock differs from physical time')
            feature = feature + self.temporal_readout(belief.temporal)
            if diagnostics:
                value = belief.temporal.value
                info.update(temporal_real_power_per_mode=value.real.square().mean((0, 1, 3)),
                            temporal_imag_power_per_mode=value.imag.square().mean((0, 1, 3)),
                            temporal_rates=self.temporal_readout.bank.log_rate.exp(),
                            temporal_frequencies=self.temporal_readout.bank.physical_frequency,
                            temporal_elapsed=belief.temporal.elapsed)
        if diagnostics and self.port_scope == 'compact':
            info.update(self.port_diagnostics(belief.medium.field))
        return self.decode(feature) if decode else feature, info

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor,
                belief: PlasticBelief | None = None, *, event_duration: float,
                substeps: int = 1) -> tuple[torch.Tensor, PlasticBelief, dict]:
        """Likelihood training at an explicit event cadence; no implicit reset.

        Output predicts the next token after the current observed token has been
        assimilated. Deployments with intervening reads use the same three APIs
        and must train at those read timestamps as well.
        """
        if input_ids.ndim != 2 or input_ids.shape != targets.shape or input_ids.shape[1] < 1:
            raise ValueError("Nonempty [B,L] observations and targets required")
        belief = self.initial_belief(input_ids.shape[0]) if belief is None else belief
        if self.medium.structural_posterior is not None:
            from ..runtime.training import quiet_training_chunk
            loss, outgoing, nll, components = quiet_training_chunk(self, input_ids, targets,
                belief, event_duration=event_duration, substeps=substeps, return_loss_components=True)
            return loss, outgoing, components
        token_features = F.normalize(self.source.embedding.weight, dim=-1)
        prepared = self.medium.prepare_evolution()
        features, terms = [], []
        info = {}
        for index in range(input_ids.shape[1]):
            duration = self.event_time(belief, event_duration)
            belief, write_info = self.assimilate(
                belief, input_ids[:, index], token_features=token_features, diagnostics=False)
            terms.append(write_info["_write_free_energy"])
            belief, info, motion = self.advance(belief, duration, substeps=substeps,
                                               prepared=prepared, return_motion=True)
            feature, _ = self.read(belief, decode=False, prepared=prepared, motion=motion)
            features.append(feature)
        logits = self.decode(torch.stack(features, 1))
        nll = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        # Preserve the existing port objective; don't rename it a full latent ELBO.
        port_objective = torch.stack(terms).mean()
        return nll + port_objective, belief, {
            **info, "token_nll": nll.detach(), "port_objective": port_objective.detach()}

    def forward_timestamped(self, input_ids: torch.Tensor, targets: torch.Tensor,
                            observation_times, read_times, *, max_step: float,
                            belief: PlasticBelief | None = None):
        """Train the same causal event engine used at deployment.

        Next-token targets remain unobserved until after their prediction time.
        Read timestamps lie at/after their observation and before the next one.
        This method does not wait for convergence or detach between events.
        """
        from ..runtime.continuous import ContinuousStream
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape != targets.shape:
            raise ValueError('Timestamped individual requires matching [1,L] ids/targets')
        observations, reads = list(map(float, observation_times)), list(map(float, read_times))
        if not observations or len(observations) != input_ids.shape[1] or len(reads) != len(observations):
            raise ValueError('One observation/read timestamp per supervised event required')
        for index, (observed, read) in enumerate(zip(observations, reads)):
            if not math.isfinite(observed) or not math.isfinite(read) or read < observed:
                raise ValueError('Finite read time at/after observation required')
            if index + 1 < len(observations) and read >= observations[index + 1]:
                raise ValueError('Next-token prediction must precede the next observation')
        stream = ContinuousStream(self, max_step=max_step, belief=belief)
        features, objectives = [], []
        for index, (observed, read) in enumerate(zip(observations, reads)):
            info = stream.observe(observed, input_ids[:, index])
            objectives.append(info['_write_free_energy'])
            features.append(stream.read_at(read, decode=False).value)
        logits = self.decode(torch.stack(features, 1))
        nll = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        port_objective = torch.stack(objectives).mean()
        return nll + port_objective, stream.belief, {
            'token_nll': nll.detach(), 'port_objective': port_objective.detach(),
            'solver_steps': stream.steps, 'physical_time': stream.time}
