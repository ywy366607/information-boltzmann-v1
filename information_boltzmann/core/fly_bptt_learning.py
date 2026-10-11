"""32-event truncated BPTT for one continuing sensory/motor COBA individual.

Physical values persist between windows. Only autograd history is detached.
The spike backward is the model's existing ATan surrogate, not a derivative
of the discontinuous hard threshold. Static edge topology remains fixed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .triton_synapse import execute_delayed_synaptic_transmission
from .gradient_norms import stable_grad_norm, stable_clip_grad_norm_

STP_PARAMETER_NAMES = ('logit_u0', 'log_tau_fac', 'log_tau_rec')


@dataclass
class FlyPhysicalState:
    h: torch.Tensor
    ring: tuple[torch.Tensor, ...]
    ge: torch.Tensor
    gi: torch.Tensor
    b: torch.Tensor
    x: torch.Tensor
    u: torch.Tensor
    baseline: torch.Tensor
    # Slow running mean of the membrane state (the read pathway's background
    # estimate).  Empty until the first event initializes it from h.
    h_mean: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    # Optional, detached DAN arrival filter. Persisted only when local learning
    # is enabled; its time scale is shared with postsynaptic current relaxation.
    dan_gate: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    # Optional 2nd-order continuous Gamma trace state (Solution 1 temporal accumulator).
    gamma_z1: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    gamma_z2: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    # Persistent prior state for Graph Observer (HX-1 / LeWM state observer).
    observer_prior: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    # Persistent recent 4-tick posterior history [H, 4, d_model] for multi-tier delays.
    observer_history: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    # Optional read bank: exponential filters of the motor membranes along the whole tick stream [F, n_read].
    read_bank: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    # Optional neuromodulator "concentration": slow trace of the modulatory cells' spikes [1, n_modulatory].
    modulator: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    # Optional top-down prediction of the next input: the expected word embedding of the brain's last spoken
    # prediction [1, d_model] (predictive-coding writer).
    expected_input: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    # Optional dopamine at every neuron along the connectome's modulatory projections: fast concentration and its slow
    # mean [2, N]; the phasic component is their difference.
    dopamine: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    def detached(self):
        return FlyPhysicalState(**{
            item.name: tuple(t.detach() for t in self.ring) if item.name == 'ring'
            else getattr(self, item.name).detach() for item in fields(self)})

    def state_dict(self):
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_online(cls, state, device):
        syn = state['syn']
        h = state['h'].to(device)
        h_mean = state.get('h_mean')
        h_mean = (h_mean.to(device) if h_mean is not None and h_mean.numel() == h.numel()
                  else torch.zeros_like(h))
        dan_gate = state.get('dan_gate', torch.empty(0)).to(device)
        gamma_z1 = state.get('gamma_z1', torch.empty(0)).to(device)
        gamma_z2 = state.get('gamma_z2', torch.empty(0)).to(device)
        observer_prior = state.get('observer_prior', torch.empty(0)).to(device)
        obs_hist = state.get('observer_history', torch.empty(0))
        if isinstance(obs_hist, (list, tuple)):
            observer_history = torch.stack(list(obs_hist), dim=0).to(device) if len(obs_hist) > 0 else torch.empty(0, device=device)
        elif isinstance(obs_hist, torch.Tensor):
            observer_history = obs_hist.to(device)
        else:
            observer_history = torch.empty(0, device=device)
        return cls(h, tuple(t.to(device) for t in state['ring']),
                   *(syn[key].to(device) for key in ('ge', 'gi', 'b', 'x', 'u')),
                   state['writer_baseline'].to(device), h_mean, dan_gate,
                   gamma_z1, gamma_z2, observer_prior, observer_history)


def step_fly_physical_tick(model, current, token, source, next_baseline, options, *,
                           h_mean_decay=0.99, base_rates=None):
    """Atomic physical step on the connectome with optional sensory drive."""
    kappa = getattr(model, 'intrinsic_kappa', None)
    if kappa is not None:
        # Intrinsic (pacemaker / tonic) current of the annotated spontaneously active cells, in units of their own
        # rheobase: the membrane alone would settle at kappa * threshold (kappa > 1 fires on its own).
        idx = model.intrinsic_index
        rates = base_rates if base_rates is not None else model.get_decay_rates()
        thresholds = options.get('thresholds')
        thresholds = thresholds if thresholds is not None else model.get_thresholds()
        g_leak = -torch.log(rates[0].expand_as(current.h)[0, idx].detach().clamp(1e-5, 1 - 1e-7))
        current_in = kappa * thresholds.expand_as(current.h)[0, idx].detach() * g_leak
        source = (torch.zeros_like(current.h) if source is None else source).index_add(
            1, idx, current_in.unsqueeze(0))
    dopamine = getattr(current, 'dopamine', torch.empty(0))
    if getattr(model, 'dopamine_sensitivity', None) is not None and dopamine.numel():
        # Dopamine modulates excitability where it is delivered: the phasic part (concentration minus its slow mean)
        # scales the threshold, theta * exp(-s_class * phasic); s starts at 0 for every superclass.
        tonic = dopamine[1:2]
        # the tonic level: mean slow concentration over the targets
        level = tonic[tonic > 0].mean().detach().clamp_min(1e-12) if bool((tonic > 0).any()) else tonic.new_tensor(1.0)
        if getattr(model, 'dopamine_adapted', False):
            # adapted receptors with half-saturation at the tonic level: a target's relative change
            # (fast - slow) / (slow + level), bounded for targets that receive only a trace of dopamine
            phasic = (dopamine[0:1] - tonic) / (tonic.detach() + level)
        else:
            # relative to the tonic level: dimensionless, parameter-free
            phasic = (dopamine[0:1] - tonic) / level
        s_class = model.dopamine_sensitivity[model.superclass_id].unsqueeze(0)
        thresholds = options.get('thresholds')
        thresholds = (thresholds if thresholds is not None else model.get_thresholds()).expand_as(current.h)
        options = {**options, 'thresholds': thresholds * torch.exp(-s_class * phasic)}
    modulator = getattr(current, 'modulator', torch.empty(0))
    if getattr(model, 'precision_u', None) is not None and source is not None:
        # Precision = attention: each sensory pathway's written innovation is scaled by a gain set by the current
        # neuromodulator concentration, a = 2 sigmoid(u . m + b) (a = 1 at u = b = 0).
        if modulator.numel() != model.modulatory_index.numel():
            modulator = current.h.new_zeros(1, model.modulatory_index.numel())
        gain = 2.0 * torch.sigmoid(modulator @ model.precision_u.T + model.precision_b)          # [1, 3]
        source = source * model.precision_scale(gain)
    dan_gate = current.dan_gate
    if getattr(model, 'dan_plastic_lr', 0.0) > 0.0 and model.has_dopamine:
        with torch.no_grad():
            arrival = execute_delayed_synaptic_transmission(
                tuple(p.detach() for p in current.ring), model.dan_edge_pre,
                model.dan_edge_post, model.dan_edge_weight, model.dan_delay_splits)
            arrival = arrival / model.dan_scale
            rho = (base_rates if base_rates is not None else model.get_decay_rates())[1].detach()
            if dan_gate.numel() != current.h.numel():
                dan_gate = torch.zeros_like(current.h)
            dan_gate = rho * dan_gate + (1.0 - rho) * arrival
    ret = model.step(
        current.h, token, spike_ring=current.ring, ge=current.ge,
        gi=current.gi, b=current.b, x=current.x, u=current.u,
        i_syn=current.ge if model.synapse_model == 'cuba' else None,
        sensory_drive=source, **options)
    h = ret[0]
    if model.synapse_model == 'coba':
        ring, ge, gi = ret[2:5]
        offset = 5
    else:
        offset = 2
        if model.has_delays:
            ring = ret[offset]
            offset += 1
        else:
            ring = (ret[1], *current.ring[:3])
        ge, gi = ret[offset], current.gi
        offset += 1
    b, x, u = current.b, current.x, current.u
    if model.use_alif:
        b = ret[offset]
        offset += 1
    if model.use_stp:
        x, u = ret[offset:offset + 2]
    h_mean = current.h_mean
    if h_mean.numel() != h.numel():
        h_mean = torch.zeros_like(h)
    h_mean = h_mean_decay * h_mean + (1.0 - h_mean_decay) * h
    gz1, gz2 = current.gamma_z1, current.gamma_z2
    if getattr(model, 'use_read_gamma_trace', False):
        read_in = (h - h_mean
                   if getattr(model, 'read_centering', False)
                   and h_mean.numel() == h.numel() else h)
        motor = read_in[:, model.read_indices] if model.read_surface != "all" else read_in
        if gz1.numel() != motor.numel():
            gz1 = torch.zeros_like(motor)
            gz2 = torch.zeros_like(motor)
        decay = model.get_read_gamma_decay()
        gz1 = decay * gz1 + (1.0 - decay) * motor
        gz2 = decay * gz2 + (1.0 - decay) * gz1
    if getattr(model, 'precision_u', None) is not None:
        rho = torch.sigmoid(model.modulator_logit_decay)
        if modulator.numel() != model.modulatory_index.numel():
            modulator = h.new_zeros(1, model.modulatory_index.numel())
        modulator = rho * modulator + (1.0 - rho) * (ring[0][:, model.modulatory_index] > 0).float()
    if getattr(model, 'dopamine_sensitivity', None) is not None:
        arrival = execute_delayed_synaptic_transmission(ring, model.dan_edge_pre, model.dan_edge_post,
                                                        model.dan_edge_weight, model.dan_delay_splits) / model.dan_scale
        fast, slow = torch.sigmoid(model.dopamine_logit_fast), torch.sigmoid(model.dopamine_logit_slow)
        if dopamine.numel() != 2 * h.numel():
            dopamine = torch.cat([arrival, arrival], 0).detach()
        conc = fast * dopamine[0:1] + (1.0 - fast) * arrival
        dopamine = torch.cat([conc, slow * dopamine[1:2] + (1.0 - slow) * conc], 0)
    bank = getattr(current, 'read_bank', torch.empty(0))
    if getattr(model, 'read_bank_gain', None) is not None:
        motor_now = h[:, model.read_indices] if model.read_surface != 'all' else h
        decay = model.read_bank_decay()
        if bank.numel() != decay.numel() * motor_now.numel():
            bank = motor_now.expand(decay.numel(), -1)
        bank = decay * bank + (1.0 - decay) * motor_now
    return FlyPhysicalState(h, ring, ge, gi, b, x, u, next_baseline, h_mean, dan_gate, gz1, gz2,
                            current.observer_prior, getattr(current, 'observer_history', torch.empty(0)), bank,
                            modulator, getattr(current, 'expected_input', torch.empty(0)), dopamine)


def extract_fly_motor_latent(model, state):
    """Extract and normalize motor readout latents from current brain state."""
    read_in = (state.h - state.h_mean
               if getattr(model, 'read_centering', False)
               and state.h_mean.numel() == state.h.numel() else state.h)
    motor = read_in[:, model.read_indices] if model.read_surface != 'all' else read_in
    lat = model.output_read(motor)
    return model.read_norm(lat) if hasattr(model, 'read_norm') else lat


def collect_fly_quiet_trajectory(model, state, num_ticks=14, *,
                                 writer_baseline_clock='input',
                                 base_rates=None, thresholds=None,
                                 conductance_gains=None, alif_params=None,
                                 stp_params=None, h_mean_decay=0.99,
                                 return_h=False, return_states=False):
    """Collect ground-truth physical motor latents or whole-brain h across num_ticks quiet steps (s=0).

    Conditioned purely on in-flight pulses already inside the brain. Zero external
    tokens enter, ensuring zero future token leakage.
    """
    quiet_source = torch.zeros_like(state.h)
    options = dict(base_rates=base_rates, thresholds=thresholds,
                   conductance_gains=conductance_gains,
                   alif_params=alif_params, stp_params=stp_params)
    current = state
    trajectory = []
    states = []
    with torch.no_grad():
        for _ in range(num_ticks):
            baseline = (model.topographic_writer.lambda_adapt * current.baseline
                        if writer_baseline_clock == 'physical'
                        and model.topographic_writer is not None else current.baseline)
            current = step_fly_physical_tick(model, current, None, quiet_source, baseline, options,
                                             h_mean_decay=h_mean_decay, base_rates=base_rates)
            if return_states:
                states.append(current.detached())
            if return_h:
                trajectory.append(current.h.clone())
            else:
                trajectory.append(extract_fly_motor_latent(model, current).clone())
    if return_states:
        return trajectory, states
    return trajectory, current


def collect_student_error_guided_queries(
    model, quiet_states, graph_observer, *,
    writer_baseline_clock='input',
    base_rates=None, thresholds=None,
    conductance_gains=None, alif_params=None, stp_params=None,
    h_mean_decay=0.99, max_queries=4, alpha=0.05,
    prior_history=None
):
    """Collects actual queried (q_history, z_target) pairs on detached copies of the physical state.

    Contract:
    S_q = admissible_physical_perturbation(copy(S_ref), student_error)
    z_q = E(S_q)
    z_target = E(F_physical(S_q))
    train G(q_history ending with z_q) -> z_target
    """
    if not quiet_states:
        return []

    with torch.no_grad():
        S_0 = quiet_states[0]
        z_0 = graph_observer.encoders(S_0)  # [1, 4, d_model]
        num_hops = min(max_queries, len(quiet_states))
        student_sim = graph_observer.simulate_hops(z_0, num_hops=num_hops, history=prior_history)

        query_pairs = []
        quiet_source = torch.zeros_like(S_0.h)
        options = dict(base_rates=base_rates, thresholds=thresholds,
                       conductance_gains=conductance_gains,
                       alif_params=alif_params, stp_params=stp_params)

        Z_ref_list = [graph_observer.encoders(s).detach() for s in quiet_states]
        base_ref_history = [h.detach() for h in prior_history] if prior_history is not None else []

        for k in range(min(max_queries, len(quiet_states) - 2)):
            # Timestamp alignment:
            # S_0 is quiet step 1 (tick 1).
            # student_sim[k] is student's simulated prediction at quiet step k + 2.
            # quiet_states[k + 1] is teacher's true physical state at quiet step k + 2.
            S_ref_k = quiet_states[k + 1]
            z_ref_k = Z_ref_list[k + 1]                  # [1, 4, d_model] (step k + 2)
            z_student_k = student_sim[k]                  # [1, 4, d_model] (step k + 2)

            # True student discrepancy at step k + 2:
            delta_z = z_student_k - z_ref_k               # [1, 4, d_model]

            # Form admissible perturbation on reference physical state at step k + 2:
            h_perturbed = graph_observer.encoders.make_admissible_perturbation(
                S_ref_k.h.squeeze(0), delta_z.squeeze(0), alpha=alpha
            ).unsqueeze(0)

            S_q = FlyPhysicalState(**{
                k_name: tuple(t.clone() for t in v) if k_name == 'ring'
                else (h_perturbed if k_name == 'h' else (v.clone() if isinstance(v, torch.Tensor) else v))
                for k_name, v in S_ref_k.state_dict().items()
            })

            # 1. Re-encode actual perturbed state:
            z_q = graph_observer.encoders(S_q).detach()  # [1, 4, d_model]

            # 2. Advance physical teacher 1 tick from S_q:
            baseline = (model.topographic_writer.lambda_adapt * S_q.baseline
                        if writer_baseline_clock == 'physical'
                        and model.topographic_writer is not None else S_q.baseline)
            S_q_next = step_fly_physical_tick(
                model, S_q, None, quiet_source, baseline, options,
                h_mean_decay=h_mean_decay, base_rates=base_rates
            )

            # 3. Expert target is the re-encoded physical state after 1 tick:
            z_target = graph_observer.encoders(S_q_next).detach()  # [1, 4, d_model]

            # 4. Construct query history: unperturbed history preceding step k+2, with z_q at step k+2
            full_past = base_ref_history + Z_ref_list[:k + 1]
            past_3 = full_past[-3:] if len(full_past) >= 3 else full_past
            q_history = past_3 + [z_q]

            query_pairs.append((q_history, z_target))

    return query_pairs


def advance_fly_input_event(model, state, token, *, settle_ticks=0,
                            writer_baseline_clock='input',
                            base_rates=None, thresholds=None,
                            conductance_gains=None, alif_params=None,
                            stp_params=None, h_mean_decay=0.99,
                            return_ticks=False):
    """One token pulse followed by quiet physical ticks, before next-token read."""
    if not isinstance(settle_ticks, int) or settle_ticks < 0:
        raise ValueError('settle_ticks must be a nonnegative integer')
    if writer_baseline_clock not in ('input', 'physical'):
        raise ValueError('writer_baseline_clock must be input or physical')
    if not 0.0 <= h_mean_decay < 1.0:
        raise ValueError('h_mean_decay must be in [0, 1)')
    drive, baseline = write_input(model, state, token)
    options = dict(base_rates=base_rates, thresholds=thresholds,
                   conductance_gains=conductance_gains,
                   alif_params=alif_params, stp_params=stp_params)

    state = step_fly_physical_tick(model, state, token, drive, baseline, options,
                                   h_mean_decay=h_mean_decay, base_rates=base_rates)
    tick_states = [state] if return_ticks else None
    if settle_ticks:
        quiet_source = torch.zeros_like(state.h)
        for _ in range(settle_ticks):
            quiet_baseline = (model.topographic_writer.lambda_adapt * state.baseline
                              if writer_baseline_clock == 'physical'
                              and model.topographic_writer is not None else state.baseline)
            state = step_fly_physical_tick(model, state, None, quiet_source, quiet_baseline, options,
                                           h_mean_decay=h_mean_decay, base_rates=base_rates)
            if return_ticks:
                tick_states.append(state)
    if return_ticks:
        return state, tick_states
    return state


def advance_token_and_read_latent(model, state, token, *, settle_ticks=0,
                                  writer_baseline_clock='input',
                                  base_rates=None, thresholds=None,
                                  conductance_gains=None, alif_params=None,
                                  stp_params=None, h_mean_decay=0.99,
                                  use_ctm=False):
    """Atomic token advancement plus motor latent extraction for gradient checkpointing."""
    next_state, tick_states = advance_fly_input_event(
        model, state, token,
        settle_ticks=settle_ticks,
        writer_baseline_clock=writer_baseline_clock,
        base_rates=base_rates,
        thresholds=thresholds,
        conductance_gains=conductance_gains,
        alif_params=alif_params,
        stp_params=stp_params,
        h_mean_decay=h_mean_decay,
        return_ticks=True
    )
    token_latents = []
    if use_ctm:
        for st in tick_states:
            if getattr(model, 'use_read_gamma_trace', False) and st.gamma_z2.numel() > 0:
                token_latents.append(model.output_read(st.gamma_z2))
            else:
                read_in = (st.h - st.h_mean
                           if getattr(model, 'read_centering', False)
                           and st.h_mean.numel() == st.h.numel() else st.h)
                motor = read_in[:, model.read_indices] if model.read_surface != 'all' else read_in
                token_latents.append(model.output_read(motor))
    else:
        if getattr(model, 'use_read_gamma_trace', False) and next_state.gamma_z2.numel() > 0:
            token_latents.append(model.output_read(next_state.gamma_z2))
        else:
            read_in = (next_state.h - next_state.h_mean
                       if getattr(model, 'read_centering', False)
                       and next_state.h_mean.numel() == next_state.h.numel() else next_state.h)
            motor = read_in[:, model.read_indices] if model.read_surface != 'all' else read_in
            token_latents.append(model.output_read(motor))
    return next_state, torch.stack(token_latents, dim=0)


def _with_transmission_reuse(inner, model, state, token, schedule_box, **kwargs):
    """Record this token's synaptic transmission outputs on its first forward and replay them when a checkpoint
    recomputes it (same values; the transmission kernels are skipped). The cache rides on the token's schedule box,
    whose first entry is the token's tick count once known."""
    from . import triton_synapse as ts
    if not (ts.REUSE_TRANSMISSION and schedule_box is not None and torch.is_grad_enabled()):
        return inner(model, state, token, schedule_box, **kwargs)
    replay = len(schedule_box) > 1 and schedule_box[0] is not None
    previous = ts.TRANSMISSION_RECORD, ts.TRANSMISSION_REPLAY
    record = None if replay else []
    ts.TRANSMISSION_RECORD, ts.TRANSMISSION_REPLAY = record, (iter(schedule_box[1]) if replay else None)
    try:
        result = inner(model, state, token, schedule_box, **kwargs)
    finally:
        ts.TRANSMISSION_RECORD, ts.TRANSMISSION_REPLAY = previous
    if record is not None:
        schedule_box.append(record)
    return result


def advance_fly_token_adaptive(model, state, token, schedule_box=None, **kwargs):
    return _with_transmission_reuse(_advance_fly_token_adaptive, model, state, token, schedule_box, **kwargs)


def advance_fly_token_stream(model, state, token, schedule_box=None, **kwargs):
    return _with_transmission_reuse(_advance_fly_token_stream, model, state, token, schedule_box, **kwargs)


GO_PARAMETER_NAMES = ('go_read.weight', 'go_drive', 'go_reset', 'go_logit_leak')


def attach_go_neuron(model, *, leak=0.85, reset=-20.0, drive=1.2):
    """Give the body a "go" neuron: a leaky integrator of the motor surface whose firing is the act of speaking the
    current prediction and then moving to the next input (a saccade).  Its membrane G starts at `reset` after every
    firing and integrates G <- leak * G + drive + w . motor; the firing probability per tick is sigmoid(G).
    Initial values (w = 0) put its firing at 6-9 ticks after an input (mean 7.9; the champion's flux rule averages
    7.2); the brain's own motor state then learns to hasten or delay it.  Also stores each neuron's share of all
    synapses, so the energy of a tick (synaptic events) can be read from its spikes."""
    device = model.edge_pre_e.device if hasattr(model, 'edge_pre_e') else model.edge_pre.device
    width = model.n_read if model.read_surface != 'all' else model.n_neurons
    model.go_read = nn.Linear(width, 1, bias=False).to(device)
    nn.init.zeros_(model.go_read.weight)
    model.go_drive = nn.Parameter(torch.tensor([drive], device=device))
    model.go_reset = nn.Parameter(torch.tensor([reset], device=device))
    model.go_logit_leak = nn.Parameter(torch.tensor([math.log(leak / (1 - leak))], device=device))
    if hasattr(model, 'edge_pre_e'):
        pre = torch.cat([model.edge_pre_e.long(), model.edge_pre_i.long()])
    else:
        pre = model.edge_pre.long()
    out_degree = torch.bincount(pre, minlength=model.n_neurons).float()
    model.register_buffer('synapse_share', (out_degree / out_degree.sum()).unsqueeze(0), persistent=False)
    return model


GIANT_FIBER_BODY_IDS = (10001, 10010)        # MaleCNS DNp01 (GF) right and left


def attach_connectome_go(model, graph_npz, body_ids=GIANT_FIBER_BODY_IDS):
    """Use existing connectome neurons as the go act (default: the two giant fibers, DNp01): their own spike -- the
    membrane integrating its synapses and crossing threshold -- is the act of speaking the prediction and taking the
    next input.  Nothing is added to the brain; only their indices are recorded."""
    import numpy as np
    ids = np.load(graph_npz)['neuron_body_ids']
    index = [int(np.flatnonzero(ids == b)[0]) for b in body_ids]
    model.register_buffer('go_neurons', torch.as_tensor(index, dtype=torch.long, device=model.read_indices.device),
                          persistent=False)
    return model


def advance_fly_token_go(model, state, token, schedule_box=None, **kwargs):
    return _with_transmission_reuse(_advance_fly_token_go, model, state, token, schedule_box, **kwargs)


def _advance_fly_token_go(model, state, token, schedule_box=None, *, max_ticks=14, writer_baseline_clock='physical',
                          base_rates=None, thresholds=None, conductance_gains=None, alif_params=None,
                          stp_params=None, h_mean_decay=0.99, silence_after=0):
    """Self-paced token: the input pulse at tick 1, then ticks run until the go neuron fires (sampled from its
    firing probability) or max_ticks.  Returns the motor read, go firing probability and energy (share of synapses
    carrying a spike) of every realized tick.  The prediction is read at the firing tick and the next token is
    injected only afterwards, so the read can never see the token it predicts.  The realized tick count is stored in
    schedule_box[0] and replayed exactly by a checkpoint recompute."""
    if writer_baseline_clock not in ('input', 'physical'):
        raise ValueError('writer_baseline_clock must be input or physical')
    options = dict(base_rates=base_rates, thresholds=thresholds, conductance_gains=conductance_gains,
                   alif_params=alif_params, stp_params=stp_params)
    writer = model.topographic_writer
    drive, baseline = write_input(model, state, token)
    current = step_fly_physical_tick(model, state, token, drive, baseline, options,
                                     h_mean_decay=h_mean_decay, base_rates=base_rates)
    replay = schedule_box is not None and len(schedule_box) > 0 and schedule_box[0] is not None
    target = schedule_box[0] if replay else None
    physical = getattr(model, 'go_neurons', None) is not None
    if physical:            # connectome go neurons: the hazard is the surrogate CDF of their own membrane
        go_threshold = (thresholds if thresholds is not None else model.get_thresholds())
        go_threshold = go_threshold.expand_as(current.h)[0, model.go_neurons]
    else:
        leak = torch.sigmoid(model.go_logit_leak)
        go = model.go_reset
    share = getattr(model, 'synapse_share', None)
    quiet_source = torch.zeros_like(current.h)
    latents, hazards, energy = [], [], []
    for tick in range(1, max_ticks + 1):
        if tick > 1:
            baseline = (writer.lambda_adapt * current.baseline
                        if writer_baseline_clock == 'physical' and writer is not None else current.baseline)
            current = step_fly_physical_tick(model, current, None, quiet_source, baseline, options,
                                             h_mean_decay=h_mean_decay, base_rates=base_rates)
        read_in = (current.h - current.h_mean if getattr(model, 'read_centering', False)
                   and current.h_mean.numel() == current.h.numel() else current.h)
        motor = read_in[:, model.read_indices] if model.read_surface != 'all' else read_in
        latents.append(model.output_read(motor_feature(model, current, motor)))
        if physical:
            # The act is the go neurons' own spike.  Their transmitted pulse is zero or positive in the forward pass and
            # carries the spike's ATan surrogate gradient backward, exactly like every other spike in the network.
            hazard = current.ring[0][0, model.go_neurons].sum().reshape(1)
            fired = bool((hazard.detach() > 0).item())
        else:
            go = leak * go + model.go_drive + model.go_read(motor).squeeze(-1)
            hazard = torch.sigmoid(go)
            fired = None
        hazards.append(hazard)
        energy.append(((current.ring[0] > 0).float() * share).sum(-1).detach() if share is not None
                      else current.h.new_zeros(1))
        if replay:
            if tick == target:
                break
        elif tick == max_ticks or (fired if physical else bool(torch.bernoulli(hazard.detach()).item())):
            target = tick
            break
    if not replay and schedule_box is not None:
        if len(schedule_box) == 0:
            schedule_box.append(target)
        else:
            schedule_box[0] = target
    after = current.h.new_zeros(0)
    if silence_after:
        # The world is silent for `silence_after` ticks before the next word arrives; the read at the end of the
        # silence is the prediction that the next word will meet (the brain must carry the context through it).
        for _ in range(silence_after):
            baseline = (writer.lambda_adapt * current.baseline
                        if writer_baseline_clock == 'physical' and writer is not None else current.baseline)
            current = step_fly_physical_tick(model, current, None, quiet_source, baseline, options,
                                             h_mean_decay=h_mean_decay, base_rates=base_rates)
        after = read_fly_motor(model, current)
    current = expect_next(model, current, after if silence_after else latents[-1])
    lookahead = current.h.new_zeros(0)
    if physical:
        # Counterfactual: what the read would have been one quiet tick later, had the go neurons not fired.  Not part
        # of the realized life (the next input comes next); used only as the baseline of the firing tick's credit.
        with torch.no_grad():
            baseline = (writer.lambda_adapt * current.baseline
                        if writer_baseline_clock == 'physical' and writer is not None else current.baseline)
            later = step_fly_physical_tick(model, current, None, quiet_source, baseline, options,
                                           h_mean_decay=h_mean_decay, base_rates=base_rates)
            read_in = (later.h - later.h_mean if getattr(model, 'read_centering', False)
                       and later.h_mean.numel() == later.h.numel() else later.h)
            later_motor = read_in[:, model.read_indices] if model.read_surface != 'all' else read_in
            lookahead = model.output_read(motor_feature(model, later, later_motor))
    return current, torch.cat(latents, 0), torch.cat(hazards, 0), torch.cat(energy, 0), lookahead, after


def go_spike_credit(ce, spikes, ce_later):
    """Realized-life objective for a go act that is a real spike.  Forward value: the prediction error at the tick
    the go neurons actually fired (nothing counterfactual enters the value).  Backward: the spike at an earlier tick t
    gets credit CE_t - CE_fire (fire then if that would have been better, hold otherwise); the firing spike gets
    CE_fire - CE_next (hold if one more tick would have helped), through the network's own surrogate gradient."""
    credit = torch.cat([ce[:-1].detach() - ce[-1].detach(), (ce[-1].detach() - ce_later).reshape(1)])
    return ce[-1] + ((spikes - spikes.detach()) * credit).sum()


def go_free_energy(ce, hazard, energy, tick_cost, reference_energy, entropy_term=True):
    """Variational free energy of one token's halting.  q(t) is the go neuron's first-firing distribution truncated to
    the realized ticks (the unrealized tail mass sits on the realized last tick).  The prior is the Boltzmann
    distribution of the energy spent, p(t) ~ exp(-E(t) / B), with B set so that one tick of average activity
    (reference_energy) costs `tick_cost` nats.  F = E_q[CE] + E_q[E(t)] / B - H(q), up to the normalizer."""
    tau = ce.numel()
    survive = torch.cumprod(torch.cat([hazard.new_ones(1), 1 - hazard[:-1]]), 0)        # P(not fired before t)
    q = survive * hazard
    q = torch.cat([q[:-1], survive[-1:]]) if tau > 1 else survive
    expected = (q * ce).sum()
    if tick_cost:
        expected = expected + (q * torch.cumsum(energy, 0) / reference_energy * tick_cost).sum()
    if entropy_term:
        expected = expected + (q * torch.log(q.clamp_min(1e-12))).sum()
    return expected, q


READ_BANK_PARAMETER_NAMES = ('read_bank_gain', 'read_bank_logit_decay')
INTRINSIC_PARAMETER_NAMES = ('intrinsic_kappa',)
PRECISION_PARAMETER_NAMES = ('precision_u', 'precision_b', 'modulator_logit_decay')
MODULATORY_TYPE_PATTERN = r'^(PAM\d|PPL\d|OA-|5-HT|DPM$)'
DOPAMINE_TYPE_PATTERN = r'^(PAM\d|PPL\d)'


def attach_precision(model, graph_npz, annotations, tau=8.0, pattern=MODULATORY_TYPE_PATTERN):
    """Precision carried by the connectome's own neuromodulatory cells (dopaminergic PAM/PPL, octopaminergic OA-*,
    serotonergic 5-HT*, DPM): their spikes form a slow concentration trace m (time constant tau ticks, learnable); the
    written innovation of each sensory pathway (visual, chemo, mechano) is scaled by a = 2 sigmoid(u . m + b), starting at
    exactly 1.  The dopaminergic part of m also sets the learning rate of the synapses (see FlyBPTTLearner)."""
    import numpy as np
    import pandas as pd
    ids = np.load(graph_npz)['neuron_body_ids']
    table = pd.read_feather(annotations)[['bodyId', 'type']]
    position = {int(b): i for i, b in enumerate(ids)}
    chosen = table[table['type'].astype(str).str.contains(pattern, regex=True)]
    rows = sorted({position[int(b)] for b in chosen['bodyId'] if int(b) in position})
    dopamine = set(position[int(b)] for b in table[table['type'].astype(str).str.contains(DOPAMINE_TYPE_PATTERN, regex=True)]['bodyId']
                   if int(b) in position)
    device = model.read_indices.device
    model.register_buffer('modulatory_index', torch.as_tensor(rows, dtype=torch.long, device=device), persistent=False)
    model.register_buffer('dopamine_mask', torch.as_tensor([r in dopamine for r in rows], device=device), persistent=False)
    model.precision_u = nn.Parameter(torch.zeros(3, len(rows), device=device))
    model.precision_b = nn.Parameter(torch.zeros(3, device=device))
    decay = math.exp(-1.0 / tau)
    model.modulator_logit_decay = nn.Parameter(torch.tensor(math.log(decay / (1 - decay)), device=device))
    writer = model.topographic_writer
    pathway = torch.zeros(model.n_neurons, dtype=torch.long, device=device)      # which pathway each neuron's input uses
    pathway[writer.idx_chemo] = 1
    pathway[writer.idx_mech] = 2
    injected = torch.zeros(model.n_neurons, dtype=torch.bool, device=device)
    injected[writer.injection_index] = True

    def scale(gain):
        per_neuron = gain[:, pathway]
        return torch.where(injected, per_neuron, torch.ones_like(per_neuron))
    model.precision_scale = scale
    return len(rows), int(sum(r in dopamine for r in rows))
# Cell types known to be spontaneously / tonically active, as annotated in the MaleCNS connectome: circadian clock
# neurons (pacemakers), dopaminergic PAM / PPL, octopaminergic OA-*, serotonergic 5-HT*, DPM and APL.
INTRINSIC_TYPE_PATTERN = r'^(s-LNv|l-LNv|5thsLNv|LNd|DN1|DN2|DN3|LPN|LN-DN1|PAM\d|PPL\d|OA-|5-HT|DPM$|APL$)'


def attach_intrinsic_drive(model, graph_npz, annotations, kappa=1.2, pattern=INTRINSIC_TYPE_PATTERN):
    """Release the spontaneously active cells the connectome already contains: each selected neuron gets a learnable
    intrinsic current kappa (in units of its rheobase, start 1.2 = slow pacemaking, period ~1.8 tau_m).  Every other
    neuron is unchanged.  Returns the selected cell types."""
    import numpy as np
    import pandas as pd
    ids = np.load(graph_npz)['neuron_body_ids']
    table = pd.read_feather(annotations)[['bodyId', 'type']]
    table = table[table['type'].astype(str).str.contains(pattern, regex=True)]
    position = {int(b): i for i, b in enumerate(ids)}
    chosen = sorted(position[int(b)] for b in table['bodyId'] if int(b) in position)
    device = model.read_indices.device
    model.register_buffer('intrinsic_index', torch.as_tensor(chosen, dtype=torch.long, device=device), persistent=False)
    model.intrinsic_kappa = nn.Parameter(torch.full((len(chosen),), float(kappa), device=device))
    return table['type'].astype(str).value_counts().to_dict()


def attach_read_bank(model, taus=(1, 2, 4, 8, 16, 32)):
    """A read without a window: every motor membrane also passes through exponential filters (initial time constants
    `taus`, learnable) that run along the whole tick stream across token boundaries and are never reset.  The read is
    output_read(sum_k g_k * f_k) with per-neuron gains; f_0 is the membrane now.  Gains start at (1, 0, ..., 0), so the
    read starts exactly as the champion's."""
    device = model.read_indices.device
    width = model.n_read if model.read_surface != 'all' else model.n_neurons
    gain = torch.zeros(1 + len(taus), width, device=device)
    gain[0] = 1.0
    model.read_bank_gain = nn.Parameter(gain)
    decay = torch.tensor([math.exp(-1.0 / t) for t in taus], device=device)
    model.read_bank_logit_decay = nn.Parameter(torch.log(decay / (1 - decay)))
    model.read_bank_decay = lambda: torch.sigmoid(model.read_bank_logit_decay)[:, None]
    return model


def motor_feature(model, state, motor):
    """The read's input: the membranes now, or the gained sum of the membrane filters when a read bank is attached."""
    gain = getattr(model, 'read_bank_gain', None)
    bank = getattr(state, 'read_bank', torch.empty(0))
    if gain is None or bank.numel() == 0:
        return motor
    return gain[0] * motor + (gain[1:] * bank).sum(0, keepdim=True)


PREDICTIVE_PARAMETER_NAMES = ('writer_prediction_logit',)
DOPAMINE_PARAMETER_NAMES = ('dopamine_sensitivity', 'dopamine_logit_fast', 'dopamine_logit_slow')


DOPAMINE_SITES = ('post', 'pre')


def dopamine_edge_scale(model, edge_suffix, local, site='post'):
    """Per-synapse scaling of the update by the dopamine at one end of each synapse: 'post' (the receiving neuron) or
    'pre' (the presynaptic terminal: in the fly dopamine acts on the Kenyon-cell axons and changes their output
    synapses onto the mushroom-body output neurons)."""
    if site not in DOPAMINE_SITES:
        raise ValueError(f'dopamine site must be one of {DOPAMINE_SITES}')
    return local[getattr(model, f'edge_{site}{edge_suffix}').long()]


def attach_dopamine_modulation(model, tau_fast=50.0, tau_slow=500.0, site='post', adapted=False, sensitivity_init=0.0):
    """Dopamine acts as a neuromodulator along the connectome's own projections (241,701 modulatory synapses of the
    392 dopaminergic cells onto 39,850 targets): the dopamine delivered to each neuron forms a fast concentration and a
    slow mean (time constants learnable, start 50 / 500 ticks); the phasic part (difference) scales that neuron's
    threshold with a per-superclass sensitivity starting at 0, and the concentration scales how much the synapses onto
    that neuron learn (FlyBPTTLearner); with site='pre' it scales how much the synapses *from* that neuron learn
    (dopamine at the presynaptic terminal, as at the fly's Kenyon-cell -> output-neuron synapses).  Nothing changes at
    the start.

    adapted=True: every target's receptors adapt to its own slow level, so both the threshold modulation and the
    plasticity scaling see the relative change (fast - slow) / slow of that target, not a fixed map of how much
    dopamine each target receives.  sensitivity_init sets the starting sensitivity of every superclass (positive =
    more excitable with more dopamine, as through the D1-like receptors that dominate in the fly)."""
    if not getattr(model, 'has_dopamine', False):
        raise ValueError('the graph has no dopaminergic projections')
    if site not in DOPAMINE_SITES:
        raise ValueError(f'dopamine site must be one of {DOPAMINE_SITES}')
    model.dopamine_site = site
    model.dopamine_adapted = bool(adapted)
    device = model.read_indices.device
    n_class = int(model.superclass_id.max()) + 1
    model.dopamine_sensitivity = nn.Parameter(torch.full((n_class,), float(sensitivity_init), device=device))
    model.dopamine_logit_fast = nn.Parameter(torch.tensor(math.log(math.exp(-1 / tau_fast) / (1 - math.exp(-1 / tau_fast))), device=device))
    model.dopamine_logit_slow = nn.Parameter(torch.tensor(math.log(math.exp(-1 / tau_slow) / (1 - math.exp(-1 / tau_slow))), device=device))
    return model


def attach_predictive_writer(model, beta=0.5):
    """Predictive-coding writer: the written innovation becomes input - expected input, where the expected input mixes
    the slow sensory adaptation baseline and the brain's own top-down prediction (its last spoken prediction turned
    into an expected word embedding and sent through the same sensory projection, like an efference copy):
    reference = (1 - beta) * baseline + beta * packet(expected), beta learnable (start 0.5)."""
    model.writer_prediction_logit = nn.Parameter(torch.tensor(math.log(beta / (1 - beta)), device=model.read_indices.device))
    return model


def write_input(model, state, token):
    """Sensory drive of one input word through the writer; predictive when a top-down expectation is available."""
    writer = model.topographic_writer
    if writer is None:
        return None, state.baseline
    expected = getattr(state, 'expected_input', torch.empty(0))
    if getattr(model, 'writer_prediction_logit', None) is not None and expected.numel():
        return writer.forward_with_state(model.embedding(token), state.h, state.baseline, expected=expected,
                                         beta=torch.sigmoid(model.writer_prediction_logit))
    return writer.forward_with_state(model.embedding(token), state.h, state.baseline)


def expect_next(model, state, latent):
    """After speaking a prediction, keep its expected word embedding as the top-down prediction of the next input."""
    if getattr(model, 'writer_prediction_logit', None) is None:
        return state
    probs = torch.softmax(model.decoder(model.read_norm(latent)).float(), -1)
    expected = probs @ model.embedding.weight.float()
    return FlyPhysicalState(**{**state.state_dict(), 'expected_input': expected.to(latent.dtype)})


def read_fly_motor(model, state):
    """Linear motor read: the Gamma integrator of the motor membranes when enabled, else the membranes now."""
    if getattr(model, 'use_read_gamma_trace', False) and state.gamma_z2.numel() > 0:
        return model.output_read(state.gamma_z2)
    read_in = (state.h - state.h_mean
               if getattr(model, 'read_centering', False) and state.h_mean.numel() == state.h.numel() else state.h)
    motor = read_in[:, model.read_indices] if model.read_surface != 'all' else read_in
    return model.output_read(motor_feature(model, state, motor))


def _advance_fly_token_stream(model, state, token, schedule_box=None, *, ticks, hold_input=False, read_after=None,
                              writer_baseline_clock='physical', base_rates=None, thresholds=None,
                              conductance_gains=None, alif_params=None, stp_params=None, h_mean_decay=0.99):
    """Streaming clock: every token occupies exactly `ticks` physical ticks, whatever the brain is doing, and the
    prediction of the next token is read just before it arrives.  Nothing waits for the brain to settle, so the
    processing of a token continues while the next ones are being sensed (latency and throughput decouple).

    hold_input=False: the token is one sensory pulse at its first tick (the archived event form), quiet afterwards.
    hold_input=True: the token stays on the sensory surface for all its ticks; the writer's own adaptation turns the
    held stimulus into a decaying innovation.
    read_after=None: read after the last tick (the prediction of the NEXT token, read just before it arrives).
    read_after=D (1 <= D <= ticks): read after D ticks of THIS token's window -- output lags input.  The read is the
    prediction of this very token, made while it is being sensed: leak-free only while D is below the earliest
    sensory -> read-surface arrival (MaleCNS: direct sensory->motor edges of delay 1, so the read surface first changes
    at tick 2 and D = 1 is the maximum; verified bitwise in tests).
    """
    if not isinstance(ticks, int) or ticks < 1:
        raise ValueError('ticks must be a positive integer')
    if writer_baseline_clock not in ('input', 'physical'):
        raise ValueError('writer_baseline_clock must be input or physical')
    options = dict(base_rates=base_rates, thresholds=thresholds, conductance_gains=conductance_gains,
                   alif_params=alif_params, stp_params=stp_params)
    writer = model.topographic_writer
    embedding = model.embedding(token) if writer is not None else None
    if read_after is not None and not 1 <= read_after <= ticks:
        raise ValueError('read_after must lie inside the token window')
    current = state
    quiet_source = None
    latent = None
    for tick in range(ticks):
        if read_after is not None and tick == read_after:
            latent = read_fly_motor(model, current)
        if tick == 0 or (hold_input and writer is not None):
            drive, baseline = write_input(model, current, token)
            current = step_fly_physical_tick(model, current, token, drive, baseline, options,
                                             h_mean_decay=h_mean_decay, base_rates=base_rates)
        else:
            if quiet_source is None:
                quiet_source = torch.zeros_like(current.h)
            baseline = (writer.lambda_adapt * current.baseline
                        if writer_baseline_clock == 'physical' and writer is not None else current.baseline)
            current = step_fly_physical_tick(model, current, None, quiet_source, baseline, options,
                                             h_mean_decay=h_mean_decay, base_rates=base_rates)
    latent = latent if latent is not None else read_fly_motor(model, current)
    return expect_next(model, current, latent), latent


def _advance_fly_token_adaptive(
    model, state, token, schedule_box=None, *,
    min_settle_ticks=3,
    max_settle_ticks=14,
    flux_baseline=0.048,
    writer_baseline_clock='physical',
    base_rates=None, thresholds=None,
    conductance_gains=None, alif_params=None,
    stp_params=None, h_mean_decay=0.99
):
    """Adaptive admission token advance with causal arrival floor and deterministic replay.

    Replay Contract:
    - If schedule_box contains an integer, executes exactly that many quiet ticks without re-evaluating stopping conditions.
    - If schedule_box is empty or contains None, evaluates adaptive stopping and populates schedule_box[0] with the chosen ticks.
    - Information Arrival Guarantee: min_settle_ticks (default: 3) ensures sensory signals reach the readout surface.
    - Attractor Settlement: stops when flux drops below flux_baseline or stops decreasing (inflection).
    - Hard Ceiling: never exceeds max_settle_ticks (default: 14).
    """
    if not 0.0 <= h_mean_decay < 1.0:
        raise ValueError('h_mean_decay must be in [0, 1)')
    if writer_baseline_clock not in ('input', 'physical'):
        raise ValueError('writer_baseline_clock must be input or physical')

    drive, baseline = write_input(model, state, token)
    options = dict(base_rates=base_rates, thresholds=thresholds,
                   conductance_gains=conductance_gains,
                   alif_params=alif_params, stp_params=stp_params)

    curr_state = step_fly_physical_tick(
        model, state, token, drive, baseline, options,
        h_mean_decay=h_mean_decay, base_rates=base_rates
    )
    quiet_source = torch.zeros_like(curr_state.h)

    if schedule_box is not None and len(schedule_box) > 0 and schedule_box[0] is not None:
        # Deterministic Recomputation Pass under Gradient Checkpointing:
        # Replays EXACTLY the recorded schedule_box[0] ticks without re-evaluating stopping criteria.
        target_ticks = schedule_box[0]
        for t in range(1, target_ticks + 1):
            quiet_baseline = (model.topographic_writer.lambda_adapt * curr_state.baseline
                              if writer_baseline_clock == 'physical'
                              and model.topographic_writer is not None else curr_state.baseline)
            curr_state = step_fly_physical_tick(
                model, curr_state, None, quiet_source, quiet_baseline, options,
                h_mean_decay=h_mean_decay, base_rates=base_rates
            )
    else:
        # Forward Pass: evaluates adaptive relaxation with zero-grad flux
        sqrt_N = math.sqrt(model.n_neurons)
        prev_h = curr_state.h.detach()
        prev_flux = None
        target_ticks = max_settle_ticks

        for t in range(1, max_settle_ticks + 1):
            quiet_baseline = (model.topographic_writer.lambda_adapt * curr_state.baseline
                              if writer_baseline_clock == 'physical'
                              and model.topographic_writer is not None else curr_state.baseline)
            curr_state = step_fly_physical_tick(
                model, curr_state, None, quiet_source, quiet_baseline, options,
                h_mean_decay=h_mean_decay, base_rates=base_rates
            )
            with torch.no_grad():
                flux = torch.linalg.vector_norm(curr_state.h - prev_h).item() / sqrt_N
                prev_h = curr_state.h.detach()
                if t >= min_settle_ticks:
                    if flux <= flux_baseline or (prev_flux is not None and flux >= prev_flux):
                        target_ticks = t
                        break
                prev_flux = flux

        if schedule_box is not None:
            if len(schedule_box) == 0:
                schedule_box.append(target_ticks)
            else:
                schedule_box[0] = target_ticks

    # Extract motor latent at terminal settled state
    if getattr(model, 'use_read_gamma_trace', False) and curr_state.gamma_z2.numel() > 0:
        latent = model.output_read(curr_state.gamma_z2)
    else:
        latent = read_fly_motor(model, curr_state)

    return expect_next(model, curr_state, latent), latent


def predict_fly_next(model, state, token, *, settle_ticks, writer_baseline_clock,
                     adaptive_admission=False, min_settle_ticks=3, flux_baseline=0.048,
                     stream_ticks=0, hold_input=False):
    """Label-free generation/read interface with the same event order as BPTT.

    The caller must supply the checkpoint's timing explicitly. Sampling or
    teacher-forcing the next token happens only after this function returns.
    """
    if stream_ticks:
        next_state, z = advance_fly_token_stream(
            model, state, token, ticks=stream_ticks, hold_input=hold_input,
            writer_baseline_clock=writer_baseline_clock, base_rates=model.get_decay_rates(),
            thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
        return model.decoder(model.read_norm(z)), next_state
    if adaptive_admission:
        rates, thresholds = model.get_decay_rates(), model.get_thresholds()
        gains, alif, stp = model.get_conductance_gains(), model.get_alif_params(), model.get_stp_params()
        next_state, z = advance_fly_token_adaptive(
            model, state, token,
            min_settle_ticks=min_settle_ticks,
            max_settle_ticks=settle_ticks if settle_ticks > 0 else 14,
            flux_baseline=flux_baseline,
            writer_baseline_clock=writer_baseline_clock,
            base_rates=rates, thresholds=thresholds,
            conductance_gains=gains, alif_params=alif, stp_params=stp
        )
        normed = model.read_norm(z) if hasattr(model, 'read_norm') else z
        return model.decoder(normed), next_state

    state, tick_states = advance_fly_input_event(
        model, state, token, settle_ticks=settle_ticks,
        writer_baseline_clock=writer_baseline_clock, return_ticks=True)
    if settle_ticks == 0 or len(tick_states) <= 1:
        if getattr(model, 'use_read_gamma_trace', False) and state.gamma_z2.numel() > 0:
            z = model.output_read(state.gamma_z2)
            if hasattr(model, 'read_norm'):
                z = model.read_norm(z)
            return model.decoder(z), state
        z = extract_fly_motor_latent(model, state)
        if getattr(model, 'graph_observer', None) is not None:
            prior = state.observer_prior if (state.observer_prior is not None and state.observer_prior.numel() > 0) else None
            history = state.observer_history if (hasattr(state, 'observer_history') and state.observer_history is not None and state.observer_history.numel() > 0) else None
            readout_features, _, _, next_prior, next_history = model.graph_observer.forward_window(
                state.h, prior_state=prior, prior_history=history, physical_states=[state],
                anchor_features=z
            )
            state.observer_prior = next_prior.detach()
            state.observer_history = next_history.detach()
            normed_readout = model.read_norm(readout_features) if hasattr(model, 'read_norm') else readout_features
            return model.decoder(normed_readout), state
        if getattr(model, 'latent_predictor', None) is not None:
            z, _, _ = model.latent_predictor.rollout_dagger(z)
        return model.decoder(z), state

    # CTM multi-tick inference: evaluate certainty across physical settling ticks and select argmax certainty
    logits_list = []
    certs = []
    log_V = math.log(model.decoder.out_features)
    for st in tick_states:
        z_k = extract_fly_motor_latent(model, st)
        logits_k = model.decoder(z_k)
        log_p = F.log_softmax(logits_k, dim=-1)
        p = torch.exp(log_p)
        entropy = -(p * log_p).sum(dim=-1)
        cert = 1.0 - entropy / log_V
        logits_list.append(logits_k)
        certs.append(cert)
    certs_t = torch.stack(certs, dim=1)
    best_k = certs_t.argmax(dim=1).item()
    return logits_list[best_k], state


class FlyBPTTLearner:
    def __init__(self, model, state, *, lr=2e-4, lr_synapse=None,
                 lr_sensory=None, max_grad_norm=1.0, adam_names=None,
                 plasticity_optimizer='adamw', lr_decoder=None, settle_ticks=0,
                 writer_baseline_clock='input', learn_stp=False,
                 lambda_jepa=0.1, use_ctm_loss=True, use_checkpointing=False,
                 lambda_mcr2=0.0, eps_mcr2=0.5,
                 adaptive_admission=False, min_settle_ticks=3, flux_baseline=0.048,
                 stream_ticks=0, hold_input=False, read_lag=0, go_neuron=False, go_tick_cost=0.0,
                 go_reference_energy=0.07, go_max_ticks=32, lr_read_bank=None):
        if not isinstance(stream_ticks, int) or stream_ticks < 0:
            raise ValueError('stream_ticks must be a nonnegative integer')
        if read_lag and not (stream_ticks and 1 <= read_lag <= stream_ticks):
            raise ValueError('read_lag needs the streaming clock and must lie inside the token window')
        # Output lag: the prediction of a token is read read_lag ticks after that token began to be sensed.
        self.read_lag = int(read_lag)
        # Streaming clock (stream_ticks > 0) replaces the serial pulse-settle-read event and adaptive admission.
        self.stream_ticks = stream_ticks
        self.hold_input = bool(hold_input)
        # Go neuron (self-paced body): the brain's own motor act decides when to speak and take the next input.
        # go_neuron: False, 'virtual' (attach_go_neuron) or 'connectome' (attach_connectome_go, e.g. giant fibers).
        self.go_neuron = ('connectome' if getattr(model, 'go_neurons', None) is not None else 'virtual') if go_neuron else False
        if self.go_neuron and stream_ticks:
            raise ValueError('the go neuron (self-paced) replaces the streaming clock')
        if self.go_neuron == 'virtual' and not hasattr(model, 'go_read'):
            raise ValueError('a virtual go neuron needs attach_go_neuron(model)')
        self.go_tick_cost, self.go_reference_energy = float(go_tick_cost), float(go_reference_energy)
        self.go_max_ticks = int(go_max_ticks)          # safety ceiling only; the act decides
        self.adaptive_admission = bool(adaptive_admission) and stream_ticks == 0 and not self.go_neuron
        self.min_settle_ticks = int(min_settle_ticks)
        self.flux_baseline = float(flux_baseline)
        self.last_adaptive_metrics = {}
        self.use_ctm_loss = bool(use_ctm_loss)
        self.use_checkpointing = bool(use_checkpointing)
        self.last_train_loss = None
        self.last_ctm_metrics = {}
        self.lambda_jepa = float(lambda_jepa)
        self.last_jepa_loss = None
        self.last_jepa_metrics = {}
        self.lambda_mcr2 = float(lambda_mcr2)
        self.eps_mcr2 = float(eps_mcr2)
        self.last_mcr2_loss = None
        self.last_mcr2_metrics = {}
        if self.lambda_mcr2 > 0:
            from .mcr2_rate_distortion import MCR2Loss
            self.mcr2_criterion = MCR2Loss(d_model=getattr(model, 'd_model', 768), eps=self.eps_mcr2, beta=1.0)
        else:
            self.mcr2_criterion = None
        if plasticity_optimizer not in ('adamw', 'sgd'):
            raise ValueError('plasticity_optimizer must be adamw or sgd')
        if not isinstance(settle_ticks, int) or settle_ticks < 0:
            raise ValueError('settle_ticks must be a nonnegative integer')
        self.settle_ticks = settle_ticks
        if writer_baseline_clock not in ('input', 'physical'):
            raise ValueError('writer_baseline_clock must be input or physical')
        self.writer_baseline_clock = writer_baseline_clock
        lr_synapse = lr if lr_synapse is None else lr_synapse
        lr_sensory = lr if lr_sensory is None else lr_sensory
        self.model, self.state = model, state.detached()
        self.observer_prior = getattr(self.state, 'observer_prior', None)
        self.observer_history = getattr(self.state, 'observer_history', None)
        if self.state.h_mean.numel() != self.state.h.numel():
            self.state.h_mean = torch.zeros_like(self.state.h)
        if getattr(model, 'dan_plastic_lr', 0.0) > 0.0 and model.has_dopamine:
            if model.dan_scale <= 0:
                raise ValueError('The graph DAN sensitivity scale must be positive')
            if self.state.dan_gate.numel() != self.state.h.numel():
                self.state.dan_gate = torch.zeros_like(self.state.h)
        model.requires_grad_(False)
        # Promote existing learned edge buffers without altering topology or
        # values.  Layout-generic: COBA models carry edge_weight_e/i, other
        # models a single signed edge_weight.
        self.edge_names = ([name for name in ('edge_weight_e', 'edge_weight_i')
                            if hasattr(model, name)] if model.synapse_model == 'coba'
                           else ['edge_weight'])
        for name in self.edge_names:
            if name in model._buffers and model._buffers[name] is not None:
                value = model._buffers[name]
                del model._buffers[name]
                model.register_parameter(name, nn.Parameter(value))
        named = dict(model.named_parameters())
        self.adam_names = list(adam_names or (
            'output_read.weight', 'read_norm.weight', 'decoder.weight', 'decoder.bias',
            'log_threshold', 'log_tau_m', 'log_beta', 'log_tau_a',
            'log_tau_s_e', 'log_tau_s_i', 'log_g_e', 'log_g_i',
            'log_tau_s',
            'topographic_writer.gate_linear.weight', 'topographic_writer.gate_linear.bias'))
        self.adam_names = [name for name in self.adam_names if name in named]
        if getattr(model, 'use_read_gamma_trace', False) and getattr(model, 'logit_read_gamma', None) is not None:
            if 'logit_read_gamma' not in self.adam_names:
                self.adam_names.append('logit_read_gamma')
        if self.go_neuron == 'virtual':
            self.adam_names += [name for name in GO_PARAMETER_NAMES if name not in self.adam_names]
        if learn_stp:
            if not getattr(model, 'use_stp', False):
                raise ValueError('STP learning requires an STP-enabled model')
            for name in STP_PARAMETER_NAMES:
                if name not in named:
                    raise ValueError(f'Missing STP parameter: {name}')
                if name not in self.adam_names:
                    self.adam_names.append(name)
        self.learn_stp = all(name in self.adam_names for name in STP_PARAMETER_NAMES)
        if getattr(model, 'latent_predictor', None) is not None:
            for p_name, _ in model.latent_predictor.named_parameters():
                full_name = f'latent_predictor.{p_name}'
                if full_name in named and full_name not in self.adam_names:
                    self.adam_names.append(full_name)
        if getattr(model, 'graph_observer', None) is not None:
            for p_name, _ in model.graph_observer.named_parameters():
                full_name = f'graph_observer.{p_name}'
                if full_name in named and full_name not in self.adam_names:
                    self.adam_names.append(full_name)
        # O(1) scalars/gains: the go neuron's drive, reset and leak, the read bank, intrinsic currents.  (go_read.weight
        # entries are ~0.04 at their natural scale, where the base rate already moves them ~1% per step.)
        go_scalars = ('go_drive', 'go_reset', 'go_logit_leak')
        bank_names = [name for name in READ_BANK_PARAMETER_NAMES + INTRINSIC_PARAMETER_NAMES + PRECISION_PARAMETER_NAMES
                      + PREDICTIVE_PARAMETER_NAMES + DOPAMINE_PARAMETER_NAMES + go_scalars if name in named
                      and name in self.adam_names + list(READ_BANK_PARAMETER_NAMES + INTRINSIC_PARAMETER_NAMES
                                                         + PRECISION_PARAMETER_NAMES + PREDICTIVE_PARAMETER_NAMES
                                                         + DOPAMINE_PARAMETER_NAMES)]
        self.adam_names = [name for name in self.adam_names if name not in bank_names]
        decayed = [named[name] for name in self.adam_names
                   if name in ('output_read.weight', 'decoder.weight') or (('latent_predictor' in name or 'graph_observer' in name) and name.endswith('.weight'))]
        undecayed = [named[name] for name in self.adam_names
                     if name not in ('output_read.weight', 'decoder.weight') and not (('latent_predictor' in name or 'graph_observer' in name) and name.endswith('.weight'))]
        for name in self.adam_names:
            named[name].requires_grad_(True)
        if lr_decoder is None:
            adam_groups = [{'params': decayed, 'weight_decay': 1e-4},
                           {'params': undecayed, 'weight_decay': 0.0}]
        else:
            adam_groups = [
                {'params': [p for p in decayed if p is not model.decoder.weight],
                 'weight_decay': 1e-4, 'lr': lr},
                {'params': [model.decoder.weight], 'weight_decay': 1e-4, 'lr': lr_decoder},
                {'params': undecayed, 'weight_decay': 0.0, 'lr': lr}]
        if bank_names:
            # Gains are O(1) where synapses are ~1e-3; Adam steps ~lr per update, so the bank takes a scale-matched
            # rate (calibrated on a frozen-brain probe: 1e-2 moves the gains, 2e-4 leaves them at ~0.01 per 4k tokens).
            for n in bank_names:
                named[n].requires_grad_(True)
            adam_groups.append({'params': [named[n] for n in bank_names], 'weight_decay': 0.0,
                                'lr': lr_read_bank if lr_read_bank is not None else 50 * lr})
            self.adam_names += bank_names
        parameter_names = {id(p): n for n, p in named.items()}
        for group in adam_groups:
            group['parameter_names'] = [parameter_names[id(p)] for p in group['params']]
        self.optimizer = torch.optim.AdamW(adam_groups, lr=lr, fused=state.h.is_cuda)
        # The topographic writer exists only for topographic injection; other
        # injection modes have no sensory projection group to train.
        self.projections = ([getattr(model.topographic_writer, name).weight
                             for name in ('proj_vis', 'proj_chemo', 'proj_mech')]
                            if model.topographic_writer is not None
                            else [model.input_proj.weight])
        # Layout-generic edge collection: COBA models split excitatory and
        # inhibitory synapses into separate tensors, other models keep a
        # single signed tensor; the signed-aware clamp handles both.
        self.edges = [getattr(model, name) for name in self.edge_names]
        self.edge_positive = {name: weight.detach().ge(0).clone()
                              for name, weight in zip(self.edge_names, self.edges)}
        for parameter in self.projections + self.edges:
            parameter.requires_grad_(True)
        plasticity_groups = [
            {'params': self.projections, 'lr': lr_sensory, 'weight_decay': 1e-4},
            {'params': self.edges, 'lr': lr_synapse, 'weight_decay': 0.0}]
        self.plasticity_optimizer_kind = plasticity_optimizer
        # Retain the legacy attribute/checkpoint key for old continuation tools.
        # New default gives sensory projections and edges adaptive scaling too.
        if plasticity_optimizer == 'adamw':
            self.sgd = torch.optim.AdamW(plasticity_groups, fused=state.h.is_cuda)
        else:
            self.sgd = torch.optim.SGD(plasticity_groups, foreach=False)
        self.trainable = [p for p in model.parameters() if p.requires_grad]
        self.max_grad_norm = max_grad_norm
        self.events = self.updates = 0
        self.physical_ticks = 0
        self.previous_token = None
        self.ema = 0.0
        self.latent_window = torch.zeros(128, model.embedding.embedding_dim, device=state.h.device)
        self.runner = None
        # Explicit optional candidate: delayed DAN-pulse EMA modulates terminal
        # membrane coactivity once per window. This is not eligibility credit.
        self.dan_plastic_lr = float(getattr(model, 'dan_plastic_lr', 0.0))
        self._dan_setup()
        self.dan_gate = None
        self.observer_prior = None
        self.observer_history = None

    def _dan_setup(self):
        import numpy as np
        model = self.model
        if not getattr(model, 'has_dopamine', False) or self.dan_plastic_lr <= 0.0:
            self.dan_eligible = None
            return
        dan_post = model.dan_edge_post.detach().cpu().numpy()
        gated_targets = np.unique(dan_post)
        gated_set = np.zeros(model.n_neurons, dtype=bool)
        gated_set[gated_targets] = True
        self.dan_gate_target_index = torch.as_tensor(gated_targets, dtype=torch.long,
                                                     device=self.state.h.device)
        edge_sets = []
        if hasattr(model, 'edge_post_e'):
            edge_sets.append(('edge_weight_e', model.edge_post_e, model.edge_pre_e, (0.0, 5.0)))
        if hasattr(model, 'edge_post_i'):
            edge_sets.append(('edge_weight_i', model.edge_post_i, model.edge_pre_i, (0.0, 5.0)))
        if hasattr(model, 'edge_post'):
            edge_sets.append(('edge_weight', model.edge_post, model.edge_pre, (-5.0, 5.0)))
        eligible = {}
        for weight_name, post_tensor, pre_tensor, clamp in edge_sets:
            post_np = post_tensor.detach().cpu().numpy()
            mask = gated_set[post_np]
            idx = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long,
                                  device=self.state.h.device)
            if idx.numel():
                eligible[weight_name] = (idx, pre_tensor, post_tensor, clamp)
        self.dan_eligible = eligible or None

    def load_adam_state(self, saved, *, newly_trainable=()):
        """Retain per-parameter Adam history across a decoder group split.

        Old two-group checkpoints have no name metadata. Their ordering is
        determined by adam_names and the original matrix weight-decay policy.
        New checkpoints record explicit names, making future regrouping exact.
        Requested learning rates remain those of the destination optimizer.
        """
        newly_trainable = set(newly_trainable)
        allowed_new = (set(STP_PARAMETER_NAMES) | {'logit_read_gamma'} | set(GO_PARAMETER_NAMES)
                       | set(READ_BANK_PARAMETER_NAMES) | set(INTRINSIC_PARAMETER_NAMES) | set(PRECISION_PARAMETER_NAMES)
                       | set(PREDICTIVE_PARAMETER_NAMES) | set(DOPAMINE_PARAMETER_NAMES))
        if getattr(self.model, 'latent_predictor', None) is not None:
            allowed_new.update(f'latent_predictor.{n}' for n, _ in self.model.latent_predictor.named_parameters())
        if getattr(self.model, 'graph_observer', None) is not None:
            allowed_new.update(f'graph_observer.{n}' for n, _ in self.model.graph_observer.named_parameters())
        if not newly_trainable.issubset(allowed_new):
            raise ValueError(f'Only explicitly activated STP/Gamma/Predictor parameters may start new moments: {newly_trainable - allowed_new}')
        groups = saved['param_groups']
        if all('parameter_names' in group for group in groups):
            name_lists = [group['parameter_names'] for group in groups]
        elif len(groups) == 2:
            name_lists = [[n for n in self.adam_names if n not in newly_trainable
                           if n in ('output_read.weight', 'decoder.weight')],
                          [n for n in self.adam_names if n not in newly_trainable
                           if n not in ('output_read.weight', 'decoder.weight')]]
        else:
            raise ValueError('Optimizer checkpoint lacks parameter-name metadata')
        old_by_name, old_options = {}, {}
        for group, names in zip(groups, name_lists):
            if len(names) != len(group['params']):
                raise ValueError('Optimizer parameter ordering mismatch')
            for name, index in zip(names, group['params']):
                if name in old_by_name:
                    raise ValueError(f'Duplicate optimizer parameter: {name}')
                old_by_name[name] = saved['state'].get(index)
                old_options[name] = group
        for name in newly_trainable:
            old_by_name.pop(name, None)
            old_options.pop(name, None)
        for name in list(old_by_name):
            if name not in self.adam_names and (name.startswith('graph_observer.') or name.startswith('latent_predictor.')):
                old_by_name.pop(name, None)
                old_options.pop(name, None)
        if set(old_by_name) != set(self.adam_names) - newly_trainable:
            raise ValueError(f'Optimizer parameter coverage changed: extra={set(old_by_name) - (set(self.adam_names) - newly_trainable)}, missing={(set(self.adam_names) - newly_trainable) - set(old_by_name)}')
        destination = self.optimizer.state_dict()
        migrated = {}
        for group in destination['param_groups']:
            for index, name in zip(group['params'], group['parameter_names']):
                entry = old_by_name.get(name)
                if entry is not None:
                    migrated[index] = entry
            retained = [name for name in group['parameter_names'] if name in old_options]
            if retained:
                options = old_options[retained[0]]
                for key in ('betas', 'eps', 'amsgrad', 'maximize'):
                    if key in options:
                        group[key] = options[key]
            group['fused'] = self.state.h.is_cuda
        self.optimizer.load_state_dict({'state': migrated, 'param_groups': destination['param_groups']})
        if self.state.h.is_cuda:
            for entry in self.optimizer.state.values():
                if isinstance(entry.get('step'), torch.Tensor):
                    entry['step'] = entry['step'].cuda()

    def forward_window(self, input_ids, targets):
        """Every score precedes the single parameter update for this window."""
        m, state = self.model, self.state
        rates, thresholds = m.get_decay_rates(), m.get_thresholds()
        gains, alif, stp = m.get_conductance_gains(), m.get_alif_params(), m.get_stp_params()
        latents = []
        h_list = []
        state_list = []

        world = getattr(self, 'world', None) or {'mode': 'read'}
        if self.go_neuron and world['mode'] == 'listen':
            # Externally paced: every word occupies world['k'] ticks whatever the brain does; the prediction of the
            # next word is read just before it arrives.  The go neuron has no say.
            stream = dict(ticks=int(world['k']), hold_input=False, read_after=None,
                          writer_baseline_clock=self.writer_baseline_clock, base_rates=rates,
                          thresholds=thresholds, conductance_gains=gains, alif_params=alif, stp_params=stp)
            for token in input_ids.unbind(1):
                box = [int(world['k'])]
                if self.use_checkpointing and m.training and torch.is_grad_enabled():
                    state, token_lat = checkpoint(advance_fly_token_stream, m, state, token, box,
                                                  use_reentrant=False, **stream)
                else:
                    state, token_lat = advance_fly_token_stream(m, state, token, box, **stream)
                latents.append(token_lat)
            features = torch.cat(latents, 0)
            scores = F.cross_entropy(m.decoder(m.read_norm(features)), targets.flatten(), reduction='none')
            self.last_train_loss = scores
            self.last_adaptive_metrics = {'world': 'listen', 'world_k': int(world['k']),
                                          'adaptive_total_ticks': int(world['k']) * input_ids.size(1)}
            self.last_ctm_metrics, self.last_jepa_loss, self.last_jepa_metrics = {}, None, {}
            return scores, state, features
        if self.go_neuron:
            gaps = list(world.get('gaps') or [0] * input_ids.size(1))
            go = dict(max_ticks=self.go_max_ticks, writer_baseline_clock=self.writer_baseline_clock, base_rates=rates,
                      thresholds=thresholds, conductance_gains=gains, alif_params=alif, stp_params=stp)
            free_energy, realized, features, ticks, expected, energies = [], [], [], [], [], []
            gap_ticks = 0
            for token, target, gap in zip(input_ids.unbind(1), targets.flatten(), gaps):
                box = [None]
                if self.use_checkpointing and m.training and torch.is_grad_enabled():
                    state, lat, hazard, energy, later, after = checkpoint(
                        advance_fly_token_go, m, state, token, box, use_reentrant=False, silence_after=int(gap), **go)
                else:
                    state, lat, hazard, energy, later, after = advance_fly_token_go(
                        m, state, token, box, silence_after=int(gap), **go)
                gap_ticks += int(gap)
                ce = F.cross_entropy(m.decoder(m.read_norm(lat)), target.expand(lat.size(0)), reduction='none')
                if self.go_neuron == 'connectome':
                    with torch.no_grad():
                        ce_later = F.cross_entropy(m.decoder(m.read_norm(later)), target.reshape(1))
                    f = go_spike_credit(ce, hazard, ce_later)
                    q = torch.zeros_like(ce)
                    q[-1] = 1.0
                else:
                    f, q = go_free_energy(ce, hazard, energy, self.go_tick_cost, self.go_reference_energy,
                                          entropy_term=bool(self.go_tick_cost))
                if gap:
                    # the word after a silence meets the read at the end of the silence: that is what is scored,
                    # and it trains carrying the context through the gap
                    ce_after = F.cross_entropy(m.decoder(m.read_norm(after)), target.reshape(1)).reshape(())
                    f = f + ce_after
                    realized.append(ce_after)
                    features.append(after)
                else:
                    realized.append(ce[-1])
                    features.append(lat[-1:])
                free_energy.append(f)
                ticks.append(lat.size(0))
                expected.append(float((q.detach() * torch.arange(1, q.numel() + 1, device=q.device)).sum()))
                energies.append(float(energy.mean()))
            scores = torch.stack(realized)
            self.last_train_loss = torch.stack(free_energy)
            self.last_adaptive_metrics = {
                'go_ticks_mean': float(np.mean(ticks)), 'go_ticks_min': int(min(ticks)),
                'go_ticks_max': int(max(ticks)), 'go_expected_ticks_mean': float(np.mean(expected)),
                'go_energy_per_tick': float(np.mean(energies)), 'adaptive_total_ticks': int(sum(ticks)) + gap_ticks,
                'world': world['mode'], 'world_gap_ticks': gap_ticks,
                'go_ceiling_hits': int(sum(t == self.go_max_ticks for t in ticks))}
            self.last_ctm_metrics = {}
            self.last_jepa_loss = None
            self.last_jepa_metrics = {}
            return scores, state, torch.cat(features, 0)

        if self.stream_ticks:
            stream = dict(ticks=self.stream_ticks, hold_input=self.hold_input, read_after=self.read_lag or None,
                          writer_baseline_clock=self.writer_baseline_clock, base_rates=rates,
                          thresholds=thresholds, conductance_gains=gains, alif_params=alif, stp_params=stp)
            for token in input_ids.unbind(1):
                box = [self.stream_ticks]
                if self.use_checkpointing and m.training and torch.is_grad_enabled():
                    state, token_lat = checkpoint(advance_fly_token_stream, m, state, token, box,
                                                  use_reentrant=False, **stream)
                else:
                    state, token_lat = advance_fly_token_stream(m, state, token, box, **stream)
                latents.append(token_lat)
            features = torch.cat(latents, dim=0)
            logits = m.decoder(m.read_norm(features))
            scores = F.cross_entropy(logits, targets.flatten(), reduction='none')
            self.last_train_loss = scores
            self.last_adaptive_metrics = {}
            self.last_ctm_metrics = {}
            self.last_jepa_loss = None
            self.last_jepa_metrics = {}
            return scores, state, features

        if self.adaptive_admission:
            window_ticks = []
            max_ticks = self.settle_ticks if self.settle_ticks > 0 else 14
            for token in input_ids.unbind(1):
                box = [None]
                if self.use_checkpointing and m.training and torch.is_grad_enabled():
                    state, token_lat = checkpoint(
                        advance_fly_token_adaptive,
                        m, state, token, box,
                        min_settle_ticks=self.min_settle_ticks,
                        max_settle_ticks=max_ticks,
                        flux_baseline=self.flux_baseline,
                        writer_baseline_clock=self.writer_baseline_clock,
                        base_rates=rates,
                        thresholds=thresholds,
                        conductance_gains=gains,
                        alif_params=alif,
                        stp_params=stp,
                        use_reentrant=False
                    )
                else:
                    state, token_lat = advance_fly_token_adaptive(
                        m, state, token, box,
                        min_settle_ticks=self.min_settle_ticks,
                        max_settle_ticks=max_ticks,
                        flux_baseline=self.flux_baseline,
                        writer_baseline_clock=self.writer_baseline_clock,
                        base_rates=rates,
                        thresholds=thresholds,
                        conductance_gains=gains,
                        alif_params=alif,
                        stp_params=stp
                    )
                window_ticks.append(box[0] if box and box[0] is not None else max_ticks)
                latents.append(token_lat)
                h_list.append(state.h)
                state_list.append(state)

            features = torch.cat(latents, dim=0)
            normed_features = m.read_norm(features)
            logits = m.decoder(normed_features)
            scores = F.cross_entropy(logits, targets.flatten(), reduction='none')
            self.last_train_loss = scores
            n_toks = max(1, len(window_ticks))
            self.last_adaptive_metrics = {
                'adaptive_ticks_mean': float(sum(window_ticks) / n_toks),
                'adaptive_ticks_min': int(min(window_ticks)) if window_ticks else 0,
                'adaptive_ticks_max': int(max(window_ticks)) if window_ticks else 0,
                'adaptive_total_ticks': int(sum(window_ticks) + len(window_ticks)),
            }
            self.last_ctm_metrics = {}
            self.last_jepa_loss = None
            self.last_jepa_metrics = {}
            return scores, state, features

        K = 1 + self.settle_ticks
        use_ctm = (self.settle_ticks > 0 and self.use_ctm_loss)
        for token in input_ids.unbind(1):
            if self.use_checkpointing and m.training and torch.is_grad_enabled():
                state, token_lat = checkpoint(
                    advance_token_and_read_latent,
                    m, state, token,
                    settle_ticks=self.settle_ticks,
                    writer_baseline_clock=self.writer_baseline_clock,
                    base_rates=rates,
                    thresholds=thresholds,
                    conductance_gains=gains,
                    alif_params=alif,
                    stp_params=stp,
                    use_ctm=use_ctm,
                    use_reentrant=False
                )
                for k in range(token_lat.size(0)):
                    latents.append(token_lat[k])
                h_list.append(state.h)
                state_list.append(state)
            else:
                state, tick_states = advance_fly_input_event(
                    m, state, token,
                    settle_ticks=self.settle_ticks, writer_baseline_clock=self.writer_baseline_clock,
                    base_rates=rates,
                    thresholds=thresholds, conductance_gains=gains,
                    alif_params=alif, stp_params=stp, return_ticks=True)
                h_list.append(state.h)
                state_list.append(state)
                if use_ctm:
                    for st in tick_states:
                        if getattr(m, 'use_read_gamma_trace', False) and st.gamma_z2.numel() > 0:
                            latents.append(m.output_read(st.gamma_z2))
                        else:
                            read_in = (st.h - st.h_mean
                                       if getattr(m, 'read_centering', False)
                                       and st.h_mean.numel() == st.h.numel() else st.h)
                            motor = read_in[:, m.read_indices] if m.read_surface != 'all' else read_in
                            latents.append(m.output_read(motor))
                else:
                    if getattr(m, 'use_read_gamma_trace', False) and state.gamma_z2.numel() > 0:
                        latents.append(m.output_read(state.gamma_z2))
                    else:
                        read_in = (state.h - state.h_mean
                                   if getattr(m, 'read_centering', False)
                                   and state.h_mean.numel() == state.h.numel() else state.h)
                        motor = read_in[:, m.read_indices] if m.read_surface != 'all' else read_in
                        latents.append(m.output_read(motor))
        features = torch.cat(latents, dim=0)
        normed_features = m.read_norm(features)
        if getattr(m, 'graph_observer', None) is not None:
            if m.training and self.lambda_jepa > 0:
                teacher_quiet_h_list, quiet_states = collect_fly_quiet_trajectory(
                    m, state, num_ticks=m.graph_observer.max_horizon,
                    writer_baseline_clock=self.writer_baseline_clock,
                    base_rates=rates, thresholds=thresholds,
                    conductance_gains=gains, alif_params=alif, stp_params=stp,
                    return_h=True, return_states=True)
                teacher_quiet_h = torch.cat(teacher_quiet_h_list, dim=0)
                window_posts = [m.graph_observer.encoders(s).detach() for s in state_list[-4:]]
                query_pairs = collect_student_error_guided_queries(
                    m, quiet_states, m.graph_observer,
                    writer_baseline_clock=self.writer_baseline_clock,
                    base_rates=rates, thresholds=thresholds,
                    conductance_gains=gains, alif_params=alif, stp_params=stp,
                    max_queries=4,
                    prior_history=window_posts)
            else:
                teacher_quiet_h = None
                quiet_states = None
                query_pairs = None
            h_seq = torch.cat(h_list, dim=0)
            prior = state.observer_prior if (state.observer_prior is not None and state.observer_prior.numel() > 0) else getattr(self, 'observer_prior', None)
            history = state.observer_history if (hasattr(state, 'observer_history') and state.observer_history is not None and state.observer_history.numel() > 0) else getattr(self, 'observer_history', None)
            readout_features, obs_loss, obs_metrics, next_prior, next_history = m.graph_observer.forward_window(
                h_seq, prior_state=prior, prior_history=history, teacher_quiet_h=teacher_quiet_h, query_pairs=query_pairs,
                physical_states=state_list, teacher_quiet_states=quiet_states,
                anchor_features=features
            )
            state.observer_prior = next_prior.detach()
            state.observer_history = next_history.detach()
            self.observer_prior = next_prior.detach()
            self.observer_history = next_history.detach()
            normed_readout = m.read_norm(readout_features)
            logits = m.decoder(normed_readout)
            scores = F.cross_entropy(logits, targets.flatten(), reduction='none')
            self.last_train_loss = scores
            self.last_ctm_metrics = {}
            self.last_jepa_loss = obs_loss
            self.last_jepa_metrics = obs_metrics
        elif getattr(m, 'latent_predictor', None) is not None:
            if m.training and self.lambda_jepa > 0:
                teacher_hops, _ = collect_fly_quiet_trajectory(
                    m, state, num_ticks=m.latent_predictor.max_horizon,
                    writer_baseline_clock=self.writer_baseline_clock,
                    base_rates=rates, thresholds=thresholds,
                    conductance_gains=gains, alif_params=alif, stp_params=stp)
                teacher_hops_tensor = torch.cat(teacher_hops, dim=0)
            else:
                teacher_hops_tensor = None
            readout_features, jepa_loss, jepa_metrics = m.latent_predictor.rollout_dagger(
                normed_features, teacher_hops=teacher_hops_tensor
            )
            logits = m.decoder(readout_features)
            scores = F.cross_entropy(logits, targets.flatten(), reduction='none')
            self.last_train_loss = scores
            self.last_ctm_metrics = {}
            self.last_jepa_loss = jepa_loss
            self.last_jepa_metrics = jepa_metrics
        else:
            if use_ctm:
                W = input_ids.size(1)
                V = m.decoder.out_features
                all_logits = m.decoder(normed_features)
                logits_window = all_logits.view(W, K, V)
                target_expanded = targets.view(W, 1).expand(W, K)
                loss_all = F.cross_entropy(
                    logits_window.reshape(W * K, V),
                    target_expanded.reshape(-1),
                    reduction='none'
                ).view(W, K)

                log_p = F.log_softmax(logits_window, dim=-1)
                p = torch.exp(log_p)
                entropy = -(p * log_p).sum(dim=-1)
                log_V = math.log(V)
                certainty = 1.0 - entropy / log_V

                with torch.no_grad():
                    k_min_loss = loss_all.argmin(dim=1, keepdim=True)
                    k_max_cert = certainty.argmax(dim=1, keepdim=True)

                loss_min = loss_all.gather(1, k_min_loss).squeeze(1)
                loss_cert = loss_all.gather(1, k_max_cert).squeeze(1)
                self.last_train_loss = 0.5 * (loss_min + loss_cert)
                scores = loss_cert

                feat_dim = features.size(-1)
                feat_by_token = features.view(W, K, feat_dim)
                chosen_features = feat_by_token.gather(
                    1, k_max_cert.unsqueeze(-1).expand(W, 1, feat_dim)
                ).squeeze(1)

                if not (target_expanded.is_cuda and torch.cuda.is_current_stream_capturing()):
                    self.last_ctm_metrics = {
                        'ctm_k_min': float(k_min_loss.float().mean().item()),
                        'ctm_k_cert': float(k_max_cert.float().mean().item()),
                        'ctm_loss_min': float(loss_min.mean().item()),
                        'ctm_loss_cert': float(loss_cert.mean().item()),
                        'ctm_loss_tick0': float(loss_all[:, 0].mean().item()),
                        'ctm_loss_tickS': float(loss_all[:, -1].mean().item()),
                        'ctm_cert_mean': float(certainty.gather(1, k_max_cert).mean().item()),
                        'ctm_entropy_mean': float(entropy.gather(1, k_max_cert).mean().item()),
                    }
                else:
                    self.last_ctm_metrics = {}
                features = chosen_features
            else:
                logits = m.decoder(normed_features)
                scores = F.cross_entropy(logits, targets.flatten(), reduction='none')
                self.last_train_loss = scores
                self.last_ctm_metrics = {}

            self.last_jepa_loss = None
            self.last_jepa_metrics = {}
        return scores, state, features

    def inputs_for_targets(self, tokens):
        """Archived post-write next-token protocol; subclasses may align events.

        With an output lag the sensed token IS the target: its prediction is read before its signal can reach the
        read surface."""
        if getattr(self, 'read_lag', 0):
            return tokens[None]
        if self.previous_token is None:
            raise ValueError('A continuing previous token is required')
        return torch.cat((tokens.new_tensor([self.previous_token]), tokens[:-1]))[None]

    def observe(self, target_tokens):
        tokens = torch.as_tensor(target_tokens, device=self.state.h.device, dtype=torch.long).flatten()
        if tokens.numel() == 0:
            raise ValueError('At least one observed token is required')
        ids = self.inputs_for_targets(tokens)
        if self.runner is None:
            self.optimizer.zero_grad(set_to_none=True)
            self.sgd.zero_grad(set_to_none=True)
            scores, next_state, features = self.forward_window(ids, tokens[None])
        else:
            scores, next_state, features = self.runner.replay(ids, tokens[None])
        train_loss = getattr(self, 'last_train_loss', None)
        if train_loss is None:
            train_loss = scores
        loss = train_loss.mean()
        if getattr(self, 'last_jepa_loss', None) is not None:
            loss = loss + self.lambda_jepa * self.last_jepa_loss
        if getattr(self, 'lambda_mcr2', 0.0) > 0.0 and self.mcr2_criterion is not None and features is not None and features.size(0) >= 2:
            mcr2_out = self.mcr2_criterion(features)
            self.last_mcr2_loss = mcr2_out['loss']
            self.last_mcr2_metrics = {
                'mcr2_delta_R': float(mcr2_out['delta_R'].item()),
                'mcr2_R_total': float(mcr2_out['R_total'].item()),
                'mcr2_R_cluster': float(mcr2_out['R_cluster'].item()),
            }
            loss = loss + self.lambda_mcr2 * self.last_mcr2_loss
        if not torch.isfinite(loss).item():
            raise FloatingPointError('Non-finite BPTT training loss')
        if self.runner is None:
            from .triton_synapse import direct_edge_grad
            with direct_edge_grad():
                loss.backward()
        def norm(parameters):
            active = [p for p in parameters if p.requires_grad and p.numel()]
            if any(p.grad is None for p in active):
                raise RuntimeError('A nonempty optimized parameter has no task gradient')
            return float(stable_grad_norm(active)) if active else 0.0

        group_norms = {'synapse_grad_norm': norm(self.edges),
                       'writer_grad_norm': norm(self.projections)}
        # Keep the historical writer projection norm, but distinguish it from
        # writer gates, output projection, normalization gain and vocabulary.
        gate_parameters = (list(self.model.topographic_writer.gate_linear.parameters())
                           if self.model.topographic_writer is not None else [])
        group_norms.update(
            writer_gate_grad_norm=norm(gate_parameters),
            read_grad_norm=float(stable_grad_norm([self.model.output_read.weight])),
            decoder_grad_norm=float(stable_grad_norm([self.model.decoder.weight])),
            read_norm_grad_norm=float(stable_grad_norm([self.model.read_norm.weight])))
        group_norms['writer_total_grad_norm'] = (
            group_norms['writer_grad_norm']**2 + group_norms['writer_gate_grad_norm']**2)**.5
        group_norms['writer_gradient_rms'] = group_norms['writer_total_grad_norm'] / (
            max(1, sum(p.numel() for p in self.projections + gate_parameters))**.5)
        group_norms['synapse_gradient_rms'] = group_norms['synapse_grad_norm'] / (
            max(1, sum(p.numel() for p in self.edges))**.5)
        group_norms['read_gradient_rms'] = group_norms['read_grad_norm'] / (
            self.model.output_read.weight.numel()**.5)
        group_norms['decoder_gradient_rms'] = group_norms['decoder_grad_norm'] / (
            self.model.decoder.weight.numel()**.5)
        if getattr(self.model, 'use_read_gamma_trace', False) and getattr(self.model, 'logit_read_gamma', None) is not None:
            if self.model.logit_read_gamma.grad is not None:
                group_norms['read_gamma_grad_norm'] = float(stable_grad_norm([self.model.logit_read_gamma]))
        if self.learn_stp:
            for name in STP_PARAMETER_NAMES:
                group_norms[f'{name}_grad_norm'] = float(stable_grad_norm([getattr(self.model, name)]))
        if getattr(self.model, 'latent_predictor', None) is not None:
            pred_params = list(self.model.latent_predictor.parameters())
            group_norms['predictor_grad_norm'] = norm(pred_params)
            group_norms['predictor_gradient_rms'] = group_norms['predictor_grad_norm'] / (
                max(1, sum(p.numel() for p in pred_params))**.5)
            if getattr(self, 'last_jepa_metrics', None):
                group_norms.update(self.last_jepa_metrics)
        if getattr(self.model, 'graph_observer', None) is not None:
            obs_params = list(self.model.graph_observer.parameters())
            group_norms['observer_grad_norm'] = norm(obs_params)
            group_norms['observer_gradient_rms'] = group_norms['observer_grad_norm'] / (
                max(1, sum(p.numel() for p in obs_params))**.5)
            if getattr(self, 'last_jepa_metrics', None):
                group_norms.update(self.last_jepa_metrics)
        grad_norm = stable_clip_grad_norm_(self.trainable, self.max_grad_norm,
                                          error_if_nonfinite=True)
        self.optimizer.step()
        plasticity = 1.0
        if getattr(self.model, 'dopamine_mask', None) is not None and next_state.modulator.numel():
            # Dopamine sets how much the synapses learn from this window: their learning rate is scaled by the
            # dopaminergic concentration relative to its own running mean (a fixed rule, never learned from the loss,
            # which would simply switch learning off).
            level = float(next_state.modulator[0, self.model.dopamine_mask].mean())
            self.dopamine_mean = level if getattr(self, 'dopamine_mean', None) is None else (
                0.99 * self.dopamine_mean + 0.01 * level)
            plasticity = min(4.0, max(0.25, level / max(self.dopamine_mean, 1e-6)))
        base_rates = [group['lr'] for group in self.sgd.param_groups]
        self.sgd.param_groups[1]['lr'] = base_rates[1] * plasticity
        local = None
        if getattr(self.model, 'dopamine_sensitivity', None) is not None and next_state.dopamine.numel():
            # A fixed rule (never learned from the loss): the update of every synapse is scaled by the dopamine
            # concentration at its postsynaptic neuron (or, with model.dopamine_site == 'pre', at its presynaptic
            # terminal) relative to the mean over the dopamine targets (adapted: relative to that neuron's own slow
            # level), within [1/4, 4]; neurons that receive no dopamine keep the plain update.
            conc = next_state.dopamine[0].detach()
            targets = conc > 0
            local = torch.ones_like(conc)
            if bool(targets.any()):
                level = conc[targets].mean().clamp_min(1e-9)
                if getattr(self.model, 'dopamine_adapted', False):
                    slow = next_state.dopamine[1].detach()[targets]
                    local[targets] = (1.0 + (conc[targets] - slow) / (slow + level)).clamp(0.25, 4.0)
                else:
                    local[targets] = (conc[targets] / level).clamp(0.25, 4.0)
            before = [w.detach().clone() for w in self.edges]
        self.sgd.step()
        if local is not None:
            with torch.no_grad():
                site = getattr(self.model, 'dopamine_site', 'post')
                for name, w, old in zip(self.edge_names, self.edges, before):
                    w.copy_(old + (w - old) * dopamine_edge_scale(self.model, name[len('edge_weight'):], local, site))
            self.last_dopamine_plasticity_spread = float(local[local != 1].std()) if bool((local != 1).any()) else 0.0
        for group, rate in zip(self.sgd.param_groups, base_rates):
            group['lr'] = rate
        self.last_plasticity = plasticity
        with torch.no_grad():
            self.clamp_edges()
            if self.model.topographic_writer is not None:
                self.model.topographic_writer.a_adapt.copy_(next_state.baseline)
            self._dan_update(next_state)
            for i, feature in enumerate(features):
                self.latent_window[(self.events + i) % 128].copy_(feature)
        self.state = next_state.detached()
        values = scores.detach().cpu().tolist()
        for value in values:
            self.ema = value if self.events == 0 else .99 * self.ema + .01 * value
            self.events += 1
        self.previous_token = int(tokens[-1].item())
        if getattr(self, 'go_neuron', False):
            self.physical_ticks += self.last_adaptive_metrics['adaptive_total_ticks']
        elif getattr(self, 'stream_ticks', 0):
            self.physical_ticks += len(values) * self.stream_ticks
        elif self.adaptive_admission and getattr(self, 'last_adaptive_metrics', None):
            self.physical_ticks += self.last_adaptive_metrics.get('adaptive_total_ticks', len(values) * (1 + self.settle_ticks))
        else:
            self.physical_ticks += len(values) * (1 + self.settle_ticks)
        self.updates += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.sgd.zero_grad(set_to_none=True)
        if getattr(self.model, 'precision_u', None) is not None:
            group_norms['dopamine_plasticity'] = float(getattr(self, 'last_plasticity', 1.0))
        if getattr(self.model, 'dopamine_sensitivity', None) is not None:
            group_norms['dopamine_plasticity_spread'] = float(getattr(self, 'last_dopamine_plasticity_spread', 0.0))
        return values, {'grad_norm_before_clip': float(grad_norm), **group_norms, **getattr(self, 'last_ctm_metrics', {}), **getattr(self, 'last_adaptive_metrics', {})}

    @torch.no_grad()
    def clamp_edges(self):
        """Retain conductance positivity or each signed edge's original sign.

        Initially zero signed edges belong to the nonnegative class. Their
        mask is saved, so a clamped negative edge reaching zero stays negative.
        """
        for name, weight in zip(self.edge_names, self.edges):
            if self.model.synapse_model == 'coba':
                weight.clamp_(0.0, 5.0)
            else:
                weight.copy_(torch.where(self.edge_positive[name], weight.clamp(0.0, 5.0),
                                         weight.clamp(-5.0, 0.0)))

    @torch.no_grad()
    def _dan_update(self, terminal_state):
        """Delayed-pulse EMA times terminal pre/post membrane coactivity.

        Applied once per window, independently of BPTT. This optional candidate
        has no reward/error factor and does not claim biological credit accuracy.
        """
        if self.dan_eligible is None:
            return {}
        h = terminal_state.h[0]
        dan_drive = terminal_state.dan_gate[0]
        applied = {}
        for name, (idx, pre_tensor, post_tensor, clamp) in self.dan_eligible.items():
            weight = getattr(self.model, name)
            gate = dan_drive[post_tensor[idx]]
            pre_act = h[pre_tensor[idx]]
            post_act = h[post_tensor[idx]]
            delta = self.dan_plastic_lr * gate * pre_act * post_act
            with torch.no_grad():
                updated = (weight[idx] + delta).clamp(*clamp)
                if self.model.synapse_model != 'coba':
                    updated = torch.where(self.edge_positive[name][idx], updated.clamp_min(0),
                                          updated.clamp_max(0))
                weight.index_put_((idx,), updated)
            applied[name] = float(delta.abs().sum())
        self.dan_gate = dan_drive.detach()
        return applied

    def state_dict(self):
        return {'physical': self.state.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'sgd': self.sgd.state_dict(), 'adam_names': self.adam_names,
                'plasticity_optimizer_kind': self.plasticity_optimizer_kind,
                'events': self.events, 'updates': self.updates,
                'physical_ticks': self.physical_ticks, 'settle_ticks': self.settle_ticks,
                'writer_baseline_clock': self.writer_baseline_clock,
                'learn_stp': self.learn_stp,
                'read_centering': getattr(self.model, 'read_centering', False),
                'dan_plastic_lr': self.dan_plastic_lr,
                'dan_rule': 'delayed-pulse-ema-terminal-coactivity-v1',
                'edge_positive': self.edge_positive,
                'previous_token': self.previous_token, 'ema': self.ema,
                'latent_window': self.latent_window,
                'use_latent_predictor': getattr(self.model, 'use_latent_predictor', False),
                'use_graph_observer': getattr(self.model, 'use_graph_observer', False),
                'lambda_jepa': self.lambda_jepa,
                'use_ctm_loss': self.use_ctm_loss,
                'use_checkpointing': getattr(self, 'use_checkpointing', False),
                'lambda_mcr2': getattr(self, 'lambda_mcr2', 0.0),
                'eps_mcr2': getattr(self, 'eps_mcr2', 0.5),
                'adaptive_admission': getattr(self, 'adaptive_admission', False),
                'min_settle_ticks': getattr(self, 'min_settle_ticks', 3),
                'flux_baseline': getattr(self, 'flux_baseline', 0.048),
                'stream_ticks': getattr(self, 'stream_ticks', 0),
                'hold_input': getattr(self, 'hold_input', False),
                'read_lag': getattr(self, 'read_lag', 0),
                'go_neuron': getattr(self, 'go_neuron', False),
                'go_tick_cost': getattr(self, 'go_tick_cost', 0.0),
                'go_max_ticks': getattr(self, 'go_max_ticks', 32),
                'go_reference_energy': getattr(self, 'go_reference_energy', 0.07)}

    def load_edge_signs(self, saved):
        masks = saved.get('edge_positive')
        if masks is None:
            if self.model.synapse_model != 'coba':
                raise ValueError('A signed continuation requires saved edge signs; migrate from the original graph explicitly')
            return
        if set(masks) != set(self.edge_names):
            raise ValueError('Saved edge sign coverage mismatch')
        for name, mask in masks.items():
            if name not in self.edge_positive or mask.shape != self.edge_positive[name].shape:
                raise ValueError('Saved edge sign coverage mismatch')
            self.edge_positive[name] = mask.to(self.state.h.device, dtype=torch.bool)


class FlyBPTTGraph:
    """Capture forward/backward only; optimizer executes once per real window.

    Warm-up never updates weights or the continuing state. Inputs and all
    physical states are copied into fixed buffers before each actual replay.
    """
    def __init__(self, learner, window=32):
        if not learner.state.h.is_cuda:
            raise ValueError('CUDA Graph requires CUDA')
        self.learner, self.window = learner, window
        self.ids = torch.zeros(1, window, dtype=torch.long, device=learner.state.h.device)
        self.targets = torch.zeros_like(self.ids)
        original = learner.state
        self.initial = FlyPhysicalState(**{
            key: tuple(t.clone() for t in value) if key == 'ring' else value.clone()
            for key, value in original.state_dict().items()})

        def capture_step():
            learner.state = self.initial
            try:
                result = learner.forward_window(self.ids, self.targets)
                train_loss = getattr(learner, 'last_train_loss', None)
                if train_loss is None:
                    train_loss = result[0]
                train_loss.mean().backward()
                return result
            finally:
                learner.state = original

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                learner.model.zero_grad(set_to_none=True)
                warm_result = capture_step()
                del warm_result
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        learner.model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = capture_step()
        self.gradients = [(parameter, parameter.grad) for parameter in learner.trainable]
        if any(gradient is None for _, gradient in self.gradients):
            raise RuntimeError('A trainable parameter has no captured gradient')

    def replay(self, ids, targets):
        if ids.shape != self.ids.shape:
            raise ValueError('Captured BPTT windows must have the registered length')
        with torch.no_grad():
            for parameter, gradient in self.gradients:
                parameter.grad = gradient
                gradient.zero_()
            self.ids.copy_(ids)
            self.targets.copy_(targets)
            for key, value in self.learner.state.state_dict().items():
                destination = getattr(self.initial, key)
                if key == 'ring':
                    for dst, src in zip(destination, value):
                        dst.copy_(src)
                else:
                    destination.copy_(value)
        self.graph.replay()
        return self.output
