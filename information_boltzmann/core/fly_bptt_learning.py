"""32-event truncated BPTT for one continuing sensory/motor COBA individual.

Physical values persist between windows. Only autograd history is detached.
The spike backward is the model's existing ATan surrogate, not a derivative
of the discontinuous hard threshold. Static edge topology remains fixed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
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
    return FlyPhysicalState(h, ring, ge, gi, b, x, u, next_baseline, h_mean, dan_gate, gz1, gz2,
                            current.observer_prior, getattr(current, 'observer_history', torch.empty(0)))


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
    if model.topographic_writer is not None:
        drive, baseline = model.topographic_writer.forward_with_state(
            model.embedding(token), state.h, state.baseline)
    else:
        drive, baseline = None, state.baseline
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


def advance_fly_token_adaptive(
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

    if model.topographic_writer is not None:
        drive, baseline = model.topographic_writer.forward_with_state(
            model.embedding(token), state.h, state.baseline)
    else:
        drive, baseline = None, state.baseline
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
        read_in = (curr_state.h - curr_state.h_mean
                   if getattr(model, 'read_centering', False)
                   and curr_state.h_mean.numel() == curr_state.h.numel() else curr_state.h)
        motor = read_in[:, model.read_indices] if model.read_surface != 'all' else read_in
        latent = model.output_read(motor)

    return curr_state, latent


def predict_fly_next(model, state, token, *, settle_ticks, writer_baseline_clock,
                     adaptive_admission=False, min_settle_ticks=3, flux_baseline=0.048):
    """Label-free generation/read interface with the same event order as BPTT.

    The caller must supply the checkpoint's timing explicitly. Sampling or
    teacher-forcing the next token happens only after this function returns.
    """
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
                 adaptive_admission=False, min_settle_ticks=3, flux_baseline=0.048):
        self.adaptive_admission = bool(adaptive_admission)
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
        allowed_new = set(STP_PARAMETER_NAMES) | {'logit_read_gamma'}
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
        """Archived post-write next-token protocol; subclasses may align events."""
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
        self.sgd.step()
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
        if self.adaptive_admission and getattr(self, 'last_adaptive_metrics', None):
            self.physical_ticks += self.last_adaptive_metrics.get('adaptive_total_ticks', len(values) * (1 + self.settle_ticks))
        else:
            self.physical_ticks += len(values) * (1 + self.settle_ticks)
        self.updates += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.sgd.zero_grad(set_to_none=True)
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
                'flux_baseline': getattr(self, 'flux_baseline', 0.048)}

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
