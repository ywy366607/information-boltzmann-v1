"""32-event truncated BPTT for one continuing sensory/motor COBA individual.

Physical values persist between windows. Only autograd history is detached.
The spike backward is the model's existing ATan surrogate, not a derivative
of the discontinuous hard threshold. Static edge topology remains fixed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import os
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
    # Optional online dopamine plasticity (O(1), never reset): presynaptic activity trace e and the release-gain memory
    # phi written at the terminals that receive dopamine [1, N] each.
    da_trace: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    da_memory: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    # Optional homeostatic intrinsic drive of every neuron, in units of its own rheobase [1, N] (never reset).
    homeostasis: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    # Optional regime homeostasis [4, N] (never reset): the free membrane u (integrated like h but never reset), its
    # running mean and second moment, and the log input gain log s of every neuron.
    regime: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    # Optional compartment memory (never reset): the Kenyon cells' presynaptic activity trace [1, n_KC] and the memory
    # phi of every plastic Kenyon -> MBON synapse [1, n_edges] (its efficacy is (1 + phi) times its weight).
    comp_trace: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    comp_memory: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    # Optional carried credit (fly_carried_credit, never reset): the slow states' sensitivities to their parameters,
    # advanced every tick (forward mode), so credit outlives the BPTT window at O(1) memory.
    credit_comp: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    credit_comp_trace: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    credit_dopamine: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    credit_stp: torch.Tensor = field(default_factory=lambda: torch.empty(0))

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


def dopamine_class_sensitivity(model, like):
    """Per-class dopamine sensitivity spread to the neurons, [1, N].  A constant within a window (a parameter): computed
    once per window and passed to every tick as options['dopamine_s_class'] (2026-10-14: per tick it was a sparse
    product and its backward on every tick of every window)."""
    # per-superclass sensitivity spread to the neurons by a fixed one-hot product (an advanced-index gather here
    # has a sort-based, synchronizing backward: 2 ms per tick)
    n_class = model.dopamine_sensitivity.numel()
    onehot = getattr(model, '_superclass_onehot', None)
    if onehot is None or onehot.shape[-1 if n_class <= 256 else 0] != like.shape[1] or             onehot.shape[0 if n_class <= 256 else 1] != n_class:
        if n_class <= 256:
            onehot = F.one_hot(model.superclass_id.long(), n_class).T.to(like.dtype)
        else:
            # one class per cell type (11.7k classes): a dense [classes, neurons] one-hot would be 14 GB; the
            # same product as a sparse [neurons, classes] matrix
            n = like.shape[1]
            idx = torch.stack([torch.arange(n, device=like.device), model.superclass_id.long()])
            onehot = torch.sparse_coo_tensor(idx, torch.ones(n, device=like.device, dtype=like.dtype),
                                             (n, n_class)).coalesce()
        model._superclass_onehot = onehot
    if n_class <= 256:
        return (model.dopamine_sensitivity.to(onehot.dtype) @ onehot).unsqueeze(0)
    return torch.sparse.mm(onehot, model.dopamine_sensitivity.to(onehot.dtype).unsqueeze(1)).T


def step_fly_physical_tick(model, current, token, source, next_baseline, options, *,
                           h_mean_decay=0.99, base_rates=None):
    """Atomic physical step on the connectome with optional sensory drive."""
    credit = getattr(model, 'carried_credit', ())                    # fly_carried_credit (off: ())
    s_class_window = options.get('dopamine_s_class')
    if 'dopamine_s_class' in options:                                # a window constant, not a model.step option
        options = {k: v for k, v in options.items() if k != 'dopamine_s_class'}
    credit_comp, credit_comp_trace = current.credit_comp, current.credit_comp_trace
    credit_dopamine, credit_stp = current.credit_dopamine, current.credit_stp
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
    homeostasis = getattr(current, 'homeostasis', torch.empty(0))
    homeostatic = getattr(model, 'homeostasis_target', None) is not None
    if homeostatic:
        if homeostasis.numel() != current.h.numel():
            homeostasis = torch.zeros_like(current.h)
        # every neuron's intrinsic drive b (in units of its rheobase: b > 1 fires on its own) enters as a current
        rates_ = base_rates if base_rates is not None else model.get_decay_rates()
        theta_ = options.get('thresholds')
        theta_ = (theta_ if theta_ is not None else model.get_thresholds()).expand_as(current.h).detach()
        g_leak_ = -torch.log(rates_[0].expand_as(current.h).detach().clamp(1e-5, 1 - 1e-7))
        drive_ = homeostasis * theta_ * g_leak_
        source = drive_ if source is None else source + drive_
    regime = getattr(current, 'regime', torch.empty(0))
    regime_on = getattr(model, 'regime_quantile', None) is not None
    if regime_on:
        # Regime homeostasis (attach_regime_homeostasis): every input to a neuron -- synaptic conductances and the
        # writer's sensory drive -- is scaled by its gain s, and its location drive b enters as a current; both are
        # constants within the tick (detached state), so nothing of the future reaches the present.
        if homeostatic or model.synapse_model != 'coba':
            raise ValueError('regime homeostasis needs COBA synapses and excludes the rate homeostasis')
        theta_static = model.get_thresholds().expand_as(current.h).detach()
        if homeostasis.numel() != current.h.numel():
            homeostasis = torch.zeros_like(current.h)
        if regime.numel() != 4 * current.h.numel():
            h0 = current.h.detach()
            regime = torch.cat([h0, h0, h0 * h0 + (model.regime_kappa * theta_static) ** 2, torch.zeros_like(h0)], 0)
        gain = torch.exp(regime[3:4])
        if source is None:
            source = (model.topographic_writer(model.embedding(token), current.h)
                      if token is not None and model.injection_mode == 'topographic' else torch.zeros_like(current.h))
        rates_ = base_rates if base_rates is not None else model.get_decay_rates()
        g_leak_ = -torch.log(rates_[0].expand_as(current.h).detach().clamp(1e-5, 1 - 1e-7))
        source = source * gain + homeostasis * theta_static * g_leak_
        gains_ = options.get('conductance_gains')
        gains_ = gains_ if gains_ is not None else model.get_conductance_gains()
        options = {**options, 'conductance_gains': (gains_[0] * gain, gains_[1] * gain)}
    external = getattr(model, 'external_drive', None)
    if external is not None:
        # a current the world injects directly (e.g. driving the dopamine cells, as optogenetics does); added after the
        # input gain, so it is the nominal current
        if source is None:
            source = (model.topographic_writer(model.embedding(token), current.h)
                      if token is not None and model.injection_mode == 'topographic' else torch.zeros_like(current.h))
        source = source + external
    comp_on = getattr(model, 'comp_pre', None) is not None
    extra_ge = None
    comp_memory = getattr(current, 'comp_memory', torch.empty(0))
    if comp_on:
        if comp_memory.numel() != model.comp_pre.numel():
            comp_memory = current.h.new_zeros(1, model.comp_pre.numel())
        # the plastic Kenyon -> MBON synapses, outside the static edge list: efficacy (1 + phi) * w, same delays
        extra_ge = execute_delayed_synaptic_transmission(current.ring, model.comp_pre, model.comp_post,
                                                         model.comp_weight * (1.0 + comp_memory[0]), model.comp_splits)
    dopamine = getattr(current, 'dopamine', torch.empty(0))
    if getattr(model, 'dopamine_sensitivity', None) is not None and dopamine.numel():
        # Dopamine modulates excitability where it is delivered: the phasic part (concentration minus its slow mean)
        # scales the threshold, theta * exp(-s_class * phasic); s starts at 0 for every superclass.
        tonic = dopamine[1:2]
        # receptor affinity: a fixed half-saturation concentration (not the current mean level, whose reciprocal makes
        # the gradient explode when dopamine fades, as in a silent newborn)
        level = tonic.new_tensor(DOPAMINE_AFFINITY)
        if getattr(model, 'dopamine_adapted', False):
            # adapted receptors with half-saturation at the tonic level: a target's relative change
            # (fast - slow) / (slow + level), bounded for targets that receive only a trace of dopamine
            phasic = (dopamine[0:1] - tonic) / (tonic.detach() + level)
        else:
            # receptor occupancy c / (c + K) saturates: the phasic part is the occupancy of the fast concentration minus
            # that of its slow mean, bounded in (-1, 1); its gradient stays finite at both ends (silent and flooded)
            phasic = dopamine_occupancy(dopamine[0:1], model.dopamine_affinity) - dopamine_occupancy(tonic, model.dopamine_affinity)
        s_class = s_class_window if s_class_window is not None else dopamine_class_sensitivity(model, current.h)
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
    da_trace = getattr(current, 'da_trace', torch.empty(0))
    da_memory = getattr(current, 'da_memory', torch.empty(0))
    online = getattr(model, 'online_da_eta', None) is not None
    if online and da_memory.numel() == current.h.numel() and model.use_stp:
        # The memory written at a dopamine-receiving terminal shifts its baseline release probability (in the fly,
        # dopamine acts on Kenyon terminals through cAMP/PKA, i.e. on release); the vesicle pool x is depleted as
        # usual, so more release runs the terminal down faster and the output caps itself.
        stp = options.get('stp_params')
        stp = stp if stp is not None else model.get_stp_params()
        options = {**options, 'stp_params': (online_da_release(stp[0], da_memory), *stp[1:])}
    if regime_on:
        # the same COBA tick as model.step, in its two halves, so the integration coefficients are at hand for the
        # free membrane
        context = model.prepare_coba_tick(current.h, current.ring, current.ge, current.gi, current.b, current.x,
                                          current.u, **options, extra_delta_ge=extra_ge)
        ret = model.finish_coba_tick(context, source)
    else:
        ret = model.step(
            current.h, token, spike_ring=current.ring, ge=current.ge,
            gi=current.gi, b=current.b, x=current.x, u=current.u,
            i_syn=current.ge if model.synapse_model == 'cuba' else None,
            sensory_drive=source, **options, **({'extra_delta_ge': extra_ge} if extra_ge is not None else {}))
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
        if 'stp' in credit and current.u.numel() == h.numel() and current.x.numel() == h.numel():
            with torch.no_grad():
                from .fly_carried_credit import stp_step, stp_parameter_slopes
                if credit_stp.numel() != 5 * h.numel():
                    credit_stp = h.new_zeros(5, h.shape[1])
                stp_now = options.get('stp_params') or model.get_stp_params()
                u0, rho_f, rho_r = (t.detach() for t in stp_now[:3])
                spk = ret[1].detach() if ret[1].shape == h.shape else (ret[2][0] > 0).to(h.dtype)
                credit_stp = stp_step(credit_stp, current.u.detach(), current.x.detach(), spk, u0, rho_f, rho_r,
                                      *stp_parameter_slopes(model))
    h_mean = current.h_mean
    if h_mean.numel() != h.numel():
        h_mean = torch.zeros_like(h)
    h_mean = h_mean_decay * h_mean + (1.0 - h_mean_decay) * h
    gz1, gz2 = current.gamma_z1, current.gamma_z2
    if getattr(model, 'use_read_gamma_trace', False):
        if getattr(model, 'read_spikes', False):
            # The read surface's OUTPUT, as a muscle sees it: the transmitted pulses (spikes; graded release of
            # graded cells) through the two-stage Gamma filter.  The membrane read was blind to the reset (detached),
            # so the gradient asked every read cell for more drive and a cell that latched (membrane pinned at 0 by
            # reset/refractory) lost its gradient for good; the pulse carries the spike's surrogate gradient instead.
            read_in = ring[0]
        else:
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
        if getattr(model, 'kcd_pre', None) is not None:
            arrival = arrival + kenyon_terminal_release(model, ring[0])
        fast, slow = torch.sigmoid(model.dopamine_logit_fast), torch.sigmoid(model.dopamine_logit_slow)
        if dopamine.numel() != 2 * h.numel():
            dopamine = torch.cat([arrival, arrival], 0).detach()
        if 'dopamine' in credit:
            with torch.no_grad():
                from .fly_carried_credit import dopamine_step
                if credit_dopamine.numel() != 3 * h.numel():
                    credit_dopamine = h.new_zeros(3, h.shape[1])
                credit_dopamine = dopamine_step(credit_dopamine, dopamine[0:1].detach(), dopamine[1:2].detach(),
                                                arrival.detach(), fast.detach(), slow.detach())
        conc = fast * dopamine[0:1] + (1.0 - fast) * arrival
        dopamine = torch.cat([conc, slow * dopamine[1:2] + (1.0 - slow) * conc], 0)
    if online and dopamine.numel() == 2 * h.numel():
        # Three-factor online rule at the dopamine-receiving terminals (fly mushroom body: presynaptic activity and
        # dopamine, no postsynaptic factor):  e <- l e + (1 - l) s;  p = occ(D_fast) - occ(D_slow), occ(c) = c / (c + K);
        # phi <- phi + 2 occ(D_slow) e (eta p - kappa phi).  Only this tick's spikes and dopamine enter.
        spikes = ret[1] if ret[1].shape == h.shape else (ring[0] > 0).to(h.dtype)
        if da_trace.numel() != h.numel():
            da_trace, da_memory = torch.zeros_like(h), torch.zeros_like(h)
        lam = torch.sigmoid(model.online_da_logit_trace)
        da_trace = lam * da_trace + (1.0 - lam) * spikes
        tonic = dopamine[1:2]
        phasic = dopamine_occupancy(dopamine[0:1], model.dopamine_affinity) - dopamine_occupancy(tonic, model.dopamine_affinity)
        # dopamine opens a plasticity window at an active terminal: window = occupancy of its tonic dopamine / 0.5 * e;
        # inside it the memory is written (eta p) and the old content can be overwritten (kappa phi); outside it
        # (terminal silent or no dopamine) the memory stays as it is
        kappa = torch.sigmoid(model.online_da_logit_forget)
        window = dopamine_occupancy(tonic.detach(), model.dopamine_affinity) / 0.5 * da_trace     # 1 at the affinity concentration
        da_memory = da_memory + model.online_da_target * window * (model.online_da_eta * phasic - kappa * da_memory)
    if homeostatic:
        # homeostatic intrinsic plasticity (integral control toward the neuron's own target rate nu*):
        # b <- b + (nu* - s) / N_SPIKES -- about 1/100 of a rheobase per spike above target, a deficit closes within
        # ~100 of the neuron's own spikes.  A fixed local rule: no gradient flows through it.
        fired = (ring[0] > 0).to(h.dtype).detach()
        homeostasis = homeostasis.detach() + (model.homeostasis_target - fired) / HOMEOSTASIS_SPIKES
    if regime_on:
        # Location: b moves the free membrane until it is above the static threshold a fraction q* of the time
        # (b <- b + eta_b (q* - [u > theta])).  Width: the input gain moves the free membrane's running SD toward
        # kappa * theta (log s <- log s + eta_s log(kappa theta / sd)).  Fixed local rules on detached quantities.
        with torch.no_grad():
            u_free = context['alpha'] * regime[0:1] + context['beta_int'] * (context['base_current'] + source)
            above = (u_free > theta_static).to(h.dtype)
            homeostasis = homeostasis.detach() + model.regime_eta_b * (model.regime_quantile - above)
            w = model.regime_window_rate
            mean = regime[1:2] + w * (u_free - regime[1:2])
            second = regime[2:3] + w * (u_free * u_free - regime[2:3])
            sd = (second - mean * mean).clamp_min(1e-12).sqrt()
            log_gain = (regime[3:4] + model.regime_eta_s * torch.log(model.regime_kappa * theta_static / sd)).clamp(
                -REGIME_LOG_GAIN_MAX, REGIME_LOG_GAIN_MAX)
            regime = torch.cat([u_free, mean, second, log_gain], 0).detach()
    comp_trace = getattr(current, 'comp_trace', torch.empty(0))
    if comp_on:
        # Compartment memory (attach_compartment_memory): e <- l e + (1 - l) s at each Kenyon cell; at each plastic
        # synapse k -> j, with the dopamine delivered to its MBON j (the MBON marks the compartment):
        # phi <- phi + w (eta p_j - kappa phi), w = occ(D_slow,j) / 0.5 * e_k, p_j = occ(D_fast,j) - occ(D_slow,j).
        spikes_c = ret[1] if ret[1].shape == h.shape else (ring[0] > 0).to(h.dtype)
        if comp_trace.numel() != model.comp_kc.numel():
            comp_trace = h.new_zeros(1, model.comp_kc.numel())
        lam_c = torch.sigmoid(model.comp_logit_trace)
        trace_before, memory_before = comp_trace.detach(), comp_memory.detach()
        comp_trace = lam_c * comp_trace + (1.0 - lam_c) * spikes_c.index_select(1, model.comp_kc)
        if dopamine.numel() == 2 * h.numel():
            d_post = dopamine.index_select(1, model.comp_post.long())
            # only dopamine above its own running level teaches (a burst after Kenyon activity): the signed fast - slow
            # difference integrates to ~0 over a burst and its long undershoot, which the 50-tick trace spans, so a
            # pairing left almost nothing
            phasic_c = F.relu(dopamine_occupancy(d_post[0:1], model.comp_affinity)
                              - dopamine_occupancy(d_post[1:2], model.comp_affinity))
            window_c = (dopamine_occupancy(d_post[1:2].detach(), model.comp_affinity) / 0.5
                        * comp_trace.index_select(1, model.comp_edge_kc))
            kappa_c = torch.sigmoid(model.comp_logit_forget)
            # depression scaled by the efficacy left, (1 + phi): phi approaches -1 and never passes it, so forgetting
            # (kappa, slow) no longer has to equal the learning rate to keep phi bounded
            if 'comp' in credit:
                with torch.no_grad():
                    from .fly_carried_credit import comp_step, occupancy_slope, COMP_PARAMS
                    if credit_comp.numel() != len(COMP_PARAMS) * comp_memory.shape[1]:
                        credit_comp = h.new_zeros(len(COMP_PARAMS), comp_memory.shape[1])
                        credit_comp_trace = h.new_zeros(1, comp_trace.shape[1])
                    post = model.comp_post.long()
                    Sd = credit_dopamine.index_select(1, post)                # [3, E]: dfast/dlf, dslow/dlf, dslow/dls
                    dpost = d_post.detach()
                    on = (phasic_c.detach() > 0).float()
                    a_f = occupancy_slope(dpost[0:1], model.comp_affinity)
                    a_s = occupancy_slope(dpost[1:2], model.comp_affinity)
                    dphasic = on * torch.cat([a_f * Sd[0:1] - a_s * Sd[1:2], -a_s * Sd[2:3]], 0)
                    lam_d = lam_c.detach()
                    credit_comp, credit_comp_trace = comp_step(
                        credit_comp, credit_comp_trace, memory_before, trace_before,
                        spikes_c.detach().index_select(1, model.comp_kc), lam_d, model.comp_eta.detach(),
                        kappa_c.detach(), phasic_c.detach(),
                        dopamine_occupancy(d_post[1:2].detach(), model.comp_affinity) / 0.5, model.comp_edge_kc,
                        dphasic)
            comp_memory = comp_memory + window_c * (model.comp_eta * phasic_c * (1.0 + comp_memory) - kappa_c * comp_memory)
    bank = getattr(current, 'read_bank', torch.empty(0))
    if getattr(model, 'read_bank_gain', None) is not None:
        motor_now = h[:, model.read_indices] if model.read_surface != 'all' else h
        decay = model.read_bank_decay()
        if bank.numel() != decay.numel() * motor_now.numel():
            bank = motor_now.expand(decay.numel(), -1)
        bank = decay * bank + (1.0 - decay) * motor_now
    return FlyPhysicalState(h, ring, ge, gi, b, x, u, next_baseline, h_mean, dan_gate, gz1, gz2,
                            current.observer_prior, getattr(current, 'observer_history', torch.empty(0)), bank,
                            modulator, getattr(current, 'expected_input', torch.empty(0)), dopamine, da_trace, da_memory,
                            homeostasis, regime, comp_trace, comp_memory,
                            credit_comp, credit_comp_trace, credit_dopamine, credit_stp)


def extract_fly_motor_latent(model, state):
    """Extract and normalize motor readout latents from current brain state."""
    if getattr(model, 'use_read_gamma_trace', False) and state.gamma_z2.numel() > 0:
        lat = model.output_read(state.gamma_z2)
        return model.read_norm(lat) if hasattr(model, 'read_norm') else lat
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
                            return_ticks=False, hold_input=False):
    """One token pulse followed by quiet physical ticks, before next-token read.  hold_input=True: the token stays on
    the sensory surface for all its ticks (the world keeps sending it; the writer's own adaptation turns it into a
    decaying innovation) instead of quiet ticks."""
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
            if hold_input and model.topographic_writer is not None:
                held, held_baseline = write_input(model, state, token)
                state = step_fly_physical_tick(model, state, token, held, held_baseline, options,
                                               h_mean_decay=h_mean_decay, base_rates=base_rates)
            else:
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
                                  use_ctm=False, hold_input=False):
    """Atomic token advancement plus motor latent extraction for gradient checkpointing."""
    next_state, tick_states = advance_fly_input_event(
        model, state, token,
        settle_ticks=settle_ticks, hold_input=hold_input,
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
# neurons (pacemakers), dopaminergic PAM / PPL, octopaminergic OA-*, serotonergic 5-HT* and DPM.  APL is not here
# (2026-10-13): it is a graded feedback interneuron driven by the Kenyon cells (attach_graded_cells); a pacemaker
# current on it made it fire at the rate cap and hold the Kenyon layer under constant inhibition.  DPM is not here
# either (2026-10-13): it is serotonergic / GABAergic, not dopaminergic (moved to fast GABA in the birth graph).
INTRINSIC_TYPE_PATTERN = r'^(s-LNv|l-LNv|5thsLNv|LNd|DN1|DN2|DN3|LPN|LN-DN1|PAM\d|PPL\d|OA-|5-HT)'


def attach_intrinsic_drive(model, graph_npz, annotations, kappa=1.2, pattern=INTRINSIC_TYPE_PATTERN, jitter=0.0, seed=0):
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
    # jitter: each cell's kappa x (1 + jitter * N(0, 1)), fixed at birth -- identical pacemakers started from the same
    # state fire in lockstep (audit 2026-10-13: 441 cells at one 27 Hz rhythm); different periods desynchronise them
    k = torch.full((len(chosen),), float(kappa))
    if jitter > 0:
        k = k * (1.0 + jitter * torch.randn(len(chosen), generator=torch.Generator().manual_seed(seed)))
    model.intrinsic_kappa = nn.Parameter(k.to(device))
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
# Receptor half-saturation concentration, in the units of the delivered dopamine: the mean tonic dopamine at the targets
# of the trained lineage (0.0102-0.0108 in damod2_gb, dasite_pre, l1_*), so trained individuals see the same scale as
# before, while in a newborn whose dopamine cells are silent the phasic signal goes to 0 instead of blowing up.
DOPAMINE_AFFINITY = 0.0108


def dopamine_occupancy(concentration, affinity=DOPAMINE_AFFINITY):
    """Receptor occupancy c / (c + K); K = the model's dopamine_affinity when set (attach_dopamine_modulation), else
    DOPAMINE_AFFINITY."""
    return concentration / (concentration + affinity)
ONLINE_DA_PARAMETER_NAMES = ('online_da_eta', 'online_da_logit_trace', 'online_da_logit_forget')
def online_da_release(u0, memory):
    """Baseline release probability of every terminal with its memory phi: sigmoid(logit(u0) + phi) -- stays in
    (0, 1) -- and exactly u0 where phi is 0 (every terminal without dopamine, and every terminal before it learns)."""
    u0 = u0.expand_as(memory)
    return torch.where(memory != 0, torch.sigmoid(torch.logit(u0, eps=1e-6) + memory), u0)


HOMEOSTASIS_SPIKES = 100.0          # a neuron corrects its excitability over ~100 of its own spikes (1/sqrt(N) = 10%)
HOMEOSTASIS_DUTY_MEDIAN = 0.05      # median duty cycle nu* x tau_m: fluctuation-driven firing, the middle of 0.01-0.2
HOMEOSTASIS_DUTY_SIGMA = 1.0        # log-normal spread of the duty cycle across neurons (rates span ~2 decades)


def attach_homeostasis(model, graph_npz, seed=0, duty_median=HOMEOSTASIS_DUTY_MEDIAN, duty_sigma=HOMEOSTASIS_DUTY_SIGMA):
    """Homeostatic intrinsic plasticity for every neuron.  Each neuron has its own target rate nu* = c / tau_m (spikes
    per tick; tau_m its anatomical membrane time constant in ticks), the duty cycle c drawn once per individual from a
    log-normal (median 0.05, sigma 1; `seed` makes the individual); its intrinsic drive b (state) integrates
    (nu* - s) / 100 every tick, so a silent neuron becomes excitable until it fires spontaneously and an over-active one
    is held down, both on the scale of ~100 of its own spikes.  The connectome carries the wiring; this supplies the
    one cell property it lacks without fixing any rate by hand."""
    import numpy as np
    tau_m = torch.as_tensor(np.load(graph_npz)['tau_m'], dtype=torch.float32)
    gen = torch.Generator().manual_seed(int(seed))
    duty = torch.exp(math.log(duty_median) + duty_sigma * torch.randn(tau_m.numel(), generator=gen))
    target = (duty / tau_m.clamp_min(1.0)).clamp(max=0.2)
    model.register_buffer('homeostasis_target', target.unsqueeze(0).to(model.read_indices.device), persistent=False)
    model.homeostasis_seed = int(seed)
    return model


REGIME_WINDOWS = 100.0              # estimates over ~100 of the neuron's own membrane time constants (SD to ~14%)
REGIME_Z_MEAN = 1.0                 # threshold ~1 SD of the free membrane above its mean (q* = P(z > 1) ~ 0.16) ...
REGIME_Z_SD = 0.25                  # ... varying from neuron to neuron, drawn once per individual
REGIME_KAPPA = 0.5                  # free-membrane SD = theta / 2: +-2 SD spans rest to threshold
REGIME_LOG_GAIN_MAX = math.log(10.0)  # input gain within [1/10, 10] (E_E / theta): a safety rail for input-free cells


REGIME_KENYON_Z = 1.645             # Kenyon cells: q* = P(Z > 1.645) = 0.05, the coding level where a sparse expansion
                                    # code has the most dimensions (Litwin-Kumar et al. 2017: f ~ 0.05-0.1 for a few claws)


def attach_regime_homeostasis(model, graph_npz, seed=0, z_mean=REGIME_Z_MEAN, z_sd=REGIME_Z_SD, kappa=REGIME_KAPPA,
                              windows=REGIME_WINDOWS, kenyon_z=None, annotations=None):
    """Regime homeostasis (replaces the rate homeostasis of attach_homeostasis; design and measurements:
    docs/information_boltzmann/FLY_CONSEQUENCE_WORLD_DESIGN_20261012.html).  What is regulated is where each neuron's
    threshold sits inside the distribution of its own free membrane u (integrated like the membrane, never reset),
    not its firing rate:  location -- the additive drive b (rheobase units) keeps u above the static threshold a
    fraction q* of the time, q* = P(Z > z) with z ~ N(1, 0.25) drawn per neuron from `seed` (the individual);
    width -- a multiplicative gain s on every input of the neuron (excitatory and inhibitory conductances alike, so
    E/I and short-term plasticity are kept, and the writer's drive for sensory neurons) keeps the running SD of u at
    theta / 2.  A neuron is then fluctuation-driven by construction; its rate follows from q* and its input's
    correlation time.  Time scales: eta_b = 1 / (100 q* tau_m), eta_s = eta_b / 3 (the slower knob does not fight
    the faster), running moments over 100 tau_m.  kenyon_z (with the annotations) moves the Kenyon cells' mean z: an
    expansion layer codes best when sparse, so their thresholds sit further out in their own distribution (the same
    per-neuron draw, only shifted; every other neuron is unchanged)."""
    import numpy as np
    graph = np.load(graph_npz, allow_pickle=True)
    tau_m = torch.as_tensor(graph['tau_m'], dtype=torch.float32).clamp_min(1.0)
    gen = torch.Generator().manual_seed(int(seed))
    z = z_mean + z_sd * torch.randn(tau_m.numel(), generator=gen)
    if kenyon_z is not None:
        import pandas as pd
        cls = pd.read_feather(annotations).set_index('bodyId').reindex(graph['neuron_body_ids'])['class'].astype(str).values
        z = torch.where(torch.as_tensor(cls == 'Kenyon_Cell'), z + (float(kenyon_z) - z_mean), z)
    q = 0.5 * torch.erfc(z / math.sqrt(2.0))
    eta_b = 1.0 / (windows * q * tau_m)
    device = model.read_indices.device
    for name, value in (('regime_quantile', q), ('regime_eta_b', eta_b), ('regime_eta_s', eta_b / 3.0),
                        ('regime_window_rate', 1.0 / (windows * tau_m))):
        model.register_buffer(name, value.unsqueeze(0).to(device), persistent=False)
    model.regime_kappa = float(kappa)
    model.regime_seed = int(seed)
    model.regime_kenyon_z = None if kenyon_z is None else float(kenyon_z)
    return model


def attach_compartment_memory(model, graph_npz, annotations, eta=-0.02, tau_trace=50.0, forget=0.02, affinity=None):
    """Three-factor memory per Kenyon -> MBON synapse (the fly's own plastic site; design:
    FLY_CONSEQUENCE_WORLD_DESIGN_20261012.html).  A Kenyon cell's axon passes through compartments, each with its own
    dopamine cells and its own MBONs; in the connectome the MBONs separate cleanly by their dopamine source (51
    PPL1-dominated, 40 PAM-dominated, 6 mixed), while every Kenyon cell receives a PAM/PPL1 mixture.  So the memory
    lives on the synapse and reads the dopamine delivered to its MBON: phi <- phi + w (eta p - kappa phi), w = occ(slow
    dopamine) / 0.5 * presynaptic trace (no postsynaptic factor, as in the fly), efficacy (1 + phi) * weight; eta < 0:
    dopamine after Kenyon activity depresses (eta = -kappa keeps phi in (-1, 1)).  The excitatory Kenyon -> MBON edges
    leave the static edge list (pinned at 0) and are transmitted separately with the same delays.  Needs
    attach_dopamine_modulation (the dopamine state).  affinity: receptor half-saturation (default the model's)."""
    import pandas as pd
    if getattr(model, 'dopamine_sensitivity', None) is None:
        raise ValueError('compartment memory needs attach_dopamine_modulation first')
    graph = np.load(graph_npz, allow_pickle=True)
    cls = pd.read_feather(annotations).set_index('bodyId').reindex(graph['neuron_body_ids'])['class'].astype(str).values
    device = model.edge_pre_e.device
    kc = torch.as_tensor(cls == 'Kenyon_Cell', device=device)
    mbon = torch.as_tensor(cls == 'MBON', device=device)
    pre, post = model.edge_pre_e.long(), model.edge_post_e.long()
    edges = torch.nonzero(kc[pre] & mbon[post]).squeeze(1)              # ascending: already grouped by delay tier
    splits = torch.as_tensor(model.splits_e, device=device)
    tier = torch.searchsorted(splits[1:], edges, right=True)
    comp_splits = tuple([0] + [int((tier <= d).sum()) for d in range(4)])
    kc_index = torch.nonzero(kc).squeeze(1)
    with torch.no_grad():
        weight = model.edge_weight_e[edges].detach().clone()
        model.edge_weight_e[edges] = 0.0
    pinned = getattr(model, 'pinned_zero_edges_e', None)
    pinned = edges if pinned is None else torch.unique(torch.cat([pinned.to(device), edges]))
    model.register_buffer('pinned_zero_edges_e', pinned, persistent=False)
    model.register_buffer('comp_pre', pre[edges].to(model.edge_pre_e.dtype), persistent=False)
    model.register_buffer('comp_post', post[edges].to(model.edge_post_e.dtype), persistent=False)
    model.register_buffer('comp_kc', kc_index, persistent=False)
    model.register_buffer('comp_edge_kc', torch.searchsorted(kc_index, pre[edges]), persistent=False)
    model.comp_splits = comp_splits
    model.comp_weight = nn.Parameter(weight)
    model.comp_eta = nn.Parameter(torch.tensor(float(eta), device=device))
    lam = math.exp(-1.0 / tau_trace)
    model.comp_logit_trace = nn.Parameter(torch.tensor(math.log(lam / (1 - lam)), device=device))
    model.comp_logit_forget = nn.Parameter(torch.tensor(math.log(forget / (1 - forget)), device=device))
    model.comp_affinity = float(affinity if affinity is not None else model.dopamine_affinity)
    return int(edges.numel())


COMPARTMENT_PARAMETER_NAMES = ('comp_weight', 'comp_eta', 'comp_logit_trace', 'comp_logit_forget')
TASTE_PARAMETER_NAMES = ('taste_gain', 'taste_bias')
TASTE_YES, TASTE_NO = 3763, 645            # GPT-2 " yes", " no"


def taste_token_ticks(model, state, token, per_tick_drive, hold, options):
    """One word on a fixed clock for a taste episode: len(per_tick_drive) ticks, the word written at the first tick
    (or every tick when held), per_tick_drive[k] the external current at tick k (None: none).  Returns the state and
    every tick's transmitted pulses of the MBONs (differentiable) [ticks, n_MBON]."""
    writer = model.topographic_writer
    quiet, acts = None, []
    for k, drive in enumerate(per_tick_drive):
        model.external_drive = drive
        if k == 0 or (hold and writer is not None):
            source, baseline = write_input(model, state, token)
            state = step_fly_physical_tick(model, state, token, source, baseline, options, base_rates=options['base_rates'])
        else:
            quiet = torch.zeros_like(state.h) if quiet is None else quiet
            baseline = writer.lambda_adapt * state.baseline if writer is not None else state.baseline
            state = step_fly_physical_tick(model, state, None, quiet, baseline, options, base_rates=options['base_rates'])
        acts.append(state.ring[0][0, model.taste_mbon])
    model.external_drive = None
    return state, torch.stack(acts)


def attach_taste_heads(model, graph_npz, annotations):
    """The act of the taste world read from the MBONs (the fly's approach / avoid output): every MBON's valence
    v_j = (Q_j - P_j) / (Q_j + P_j) from the PPL1 (Q) and PAM (P) dopamine it receives (PPL1 compartments +: approach;
    PAM compartments -: avoid), fixed by the connectome; a learned gain and bias make the go logit."""
    import pandas as pd
    graph = np.load(graph_npz, allow_pickle=True)
    ann = pd.read_feather(annotations).set_index('bodyId').reindex(graph['neuron_body_ids'])
    typ, cls = ann['type'].astype(str).values, ann['class'].astype(str).values
    n = len(cls)
    is_dan = np.zeros(n, bool); is_dan[graph['dan_indices']] = True
    pam = np.char.startswith(typ.astype(str), 'PAM') & is_dan
    ppl1 = np.char.startswith(typ.astype(str), 'PPL1') & is_dan
    mbon = np.flatnonzero(cls == 'MBON')
    pre, post, w = graph['dan_edge_pre'], graph['dan_edge_post'], np.abs(graph['dan_edge_weight']).astype(np.float64)
    P = np.bincount(post[pam[pre]], weights=w[pam[pre]], minlength=n)[mbon]
    Q = np.bincount(post[ppl1[pre]], weights=w[ppl1[pre]], minlength=n)[mbon]
    v = np.where(P + Q > 0, (Q - P) / np.maximum(P + Q, 1e-12), 0.0)
    device = model.read_indices.device
    model.register_buffer('taste_mbon', torch.as_tensor(mbon, device=device), persistent=False)
    model.register_buffer('taste_valence', torch.as_tensor(v, dtype=torch.float32, device=device), persistent=False)
    model.register_buffer('taste_pam', torch.as_tensor(pam, device=device), persistent=False)
    model.register_buffer('taste_ppl1', torch.as_tensor(ppl1, device=device), persistent=False)
    model.taste_gain = nn.Parameter(torch.tensor(1.0, device=device))
    model.taste_bias = nn.Parameter(torch.tensor(0.0, device=device))
    return int(len(mbon))


TIED_WRITER_SCALE = 0.1             # birth scale of the tied rows: the input gain the regime-gestated newborn chose for
                                    # its written sensory neurons (on the 1/10 rail: the untied drive was ~10x too big)


class TiedProjection(nn.Module):
    """One projection row per sensory cell type, shared by every cell of that type (index: cell -> row)."""

    def __init__(self, weight, index):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.register_buffer('index', index, persistent=False)

    def forward(self, x):
        return F.linear(x, self.weight)[..., self.index]


TIED_WRITER_MODALITIES = ('proj_chemo',)


def attach_tied_writer(model, graph_npz, annotations, convert=True, scale=TIED_WRITER_SCALE,
                       modalities=TIED_WRITER_MODALITIES):
    """The writer drives sensory cells by type, as a stimulus drives all receptors of one kind together (in the fly
    the ~35 olfactory receptor neurons of a glomerulus share one receptor and fire together; projection neurons sum
    them).  Only where a type is a tuning identity: the chemosensory cells (olfactory and gustatory receptor types)
    get one row per annotated type; visual and mechanosensory types are places on a map (photoreceptors of one type
    in different ommatidia see different points), so those cells keep their own rows.  Untyped cells keep a row of
    their own.  Measured on the regime-gestated
    newborn: driving all receptors of one type together moves the matched projection neurons +25 Hz, the same
    number scattered across types ~0 Hz (FLY_CONSEQUENCE_WORLD_DESIGN_20261012.html).  convert=True builds the rows
    from the current per-cell rows (type mean x sqrt(cells), so a cell's drive keeps its size, times `scale`);
    convert=False makes empty rows of the right shape for loading a tied checkpoint."""
    import pandas as pd
    graph = np.load(graph_npz, allow_pickle=True)
    types = pd.read_feather(annotations).set_index('bodyId').reindex(graph['neuron_body_ids'])['type'].astype(str).values
    writer = model.topographic_writer
    counts = {}
    for name, cells in (('proj_vis', writer.idx_vis), ('proj_chemo', writer.idx_chemo), ('proj_mech', writer.idx_mech)):
        if name not in modalities:
            continue
        labels = types[cells.cpu().numpy()]
        keys = [t if t not in ('None', 'nan', '') else f'cell{i}' for i, t in enumerate(labels)]
        uniq = {k: j for j, k in enumerate(dict.fromkeys(keys))}
        index = torch.as_tensor([uniq[k] for k in keys], dtype=torch.long, device=cells.device)
        old = getattr(writer, name)
        if isinstance(old, TiedProjection):
            raise ValueError('the writer is already tied')
        if convert:
            w = old.weight.detach()
            total = torch.zeros(len(uniq), w.shape[1], device=w.device, dtype=w.dtype).index_add_(0, index, w)
            n = torch.bincount(index, minlength=len(uniq)).to(w.dtype).unsqueeze(1)
            rows = total / n * n.sqrt() * scale
        else:
            rows = torch.zeros(len(uniq), old.weight.shape[1], device=old.weight.device, dtype=old.weight.dtype)
        setattr(writer, name, TiedProjection(rows, index))
        counts[name] = (len(keys), len(uniq))
    model.tied_writer = True
    return counts


ONLINE_DA_CELL_CLASSES = ('Kenyon_Cell',)


def kenyon_dendritic_share(model, graph_npz, annotations, dendritic_feather):
    """Per excitatory edge: the share of a Kenyon -> Kenyon connection's synapses that lie in the calyx (Kenyon
    dendrites), from neuPrint male-cns v1.0 roiInfo (data/malecns_v1/kc_kc_dendritic_synapses.feather); 97% of these
    synapses are axo-axonal (lobes, peduncle) and do not drive the soma in the fly.  Returns (edge index, share) for
    every Kenyon -> Kenyon excitatory edge (share 0 where no synapse is in the calyx)."""
    import numpy as np
    import pandas as pd
    graph = np.load(graph_npz, allow_pickle=True)
    ids = graph['neuron_body_ids']
    table = pd.read_feather(annotations).set_index('bodyId').reindex(ids)
    kc = torch.as_tensor(table['class'].astype(str).values == 'Kenyon_Cell', device=model.edge_pre_e.device)
    pre, post = model.edge_pre_e.long(), model.edge_post_e.long()
    edges = torch.nonzero(kc[pre] & kc[post]).squeeze(1)
    dend = pd.read_feather(dendritic_feather)
    pos = {int(b): i for i, b in enumerate(ids)}
    n = model.n_neurons
    keys = torch.as_tensor([pos[int(a)] * n + pos[int(b)] for a, b in zip(dend.body_pre, dend.body_post)
                            if int(a) in pos and int(b) in pos], device=pre.device)
    vals = torch.as_tensor([d / t for a, b, d, t in zip(dend.body_pre, dend.body_post, dend.dendritic_synapses,
                                                        dend.total_synapses) if int(a) in pos and int(b) in pos],
                           dtype=torch.float32, device=pre.device)
    order = torch.argsort(keys)
    keys, vals = keys[order], vals[order]
    edge_keys = pre[edges] * n + post[edges]
    at = torch.searchsorted(keys, edge_keys).clamp(max=len(keys) - 1)
    share = torch.where(keys[at] == edge_keys, vals[at], torch.zeros_like(vals[at]))
    return edges, share


TIED_RECEPTOR_CLASSES = ('olfactory', 'gustatory', 'chemosensory', 'hygrosensory', 'thermosensory')


CHEMICAL_SENSE_CLASSES = ('olfactory', 'gustatory', 'chemosensory')
TASTE_SENSE_CLASSES = ('gustatory', 'chemosensory')
WRITER_SENSE = {'receptor': 'olfactory', 'receptor_all': 'all', 'receptor_text': 'text', 'receptor_lang': 'lang'}


def attach_receptor_writer(model, graph_npz, annotations, sense_class='olfactory', seed=0, **kw):
    """Replace the topographic writer by the receptor-transduction writer (ReceptorWriter).  sense_class: one sensory
    class (e.g. 'olfactory'), or 'all' -- every sensory cell of the old writer (data/malecns_v1/sensory_partitions.npz),
    the brain learning which receptors the text uses (user 2026-10-13: shared senses must stay learnable, as in sensory
    substitution).  Rows: one per receptor type where the type is a tuning identity (TIED_RECEPTOR_CLASSES: receptors of
    one kind respond alike), one per cell elsewhere (visual and mechanosensory types are places on a map).  Each row's
    projection is the learned route: a row at zero leaves its receptors at spontaneous firing.  Returns (cells, rows)."""
    import pandas as pd
    from .fly_reservoir import ReceptorWriter
    graph = np.load(graph_npz, allow_pickle=True)
    ann = pd.read_feather(annotations).set_index('bodyId').reindex(graph['neuron_body_ids'])
    cls, typ = ann['class'].astype(str).values, ann['type'].astype(str).values
    if sense_class in ('all', 'text', 'lang'):
        from pathlib import Path
        parts = np.load(Path(graph_npz).parent / 'sensory_partitions.npz')
        cells = np.concatenate([parts['visual_idx'], parts['chemo_idx'], parts['mechano_idx']]).astype(np.int64)
        if sense_class == 'text':
            # the chemical senses are kept for the world (odours, tastes: valence and the mushroom body's teacher);
            # the text goes through every other sense (user 2026-10-13: dedicated smell and taste are fine)
            cells = cells[~np.isin(cls[cells], CHEMICAL_SENSE_CLASSES)]
        elif sense_class == 'lang':
            # language through smell too (the mushroom body's main input is olfactory: without it the Kenyon cells
            # never hear the text), taste kept for the world as the valence / dopamine teacher (user 2026-10-14)
            cells = cells[~np.isin(cls[cells], TASTE_SENSE_CLASSES)]
    else:
        cells = np.nonzero(cls == sense_class)[0]
    tied = np.isin(cls[cells], TIED_RECEPTOR_CLASSES)
    keys = [f'{cls[c]}:{t}' if (k and t not in ('None', 'nan', '')) else f'cell{c}'
            for c, t, k in zip(cells, typ[cells], tied)]
    uniq = {k: j for j, k in enumerate(dict.fromkeys(keys))}
    old = model.topographic_writer
    d_model = model.embedding.weight.shape[1]
    writer = ReceptorWriter(d_model, cells, [uniq[k] for k in keys], len(uniq), E_E=model.E_E, seed=seed, **kw)
    model.topographic_writer = writer.to(model.embedding.weight.device)
    model.receptor_writer = True
    del old
    return int(len(cells)), int(len(uniq))


GRADED_TYPES = ('APL', 'CT1')
GRADED_PARAMETER_NAMES = ('log_graded_gain',)


# graded (non-spiking) early visual system: photoreceptors, lamina monopolar cells, and the medulla columnar cells whose
# recorded responses are graded (Behnia et al. 2014: Tm1, Tm2, Tm4, Tm9; Strother et al. 2017: Mi1, Tm3, Mi4, Mi9)
VISUAL_GRADED_TYPES = ('~R1-R6', '~R7*', '~R8*', '~L1', '~L2', '~L3', '~L4', '~L5',
                       '~Mi1', '~Mi4', '~Mi9', '~Tm1', '~Tm2', '~Tm3', '~Tm4', '~Tm9')


def attach_graded_cells(model, graph_npz, annotations, types=GRADED_TYPES, gain=1.0, tonic_gain=None):
    """The fly's non-spiking giant inhibitory interneurons (APL in the mushroom body, CT1 in the optic lobe) as graded
    cells (FlyReservoirLM.attach_graded): release proportional to depolarization instead of spikes.  Returns the
    number of cells per type.

    A type written '~T' (patterns allowed, e.g. '~R7*') is graded around a tonic operating point: release floor at E_I
    (photoreceptors and lamina monopolar cells, 2026-10-13: as spiking cells the photoreceptors' histamine onto L1-L3 only
    ever inhibited silent cells, so the retina sent nothing on); its gain is tonic_gain (default gain).  Cells already
    graded keep their learned gains when a second call extends the set."""
    import fnmatch
    import pandas as pd
    graph = np.load(graph_npz, allow_pickle=True)
    typ = pd.read_feather(annotations).set_index('bodyId').reindex(graph['neuron_body_ids'])['type'].astype(str).values
    plain = [t for t in types if not t.startswith('~')]
    tonic = [t[1:] for t in types if t.startswith('~')]
    match = lambda pats: (np.array([any(fnmatch.fnmatchcase(x, p) for p in pats) for x in typ]) if pats
                          else np.zeros(len(typ), bool))
    is_plain = match(plain)
    is_tonic = match(tonic) & ~is_plain
    index = np.nonzero(is_plain | is_tonic)[0]
    gains = np.where(is_tonic[index], float(tonic_gain if tonic_gain is not None else gain), float(gain))
    floors = np.where(is_tonic[index], float(model.E_I), 0.0)
    if getattr(model, 'graded_index', None) is not None:
        old = dict(zip(model.graded_index.tolist(), model.log_graded_gain.detach().exp().tolist()))
        gains = np.array([old.get(int(i), g) for i, g in zip(index, gains)])
    model.attach_graded(index, gain=gains, floor=floors)
    counts = {t: int((typ == t).sum()) for t in plain}
    counts.update({'~' + p: int(np.array([fnmatch.fnmatchcase(x, p) for x in typ]).sum()) for p in tonic})
    return counts


def attach_kenyon_dendritic_only(model, graph_npz, annotations, dendritic_feather, rescale=True):
    """Kenyon -> Kenyon excitation through the soma only for the calyx (dendritic) synapses: with rescale=True the
    learned weights are scaled once by the calyx share; connections with no calyx synapse are pinned at 0 (the learner
    keeps them there, so learning cannot regrow an axo-axonal drive)."""
    edges, share = kenyon_dendritic_share(model, graph_npz, annotations, dendritic_feather)
    if rescale:
        with torch.no_grad():
            model.edge_weight_e[edges] = model.edge_weight_e[edges] * share
    model.register_buffer('pinned_zero_edges_e', edges[share == 0], persistent=False)
    return int(edges.numel()), int((share == 0).sum())


def pin_kenyon_to_dopamine(model, graph_npz, annotations):
    """Kenyon -> dopamine-neuron synapses sit on the dopamine terminals in the mushroom-body lobes (axo-axonal); the point
    model made them somatic, so Kenyon activity (mostly text-driven) supplied ~80% of the dopamine cells' excitation and
    the teaching signal followed the text instead of the outcome (audit, FLY_CONNECTOME_AUDIT_20261012.md).  Pinned at 0
    like the axo-axonal Kenyon -> Kenyon synapses.  Returns the number of edges pinned."""
    import numpy as np
    import pandas as pd
    graph = np.load(graph_npz, allow_pickle=True)
    cls = pd.read_feather(annotations).set_index('bodyId').reindex(graph['neuron_body_ids'])['class'].astype(str).values
    device = model.edge_pre_e.device
    kc = torch.as_tensor(cls == 'Kenyon_Cell', device=device)
    dan = torch.zeros(model.n_neurons, dtype=torch.bool, device=device)
    dan[torch.as_tensor(graph['dan_indices'].astype(np.int64), device=device)] = True
    edges = torch.nonzero(kc[model.edge_pre_e.long()] & dan[model.edge_post_e.long()]).squeeze(1)
    with torch.no_grad():
        model.edge_weight_e[edges] = 0.0
    pinned = getattr(model, 'pinned_zero_edges_e', None)
    pinned = edges if pinned is None else torch.unique(torch.cat([pinned.to(device), edges]))
    model.register_buffer('pinned_zero_edges_e', pinned, persistent=False)
    return int(edges.numel())


def attach_kenyon_terminal_release(model, graph_npz, annotations, gain=0.1):
    """Kenyon -> dopamine-terminal synapses as what they are: axo-axonal input that raises dopamine RELEASE at the
    terminal in the lobe (Cervantes-Sandoval et al. 2017: functional cholinergic KC -> DAN synapses, local release, a
    positive feedback needed for learning), not somatic drive (pinned at 0 since connectome v2).  They are ~80% of the
    mushroom-body dopamine cells' excitatory synapses, and surprise reaches the Kenyon cells (2026-10-14 path probe:
    0.13) but not the dopamine cells (~0): the point model had cut the fly's own route from the Kenyon code to local
    dopamine (the alpha'3 novelty habituation, Hattori et al. 2017).
    Per tick: drive_d = sum_k w_kd s_k (the connectome's own Kenyon -> DAN weights, this tick's Kenyon spikes); the
    extra release g_d drive_d reaches each of the dopamine cell's targets through its own dopamine edges (local, no
    delay).  g per cell type (log_kc_release, a class parameter: learned per type), starting at `gain` (0.1: a tenth
    of a spike per unit drive, so the outcome lines are not swamped at birth).  The dopamine cells' spiking is not
    touched.  Returns (edges, dopamine cells reached)."""
    import numpy as np
    import pandas as pd
    graph = np.load(graph_npz, allow_pickle=True)
    cls = pd.read_feather(annotations).set_index('bodyId').reindex(graph['neuron_body_ids'])['class'].astype(str).values
    pre, post = graph['edge_pre_e'].astype(np.int64), graph['edge_post_e'].astype(np.int64)
    dan = np.zeros(len(cls), dtype=bool)
    dan[graph['dan_indices'].astype(np.int64)] = True
    m = (cls[pre] == 'Kenyon_Cell') & dan[post]
    device = model.edge_pre_e.device
    model.register_buffer('kcd_pre', torch.as_tensor(pre[m], device=device), persistent=False)
    model.register_buffer('kcd_post', torch.as_tensor(post[m], device=device), persistent=False)
    model.register_buffer('kcd_weight', torch.as_tensor(graph['edge_weight_e'][m], dtype=torch.float32, device=device),
                          persistent=False)
    if getattr(model, 'log_kc_release', None) is None:
        model.log_kc_release = torch.nn.Parameter(torch.full((model.n_superclasses,), float(np.log(gain)), device=device))
    return int(m.sum()), int(np.unique(post[m]).size)


def kenyon_terminal_release(model, spikes):
    """The dopamine arriving at every cell from the Kenyon-driven release at the dopamine terminals [1, N]."""
    drive = torch.zeros_like(spikes).index_add(1, model.kcd_post, spikes.index_select(1, model.kcd_pre) * model.kcd_weight)
    release = drive * torch.exp(model.log_kc_release)[model.superclass_id].unsqueeze(0)
    pre, post = model.dan_edge_pre.long(), model.dan_edge_post.long()
    return torch.zeros_like(spikes).index_add(1, post, release.index_select(1, pre) * model.dan_edge_weight) / model.dan_scale


def pin_edges(model, suffix, edges):
    """Pin edges of the excitatory ('e') or inhibitory ('i') list at 0 for good (merged into pinned_zero_edges_*, which
    the learner re-zeroes after every update)."""
    w = getattr(model, f'edge_weight_{suffix}')
    edges = edges.to(w.device)
    with torch.no_grad():
        w[edges] = 0.0
    name = f'pinned_zero_edges_{suffix}'
    old = getattr(model, name, None)
    model.register_buffer(name, edges if old is None else torch.unique(torch.cat([old.to(w.device), edges])), persistent=False)
    return int(edges.numel())


def pin_sensory_terminals(model, graph_npz):
    """Synapses onto sensory neurons sit on their axon terminals (their dendrites are peripheral): they modulate
    release, they do not make the sensory cell fire.  As somatic input they formed self-exciting loops (receptor ->
    receptor ~43% of olfactory receptor input).  Removed from the membrane (release modulation can be added later)."""
    e, i = sensory_axonal_edges(model, graph_npz)
    return pin_edges(model, 'e', e) + pin_edges(model, 'i', i)


MODULATORY_TYPE_PATTERN = r'^(OA-|5-HT|PAM\d|PPL\d)'


def pin_modulatory_fast_edges(model, graph_npz, modulatory_feather, transmitters=('octopamine', 'serotonin'),
                              type_pattern=None, annotations=None):
    """Octopamine and serotonin act through metabotropic receptors (volume transmission), not fast synapses; as fast
    excitation they drove e.g. the Kenyon cells directly (OA-VPM3/VPM4/VUMa2: ~half of the non-PN Kenyon drive).  Their
    fast output edges are removed (as dopamine's were); their modulation is not modelled yet."""
    import numpy as np
    import pandas as pd
    ids = np.load(graph_npz, allow_pickle=True)['neuron_body_ids']
    table = pd.read_feather(modulatory_feather)
    bodies = set(int(b) for b in table[table.consensusNt.isin(transmitters)].bodyId)
    if type_pattern is not None:
        # by cell type as well (audit 2026-10-13): OA-AL2i1 / OA-ASM3 and PPL203 have an 'unclear' prediction and
        # 5-HTPLP01 a glutamate one, so the transmitter alone left their fast edges in place
        ann = pd.read_feather(annotations)[['bodyId', 'type']]
        bodies |= set(int(b) for b in ann[ann['type'].astype(str).str.contains(type_pattern, regex=True)].bodyId)
    mask = torch.as_tensor(np.array([int(b) in bodies for b in ids]), device=model.edge_pre_e.device)
    n = 0
    for sfx in ('e', 'i'):
        edges = torch.nonzero(mask[getattr(model, f'edge_pre_{sfx}').long()]).squeeze(1)
        n += pin_edges(model, sfx, edges)
    return n


def apply_connectome_fixes(model, graph_npz, annotations, *, kenyon_dendritic=True, kenyon_renorm=True, reset_al=False):
    """The high-severity point-model fixes of the audit (FLY_CONNECTOME_AUDIT_20261012.md) in one call: Kenyon
    axo-axonal input cut and the Kenyon inputs renormalized over dendritic partners (only when asked: they rescale the
    learned weights, once per lineage), Kenyon -> dopamine terminal synapses, synapses onto sensory terminals and the
    fast edges of octopamine / serotonin neurons pinned at 0 (idempotent: call on every load), optionally the antennal
    lobe's inputs back to birth (once).  Returns the counts."""
    from pathlib import Path
    graph_npz = Path(graph_npz); folder = graph_npz.parent
    dend = folder / 'kc_kc_dendritic_synapses.feather'
    done = {}
    if kenyon_dendritic:
        done['kenyon_axo_axonal'] = attach_kenyon_dendritic_only(model, graph_npz, annotations, dend, rescale=kenyon_renorm)[1]
    else:
        done['kenyon_axo_axonal'] = attach_kenyon_dendritic_only(model, graph_npz, annotations, dend, rescale=False)[1]
    if kenyon_renorm:
        done['kenyon_renorm_factor'] = renormalize_kenyon_inputs(model, graph_npz, annotations, dend)
    done['kenyon_to_dopamine'] = pin_kenyon_to_dopamine(model, graph_npz, annotations)
    done['sensory_terminals'] = pin_sensory_terminals(model, graph_npz)
    done['octopamine_serotonin_fast'] = pin_modulatory_fast_edges(model, graph_npz, folder / 'modulatory_neurons.feather')
    if reset_al:
        done['antennal_lobe_reset'] = reset_inputs_to_birth(model, graph_npz, annotations)
    return done


ANTENNAL_LOBE_CLASSES = ('olfactory', 'ALLN', 'ALPN', 'ALIN', 'ALON')


def reset_inputs_to_birth(model, graph_npz, annotations, classes=ANTENNAL_LOBE_CLASSES):
    """Every synapse onto the cells of the given classes back to its birth weight (the rest of the brain keeps what it
    learned).  For the antennal lobe of a lineage trained with absolute Adam steps: its excitation grew 5-8x (inhibition
    1.3-2x) into a self-sustaining excitatory loop that drowns odours.  Returns the number of edges reset."""
    import numpy as np
    import pandas as pd
    graph = np.load(graph_npz, allow_pickle=True)
    cls = pd.read_feather(annotations).set_index('bodyId').reindex(graph['neuron_body_ids'])['class'].astype(str).values
    target = torch.as_tensor(np.isin(cls, list(classes)), device=model.edge_post_e.device)
    n = 0
    with torch.no_grad():
        for sfx in ('e', 'i'):
            w = getattr(model, f'edge_weight_{sfx}')
            birth = torch.as_tensor(np.abs(graph[f'edge_weight_{sfx}']), dtype=w.dtype, device=w.device)
            sel = target[getattr(model, f'edge_post_{sfx}').long()]
            w[sel] = birth[sel]
            n += int(sel.sum())
    return n


SENSORY_SUPERCLASSES = ('cb_sensory', 'ol_sensory', 'vnc_sensory', 'sensory_ascending', 'sensory_descending')


def sensory_axonal_edges(model, graph_npz):
    """Edges whose target is a sensory neuron.  A sensory neuron's dendrites are peripheral, so every connectome synapse
    onto it sits on its axon terminal (presynaptic modulation of release), not where its spikes are generated.
    Returns (excitatory edge indices, inhibitory edge indices)."""
    import numpy as np
    graph = np.load(graph_npz, allow_pickle=True)
    names = list(graph['superclass_names'])
    sensory = np.isin(np.array(names)[graph['superclass_id']], SENSORY_SUPERCLASSES)
    mask = torch.as_tensor(sensory, device=model.edge_post_e.device)
    return (torch.nonzero(mask[model.edge_post_e.long()]).squeeze(1),
            torch.nonzero(mask[model.edge_post_i.long()]).squeeze(1))


def renormalize_kenyon_inputs(model, graph_npz, annotations, dendritic_feather):
    """Undo a birth-normalization artifact, once per lineage: every edge was divided by sqrt(K_in) of its target, and a
    Kenyon cell's K_in counted its axo-axonal Kenyon partners (median 162 partners; 13 without any Kenyon partner),
    which do not drive the soma (attach_kenyon_dendritic_only).  With the dendritic partners only (non-Kenyon partners
    plus Kenyon partners with calyx synapses) every remaining input onto a Kenyon cell is sqrt(K_all / K_dendritic)
    larger (the original count also included the dopamine partners, median 35, whose edges now live in the modulatory
    list).  Scales the current (learned) weights of all non-pinned edges onto Kenyon cells; returns the median factor."""
    import numpy as np
    import pandas as pd
    graph = np.load(graph_npz, allow_pickle=True)
    n = model.n_neurons
    cls = pd.read_feather(annotations).set_index('bodyId').reindex(graph['neuron_body_ids'])['class'].astype(str).values
    kc = cls == 'Kenyon_Cell'
    # the birth K_in counted every partner, the dopamine cells included (their edges were moved to the modulatory list
    # afterwards): add them back to the original denominator
    k_fast = np.bincount(graph['edge_post'], minlength=n).astype(np.float64)
    k_all = k_fast + np.bincount(np.unique(graph['dan_edge_pre'].astype(np.int64) * n + graph['dan_edge_post']) % n,
                                 minlength=n)
    edges, share = kenyon_dendritic_share(model, graph_npz, annotations, dendritic_feather)
    dead = torch.zeros(model.edge_pre_e.numel(), dtype=torch.bool, device=model.edge_pre_e.device)
    dead[edges[share == 0]] = True
    # Kenyon partners without calyx synapses, per target (counted on the excitatory list, where Kenyon edges live)
    post_e = model.edge_post_e.long()
    axo_t = torch.bincount(post_e[dead], minlength=n).cpu().numpy().astype(np.float64)
    k_den = np.maximum(k_fast - axo_t, 1.0)          # dendritic: fast partners without the axo-axonal Kenyon ones
    factor = torch.as_tensor(np.where(kc, np.sqrt(k_all / k_den), 1.0), dtype=torch.float32, device=post_e.device)
    with torch.no_grad():
        for sfx in ('e', 'i'):
            w = getattr(model, f'edge_weight_{sfx}')
            post = getattr(model, f'edge_post_{sfx}').long()
            scale = factor[post]
            if sfx == 'e':
                scale = torch.where(dead, torch.ones_like(scale), scale)
            w.mul_(scale)
    return float(np.median(factor.cpu().numpy()[kc]))


def renormalize_dopamine_inputs(model, graph_npz, annotations):
    """The same birth-normalization artifact as renormalize_kenyon_inputs, for the dopamine cells, once per lineage:
    every edge was divided by sqrt(K_in) of its target, and a dopamine cell's K_in counted its Kenyon partners (median
    319 of 479), whose synapses sit on its terminals in the lobes and are pinned (pin_kenyon_to_dopamine).  Over the
    remaining partners every remaining input onto a dopamine cell is sqrt(K_all / K_kept) larger (median 2.1).
    Scales the current weights of all non-pinned edges onto dopamine cells; returns the median factor."""
    import pandas as pd
    graph = np.load(graph_npz, allow_pickle=True)
    n = model.n_neurons
    cls = pd.read_feather(annotations).set_index('bodyId').reindex(graph['neuron_body_ids'])['class'].astype(str).values
    kc = cls == 'Kenyon_Cell'
    dan = np.zeros(n, bool)
    dan[graph['dan_indices']] = True
    k_fast = np.bincount(graph['edge_post'], minlength=n).astype(np.float64)
    k_all = k_fast + np.bincount(np.unique(graph['dan_edge_pre'].astype(np.int64) * n + graph['dan_edge_post']) % n,
                                 minlength=n)
    pre, post = graph['edge_pre'], graph['edge_post']
    k_kc = np.bincount(post[kc[pre]], minlength=n).astype(np.float64)
    factor = np.where(dan, np.sqrt(k_all / np.maximum(k_fast - k_kc, 1.0)), 1.0)
    factor_t = torch.as_tensor(factor, dtype=torch.float32, device=model.edge_pre_e.device)
    pinned = getattr(model, 'pinned_zero_edges_e', None)
    with torch.no_grad():
        for sfx in ('e', 'i'):
            w = getattr(model, f'edge_weight_{sfx}')
            scale = factor_t[getattr(model, f'edge_post_{sfx}').long()]
            if sfx == 'e' and pinned is not None:
                scale[pinned.long()] = 1.0
            w.mul_(scale)
    return float(np.median(factor[dan]))


def attach_online_da_plasticity(model, graph_npz=None, annotations=None, eta=-0.02, tau_trace=50.0, forget=0.02,
                                cell_classes=ONLINE_DA_CELL_CLASSES):
    """O(1) online learning with dopamine as the third factor (design: FLY_ONLINE_DA_LEARNING_DESIGN.md).  The
    terminals of the Kenyon cells that receive the connectome's dopamine projections (the mushroom body's own plastic
    site; its loop Kenyon -> MBON -> dopamine cells -> Kenyon and MBON -> motor is native) keep a presynaptic activity
    trace e (time constant learnable, start 50 ticks) and a memory phi.  Dopamine opens a plasticity window at an
    active terminal, w = (receptor occupancy of the tonic dopamine there / 0.5) * e; phi <- phi + w * (eta * phasic
    dopamine - kappa * phi), the phasic part being the occupancy of the fast minus that of the slow concentration:
    written and overwritten only while the terminal is active and dopamine arrives, otherwise kept (eta < 0: the
    fly's depression after activity-then-dopamine).  eta = -kappa (start 0.02): phi is then the running average of minus
    the occupancy change met inside the windows, so it stays within (-1, 1) with no imposed bound (each update is a
    convex mix since window * kappa <= 0.04); both learnable.  phi shifts the
    terminal's baseline release probability, sigmoid(logit(u0) + phi); the vesicle pool is depleted as usual.  Needs
    attach_dopamine_modulation.  Within a window the updates are differentiable (BPTT learns eta, the trace time
    constant and what drives the dopamine cells); across windows phi persists.  cell_classes=None: every dopamine
    target (all 39,850; includes the core, whose loop through the dopamine cells makes BPTT gradients diverge)."""
    import numpy as np
    import pandas as pd
    if getattr(model, 'dopamine_sensitivity', None) is None:
        raise ValueError('online dopamine plasticity needs attach_dopamine_modulation first')
    device = model.read_indices.device
    receives = torch.zeros(model.n_neurons, dtype=torch.bool, device=device)
    receives[torch.unique(model.dan_edge_post.long())] = True
    if cell_classes is not None:
        graph = np.load(graph_npz, allow_pickle=True)
        table = pd.read_feather(annotations).set_index('bodyId').reindex(graph['neuron_body_ids'])
        chosen = np.isin(table['class'].astype(str).values, list(cell_classes))
        receives &= torch.as_tensor(chosen, device=device)
    target = receives.float().unsqueeze(0)
    model.register_buffer('online_da_target', target, persistent=False)
    model.online_da_eta = nn.Parameter(torch.tensor(float(eta), device=device))
    lam = math.exp(-1.0 / tau_trace)
    model.online_da_logit_trace = nn.Parameter(torch.tensor(math.log(lam / (1.0 - lam)), device=device))
    model.online_da_logit_forget = nn.Parameter(torch.tensor(math.log(forget / (1.0 - forget)), device=device))
    return model


DOPAMINE_SITES = ('post', 'pre')


def dopamine_edge_scale(model, edge_suffix, local, site='post'):
    """Per-synapse scaling of the update by the dopamine at one end of each synapse: 'post' (the receiving neuron) or
    'pre' (the presynaptic terminal: in the fly dopamine acts on the Kenyon-cell axons and changes their output
    synapses onto the mushroom-body output neurons)."""
    if site not in DOPAMINE_SITES:
        raise ValueError(f'dopamine site must be one of {DOPAMINE_SITES}')
    return local[getattr(model, f'edge_{site}{edge_suffix}').long()]


def compose_edge_updates(model, edge_names, edges, before, relative=None, local=None, site='post'):
    """Scale every synapse's optimizer update (w - before) by one composed factor: the relative step (rate x its birth
    weight / lr, set_relative_edge_steps) times the dopamine at its chosen end (dopamine_edge_scale).  In place."""
    with torch.no_grad():
        for i, (name, w, old) in enumerate(zip(edge_names, edges, before)):
            scale = relative[i] if relative is not None else 1.0
            if local is not None:
                scale = scale * dopamine_edge_scale(model, name[len('edge_weight'):], local, site)
            w.copy_(old + (w - old) * scale)


def attach_dopamine_modulation(model, tau_fast=50.0, tau_slow=500.0, site='post', adapted=False, sensitivity_init=0.0,
                               affinity=None):
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
    model.dopamine_affinity = float(affinity if affinity is not None else DOPAMINE_AFFINITY)
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


def word_packets(model, state, token):
    """A held word's writer projections, once per word: (packet of the word, packet of the expectation or None).
    Every tick of the word sees the same embedding and the same expectation (updated once per word), so projecting
    them every tick only repeated two matrix-vector products and their weight gradients (2026-10-14 profile)."""
    writer = model.topographic_writer
    if writer is None or not getattr(writer, 'accepts_packets', False) or os.environ.get('FLY_WORD_PACKET', '1') == '0':
        return None
    expected = getattr(state, 'expected_input', torch.empty(0))
    predictive = getattr(model, 'writer_prediction_logit', None) is not None and expected.numel()
    return writer.packet(model.embedding(token)), (writer.packet(expected) if predictive else None)


def write_input(model, state, token, emb=None, packets=None):
    """Sensory drive of one input word through the writer; predictive when a top-down expectation is available.
    emb: the sample the senses deliver this tick (default: the word's clean embedding).  packets: word_packets(...)
    (a clean held word only)."""
    writer = model.topographic_writer
    if writer is None:
        return None, state.baseline
    if packets is not None and emb is None:
        beta = (torch.sigmoid(model.writer_prediction_logit) if packets[1] is not None else None)
        return writer.forward_with_state(None, state.h, state.baseline, beta=beta, packets=packets)
    sample = model.embedding(token) if emb is None else emb
    expected = getattr(state, 'expected_input', torch.empty(0))
    if getattr(model, 'writer_prediction_logit', None) is not None and expected.numel():
        return writer.forward_with_state(sample, state.h, state.baseline, expected=expected,
                                         beta=torch.sigmoid(model.writer_prediction_logit))
    return writer.forward_with_state(sample, state.h, state.baseline)


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
    motor = read_in.index_select(1, model.read_indices) if model.read_surface != 'all' else read_in
    return model.output_read(motor_feature(model, state, motor))


def _advance_fly_token_stream(model, state, token, schedule_box=None, *, ticks, hold_input=False, read_after=None,
                              writer_baseline_clock='physical', base_rates=None, thresholds=None,
                              conductance_gains=None, alif_params=None, stp_params=None, h_mean_decay=0.99,
                              read_every=False, world=None, world_tape=None, act_g=None, fresh=True,
                              ai_g=None, ai_tape=None, eye_forced=None, eye_binary=False, eye_tape=None,
                              eye_sample=False, sense_noise=None, dopamine_s_class=None):
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
    read_every=True: also return the motor read after every tick ([ticks * B, d]): each is the expectation that would
    meet the next word had it arrived then.
    world (fly_odour_world.OdourWorld): the odour world runs alongside, every tick: its current is injected, the act
    head integrates (act_g carried in [1, 2]), the realized act steps the world.  world_tape (a fresh list per token)
    records each tick's (drive spec, record, act) on the first pass and replays it on a checkpoint recompute.  Then
    also returned: the act hazards [ticks, 2] and the new act_g.
    fresh=False: a continuation segment of the same token (no onset write; quiet or held ticks only) -- long intervals
    are split into segments so a checkpoint recompute never holds more than one segment's activations.
    ai_g [1, 2] (fly_active_inference): the eyes / output act head runs: before a tick that writes the word, the eye
    gain e (from the integrator as it stands; eye_forced fixes it) scales the written drive -- closed eyes miss the
    word, the world does not wait; after every tick both channels integrate and the output channel may commit (taped in
    ai_tape for the recompute).  Then also returned: the eye gain in force at every tick [ticks], whether the tick
    wrote the word [ticks], the output hazards [ticks], the new ai_g.
    sense_noise [ticks, d]: with held input, the senses deliver a noisy sample of the word every tick (embedding +
    sense_noise[tick]); evidence accumulates over the word's ticks (fly_active_inference, sustained noisy input).
    """
    if not isinstance(ticks, int) or ticks < 1:
        raise ValueError('ticks must be a positive integer')
    if writer_baseline_clock not in ('input', 'physical'):
        raise ValueError('writer_baseline_clock must be input or physical')
    options = dict(base_rates=base_rates, thresholds=thresholds, conductance_gains=conductance_gains,
                   alif_params=alif_params, stp_params=stp_params)
    if dopamine_s_class is not None:
        options['dopamine_s_class'] = dopamine_s_class
    writer = model.topographic_writer
    embedding = model.embedding(token) if writer is not None else None
    if read_after is not None and not 1 <= read_after <= ticks:
        raise ValueError('read_after must lie inside the token window')
    current = state
    quiet_source = None
    latent = None
    packets = None
    reads, hazards = [], []
    eyes, onsets, out_hazards = [], [], []
    if world is not None:
        from .fly_odour_world import act_step, dense
        replay = len(world_tape) > 0
    if ai_g is not None:
        from .fly_active_inference import ai_eye, ai_step
        ai_replay = len(ai_tape) > 0
    for tick in range(ticks):
        if read_after is not None and tick == read_after:
            latent = read_fly_motor(model, current)
        if world is not None:
            spec, record = world_tape[tick][:2] if replay else world.drive()
            model.external_drive = dense(spec, model.n_neurons, current.h.device)
        writes = (tick == 0 and fresh) or (hold_input and writer is not None)
        if ai_g is not None:
            if eye_binary:
                # M3: the eyes as a decision, exactly 0 / 1 forward (taped for the recompute), straight-through
                from .fly_active_inference import eye_act
                eye, look = eye_act(model, ai_g, eye_forced, eye_tape[tick] if ai_replay else None, sample=eye_sample)
                if not ai_replay:
                    eye_tape.append(look)
            else:
                eye = ai_eye(model, ai_g, eye_forced)
            eyes.append(eye)
            onsets.append(writes)
        if writes:
            sample = None if sense_noise is None else model.embedding(token) + sense_noise[tick:tick + 1]
            if sample is None and packets is None and hold_input:
                packets = word_packets(model, current, token)
            drive, baseline = write_input(model, current, token, sample, packets=packets)
            if ai_g is not None:
                # precision of the text input: the surface receives e * the word, so the drive and the sensory
                # adaptation both see e * the packet; closed eyes (e = 0) are exactly a quiet tick, the word missed
                quiet_baseline = (writer.lambda_adapt * current.baseline
                                  if writer_baseline_clock == 'physical' and writer is not None else current.baseline)
                drive = drive * eye
                baseline = eye * baseline + (1.0 - eye) * quiet_baseline
            current = step_fly_physical_tick(model, current, token, drive, baseline, options,
                                             h_mean_decay=h_mean_decay, base_rates=base_rates)
        else:
            if quiet_source is None:
                quiet_source = torch.zeros_like(current.h)
            baseline = (writer.lambda_adapt * current.baseline
                        if writer_baseline_clock == 'physical' and writer is not None else current.baseline)
            current = step_fly_physical_tick(model, current, None, quiet_source, baseline, options,
                                             h_mean_decay=h_mean_decay, base_rates=base_rates)
        if world is not None:
            model.external_drive = None
            hazard, act_g, act = act_step(model, current.h, act_g, world_tape[tick][2] if replay else None,
                                          noise=record.get('noise'))
            hazards.append(hazard)
            if not replay:
                world.observe(act)
                world_tape.append((spec, record, act))
        if ai_g is not None:
            hz, ai_g, committed = ai_step(model, current, ai_g, ai_tape[tick] if ai_replay else None)
            out_hazards.append(hz)
            if not ai_replay:
                ai_tape.append(committed)
        if read_every:
            reads.append(read_fly_motor(model, current))
    extra = (torch.stack(hazards), act_g) if world is not None else ()
    if ai_g is not None:
        extra = extra + (torch.cat(eyes), torch.tensor(onsets, device=current.h.device), torch.cat(out_hazards), ai_g)
    if read_every:
        latent = reads[-1]
        return (expect_next(model, current, latent), latent, torch.cat(reads, 0), *extra)
    latent = latent if latent is not None else read_fly_motor(model, current)
    return (expect_next(model, current, latent), latent, *extra)


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
                 go_reference_energy=0.07, go_max_ticks=32, lr_read_bank=None, lr_physiology=None):
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
        if getattr(model, 'act_w', None) is not None:
            # the act head, when attached, always learns (fly_odour_world.attach_act_head)
            self.adam_names += [n for n in ('act_w', 'act_b', 'act_logit_leak') if n not in self.adam_names]
        if getattr(model, 'sense_head', None) is not None:
            self.adam_names += [n for n in ('sense_head.weight', 'sense_head.bias') if n not in self.adam_names]
        if getattr(model, 'ai_w', None) is not None:
            self.adam_names += [n for n in ('ai_w', 'ai_b', 'ai_logit_leak')
                                if n not in self.adam_names and getattr(model, n, None) is not None]
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
                      + PREDICTIVE_PARAMETER_NAMES + DOPAMINE_PARAMETER_NAMES + ONLINE_DA_PARAMETER_NAMES + go_scalars
                      if name in named
                      and name in self.adam_names + list(READ_BANK_PARAMETER_NAMES + INTRINSIC_PARAMETER_NAMES
                                                         + PRECISION_PARAMETER_NAMES + PREDICTIVE_PARAMETER_NAMES
                                                         + DOPAMINE_PARAMETER_NAMES + ONLINE_DA_PARAMETER_NAMES)]
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
        if lr_physiology is not None:
            # Class physiology (log time constants, log thresholds, logit release probabilities, dopamine and memory
            # clocks) lives in log / logit space, where an Adam step of lr is a relative change of ~lr: at the shared
            # 2e-4 a time constant moves ~1% per 4k-token chunk (D2: STP moved <= 0.016 in two chunks), 100x slower
            # than the synapses' relative steps (2% of the birth weight).  Their own rate makes them evolvable.
            physio = set(CLASS_PARAMETER_NAMES) | {'dopamine_logit_fast', 'dopamine_logit_slow', 'comp_logit_trace',
                                                   'comp_logit_forget'}
            moved = [named[n] for n in self.adam_names if n in physio]
            self.physio_names = [n for n in self.adam_names if n in physio]
            self.physio_lr = float(lr_physiology)
            moved_ids = {id(p) for p in moved}
            for group in adam_groups:
                group['params'] = [p for p in group['params'] if id(p) not in moved_ids]
            adam_groups.append({'params': moved, 'weight_decay': 0.0, 'lr': float(lr_physiology)})
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
        if model.topographic_writer is None:
            self.projections = [model.input_proj.weight]
        elif hasattr(model.topographic_writer, 'proj'):              # receptor writer: one row per receptor type
            self.projections = [model.topographic_writer.proj.weight]
        else:
            self.projections = [getattr(model.topographic_writer, name).weight
                                for name in ('proj_vis', 'proj_chemo', 'proj_mech')]
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
                       | set(PREDICTIVE_PARAMETER_NAMES) | set(DOPAMINE_PARAMETER_NAMES) | set(ONLINE_DA_PARAMETER_NAMES)
                       | set(COMPARTMENT_PARAMETER_NAMES) | set(TASTE_PARAMETER_NAMES)
                       | {'act_w', 'act_b', 'act_logit_leak', 'ai_w', 'ai_b', 'ai_logit_leak'})
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
            if name not in self.adam_names and (name.startswith('graph_observer.') or name.startswith('latent_predictor.')
                                                or name not in dict(self.model.named_parameters())):
                old_by_name.pop(name, None)
                old_options.pop(name, None)
        if set(old_by_name) != set(self.adam_names) - newly_trainable:
            raise ValueError(f'Optimizer parameter coverage changed: extra={set(old_by_name) - (set(self.adam_names) - newly_trainable)}, missing={(set(self.adam_names) - newly_trainable) - set(old_by_name)}')
        destination = self.optimizer.state_dict()
        migrated = {}
        named = dict(self.model.named_parameters())
        sources = getattr(self.model, 'class_split_sources', None)
        for group in destination['param_groups']:
            for index, name in zip(group['params'], group['parameter_names']):
                entry = old_by_name.get(name)
                if entry is not None and sources and name in CLASS_PARAMETER_NAMES:
                    # a split class: the new classes start with the moments of the class they left
                    width = named[name].shape[0]
                    entry = {k: (torch.cat([v, v[sources]]) if isinstance(v, torch.Tensor) and v.ndim == 1
                                 and v.shape[0] + len(sources) == width else v) for k, v in entry.items()}
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
        self._credit = None
        if getattr(m, 'carried_credit', ()) and m.training and torch.is_grad_enabled():
            # credit beyond the window: the slow states open the window as leaves (fly_carried_credit)
            from .fly_carried_credit import open_window
            state, leaves, start = open_window(state, m.carried_credit)
            self._credit = (leaves, start)
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
            if getattr(m, 'dopamine_sensitivity', None) is not None and os.environ.get('FLY_WINDOW_S_CLASS', '1') != '0':
                stream['dopamine_s_class'] = dopamine_class_sensitivity(m, state.h)
            pace = float(getattr(self, 'stream_pace_mean', 0.0) or 0.0)
            pace_min = float(getattr(self, 'stream_pace_min', 0.0) or 0.0)
            if pace > 0 and getattr(self, 'pace_rng', None) is None:
                self.pace_rng = np.random.default_rng(int(getattr(self, 'pace_seed', 0)))
            if pace > 0 and 0 < pace_min < pace:
                # the world's tempo changes from stretch to stretch: this window's mean interval, log-uniform
                pace = float(math.exp(self.pace_rng.uniform(math.log(pace_min), math.log(pace))))
            total_ticks = 0
            reads, intervals = [], []
            word_da = [] if getattr(self, 'record_word_da', False) else None
            if pace > 0:
                stream['read_every'] = True
            world = getattr(self, 'odour_world', None) if torch.is_grad_enabled() else None
            hazards, records = [], []
            ai_on = getattr(m, 'ai_w', None) is not None
            noise_targets = self.__dict__.pop('noise_mask', None)      # set by observe() for this window only
            sense_sigmas = []
            ai_eyes, ai_onsets, ai_out, ai_tokens = [], [], [], []
            if ai_on:
                if getattr(m, 'ai_mode', None) == 'motor':
                    from .fly_active_inference import motor_rest
                    rest = motor_rest(m, state.h)
                    if getattr(self, 'ai_g', None) is None or self.ai_g.shape != rest.shape:
                        self.ai_g = rest                                            # muscles at rest
                if getattr(self, 'ai_g', None) is None:
                    p_open = float(getattr(self, 'eye_init', 0.95))
                    self.ai_g = state.h.new_tensor([[math.log(p_open / (1 - p_open)), 0.0]])     # eyes open at the start
                ai_g = self.ai_g
            if world is not None:
                if getattr(self, 'act_g', None) is None:
                    self.act_g = state.h.new_zeros(1, 2)
                act_g = self.act_g
            for token in input_ids.unbind(1):
                # the world's pace: a fixed clock, or (pace > 0) a memoryless interval k >= 1 with mean `pace` drawn per
                # word, so that elapsed time tells nothing about when the next word comes; the read is the expectation
                # standing just before the next word arrives
                k = int(self.pace_rng.geometric(1.0 / pace)) if pace > 0 else self.stream_ticks
                total_ticks += k
                word_noise = None
                if float(getattr(self, 'sense_noise_max', 0.0) or 0.0) > 0 and self.hold_input:
                    # each word is heard over its whole interval as noisy samples; its clarity is drawn per word
                    # (sigma = max * u, u ~ U(0, 1)), the noise is relative to the word's embedding norm per dimension
                    if getattr(self, 'sense_rng', None) is None:
                        self.sense_rng = np.random.default_rng(int(getattr(self, 'pace_seed', 0)) + 17)
                    sigma = float(self.sense_noise_max) * float(self.sense_rng.random())
                    with torch.no_grad():
                        emb = m.embedding(token)
                        scale = sigma * float(emb.norm()) / math.sqrt(emb.shape[-1])
                        word_noise = torch.as_tensor(self.sense_rng.standard_normal((k, emb.shape[-1])),
                                                     dtype=emb.dtype, device=emb.device) * scale
                    sense_sigmas.append(sigma)
                seg = int(getattr(self, 'stream_segment_ticks', 16) or k)
                done = 0
                while done < k:
                    # a long interval in segments of <= seg ticks, each its own checkpoint (same computation)
                    n = min(seg, k - done)
                    stream['ticks'] = n
                    stream['fresh'] = done == 0
                    stream['sense_noise'] = None if word_noise is None else word_noise[done:done + n]
                    box = [n]
                    if world is not None:
                        tape = []
                        stream.update(world=world, world_tape=tape, act_g=act_g)
                    if ai_on:
                        stream.update(ai_g=ai_g, ai_tape=[], eye_forced=getattr(self, 'eye_forced', None))
                        if getattr(self, 'eye_binary', False):
                            stream.update(eye_binary=True, eye_tape=[],
                                          eye_sample=bool(m.training and torch.is_grad_enabled()))
                    if self.use_checkpointing and m.training and torch.is_grad_enabled():
                        out = checkpoint(advance_fly_token_stream, m, state, token, box, use_reentrant=False, **stream)
                    else:
                        out = advance_fly_token_stream(m, state, token, box, **stream)
                    state, token_lat = out[0], out[1]
                    if pace > 0:
                        reads.append(out[2])
                    if ai_on:
                        ai_eyes.append(out[-4]); ai_onsets.append(out[-3]); ai_out.append(out[-2]); ai_g = out[-1]
                        ai_tokens += [int(token[0])] * n
                    wi = -4 if ai_on else 0
                    if world is not None:
                        hazards.append(out[wi - 2]); act_g = out[wi - 1]
                        records += [rec for _, rec, _ in tape]
                    done += n
                if pace > 0:
                    intervals.append(k)
                latents.append(token_lat)
                if word_da is not None:
                    # the living causal gate (2): each word's phasic dopamine (fast - slow) / slow at the end of its
                    # ticks and the dopamine cells' firing over the spike ring -- kept on the device, one copy per window
                    with torch.no_grad():
                        fast, slow = state.dopamine[0].detach(), state.dopamine[1].detach()
                        on = slow > 0
                        phasic = torch.where(on, (fast - slow) / slow.clamp_min(1e-30), torch.zeros_like(fast)).sum()                             / on.sum().clamp_min(1)
                        dan = getattr(m, 'dan_indices', None)
                        rate = (torch.stack([r[0] for r in state.ring]).index_select(1, dan).gt(0).float().mean()
                                if dan is not None else phasic.new_zeros(()))
                        word_da.append(torch.stack([phasic.float(), rate.float()]))
            if word_da is not None:
                self.last_word_da = torch.stack(word_da).cpu().tolist() if word_da else None
            features = torch.cat(latents, dim=0)
            logits = m.decoder(m.read_norm(features))
            scores = F.cross_entropy(logits, targets.flatten(), reduction='none')
            self.last_train_loss = scores
            self.last_adaptive_metrics = {}
            if pace > 0:
                # every tick before the next word arrives carries an expectation that would have met it had it come
                # then: each scores with the arrival hazard p = 1/pace.  P(K >= t) = (1-p)^(t-1), so the expectation of
                # sum_{t<=K} p CE_t is the expected arrival CE -- no imposed read time, no imposed waiting cost, and
                # the world never pauses (no counterfactual branch past the real arrival)
                tick_targets = torch.repeat_interleave(targets.flatten(), torch.as_tensor(intervals, device=targets.device))
                tick_ce = F.cross_entropy(m.decoder(m.read_norm(torch.cat(reads, 0))), tick_targets, reduction='none')
                owner = torch.repeat_interleave(torch.arange(len(intervals), device=targets.device),
                                                torch.as_tensor(intervals, device=targets.device))
                self.last_train_loss = torch.zeros_like(scores).index_add(0, owner, tick_ce) / pace
                with torch.no_grad():
                    onset = torch.cat([torch.arange(1, k + 1, device=targets.device) for k in intervals])
                    early = tick_ce[onset <= 2].mean() if bool((onset <= 2).any()) else tick_ce.new_tensor(float('nan'))
                self.last_adaptive_metrics = {'stream_pace_mean': pace, 'adaptive_total_ticks': total_ticks,
                                              'survival_ce': float(self.last_train_loss.detach().mean()),
                                              'ce_ticks_1_2': float(early)}
            self.last_act_loss = None
            if ai_on:
                # perception VFE complexity (SIGReg on the per-tick read latents) and the epistemic value of looking:
                # each word onset is worth e * novelty(word), novelty from what was actually seen so far
                from .fly_active_inference import Familiarity
                if getattr(self, 'familiarity', None) is None:
                    self.familiarity = Familiarity()
                e = torch.cat(ai_eyes); on = torch.cat(ai_onsets)
                tok = np.array(ai_tokens)
                e_on = e[on]; tok_on = tok[on.cpu().numpy()]
                nov = torch.as_tensor(self.familiarity.novelty(tok_on), dtype=e.dtype, device=e.device)
                prior = torch.as_tensor(1.0 + self.familiarity.n[tok_on], dtype=e.dtype, device=e.device)
                if getattr(self, 'epistemic_source', 'count') == 'belief' and reads:
                    # the brain's own Bayesian surprise (Itti & Baldi): what a word told the brain is how far it moved
                    # the brain's prediction of what comes next, KL(p after the word's interval || p just before it);
                    # a background word the brain has learned to ignore moves nothing and is worth ~0 (the entropy of
                    # the prediction would count unlearnable noise as information).  Evaluated under the beliefs (not
                    # trained by them), weighted by the eye gain over the word's ticks; per word, as the pragmatic J.
                    with torch.no_grad():
                        lp = F.log_softmax(m.decoder(m.read_norm(torch.cat(reads, 0))).float(), -1)
                    wstart = np.cumsum([0] + list(intervals))
                    epistemic = e.new_zeros(())
                    for i in range(1, len(intervals)):
                        before, after = lp[wstart[i] - 1], lp[wstart[i + 1] - 1]
                        kl = (after.exp() * (after - before)).sum()
                        sl = slice(int(wstart[i]), int(wstart[i + 1]))
                        seen = on[sl]
                        gain = e[sl][seen].mean() if bool(seen.any()) else e.new_zeros(())
                        epistemic = epistemic + gain * kl.to(e.dtype)
                    epistemic = epistemic / max(1, len(intervals))
                else:
                    # information gain of seeing the word with gain e: e pseudo-counts on a count of 1 + n(word)
                    epistemic = (0.5 * torch.log1p(e_on / prior)).sum() / max(1, len(tok_on))
                ai_loss = -float(getattr(self, 'ai_epistemic', 0.0)) * epistemic
                lam_sig = float(getattr(self, 'ai_sigreg', 0.0))
                if lam_sig > 0 and reads:
                    if getattr(self, 'sigreg', None) is None:
                        from .sigreg import SIGReg
                        self.sigreg = SIGReg(dim=reads[0].shape[1]).to(e.device)
                    ai_loss = ai_loss + lam_sig * self.sigreg(torch.cat(reads, 0).float())
                lam_out = float(getattr(self, 'ai_pragmatic', 0.0))
                commit_tick, extended = [], []
                depth = int(getattr(self, 'ai_depth', 0) or 0)
                n_words = len(intervals)
                tnoise = noise_targets if noise_targets is not None else [False] * n_words
                # observe() never puts a noisy stretch at a window's edges, so the first input (the previous window's
                # last target) and the last target are main words: every question has its answer inside the window
                assert not tnoise[0] and not tnoise[-1], 'noisy TV at a window edge'
                input_noise = [False] + list(tnoise[:-1])         # input i is the word that arrived as target i - 1
                motor = getattr(m, 'ai_mode', None) == 'motor'
                if pace > 0 and ai_out and (getattr(self, 'eye_binary', False) or motor):
                    # M3: the main speaker's questions stay open while the eyes are closed (the brain is not told the
                    # answer) and across background words; a later commit answers the oldest open question against its
                    # own answer, the main speaker's next word (fly_active_inference.pragmatic_value)
                    from .fly_active_inference import pragmatic_value
                    hz = torch.cat(ai_out)
                    # the answers' values are evaluated under the beliefs, never trained by them: preferences select
                    # acts, they do not bend predictions (else the brain trades later-tick accuracy for quick answers)
                    with torch.no_grad():
                        logp = F.log_softmax(m.decoder(m.read_norm(torch.cat(reads, 0))).float(), -1)
                    flat = targets.flatten().tolist()
                    answers, nxt = [0] * n_words, flat[-1]
                    for j in range(n_words - 1, -1, -1):
                        if not tnoise[j]:
                            nxt = flat[j]
                        answers[j] = nxt
                    asks = [not x for x in input_noise]
                    miss = -math.log(m.decoder.out_features)
                    if getattr(self, 'unigram_ce', None) is not None:
                        # not answering leaves the listener with its prior: an answer is worth what it adds to that
                        miss = [-float(self.unigram_ce[a]) for a in answers]
                    J_out, commit_tick, extended = pragmatic_value(hz, e, on, logp.detach(), intervals, answers, asks,
                                                                   miss, depth)
                    J_out = J_out / len(intervals)
                    if lam_out > 0:
                        ai_loss = ai_loss - lam_out * J_out
                    self.last_adaptive_metrics['pragmatic'] = float(J_out.detach())
                    self.last_adaptive_metrics['questions_extended'] = int(sum(x > 0 for x in extended))
                    if motor:
                        # the policy prior (habit) of active inference keeps the acts sampled: 1/gamma x KL per tick
                        from .fly_active_inference import habit_kl
                        habit = float(getattr(self, 'act_habit', 0.2))
                        kl = habit_kl(hz, habit).mean()
                        ai_loss = ai_loss + float(getattr(self, 'act_gamma_inv', 0.05)) * kl
                        with torch.no_grad():
                            hz_d = hz.detach()
                            ent = -(hz_d * hz_d.clamp_min(1e-6).log() + (1 - hz_d) * (1 - hz_d).clamp_min(1e-6).log())
                            from .fly_active_inference import precision_of
                            gamma = precision_of(m, ai_g)
                            if gamma is not None:
                                self.last_adaptive_metrics['gamma'] = gamma
                            self.last_adaptive_metrics.update({'habit_kl': float(kl.detach()),
                                                               'hazard_mean': float(hz_d.mean()),
                                                               'hazard_sd': float(hz_d.std()),
                                                               'hazard_entropy': float(ent.mean())})
                    if any(input_noise):
                        word_of = torch.repeat_interleave(torch.as_tensor(input_noise, device=e.device),
                                                          torch.as_tensor(intervals, device=e.device))
                        nz = word_of[on]
                        self.last_adaptive_metrics['eye_open_noise'] = float((e_on.detach()[nz] >= 0.5).float().mean())
                        self.last_adaptive_metrics['eye_open_text'] = float((e_on.detach()[~nz] >= 0.5).float().mean())
                elif pace > 0 and ai_out:
                    # pragmatic value of output: every arriving word is a question; committing at tick t answers it
                    # with the read standing then (value -CE_t); if the word arrives first, the answer is missed (value
                    # -log V).  J = sum_t S_{t-1} h_t (-CE_t) + S_K (-log V), S = P(not committed yet) for this word
                    hz = torch.cat(ai_out)
                    miss_value = -math.log(m.decoder.out_features)
                    J_out, start = hz.new_zeros(()), 0
                    for k in intervals:
                        h_w, ce_w = hz[start:start + k], tick_ce[start:start + k].detach()   # values under beliefs
                        S = torch.cumprod(torch.cat([h_w.new_ones(1), 1 - h_w[:-1]]), 0)
                        J_out = J_out + (S * h_w * (-ce_w)).sum() + torch.prod(1 - h_w) * miss_value
                        hits = torch.nonzero(h_w.detach() > 0.5)
                        commit_tick.append(int(hits[0]) + 1 if len(hits) else -1)
                        start += k
                    J_out = J_out / len(intervals)
                    if lam_out > 0:
                        ai_loss = ai_loss - lam_out * J_out
                    self.last_adaptive_metrics['pragmatic'] = float(J_out.detach())
                self.last_act_loss = ai_loss
                self.familiarity.see(tok_on, e_on.detach().cpu().numpy())
                self.ai_g = ai_g.detach()
                with torch.no_grad():
                    commits = torch.cat(ai_out).detach().round()
                    self.last_adaptive_metrics.update({
                        'eye_mean_onset': float(e_on.mean()), 'eye_min_onset': float(e_on.min()),
                        'eye_closed_frac_onset': float((e_on < 0.5).float().mean()), 'eye_mean_tick': float(e.mean()),
                        'novelty_mean': float(nov.mean()), 'epistemic': float(epistemic), 'commits': int(commits.sum())})
                    # the per-word trace for the analysis: eye at its onset, its arrival CE, its pace
                    self.last_ai_trace = {'eye': e_on.cpu().tolist(), 'token': tok_on.tolist(),
                                          'ce': scores.detach().cpu().tolist(), 'pace': pace,
                                          'interval': list(intervals), 'commit_tick': commit_tick,
                                          'extended': extended, 'noise': input_noise, 'sigma': sense_sigmas,
                                          'eye_tick': e.detach().cpu().tolist() if self.hold_input else None,
                                          'hazard': (torch.cat(ai_out).detach().cpu().tolist()
                                                     if getattr(m, 'ai_mode', None) == 'motor' else None),
                                          'gamma': self.last_adaptive_metrics.get('gamma'),
                                          # every tick's read CE against its word's target: the value of waiting
                                          'tick_ce': tick_ce.detach().cpu().tolist() if pace > 0 else None}
            if world is not None:
                # the act's fitness, as evolution sees it (never an input to the brain); BPTT through the hazards
                from .fly_odour_world import act_return
                h = torch.cat(hazards, 0)
                J, self.act_carry = act_return(h[:, 0], h[:, 1], records, getattr(self, 'act_carry', {}))
                self.last_act_loss = (self.last_act_loss if self.last_act_loss is not None else 0.0) \
                    - float(getattr(self, 'act_lambda', 1.0)) * J
                if getattr(m, 'sense_head', None) is not None and reads:
                    # the brain's expectation of its other senses: after tick t it predicts what tick t + 1 brings
                    # (odour levels, taste); scored against what came.  Self-caused rises (its own approach) vs
                    # world-caused (gusts) tell whether it models its own effect on its input.
                    pred = m.sense_head(m.read_norm(torch.cat(reads, 0)))[:-1]
                    target = torch.as_tensor(np.stack([r['sense'] for r in records[1:]]), device=pred.device)
                    err = ((pred - target) ** 2).sum(1)
                    self.last_act_loss = self.last_act_loss + float(getattr(self, 'sense_lambda', 1.0)) * err.mean()
                    with torch.no_grad():
                        ramp = np.array([r['ramp'] or '' for r in records[1:]])
                        sense_metrics = {'sense_err': float(err.mean())}
                        for kind in ('self', 'world'):
                            sel = torch.as_tensor(ramp == kind, device=err.device)
                            sense_metrics[f'sense_err_{kind}_ramp'] = float(err[sel].mean()) if bool(sel.any()) else None
                            sense_metrics[f'{kind}_ramp_ticks'] = int(sel.sum())
                        on = torch.as_tensor(target.sum(1) > 0)
                        sense_metrics['sense_err_on_input'] = float(err[on].mean()) if bool(on.any()) else None
                self.act_g = act_g.detach()
                acts = h.detach().round()
                self.last_adaptive_metrics.update({
                    'act_return': float(J.detach()), 'acts_approach': int(acts[:, 0].sum()), 'acts_avoid': int(acts[:, 1].sum()),
                    'live_ticks': int(sum(r['live'] for r in records)), 'gap_ticks': int(sum(r.get('gap', False) for r in records)),
                    'act_g_mean': float(act_g.detach().mean())})
                if getattr(m, 'sense_head', None) is not None and reads:
                    self.last_adaptive_metrics.update(sense_metrics)
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
                    hold_input=self.hold_input,
                    use_reentrant=False
                )
                for k in range(token_lat.size(0)):
                    latents.append(token_lat[k])
                h_list.append(state.h)
                state_list.append(state)
            else:
                state, tick_states = advance_fly_input_event(
                    m, state, token, hold_input=self.hold_input,
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
        noise = getattr(self, 'noise_world', None)
        self.noise_mask = None
        if noise is not None and tokens.numel() > noise['length'] + 2:
            # the noisy TV: with probability `rate` per window a stretch of `length` words drawn uniformly from a small
            # fixed alphabet -- the words are learnable as a set, their order is not
            if getattr(self, 'noise_rng', None) is None:
                self.noise_rng = np.random.default_rng(int(noise.get('seed', 0)) + 1 + int(noise.get('offset', 0)))
            mask = np.zeros(tokens.numel(), bool)
            if self.noise_rng.random() < noise['rate']:
                # never at the window's edges: a main word comes before and after the stretch inside the window
                at = int(self.noise_rng.integers(1, tokens.numel() - noise['length']))
                draw = self.noise_rng.choice(np.asarray(noise['alphabet']), noise['length'])
                tokens = tokens.clone()
                tokens[at:at + noise['length']] = torch.as_tensor(draw, device=tokens.device)
                mask[at:at + noise['length']] = True
            self.noise_mask = mask.tolist()
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
        if getattr(self, 'last_act_loss', None) is not None:
            loss = loss + self.last_act_loss
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
        return self._finish_update(loss, scores, next_state, features, tokens)

    def _finish_update(self, loss, scores, next_state, features, tokens, ticks=None):
        """Backward, one parameter update and the move to the next state (shared by the text window and the taste
        episode).  ticks: physical ticks this update covered when the clock is the caller's own."""
        if not torch.isfinite(loss).item():
            raise FloatingPointError('Non-finite BPTT training loss')
        if self.runner is None:
            from .triton_synapse import direct_edge_grad
            with direct_edge_grad():
                loss.backward()
            if getattr(self, '_credit', None) is not None:
                from .fly_carried_credit import apply_carried_credit
                added = apply_carried_credit(self.model, *self._credit)
                self._credit = None
                if isinstance(getattr(self, 'last_adaptive_metrics', None), dict):
                    self.last_adaptive_metrics.update({f'carried_{k}': v for k, v in added.items()})
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
        try:
            grad_norm = stable_clip_grad_norm_(self.trainable, self.max_grad_norm,
                                              error_if_nonfinite=True)
        except FloatingPointError:
            # name the parameters (FLY_DETECT_ANOMALY=1 FLY_COMPILE_TICK=0 names the operation); a living brain with the
            # 'absolute' surrogate (unit width against a 0.1 threshold) explodes to inf through 256 ticks
            bad = [name for name, p in self.model.named_parameters()
                   if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()]
            print(f'non-finite gradient in: {bad}', flush=True)
            raise
        if os.environ.get('FLY_GRAD_PROBE') and hasattr(self.model, 'edge_post_e'):
            # does the loss push the drive onto the read cells one way?  (grad < 0 = more weight lowers the loss)
            with torch.no_grad():
                g = self.model.edge_weight_e.grad
                if g is not None:
                    onto = torch.zeros(self.model.n_neurons, dtype=torch.bool, device=g.device)
                    onto[self.model.read_indices] = True
                    m = onto[self.model.edge_post_e.long()]
                    groups = [('onto read', m), ('elsewhere', ~m)]
                    if os.environ.get('FLY_GRAD_PROBE_MASK'):
                        lat = torch.as_tensor(np.load(os.environ['FLY_GRAD_PROBE_MASK']), device=g.device)
                        a, b = lat[self.model.edge_pre_e.long()], lat[self.model.edge_post_e.long()]
                        groups += [('latched->latched', a & b), ('latched->other', a & ~b)]
                    for label, sel in groups:
                        gs = g[sel]; nz = gs != 0
                        print(f'GRAD PROBE update {self.updates} {label}: edges with gradient {float(nz.float().mean()):.3f}, '
                              f'push up {float((gs[nz] < 0).float().mean()):.3f}, mean grad / rms '
                              f'{float(gs.mean() / gs.pow(2).mean().sqrt().clamp_min(1e-30)):+.4f}', flush=True)
        evidence = getattr(self, 'evidence', False)
        if evidence:
            physio_grads = self._evidence_take(self.physio_params)
        self.optimizer.step()
        if evidence:
            self._evidence_apply(self.physio_names, self.physio_params, physio_grads, self.physio_lr)
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
            before = None if getattr(self, 'eg', False) else [w.detach().clone() for w in self.edges]
        relative = getattr(self, 'edge_step_scale', None)
        eg = getattr(self, 'eg', False)
        if relative is not None and local is None and not eg:
            before = [w.detach().clone() for w in self.edges]
        if (evidence and getattr(self, 'evidence_edges', True)) or eg:
            edge_grads = self._evidence_take(self.edges)
        self.sgd.step()
        if eg:
            self._eg_apply(edge_grads, local)
        if evidence:
            if getattr(self, 'evidence_edges', True):
                # raw evidence step; the relative / dopamine scales below multiply it like any other synapse update
                self._evidence_apply(self.edge_names, self.edges, edge_grads, self.sgd.param_groups[1]['lr'])
            self.evidence_count += 1
        if not eg and (relative is not None or local is not None):
            # one composed scale per synapse on its Adam update: relative plasticity (each synapse moves by rate x its
            # own birth weight per full Adam step, so the connectome's scale is kept) times the dopamine at its
            # terminal (local, within [1/4, 4]); both multiply the same update from the weights before the step
            compose_edge_updates(self.model, self.edge_names, self.edges, before, relative, local,
                                 getattr(self.model, 'dopamine_site', 'post'))
        if local is not None:
            self.last_dopamine_plasticity_spread = float(local[local != 1].std()) if bool((local != 1).any()) else 0.0
        for group, rate in zip(self.sgd.param_groups, base_rates):
            group['lr'] = rate
        self.last_plasticity = plasticity
        with torch.no_grad():
            if evidence and self.pull_tau > 0:
                self._evidence_pull()
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
        if ticks is not None:
            self.physical_ticks += int(ticks)
        elif getattr(self, 'go_neuron', False):
            self.physical_ticks += self.last_adaptive_metrics['adaptive_total_ticks']
        elif getattr(self, 'stream_ticks', 0):
            self.physical_ticks += (self.last_adaptive_metrics.get('adaptive_total_ticks', len(values) * self.stream_ticks)
                                    if getattr(self, 'last_adaptive_metrics', None) else len(values) * self.stream_ticks)
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
        if getattr(self, 'eg', False) and self.eg_moved:
            group_norms['eg_log_change_mean_p99'] = dict(self.eg_moved)
        if getattr(self, 'evidence', False) and self.evidence_passed:
            group_norms['evidence_passed'] = dict(self.evidence_passed)
        if getattr(self.model, 'online_da_eta', None) is not None and next_state.da_memory.numel():
            phi = next_state.da_memory[0][self.model.online_da_target[0] > 0].detach()
            group_norms['online_da_memory_sd'] = float(phi.std())
            group_norms['online_da_memory_absmax'] = float(phi.abs().max())
            kc = self.model.online_da_target[0] > 0
            dop = next_state.dopamine.detach()
            tonic = dop[1]
            group_norms['online_da_phasic_abs'] = float((dopamine_occupancy(dop[0], self.model.dopamine_affinity) - dopamine_occupancy(tonic, self.model.dopamine_affinity))[kc].abs().mean())
            group_norms['online_da_trace_mean'] = float(next_state.da_trace[0][kc].mean())
            group_norms['online_da_eta'] = float(self.model.online_da_eta.detach())
            group_norms['online_da_memory_mean'] = float(phi.mean())
        return values, {'grad_norm_before_clip': float(grad_norm), **group_norms, **getattr(self, 'last_ctm_metrics', {}), **getattr(self, 'last_adaptive_metrics', {})}

    def observe_taste(self, tokens, drives, held, targets, read_windows, baseline_window, token_ticks=8):
        """One taste episode as one update (design: FLY_CONSEQUENCE_WORLD_DESIGN_20261012.html).  tokens: the episode's
        input words on a fixed clock of token_ticks ticks each; drives: {tick: external current} (the dopamine cells
        driven during pairing); held: words held on the sensory surface for all their ticks (a sustained odour);
        targets: 1 (go: the cue was sweet) or 0 for each test; read_windows: (start, end) ticks of each test's read;
        baseline_window: (start, end) of the pre-test baseline.  Two heads, both trained: the MBONs' valence balance
        A = sum v_j (n_j(read) - n_j(baseline) * len(read) / len(baseline)) / sum |v_j| (v_j fixed by the connectome,
        gain and bias learned) and the text read's logit(yes) - logit(no) at the end of each read window."""
        m = self.model
        rates, thresholds = m.get_decay_rates(), m.get_thresholds()
        gains, alif, stp = m.get_conductance_gains(), m.get_alif_params(), m.get_stp_params()
        opts = dict(base_rates=rates, thresholds=thresholds, conductance_gains=gains, alif_params=alif, stp_params=stp)
        state, acts, reads, t = self.state, [], {}, 0
        ends = {end for _, end in read_windows}
        for i, token in enumerate(tokens):
            tok = torch.tensor([int(token)], device=state.h.device)
            per_tick = [drives.get(t + k) for k in range(token_ticks)]
            hold = int(token) in held
            if self.use_checkpointing and m.training and torch.is_grad_enabled():
                state, act = checkpoint(taste_token_ticks, m, state, tok, per_tick, hold, opts, use_reentrant=False)
            else:
                state, act = taste_token_ticks(m, state, tok, per_tick, hold, opts)
            acts.append(act)
            t += token_ticks
            if t in ends:
                reads[t] = read_fly_motor(m, state)
        A = torch.cat(acts, 0)                                        # ticks x n_MBON (transmitted pulses)
        b0, b1 = baseline_window
        base = A[b0:b1].sum(0)
        loss, out = 0.0, []
        for (r0, r1), target in zip(read_windows, targets):
            dn = A[r0:r1].sum(0) - base * ((r1 - r0) / (b1 - b0))
            a = (m.taste_valence * dn).sum() / m.taste_valence.abs().sum()
            logit_mb = m.taste_gain * a + m.taste_bias
            lg = m.decoder(m.read_norm(reads[r1])).float().reshape(-1)
            logit_text = lg[TASTE_YES] - lg[TASTE_NO]
            y = torch.tensor(float(target), device=a.device)
            loss = loss + F.binary_cross_entropy_with_logits(logit_mb, y) + F.binary_cross_entropy_with_logits(logit_text, y)
            out.append({'target': int(target), 'mb_logit': float(logit_mb.detach()), 'text_logit': float(logit_text.detach())})
        loss = loss / len(read_windows)
        token_tensor = torch.as_tensor(list(tokens), device=state.h.device)
        empty = torch.empty(0, device=state.h.device)
        _, metrics = self._finish_update(loss, empty, state, [], token_tensor, ticks=t)
        metrics.update(taste_loss=float(loss.detach()),
                       taste_mb_correct=float(np.mean([(o['mb_logit'] > 0) == bool(o['target']) for o in out])),
                       taste_text_correct=float(np.mean([(o['text_logit'] > 0) == bool(o['target']) for o in out])),
                       taste_tests=out)
        return metrics

    @torch.no_grad()
    def set_relative_edge_steps(self, graph_npz, rate):
        """Per-synapse step: rate x |birth weight| per full Adam step (Adam's own steps are ~lr whatever the weight;
        the birth weights' median is 7.4e-4 against lr 2e-4, so every update moved a typical synapse by ~27%).  Stores
        the multiplier |w0| * rate / lr_group for every edge list."""
        import numpy as np
        graph = np.load(graph_npz, allow_pickle=True)
        lr = self.sgd.param_groups[1]['lr']
        scales = []
        for name, w in zip(self.edge_names, self.edges):
            birth = torch.as_tensor(np.abs(graph[name]), dtype=w.dtype, device=w.device)
            if birth.shape != w.shape:
                raise ValueError(f'{name}: birth weights {tuple(birth.shape)} vs {tuple(w.shape)}')
            scales.append(birth * (float(rate) / lr))
        self.edge_step_scale = scales
        self.edge_step_rate = float(rate)
        return float(torch.cat([s.flatten() for s in scales]).median() * lr)

    @torch.no_grad()
    def attach_ablation(self, pre_cells=None, read_cells=None):
        """A living ablation (the causal gates): the individual keeps learning, with a part of its brain removed --
        pre_cells (bool over neurons): their output synapses cut; read_cells: their read weights zeroed.  Re-applied
        after every update, so learning cannot regrow what was removed; under EG each cell keeps the input totals of
        the ablated brain (the remaining synapses are not scaled up to replace the removed ones)."""
        self.ablate_pre, self.ablate_read = pre_cells, read_cells
        self._apply_ablation()
        if getattr(self, 'eg', False):
            for n, w in zip(self.edge_names, self.edges):
                post = getattr(self.model, 'edge_post' + n[len('edge_weight'):]).long()
                self.eg_total[n] = torch.zeros(self.model.n_neurons, device=w.device).index_add_(0, post, w.float())

    @torch.no_grad()
    def _apply_ablation(self):
        if getattr(self, 'ablate_pre', None) is not None:
            for n, w in zip(self.edge_names, self.edges):
                w[self.ablate_pre[getattr(self.model, 'edge_pre' + n[len('edge_weight'):]).long()]] = 0.0
        if getattr(self, 'ablate_read', None) is not None and self.model.read_surface != 'all':
            self.model.output_read.weight[:, self.ablate_read[self.model.read_indices]] = 0.0

    def clamp_edges(self):
        """Retain conductance positivity or each signed edge's original sign (and keep the plastic Kenyon -> MBON
        weights excitatory).

        Initially zero signed edges belong to the nonnegative class. Their
        mask is saved, so a clamped negative edge reaching zero stays negative.
        """
        if getattr(self.model, 'comp_weight', None) is not None:
            self.model.comp_weight.clamp_(min=0.0)
        for name, weight in zip(self.edge_names, self.edges):
            if self.model.synapse_model == 'coba':
                cap = getattr(self, 'edge_cap', {}).get(name)
                if getattr(self, 'eg', False):
                    weight.clamp_(min=0.0)
                elif cap is None:
                    weight.clamp_(0.0, 5.0)
                else:
                    # the conductance bound never cuts a birth synapse (the rescaled graph has inhibitory edges to
                    # 14.6); learned growth is bounded at 5 as before
                    weight.clamp_(min=0.0)
                    torch.minimum(weight, cap, out=weight)
                # edges pinned at 0 (Kenyon axo-axonal synapses, attach_kenyon_dendritic_only)
                pinned = getattr(self.model, 'pinned_zero_edges' + name[len('edge_weight'):], None)
                if pinned is not None:
                    weight[pinned] = 0.0
            else:
                weight.copy_(torch.where(self.edge_positive[name], weight.clamp(0.0, 5.0),
                                         weight.clamp(-5.0, 0.0)))
        self._apply_ablation()

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

    def attach_evidence_plasticity(self, pull_tau=0.0, saved=None, z=3.0, beta=0.99, edges=True):
        """Evidence-gated plasticity for the synapses and the class physiology (R3 runaway, 2026-10-13).

        R3 measured: the gradient on a synapse is noise from update to update (sign flips, |m|/sqrt(v) at the noise
        level, only chance-level synapses significant), yet Adam moves every synapse a full step on it -- a random walk.
        Latching is an absorbing side of that walk (above threshold is unbounded and nothing pushes back), so the walk
        ratchets cells into latching, and sorting synapses by the outcome makes the walk look directional (the loops of
        cells that latched are the ones that happened to grow).  Rule here: a parameter keeps a running mean and second
        moment of its gradient (EMA beta, ~1/(1-beta) updates) and steps only when the mean is significant, |mean| > z
        standard errors of that running mean; then by one nominal step (lr; synapses: the relative and dopamine scales
        below) against the mean's sign.  Pure noise passes ~0.3% of the time at z = 3; consistent evidence moves at
        the full rate (synaptic tagging / consolidation: change needs repeated agreement).  pull_tau > 0: change decays
        toward the birth value with that time constant in updates (backstop for anything that drifted).  The birth
        values and the running moments travel with the learner's checkpoint."""
        named = dict(self.model.named_parameters())
        self.physio_names = list(getattr(self, 'physio_names', []))
        if not self.physio_names:
            print('evidence plasticity: no separate physiology group (lr_physiology unset); synapses only', flush=True)
        self.physio_params = [named[n] for n in self.physio_names]
        self.physio_lr = float(getattr(self, 'physio_lr', 0.0))
        ev = (saved or {}).get('evidence') or {}
        device = self.state.h.device
        # edges=False: the synapses learn by another rule (EG); the gate then governs the class physiology only
        self.evidence_edges = bool(edges)
        names = (list(self.edge_names) if edges else []) + self.physio_names
        params = (list(self.edges) if edges else []) + self.physio_params
        self.evidence_names, self.evidence_params = names, params
        self.evidence_z, self.evidence_beta = float(ev.get('z', z)), float(ev.get('beta', beta))
        self.evidence_count = int(ev.get('count', 0))
        moments = ev.get('moments') or {}
        self.evidence_moments = {n: (moments[n][0].to(device), moments[n][1].to(device)) if n in moments
                                 else (torch.zeros_like(p, dtype=torch.float32), torch.zeros_like(p, dtype=torch.float32))
                                 for n, p in zip(names, params)}
        self.pull_tau = float(pull_tau)
        if ev.get('birth') is not None:
            self.evidence_birth = {n: ev['birth'][n].to(device) for n in names}
        else:
            if saved is not None:
                print('evidence plasticity: no birth values in the checkpoint -- the current values become the reference',
                      flush=True)
            self.evidence_birth = {n: p.detach().clone() for n, p in zip(names, params)}
        if edges:
            self.edge_cap = {n: self.evidence_birth[n].clamp_min(5.0) for n in self.edge_names}
        self.evidence_passed = {}
        self.evidence = True

    def attach_eg_plasticity(self, rate, tau=0.0, saved=None, momentum=0.9):
        """Normalized exponentiated gradient on every cell's input, with the birth connectome as its prior
        (Kivinen & Warmuth 1997; Cornford, Pogodin et al. 2024 'Brain-like learning with exponentiated gradients').

        R3 (2026-10-13): the gradient of synapse j -> i is delta_i x a_j, so a cell's inputs share the sign of its own
        error (per-cell sign agreement median 0.79, 46% of the synapse momentum variance is the post cell): the noise
        lives in each cell's gain, and per-parameter Adam turned it into full steps -> every cell's total drive walked
        and the ones that crossed latched.  Rule: log w <- log w - eta m (m: gradient momentum; one eta per tensor,
        rate / running RMS of m, so a synapse moves with its own evidence and is never normalised alone) - (log w -
        log w_birth) / tau (KL to the birth connectome: unsupported change returns, consistent evidence moves far);
        then every cell's excitatory and inhibitory input totals are rescaled to their birth totals (the normalized
        EG: learning redistributes what a cell listens to, it cannot change how much it is driven).  Multiplicative:
        signs never change (Dale), weights stay log-normal, zero stays zero.  The dopamine at the chosen end scales
        each synapse's step (fixed rule)."""
        ev = (saved or {}).get('eg') or {}
        device = self.state.h.device
        self.eg_rate, self.eg_tau, self.eg_momentum = float(rate), float(tau), float(momentum)
        self.eg_count = int(ev.get('count', 0))
        self.eg_rms = {n: torch.as_tensor(v, device=device) for n, v in ev.get('rms', {}).items()}
        mom = ev.get('mom') or {}
        self.eg_mom = {n: mom[n].to(device) if n in mom else torch.zeros_like(w, dtype=torch.float32)
                       for n, w in zip(self.edge_names, self.edges)}
        if ev.get('birth') is not None:
            self.eg_birth = {n: ev['birth'][n].to(device) for n in self.edge_names}
        else:
            if saved is not None:
                print('EG plasticity: no birth values in the checkpoint -- the current weights become the prior', flush=True)
            self.eg_birth = {n: w.detach().clone() for n, w in zip(self.edge_names, self.edges)}
        self.eg_total = {}
        for n in self.edge_names:
            post = getattr(self.model, 'edge_post' + n[len('edge_weight'):]).long()
            self.eg_total[n] = torch.zeros(self.model.n_neurons, device=device).index_add_(0, post, self.eg_birth[n].float())
        self.eg_moved = {}
        self.eg = True

    @torch.no_grad()
    def _eg_apply(self, grads, local):
        k = 1.0 / self.eg_tau if self.eg_tau > 0 else 0.0
        site = getattr(self.model, 'dopamine_site', 'post')
        for n, w, g in zip(self.edge_names, self.edges, grads):
            sfx = n[len('edge_weight'):]
            mom = self.eg_mom[n]
            mom.mul_(self.eg_momentum)
            if g is not None:
                mom.add_(g.float())
            ms = mom.pow(2).mean()
            prev = self.eg_rms.get(n)
            ms = ms if prev is None else 0.99 * prev + 0.01 * ms
            self.eg_rms[n] = ms
            dlog = mom * (-self.eg_rate / float(ms.sqrt().clamp_min(1e-30)))
            if local is not None:
                dlog.mul_(dopamine_edge_scale(self.model, sfx, local, site))
            alive = w > 0
            if k > 0:
                dlog.add_(torch.where(alive, torch.log(self.eg_birth[n].float().clamp_min(1e-30))
                                      - torch.log(w.float().clamp_min(1e-30)), torch.zeros_like(dlog)), alpha=k)
            w.mul_(torch.exp(dlog.clamp_(-1.0, 1.0)).to(w.dtype))   # at most e-fold per update
            post = getattr(self.model, 'edge_post' + sfx).long()
            total = torch.zeros(self.model.n_neurons, device=w.device).index_add_(0, post, w.float())
            w.mul_((self.eg_total[n] / total.clamp_min(1e-30))[post].to(w.dtype))
            ratio = w.float() / self.eg_birth[n].float().clamp_min(1e-30)
            lr_ = torch.log(ratio[alive].clamp_min(1e-30))
            if lr_.numel() == 0:                          # every synapse removed (the living noedges gate)
                self.eg_moved[n] = [0.0, 0.0]
                continue
            self.eg_moved[n] = [float(lr_.abs().mean()), float(lr_.abs().quantile(0.99)) if lr_.numel() < 2**24 else
                                float(lr_.abs()[torch.randint(0, lr_.numel(), (1 << 20,), device=lr_.device)].quantile(0.99))]
        self.eg_count += 1

    def eg_state(self):
        if not getattr(self, 'eg', False):
            return None
        return {'rate': self.eg_rate, 'tau': self.eg_tau, 'count': self.eg_count,
                'rms': {n: float(v) for n, v in self.eg_rms.items()}, 'mom': self.eg_mom, 'birth': self.eg_birth}

    @staticmethod
    def _evidence_take(params):
        grads = []
        for p in params:
            grads.append(p.grad)
            p.grad = None                       # the Adam optimizers skip a parameter without a gradient
        return grads

    @torch.no_grad()
    def _evidence_apply(self, names, params, grads, lr):
        beta, c = self.evidence_beta, self.evidence_count + 1
        correction = 1.0 - beta ** c
        # the bias-corrected EMA over c updates is a weighted mean with weights beta^k: its effective sample size is
        # (sum w)^2 / sum w^2, its standard error sd / sqrt(n_eff), and the weighted variance is unbiased by
        # n_eff / (n_eff - 1) (-> sqrt((1 - beta) / (1 + beta)) for c >> 1 / (1 - beta))
        n_eff = correction ** 2 / (1.0 - beta) ** 2 / ((1.0 - beta ** (2 * c)) / (1.0 - beta ** 2))
        se_factor = (1.0 / max(n_eff - 1.0, 1e-6)) ** 0.5
        for n, p, g in zip(names, params, grads):
            m, v = self.evidence_moments[n]
            g = torch.zeros_like(m) if g is None else g.float()
            m.mul_(beta).add_(g, alpha=1.0 - beta)
            v.mul_(beta).addcmul_(g, g, value=1.0 - beta)
            if c < 32:                          # no verdict before ~32 samples (small-sample tails: 2% false at 8)
                continue
            mean = m / correction
            sd = (v / correction - mean * mean).clamp_min(0.0).sqrt()      # se = sd * sqrt(n/(n-1)) / sqrt(n)
            passed = mean.abs() > self.evidence_z * se_factor * sd
            self.evidence_passed[n] = float(passed.float().mean())
            p.sub_((float(lr) * torch.sign(mean) * passed).to(p.dtype))

    @torch.no_grad()
    def _evidence_pull(self):
        k = 1.0 / self.pull_tau
        for n, p in zip(self.evidence_names, self.evidence_params):
            p.lerp_(self.evidence_birth[n].to(p.dtype), k)

    def evidence_state(self):
        if not getattr(self, 'evidence', False):
            return None
        return {'z': self.evidence_z, 'beta': self.evidence_beta, 'count': self.evidence_count,
                'pull_tau': self.pull_tau, 'moments': self.evidence_moments, 'birth': self.evidence_birth}

    def state_dict(self):
        return {'evidence': self.evidence_state(), 'eg': self.eg_state(),
                'physical': self.state.state_dict(), 'optimizer': self.optimizer.state_dict(),
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
                'go_reference_energy': getattr(self, 'go_reference_energy', 0.07),
                # the act head's integrator and the lifetime count of what was seen: never reset across chunks
                'ai_g': getattr(self, 'ai_g', None),
                'familiarity_n': (self.familiarity.n if getattr(self, 'familiarity', None) is not None else None)}

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


# Physiology that is learned per cell class (time constants, threshold, adaptation, short-term plasticity, dopamine
# sensitivity); splitting a class gives its cells their own values.
CLASS_PARAMETER_NAMES = ('log_tau_m', 'log_tau_s_e', 'log_tau_s_i', 'log_threshold', 'log_tau_a', 'log_beta',
                         'logit_u0', 'log_tau_fac', 'log_tau_rec', 'dopamine_sensitivity', 'log_input_gain',
                         'log_kc_release')


def attach_input_gain(model):
    """A learnable input gain per physiology class, 1 at attachment (call before split_by_type so the type classes
    inherit it).  2026-10-13: with one leak for every cell a big neuron's drive grows with its synapse count (total input
    ~ size^0.93: motor / descending 21-26 thresholds, they latch); scaling every input by median size / size removed the
    latching but silenced the output cells and the taste lines.  The mechanism (input conductance grows with the cell)
    is fixed; how much is a number the individual learns."""
    model.log_input_gain = nn.Parameter(torch.zeros(model.n_superclasses, device=model.read_indices.device))
    return model


def init_input_gain_from_size(model, graph_npz, exponent=0.5):
    """Start every non-sensory class at gain (median size / class median size)^exponent (0 = uniform synapses, 1 =
    inputs inverse to size; 0.5 = halfway, in log).  Classes without size data, and sensory classes, stay at 1."""
    import pandas as pd
    from pathlib import Path
    graph = np.load(graph_npz, allow_pickle=True)
    size = pd.read_feather(Path(graph_npz).parent / 'neuron_sizes.feather').set_index('bodyId').reindex(
        graph['neuron_body_ids'])['size'].values.astype(float)
    sc_name = np.array(graph['superclass_names'])[graph['superclass_id']].astype(str)
    cls = model.superclass_id.cpu().numpy().astype(np.int64)
    ok = np.isfinite(size) & (size > 0) & (np.char.find(sc_name, 'sensory') < 0)
    med = np.median(size[np.isfinite(size) & (size > 0)])
    gains = np.zeros(model.n_superclasses)
    df = pd.DataFrame({'c': cls[ok], 's': size[ok]}).groupby('c')['s'].median()
    gains[df.index.values] = exponent * np.log(med / df.values)
    with torch.no_grad():
        model.log_input_gain.copy_(torch.as_tensor(gains, dtype=torch.float32, device=model.log_input_gain.device))
    return float(np.exp(gains[df.index.values]).min()), float(np.exp(np.median(gains[df.index.values]))), float(np.exp(gains[df.index.values]).max())
KENYON_SUBTYPES = (('KC_ab_core', ('KCab-c',)), ('KC_ab_surface', ('KCab-s', 'KCab-m')), ('KC_ab_posterior', ('KCab-p',)),
                   ("KC_apbp", ("KCa'b'",)), ('KC_g', ('KCg',)))


def split_kenyon_subtypes(model, graph_npz, annotations, record=True):
    """Connectome audit (point-model coarse-graining): all 4,064 Kenyon cells sat in cb_intrinsic with 32,160 other
    central-brain neurons, sharing one membrane time constant, threshold, adaptation and STP -- the alpha/beta core
    (FoxP; the population that accumulates evidence, DasGupta et al. 2014; Groschner et al. 2018) could not differ from
    anything.  The connectome records wiring, not channels, so these values are for evolution (BPTT) to find.  Each
    subtype becomes its own class, initialised from the class it leaves (dynamics unchanged at the split).  Records the
    source class of every new class in model.class_split_sources (for the optimizer's moments).  Returns the counts."""
    import pandas as pd
    graph = np.load(graph_npz, allow_pickle=True)
    ann = pd.read_feather(annotations).set_index('bodyId').reindex(graph['neuron_body_ids'])
    typ = ann['type'].astype(str).values.astype(str)
    groups = []
    for name, prefixes in KENYON_SUBTYPES:
        mask = np.zeros(len(typ), bool)
        for p in prefixes:
            mask |= np.char.startswith(typ, p)
        groups.append((name, mask))
    return split_classes(model, groups, record=record)


def split_classes(model, groups, record=True):
    """Each (name, mask) group leaves its class and becomes its own class, initialised from the class it leaves
    (dynamics unchanged at the split).  record: the new classes' sources are appended to model.class_split_sources
    (the optimizer's moments of a checkpoint without them are padded from their source class); a split carried over
    from a lineage whose checkpoint already has the classes is not recorded."""
    sc = model.superclass_id
    sources, counts = [], {}
    with torch.no_grad():
        for name, mask in groups:
            if not mask.any():
                continue
            m = torch.as_tensor(mask, device=sc.device)
            sources.append(int(torch.mode(sc[m]).values))
            sc[m] = model.n_superclasses + len(sources) - 1
            counts[name] = int(mask.sum())
        for pname in CLASS_PARAMETER_NAMES:
            p = getattr(model, pname, None)
            if p is None:
                continue
            grown = torch.cat([p.data, p.data[sources]])
            setattr(model, pname, nn.Parameter(grown, requires_grad=p.requires_grad))
    model.n_superclasses += len(sources)
    if record:
        model.class_split_sources = list(getattr(model, 'class_split_sources', None) or []) + sources
    for attr in ('_compiled_coefficients', '_compiled_finish'):
        if hasattr(model, attr):
            setattr(model, attr, None)
    return counts


def split_by_type(model, graph_npz, annotations, record=True):
    """Every annotated cell type gets its own physiology class (time constants, threshold, adaptation, dopamine
    sensitivity), initialised from the superclass most of its cells belong to; untyped cells keep their superclass.
    Audit 2026-10-13: one class for 89k optic-lobe cells (or 32k central-brain cells) let learning move them only
    together.  Vectorised (11.7k types); supersedes the Kenyon-subtype and modulatory splits.  Returns the number of
    type classes."""
    import pandas as pd
    graph = np.load(graph_npz, allow_pickle=True)
    typ = pd.read_feather(annotations).set_index('bodyId').reindex(graph['neuron_body_ids'])['type'].astype(str).values
    typed = ~np.isin(typ, ['None', 'nan', ''])
    names, inv = np.unique(typ[typed], return_inverse=True)
    sc = model.superclass_id
    old = sc.cpu().numpy().astype(np.int64)
    # source class of each type: the most common superclass among its cells
    pair = inv.astype(np.int64) * (old.max() + 1) + old[typed]
    counts = np.bincount(pair, minlength=len(names) * (old.max() + 1)).reshape(len(names), old.max() + 1)
    sources = counts.argmax(1).tolist()
    new = old.copy()
    new[typed] = model.n_superclasses + inv
    with torch.no_grad():
        sc.copy_(torch.as_tensor(new, dtype=sc.dtype, device=sc.device))
        for pname in CLASS_PARAMETER_NAMES:
            p = getattr(model, pname, None)
            if p is None:
                continue
            grown = torch.cat([p.data, p.data[sources]])
            setattr(model, pname, nn.Parameter(grown, requires_grad=p.requires_grad))
    model.n_superclasses += len(sources)
    if record:
        model.class_split_sources = list(getattr(model, 'class_split_sources', None) or []) + sources
    for attr in ('_compiled_coefficients', '_compiled_finish'):
        if hasattr(model, attr):
            setattr(model, attr, None)
    return len(sources)


def split_modulatory_classes(model, graph_npz, annotations, record=True):
    """Connectome audit (same coarse-graining as the Kenyon cells): 384 of the 392 dopamine cells, the octopaminergic
    and the serotonergic cells sat in cb_intrinsic with ~32,000 other neurons and shared its physiology -- the cells
    that are tonically active in the fly and set the brain's state could not differ from anything.  PAM (reward), PPL1
    (punishment), the other dopamine cells, OA and 5-HT each become a class, initialised from the class they leave."""
    import pandas as pd
    graph = np.load(graph_npz, allow_pickle=True)
    typ = pd.read_feather(annotations).set_index('bodyId').reindex(graph['neuron_body_ids'])['type'].astype(str).values.astype(str)
    dan = np.zeros(len(typ), bool)
    dan[graph['dan_indices']] = True
    pam, ppl1 = dan & np.char.startswith(typ, 'PAM'), dan & np.char.startswith(typ, 'PPL1')
    groups = [('DAN_PAM', pam), ('DAN_PPL1', ppl1), ('DAN_other', dan & ~pam & ~ppl1),
              ('OA', ~dan & np.char.startswith(typ, 'OA-')), ('5HT', ~dan & np.char.startswith(typ, '5-HT'))]
    return split_classes(model, groups, record=record)
