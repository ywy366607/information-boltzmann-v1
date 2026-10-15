"""Credit beyond the BPTT window at O(1) memory (GOAL: docs/information_boltzmann/FLY_ACTIVE_INFERENCE_GOAL_20261013.md,
section 'Credit beyond the 32-word window').

BPTT sees a parameter's effects inside one window.  A slow state carried across windows (the compartment memory phi,
the dopamine concentrations, the STP resources) also carries the parameter's past effects into the window: with the
window's opening state s_0 a leaf, BPTT gives G = dL_w / ds_0, and the missing credit is G . S_0, S_0 = ds_0 / dtheta
from all earlier windows.  S is advanced every tick by the state's own local recursion (forward mode, as e-prop's
eligibility traces), holding the spikes and the dopamine arriving from other cells fixed (conditional sensitivity:
exact for the parameters' direct paths, no multi-hop paths through spikes).  Memory: a few numbers per state element,
independent of how long the stream runs.  Total gradient per window: dL_w/dtheta (in window) + G . S_0 (carried).

Mechanisms and their parameters (only those whose state outlives the window; ALIF, muscles, writer adaptation are much
shorter than a window):
  comp      phi on the Kenyon -> MBON synapses and the Kenyon trace: comp_eta, comp_logit_forget, comp_logit_trace,
            and (through the phasic dopamine that writes phi) dopamine_logit_fast / slow
  dopamine  the fast and slow dopamine concentrations: dopamine_logit_fast, dopamine_logit_slow
  stp       release probability u and resources x per neuron: logit_u0, log_tau_fac, log_tau_rec (per superclass)
"""
import torch

COMP_PARAMS = ('comp_eta', 'comp_logit_forget', 'comp_logit_trace', 'dopamine_logit_fast', 'dopamine_logit_slow',
               'comp_logit_tonic')
DOPAMINE_PARAMS = ('dopamine_logit_fast', 'dopamine_logit_slow')
STP_PARAMS = ('logit_u0', 'log_tau_fac', 'log_tau_rec')
STATE_OF = {'comp': ('comp_trace', 'comp_memory'), 'dopamine': ('dopamine',), 'stp': ('u', 'x')}


def occupancy_slope(c, affinity):
    """d occ / dc for occ(c) = c / (c + K)."""
    return affinity / (c + affinity).square()


def dopamine_step(S, conc, slow, arrival, f, s):
    """S [3, N]: d conc / d lf, d slow / d lf, d slow / d ls for conc' = f conc + (1 - f) arrival, slow' = s slow +
    (1 - s) conc' (f, s = sigmoid of the logits).  conc, slow: the values before the tick."""
    conc_new = f * conc + (1.0 - f) * arrival
    dc_f = f * S[0] + f * (1.0 - f) * (conc - arrival)
    ds_f = s * S[1] + (1.0 - s) * dc_f
    ds_s = s * S[2] + s * (1.0 - s) * (slow - conc_new)
    return torch.cat([dc_f, ds_f, ds_s], 0)


def comp_step(S_phi, S_tr, phi, trace, spikes_kc, lam, eta, kappa, phasic, gate, edge_kc, dphasic, tonic=None,
              tonic_rate=None):
    """One tick of the compartment memory's sensitivities.
    trace' = lam trace + (1 - lam) s;  phi' = phi + w (eta p (1 + phi) - kappa phi),  w = gate * trace'[edge_kc]
    (gate = occ(slow dopamine at the MBON) / 0.5, detached in the model too).
    S_tr [1, n_kc]: d trace / d logit_trace.  S_phi [5, E] for COMP_PARAMS.  dphasic [2, E]: d p / d (lf, ls).
    phi, trace: the values before the tick; p, gate at this tick.
    tonic [1, E] (active forgetting, gate x its rate): phi' -= tonic phi, ungated by the trace; it enters keep (the
    carried sensitivities decay with it); tonic_rate sigma: its own row d phi' / d logit = -tonic (1 - sigma) phi."""
    S_tr_new = lam * S_tr + lam * (1.0 - lam) * (trace - spikes_kc)
    trace_new = lam * trace + (1.0 - lam) * spikes_kc
    w = gate * trace_new.index_select(1, edge_kc)
    dw_trace = gate * S_tr_new.index_select(1, edge_kc)
    A = eta * phasic * (1.0 + phi) - kappa * phi
    keep = 1.0 + w * (eta * phasic - kappa) - (tonic if tonic is not None else 0.0)
    direct = torch.cat([w * phasic * (1.0 + phi),                    # eta
                        -w * kappa * (1.0 - kappa) * phi,             # logit of the forgetting rate
                        dw_trace * A,                                 # logit of the trace's decay
                        w * eta * (1.0 + phi) * dphasic,              # the dopamine clocks, through p
                        (-tonic * (1.0 - tonic_rate) * phi) if tonic is not None else torch.zeros_like(phi)], 0)
    if S_phi.shape[0] != direct.shape[0]:            # sensitivities saved before the tonic row existed
        S_phi = torch.cat([S_phi, S_phi.new_zeros(direct.shape[0] - S_phi.shape[0], S_phi.shape[1])], 0)
    return keep * S_phi + direct, S_tr_new


def stp_step(S, u, x, spikes, u0, rho_f, rho_r, du0, drf, drr):
    """S [5, N]: du/da, du/dbf, dx/da, dx/dbf, dx/dbr for a = logit_u0, bf = log_tau_fac, br = log_tau_rec (each
    neuron's own class's parameter).  u_a = u + u0 (1 - u) s, x_p = x - u_a x s, u' = u0 + (u_a - u0) rho_f,
    x' = 1 + (x_p - 1) rho_r.  du0 = du0/da, drf = d rho_f / dbf, drr = d rho_r / dbr (0 where clamped)."""
    u_a = u + u0 * (1.0 - u) * spikes
    x_p = x - u_a * x * spikes
    zero = torch.zeros_like(u)
    out = []
    dua = []
    for k, (U, du0_k, drf_k) in enumerate(((S[0], du0, zero), (S[1], zero, drf))):
        d_ua = U * (1.0 - u0 * spikes) + du0_k * (1.0 - u) * spikes
        dua.append(d_ua)
        out.append(du0_k + (d_ua - du0_k) * rho_f + (u_a - u0) * drf_k)
    for k, (X, d_ua, drr_k) in enumerate(((S[2], dua[0], zero), (S[3], dua[1], zero), (S[4], zero, drr))):
        d_xp = X * (1.0 - u_a * spikes) - spikes * x * d_ua
        out.append(d_xp * rho_r + (x_p - 1.0) * drr_k)
    return torch.cat(out, 0)


def stp_parameter_slopes(model):
    """Per neuron: d u0 / d logit_u0, d rho_fac / d log_tau_fac, d rho_rec / d log_tau_rec (0 where clamped)."""
    u0 = torch.sigmoid(model.logit_u0)
    du0 = u0 * (1.0 - u0) * ((u0 > 0.05) & (u0 < 0.95)).float()
    tf, tr = torch.exp(model.log_tau_fac), torch.exp(model.log_tau_rec)
    drf = torch.exp(-1.0 / tf) / tf * ((tf > 5.0) & (tf < 1000.0)).float()
    drr = torch.exp(-1.0 / tr) / tr * ((tr > 5.0) & (tr < 2000.0)).float()
    sc = model.superclass_id.long()
    return du0[sc].unsqueeze(0), drf[sc].unsqueeze(0), drr[sc].unsqueeze(0)


def attach_carried_credit(model, mechanisms):
    """mechanisms: subset of ('comp', 'dopamine', 'stp'); each needs its state and parameters to exist."""
    mechanisms = tuple(mechanisms)
    for m in mechanisms:
        if m == 'comp' and getattr(model, 'comp_eta', None) is None:
            raise ValueError('carried credit for comp needs the compartment memory')
        if m in ('comp', 'dopamine') and getattr(model, 'dopamine_logit_slow', None) is None:
            raise ValueError('carried credit needs the dopamine state (--dopamine-modulation)')
        if m == 'stp' and not getattr(model, 'use_stp', False):
            raise ValueError('carried credit for stp needs STP')
    if 'comp' in mechanisms and 'dopamine' not in mechanisms:
        mechanisms = mechanisms + ('dopamine',)          # p depends on the dopamine clocks: carry their sensitivity
    model.carried_credit = mechanisms
    return mechanisms


def open_window(state, mechanisms):
    """The window's opening slow states become leaves (BPTT then gives dL/ds_0); returns (new state, leaves, the
    sensitivities at the window's start)."""
    import dataclasses
    model_fields = {}
    leaves = {}
    start = {name: getattr(state, name) for name in ('credit_comp', 'credit_comp_trace', 'credit_dopamine', 'credit_stp')}
    for mech in mechanisms:
        for name in STATE_OF[mech]:
            t = getattr(state, name, None)
            if t is not None and t.numel():
                leaf = t.detach().requires_grad_(True)
                model_fields[name] = leaf
                leaves[name] = leaf
    return dataclasses.replace(state, **model_fields), leaves, start


def apply_carried_credit(model, leaves, start, scale=1.0):
    """Adds G . S_0 to the parameters' gradients.  leaves: the window's opening states (with .grad after backward);
    start: the sensitivities at the window's start (credit_* fields of the opening state).  Returns the norms."""
    mech = getattr(model, 'carried_credit', ())
    added = {}

    def add(name, value):
        p = getattr(model, name, None)
        if p is None or not p.requires_grad:
            return
        value = value.reshape(p.shape).to(p.dtype) * scale
        if f'inwindow_{name}' not in added:          # the window's own BPTT credit, for the ratio
            added[f'inwindow_{name}'] = float(p.grad.norm()) if p.grad is not None else 0.0
        p.grad = value.clone() if p.grad is None else p.grad + value
        added[name] = added.get(name, 0.0) + float(value.norm())

    def grad(name):
        t = leaves.get(name)
        return None if t is None or t.grad is None else t.grad

    if 'comp' in mech and start.get('credit_comp') is not None and start['credit_comp'].numel():
        G_phi, G_tr = grad('comp_memory'), grad('comp_trace')
        S_phi, S_tr = start['credit_comp'], start['credit_comp_trace']
        if G_phi is not None:
            for k, name in enumerate(COMP_PARAMS[:S_phi.shape[0]]):     # old states carry fewer rows (no tonic)
                add(name, (G_phi[0] * S_phi[k]).sum())
        if G_tr is not None:
            add('comp_logit_trace', (G_tr[0] * S_tr[0]).sum())
    if 'dopamine' in mech and start.get('credit_dopamine') is not None and start['credit_dopamine'].numel():
        G = grad('dopamine')
        if G is not None:
            S = start['credit_dopamine']
            add('dopamine_logit_fast', (G[0] * S[0]).sum() + (G[1] * S[1]).sum())
            add('dopamine_logit_slow', (G[1] * S[2]).sum())
    if 'stp' in mech and start.get('credit_stp') is not None and start['credit_stp'].numel():
        Gu, Gx = grad('u'), grad('x')
        if Gu is not None and Gx is not None:
            S = start['credit_stp']
            sc = model.superclass_id.long()
            n = model.n_superclasses
            per = lambda v: torch.zeros(n, device=v.device, dtype=v.dtype).index_add_(0, sc, v[0] if v.dim() == 2 else v)
            add('logit_u0', per(Gu[0] * S[0] + Gx[0] * S[2]))
            add('log_tau_fac', per(Gu[0] * S[1] + Gx[0] * S[3]))
            add('log_tau_rec', per(Gx[0] * S[4]))
    return added
