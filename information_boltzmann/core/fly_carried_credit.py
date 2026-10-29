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
  DAN source  full local receptor/calcium/resource/delayed-packet conditional
              material history; six per-cell parameters and existing class gain
"""
import torch

COMP_PARAMS = ('comp_eta', 'comp_logit_forget', 'comp_logit_trace', 'dopamine_logit_fast', 'dopamine_logit_slow',
               'comp_logit_tonic', 'comp_eta_pot', 'comp_burst_k', 'comp_logit_camp')
# rows of dphasic (d p / d theta through the dopamine and the threshold) -> their COMP_PARAMS index
DPHASIC_ROWS = (3, 4, 7, 8)       # dopamine_logit_fast, dopamine_logit_slow, comp_burst_k, comp_logit_camp
DOPAMINE_PARAMS = ('dopamine_logit_fast', 'dopamine_logit_slow')
STP_PARAMS = ('logit_u0', 'log_tau_fac', 'log_tau_rec')
STATE_OF = {'comp': ('comp_trace', 'comp_memory', 'comp_da_trace', 'comp_camp', 'kc_contact_memory', 'kc_contact_trace', 'kc_contact_da_trace', 'kc_contact_camp'), 'dopamine': ('dopamine', 'kc_dopamine_signal'), 'stp': ('u', 'x')}


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


def split_comp_rows(S, n):
    """The carried rows in the current layout: (phi rows [n, E], D rows [n, E] or None).  Earlier layouts carried fewer
    parameters (5, 6 or 8 phi rows; with the potentiation path 2 x 8): each block keeps its order, new rows start at 0."""
    E = S.shape[1]
    pad = lambda B: torch.cat([B[:n], B.new_zeros(max(0, n - B.shape[0]), E)], 0)
    rows = S.shape[0]
    if rows == 2 * n:
        return S[:n], S[n:]
    if rows >= 16 and rows % 2 == 0 and rows // 2 < n:      # an older potentiation layout (2 x k, k < n)
        k = rows // 2
        return pad(S[:k]), pad(S[k:])
    return pad(S), None


def comp_step(S_phi, S_tr, phi, trace, spikes_kc, lam, eta, kappa, phasic, gate, edge_kc, dphasic, tonic=None,
              tonic_rate=None, unit=False, pot=None):
    """One tick of the compartment memory's sensitivities, rows in COMP_PARAMS order (P rows).
    trace' = lam trace + (1 - lam) s  (unit: lam trace + (1 - lam trace) s);  w = gate * trace'[edge_kc]
    phi' = phi + w (eta p (1 + phi) - kappa phi) + gate P D s_e (1 - phi) - tonic phi
    (gate = occ(slow dopamine at the MBON) / 0.5, detached in the model too; kappa = 0 when the use-dependent
    forgetting is off; the potentiation term only with pot).
    S_tr [1, n_kc]: d trace / d logit_trace.  dphasic [2 or 3, E]: d p / d (lf, ls[, burst_k]).
    phi, trace: the values before the tick; p, gate at this tick.
    tonic [1, E] (active forgetting, gate x its rate): its own row d phi' / d logit = -tonic (1 - sigma) phi.
    pot (the timing rule's potentiation, attach_comp_timing): dict with P = softplus(eta_pot), dP = sigmoid(eta_pot),
    D = the dopamine trace before the tick [1, E], s_edge = this tick's Kenyon spike per edge [1, E].  Then S_phi
    carries 2P rows: P for phi, P for D, with D' = lam D + (1 - lam D) min(p, 1) (unit) or lam D + (1 - lam) p.
    Returns (S_phi', S_tr')."""
    n = len(COMP_PARAMS)
    S_p0, S_D0 = split_comp_rows(S_phi, n)
    if unit:      # trace' = lam trace + (1 - lam trace) s  (attach_comp_unit_traces)
        S_tr_new = lam * (1.0 - spikes_kc) * S_tr + lam * (1.0 - lam) * trace * (1.0 - spikes_kc)
        trace_new = lam * trace + (1.0 - lam * trace) * spikes_kc
    else:
        S_tr_new = lam * S_tr + lam * (1.0 - lam) * (trace - spikes_kc)
        trace_new = lam * trace + (1.0 - lam) * spikes_kc
    w = gate * trace_new.index_select(1, edge_kc)
    dw_trace = gate * S_tr_new.index_select(1, edge_kc)
    A = eta * phasic * (1.0 + phi) - kappa * phi
    z = torch.zeros_like(phi)
    dp = [z] * n                                            # d p / d theta per COMP_PARAMS row
    for r, idx in enumerate(DPHASIC_ROWS[:dphasic.shape[0]]):
        dp[idx] = dphasic[r:r + 1]
    keep = 1.0 + w * (eta * phasic - kappa) - (tonic if tonic is not None else 0.0)
    direct = [z] * n
    for r, idx in enumerate(DPHASIC_ROWS[:dphasic.shape[0]]):
        direct[idx] = w * eta * (1.0 + phi) * dphasic[r:r + 1]
    direct[0] = w * phasic * (1.0 + phi)                    # eta
    direct[1] = -w * kappa * (1.0 - kappa) * phi            # logit of the use-dependent forgetting rate
    direct[2] = dw_trace * A                                # logit of the trace's decay
    direct[5] = (-tonic * (1.0 - tonic_rate) * phi) if tonic is not None else z
    direct[6] = z                                           # eta_pot (below)
    # Fill the final state row-by-row instead of concatenating multiple full
    # parameter-by-contact temporaries. These are independent conditional rows;
    # no state/parameter path is removed or shortened.
    output = S_phi.new_empty(n * (2 if pot is not None else 1), phi.shape[1])
    if pot is None:
        for row in range(n):
            output[row:row+1] = keep * S_p0[row:row+1] + direct[row]
        return output, S_tr_new
    P, dP, D, s_e = pot['P'], pot['dP'], pot['D'], pot['s_edge']
    S_p, S_D = S_p0, S_D0 if S_D0 is not None else torch.zeros_like(S_p0)
    keep = keep - gate * P * D * s_e
    direct[6] = gate * dP * D * s_e * (1.0 - phi)
    cross = gate * P * s_e * (1.0 - phi)                    # d phi' / d D
    for row in range(n):
        output[row:row+1] = keep * S_p[row:row+1] + cross * S_D[row:row+1] + direct[row]
    if unit:
        q_on = (phasic < 1.0).to(phasic.dtype)
        q = phasic.clamp(max=1.0)
        keep_D = lam * (1.0 - q)
        d_lam = lam * (1.0 - lam) * D * (1.0 - q)
        d_q = (1.0 - lam * D) * q_on
    else:
        keep_D = lam
        d_lam = lam * (1.0 - lam) * (D - phasic)
        d_q = (1.0 - lam) * torch.ones_like(phasic)
    for row in range(n):
        direct_d = (d_lam.expand_as(phi) if row == 2 else
                    d_q * dp[row] if row in DPHASIC_ROWS else z)
        output[n+row:n+row+1] = keep_D * S_D[row:row+1] + direct_d
    return output, S_tr_new


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
    if getattr(model, 'dan_carried_credit_enabled', False) and 'dopamine' not in mechanisms:
        raise ValueError('DAN carried material credit requires opening dopamine windows')
    return mechanisms


def open_window(state, mechanisms):
    """The window's opening slow states become leaves (BPTT then gives dL/ds_0); returns (new state, leaves, the
    sensitivities at the window's start)."""
    import dataclasses
    model_fields = {}
    leaves = {}
    start = {name: getattr(state, name, None) for name in ('credit_comp', 'credit_comp_trace', 'credit_dopamine', 'credit_stp',
                                                         'credit_comp_camp', 'kc_contact_credit', 'kc_contact_trace_credit',
                                                         'kc_contact_camp_credit', 'kc_dopamine_credit')}
    from .fly_dan_credit import DAN_CREDIT_FIELDS
    from .fly_dan_release import DAN_RELEASE_STATE_FIELDS
    start.update({name: getattr(state, name, None) for name in DAN_CREDIT_FIELDS})
    start['dan_credit_dopamine'] = getattr(state, 'dan_credit_dopamine', None)
    for mech in mechanisms:
        for name in STATE_OF[mech]:
            t = getattr(state, name, None)
            if t is not None and t.numel():
                leaf = t.detach().requires_grad_(True)
                model_fields[name] = leaf
                leaves[name] = leaf
    if 'dopamine' in mechanisms and any(start[n] is not None and start[n].numel() for n in DAN_CREDIT_FIELDS):
        for name in DAN_RELEASE_STATE_FIELDS:
            t = getattr(state, name)
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

    layouts = (
        ('comp_memory', 'comp_trace', 'comp_da_trace', 'comp_camp',
         'credit_comp', 'credit_comp_trace', 'credit_comp_camp'),
        ('kc_contact_memory', 'kc_contact_trace', 'kc_contact_da_trace', 'kc_contact_camp',
         'kc_contact_credit', 'kc_contact_trace_credit', 'kc_contact_camp_credit'))
    for phi_name, trace_name, da_name, camp_name, credit_name, trace_credit_name, camp_credit_name in layouts:
        if 'comp' not in mech or start.get(credit_name) is None or not start[credit_name].numel():
            continue
        G_phi, G_tr, G_D = grad(phi_name), grad(trace_name), grad(da_name)
        S_all, S_tr = start[credit_name], start[trace_credit_name]
        n = len(COMP_PARAMS)
        S_phi, S_D = split_comp_rows(S_all, n)
        if G_phi is not None:
            for k, name in enumerate(COMP_PARAMS):
                add(name, (G_phi[0] * S_phi[k]).sum())
        if G_D is not None and S_D is not None:
            for k, name in enumerate(COMP_PARAMS):                  # the potentiation's dopamine trace
                add(name, (G_D[0] * S_D[k]).sum())
        G_I, S_I = grad(camp_name), start.get(camp_credit_name)
        if G_I is not None and S_I is not None and S_I.numel():
            # the cAMP integrator of the burst judgement: rows dopamine_logit_fast, _slow, comp_logit_camp
            for k, name in enumerate(('dopamine_logit_fast', 'dopamine_logit_slow', 'comp_logit_camp')):
                add(name, (G_I[0] * S_I[k]).sum())
        if G_tr is not None:
            add('comp_logit_trace', (G_tr[0] * S_tr[0]).sum())
    if 'dopamine' in mech and start.get('credit_dopamine') is not None and start['credit_dopamine'].numel():
        G = grad('dopamine')
        if G is not None:
            S = start['credit_dopamine']
            add('dopamine_logit_fast', (G[0] * S[0]).sum() + (G[1] * S[1]).sum())
            add('dopamine_logit_slow', (G[1] * S[2]).sum())
    if 'dopamine' in mech and start.get('kc_dopamine_credit') is not None and start['kc_dopamine_credit'].numel():
        G = grad('kc_dopamine_signal')
        if G is not None:
            S = start['kc_dopamine_credit']
            add('dopamine_logit_fast', (G[0]*S[0]).sum()+(G[1]*S[1]).sum())
            add('dopamine_logit_slow', (G[1]*S[2]).sum())
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
    from .fly_dan_credit import carried_gradients
    for name, value in carried_gradients(model, {name: leaf.grad for name, leaf in leaves.items()}, start).items():
        add(name, value)
    return added
