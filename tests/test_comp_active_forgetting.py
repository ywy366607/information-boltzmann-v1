"""Active forgetting of the Kenyon -> MBON memory (attach_comp_tonic_forget) keeps the carried credit exact, and a
repeated pairing leaves a deeper mark than a rare one (familiarity), which the trace-gated forgetting alone did not."""
import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize('unit', [False, True])
def test_carried_credit_with_active_forgetting(unit):
    """With active forgetting (phi -= gate x rate x phi each tick, ungated by the trace) the carried credit still equals
    backprop through both windows.  Credit beyond the window (fly_carried_credit): with the spikes and the arriving dopamine fixed, the gradient of
    the second window's loss through BOTH windows equals the second window's own BPTT (opening states as leaves) plus
    G . S_0, S advanced by the forward recursions over the first window -- for the compartment memory with its
    dopamine clocks, and for STP with per-class parameters."""
    from information_boltzmann.core import fly_carried_credit as cc
    torch.manual_seed(3)
    N, E, nkc, K = 7, 6, 3, 0.4
    post = torch.tensor([4, 4, 5, 6, 5, 6]); edge_kc = torch.tensor([0, 1, 1, 2, 0, 2]); kc = torch.tensor([0, 1, 2])
    T1, T2 = 23, 17
    spikes = (torch.rand(T1 + T2, N) < 0.3).float()
    arrival = torch.rand(T1 + T2, N) * 0.5
    occ = lambda c: c / (c + K)
    model = torch.nn.Module()
    for name, v in (('comp_eta', -0.4), ('comp_logit_forget', -2.0), ('comp_logit_trace', 2.5),
                    ('dopamine_logit_fast', 1.5), ('dopamine_logit_slow', 3.0), ('comp_logit_tonic', -3.0)):
        setattr(model, name, torch.nn.Parameter(torch.tensor(v)))
    model.carried_credit = ('comp', 'dopamine')

    def tick(st, t):
        D, tr, phi = st
        f, s = torch.sigmoid(model.dopamine_logit_fast), torch.sigmoid(model.dopamine_logit_slow)
        conc = f * D[0:1] + (1 - f) * arrival[t:t + 1]
        D = torch.cat([conc, s * D[1:2] + (1 - s) * conc], 0)
        lam = torch.sigmoid(model.comp_logit_trace)
        tr = (lam * tr + (1 - lam * tr) * spikes[t:t + 1, kc]) if unit else (lam * tr + (1 - lam) * spikes[t:t + 1, kc])
        dp = D[:, post]
        p = F.relu(occ(dp[0:1]) - occ(dp[1:2]))
        w = occ(dp[1:2].detach()) / 0.5 * tr[:, edge_kc]
        kap = torch.sigmoid(model.comp_logit_forget)
        phi = phi + w * (model.comp_eta * p * (1 + phi) - kap * phi)             - occ(dp[1:2].detach()) / 0.5 * torch.sigmoid(model.comp_logit_tonic) * phi      # live phi, as the model
        return D, tr, phi

    def sens(st, S, t):
        D, tr, phi = (x.detach() for x in st)
        Sd, Sphi, Str = S
        f, s = torch.sigmoid(model.dopamine_logit_fast).detach(), torch.sigmoid(model.dopamine_logit_slow).detach()
        Sd = cc.dopamine_step(Sd, D[0:1], D[1:2], arrival[t:t + 1], f, s)
        new = tick(st, t)
        Dn = new[0].detach(); dp = Dn[:, post]
        p = F.relu(occ(dp[0:1]) - occ(dp[1:2]))
        Sdp = Sd[:, post]
        on = (p > 0).float()
        dphasic = on * torch.cat([cc.occupancy_slope(dp[0:1], K) * Sdp[0:1] - cc.occupancy_slope(dp[1:2], K) * Sdp[1:2],
                                  -cc.occupancy_slope(dp[1:2], K) * Sdp[2:3]], 0)
        Sphi, Str = cc.comp_step(Sphi, Str, phi, tr, spikes[t:t + 1, kc], torch.sigmoid(model.comp_logit_trace).detach(),
                                 model.comp_eta.detach(), torch.sigmoid(model.comp_logit_forget).detach(), p,
                                 occ(dp[1:2]) / 0.5, edge_kc, dphasic, tonic=occ(dp[1:2]) / 0.5 * torch.sigmoid(model.comp_logit_tonic).detach(),
                                 tonic_rate=float(torch.sigmoid(model.comp_logit_tonic)), unit=unit)
        return tuple(x.detach() for x in new), (Sd, Sphi, Str)

    start = (torch.rand(2, N) * 0.3, torch.rand(1, nkc) * 0.1, torch.zeros(1, E))
    wD, wtr, wphi = torch.randn(2, N), torch.randn(1, nkc), torch.randn(1, E)
    loss = lambda st: (st[0] * wD).sum() + (st[1] * wtr).sum() + (st[2] * wphi).sum() + (st[2] ** 2).sum()
    # (a) through both windows
    st = start
    for t in range(T1 + T2):
        st = tick(st, t)
    names = [n for n in cc.COMP_PARAMS if hasattr(model, n)]
    full = torch.autograd.grad(loss(st), [getattr(model, n) for n in names])
    # (b) window 1 forward-only with the recursions, then window 2 BPTT from leaves, plus G . S_0
    st, S = start, (torch.zeros(3, N), torch.zeros(len(cc.COMP_PARAMS), E), torch.zeros(1, nkc))
    with torch.no_grad():
        for t in range(T1):
            st, S = sens(st, S, t)
    leaves = {'dopamine': st[0].clone().requires_grad_(), 'comp_trace': st[1].clone().requires_grad_(),
              'comp_memory': st[2].clone().requires_grad_()}
    s2 = (leaves['dopamine'], leaves['comp_trace'], leaves['comp_memory'])
    for t in range(T1, T1 + T2):
        s2 = tick(s2, t)
    model.zero_grad(set_to_none=True)
    loss(s2).backward()
    cc.apply_carried_credit(model, leaves, {'credit_dopamine': S[0], 'credit_comp': S[1], 'credit_comp_trace': S[2]})
    for name, g in zip(names, full):
        assert torch.allclose(getattr(model, name).grad, g, rtol=1e-4, atol=1e-6), (name, getattr(model, name).grad, g)
    assert all(float(g.abs()) > 0 for g in full)


def test_repetition_leaves_a_deeper_mark_only_with_active_forgetting():
    """Two Kenyon edges paired with the same burst: one every 20 ticks, one every 200 ticks.  Trace-gated forgetting
    alone drives both toward the same equilibrium; with active forgetting the frequent one is deeper."""
    eta, kap, lam = -2.0, 0.0355, 0.963

    def run(kt, period, T=60000):
        tr, phi = 0.0, 0.0
        for t in range(T):
            s = 1.0 if t % period == 0 else 0.0
            tr = lam * tr + (1 - lam) * s
            p = 0.05 if t % period < 4 else 0.0
            w = 1.0 * tr
            phi = phi + w * (eta * p * (1 + phi) - kap * phi) - kt * phi
        return phi
    gated = run(0.0, 20), run(0.0, 200)
    active = run(1 / 2048, 20), run(1 / 2048, 200)
    # (toy: 0.42 vs 0.28 gated, 0.36 vs 0.09 active; in Mengli after 50k words every fired edge sat at -0.10)
    assert gated[0] / gated[1] < 1.6                               # frequency barely coded, and less as it settles
    assert active[0] < active[1] < 0 and active[0] / active[1] > 3   # the frequent edge is clearly deeper


@pytest.mark.parametrize('unit', [False, True])
def test_carried_credit_with_relative_burst_threshold(unit):
    """The burst judged on the linear contrast (f - s) / (s + K) with a ramp over one sd above mean + k sd
    (attach_comp_relative_burst): the carried credit (the model's dphasic: ramp d(raw) + relu(raw) d(ramp)) equals
    backprop through both windows.  Running statistics are detached in both."""
    from information_boltzmann.core import fly_carried_credit as cc
    torch.manual_seed(5)
    N, E, nkc, K = 7, 6, 3, 0.4
    post = torch.tensor([4, 4, 5, 6, 5, 6]); edge_kc = torch.tensor([0, 1, 1, 2, 0, 2]); kc = torch.tensor([0, 1, 2])
    T1, T2 = 23, 17
    spikes = (torch.rand(T1 + T2, N) < 0.3).float()
    arrival = torch.rand(T1 + T2, N) * 0.5 * (torch.rand(T1 + T2, N) < 0.4).float()
    occ = lambda c: c / (c + K)
    kb = 0.3
    model = torch.nn.Module()
    for name, v in (('comp_eta', -0.4), ('comp_logit_forget', -2.0), ('comp_logit_trace', 2.5),
                    ('dopamine_logit_fast', 1.5), ('dopamine_logit_slow', 3.0), ('comp_logit_tonic', -3.0)):
        setattr(model, name, torch.nn.Parameter(torch.tensor(v)))
    model.carried_credit = ('comp', 'dopamine')
    seen = []

    def tick(st, t):
        D, tr, phi, stats = st
        f, s = torch.sigmoid(model.dopamine_logit_fast), torch.sigmoid(model.dopamine_logit_slow)
        conc = f * D[0:1] + (1 - f) * arrival[t:t + 1]
        D = torch.cat([conc, s * D[1:2] + (1 - s) * conc], 0)
        lam = torch.sigmoid(model.comp_logit_trace)
        sk = spikes[t:t + 1, kc]
        tr = (lam * tr + (1 - lam * tr) * sk) if unit else (lam * tr + (1 - lam) * sk)
        dp = D[:, post]
        raw = occ(dp[0:1]) - occ(dp[1:2])
        rel = (dp[0:1] - dp[1:2]) / (dp[1:2] + K)
        with torch.no_grad():
            stats = torch.cat([0.8 * stats[0:1] + 0.2 * rel, 0.8 * stats[1:2] + 0.2 * rel ** 2], 0)
            sd = (stats[1:2] - stats[0:1] ** 2).clamp_min(1e-8).sqrt()
        ramp = ((rel - stats[0:1] - kb * sd) / sd).clamp(0.0, 1.0)
        seen.append(int(((ramp > 0) & (ramp < 1) & (raw > 0)).sum()))
        p = F.relu(raw) * ramp
        w = occ(dp[1:2].detach()) / 0.5 * tr[:, edge_kc]
        kap = torch.sigmoid(model.comp_logit_forget)
        phi = phi + w * (model.comp_eta * p * (1 + phi) - kap * phi) \
            - occ(dp[1:2].detach()) / 0.5 * torch.sigmoid(model.comp_logit_tonic) * phi
        return D, tr, phi, stats, (raw, ramp, sd, p)

    def sens(st, S, t):
        D, tr, phi, stats = (x.detach() for x in st)
        Sd, Sphi, Str = S
        f, s = torch.sigmoid(model.dopamine_logit_fast).detach(), torch.sigmoid(model.dopamine_logit_slow).detach()
        Sd = cc.dopamine_step(Sd, D[0:1], D[1:2], arrival[t:t + 1], f, s)
        *new, (raw, ramp, sd, p) = tick(st, t)
        new = tuple(x.detach() for x in new)
        dp = new[0][:, post]
        Sdp = Sd[:, post]
        a_f, a_s = cc.occupancy_slope(dp[0:1], K), cc.occupancy_slope(dp[1:2], K)
        draw = torch.cat([a_f * Sdp[0:1] - a_s * Sdp[1:2], -a_s * Sdp[2:3]], 0)
        r_ = ramp.detach(); raw = raw.detach()
        den = dp[1:2] + K
        b_s = (dp[0:1] + K) / den.square()
        drel = torch.cat([Sdp[0:1] / den - b_s * Sdp[1:2], -b_s * Sdp[2:3]], 0)
        in_ramp = ((r_ > 0) & (r_ < 1)).float() / sd
        dphasic = (raw > 0).float() * r_ * draw + F.relu(raw) * in_ramp * drel
        Sphi, Str = cc.comp_step(Sphi, Str, phi, tr, spikes[t:t + 1, kc], torch.sigmoid(model.comp_logit_trace).detach(),
                                 model.comp_eta.detach(), torch.sigmoid(model.comp_logit_forget).detach(), p.detach(),
                                 occ(dp[1:2]) / 0.5, edge_kc, dphasic,
                                 tonic=occ(dp[1:2]) / 0.5 * torch.sigmoid(model.comp_logit_tonic).detach(),
                                 tonic_rate=float(torch.sigmoid(model.comp_logit_tonic)), unit=unit)
        return new, (Sd, Sphi, Str)

    start = (torch.rand(2, N) * 0.3, torch.rand(1, nkc) * 0.1, torch.zeros(1, E), torch.cat([torch.zeros(1, E), torch.full((1, E), 0.5)], 0))
    wD, wtr, wphi = torch.randn(2, N), torch.randn(1, nkc), torch.randn(1, E)
    loss = lambda st: (st[0] * wD).sum() + (st[1] * wtr).sum() + (st[2] * wphi).sum() + (st[2] ** 2).sum()
    st = start
    for t in range(T1 + T2):
        st = tick(st, t)[:4]
    names = [n for n in cc.COMP_PARAMS if hasattr(model, n)]
    full = torch.autograd.grad(loss(st), [getattr(model, n) for n in names])
    st, S = start, (torch.zeros(3, N), torch.zeros(len(cc.COMP_PARAMS), E), torch.zeros(1, nkc))
    with torch.no_grad():
        for t in range(T1):
            st, S = sens(st, S, t)
    leaves = {'dopamine': st[0].clone().requires_grad_(), 'comp_trace': st[1].clone().requires_grad_(),
              'comp_memory': st[2].clone().requires_grad_()}
    s2 = (leaves['dopamine'], leaves['comp_trace'], leaves['comp_memory'], st[3])
    for t in range(T1, T1 + T2):
        s2 = tick(s2, t)[:4]
    model.zero_grad(set_to_none=True)
    loss(s2).backward()
    cc.apply_carried_credit(model, leaves, {'credit_dopamine': S[0], 'credit_comp': S[1], 'credit_comp_trace': S[2]})
    for name, g in zip(names, full):
        assert torch.allclose(getattr(model, name).grad, g, rtol=1e-4, atol=1e-6), (name, getattr(model, name).grad, g)
    assert all(float(g.abs()) > 0 for g in full)
    assert sum(seen) > 10                    # the ramp's own slope was exercised


@pytest.mark.parametrize('unit,rel,camp', [(False, False, False), (True, False, False), (False, True, False),
                                           (True, True, False), (False, True, True), (True, True, True)])
def test_carried_credit_newborn_compartment(unit, rel, camp):
    """The newborn's full compartment rule (third review 2026-10-15): timing rule with the potentiation path (its own
    dopamine trace D, eta_pot), use-dependent forgetting off (kappa 0 although its parameter exists), adapted receptors
    K_j = max(slow, K), active forgetting, burst threshold k (absolute or on the contrast), unit or rate traces.
    Carried credit for every COMP parameter equals backprop through both windows."""
    from information_boltzmann.core import fly_carried_credit as cc
    torch.manual_seed(7 + 2 * unit + rel + 4 * camp)
    N, E, nkc, K = 7, 6, 3, 0.05
    post = torch.tensor([4, 4, 5, 6, 5, 6]); edge_kc = torch.tensor([0, 1, 1, 2, 0, 2]); kc = torch.tensor([0, 1, 2])
    T1, T2 = 29, 19
    spikes = (torch.rand(T1 + T2, N) < 0.3).float()
    arrival = torch.rand(T1 + T2, N) * 0.6 * (torch.rand(T1 + T2, N) < 0.5).float()
    occ = lambda c, k: c / (c + k)
    sp = torch.nn.functional.softplus
    model = torch.nn.Module()
    for name, v in (('comp_eta', -0.6), ('comp_logit_forget', -2.0), ('comp_logit_trace', 2.0),
                    ('dopamine_logit_fast', 1.2), ('dopamine_logit_slow', 2.5), ('comp_logit_tonic', -3.0),
                    ('comp_eta_pot', 0.3), ('comp_burst_k', -1.0)) + ((('comp_logit_camp', 1.5),) if camp else ()):
        setattr(model, name, torch.nn.Parameter(torch.tensor(v)))
    model.carried_credit = ('comp', 'dopamine')
    seen = {'ramp': 0, 'pot': 0}

    def tick(st, t):
        Dc, tr, phi, Dt, stats, I = st
        f, s = torch.sigmoid(model.dopamine_logit_fast), torch.sigmoid(model.dopamine_logit_slow)
        conc = f * Dc[0:1] + (1 - f) * arrival[t:t + 1]
        Dc = torch.cat([conc, s * Dc[1:2] + (1 - s) * conc], 0)
        lam = torch.sigmoid(model.comp_logit_trace)
        sk = spikes[t:t + 1, kc]
        tr = (lam * tr + (1 - lam * tr) * sk) if unit else (lam * tr + (1 - lam) * sk)
        dp = Dc[:, post]
        Kj = torch.maximum(dp[1:2], torch.tensor(K))
        occ_s = occ(dp[1:2], Kj)
        raw = occ(dp[0:1], Kj) - occ_s
        kb = sp(model.comp_burst_k)
        I_before = I
        if rel:
            rel_v = (dp[0:1] - dp[1:2]) / (dp[1:2] + K)
            if camp:
                li = torch.sigmoid(model.comp_logit_camp)
                I = li * I + (1 - li) * rel_v
                sig = I
            else:
                sig = rel_v
        else:
            rel_v = None
            sig = raw
        with torch.no_grad():
            stats = torch.cat([0.8 * stats[0:1] + 0.2 * sig, 0.8 * stats[1:2] + 0.2 * sig ** 2], 0)
            sd = (stats[1:2] - stats[0:1] ** 2).clamp_min(1e-8).sqrt()
        if rel:
            ramp = ((sig - stats[0:1] - kb * sd) / sd).clamp(0.0, 1.0)
            p = F.relu(raw) * ramp
            seen['ramp'] += int(((ramp > 0) & (ramp < 1) & (raw > 0)).sum())
        else:
            ramp = None
            p = F.relu(raw - stats[0:1] - kb * sd)
        g = occ_s.detach() / 0.5
        w = g * tr[:, edge_kc]
        phi_prev = phi
        phi = phi + w * (model.comp_eta * p * (1 + phi) - 0.0 * phi)
        s_e = sk[:, edge_kc]
        phi = phi + g * sp(model.comp_eta_pot) * Dt * s_e * (1 - phi_prev)
        seen['pot'] += int((Dt * s_e > 0).sum())
        Dt = (lam * Dt + (1 - lam * Dt) * p.clamp(max=1.0)) if unit else (lam * Dt + (1 - lam) * p)
        phi = phi - g * torch.sigmoid(model.comp_logit_tonic) * phi_prev
        return (Dc, tr, phi, Dt, stats, I), (raw, ramp, sd, p, Kj, g, rel_v, I_before)

    def sens(st, S, t):
        Dc, tr, phi, Dt, stats, I = (x.detach() for x in st)
        Sd, Sphi, Str, SI = S
        f, s = torch.sigmoid(model.dopamine_logit_fast).detach(), torch.sigmoid(model.dopamine_logit_slow).detach()
        Sd = cc.dopamine_step(Sd, Dc[0:1], Dc[1:2], arrival[t:t + 1], f, s)
        new, (raw, ramp, sd, p, Kj, g, rel_v, I_before) = tick(st, t)
        new = tuple(x.detach() for x in new)
        dp = new[0][:, post]
        Sdp = Sd[:, post]
        raw, p = raw.detach(), p.detach()
        on = (p > 0).float()
        a_f = cc.occupancy_slope(dp[0:1], Kj)
        adapted = (dp[1:2] >= K).float()
        a_s = (1 - adapted) * cc.occupancy_slope(dp[1:2], K) + adapted * dp[0:1] / (dp[0:1] + Kj).square()
        draw = torch.cat([a_f * Sdp[0:1] - a_s * Sdp[1:2], -a_s * Sdp[2:3]], 0)
        sk_ = torch.sigmoid(model.comp_burst_k).detach()
        if rel:
            r_ = ramp.detach()
            den = dp[1:2] + K
            b_s = (dp[0:1] + K) / den.square()
            drel = torch.cat([Sdp[0:1] / den - b_s * Sdp[1:2], -b_s * Sdp[2:3]], 0)
            in_ramp = ((r_ > 0) & (r_ < 1)).float() / sd
            if camp:
                li = torch.sigmoid(model.comp_logit_camp).detach()
                SI = torch.cat([li * SI[0:2] + (1 - li) * drel,
                                li * SI[2:3] + li * (1 - li) * (I_before.detach() - rel_v.detach())], 0)
                dsig = SI[0:2]
            else:
                dsig = drel
            dph = (raw > 0).float() * r_ * draw + F.relu(raw) * in_ramp * dsig
            dk = -F.relu(raw) * in_ramp * sd * sk_
            extra = [F.relu(raw) * in_ramp * SI[2:3]] if camp else []
        else:
            dph = on * draw
            dk = -on * sk_ * sd
            extra = []
        dph = torch.cat([dph, dk] + extra, 0)
        pot = {'P': sp(model.comp_eta_pot).detach(), 'dP': torch.sigmoid(model.comp_eta_pot).detach(), 'D': Dt,
               's_edge': spikes[t:t + 1, kc][:, edge_kc]}
        Sphi, Str = cc.comp_step(Sphi, Str, phi, tr, spikes[t:t + 1, kc], torch.sigmoid(model.comp_logit_trace).detach(),
                                 model.comp_eta.detach(), torch.tensor(0.0), p, g, edge_kc, dph,
                                 tonic=g * torch.sigmoid(model.comp_logit_tonic).detach(),
                                 tonic_rate=float(torch.sigmoid(model.comp_logit_tonic)), unit=unit, pot=pot)
        return new, (Sd, Sphi, Str, SI)

    start = (torch.rand(2, N) * 0.3, torch.rand(1, nkc) * 0.1, torch.zeros(1, E), torch.zeros(1, E),
             torch.cat([torch.zeros(1, E), torch.full((1, E), 0.05)], 0), torch.zeros(1, E))
    wD, wtr, wphi, wDt = torch.randn(2, N), torch.randn(1, nkc), torch.randn(1, E), torch.randn(1, E)
    loss = lambda st: (st[0] * wD).sum() + (st[1] * wtr).sum() + (st[2] * wphi).sum() + (st[2] ** 2).sum() + (st[3] * wDt).sum()
    names = [n for n in cc.COMP_PARAMS if hasattr(model, n)]
    params = [getattr(model, n) for n in names]
    st = start
    for t in range(T1 + T2):
        st = tick(st, t)[0]
    full = torch.autograd.grad(loss(st), params, allow_unused=True)
    full = [torch.zeros(()) if g is None else g for g in full]
    st, S = start, (torch.zeros(3, N), torch.zeros(len(cc.COMP_PARAMS), E), torch.zeros(1, nkc), torch.zeros(3, E))
    with torch.no_grad():
        for t in range(T1):
            st, S = sens(st, S, t)
    leaves = {'dopamine': st[0].clone().requires_grad_(), 'comp_trace': st[1].clone().requires_grad_(),
              'comp_memory': st[2].clone().requires_grad_(), 'comp_da_trace': st[3].clone().requires_grad_()}
    if camp:
        leaves['comp_camp'] = st[5].clone().requires_grad_()
    s2 = (leaves['dopamine'], leaves['comp_trace'], leaves['comp_memory'], leaves['comp_da_trace'], st[4],
          leaves['comp_camp'] if camp else st[5])
    for t in range(T1, T1 + T2):
        s2 = tick(s2, t)[0]
    model.zero_grad(set_to_none=True)
    loss(s2).backward()
    cc.apply_carried_credit(model, leaves, {'credit_dopamine': S[0], 'credit_comp': S[1], 'credit_comp_trace': S[2],
                                            'credit_comp_camp': S[3] if camp else None})
    for name, g in zip(names, full):
        got = getattr(model, name).grad
        got = torch.zeros(()) if got is None else got
        assert torch.allclose(got, g, rtol=1e-4, atol=1e-6), (name, got, g)
    assert float(full[1].abs()) == 0.0                       # the use-dependent forgetting is off: no credit
    assert all(float(full[i].abs()) > 0 for i in range(len(names)) if i != 1)
    assert seen['pot'] > 5 and (not rel or seen['ramp'] > 5)
