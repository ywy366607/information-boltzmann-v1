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
    full = torch.autograd.grad(loss(st), [getattr(model, n) for n in cc.COMP_PARAMS])
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
    for name, g in zip(cc.COMP_PARAMS, full):
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
    full = torch.autograd.grad(loss(st), [getattr(model, n) for n in cc.COMP_PARAMS])
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
    for name, g in zip(cc.COMP_PARAMS, full):
        assert torch.allclose(getattr(model, name).grad, g, rtol=1e-4, atol=1e-6), (name, getattr(model, name).grad, g)
    assert all(float(g.abs()) > 0 for g in full)
    assert sum(seen) > 10                    # the ramp's own slope was exercised
