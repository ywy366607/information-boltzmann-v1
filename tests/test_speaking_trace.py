"""The continuous speaking trace (accumulate_speaking_trace) is the horizon form (accumulate_speaking_credit) with the
horizon taken to infinity: the same sum_d (R_d - b) grad log p_d, delivered word by word across windows."""
import types

import torch

from information_boltzmann.core.fly_active_inference import delayed_speaking_value
from information_boltzmann.core.fly_mother_world import accumulate_speaking_credit, accumulate_speaking_trace


def _model(seed=0):
    torch.manual_seed(seed)
    m = torch.nn.Module()
    m.ai_w = torch.nn.Parameter(torch.randn(2, 5) * 0.3)
    m.ai_b = torch.nn.Parameter(torch.tensor([0.0, -1.0]))
    m.ai_mu = None
    return m


def _windows():
    torch.manual_seed(1)
    # window 1: child decisions interleaved with mother replies; window 2: mother replies only (everything matures)
    w1 = {'intervals': [6, 5, 7, 4, 6, 8], 'speakers': ['mother', 'fly', 'mother', 'mother', 'fly', 'mother'],
          'sources': ['mother_reply', 'self_hearing', 'mother_reply', 'mother_call', 'self_hearing', 'mother_reply'],
          'decisions': [(1, 6, 5, 1), (2, 11, 7, 0), (4, 22, 6, 1), (5, 28, 8, 0)]}
    w1['g'] = torch.randn(sum(w1['intervals']), 5)
    w2 = {'intervals': [40] * 8, 'speakers': ['mother'] * 8, 'sources': ['mother_reply'] * 8, 'decisions': []}
    w2['g'] = torch.randn(320, 5)
    return [(w1, [0.3, 0.0, -0.2, 0.9, 0.0, 0.5]), (w2, [0.1, 0.4, -0.3, 0.2, 0.05, 0.0, 0.1, 0.2])]


def test_trace_equals_infinite_horizon():
    tau, b0 = 32.0, 0.07
    # horizon form, horizon long enough that nothing is cut (exp(-320/32) ~ 5e-5 of the tail is beyond window 2)
    m = _model()
    learner = types.SimpleNamespace(model=m, speak_epistemic_baseline=b0)
    for rec, gains in _windows():
        accumulate_speaking_credit(learner, rec, gains, horizon_ticks=100000, tau_ticks=tau)
    learner.speak_credit_clock += 100000
    accumulate_speaking_credit(learner, {'intervals': [1], 'speakers': ['mother'], 'sources': ['mother_call'],
                                         'decisions': [], 'g': torch.zeros(1, 5)}, [0.0],
                               horizon_ticks=100000, tau_ticks=tau)
    g_, decisions, adv = learner.pending_speak_credit
    J, _ = delayed_speaking_value(m, g_, decisions, adv, n_words=1)
    ref = torch.autograd.grad(J, [m.ai_w, m.ai_b])
    # trace form
    m2 = _model()
    learner2 = types.SimpleNamespace(model=m2, speak_trace={'baseline': b0})
    total = {'ai_w': torch.zeros_like(m2.ai_w), 'ai_b': torch.zeros_like(m2.ai_b)}
    for rec, gains in _windows():
        stats = accumulate_speaking_trace(learner2, rec, gains, tau)
        for k, v in learner2.pending_speak_credit['trace'].items():
            total[k] += v
    assert torch.allclose(total['ai_w'], ref[0], atol=1e-4), (total['ai_w'], ref[0])
    assert torch.allclose(total['ai_b'], ref[1], atol=1e-4), (total['ai_b'], ref[1])
    assert stats['credit_trace_norm'] >= 0.0


def test_trace_carries_across_windows_in_o1_state():
    m = _model()
    learner = types.SimpleNamespace(model=m)
    (w1, g1), (w2, g2) = _windows()
    accumulate_speaking_trace(learner, w1, g1, 32.0)
    E1 = {k: v.clone() for k, v in learner.speak_trace['E'].items()}
    assert set(E1) == {'ai_w', 'ai_b'} and all(v.abs().sum() > 0 for v in E1.values())
    accumulate_speaking_trace(learner, w2, g2, 32.0)
    # window 2 has no decisions: its credit is window 1's trace, decayed, times the replies' progress
    G = learner.pending_speak_credit['trace']['ai_b']
    assert G.abs().sum() > 0
    assert all(v.shape == p.shape for v, p in zip(learner.speak_trace['E'].values(), (m.ai_w, m.ai_b)))


def test_differential_trace_is_the_average_reward_return():
    """R_d = sum_words g e^{-delay/tau} - r_bar int_end^T e^{-(s - end)/tau} ds (T: the last event), by brute force."""
    import math
    from information_boltzmann.core.fly_active_inference import motor_logits
    tau, b0, r_bar = 32.0, 0.05, 0.004
    m = _model()
    learner = types.SimpleNamespace(model=m, speak_trace={'baseline': b0, 'r_bar': r_bar})
    total = {'ai_w': torch.zeros_like(m.ai_w), 'ai_b': torch.zeros_like(m.ai_b)}
    clock, decisions, words = 0, [], []
    for rec, gains in _windows():
        arrival = clock + torch.tensor(rec['intervals']).cumsum(0)
        for i, t0, n, d in rec['decisions']:
            decisions.append((clock + t0 + n, rec['g'][t0:t0 + n], d))
        words += [(int(t), g) for t, g, s in zip(arrival, gains, rec['sources']) if s == 'mother_reply']
        accumulate_speaking_trace(learner, rec, gains, tau, differential=True, rate_beta=1.0)
        for k, v in learner.pending_speak_credit['trace'].items():
            total[k] += v
        clock = int(arrival[-1])
    T = max([t for t, _ in words] + [e for e, _, _ in decisions])
    J = 0.0
    for end, g_, d in decisions:
        R = sum(g * math.exp(-(t - end) / tau) for t, g in words if t >= end)
        R -= r_bar * tau * (1.0 - math.exp(-(T - end) / tau))
        hz = torch.sigmoid(motor_logits(m, g_)[:, 1]).clamp(1e-6, 1 - 1e-6)
        lq = torch.log1p(-hz).sum()
        J = J + (R - b0) * (torch.log((-torch.expm1(lq)).clamp_min(1e-12)) if d else lq)
    ref = torch.autograd.grad(J, [m.ai_w, m.ai_b])
    assert torch.allclose(total['ai_w'], ref[0], atol=1e-4), (total['ai_w'], ref[0])
    assert torch.allclose(total['ai_b'], ref[1], atol=1e-4)
    # a stretch with no teaching (only calls) makes silence worth less than her average
    learner3 = types.SimpleNamespace(model=_model(), speak_trace={'r_bar': 0.01})
    dark = {'intervals': [8] * 6, 'speakers': ['fly', 'mother'] * 3, 'sources': ['self_hearing', 'mother_call'] * 3,
            'decisions': [(0, 0, 8, 0), (2, 16, 8, 0), (4, 32, 8, 0)], 'g': torch.randn(48, 5)}
    accumulate_speaking_trace(learner3, dark, [0.0] * 6, tau, differential=True)
    assert all(it['return'] < 0 for it in learner3.speak_trace['diag'][:-1])
