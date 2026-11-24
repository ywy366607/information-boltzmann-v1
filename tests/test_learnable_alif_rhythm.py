"""Unit tests for Isomorphic Wood-Sub-Pentad Learnable ALIF Rhythm."""
from __future__ import annotations

from types import SimpleNamespace
import torch

from scripts.ib_local.alif_executive import DEFAULT_PARAMS, validate_params
from scripts.ib_local.learnable_alif_rhythm import LearnableAlifRhythm


def test_safety_envelope_holds_under_extreme_adversarial_phi():
    """Lean Theorems 1 & 2: zero starvation and refractory anti-storming for ALL phi."""
    mod = LearnableAlifRhythm()
    torch.manual_seed(0)
    for scale in (0.0, 1.0, 10.0, 50.0, -50.0):
        with torch.no_grad():
            mod.phi.copy_(torch.randn_like(mod.phi) * scale)
        p = mod.decoded_floats()
        # Must pass strict Rheobase + Refractory validation unconditionally!
        validate_params({**DEFAULT_PARAMS, **p})
        assert p['delta0'] > (1.0 - p['lam_v']) * p['theta0']
        assert p['i_max'] < p['theta0'] + p['beta']


def test_forward_sensitivity_matches_unrolled_autograd_exactly():
    """Lean Theorem 3: O(1) forward sensitivity `e_V, e_a` matches full BPTT autograd."""
    mod = LearnableAlifRhythm(tau=0.25)
    inputs = [
        (0.1, 0.0, 0.002, 0.05),
        (0.3, 0.2, 0.004, 0.10),
        (0.0, 0.1, 0.001, 0.00),
        (0.2, 0.15, 0.003, 0.08),
    ]
    v, a = 0.0, 1.0
    for zs, zb, pk, ze in inputs:
        out = mod.step_with_trace(
            v_before=v, a_before=a, z_sens=zs, z_burst=zb, p_rel_delta=pk, z_epi=ze,
            deterministic_eval=True,
        )
        assert not out['spiked']
        v, a = out['v_after'], out['a_after']

    # Reference full unrolled autograd through the 4 steps
    phi_ref = mod.phi.detach().clone().requires_grad_(True)
    d = mod.decode(phi_ref)
    vr = torch.tensor(0.0)
    ar = torch.tensor(1.0)
    for zs, zb, pk, ze in inputs:
        raw = d['delta0'] + d['w_sens'] * zs + d['w_burst'] * zb + d['w_kin'] * pk + d['w_epi'] * ze
        drv = torch.clamp(raw, min=d['delta0'], max=d['i_max'])
        vr = d['lam_v'] * vr + drv
        ar = d['lam_a'] * ar
    ev_ref = torch.autograd.grad(vr, phi_ref, retain_graph=True)[0]
    ea_ref = torch.autograd.grad(ar, phi_ref)[0]

    assert torch.allclose(mod.e_v, ev_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(mod.e_a, ea_ref, atol=1e-5, rtol=1e-5)


def test_three_factor_credit_raises_threshold_on_wasteful_spikes_and_lowers_on_useful():
    """Lean Theorem 4: negative net utility suppresses wasteful spikes; positive reinforces."""
    dummy = SimpleNamespace(optimizer=None, alif_executive_state=dict(DEFAULT_PARAMS))

    # Case A: Repeated wasteful wakeups (silence + negative metal_g) -> firing probability should decrease.
    mod_waste = LearnableAlifRhythm(tau=0.30, credit_scale=4.0)
    p0_theta = mod_waste.decoded_floats()['theta0']
    for _ in range(8):
        v, a = 0.0, 0.0
        for step_i in range(6):
            out = mod_waste.step_with_trace(
                v_before=v, a_before=a, z_sens=0.4, z_burst=0.8, p_rel_delta=0.005, z_epi=0.1,
                deterministic_eval=True,
            )
            v, a = out['v_after'], out['a_after']
            if out['spiked']:
                mod_waste.settle_executive_credit(
                    dummy, metal_g=-0.05, choice='silence', preceding_calm_pieces=step_i + 1,
                )
                break
    # After repeated negative advantages on spikes, theta0 must increase (more conservative waking)
    assert mod_waste.decoded_floats()['theta0'] > p0_theta

    # Case B: Repeated high-value wakeups (active generate + strong positive metal_g) -> theta0 decreases.
    mod_good = LearnableAlifRhythm(tau=0.30, credit_scale=4.0)
    # Seed baseline lower so positive spikes get positive advantage immediately
    mod_good.utility_ema.fill_(-0.02)
    mod_good.utility_count.fill_(5)
    p0_good = mod_good.decoded_floats()['theta0']
    for _ in range(8):
        v, a = 0.0, 0.0
        for step_i in range(6):
            out = mod_good.step_with_trace(
                v_before=v, a_before=a, z_sens=0.4, z_burst=0.8, p_rel_delta=0.005, z_epi=0.1,
                deterministic_eval=True,
            )
            v, a = out['v_after'], out['a_after']
            if out['spiked']:
                mod_good.settle_executive_credit(
                    dummy, metal_g=+0.08, choice='generate', preceding_calm_pieces=step_i + 1,
                )
                break
    assert mod_good.decoded_floats()['theta0'] < p0_good
