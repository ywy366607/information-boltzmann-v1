"""Equation/interface checks only; no synthetic capability claims."""
import torch

from scripts.ib_local.bfss_gdn_reference import (
    coupling_potential_and_gradient, energy, event_step, quiet_step, rotation_charge,
    symmetric_lift,
)


def state():
    generator = torch.Generator().manual_seed(1110)
    return tuple(.2 * torch.randn(2, 3, 3, 2, generator=generator, dtype=torch.float64)
                 for _ in range(2))


def test_block_energy_and_force_match_symmetric_matrix_hamiltonian():
    s, _ = state()
    s.requires_grad_()
    x = symmetric_lift(s)
    reference = s.new_zeros(2)
    for i in range(3):
        for j in range(3):
            comm = x[:, i] @ x[:, j] - x[:, j] @ x[:, i]
            reference = reference + .7 / 4 * comm.square().sum((-2, -1))
    potential, gradient = coupling_potential_and_gradient(s, .7)
    expected, = torch.autograd.grad(reference.sum(), s)
    torch.testing.assert_close(potential, reference)
    torch.testing.assert_close(gradient, expected)


def test_zero_internal_time_restores_native_delta_write():
    s, p = state()
    k = torch.nn.functional.normalize(s[..., 0], dim=-1)
    v = s[..., 0, :]
    alpha = torch.full((2, 3), .9, dtype=s.dtype)
    beta = torch.full_like(alpha, .6)
    prior = alpha[..., None, None] * s
    e = v - (prior.transpose(-1, -2) @ k.unsqueeze(-1)).squeeze(-1)
    expected = prior + beta[..., None, None] * k.unsqueeze(-1) * e.unsqueeze(-2)
    actual, _ = event_step(s, p, k, v, alpha, beta, dt=0., frequency=.8, coupling=.7)
    assert torch.equal(actual, expected)


def test_quiet_step_reverses_without_external_input():
    s, p = state()
    sn, pn = quiet_step(s, p, dt=.03, frequency=.8, coupling=.7)
    sr, pr = quiet_step(sn, pn, dt=-.03, frequency=.8, coupling=.7)
    torch.testing.assert_close(sr, s, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(pr, p, atol=1e-14, rtol=1e-14)
    assert (sn - s).norm() > 0


def test_uncoupled_quiet_rotation_preserves_energy():
    s, p = state()
    e = energy(s, p, frequency=.8, coupling=0.)
    for _ in range(100):
        s, p = quiet_step(s, p, dt=.2, frequency=.8, coupling=0.)
    torch.testing.assert_close(energy(s, p, frequency=.8, coupling=0.), e,
                               atol=1e-12, rtol=1e-12)


def test_coupled_energy_error_decreases_with_step_size():
    s, p = state()
    e = energy(s, p, frequency=.8, coupling=.7)
    errors = []
    for step, count in ((.05, 100), (.025, 200)):
        sn, pn = s, p
        maximum = 0.
        for _ in range(count):
            sn, pn = quiet_step(sn, pn, dt=step, frequency=.8, coupling=.7)
            maximum = max(maximum, float(((energy(sn, pn, frequency=.8, coupling=.7)-e)/e).abs().max()))
        errors.append(maximum)
    assert errors[0] < .001
    assert errors[1] < .3 * errors[0]


def test_one_head_has_no_commutator_force():
    s, _ = state()
    potential, gradient = coupling_potential_and_gradient(s[:, :1], .7)
    assert torch.equal(potential, torch.zeros_like(potential))
    assert torch.equal(gradient, torch.zeros_like(gradient))


def test_cross_head_influence_and_differentiable_parameters():
    s, p = state()
    s.requires_grad_()
    dt, omega, coupling = [torch.tensor(v, dtype=s.dtype, requires_grad=True)
                            for v in (.03, .8, .7)]
    sn, pn = quiet_step(s, p, dt=dt, frequency=omega, coupling=coupling)
    gs, *parameters = torch.autograd.grad(sn[:, 0].square().sum(), (s, dt, omega, coupling))
    assert gs[:, 1:].norm() > 0
    assert all(torch.isfinite(g) and g.abs() > 0 for g in parameters)
    assert torch.isfinite(pn).all()


def test_internal_flow_preserves_matrix_rotation_charge():
    s, p = state()
    original = rotation_charge(s, p)
    for _ in range(20):
        s, p = quiet_step(s, p, dt=.03, frequency=.8, coupling=.7)
    for got, expected in zip(rotation_charge(s, p), original):
        torch.testing.assert_close(got, expected, atol=1e-13, rtol=1e-13)


def test_boundary_evidence_can_change_the_preserved_charge():
    s, p = state()
    original = rotation_charge(s, p)
    k = torch.nn.functional.normalize(s[..., 0], dim=-1)
    alpha = torch.full((2, 3), .9, dtype=s.dtype)
    beta = torch.full_like(alpha, .6)
    sn, pn = event_step(s, p, k, s[..., 0, :], alpha, beta,
                       dt=.03, frequency=.8, coupling=.7)
    assert sum((a-b).square().sum() for a, b in zip(rotation_charge(sn, pn), original)) > 0
