"""Numerical identities and persistence contracts, without task training."""
import copy
import math

import pytest
import torch
from torch.distributions import Normal, kl_divergence

from information_boltzmann.core.plastic_medium import PlasticMedium3D
from information_boltzmann.core.structural_posterior import StructuralPosterior


def posterior(k=3, *, dtype=torch.float64, **overrides):
    arguments = dict(resource_density=2., speed_reference=.7, structure_time=3.,
                     prior_std=.4, initial_std=.2, maintenance_supply=1.3,
                     initial_dual=.6, initial_mean=torch.zeros(k, 3, dtype=dtype))
    arguments.update(overrides)
    return StructuralPosterior(k, **arguments)


def snapshot(module):
    return {name: value.clone() for name, value in module.state_dict().items()}


def assert_state_equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0, equal_nan=True)


def test_reuses_material_basis_without_registering_material_or_grid():
    medium = PlasticMedium3D((8, 8, 8), channels=2, material_width=2, hidden=4,
                             anisotropic_transport=True).double()
    basis = medium.material.basis(medium.coordinates)
    assert basis.shape[-1] == 256
    # The canonical basis supplies 256 independent fields on the 512-site grid;
    # this is a spatial subspace, not 512 independent allocation logits.
    assert torch.linalg.matrix_rank(basis.flatten(0, -2)) == 256
    q = posterior(basis.shape[-1])
    assert dict(q.named_parameters()).keys() == {'mean', 'log_std'}
    assert sum(p.numel() for p in q.parameters()) == 1536
    assert not any('basis' in name or 'material' in name for name in q.state_dict())
    before = snapshot(medium)
    allocation = q.allocation(basis)
    assert allocation.shape == (*medium.shape, 4)
    assert_state_equal(before, snapshot(medium))


def test_simplex_competition_idle_and_volume_weighted_maintenance():
    q = posterior(1)
    coarse = torch.ones(4, 4, 4, 1, dtype=torch.float64)
    fine = torch.ones(8, 8, 8, 1, dtype=torch.float64)
    allocation = q.allocation(coarse)
    torch.testing.assert_close(allocation.sum(-1), torch.full((4, 4, 4), 2., dtype=torch.float64))
    assert (allocation > 0).all()
    torch.testing.assert_close(q.maintenance(allocation, 1 / 4**3), torch.tensor(1.5, dtype=torch.float64))
    torch.testing.assert_close(q.maintenance(q.allocation(fine), 1 / 8**3), q.maintenance(allocation, 1 / 4**3))
    volumes = torch.linspace(.001, .005, 4**3, dtype=torch.float64).reshape(4, 4, 4)
    torch.testing.assert_close(q.maintenance(allocation, volumes), 1.5 * volumes.sum())
    idle = torch.zeros_like(allocation)
    idle[..., 3] = 2
    assert q.maintenance(idle, 1 / 4**3) == 0
    elevated = q.allocation(coarse, torch.tensor([[2., 0., 0.]], dtype=torch.float64))
    assert (elevated[..., 0] > allocation[..., 0]).all()
    assert (elevated[..., 1:] < allocation[..., 1:]).all()


def test_explicit_sample_reconstructs_graph_and_forward_does_not_advance_state():
    q = posterior(2)
    noise = torch.tensor([[.3, -.7, 1.1], [-.4, .5, .8]], dtype=torch.float64)
    q.begin_window(noise)
    before = snapshot(q)
    first = q.sample_coefficients()
    second = q.sample_coefficients()
    assert first is not second
    torch.testing.assert_close(first, q.mean + q.log_std.exp() * noise, rtol=0, atol=0)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    basis = torch.tensor([[1., .4], [1., -.2]], dtype=torch.float64)
    for _ in range(2):
        allocation = q.allocation(basis, q.sample_coefficients())
        loss = q.objective(allocation[0, 0], q.maintenance(allocation, .5), 11)
        gradients = torch.autograd.grad(loss, (q.mean, q.log_std))
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert all(gradient.abs().sum() > 0 for gradient in gradients)
    assert_state_equal(before, snapshot(q))


def test_full_gaussian_kl_matches_distribution_and_analytic_gradients():
    q = posterior(2)
    with torch.no_grad():
        q.mean.copy_(torch.tensor([[.2, -.1, .8], [-.3, .5, .7]], dtype=torch.float64))
        q.log_std.copy_(torch.tensor([[.1, -.8, .3], [-1.2, -.4, .6]], dtype=torch.float64))
        q.prior_mean.copy_(torch.tensor([[-.1, .2, .1], [.3, -.7, .4]], dtype=torch.float64))
        q.prior_variance.copy_(torch.tensor([[.7, .3, .9], [1.3, .4, .8]], dtype=torch.float64))
    q.begin_window(torch.zeros_like(q.mean))
    actual = q.kl_divergence()
    expected = kl_divergence(Normal(q.mean, q.log_std.exp()),
                             Normal(q.window_prior_mean, q.window_prior_variance.sqrt())).sum()
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    d_mean, d_log_std = torch.autograd.grad(actual, (q.mean, q.log_std))
    torch.testing.assert_close(d_mean, (q.mean - q.window_prior_mean) / q.window_prior_variance)
    torch.testing.assert_close(d_log_std, (2 * q.log_std).exp() / q.window_prior_variance - 1)
    with torch.no_grad():
        q.mean.copy_(q.window_prior_mean)
        q.log_std.copy_(.5 * q.window_prior_variance.log())
    assert abs(q.kl_divergence()) < 1e-25


def test_chunk_weighting_charges_one_window_kl_and_maintenance_with_matching_gradients():
    q = posterior(1)
    q.begin_window(torch.ones_like(q.mean))
    basis = torch.ones(2, 1, dtype=torch.float64)
    allocation = q.allocation(basis, q.sample_coefficients())
    cost = q.maintenance(allocation, .5)
    ce_a, ce_b = allocation[0, 0].square(), allocation[1, 1].square()
    expected = q.objective((3 * ce_a + 5 * ce_b) / 8, cost, 8)
    actual = (3 / 8) * q.objective(ce_a, cost, 8) + (5 / 8) * q.objective(ce_b, cost, 8)
    torch.testing.assert_close(actual, expected)
    expected_grads = torch.autograd.grad(expected, tuple(q.parameters()), retain_graph=True)
    actual_grads = torch.autograd.grad(actual, tuple(q.parameters()))
    for left, right in zip(expected_grads, actual_grads):
        torch.testing.assert_close(left, right)
    assert q.evidence_events == 0 and q.window_event_count == 0


def test_ou_commit_inherits_post_optimizer_posterior_at_actual_physical_interval():
    q = posterior(2, stationary_mean=torch.full((2, 3), .1, dtype=torch.float64))
    q.begin_window(torch.full_like(q.mean, .7))
    old_prior = q.prior_mean.clone()
    optimizer = torch.optim.SGD(q.parameters(), lr=.04)
    q.sample_coefficients().square().sum().backward()
    q.record_window_evidence(7, .16, torch.tensor(1.7, dtype=torch.float64))
    torch.testing.assert_close(q.prior_mean, old_prior, rtol=0, atol=0)
    optimizer.step()
    updated_mean = q.mean.detach().clone()
    updated_variance = (2 * q.log_std.detach()).exp()
    rho = math.exp(-.16 / 3)
    q.commit_window(dual_learning_rate=.2)
    torch.testing.assert_close(q.prior_mean, .1 + rho * (updated_mean - .1))
    torch.testing.assert_close(q.prior_variance, rho**2 * updated_variance + (1 - rho**2) * .4**2)
    torch.testing.assert_close(q.dual, torch.tensor(.68, dtype=torch.float64))
    assert q.windows_committed == 1 and q.evidence_events == 7
    assert q.elapsed == .16 and not q.window_active and not q.window_evidence_recorded


@pytest.mark.parametrize('duration', [0., 1e-12, 1e6])
def test_ou_stationary_fixed_point_and_zero_long_duration_limits(duration):
    stationary = torch.full((1, 3), .3, dtype=torch.float64)
    q = posterior(1, initial_mean=stationary, stationary_mean=stationary, initial_std=.4)
    q.begin_window(torch.zeros_like(q.mean))
    q.record_window_evidence(1, duration, torch.tensor(0.))
    q.commit_window(dual_learning_rate=2.)
    torch.testing.assert_close(q.prior_mean, stationary)
    torch.testing.assert_close(q.prior_variance, torch.full_like(stationary, .4**2))
    assert q.dual == 0


def test_pending_window_state_optimizer_and_gradient_resume_are_exact():
    q = posterior(2)
    optimizer = torch.optim.Adam(q.parameters(), lr=.01)
    noise = torch.tensor([[.1, -.3, .8], [.7, -.4, .5]], dtype=torch.float64)
    q.begin_window(noise)
    basis = torch.tensor([[1., .5], [1., -.7]], dtype=torch.float64)
    allocation = q.allocation(basis, q.sample_coefficients())
    q.objective(allocation.square().sum(), q.maintenance(allocation, .5), 5).backward()
    q.record_window_evidence(5, .23, q.maintenance(allocation.detach(), .5))
    saved_module = snapshot(q)
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_grads = [parameter.grad.clone() for parameter in q.parameters()]
    restored = posterior(2)
    restored.load_state_dict(saved_module, strict=True)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=.01)
    restored_optimizer.load_state_dict(saved_optimizer)
    for parameter, gradient in zip(restored.parameters(), saved_grads):
        parameter.grad = gradient.clone()
    torch.testing.assert_close(q.sample_coefficients(), restored.sample_coefficients(), rtol=0, atol=0)
    for item in (q, restored):
        sampled = item.sample_coefficients()
        gradients = torch.autograd.grad(sampled.square().sum() + item.kl_divergence(), tuple(item.parameters()))
        if item is q:
            reference_grads = gradients
        else:
            for expected, actual in zip(reference_grads, gradients):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    optimizer.step()
    restored_optimizer.step()
    q.commit_window(dual_learning_rate=.05)
    restored.commit_window(dual_learning_rate=.05)
    assert_state_equal(snapshot(q), snapshot(restored))


def test_window_ledger_rejects_duplicate_or_unordered_actions():
    q = posterior(1)
    with pytest.raises(RuntimeError):
        q.sample_coefficients()
    with pytest.raises(RuntimeError):
        q.commit_window()
    q.begin_window(torch.zeros_like(q.mean))
    with pytest.raises(RuntimeError):
        q.begin_window(torch.ones_like(q.mean))
    with pytest.raises(RuntimeError):
        q.commit_window()
    q.record_window_evidence(3, .2, torch.tensor(1.))
    with pytest.raises(RuntimeError):
        q.record_window_evidence(3, .2, torch.tensor(1.))
    q.commit_window()
    with pytest.raises(RuntimeError):
        q.commit_window()
    assert q.evidence_events == 3 and q.windows_committed == 1


def test_unit_directions_enforce_capacity_budget_even_under_autocast():
    q = posterior(1, dtype=torch.float32)
    basis = torch.ones(2, 1)
    directions = torch.eye(3).expand(2, 3, 3)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        allocation = q.allocation(basis)
        factor = q.slow_factor(basis, directions)
        assert allocation.dtype == torch.float32 and factor.dtype == torch.float32
        torch.testing.assert_close(factor.norm(dim=-1), q.speed_reference * allocation[..., :3])
        with pytest.raises(RuntimeError, match='unit row norms'):
            q.slow_factor(basis, 1.5 * directions)
    with pytest.raises(RuntimeError, match='unit row norms'):
        q.slow_factor(basis, 1.5 * directions.to(torch.bfloat16))


def test_structural_factor_passive_transport_has_posterior_credit():
    torch.manual_seed(81)
    medium = PlasticMedium3D((2, 2, 2), channels=2, material_width=2, hidden=4,
                             anisotropic_transport=True).double()
    basis = medium.material.basis(medium.coordinates)
    q = posterior(basis.shape[-1])
    q.begin_window(torch.full_like(q.mean, .2))
    shear = torch.tensor([[1., 0., 0.], [.3, 1., 0.], [-.1, .2, 1.]], dtype=torch.float64)
    directions = (shear / shear.norm(dim=-1, keepdim=True)).expand(*medium.shape, 3, 3)
    factor = q.slow_factor(basis, directions, q.sample_coefficients())
    gram = factor @ factor.transpose(-1, -2)
    assert (torch.linalg.eigvalsh(gram) > 0).all()
    state = medium.initial_state(1)
    state = state.with_field(torch.randn_like(state.field))
    evolved = medium._tensor_transport(state, medium._duration(.03, state.field), factor[None])
    energy = lambda current: current.field.square().sum() + sum(item.square().sum() for item in current.flux)
    torch.testing.assert_close(energy(evolved), energy(state), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(evolved.field.sum((1, 2, 3)), state.field.sum((1, 2, 3)), rtol=1e-12, atol=1e-12)
    gradients = torch.autograd.grad(evolved.field[0, 0, 0, 0, 0], tuple(q.parameters()))
    assert all(torch.isfinite(item).all() and item.abs().max() > 1e-9 for item in gradients)
    epsilon = 1e-5
    values = []
    with torch.no_grad():
        original = q.mean[0, 0].clone()
        for shift in (epsilon, -epsilon):
            q.mean[0, 0].copy_(original + shift)
            shifted_factor = q.slow_factor(basis, directions, q.sample_coefficients())
            result = medium._tensor_transport(state, medium._duration(.03, state.field), shifted_factor[None])
            values.append(result.field[0, 0, 0, 0, 0])
        q.mean[0, 0].copy_(original)
    torch.testing.assert_close(gradients[0][0, 0], (values[0] - values[1]) / (2 * epsilon),
                               rtol=1e-5, atol=1e-10)


@pytest.mark.parametrize('failure', ['mean', 'dual'])
def test_invalid_commit_is_transactional(failure):
    q = posterior(1, dtype=torch.float32)
    q.begin_window(torch.zeros_like(q.mean))
    q.record_window_evidence(3, .1, torch.tensor(1.7))
    if failure == 'mean':
        with torch.no_grad():
            q.mean[0, 0] = float('nan')
        learning_rate = None
    else:
        learning_rate = 1e40
    before = snapshot(q)
    with pytest.raises(FloatingPointError):
        q.commit_window(dual_learning_rate=learning_rate)
    assert_state_equal(before, snapshot(q))
    assert q.window_active and q.window_evidence_recorded


def test_invalid_configuration_noise_volume_and_coefficient_precision_are_rejected():
    for arguments in ({'initial_std': 1e-30}, {'prior_std': 1e-30},
                      {'structure_time': 0.}, {'maintenance_supply': -1.}):
        with pytest.raises(ValueError):
            posterior(1, dtype=torch.float32, **arguments)
    with pytest.raises(ValueError):
        posterior(1, dtype=torch.bfloat16)
    q = posterior(1)
    for noise in (torch.ones(2, 3, dtype=torch.float64),
                  torch.ones_like(q.mean, dtype=torch.float32),
                  torch.full_like(q.mean, float('nan'))):
        with pytest.raises(ValueError):
            q.begin_window(noise)
    with pytest.raises(RuntimeError):
        q.maintenance(torch.ones(2, 4, dtype=torch.float64), 0.)
    with pytest.raises(ValueError):
        q.allocation(torch.ones(2, 2, dtype=torch.float64))
    with pytest.raises(ValueError):
        q.half().allocation(torch.ones(2, 1))


def test_aot_eager_fullgraph_rebuilds_pending_sample_and_gradients():
    q = posterior(2)
    q.begin_window(torch.full_like(q.mean, .2))
    basis = torch.tensor([[1., .2], [1., -.4]], dtype=torch.float64)

    def evaluate():
        allocation = q.allocation(basis, q.sample_coefficients())
        return q.objective(allocation.square().sum(), q.maintenance(allocation, .5), 5)

    expected = evaluate()
    expected_grads = torch.autograd.grad(expected, tuple(q.parameters()))
    actual = torch.compile(evaluate, backend='aot_eager', fullgraph=True)()
    actual_grads = torch.autograd.grad(actual, tuple(q.parameters()))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for left, right in zip(expected_grads, actual_grads):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
