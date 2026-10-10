"""Numerical, causal and continuation contracts; no capability experiment."""
import copy
import math

import numpy as np
import pytest
import torch

from information_boltzmann.core.capacity_growth import (
    allocation_log_prob, simplex_pullback_metric)
from information_boltzmann.core.plastic_medium import PlasticMedium3D
from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.core.structural_posterior import StructuralPosterior
from information_boltzmann.runtime.active_medium_training import ActiveMediumTrainer
from information_boltzmann.runtime.optimization import make_medium_optimizer
from information_boltzmann.runtime.optimization import initialize_capacity_growth_branch


OPTIONS = dict(resource_density=2., speed_reference=1., structure_time=3.,
               prior_std=.4, initial_std=.4, maintenance_supply=1.4, initial_dual=.1)
GROWTH = dict(step_size=1., max_capacity_kl=.001)


def posterior(count=1, growth=True):
    return StructuralPosterior(count, **OPTIONS,
                               capacity_growth=GROWTH if growth else None).double()


def record(q, count=4):
    q.begin_window(torch.zeros_like(q.mean))
    q.record_window_evidence(count, .02, q.mean.new_tensor(1.5))


def test_geometry_matches_log_probability_hessian_and_singular_chart_is_regularized():
    phi = torch.tensor([[1., .3], [1., -.8], [1., .1]], dtype=torch.float64)
    mean = torch.tensor([[.2, -.1, .6], [.3, .9, -.2]], dtype=torch.float64)
    shares = allocation_log_prob(phi, mean).exp().detach()
    volumes = torch.tensor([.2, .3, .5], dtype=torch.float64)
    hessian = torch.autograd.functional.hessian(
        lambda z: -(volumes[:, None] * shares * allocation_log_prob(phi, z)).sum(), mean)
    actual = simplex_pullback_metric(phi, shares, volumes)
    torch.testing.assert_close(actual, hessian.reshape(6, 6), atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(actual, actual.T)
    assert torch.linalg.eigvalsh(actual).min() > 0
    q = posterior(2)
    record(q)
    q.mean.grad = torch.tensor([[1., -2., .3], [.1, .4, -.8]], dtype=torch.float64)
    proposal = q.capacity_growth.propose(q, torch.ones(3, 2, dtype=torch.float64), 1 / 3)
    assert proposal['gradient_dot_update'] < 0
    assert proposal['solve_residual'] < 1e-12
    assert proposal['max_site_kl'] <= GROWTH['max_capacity_kl']


def test_signed_value_grows_favored_share_with_finite_budget_and_frozen_variance():
    q = posterior()
    record(q)
    phi = torch.ones(8, 1, dtype=torch.float64)
    before = q.allocation(phi).clone()
    variance = q.log_std.clone()
    q.mean.grad = torch.tensor([[-1., 0., 0.]], dtype=torch.float64)
    proposal = q.capacity_growth.propose(q, phi, 1 / 8)
    q.capacity_growth.apply(q, proposal)
    after = q.allocation(phi)
    assert (after[:, 0] > before[:, 0]).all()
    assert (after[:, 1:] < before[:, 1:]).all()
    torch.testing.assert_close(after.sum(-1), before.sum(-1))
    torch.testing.assert_close(q.log_std, variance, atol=0, rtol=0)
    assert torch.isfinite(after).all() and (after > 0).all()
    with pytest.raises(RuntimeError, match='ownership'):
        q.capacity_growth.apply(q, proposal)


def test_grid_refinement_and_off_grid_queries_preserve_continuous_chart():
    q = posterior(3)
    record(q)
    with torch.no_grad():
        q.mean.copy_(torch.tensor([[.2, -.1, .4], [.01, .02, -.01], [-.02, .01, .01]]))
    q.mean.grad = torch.ones_like(q.mean) * .1
    def basis(n):
        x = torch.arange(n, dtype=torch.float64) / n
        return torch.stack((torch.ones_like(x), (2 * math.pi * x).cos(),
                            (2 * math.pi * x).sin()), -1)
    coarse = q.capacity_growth.propose(q, basis(32), 1 / 32)
    fine = q.capacity_growth.propose(q, basis(64), 1 / 64)
    torch.testing.assert_close(coarse['mean'], fine['mean'], atol=1e-13, rtol=1e-13)
    q.capacity_growth.apply(q, fine)
    x = torch.tensor([.013, .732, 1.013], dtype=torch.float64)
    phi = torch.stack((torch.ones_like(x), (2 * math.pi * x).cos(),
                       (2 * math.pi * x).sin()), -1)
    allocation = q.allocation(phi)
    torch.testing.assert_close(allocation[0], allocation[2], atol=1e-14, rtol=1e-14)
    assert allocation.shape == (3, 4)


def make_learner(growth=True, checkpoint=False):
    torch.manual_seed(105)
    options = {**OPTIONS, **({'capacity_growth': GROWTH} if growth else {})}
    model = PlasticMediumPorts3D(vocab_size=11, shape=(2, 2, 2), channels=4,
                                hidden=4, material_width=2, anisotropic_transport=True,
                                bath_type='conductance', activity_adaptation=True,
                                short_term_plasticity=True, structure_options=options).double()
    optimizer = make_medium_optimizer(model, lr=.0002)
    return ActiveMediumTrainer(model, optimizer, model.initial_belief(), carry_token=1,
                               event_duration=.005, chunk_tokens=2, tokens_per_update=4,
                               activation_checkpointing=checkpoint)


def test_capacity_keeps_task_vjp_but_has_one_update_owner_and_cleared_credit():
    learner = make_learner()
    q = learner.model.medium.structural_posterior
    assert q.mean.requires_grad and not q.log_std.requires_grad
    names = dict(learner.model.learning_named_parameters())
    assert names['medium.structural_posterior.mean'] is q.mean
    assert not any(p is q.mean or p is q.log_std
                   for g in learner.optimizer.param_groups for p in g['params'])
    original = q.mean.clone()
    learner.consume(torch.tensor([2, 3]), phase='train_first_pass', prior_nll=np.ones(11))
    assert q.mean.grad is not None and q.mean.grad.norm() > 0
    torch.testing.assert_close(original, q.mean, atol=0, rtol=0)
    learner.consume(torch.tensor([4, 5]), phase='fresh_B', prior_nll=np.ones(11))
    assert q.capacity_growth.updates == 1 and q.windows_committed == 1
    assert q.mean.grad is None and q.log_std.grad is None
    assert not torch.equal(original, q.mean)
    assert learner.summary()['capacity_growth']['last_gradient_dot_update'] < 0
    learner.consume(torch.tensor([6, 7, 8, 9]), phase='train_first_pass', prior_nll=np.ones(11))
    assert q.capacity_growth.updates == 2 and q.windows_committed == 2 and q.evidence_events == 8
    assert q.mean.grad is None


def test_pending_window_checkpoint_and_recomputation_match_uninterrupted_learning():
    a, checkpointed = make_learner(), make_learner(checkpoint=True)
    prior = np.ones(11)
    initial_rng = torch.get_rng_state()
    a.consume(torch.tensor([2, 3]), phase='train_first_pass', prior_nll=prior)
    after_first_rng = torch.get_rng_state()
    torch.set_rng_state(initial_rng)
    checkpointed.consume(torch.tensor([2, 3]), phase='train_first_pass', prior_nll=prior)
    assert torch.equal(after_first_rng, torch.get_rng_state())
    qa = a.model.medium.structural_posterior
    qc = checkpointed.model.medium.structural_posterior
    torch.testing.assert_close(qa.mean.grad, qc.mean.grad, atol=1e-12, rtol=1e-12)
    checkpoint_rng = torch.get_rng_state()
    restored = make_learner()
    restored.model.load_state_dict(copy.deepcopy(a.model.state_dict()))
    restored.optimizer.load_state_dict(copy.deepcopy(a.optimizer.state_dict()))
    restored.belief = copy.deepcopy(a.belief)
    restored.load_state_dict(copy.deepcopy(a.state_dict()))
    torch.set_rng_state(checkpoint_rng)
    left = a.consume(torch.tensor([4, 5, 6, 7]), phase='fresh_B', prior_nll=prior)
    uninterrupted_rng = torch.get_rng_state()
    torch.set_rng_state(checkpoint_rng)
    right = restored.consume(torch.tensor([4, 5, 6, 7]), phase='fresh_B', prior_nll=prior)
    assert left == right and a.summary() == restored.summary()
    assert torch.equal(uninterrupted_rng, torch.get_rng_state())
    for x, y in zip(a.model.state_dict().values(), restored.model.state_dict().values()):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_new_capacity_preserves_wave_energy_and_physical_state_values():
    medium = PlasticMedium3D((2, 2, 2), 2, material_width=2, hidden=4,
                             anisotropic_transport=True,
                             structure_options={**OPTIONS, 'capacity_growth': GROWTH}).double()
    q = medium.structural_posterior
    record(q)
    state = medium.initial_state(1).with_field(torch.randn(1, 2, 2, 2, 2, dtype=torch.float64))
    original = state.field.clone()
    q.mean.grad = torch.ones_like(q.mean) * .1
    proposal = q.capacity_growth.propose(q, medium.material.basis(medium.coordinates), 1 / 8)
    q.capacity_growth.apply(q, proposal)
    torch.testing.assert_close(state.field, original, atol=0, rtol=0)
    factor = medium.prepare_evolution().structural_factor[None]
    evolved = medium._tensor_transport(state, medium._duration(.02, state.field), factor)
    def energy(s):
        return s.field.square().sum() + sum(v.square().sum() for v in s.flux)
    torch.testing.assert_close(energy(evolved), energy(state), atol=1e-12, rtol=1e-12)


def test_bad_credit_rejected_without_parameter_or_ledger_mutation():
    q = posterior()
    record(q)
    before = copy.deepcopy(q.state_dict())
    q.mean.grad = torch.full_like(q.mean, float('nan'))
    with pytest.raises(FloatingPointError):
        q.capacity_growth.propose(q, torch.ones(2, 1, dtype=torch.float64), .5)
    for name, old in before.items():
        torch.testing.assert_close(q.state_dict()[name], old, atol=0, rtol=0)


def test_prospective_ou_validation_rejects_before_any_optimizer_mutation():
    learner = make_learner()
    q = learner.model.medium.structural_posterior
    learner.model.structure_dual_learning_rate = float('nan')
    before = [p.detach().clone() for p in learner.model.parameters()]
    with pytest.raises(ValueError, match='dual learning rate'):
        learner.consume(torch.tensor([2, 3, 4, 5]), phase='train_first_pass', prior_nll=np.ones(11))
    assert learner.optimizer_updates == 0 and q.capacity_growth.updates == 0
    assert not learner.optimizer.state and q.windows_committed == 0
    for old, parameter in zip(before, learner.model.parameters()):
        torch.testing.assert_close(old, parameter, atol=0, rtol=0)


def test_completed_rule_migration_preserves_weights_and_all_other_optimizer_moments():
    source = make_learner(growth=False)
    source.consume(torch.tensor([2, 3, 4, 5]), phase='train_first_pass', prior_nll=np.ones(11))
    weights = copy.deepcopy(source.model.state_dict())
    old = copy.deepcopy(source.optimizer.state_dict())
    destination = make_learner()
    new = initialize_capacity_growth_branch(destination.model, weights, old, source.state_dict())
    excluded = {'medium.structural_posterior.mean', 'medium.structural_posterior.log_std'}
    removed = {index for group in old['param_groups']
               for name, index in zip(group['param_names'], group['params']) if name in excluded}
    assert set(new['state']) == set(old['state']) - removed
    for index, state in new['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, old['state'][index][name], atol=0, rtol=0)
    for name, value in weights.items():
        torch.testing.assert_close(destination.model.state_dict()[name], value, atol=0, rtol=0)
    destination.optimizer = make_medium_optimizer(destination.model, lr=.0002, saved_state=new)
    destination.belief = copy.deepcopy(source.belief)
    destination.load_state_dict(source.state_dict())
    destination.model.medium.structural_posterior.mean.grad = None
    destination.model.medium.structural_posterior.log_std.grad = None
    destination.consume(torch.tensor([6, 7, 8, 9]), phase='fresh_B', prior_nll=np.ones(11))
    assert destination.model.medium.structural_posterior.capacity_growth.updates == 1
    source.consume(torch.tensor([6]), phase='fresh_B', prior_nll=np.ones(11))
    with pytest.raises(ValueError, match='completed'):
        initialize_capacity_growth_branch(make_learner().model, source.model.state_dict(),
                                          source.optimizer.state_dict(), source.state_dict())
    with pytest.raises(ValueError, match='normal exact resume'):
        initialize_capacity_growth_branch(make_learner().model, destination.model.state_dict(),
                                          destination.optimizer.state_dict(), destination.state_dict())


def test_float32_guard_checks_the_reconstructed_physical_sample_and_quantized_noop():
    q = posterior().float()
    with torch.no_grad():
        q.mean.copy_(torch.tensor([[.3, -.2, .1]]))
    q.begin_window(torch.tensor([[.37, -.83, .22]]))
    q.record_window_evidence(4, .02, q.mean.new_tensor(1.5))
    phi = torch.tensor([[1.], [1.]], dtype=torch.float32)
    old = q.allocation(phi, q.sample_coefficients()).detach().double()
    old = old / old.sum(-1, keepdim=True)
    old_mean = q.allocation(phi, q.mean).detach().double()
    old_mean = old_mean / old_mean.sum(-1, keepdim=True)
    q.mean.grad = torch.tensor([[-10., 12., 3.]])
    proposal = q.capacity_growth.propose(q, phi, .5)
    q.capacity_growth.apply(q, proposal)
    new = q.allocation(phi, q.sample_coefficients()).detach().double()
    new = new / new.sum(-1, keepdim=True)
    actual = (new * (new.log() - old.log())).sum(-1).max()
    new_mean = q.allocation(phi, q.mean).detach().double()
    new_mean = new_mean / new_mean.sum(-1, keepdim=True)
    actual_mean = (new_mean * (new_mean.log() - old_mean.log())).sum(-1).max()
    assert actual <= GROWTH['max_capacity_kl']
    assert actual_mean <= GROWTH['max_capacity_kl']
    assert abs(float(torch.maximum(actual, actual_mean)) - proposal['max_site_kl']) < 1e-14
    q.mean.grad = torch.ones_like(q.mean) * 1e-30
    no_op = q.capacity_growth.propose(q, phi, .5)
    assert no_op['update_norm'] == 0 and no_op['step_fraction'] == 0
    q.capacity_growth.apply(q, no_op)
    assert q.capacity_growth.zero_updates == 1
