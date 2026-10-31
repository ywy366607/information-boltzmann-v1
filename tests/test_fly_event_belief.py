"""Probability/interface tests only; no synthetic capability experiment or training."""
import copy
import io
import math

import numpy as np
import pytest
import torch

from information_boltzmann.core.fly_event_belief import (
    FORMAT, NormalInverseGammaMarks, ProbitEventBelief,
    affine_candidate_moments, probit_next_event,
)

def test_pending_non_gaussian_survival_cannot_masquerade_as_exact_prediction():
    state = {}
    model = ProbitEventBelief(state, feature_dim=1)
    x = torch.ones(1, dtype=torch.float64)
    before = model.predict(x, 3)['probabilities'].clone()
    model.begin(x)
    torch.testing.assert_close(model.predict(x, 3)['probabilities'], before)
    model.advance(4)
    with pytest.raises(ValueError, match='pending survival'):
        model.predict(x, 3)
    model.resolve(False)
    after = model.predict(x, 3)['probabilities']
    assert not torch.equal(before, after)

def assert_tree_equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    else:
        assert a == b


@pytest.mark.parametrize('horizon', [0, 1, 8, 32])
def test_next_event_normalizes_including_no_event(horizon):
    result = probit_next_event(torch.tensor([-50., -2., 0., 3., 50.]),
                              torch.tensor([.001, .2, 1., 3., .001]), horizon)
    assert result['probabilities'].shape == (5, horizon + 1)
    torch.testing.assert_close(result['probabilities'].sum(-1), torch.ones(5, dtype=torch.float64),
                               rtol=1e-12, atol=1e-12)
    assert torch.isfinite(result['timing_information']).all()
    assert (result['timing_information'] >= 0).all()


def test_hazard_mixture_matches_uniform_hazard_analytic_mass_and_information():
    # For Z~N(0,1), Phi(Z) is uniform: exact geometric-mixture integrals.
    horizon = 8
    result = probit_next_event(0., 1., horizon, quadrature_points=96)
    masses = [1 / (t * (t + 1)) for t in range(1, horizon + 1)] + [1 / (horizon + 1)]
    torch.testing.assert_close(result['probabilities'], torch.tensor(masses, dtype=torch.float64),
                               rtol=1e-9, atol=1e-10)
    marginal_entropy = -sum(p * math.log(p) for p in masses)
    conditional_entropy = horizon / (horizon + 1) ** 2
    for t in range(1, horizon + 1):
        harmonic = sum(1 / j for j in range(1, t + 2))
        conditional_entropy += ((t - 1) * (1 / t + 1 / (t + 1)) + harmonic - 1) / (t * (t + 1))
    assert float(result['timing_information']) == pytest.approx(
        marginal_entropy - conditional_entropy, abs=1e-9)
    # Marginalization occurs after accumulating conditional survival.
    assert float(result['probabilities'][-1]) != pytest.approx(.5 ** horizon)


def test_expected_information_matches_independent_legendre_integration():
    mean, variance, horizon = -.7, 1.3, 5
    nodes, weights = np.polynomial.legendre.leggauss(512)
    z = torch.as_tensor(nodes * 12., dtype=torch.float64)
    density = torch.exp(-.5 * (z - mean).square() / variance) / math.sqrt(2 * math.pi * variance)
    weights = torch.as_tensor(weights * 12., dtype=torch.float64) * density
    hazard = torch.special.ndtr(z)
    masses = torch.stack([(1 - hazard) ** t * hazard for t in range(horizon)]
                         + [(1 - hazard) ** horizon], -1)
    marginal = (weights[:, None] * masses).sum(0)
    entropy = -torch.special.xlogy(marginal, marginal).sum()
    conditional = -(weights[:, None] * torch.special.xlogy(masses, masses)).sum()
    result = probit_next_event(mean, variance, horizon, quadrature_points=96)
    torch.testing.assert_close(result['probabilities'], marginal, rtol=1e-8, atol=1e-9)
    assert float(result['timing_information']) == pytest.approx(float(entropy - conditional), abs=1e-8)


def test_random_outcomes_without_parameter_uncertainty_have_zero_information():
    known = probit_next_event(0., 0., 16)
    unknown = probit_next_event(0., 1., 16)
    assert float(known['timing_information']) == pytest.approx(0., abs=1e-12)
    assert float(unknown['timing_information']) > .1
    assert float(known['event_probability']) == pytest.approx(1 - .5 ** 16)


def test_expected_timing_information_equals_average_exact_realized_posterior_kl():
    horizon = 5
    model = ProbitEventBelief({}, feature_dim=1, quadrature_points=96)
    prediction = model.predict([1.], horizon)
    realized = []
    for quiet in range(horizon):
        branch = ProbitEventBelief(copy.deepcopy(model.state))
        realized.append(branch.observe([1.], quiet, True)['realized_information'])
    quiet_branch = ProbitEventBelief(copy.deepcopy(model.state))
    realized.append(quiet_branch.observe([1.], horizon, False)['realized_information'])
    weighted = (prediction['probabilities'] * torch.tensor(realized)).sum()
    assert float(prediction['timing_information']) == pytest.approx(float(weighted), abs=2e-8)


def test_single_event_adf_moments_and_exact_unprojected_information():
    model = ProbitEventBelief({}, feature_dim=1, quadrature_points=96)
    result = model.observe([1.], quiet_ticks=0, event=True)
    assert result['log_evidence'] == pytest.approx(-math.log(2), abs=1e-10)
    assert result['posterior_mean'] == pytest.approx(1 / math.sqrt(math.pi), abs=1e-9)
    assert result['posterior_variance'] == pytest.approx(1 - 1 / math.pi, abs=1e-9)
    assert result['realized_information'] == pytest.approx(math.log(2) - .5, abs=1e-9)
    assert result['adf_information'] < result['realized_information']


def test_aggregate_survival_and_event_match_beta_integral_not_independent_trials():
    quiet = 11
    model = ProbitEventBelief({}, feature_dim=1, quadrature_points=96)
    result = model.observe([1.], quiet_ticks=quiet, event=True)
    assert result['log_evidence'] == pytest.approx(-math.log((quiet + 1) * (quiet + 2)), abs=1e-8)
    a, b = torch.tensor(2., dtype=torch.float64), torch.tensor(quiet + 1., dtype=torch.float64)
    expected_log_likelihood = torch.digamma(a) - torch.digamma(a + b)
    expected_log_likelihood += quiet * (torch.digamma(b) - torch.digamma(a + b))
    expected_kl = float(expected_log_likelihood) - result['log_evidence']
    assert result['realized_information'] == pytest.approx(expected_kl, abs=1e-8)


def test_quiet_is_evidence_and_zero_ticks_cannot_invent_it():
    model = ProbitEventBelief({}, feature_dim=2)
    before = copy.deepcopy(model.state)
    no_observation = model.observe([1., .2], 0, False)
    assert no_observation['realized_information'] == 0
    torch.testing.assert_close(model.state['mean'], before['mean'], rtol=0, atol=0)
    prediction_before = float(model.predict([1., .2], 1)['event_probability'])
    model.observe([1., .2], 8, False)
    assert float(model.predict([1., .2], 1)['event_probability']) < prediction_before
    assert model.state['quiet_ticks'] == 8
    assert model.state['events'] == 0


def test_pending_survival_split_and_torch_resume_equal_aggregate():
    aggregate = ProbitEventBelief({}, feature_dim=2, mark_dim=2)
    split = ProbitEventBelief({}, feature_dim=2, mark_dim=2)
    x, y = [1., -.4], [.3, -2.]
    aggregate.observe(x, 17, True, mark=y)
    split.begin(x)
    first = split.advance(5)
    buffer = io.BytesIO()
    torch.save(split.state, buffer)
    buffer.seek(0)
    restored = ProbitEventBelief(torch.load(buffer, weights_only=True))
    second = restored.advance(7)
    third = restored.resolve(True, mark=y, quiet_ticks=5)
    assert first['incremental_log_evidence'] + second['incremental_log_evidence'] + third['incremental_log_evidence'] == pytest.approx(third['log_evidence'])
    assert_tree_equal(aggregate.state, restored.state)


def test_long_survival_has_finite_positive_covariance():
    model = ProbitEventBelief({}, feature_dim=2, quadrature_points=96)
    result = model.observe([1., .5], 10000, True)
    assert math.isfinite(result['log_evidence'])
    assert math.isfinite(result['realized_information'])
    assert torch.linalg.eigvalsh(model.state['covariance']).min() > 0


def test_prediction_is_causal_and_mark_information_is_event_weighted():
    model = ProbitEventBelief({}, feature_dim=2, mark_dim=2)
    before = copy.deepcopy(model.state)
    prediction = model.predict([1., .4], 5)
    assert_tree_equal(model.state, before)
    expected = prediction['event_probability'] * model.mark.expected_information([1., .4])
    torch.testing.assert_close(prediction['expected_mark_information'], expected)
    torch.testing.assert_close(prediction['total_information'], prediction['timing_information'] + expected)
    empty = model.predict([1., .4], 0)
    assert float(empty['total_information']) == 0
    other = ProbitEventBelief(copy.deepcopy(before))
    a = model.observe([1., .4], 2, True, mark=[1., -1.])
    b = other.observe([1., .4], 2, True, mark=[100., -70.])
    assert a['mark']['expected_information_before'] == b['mark']['expected_information_before']
    assert a['mark']['realized_information'] != b['mark']['realized_information']


def test_quiet_does_not_update_conditional_event_mark():
    model = ProbitEventBelief({}, feature_dim=2, mark_dim=1)
    before = copy.deepcopy(model.mark.state)
    model.observe([1., .1], 7, False)
    assert_tree_equal(model.mark.state, before)


def test_nig_update_realized_kl_matches_general_matrix_formula():
    model = NormalInverseGammaMarks({}, feature_dim=2, mark_dim=2)
    before = copy.deepcopy(model.state)
    result = model.observe([1., -.6], [.5, -2.])
    old, new = before, model.state
    ao, an = torch.tensor(old['alpha']), torch.tensor(new['alpha'])
    # Inverse-gamma KL equals gamma precision KL under the reciprocal map.
    old_gamma = torch.distributions.Gamma(ao.double(), old['beta'])
    new_gamma = torch.distributions.Gamma(an.double(), new['beta'])
    kl = torch.distributions.kl_divergence(new_gamma, old_gamma)
    inv = torch.linalg.inv(old['covariance'])
    shift = new['mean'] - old['mean']
    gaussian = (torch.trace(inv @ new['covariance']) - 2
                + torch.linalg.slogdet(old['covariance']).logabsdet
                - torch.linalg.slogdet(new['covariance']).logabsdet)
    gaussian = .5 * (gaussian + new['alpha'] / new['beta'] * (shift * (inv @ shift)).sum(0))
    assert result['realized_information'] == pytest.approx(float((kl + gaussian).sum()), abs=1e-10)
    assert new['updates'] == 1
    assert new['alpha'] == old['alpha'] + .5


def test_nig_expected_information_equals_predictive_average_exact_posterior_kl():
    template = NormalInverseGammaMarks({}, feature_dim=1, mark_dim=1)
    x = torch.tensor([.7], dtype=torch.float64)
    scale = math.sqrt(float(template.state['beta'][0]) / template.state['alpha'] * (1 + .7 ** 2))
    nodes, weights = np.polynomial.legendre.leggauss(160)
    # Integrate the Student-t predictive on the real line via y=scale*tan(angle).
    angle = torch.tensor(nodes * math.pi / 2, dtype=torch.float64)
    y = scale * angle.tan()
    density = torch.distributions.StudentT(4., 0., scale).log_prob(y).exp()
    jacobian = scale / angle.cos().square()
    values = []
    for value in y:
        model = NormalInverseGammaMarks(copy.deepcopy(template.state))
        values.append(model.observe(x, value.reshape(1))['realized_information'])
    integral = (torch.tensor(values) * density * jacobian
                * torch.tensor(weights * math.pi / 2, dtype=torch.float64)).sum()
    assert float(template.expected_information(x)) == pytest.approx(float(integral), abs=2e-7)


def test_mark_scale_rescales_aleatoric_and_coefficient_uncertainty_together():
    low = NormalInverseGammaMarks({}, feature_dim=1, mark_dim=1, beta=1.)
    high = NormalInverseGammaMarks({}, feature_dim=1, mark_dim=1, beta=1e8)
    assert float(low.expected_information([1.])) == float(high.expected_information([1.]))


def test_candidate_quadratic_matches_materialized_features_and_full_prediction():
    generator = torch.Generator().manual_seed(11)
    d, c, count = 7, 3, 19
    model = ProbitEventBelief({}, feature_dim=d, mark_dim=2)
    base = torch.randn(d, generator=generator, dtype=torch.float64)
    basis = torch.randn(d, c, generator=generator, dtype=torch.float64)
    candidates = torch.randn(count, c, generator=generator, dtype=torch.float64)
    factor = torch.randn(d, d, generator=generator, dtype=torch.float64)
    model.state['covariance'] = factor @ factor.T + torch.eye(d)
    model.state['mean'] = torch.randn(d, generator=generator, dtype=torch.float64)
    features = base + candidates @ basis.T
    mean, variance = affine_candidate_moments(model.state['mean'], model.state['covariance'], base, basis, candidates)
    _, leverage = affine_candidate_moments(model.mark.state['mean'], model.mark.state['covariance'], base, basis, candidates)
    expected = model.predict(features, 4)
    actual = model.predict_from_moments(mean, variance, 4, leverage)
    for key in ('probabilities', 'total_information', 'expected_mark_information'):
        torch.testing.assert_close(actual[key], expected[key])


@pytest.mark.parametrize('mutate', [
    lambda s: s.update(format='unknown'),
    lambda s: s.update(feature_dim=3),
    lambda s: s.update(quiet_ticks=-1),
    lambda s: s.update(covariance=torch.tensor([[1., 2.], [0., 1.]], dtype=torch.float64)),
    lambda s: s.update(mean=torch.tensor([float('nan'), 0.], dtype=torch.float64)),
    lambda s: s.pop('trial'),
])
def test_resume_fails_fast_for_changed_or_invalid_schema(mutate):
    model = ProbitEventBelief({}, feature_dim=2)
    state = copy.deepcopy(model.state)
    mutate(state)
    with pytest.raises(ValueError):
        ProbitEventBelief(state)


@pytest.mark.parametrize('kwargs', [
    {'quiet_ticks': -1, 'event': True},
    {'quiet_ticks': 2, 'event': 1},
    {'quiet_ticks': 2, 'event': False, 'mark': [1.]},
    {'quiet_ticks': 2, 'event': True, 'mark': [float('nan')]},
])
def test_invalid_observation_is_transactional(kwargs):
    model = ProbitEventBelief({}, feature_dim=2, mark_dim=1)
    before = copy.deepcopy(model.state)
    with pytest.raises(ValueError):
        model.observe([1., .2], **kwargs)
    assert_tree_equal(model.state, before)


def test_pending_trial_schema_and_nested_trials_fail_fast():
    model = ProbitEventBelief({}, feature_dim=2)
    model.begin([1., 0.])
    with pytest.raises(ValueError):
        model.begin([0., 1.])
    state = copy.deepcopy(model.state)
    state['trial']['quiet_ticks'] = -1
    with pytest.raises(ValueError):
        ProbitEventBelief(state)
    assert model.state['format'] == FORMAT
