import copy
import math

import pytest
import torch

from information_boltzmann.core.fly_context_filter import (
    DirectionalContextFilter, NODES, WEIGHTS, categorical_information,
)


def test_quadrature_has_gaussian_first_two_moments():
    assert float((WEIGHTS * NODES).sum()) == 0.
    assert float((WEIGHTS * NODES.square()).sum()) == pytest.approx(1.)


def test_categorical_information_is_disagreement_not_noise_entropy():
    iid = torch.tensor([[.2, .8]] * 3, dtype=torch.float64).log()
    assert categorical_information(iid)['expected_information'] == pytest.approx(0., abs=1e-14)
    distinct = torch.eye(3, dtype=torch.float64).log()
    assert categorical_information(distinct)['expected_information'] == pytest.approx(
        float(-(WEIGHTS * WEIGHTS.log()).sum()))


def test_expected_information_equals_expected_realized_bayes_gain():
    lp = torch.tensor([[.8, .2], [.5, .5], [.1, .9]], dtype=torch.float64).log()
    expected = categorical_information(lp)
    realized = []
    means, covariances = [], []
    for target in range(2):
        f = DirectionalContextFilter(dimension=1, process_variance=0.)
        f.begin_transition([0.], 1, [1.])
        f.forecast(lp)
        post = f.observe(target)
        realized.append(post['realized_information'])
        means.append(post['mean'][0])
        covariances.append(post['covariance'][0, 0])
    probabilities = expected['log_predictive'].exp()
    assert float(probabilities @ torch.tensor(realized, dtype=torch.float64)) == pytest.approx(
        expected['expected_information'])
    means = torch.stack(means)
    assert float(probabilities @ means) == pytest.approx(0., abs=1e-14)
    assert float(probabilities @ (torch.stack(covariances) + means.square())) == pytest.approx(1.)


def test_tiny_likelihoods_remain_in_log_domain():
    logits = torch.tensor([[-1000., 0.], [-1002., 0.], [-1004., 0.]], dtype=torch.float64)
    lp = torch.log_softmax(logits, -1)
    f = DirectionalContextFilter(dimension=1)
    f.begin_transition([0.], 2, [1.])
    f.forecast(lp)
    post = f.observe(0)
    assert math.isfinite(post['prequential_logp'])
    assert torch.isfinite(post['mean']).all()
    assert torch.isfinite(post['covariance']).all()


def test_directional_update_preserves_orthogonal_covariance():
    f = DirectionalContextFilter(dimension=2, process_variance=0.)
    nodes = f.begin_transition([0., 0.], 4, [1., 0.])
    assert torch.equal(nodes[:, 1], torch.zeros(3, dtype=torch.float64))
    lp = torch.tensor([[.8, .2], [.5, .5], [.1, .9]], dtype=torch.float64).log()
    f.forecast(lp)
    result = f.observe(1)
    assert result['mean'][0] > 0
    assert result['covariance'][1, 1] == 1.
    assert result['covariance'][0, 1] == 0.
    assert torch.linalg.eigvalsh(result['covariance']).min() >= 0


def test_iid_observation_preserves_prior_and_process_estimate():
    f = DirectionalContextFilter(dimension=3, process_variance=.02)
    f.begin_transition([2., -1., 3.], 7, [1., 2., 1.])
    prior = f.covariance.clone()
    f.forecast(torch.tensor([[.4, .6]] * 3, dtype=torch.float64).log())
    result = f.observe(1)
    torch.testing.assert_close(result['mean'], torch.tensor([2., -1., 3.], dtype=torch.float64))
    torch.testing.assert_close(result['covariance'], prior)
    assert result['realized_information'] == pytest.approx(0., abs=1e-14)
    assert result['process_variance_em'] == pytest.approx(.02)


def test_em_matches_direct_conditional_increment_moment():
    f = DirectionalContextFilter(dimension=1, initial_variance=.7, process_variance=.04)
    f.begin_transition([.1], 5, [1.])
    f.forecast(torch.tensor([[.9, .1], [.5, .5], [.1, .9]], dtype=torch.float64).log())
    result = f.observe(1)
    C, Q = .9, .2
    expected = (Q - Q * Q / C + (Q / C) ** 2
                * (float(result['covariance'][0, 0]) + (float(result['mean'][0]) - .1) ** 2)) / 5
    assert result['process_variance_em'] == pytest.approx(expected)


def test_pending_resume_is_exact_and_actual_clock_counted_once():
    f = DirectionalContextFilter(dimension=2)
    f.begin_transition([.1, .2], 13, [1., -1.])
    lp = torch.tensor([[.9, .1], [.6, .4], [.2, .8]], dtype=torch.float64).log()
    f.forecast(lp)
    restored = DirectionalContextFilter(f.state_dict(), dimension=2)
    a, b = f.observe(1), restored.observe(1)
    torch.testing.assert_close(a['mean'], b['mean'], rtol=0, atol=0)
    torch.testing.assert_close(a['covariance'], b['covariance'], rtol=0, atol=0)
    assert f.state['clock_ticks'] == restored.state['clock_ticks'] == 13
    with pytest.raises(ValueError):
        f.observe(1)
    restored.begin_transition(restored.mean, 4, [0., 1.])
    assert restored.state['clock_ticks'] == 17


def test_no_positive_floor_or_fake_noise_for_zero_variance():
    f = DirectionalContextFilter(dimension=2, initial_variance=0., process_variance=0.)
    nodes = f.begin_transition([1., 2.], 9, [1., 0.])
    assert torch.equal(nodes, torch.tensor([[1., 2.]] * 3, dtype=torch.float64))
    f.forecast(torch.tensor([[.5, .5]] * 3, dtype=torch.float64).log())
    f.observe(0)
    assert f.state['process_variance'] == 0.
    assert torch.count_nonzero(f.covariance) == 0


def test_discard_preserves_elapsed_prior_without_learning_or_process_em():
    f = DirectionalContextFilter(dimension=2, process_variance=.03)
    f.begin_transition([.1, .2], 7, [1., 0.])
    f.forecast(torch.tensor([[.5, .5]] * 3, dtype=torch.float64).log())
    before = f.state_dict()
    f.discard_unobserved()
    assert f.state['pending'] is None
    for key in ('mean', 'covariance'):
        torch.testing.assert_close(f.state[key], before[key], rtol=0, atol=0)
    for key in ('clock_ticks', 'updates', 'process_variance', 'process_em_sum', 'process_em_count'):
        assert f.state[key] == before[key]
    with pytest.raises(ValueError):
        f.observe(0)
    with pytest.raises(ValueError):
        f.cancel_pending()
    f.begin_transition([.1, .2], 2, [0., 1.])
    assert f.state['clock_ticks'] == 9


def test_speculative_clone_cancellation_does_not_mutate_actual_pending():
    actual = DirectionalContextFilter(dimension=1)
    actual.begin_transition([0.], 5, [1.])
    saved = actual.state_dict()
    clone = DirectionalContextFilter(actual.state_dict(), dimension=1)
    clone.cancel_pending()
    clone.begin_transition([0.], 12, [1.])
    assert actual.state['pending'] is not None
    assert actual.state['clock_ticks'] == saved['clock_ticks'] == 5
    assert clone.state['clock_ticks'] == 17


def test_cannot_overwrite_forecast_or_advance_pending_transition():
    f = DirectionalContextFilter(dimension=1)
    with pytest.raises(ValueError):
        f.observe(0)
    f.begin_transition([0.], 1, [1.])
    with pytest.raises(ValueError):
        f.begin_transition([0.], 1, [1.])
    lp = torch.tensor([[.5, .5]] * 3, dtype=torch.float64).log()
    f.forecast(lp)
    with pytest.raises(ValueError):
        f.forecast(torch.tensor([[.6, .4]] * 3, dtype=torch.float64).log())


@pytest.mark.parametrize('bad', [float('nan'), -1., float('inf')])
def test_rejects_invalid_process_variance(bad):
    with pytest.raises(ValueError):
        DirectionalContextFilter(dimension=1, process_variance=bad)


def test_guards_shapes_clock_and_log_probabilities():
    f = DirectionalContextFilter(dimension=2)
    for mean, ticks, direction in [([0.], 1, [1., 0.]), ([0., 0.], -1, [1., 0.]),
                                    ([0., 0.], 1.5, [1., 0.]), ([0., 0.], 1, [0., 0.])]:
        with pytest.raises(ValueError):
            f.begin_transition(mean, ticks, direction)
    f.begin_transition([0., 0.], 1, [1., 0.])
    with pytest.raises(ValueError):
        f.forecast(torch.zeros(3, 2))
    f.forecast(torch.tensor([[1., 0.]] * 3, dtype=torch.float64).log())
    before = f.state_dict()
    with pytest.raises(ValueError):
        f.observe(1)
    assert f.state['updates'] == before['updates']
    assert f.state['pending'] is not None


def test_rejects_invalid_saved_covariance():
    f = DirectionalContextFilter(dimension=2)
    state = copy.deepcopy(f.state)
    state['covariance'][0, 0] = -1.
    with pytest.raises(ValueError):
        DirectionalContextFilter(state, dimension=2)
