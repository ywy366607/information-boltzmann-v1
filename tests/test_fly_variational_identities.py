"""Numerical probability identities, not memory/capability experiments.

These reference calculations do not invoke the fly model, train parameters,
allocate CUDA tensors, or claim a finite implementation of full-brain inference.
"""

import itertools

import numpy as np


def _normal_log_prob(value, mean, covariance):
    residual = value - mean
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign > 0
    return -.5 * (
        len(value) * np.log(2 * np.pi)
        + logdet + residual @ np.linalg.solve(covariance, residual)
    )


def _gaussian_fixture():
    mean = np.array([.2, -.3])
    previous_cov = np.array([[.7, .1], [.1, .5]])
    transition = np.array([[.5, -.2], [.3, .6]])
    process_cov = np.diag([.4, .3])
    # Only the designated motor coordinate is observed.
    read = np.array([[0., 1.]])
    observation_cov = np.array([[.2]])
    observation = np.array([.8])
    return mean, previous_cov, transition, process_cov, read, observation_cov, observation


def _gaussian_posterior(mean, previous_cov, transition, process_cov,
                        read, observation_cov, observation):
    n = len(mean)
    joint_mean = np.concatenate([mean, transition @ mean])
    joint_cov = np.block([
        [previous_cov, previous_cov @ transition.T],
        [transition @ previous_cov,
         transition @ previous_cov @ transition.T + process_cov],
    ])
    observer = np.concatenate([np.zeros((len(observation), n)), read], axis=1)
    evidence_cov = observer @ joint_cov @ observer.T + observation_cov
    gain = np.linalg.solve(evidence_cov, observer @ joint_cov).T
    posterior_mean = joint_mean + gain @ (observation - observer @ joint_mean)
    posterior_cov = joint_cov - gain @ observer @ joint_cov
    log_evidence = _normal_log_prob(observation, observer @ joint_mean, evidence_cov)
    return joint_mean, joint_cov, observer, posterior_mean, posterior_cov, log_evidence


def test_exact_posterior_free_energy_equals_negative_log_evidence():
    fixture = _gaussian_fixture()
    *_, read, observation_cov, observation = fixture
    jm, jc, observer, pm, pc, log_evidence = _gaussian_posterior(*fixture)
    delta = pm - jm
    dimension = len(pm)
    kl = .5 * (
        np.trace(np.linalg.solve(jc, pc))
        + delta @ np.linalg.solve(jc, delta) - dimension
        + np.linalg.slogdet(jc)[1] - np.linalg.slogdet(pc)[1]
    )
    residual = observation - observer @ pm
    expected_nll = .5 * (
        len(observation) * np.log(2 * np.pi)
        + np.linalg.slogdet(observation_cov)[1]
        + residual @ np.linalg.solve(observation_cov, residual)
        + np.trace(np.linalg.solve(observation_cov, observer @ pc @ observer.T))
    )
    np.testing.assert_allclose(kl + expected_nll, -log_evidence, atol=2e-14)
    assert read[0, 0] == 0  # The reference has no all-site output.


def test_local_posterior_score_equals_marginal_likelihood_derivative():
    mean, p, w, q, c, r, y = _gaussian_fixture()
    _, _, _, pm, pc, _ = _gaussian_posterior(mean, p, w, q, c, r, y)
    n = len(mean)
    expected_uu = pc[:n, :n] + np.outer(pm[:n], pm[:n])
    expected_vu = pc[n:, :n] + np.outer(pm[n:], pm[:n])
    local_score = np.linalg.solve(q, expected_vu - w @ expected_uu)
    finite_difference = np.empty_like(w)
    eps = 1e-5
    for i, j in np.ndindex(w.shape):
        plus, minus = w.copy(), w.copy()
        plus[i, j] += eps
        minus[i, j] -= eps
        lp = _gaussian_posterior(mean, p, plus, q, c, r, y)[-1]
        lm = _gaussian_posterior(mean, p, minus, q, c, r, y)[-1]
        finite_difference[i, j] = (lp - lm) / (2 * eps)
    np.testing.assert_allclose(local_score, finite_difference, rtol=2e-8, atol=2e-10)


def test_posterior_mean_alone_is_not_the_complete_local_score():
    mean, p, w, q, c, r, y = _gaussian_fixture()
    _, _, _, pm, pc, _ = _gaussian_posterior(mean, p, w, q, c, r, y)
    n = len(mean)
    mean_only = np.linalg.solve(q, np.outer(pm[n:] - w @ pm[:n], pm[:n]))
    covariance_term = np.linalg.solve(q, pc[n:, :n] - w @ pc[:n, :n])
    assert np.linalg.norm(covariance_term) > 1e-2
    assert np.linalg.norm(mean_only + covariance_term - mean_only) > 1e-2


def test_prior_expected_local_score_is_zero():
    mean, p, w, q, *_ = _gaussian_fixture()
    expected_uu = p + np.outer(mean, mean)
    expected_vu = w @ expected_uu
    np.testing.assert_allclose(
        np.linalg.solve(q, expected_vu - w @ expected_uu), 0., atol=1e-14
    )


def test_motor_evidence_does_not_create_unrelated_current_factor_credit():
    mean, p, w, q, c, r, y = _gaussian_fixture()
    _, _, _, pm, pc, _ = _gaussian_posterior(mean, p, w, q, c, r, y)
    n = len(mean)
    expected_uu = pc[:n, :n] + np.outer(pm[:n], pm[:n])
    expected_vu = pc[n:, :n] + np.outer(pm[n:], pm[:n])
    local_score = np.linalg.solve(q, expected_vu - w @ expected_uu)
    # Current coordinate 0 is unobserved in this reference transition model.
    # Its factor score averages to zero, although shared previous-state
    # uncertainty can still induce posterior correlations with the motor.
    np.testing.assert_allclose(local_score[0], 0., atol=1e-14)
    assert np.linalg.norm(local_score[1]) > 1e-2


def test_forward_conditional_statistics_equal_direct_path_expectation():
    # A fixed finite probability table verifies a tower-property identity.
    # No optimizer, dataset, learning-curve or capability metric is involved.
    transition = np.array([[.6, .3, .1], [.2, .5, .3], [.1, .2, .7]])
    emission = np.array([[.5, .3, .2], [.2, .6, .2], [.3, .2, .5]])
    initial = np.array([.2, .5, .3])
    observations = (0, 2, 1)
    local = np.arange(9, dtype=float).reshape(3, 3) / 7
    filtering = initial.copy()
    statistic = np.zeros(3)
    for observation in observations:
        pair_prior = filtering[:, None] * transition
        predicted = pair_prior.sum(axis=0)
        backward_kernel = pair_prior / predicted[None, :]
        statistic = (backward_kernel * (statistic[:, None] + local)).sum(axis=0)
        unnormalized = predicted * emission[:, observation]
        filtering = unnormalized / unnormalized.sum()
        np.testing.assert_allclose(backward_kernel.sum(axis=0), 1., atol=1e-14)
    numerator = denominator = 0.
    for path in itertools.product(range(3), repeat=len(observations) + 1):
        probability = initial[path[0]]
        total = 0.
        for t, observation in enumerate(observations):
            left, right = path[t:t + 2]
            probability *= transition[left, right] * emission[right, observation]
            total += local[left, right]
        numerator += probability * total
        denominator += probability
    np.testing.assert_allclose(filtering @ statistic, numerator / denominator, atol=2e-14)
