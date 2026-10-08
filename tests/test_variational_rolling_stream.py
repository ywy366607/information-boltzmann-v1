"""Deterministic numerical unit tests for Variational Rolling Stream Learning Mechanism."""

import math
import pytest
import torch
import torch.distributions as dist
from torch.distributions.kl import kl_divergence

from information_boltzmann.core.variational_rolling_stream import (
    VariationalGaussianBelief,
    VariationalBeliefModulator,
    RollingStreamTransformer,
    VariationalRollingStreamLearner,
)


def test_analytical_kl_exactness():
    """Verify analytical KL divergence formula against torch.distributions.kl."""
    dim = 16
    torch.manual_seed(42)

    mu_p = torch.randn(dim)
    log_std_p = torch.randn(dim) * 0.5
    prior = VariationalGaussianBelief(mean=mu_p, log_std=log_std_p)

    # 1. Identity check: KL(p || p) == 0
    kl_zero = prior.kl_divergence(prior)
    assert torch.isclose(kl_zero, torch.tensor(0.0), atol=1e-6)

    # 2. General check: KL(q || p) matching PyTorch Normal distribution KL
    mu_q = torch.randn(dim)
    log_std_q = torch.randn(dim) * 0.5
    posterior = VariationalGaussianBelief(mean=mu_q, log_std=log_std_q)

    kl_analytical = posterior.kl_divergence(prior)

    # PyTorch reference
    q_dist = dist.Normal(mu_q, torch.exp(log_std_q))
    p_dist = dist.Normal(mu_p, torch.exp(log_std_p))
    kl_reference = kl_divergence(q_dist, p_dist).sum()

    assert torch.isclose(kl_analytical, kl_reference, rtol=1e-5, atol=1e-5)
    assert kl_analytical.item() > 0.0


def test_temporal_transition_retention():
    """Verify Ornstein-Uhlenbeck retention dynamics at boundary transitions."""
    dim = 8
    mu_q = torch.full((dim,), 2.5)
    log_std_q = torch.full((dim,), 0.5)  # std = exp(0.5) ~= 1.6487, var ~= 2.718
    belief = VariationalGaussianBelief(mean=mu_q, log_std=log_std_q)

    # Case 1: rho = 1.0 (Full retention)
    trans_1 = belief.transition(retention=1.0, base_mean=0.0, base_log_std=0.0)
    assert torch.allclose(trans_1.mean, mu_q)
    assert torch.allclose(trans_1.log_std, log_std_q)

    # Case 2: rho = 0.0 (Complete relaxation to baseline)
    trans_0 = belief.transition(retention=0.0, base_mean=0.0, base_log_std=0.0)
    assert torch.allclose(trans_0.mean, torch.zeros(dim))
    assert torch.allclose(trans_0.log_std, torch.zeros(dim), atol=1e-5)

    # Case 3: rho = 0.5
    rho = 0.5
    trans_half = belief.transition(retention=rho, base_mean=0.0, base_log_std=0.0)
    expected_mean = rho * 2.5
    expected_var = (rho ** 2) * math.exp(1.0) + (1.0 - rho ** 2) * 1.0
    expected_log_std = 0.5 * math.log(expected_var)

    assert torch.allclose(trans_half.mean, torch.full((dim,), expected_mean), atol=1e-5)
    assert torch.allclose(trans_half.log_std, torch.full((dim,), expected_log_std), atol=1e-5)


def test_frozen_prior_invariant():
    """Verify that prior remains strictly frozen and untouched during inner assimilation."""
    torch.manual_seed(101)
    vocab_size = 100
    model = RollingStreamTransformer(
        vocab_size=vocab_size,
        dim=32,
        num_layers=1,
        num_heads=2,
        max_len=16,
        latent_dim=8,
    )
    learner = VariationalRollingStreamLearner(
        model=model,
        latent_dim=8,
        window_size=16,
        stride=8,
        inner_lr=0.1,
        inner_max_steps=5,
        device="cpu",
    )

    dummy_tokens = torch.randint(0, vocab_size, (1, 16))
    initial_prior = learner.current_prior.clone()

    # Run inner loop
    converged_q, steps = learner.inner_assimilation_loop(
        tokens=dummy_tokens, prior=learner.current_prior
    )

    # Prior must remain strictly unchanged
    assert torch.equal(learner.current_prior.mean, initial_prior.mean)
    assert torch.equal(learner.current_prior.log_std, initial_prior.log_std)
    assert learner.current_prior.mean.grad is None
    assert learner.current_prior.log_std.grad is None


def test_plateau_early_termination():
    """Verify that inner assimilation terminates early when delta FE falls below threshold."""
    vocab_size = 100
    model = RollingStreamTransformer(
        vocab_size=vocab_size,
        dim=32,
        num_layers=1,
        num_heads=2,
        max_len=16,
        latent_dim=8,
    )
    learner = VariationalRollingStreamLearner(
        model=model,
        latent_dim=8,
        window_size=16,
        stride=8,
        inner_lr=1e-5,  # Tiny LR so FE barely changes
        inner_max_steps=20,
        plateau_delta=1.0,  # High threshold guarantees early exit
        plateau_patience=2,
        device="cpu",
    )

    dummy_tokens = torch.randint(0, vocab_size, (1, 16))
    _, steps = learner.inner_assimilation_loop(
        tokens=dummy_tokens, prior=learner.current_prior
    )

    # Must terminate well before 20 steps (patience=2 implies step 3 or earlier)
    assert len(steps) < 20
    assert steps[-1].plateau_triggered is True


def test_stream_continuous_rolling_steps():
    """Verify end-to-end rolling stream progression over multiple consecutive windows."""
    vocab_size = 200
    model = RollingStreamTransformer(
        vocab_size=vocab_size,
        dim=32,
        num_layers=1,
        num_heads=2,
        max_len=16,
        latent_dim=8,
    )
    learner = VariationalRollingStreamLearner(
        model=model,
        latent_dim=8,
        window_size=16,
        stride=8,
        inner_lr=0.05,
        inner_max_steps=5,
        device="cpu",
    )

    # Stream of 32 tokens = 3 rolling windows (0..16, 8..24, 16..32)
    stream_tokens = torch.randint(0, vocab_size, (32,))

    for w_idx in range(3):
        start_idx = w_idx * 8
        window = stream_tokens[start_idx : start_idx + 16]
        report = learner.step(window)

        assert report.window_idx == w_idx + 1
        assert report.inner_steps_taken >= 1
        assert report.prequential_nll > 0.0
        assert report.plateau_nll > 0.0
        assert report.wall_time_ms > 0.0
        assert isinstance(report.adaptation_gain, float)

    assert learner.total_windows == 3
    assert learner.total_tokens_seen == 24
