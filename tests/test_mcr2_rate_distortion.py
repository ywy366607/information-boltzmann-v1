"""Unit tests for Rate-Distortion & MCR^2 Dual-Engine operators.

Tests:
1. Deterministic properties & Sylvester determinant equivalence in Gaussian coding rate.
2. Differentiability and finite gradient flow for MCR^2 loss.
3. Dimensional collapse detection: orthogonal vs collapsed subspace volume.
4. Natural emergence of admission gating in FlyAdaptiveAdmissionGatekeeper without artificial clamps.
"""

import math
import pytest
import torch

from information_boltzmann.core.mcr2_rate_distortion import (
    FlyAdaptiveAdmissionGatekeeper,
    MCR2Loss,
    compute_gaussian_coding_rate,
)


def test_gaussian_coding_rate_sylvester_equivalence():
    """Verify that Sylvester's theorem det(I + alpha Z Z^T) == det(I + alpha Z^T Z) holds."""
    torch.manual_seed(42)
    d, m = 64, 16
    eps = 0.5
    alpha = d / (float(m) * (eps ** 2))
    Z = torch.randn(d, m)

    # 1. Direct computation in R^{d x d}
    gram_d = torch.eye(d) + alpha * (Z @ Z.t())
    logdet_d = 0.5 * torch.linalg.slogdet(gram_d).logabsdet

    # 2. Sylvester computation in R^{m x m}
    gram_m = torch.eye(m) + alpha * (Z.t() @ Z)
    logdet_m = 0.5 * torch.linalg.slogdet(gram_m).logabsdet

    # 3. Our operator compute_gaussian_coding_rate (which automatically switches)
    rate_op = compute_gaussian_coding_rate(Z, eps=eps)

    assert torch.isclose(logdet_d, logdet_m, rtol=1e-4, atol=1e-4), "Sylvester equivalence violated"
    assert torch.isclose(rate_op, logdet_m, rtol=1e-4, atol=1e-4), "Operator did not match Sylvester logdet"


def test_gaussian_coding_rate_orthogonal_vs_collapsed():
    """Verify that orthogonal vectors maximize R(Z) while collapsed vectors shrink R(Z)."""
    torch.manual_seed(42)
    d, m = 32, 8
    eps = 0.5

    # Case A: Orthogonal vectors (maximal subspace expansion)
    Q, _ = torch.linalg.qr(torch.randn(d, m))
    Z_ortho = Q[:, :m] * 2.0
    rate_ortho = compute_gaussian_coding_rate(Z_ortho, eps=eps)

    # Case B: Rank-1 collapsed vectors (all vectors identical along 1 direction)
    v = torch.randn(d, 1)
    v = v / v.norm()
    Z_collapsed = v.repeat(1, m) * 2.0
    rate_collapsed = compute_gaussian_coding_rate(Z_collapsed, eps=eps)

    assert rate_ortho > rate_collapsed, (
        f"Orthogonal coding rate ({rate_ortho.item():.3f}) must exceed "
        f"collapsed coding rate ({rate_collapsed.item():.3f})"
    )


def test_mcr2_loss_differentiability_and_gradient_flow():
    """Verify that MCR^2 loss is fully differentiable and produces finite, non-zero gradients."""
    torch.manual_seed(42)
    d_model = 64
    K = 8
    loss_fn = MCR2Loss(d_model=d_model, eps=0.5, beta=0.1)

    latents = torch.randn(K, d_model, requires_grad=True)

    # Forward pass
    out = loss_fn(latents)
    loss = out["loss"]
    delta_R = out["delta_R"]

    assert loss.ndim == 0, "Loss must be scalar"
    assert delta_R.item() > 0.0, "Delta R must be positive for non-collapsed vectors"
    assert not torch.isnan(loss), "Loss must not be NaN"

    # Backward pass
    loss.backward()

    assert latents.grad is not None, "Gradients must exist"
    assert not torch.isnan(latents.grad).any(), "Gradients must not contain NaN"
    assert not torch.isinf(latents.grad).any(), "Gradients must not contain Inf"
    assert latents.grad.norm().item() > 1e-6, "Gradients must be non-zero"


def test_mcr2_loss_with_trajectory_clusters():
    """Verify MCR^2 intra-trajectory compression with settling point groups."""
    torch.manual_seed(42)
    d_model = 32
    K = 4
    loss_fn = MCR2Loss(d_model=d_model, eps=0.5, beta=0.05)

    token_latents = torch.randn(K, d_model)
    # 4 tokens, each with 3 settling points
    trajectories = [token_latents[i:i+1].repeat(3, 1) + 0.01 * torch.randn(3, d_model) for i in range(K)]

    out = loss_fn(token_latents, trajectory_latents=trajectories)

    assert out["R_total"].item() > 0.0
    assert out["R_cluster"].item() >= 0.0
    assert out["delta_R"].item() > 0.0


def test_admission_gatekeeper_zero_rule_emergence():
    """Verify that high kinetic flux naturally prevents premature exit at t <= 2 with ZERO rules."""
    gatekeeper = FlyAdaptiveAdmissionGatekeeper(
        num_neurons=1000,
        vocab_size=100,
        lambda_flux=8.0,
        flux_baseline_threshold=0.036,
        max_settle_ticks=14,
    )

    # Simulate realistic token entry and physical relaxation:
    # t=0: injection flux ~ 0.50
    # t=1..5: exponential decay relaxation towards attractor
    h_sequence = [torch.zeros(1000)]
    h = torch.randn(1000) * 0.50 # t=0 shock
    h_sequence.append(h)
    for _ in range(6):
        h = 0.91 * h # smooth relaxation
        h_sequence.append(h)

    gatekeeper.reset()
    admit_ticks = []

    for t in range(len(h_sequence) - 1):
        h_prev = h_sequence[t]
        h_curr = h_sequence[t + 1]
        dummy_logits = torch.randn(100)
        dummy_latent = torch.randn(64)

        metrics = gatekeeper.compute_metrics(h_curr, h_prev, dummy_logits, dummy_latent)
        admit, reason = gatekeeper.should_admit_next_token()

        if admit:
            admit_ticks.append((t, reason))
            break

    # Crucial property: Gatekeeper MUST NOT admit at t=0, 1, or 2 because dF/dt is strongly negative!
    assert len(admit_ticks) > 0, "Gatekeeper must eventually admit"
    first_admit_tick, reason = admit_ticks[0]
    assert first_admit_tick >= 3, (
        f"Gatekeeper admitted prematurely at tick {first_admit_tick} ({reason})! "
        f"Must naturally emerge at t >= 3 without hardcoding."
    )
