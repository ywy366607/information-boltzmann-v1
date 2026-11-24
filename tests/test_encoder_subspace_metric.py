"""Unit tests for Closed-Form Subspace Natural Gradient on `model.encoder`."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts.ib_local import encoder_subspace_metric as esm


def _make_engine(vectors: torch.Tensor, damping: float = 0.25):
    n, d = vectors.shape
    mem = SimpleNamespace(
        occupied=torch.ones(n, dtype=torch.bool),
        source=vectors.clone(),
        action=vectors.clone(),
        successor=vectors.clone(),
        long_successor=vectors.clone(),
        episodes=torch.tensor(n),
    )
    enc = torch.nn.Linear(d, 16, bias=False)
    model = SimpleNamespace(memory=mem, encoder=enc)
    eng = SimpleNamespace(model=model)
    esm.configure(eng, damping=damping)
    return eng


def test_orthogonal_direction_preserves_unit_plasticity_and_occupied_contracts():
    d = 32
    # Orthogonal memories: first half along e_0 (μ_0 = 16), second half along e_1 (μ_1 = 1).
    vecs = torch.zeros(64, d)
    vecs[:32, 0] = 4.0 * (2.0 ** 0.5)   # mean x_0^2 = 16.0
    vecs[32:, 1] = 1.0 * (2.0 ** 0.5)   # mean x_1^2 = 1.0
    eng = _make_engine(vecs, damping=0.25)

    g = torch.zeros(16, d)
    g[:, 0] = 1.0  # heavily occupied
    g[:, 1] = 1.0  # moderately occupied
    g[:, 2] = 1.0  # strictly orthogonal novel direction!

    p = esm.project_gradient(eng, g)
    # Exact analytic eigenvalues: λ / (λ + μ)
    assert torch.allclose(p[:, 0], torch.full((16,), 0.25 / 16.25), atol=1e-5)
    assert torch.allclose(p[:, 1], torch.full((16,), 0.25 / 1.25), atol=1e-5)
    assert torch.allclose(p[:, 2], torch.ones(16), atol=1e-6)  # 100% unit plasticity!


def test_amgm_uniform_drift_energy_bound_and_strict_descent():
    torch.manual_seed(20261124)
    d = 64
    vecs = torch.randn(200, d) * torch.linspace(0.01, 5.0, d)
    lam = 0.25
    eng = _make_engine(vecs, damping=lam)
    c_mean, _ = esm.memory_gram_matrix(eng.model.memory)

    g = torch.randn(16, d)
    p = esm.project_gradient(eng, g)

    # 1. Strict descent: <g, p>_F > 0
    inner = float((g * p).sum())
    assert inner > 0.0

    # 2. Universal AM-GM bound: tr(p C̄ pᵀ) ≤ (λ / 4) ‖g‖_F²
    drift = float(((p @ c_mean) * p).sum())
    amgm_cap = (lam / 4.0) * float(g.square().sum())
    assert drift <= amgm_cap + 1e-6

    # 3. Massive drift suppression compared to unprojected raw gradient
    raw_drift = float(((g @ c_mean) * g).sum())
    assert drift < 0.05 * raw_drift

    # 4. Exact Cholesky parity vs explicit dense inverse
    eye = torch.eye(d)
    p_explicit = g @ (lam * torch.linalg.inv(lam * eye + c_mean))
    assert float((p - p_explicit).abs().max()) < 1e-5


def test_apply_during_backward_hook_projects_and_cleans_up():
    torch.manual_seed(7)
    d = 16
    vecs = torch.zeros(32, d)
    vecs[:, 0] = 3.0
    eng = _make_engine(vecs, damping=0.25)

    x = torch.ones(1, d)
    with esm.apply_during_backward(eng):
        loss = eng.model.encoder(x).sum()
        loss.backward()
    proj_grad = eng.model.encoder.weight.grad.clone()

    # After context exits, hook must be cleanly removed (raw unprojected gradient).
    eng.model.encoder.weight.grad = None
    loss2 = eng.model.encoder(x).sum()
    loss2.backward()
    raw_grad = eng.model.encoder.weight.grad.clone()

    assert torch.allclose(proj_grad[:, 0], raw_grad[:, 0] * (0.25 / 9.25), atol=1e-5)
    assert torch.allclose(proj_grad[:, 1:], raw_grad[:, 1:], atol=1e-6)
    assert eng.encoder_subspace_metric_state['applications'] == 1


def test_post_adam_trust_region_passes_small_steps_and_clamps_large_drift():
    torch.manual_seed(42)
    d = 32
    vecs = torch.randn(128, d) * torch.linspace(0.05, 4.0, d)
    eng = _make_engine(vecs, damping=0.25)
    esm.configure(eng, damping=0.25, drift_budget=1e-3, post_step=True)
    c_mean, _ = esm.memory_gram_matrix(eng.model.memory)

    # 1. Tiny step well inside drift budget -> exact pass-through (zero distortion).
    w0 = eng.model.encoder.weight.detach().clone()
    tiny_step = torch.randn_like(w0) * 1e-5
    with esm.apply_to_step(eng):
        with torch.no_grad():
            eng.model.encoder.weight.add_(tiny_step)
    actual_tiny = eng.model.encoder.weight.detach() - w0
    assert torch.allclose(actual_tiny, tiny_step, atol=1e-7)
    assert eng.encoder_subspace_metric_state['pass_throughs'] == 1

    # 2. Large step exceeding drift budget -> exact trust-region clamp to drift == budget.
    w1 = eng.model.encoder.weight.detach().clone()
    big_step = torch.randn_like(w1) * 0.2
    with esm.apply_to_step(eng):
        with torch.no_grad():
            eng.model.encoder.weight.add_(big_step)
    actual_big = eng.model.encoder.weight.detach() - w1
    actual_drift = float(((actual_big @ c_mean) * actual_big).sum().item())
    assert actual_drift == pytest.approx(1e-3, rel=1e-3)
    assert eng.encoder_subspace_metric_state['trust_region_clamps'] == 1
    # Must remain strict descent with respect to proposed step D₀.
    assert float((actual_big * big_step).sum().item()) > 0.0

