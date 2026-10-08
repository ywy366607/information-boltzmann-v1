"""Smoke test verifying ATan surrogate, learnable thresholds and conductance gains in COBA."""
import math
import numpy as np
import pytest
import torch

from information_boltzmann.core.fly_reservoir import FlyReservoirLM


def test_coba_full_plasticity_smoke(tmp_path):
    n = 40
    rng = np.random.default_rng(123)
    pre = rng.integers(0, n, 120).astype(np.int32)
    post = rng.integers(0, n, 120).astype(np.int32)
    weight = np.abs(rng.standard_normal(120)).astype(np.float32)
    nt_sign = np.array([1] * 20 + [-1] * 20, dtype=np.int8)
    superclass_id = np.array([0] * 15 + [1] * 15 + [2] * 10, dtype=np.int8)

    path = tmp_path / "tiny_coba_smoke.npz"
    np.savez_compressed(
        path, edge_pre=pre, edge_post=post, edge_weight=weight,
        edge_delay=np.ones_like(pre, dtype=np.int32),
        neuron_body_ids=np.arange(n, dtype=np.int64),
        nt_sign=nt_sign,
        superclass_id=superclass_id,
        superclass_names=np.array(["sensory", "interneuron", "output"]),
        meta=np.array("{}")
    )

    model = FlyReservoirLM(
        path, vocab_size=32, d_model=16,
        learnable_time_constants=True,
        learnable_thresholds=True,
        learnable_conductance_gains=True,
        synapse_model="coba",
        threshold=0.05
    )

    # 1. Verify parameter shapes and initial values
    assert model.learnable_thresholds is True
    assert model.learnable_conductance_gains is True
    assert model.log_threshold.shape == (3,)
    assert torch.allclose(torch.exp(model.log_threshold), torch.full((3,), 0.05), atol=1e-5)
    assert model.log_g_e.item() == 0.0  # exp(0) = 1.0
    assert model.log_g_i.item() == 0.0  # exp(0) = 1.0

    # 2. Trainable parameters collection
    trainable = [p for p in model.parameters() if p.requires_grad]
    param_names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert "log_threshold" in param_names
    assert "log_g_e" in param_names
    assert "log_g_i" in param_names
    assert "log_tau_m" in param_names

    optimizer = torch.optim.AdamW(trainable, lr=1e-2)

    # 3. Step forward and backward for 3 steps
    h = torch.zeros(1, n)
    ring = tuple(torch.ones(1, n) for _ in range(4))  # warm-in spikes
    ge = torch.zeros(1, n)
    gi = torch.zeros(1, n)

    initial_thresh = model.log_threshold.clone()
    initial_ge = model.log_g_e.clone()
    initial_gi = model.log_g_i.clone()

    for step in range(3):
        optimizer.zero_grad()
        ids = torch.randint(0, 32, (1, 8))
        targets = torch.randint(0, 32, (1, 8))
        loss, h, diag, ring, ge, gi = model.forward_chunk(
            ids, targets, h.detach(),
            spike_ring=tuple(r.detach() for r in ring),
            ge=ge.detach(), gi=gi.detach())
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()

    # 4. Verify all new plasticity parameters evolved
    assert not torch.equal(model.log_threshold, initial_thresh)
    assert not torch.equal(model.log_g_e, initial_ge)
    assert not torch.equal(model.log_g_i, initial_gi)

    # 5. Check diag keys
    assert "threshold_mean" in diag
    assert "g_e" in diag
    assert "g_i" in diag
    assert "firing_rate" in diag


def test_exact_exponential_integrator_properties(tmp_path):
    n = 10
    path = tmp_path / "tiny_coba_exact.npz"
    np.savez_compressed(
        path, edge_pre=np.array([0, 1], dtype=np.int32),
        edge_post=np.array([1, 0], dtype=np.int32),
        edge_weight=np.array([10.0, 10.0], dtype=np.float32),  # huge conductances!
        edge_delay=np.array([1, 1], dtype=np.int32),
        neuron_body_ids=np.arange(n, dtype=np.int64),
        nt_sign=np.array([1, -1] + [1] * 8, dtype=np.int8),
        superclass_id=np.zeros(n, dtype=np.int8),
        superclass_names=np.array(["all"]),
        meta=np.array("{}")
    )
    model = FlyReservoirLM(path, vocab_size=16, d_model=8, synapse_model="coba")

    # Under Forward Euler, G_E + G_I = 10 + 10 = 20 > 1, so (lambda - 20) would be -19 (negative ringing).
    # Under Exact Exponential Integration, alpha = exp(-g_total) in (0, 1] strictly!
    h0 = torch.full((1, n), 0.5)
    ring0 = tuple(torch.ones(1, n) for _ in range(4))
    ge0 = torch.full((1, n), 5.0)  # huge active conductance
    gi0 = torch.full((1, n), 5.0)

    token = torch.tensor([0])
    h_next, spike, ring_next, ge_next, gi_next = model.step(
        h0, token, spike_ring=ring0, ge=ge0, gi=gi0)

    # h_next must be strictly finite and non-oscillating
    assert torch.isfinite(h_next).all()
    assert (h_next >= -1.0).all() and (h_next <= 2.0).all()


def test_alif_dynamics_and_gradients(tmp_path):
    n = 30
    rng = np.random.default_rng(42)
    pre = rng.integers(0, n, 60).astype(np.int32)
    post = rng.integers(0, n, 60).astype(np.int32)
    weight = np.abs(rng.standard_normal(60)).astype(np.float32)
    nt_sign = np.array([1] * 15 + [-1] * 15, dtype=np.int8)
    superclass_id = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int8)

    path = tmp_path / "tiny_coba_alif.npz"
    np.savez_compressed(
        path, edge_pre=pre, edge_post=post, edge_weight=weight,
        edge_delay=np.ones_like(pre, dtype=np.int32),
        neuron_body_ids=np.arange(n, dtype=np.int64),
        nt_sign=nt_sign,
        superclass_id=superclass_id,
        superclass_names=np.array(["sensory", "interneuron", "output"]),
        meta=np.array("{}")
    )

    model = FlyReservoirLM(
        path, vocab_size=32, d_model=16,
        learnable_time_constants=True,
        learnable_thresholds=True,
        learnable_conductance_gains=True,
        use_alif=True,
        synapse_model="coba",
        threshold=0.08
    )

    assert model.use_alif is True
    assert hasattr(model, "log_tau_a")
    assert hasattr(model, "log_beta")
    assert model.log_tau_a.shape == (3,)
    assert model.log_beta.shape == (3,)

    # Test single-step threshold adaptation
    h0 = torch.zeros(1, n)
    token = torch.tensor([1])
    ring0 = tuple(torch.zeros(1, n) for _ in range(4))
    ge0 = torch.zeros(1, n)
    gi0 = torch.zeros(1, n)
    b0 = torch.zeros(1, n)

    h1, s1, ring1, ge1, gi1, b1 = model.step(h0, token, ring0, ge=ge0, gi=gi0, b=b0)
    assert b1.shape == (1, n)
    # Boundedness invariant: b must be in [0, 1]
    assert (b1 >= 0.0).all() and (b1 <= 1.0).all()

    # Step repeatedly and verify b stays strictly in [0, 1]
    b_curr = b1
    h_curr = h1
    ring_curr = ring1
    ge_curr = ge1
    gi_curr = gi1
    for step in range(50):
        h_curr, s_curr, ring_curr, ge_curr, gi_curr, b_curr = model.step(
            h_curr, token, ring_curr, ge=ge_curr, gi=gi_curr, b=b_curr)
        assert (b_curr >= 0.0).all() and (b_curr <= 1.0).all()

    # Test forward_chunk and gradient flow into log_tau_a and log_beta
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    ids = torch.randint(0, 32, (1, 8))
    targets = torch.randint(0, 32, (1, 8))
    h_init = torch.zeros(1, n)
    b_init = torch.zeros(1, n)
    res = model.forward_chunk(ids, targets, h_init, b=b_init)
    loss, h_final, diag, next_ring, next_ge, next_gi, next_b = res
    assert torch.isfinite(loss)
    assert "b_mean" in diag
    assert "tau_a_mean" in diag
    assert "beta_mean" in diag

    optimizer.zero_grad()
    loss.backward()
    assert model.log_tau_a.grad is not None
    assert model.log_beta.grad is not None
    assert torch.isfinite(model.log_tau_a.grad).all()
    assert torch.isfinite(model.log_beta.grad).all()
    optimizer.step()


def test_stp_biological_invariants_and_plasticity(tmp_path):
    n = 30
    rng = np.random.default_rng(777)
    pre = rng.integers(0, n, 90).astype(np.int32)
    post = rng.integers(0, n, 90).astype(np.int32)
    weight = np.abs(rng.standard_normal(90)).astype(np.float32)
    nt_sign = np.array([1] * 15 + [-1] * 15, dtype=np.int8)
    superclass_id = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int8)

    path = tmp_path / "tiny_coba_stp.npz"
    np.savez_compressed(
        path, edge_pre=pre, edge_post=post, edge_weight=weight,
        edge_delay=np.ones_like(pre, dtype=np.int32),
        neuron_body_ids=np.arange(n, dtype=np.int64),
        nt_sign=nt_sign,
        superclass_id=superclass_id,
        superclass_names=np.array(["sensory", "interneuron", "output"]),
        meta=np.array("{}")
    )

    model = FlyReservoirLM(
        path, vocab_size=32, d_model=16,
        learnable_time_constants=True,
        learnable_thresholds=True,
        learnable_conductance_gains=True,
        use_alif=True,
        use_stp=True,
        synapse_model="coba",
        threshold=0.08
    )

    assert model.use_stp is True
    assert hasattr(model, "logit_u0")
    assert hasattr(model, "log_tau_fac")
    assert hasattr(model, "log_tau_rec")
    assert model.logit_u0.shape == (3,)
    assert model.log_tau_fac.shape == (3,)
    assert model.log_tau_rec.shape == (3,)

    # 1. Step and verify (x, u) strictly bounded in [0, 1]
    h0 = torch.zeros(1, n)
    token = torch.tensor([1])
    ring0 = tuple(torch.zeros(1, n) for _ in range(4))
    ge0 = torch.zeros(1, n)
    gi0 = torch.zeros(1, n)
    b0 = torch.zeros(1, n)
    x0 = torch.ones(1, n)
    u0_init, _, _, _ = model.get_stp_params()
    u0 = u0_init.clone()

    res = model.step(h0, token, ring0, ge=ge0, gi=gi0, b=b0, x=x0, u=u0)
    h1, s1, ring1, ge1, gi1, b1, x1, u1 = res
    assert (x1 >= 0.0).all() and (x1 <= 1.0).all()
    assert (u1 >= 0.0).all() and (u1 <= 1.0).all()

    # Step repeatedly and verify invariant holds under driving inputs
    h_curr, ring_curr, ge_curr, gi_curr, b_curr, x_curr, u_curr = h1, ring1, ge1, gi1, b1, x1, u1
    for step in range(30):
        h_curr, s_curr, ring_curr, ge_curr, gi_curr, b_curr, x_curr, u_curr = model.step(
            h_curr, token, ring_curr, ge=ge_curr, gi=gi_curr, b=b_curr, x=x_curr, u=u_curr)
        assert (x_curr >= 0.0).all() and (x_curr <= 1.0).all()
        assert (u_curr >= 0.0).all() and (u_curr <= 1.0).all()
        assert (ring_curr[0] >= 0.0).all() and (ring_curr[0] <= 3.0).all()

    # 2. Test forward_chunk and gradient flow into STP parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    ids = torch.randint(0, 32, (1, 8))
    targets = torch.randint(0, 32, (1, 8))
    h_init = torch.zeros(1, n)
    b_init = torch.zeros(1, n)
    x_init = torch.ones(1, n)
    u_init = u0_init.clone()

    res_chunk = model.forward_chunk(ids, targets, h_init, b=b_init, x=x_init, u=u_init)
    loss, h_final, diag, next_ring, next_ge, next_gi, next_b, next_x, next_u = res_chunk
    assert torch.isfinite(loss)
    assert "x_mean" in diag
    assert "u_mean" in diag
    assert "u0_mean" in diag
    assert "tau_fac_mean" in diag
    assert "tau_rec_mean" in diag
    assert "pulse_mean" in diag

    optimizer.zero_grad()
    loss.backward()
    assert model.logit_u0.grad is not None
    assert model.log_tau_fac.grad is not None
    assert model.log_tau_rec.grad is not None
    assert torch.isfinite(model.logit_u0.grad).all()
    assert torch.isfinite(model.log_tau_fac.grad).all()
    assert torch.isfinite(model.log_tau_rec.grad).all()
    optimizer.step()


