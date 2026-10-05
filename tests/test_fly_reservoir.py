"""Deterministic tests for the level-0 fly reservoir (frozen signed wiring,
LIF dynamics, trained broadcast input and weighted readout only)."""
import numpy as np
import pytest
import torch

from information_boltzmann.core.fly_reservoir import (
    FlyReservoirLM,
    participation_ratio,
)


def make_tiny(tmp_path, n=60, edges=300, seed=3):
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n, edges).astype(np.int32)
    post = rng.integers(0, n, edges).astype(np.int32)
    self_loop = pre == post
    pre, post = pre[~self_loop], post[~self_loop]
    weight = (rng.standard_normal(pre.size) * 0.1).astype(np.float32)
    path = tmp_path / "tiny_graph.npz"
    np.savez_compressed(path, edge_pre=pre, edge_post=post,
                        edge_weight=weight,
                        neuron_body_ids=np.arange(n, dtype=np.int64),
                        nt_sign=np.ones(n, dtype=np.int8),
                        superclass_id=np.zeros(n, dtype=np.int8),
                        superclass_names=np.array(["test"]),
                        meta=np.array("{}"))
    return path, pre.size


def test_frozen_wiring_and_shapes(tmp_path):
    path, edges = make_tiny(tmp_path)
    model = FlyReservoirLM(path, vocab_size=23, d_model=16)
    assert model.n_neurons == 60
    assert model.edge_weight.requires_grad is False
    h = torch.zeros(1, model.n_neurons)
    loss, h_next, diag = model.forward_chunk(
        torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]), h)
    assert loss.isfinite()
    assert h_next.shape == h.shape and torch.isfinite(h_next).all()
    assert diag["firing_rate"].shape == ()


def test_edge_weights_receive_no_gradient(tmp_path):
    path, _ = make_tiny(tmp_path)
    model = FlyReservoirLM(path, vocab_size=23, d_model=16)
    loss, _, _ = model.forward_chunk(
        torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]),
        torch.zeros(1, model.n_neurons))
    loss.backward()
    assert model.edge_weight.grad is None
    for name, parameter in model.named_parameters():
        if name.startswith(("input_proj", "output_read", "embedding", "decoder", "read_norm")):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_state_persists_between_chunks(tmp_path):
    path, _ = make_tiny(tmp_path)
    model = FlyReservoirLM(path, vocab_size=23, d_model=16)
    h0 = torch.zeros(1, model.n_neurons)
    torch.manual_seed(0)
    with torch.no_grad():
        _, h1, _ = model.forward_chunk(
            torch.tensor([[1]]), torch.tensor([[2]]), h0)
        h1_a = h1.clone()
        _, h2, _ = model.forward_chunk(
            torch.tensor([[1]]), torch.tensor([[2]]), h1)
    assert not torch.equal(h1_a, h0)
    assert not torch.equal(h2, h1_a)


def test_participation_ratio_extremes():
    assert participation_ratio(torch.ones(40)).item() == pytest.approx(40.0)
    one_hot = torch.zeros(40)
    one_hot[7] = 5.0
    assert participation_ratio(one_hot).item() == pytest.approx(1.0)
    # Branchless sentinel: zero mass is reported at the clamp ceiling.
    assert participation_ratio(torch.zeros(40)).item() == 1e9


def test_broadcast_is_not_uniform_on_heterogeneous_wiring(tmp_path):
    """On heterogeneous signed wiring a broadcast input must yield
    neuron-differentiated responses (the graph analog of '广播≠均匀')."""
    path, _ = make_tiny(tmp_path, n=60, edges=800, seed=9)
    model = FlyReservoirLM(path, vocab_size=23, d_model=16)
    h = torch.zeros(1, model.n_neurons)
    with torch.no_grad():
        for token in (1, 2, 3, 4, 5):
            h, _ = model.step(h, torch.tensor([token]))
    activity = h.abs().flatten()
    assert activity.std() > 0 and activity.max() > 0
    assert participation_ratio(activity).item() < model.n_neurons


def test_learnable_time_constants_gradients(tmp_path):
    """Test that cell-type time constants log_tau_m and log_tau_s receive valid gradients."""
    n = 30
    rng = np.random.default_rng(42)
    pre = rng.integers(0, n, 100).astype(np.int32)
    post = rng.integers(0, n, 100).astype(np.int32)
    self_loop = pre == post
    pre, post = pre[~self_loop], post[~self_loop]
    weight = (rng.standard_normal(pre.size) * 0.1).astype(np.float32)
    superclass_id = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int8)
    tau_m = np.array([10.0] * 10 + [25.0] * 10 + [50.0] * 10, dtype=np.float32)
    path = tmp_path / "tiny_cuba.npz"
    np.savez_compressed(
        path, edge_pre=pre, edge_post=post, edge_weight=weight,
        neuron_body_ids=np.arange(n, dtype=np.int64),
        nt_sign=np.ones(n, dtype=np.int8),
        superclass_id=superclass_id,
        superclass_names=np.array(["classA", "classB", "classC"]),
        tau_m=tau_m,
        meta=np.array("{}")
    )

    model = FlyReservoirLM(path, vocab_size=23, d_model=16, learnable_time_constants=True, synapse_model="cuba")
    assert model.learnable_time_constants is True
    assert model.log_tau_m.shape == (3,)
    assert model.log_tau_s.shape == (3,)
    # Verify initial values match prior
    assert torch.allclose(torch.exp(model.log_tau_m), torch.tensor([10.0, 25.0, 50.0]), rtol=1e-3)
    assert torch.allclose(torch.exp(model.log_tau_s), torch.tensor([2.5, 6.25, 12.5]), rtol=1e-3)

    h0 = torch.zeros(1, model.n_neurons)
    loss, h_next, diag = model.forward_chunk(
        torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]), h0
    )
    loss.backward()

    assert model.log_tau_m.grad is not None
    assert torch.isfinite(model.log_tau_m.grad).all()
    assert model.log_tau_s.grad is not None
    assert torch.isfinite(model.log_tau_s.grad).all()
    assert "tau_m_mean" in diag and diag["tau_m_mean"] > 0
    assert "tau_s_mean" in diag and diag["tau_s_mean"] > 0


def test_coba_learnable_time_constants_gradients(tmp_path):
    """Test COBA conductance-based LIF with dual synaptic time constants."""
    n = 30
    rng = np.random.default_rng(42)
    pre = rng.integers(0, n, 100).astype(np.int32)
    post = rng.integers(0, n, 100).astype(np.int32)
    self_loop = pre == post
    pre, post = pre[~self_loop], post[~self_loop]
    weight = (rng.standard_normal(pre.size) * 0.1).astype(np.float32)
    nt_sign = np.array([1] * 15 + [-1] * 15, dtype=np.int8)
    superclass_id = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int8)
    tau_m = np.array([10.0] * 10 + [25.0] * 10 + [50.0] * 10, dtype=np.float32)
    path = tmp_path / "tiny_coba.npz"
    np.savez_compressed(
        path, edge_pre=pre, edge_post=post, edge_weight=weight,
        edge_delay=np.ones_like(pre, dtype=np.int32),
        neuron_body_ids=np.arange(n, dtype=np.int64),
        nt_sign=nt_sign,
        superclass_id=superclass_id,
        superclass_names=np.array(["classA", "classB", "classC"]),
        tau_m=tau_m,
        meta=np.array("{}")
    )

    model = FlyReservoirLM(path, vocab_size=23, d_model=16, learnable_time_constants=True, synapse_model="coba")
    assert model.learnable_time_constants is True
    assert model.synapse_model == "coba"
    assert model.log_tau_m.shape == (3,)
    assert model.log_tau_s_e.shape == (3,)
    assert model.log_tau_s_i.shape == (3,)
    assert torch.allclose(torch.exp(model.log_tau_m), torch.tensor([10.0, 25.0, 50.0]), rtol=1e-3)
    assert torch.allclose(torch.exp(model.log_tau_s_e), torch.tensor([2.5, 6.25, 12.5]), rtol=1e-3)
    assert torch.allclose(torch.exp(model.log_tau_s_i), torch.tensor([5.0, 12.5, 25.0]), rtol=1e-3)

    h0 = torch.zeros(1, model.n_neurons)
    ring0 = tuple(torch.zeros(1, model.n_neurons) for _ in range(4))
    ge0 = torch.zeros(1, model.n_neurons)
    gi0 = torch.zeros(1, model.n_neurons)
    loss, h_next, diag, ring_next, ge_next, gi_next = model.forward_chunk(
        torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]), h0,
        spike_ring=ring0, ge=ge0, gi=gi0
    )
    loss.backward()

    assert model.log_tau_m.grad is not None and torch.isfinite(model.log_tau_m.grad).all()
    assert model.log_tau_s_e.grad is not None and torch.isfinite(model.log_tau_s_e.grad).all()
    assert model.log_tau_s_i.grad is not None and torch.isfinite(model.log_tau_s_i.grad).all()
    assert "tau_m_mean" in diag and diag["tau_m_mean"] > 0
    assert "tau_s_e_mean" in diag and diag["tau_s_e_mean"] > 0
    assert "tau_s_i_mean" in diag and diag["tau_s_i_mean"] > 0
    assert "ge_energy" in diag and "gi_energy" in diag


def test_atan_surrogate_properties():
    from information_boltzmann.core.fly_reservoir import SpikeFn
    import math

    # 1. Forward Heaviside step
    v = torch.tensor([-1.0, -0.01, 0.0, 0.01, 1.0], requires_grad=True)
    spikes = SpikeFn.apply(v)
    assert torch.equal(spikes, torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]))

    # 2. Backward ATan peak normalization: at x = 0, grad is 1.0
    spikes.sum().backward()
    expected_grad = 1.0 / (1.0 + (math.pi * v.detach()) ** 2)
    assert torch.allclose(v.grad, expected_grad, atol=1e-6)
    assert math.isclose(float(v.grad[2]), 1.0, abs_tol=1e-6)  # at x=0, exactly 1.0
    assert float(v.grad[0]) > 0.09  # heavy tail at x = -1.0


def test_learnable_thresholds_and_conductance_gains(tmp_path):
    n = 30
    rng = np.random.default_rng(42)
    pre = rng.integers(0, n, 100).astype(np.int32)
    post = rng.integers(0, n, 100).astype(np.int32)
    weight = np.abs(rng.standard_normal(100)).astype(np.float32)
    nt_sign = np.array([1] * 15 + [-1] * 15, dtype=np.int8)
    superclass_id = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int8)
    path = tmp_path / "tiny_coba_thresholds.npz"
    np.savez_compressed(
        path, edge_pre=pre, edge_post=post, edge_weight=weight,
        edge_delay=np.ones_like(pre, dtype=np.int32),
        neuron_body_ids=np.arange(n, dtype=np.int64),
        nt_sign=nt_sign,
        superclass_id=superclass_id,
        superclass_names=np.array(["classA", "classB", "classC"]),
        meta=np.array("{}")
    )

    model = FlyReservoirLM(
        path, vocab_size=23, d_model=16,
        learnable_time_constants=True,
        learnable_thresholds=True,
        learnable_conductance_gains=True,
        synapse_model="coba",
        threshold=0.15
    )

    assert model.learnable_thresholds is True
    assert model.learnable_conductance_gains is True
    assert model.log_threshold.shape == (3,)
    assert torch.allclose(torch.exp(model.log_threshold), torch.tensor([0.15, 0.15, 0.15]), atol=1e-5)
    assert model.log_g_e.shape == ()
    assert model.log_g_i.shape == ()

    h0 = torch.zeros(1, model.n_neurons)
    ring0 = tuple(torch.zeros(1, model.n_neurons) for _ in range(4))
    ge0 = torch.zeros(1, model.n_neurons)
    gi0 = torch.zeros(1, model.n_neurons)

    loss, h_next, diag, ring_next, ge_next, gi_next = model.forward_chunk(
        torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]), h0,
        spike_ring=ring0, ge=ge0, gi=gi0
    )
    loss.backward()

    assert model.log_threshold.grad is not None and torch.isfinite(model.log_threshold.grad).all()
    assert model.log_g_e.grad is not None and torch.isfinite(model.log_g_e.grad).all()
    assert model.log_g_i.grad is not None and torch.isfinite(model.log_g_i.grad).all()
    assert "threshold_mean" in diag and float(diag["threshold_mean"]) > 0
    assert "g_e" in diag and float(diag["g_e"]) > 0
    assert "g_i" in diag and float(diag["g_i"]) > 0



