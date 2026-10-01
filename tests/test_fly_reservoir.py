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
        if name.startswith(("input_proj", "output_read", "embedding", "decoder")):
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
