"""Deterministic unit test for BiologicalTopographicWriter and FlyReservoirLM(injection='topographic')."""
import pytest
import torch
from pathlib import Path

from information_boltzmann.core.fly_reservoir import (
    FlyReservoirLM, BiologicalTopographicWriter
)

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data/malecns_v1/fly_reservoir_biological.npz"


def test_topographic_writer_forward_and_grad():
    writer = BiologicalTopographicWriter(d_model=128)
    token_emb = torch.randn(1, 128, requires_grad=True)
    h = torch.zeros(1, 165122)

    drive = writer(token_emb, h)
    assert drive.shape == (1, 165122)
    assert drive.requires_grad

    # Check non-zero drive in the 3 sensory modalities
    vis_drive = drive[:, writer.idx_vis]
    chemo_drive = drive[:, writer.idx_chemo]
    mech_drive = drive[:, writer.idx_mech]

    assert vis_drive.abs().sum() > 0
    assert chemo_drive.abs().sum() > 0
    assert mech_drive.abs().sum() > 0

    # Backprop test
    loss = drive.sum()
    loss.backward()
    assert writer.gate_linear.weight.grad is not None
    assert writer.proj_vis.weight.grad is not None
    assert writer.proj_chemo.weight.grad is not None
    assert writer.proj_mech.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for full graph test")
def test_fly_reservoir_topographic_step_and_chunk():
    model = FlyReservoirLM(
        GRAPH_PATH, vocab_size=50257, d_model=128,
        injection="topographic", read_surface="output"
    ).cuda()

    assert model.injection_mode == "topographic"
    assert model.topographic_writer is not None
    assert model.input_proj is None

    # Step test
    h = torch.zeros(1, model.n_neurons, device="cuda")
    token = torch.tensor([100], device="cuda")
    ring = tuple(torch.zeros_like(h) for _ in range(4))
    i_syn = torch.zeros_like(h)

    h_next, spike_next, ring_next, i_syn_next = model.step(h, token, ring, i_syn)
    assert h_next.shape == (1, model.n_neurons)
    assert spike_next.shape == (1, model.n_neurons)

    # Forward chunk test
    input_ids = torch.tensor([[100, 200, 300, 400]], device="cuda")
    targets = torch.tensor([[200, 300, 400, 500]], device="cuda")
    loss, h_chunk, diag, ring_chunk, i_syn_chunk = model.forward_chunk(
        input_ids, targets, h, ring, i_syn
    )
    assert loss > 0
    assert not torch.isnan(loss)
    loss.backward()

    # Check gradients on topographic writer parameters
    assert model.topographic_writer.proj_vis.weight.grad is not None
    assert model.topographic_writer.gate_linear.weight.grad is not None
