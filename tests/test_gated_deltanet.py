"""Unit tests for GatedDeltaNetLM capacity-matched baseline."""
import math
import pytest
import torch

from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.cbim_torus3d import CBIMTorus3D


def test_parameter_count_matching():
    """Verify that GatedDeltaNetLM matches CBIM Torus3D non-embedding parameters within 0.1%."""
    cbim = CBIMTorus3D(shape=(8, 8, 4), velocities=8, content_dim=16)
    gdn = GatedDeltaNetLM(vocab_size=50257, d=128, layers=4, heads=4)

    cbim_non_emb = sum(
        p.numel() for n, p in cbim.named_parameters()
        if "embedding" not in n and "decoder" not in n
    )
    gdn_non_emb = sum(
        p.numel() for n, p in gdn.named_parameters()
        if "embedding" not in n and "decoder" not in n
    )

    print(f"CBIM Torus3D non-embedding: {cbim_non_emb:,}")
    print(f"GatedDeltaNetLM non-embedding: {gdn_non_emb:,}")
    diff = abs(cbim_non_emb - gdn_non_emb)
    ratio = diff / cbim_non_emb
    print(f"Difference: {diff} ({ratio:.4%})")

    assert cbim_non_emb == 332724
    assert gdn_non_emb == 332448
    assert ratio < 0.001  # Within 0.1%!


def test_forward_backward_cpu():
    """Verify CPU forward and backward execution, loss scale, and state shape."""
    model = GatedDeltaNetLM(vocab_size=1000, d=64, layers=2, heads=2)
    ids = torch.randint(0, 1000, (2, 8))
    targets = torch.randint(0, 1000, (2, 8))

    loss, next_state, diags = model(ids, targets)
    assert loss is not None
    assert math.isfinite(loss.item())
    assert next_state.shape == (2, 2, 2, 32, 32)
    assert "alpha_mean" in diags
    assert "beta_mean" in diags
    assert "state_norm" in diags

    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Missing gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU test")
def test_cuda_graph_trainer_compatibility():
    """Verify TruncatedInternalTimeGraphTrainer capture and replay on GPU."""
    from scripts.ib_local.train_cbim_malecns_internal_time import TruncatedInternalTimeGraphTrainer

    model = GatedDeltaNetLM(vocab_size=50257, d=128, layers=4, heads=4).cuda()
    trainer = TruncatedInternalTimeGraphTrainer(model, tokens=128, chunk_tokens=128)

    ids = torch.randint(0, 50257, (1, 128), device="cuda")
    targets = torch.randint(0, 50257, (1, 128), device="cuda")

    loss, state, diags = trainer.step(ids, targets)
    assert math.isfinite(float(loss))
    assert math.isfinite(float(trainer.grad_norm))
    assert 10.0 < float(loss) < 12.0  # Initial loss around ln(50257) = 10.82


def test_initial_loss_near_entropy():
    """Verify initial loss matches theoretical entropy ln(50257) = 10.82."""
    model = GatedDeltaNetLM(vocab_size=50257, d=128, layers=4, heads=4)
    ids = torch.randint(0, 50257, (1, 32))
    targets = torch.randint(0, 50257, (1, 32))

    with torch.no_grad():
        loss, _, _ = model(ids, targets)
    expected = math.log(50257)
    assert abs(loss.item() - expected) < 0.5
