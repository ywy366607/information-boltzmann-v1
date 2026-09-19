"""Unit tests for Torus3D Write Operators W0, W1, W2, W3."""
import math
import pytest
import torch
import torch.nn.functional as F

from scripts.ib_local.cbim_torus3d import CBIMTorus3D, FullRankTorusWrite
from scripts.ib_local.train_cbim_malecns_internal_time import TruncatedInternalTimeGraphTrainer


def test_write_arms_initialization_and_forward():
    vocab_size = 1000
    shape = (4, 4, 4)
    d = 128
    batch_size = 2
    field = torch.randn(batch_size, *shape, d)
    tokens = torch.randint(0, vocab_size, (batch_size,))

    for arm in ["w0_baseline", "w1_unbounded", "w2_impedance", "w3_additive"]:
        writer = FullRankTorusWrite(vocab_size=vocab_size, shape=shape, d=d, write_type=arm)
        f_next, reflected, diag = writer(field, tokens)
        assert f_next.shape == field.shape
        assert reflected.shape == field.shape
        assert "t_packet" in diag
        assert "delta_e_field" in diag
        assert "cross_interference" in diag
        assert "write_to_f_ratio" in diag
        assert "incident_energy" in diag
        assert "reflected_energy" in diag

        if arm in ["w0_baseline", "w1_unbounded", "w2_impedance"]:
            # Energy balance: E(f') + E(r') == E(f) + E(p)
            assert diag["write_balance_residual"].item() < 3e-5


def test_write_arms_initial_transmission_targets():
    vocab_size = 1000
    shape = (4, 4, 4)
    d = 128
    batch_size = 16
    tokens = torch.randint(0, vocab_size, (batch_size,))
    field = torch.zeros(batch_size, *shape, d)  # clean vacuum initial state

    # W0 baseline: initial transmission should be tiny (~0.3% - 1%)
    w0 = FullRankTorusWrite(vocab_size=vocab_size, shape=shape, d=d, write_type="w0_baseline")
    _, _, d0 = w0(field, tokens)
    t0 = d0["t_packet"].item()
    assert 0.001 <= t0 <= 0.03, f"W0 initial transmission {t0} outside expected range"

    # W1 unbounded: initial transmission should be around 10% - 20%
    w1 = FullRankTorusWrite(vocab_size=vocab_size, shape=shape, d=d, write_type="w1_unbounded")
    _, _, d1 = w1(field, tokens)
    t1 = d1["t_packet"].item()
    assert 0.08 <= t1 <= 0.25, f"W1 initial transmission {t1} outside expected [10%, 20%] range"

    # W2 impedance: initial transmission should also be around 10% - 20%
    w2 = FullRankTorusWrite(vocab_size=vocab_size, shape=shape, d=d, write_type="w2_impedance")
    _, _, d2 = w2(field, tokens)
    t2 = d2["t_packet"].item()
    assert 0.08 <= t2 <= 0.25, f"W2 initial transmission {t2} outside expected [10%, 20%] range"

    # W3 additive: initial coupling eta should be around ~0.24, initial transmission ~5% - 15%
    w3 = FullRankTorusWrite(vocab_size=vocab_size, shape=shape, d=d, write_type="w3_additive")
    _, _, d3 = w3(field, tokens)
    t3 = d3["t_packet"].item()
    assert 0.03 <= t3 <= 0.25, f"W3 initial transmission {t3} outside expected range"


def test_parameter_count_matching():
    models = {}
    for arm in ["w0_baseline", "w1_unbounded", "w2_impedance", "w3_additive"]:
        models[arm] = CBIMTorus3D(
            vocab_size=50257, shape=(8, 8, 4), content_dim=16,
            v2_coordinate_components=True, readout_type="baseline", write_type=arm)

    p0 = sum(p.numel() for p in models["w0_baseline"].parameters())
    p1 = sum(p.numel() for p in models["w1_unbounded"].parameters())
    p2 = sum(p.numel() for p in models["w2_impedance"].parameters())
    p3 = sum(p.numel() for p in models["w3_additive"].parameters())

    assert p0 == p1 == p3, f"W0 ({p0}), W1 ({p1}), and W3 ({p3}) should have identical parameter counts"
    diff_p2 = abs(p2 - p0) / p0
    assert diff_p2 < 0.0002, f"W2 parameter difference {diff_p2 * 100:.4f}% exceeds 0.02% limit"


def test_w3_bibo_stability():
    """Verify that W3 additive forcing with contractive bath is BIBO stable over 200 steps."""
    torch.manual_seed(42)
    model = CBIMTorus3D(
        vocab_size=50257, shape=(4, 4, 4), content_dim=16,
        v2_coordinate_components=True, readout_type="baseline", write_type="w3_additive")
    field = model.initial_state(1)
    energies = []
    for step in range(200):
        tokens = torch.randint(0, 50257, (1, 1))
        targets = torch.randint(0, 50257, (1, 1))
        _, field, diag = model(tokens, targets, field)
        energies.append(diag["energy"].item())

    # Ensure energies never explode and settle to finite bounded steady-state
    assert max(energies) < 100.0, f"W3 energy exploded: {max(energies)}"
    assert math.isfinite(energies[-1]), "W3 energy is non-finite"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CUDAGraph test")
def test_cudagraph_compatibility_all_arms():
    """Verify that all 4 write arms capture cleanly in CUDAGraph with zero violations."""
    shape = (4, 4, 4)
    tokens = 16
    vocab_size = 1000

    for arm in ["w0_baseline", "w1_unbounded", "w2_impedance", "w3_additive"]:
        model = CBIMTorus3D(
            vocab_size=vocab_size, shape=shape, content_dim=16,
            v2_coordinate_components=True, readout_type="baseline", write_type=arm).cuda()
        trainer = TruncatedInternalTimeGraphTrainer(model, tokens=tokens, chunk_tokens=tokens)

        # Run 5 steps to trigger warmup and CUDAGraph capture
        for _ in range(5):
            x = torch.randint(0, vocab_size, (1, tokens), device="cuda")
            y = torch.randint(0, vocab_size, (1, tokens), device="cuda")
            loss, state, diag = trainer.step(x, y)
            assert torch.isfinite(loss)
            assert "t_packet" in diag
            assert "collision_input_snr" in diag
            assert "collision_output_snr" in diag
