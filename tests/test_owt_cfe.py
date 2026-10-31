import pytest
import torch
from fine_grain.information_boltzmann.owt_cfe import OWTConstructalEngine

def test_owt_cfe_initialization():
    engine = OWTConstructalEngine(vocab_size=1000, dim=32, init_slots=16, max_slots=64, device="cpu")
    assert engine.num_active == 16
    assert engine.k_sig.shape == (64, 32)
    assert engine.c_sig.shape == (64, 32)
    norms = torch.norm(engine.k_sig[:16], dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

def test_owt_cfe_prequential_prediction():
    engine = OWTConstructalEngine(vocab_size=1000, dim=32, init_slots=16, max_slots=64, device="cpu")
    logits, sparse_idx, sparse_act, beta_star, k_in, full_act = engine.prequential_predict(current_token=42)
    assert logits.shape == (1000,)
    assert len(sparse_act) == 16
    assert beta_star > 0.0
    assert torch.isfinite(logits).all()

def test_owt_cfe_step_stream():
    engine = OWTConstructalEngine(vocab_size=1000, dim=32, init_slots=16, max_slots=64, device="cpu")
    res = engine.step_stream(current_token=42, next_token=88)
    assert res["surprisal_nats"] > 0.0
    assert res["surprisal_bits"] > 0.0
    assert res["num_slots"] >= 16
    assert res["resistance"] >= 0.0
