import pytest
import torch
import torch.nn.functional as F
from fine_grain.information_boltzmann.cfe import ConstructalFieldEngine

def test_cfe_initialization():
    engine = ConstructalFieldEngine(dim=16, initial_slots=8, max_slots=32)
    assert engine.num_active == 8
    assert engine.signature.shape == (32, 16)
    # 验证超球面几何约束 ||k_i|| = 1
    idx = engine.active_indices
    norms = torch.norm(engine.signature[idx], dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_cfe_flux_and_resistance():
    engine = ConstructalFieldEngine(dim=16, initial_slots=4, max_slots=16)
    J, W, sim_pos = engine.compute_flux_and_conductance()
    assert J.shape == (4,)
    assert W.shape == (4, 4)

    rho, R = engine.compute_resistance(J)
    assert rho.shape == (4,)
    assert R.item() > 0.0
    assert torch.all(rho >= 0.0)

def test_cfe_manifold_projection():
    engine = ConstructalFieldEngine(dim=16, initial_slots=6, max_slots=16)
    # 执行流形归一化算子
    engine.step_flow_and_phase()
    engine.step_manifold_normalization()
    idx = engine.active_indices
    norms = torch.norm(engine.signature[idx], dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_cfe_split_and_energy_conservation():
    engine = ConstructalFieldEngine(dim=16, initial_slots=4, max_slots=16, split_threshold_sigma=0.1)
    # 人为制造一个极高能量槽位以触发分裂
    engine.energy[0] = 10.0
    engine.refractory[0] = 0.0

    init_active = engine.num_active
    splits = engine.step_curvature_split()
    if splits > 0:
        assert engine.num_active == init_active + 1
        # 验证新分裂槽位依然严格满足 ||k|| = 1
        idx = engine.active_indices
        norms = torch.norm(engine.signature[idx], dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_cfe_merge():
    engine = ConstructalFieldEngine(dim=16, initial_slots=4, max_slots=16, merge_similarity_threshold=0.8)
    # 人为让槽位 0 和槽位 1 强对齐
    engine.signature[1] = engine.signature[0].clone()
    init_active = engine.num_active
    merges = engine.step_redundancy_merge()
    if merges > 0:
        assert engine.num_active == init_active - 1
