"""Deterministic tensor and interface tests for Solution 1 Continuous Gamma-trace Readout.

Verifies:
1. Shape, initialization, and sigmoid bounds of logit_read_gamma.
2. DC gain normalization of the 2nd-order Erlang filter (sum of impulse weights = 1.0).
3. Smooth autograd backpropagation to logit_read_gamma and motor projections.
4. Window splitting equivalence: running two half-windows with detached physical
   state produces identical scores and terminal states to the full window.
5. Physical settle ticks integration: continuous wave accumulation without dropping ticks.
"""
import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from information_boltzmann.core.fly_reservoir import FlyReservoirLM, BiologicalTopographicWriter
from information_boltzmann.core.fly_bptt_learning import (
    FlyPhysicalState, FlyBPTTLearner, advance_fly_input_event, predict_fly_next,
)


def make_gamma_model(tmp_path, use_gamma=True, init_gamma=0.7):
    graph = tmp_path / "graph.npz"
    np.savez(
        graph,
        neuron_body_ids=np.arange(12),
        edge_pre=np.arange(12),
        edge_post=np.roll(np.arange(12), 1),
        edge_weight=np.full(12, 0.1, dtype=np.float32),
        nt_sign=np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]),
        edge_delay=np.ones(12, dtype=np.int32),
        superclass_id=np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]),
        superclass_names=np.array(["cb_sensory", "cb_motor"]),
    )
    parts = tmp_path / "parts.npz"
    np.savez(parts, visual_idx=np.array([0]), chemo_idx=np.array([1]), mechano_idx=np.array([2]))
    model = FlyReservoirLM(
        graph,
        vocab_size=32,
        d_model=8,
        injection="sensory",
        read_surface="output",
        synapse_model="coba",
        use_alif=True,
        use_stp=True,
        read_centering=True,
        use_read_gamma_trace=use_gamma,
        init_read_gamma=init_gamma,
    )
    model.topographic_writer = BiologicalTopographicWriter(8, parts)
    model.injection_mode = "topographic"
    return model


def make_initial_state(model):
    h = torch.rand(1, model.n_neurons) * 0.1
    u = model.get_stp_params()[0].detach().expand_as(h).clone()
    return FlyPhysicalState(
        h=h,
        ring=tuple(h.clone() for _ in range(4)),
        ge=h.clone(),
        gi=h.clone(),
        b=h.clone(),
        x=torch.ones_like(h),
        u=u,
        baseline=torch.zeros(1, model.topographic_writer.n_total),
        h_mean=torch.zeros_like(h),
    )


def test_gamma_decay_initialization_and_bounds(tmp_path):
    model = make_gamma_model(tmp_path, use_gamma=True, init_gamma=0.7)
    assert model.use_read_gamma_trace is True
    assert model.logit_read_gamma.shape == (1, model.n_read)
    decay = model.get_read_gamma_decay()
    assert decay.shape == (1, model.n_read)
    torch.testing.assert_close(decay, torch.full((1, model.n_read), 0.7), atol=1e-5, rtol=1e-5)
    
    # Boundary clamping check
    model.logit_read_gamma.data.fill_(100.0)
    assert float(model.get_read_gamma_decay().detach().max()) <= 0.995 + 1e-6
    assert float(model.get_read_gamma_decay().detach().max()) >= 0.995 - 1e-6
    model.logit_read_gamma.data.fill_(-100.0)
    assert float(model.get_read_gamma_decay().detach().min()) >= 0.005 - 1e-6
    assert float(model.get_read_gamma_decay().detach().min()) <= 0.005 + 1e-6


def test_gamma_impulse_response_dc_gain_normalization():
    """Verify that 2nd-order Erlang discrete filter has exact DC gain = 1.0."""
    gamma = 0.75
    z1, z2 = 0.0, 0.0
    accumulated = 0.0
    for t in range(200):
        pulse = 1.0 if t == 0 else 0.0
        z1 = gamma * z1 + (1.0 - gamma) * pulse
        z2 = gamma * z2 + (1.0 - gamma) * z1
        accumulated += z2
    assert abs(accumulated - 1.0) < 1e-4


def test_gamma_trace_gradients_and_parameter_coverage(tmp_path):
    torch.manual_seed(42)
    model = make_gamma_model(tmp_path, use_gamma=True, init_gamma=0.7)
    initial = make_initial_state(model)
    learner = FlyBPTTLearner(model, initial, settle_ticks=2)
    
    assert "logit_read_gamma" in learner.adam_names
    
    ids = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4]])
    
    scores, final_state, features = learner.forward_window(ids, targets)
    assert scores.shape == (4,)
    assert features.shape == (4, 8)
    assert final_state.gamma_z1.shape == (1, model.n_read)
    assert final_state.gamma_z2.shape == (1, model.n_read)
    
    loss = scores.mean()
    loss.backward()
    
    assert model.logit_read_gamma.grad is not None
    assert torch.isfinite(model.logit_read_gamma.grad).all()
    assert model.logit_read_gamma.grad.abs().sum() > 0
    assert model.output_read.weight.grad is not None
    assert model.output_read.weight.grad.abs().sum() > 0


def test_gamma_window_continuation_and_split_equivalence(tmp_path):
    torch.manual_seed(99)
    model = make_gamma_model(tmp_path, use_gamma=True, init_gamma=0.7)
    initial = make_initial_state(model)
    learner = FlyBPTTLearner(model, initial, settle_ticks=1)
    
    ids = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4]])
    
    # 1. Full window run
    scores_full, state_full, feats_full = learner.forward_window(ids, targets)
    
    # 2. Split window run with detached state handover
    scores_part1, state_mid, feats_part1 = learner.forward_window(ids[:, :2], targets[:, :2])
    learner.state = state_mid.detached()
    scores_part2, state_split, feats_part2 = learner.forward_window(ids[:, 2:], targets[:, 2:])
    
    torch.testing.assert_close(torch.cat((scores_part1, scores_part2)), scores_full, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.cat((feats_part1, feats_part2)), feats_full, atol=1e-5, rtol=1e-5)
    
    # Terminal physical state match (including gamma_z1 and gamma_z2)
    torch.testing.assert_close(state_split.gamma_z1, state_full.gamma_z1, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(state_split.gamma_z2, state_full.gamma_z2, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(state_split.h, state_full.h, atol=1e-5, rtol=1e-5)


def test_predict_fly_next_matches_bptt_scoring(tmp_path):
    torch.manual_seed(77)
    model = make_gamma_model(tmp_path, use_gamma=True, init_gamma=0.7)
    initial = make_initial_state(model)
    learner = FlyBPTTLearner(model, initial, settle_ticks=1)
    
    token = torch.tensor([0])
    target = torch.tensor([1])
    
    scores, next_state, _ = learner.forward_window(token[None], target[None])
    logits, pred_state = predict_fly_next(model, initial, token, settle_ticks=1,
                                         writer_baseline_clock='input')
    pred_score = F.cross_entropy(logits, target)
    torch.testing.assert_close(scores[0], pred_score, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(pred_state.gamma_z2, next_state.gamma_z2, atol=1e-5, rtol=1e-5)


def test_anatomical_spatial_gamma_initialization(tmp_path):
    graph = tmp_path / "spatial_graph.npz"
    # Create 3 sensory neurons at x=0, and 3 motor neurons at x=10, 50, 100 um
    coords = np.zeros((6, 3), dtype=np.float32)
    coords[0:3, 0] = 0.0    # sensory center at x=0
    coords[3, 0] = 10.0     # close motor
    coords[4, 0] = 50.0     # medium motor
    coords[5, 0] = 100.0    # distal motor
    np.savez(
        graph,
        neuron_body_ids=np.arange(6),
        coords_um=coords,
        edge_pre=np.array([0, 1, 2, 3, 4]),
        edge_post=np.array([1, 2, 3, 4, 5]),
        edge_weight=np.full(5, 0.1, dtype=np.float32),
        nt_sign=np.array([1, 1, 1, 1, 1]),
        edge_delay=np.ones(5, dtype=np.int32),
        superclass_id=np.array([0, 0, 0, 1, 1, 1]),
        superclass_names=np.array(["cb_sensory", "cb_motor"]),
    )
    parts = tmp_path / "parts.npz"
    np.savez(parts, visual_idx=np.array([0]), chemo_idx=np.array([1]), mechano_idx=np.array([2]))
    model = FlyReservoirLM(
        graph,
        vocab_size=32,
        d_model=8,
        injection="sensory",
        read_surface="output",
        synapse_model="coba",
        use_read_gamma_trace=True,
        init_read_gamma="anatomical",
    )
    model.topographic_writer = BiologicalTopographicWriter(8, parts)
    model.injection_mode = "topographic"

    assert model.n_read == 3
    assert model.logit_read_gamma.shape == (1, 3)
    decays = model.get_read_gamma_decay().detach().cpu().numpy()[0]
    # Verify monotonic ordering: closer motor neuron has smaller tau/gamma, distal has larger
    assert decays[0] < decays[1] < decays[2]
    # Boundaries: tau in [1.0, 115.0], gamma between ~0.50 and ~0.992
    assert 0.49 <= decays[0] <= 0.51
    assert 0.985 <= decays[2] <= 0.995

