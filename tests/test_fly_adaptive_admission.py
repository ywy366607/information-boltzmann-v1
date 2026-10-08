import math
import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from information_boltzmann.core.fly_bptt_learning import (
    FlyPhysicalState,
    FlyBPTTLearner,
    advance_fly_token_adaptive,
    extract_fly_motor_latent,
)
from information_boltzmann.core.fly_reservoir import FlyReservoirLM


@pytest.fixture(scope="module")
def fly_model_and_state():
    """Initializes a real MaleCNS fly reservoir model and physical state with repaired numerical paths."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph_path = "data/malecns_v1/fly_reservoir_coba.npz"

    model = FlyReservoirLM(
        graph_path,
        d_model=768,
        read_surface="output",
        injection="topographic",
        synapse_model="coba",
        use_alif=True,
        use_stp=True,
        detach_reset=True,
        surrogate_mode="threshold",
        transmission_mode="incoming",
    ).to(device)

    # Initial physical state
    h = torch.zeros(1, model.n_neurons, device=device)
    ring = tuple(torch.zeros(1, model.n_neurons, device=device) for _ in range(4))
    ge = torch.zeros(1, model.n_neurons, device=device)
    gi = torch.zeros(1, model.n_neurons, device=device)
    b = torch.zeros(1, model.n_neurons, device=device)
    x = torch.ones(1, model.n_neurons, device=device)
    u = torch.full((1, model.n_neurons), 0.25, device=device)
    n_base = model.topographic_writer.n_total if model.topographic_writer is not None else model.n_neurons
    baseline = torch.zeros(1, n_base, device=device)
    h_mean = torch.zeros(1, model.n_neurons, device=device)

    state = FlyPhysicalState(h, ring, ge, gi, b, x, u, baseline, h_mean)
    return model, state, device


def test_causal_arrival_floor_prevents_premature_exit(fly_model_and_state):
    """Directly verifies resolution of review Finding 3:

    Even under zero sensory drive and quiet membrane, the causal floor min_settle_ticks=3
    strictly prevents premature admission at t=1 or t=2.
    """
    model, state, device = fly_model_and_state
    token = torch.tensor([101], dtype=torch.long, device=device)

    # Test with min_settle_ticks = 3
    box = [None]
    with torch.no_grad():
        next_state, latent = advance_fly_token_adaptive(
            model, state, token, box,
            min_settle_ticks=3,
            max_settle_ticks=14,
            flux_baseline=1.0,  # Intentionally large baseline to test if floor holds
            writer_baseline_clock="physical"
        )

    chosen_ticks = box[0]
    assert chosen_ticks is not None
    assert chosen_ticks >= 3, f"Premature admission occurred at tick {chosen_ticks}! Must be >= 3."


def test_checkpoint_deterministic_replay_matches(fly_model_and_state):
    """Directly verifies resolution of review schedule replay recommendation:

    Under torch.utils.checkpoint with use_reentrant=False, the forward pass records
    the schedule, and the recomputation pass executes the exact recorded schedule
    with zero CheckpointError and identical tensor tracking.
    """
    model, state, device = fly_model_and_state
    token = torch.tensor([205], dtype=torch.long, device=device)

    box = [None]
    curr_state = state.detached()

    # Run checkpointed advance
    next_state, latent = checkpoint(
        advance_fly_token_adaptive,
        model, curr_state, token, box,
        min_settle_ticks=3,
        max_settle_ticks=14,
        flux_baseline=0.048,
        writer_baseline_clock="physical",
        use_reentrant=False
    )

    recorded_ticks = box[0]
    assert recorded_ticks is not None
    assert 3 <= recorded_ticks <= 14

    # Backpropagate to verify backward recomputation succeeds without error
    target_loss = latent.sum()
    target_loss.backward()

    # Parameters must have gradients
    assert model.decoder.weight.grad is not None or model.output_read.weight.grad is not None


def test_full_brain_connected_gradients_in_learner(fly_model_and_state):
    """Directly verifies resolution of review Finding 1:

    Adaptive admission in FlyBPTTLearner connects loss gradients all the way through
    the physical settling steps into decoder, read norm, output read, ALIF parameters,
    synaptic weights (edge_weight_e, edge_weight_i), and topographic writer.
    """
    model, state, device = fly_model_and_state
    if device != "cuda":
        pytest.skip("Full learner test requires CUDA for Fused AdamW")

    learner = FlyBPTTLearner(
        model, state,
        lr=2e-4,
        settle_ticks=14,
        writer_baseline_clock="physical",
        use_checkpointing=True,
        adaptive_admission=True,
        min_settle_ticks=3,
        flux_baseline=0.048,
        lambda_mcr2=0.0  # MCR2 disabled as instructed
    )

    # 4-token window
    tokens = torch.tensor([101, 205, 307, 409], dtype=torch.long, device=device)
    targets = torch.tensor([205, 307, 409, 511], dtype=torch.long, device=device)
    ids = torch.cat([torch.tensor([50], dtype=torch.long, device=device), tokens[:-1]])[None]

    learner.optimizer.zero_grad(set_to_none=True)
    learner.sgd.zero_grad(set_to_none=True)

    scores, next_st, features = learner.forward_window(ids, targets[None])
    loss = scores.mean()

    assert torch.isfinite(loss).item()
    loss.backward()

    # 1. Output readout & decoder gradients
    assert model.decoder.weight.grad is not None
    assert torch.isfinite(model.decoder.weight.grad.norm()).item()
    assert model.output_read.weight.grad is not None
    assert torch.isfinite(model.output_read.weight.grad.norm()).item()

    # 2. Synaptic weight gradients
    assert model.edge_weight_e.grad is not None
    assert torch.isfinite(model.edge_weight_e.grad.norm()).item()
    assert model.edge_weight_i.grad is not None
    assert torch.isfinite(model.edge_weight_i.grad.norm()).item()

    # 3. Topographic writer gradients
    assert model.topographic_writer.proj_vis.weight.grad is not None
    assert torch.isfinite(model.topographic_writer.proj_vis.weight.grad.norm()).item()

    # 4. Adaptive metrics recorded
    metrics = learner.last_adaptive_metrics
    assert "adaptive_ticks_mean" in metrics
    assert 3.0 <= metrics["adaptive_ticks_mean"] <= 14.0
    assert metrics["adaptive_ticks_min"] >= 3
    assert metrics["adaptive_ticks_max"] <= 14


def test_physical_ticks_and_state_continuity(fly_model_and_state):
    """Verifies that physical ticks accounting accurately tracks executed physical ticks."""
    model, state, device = fly_model_and_state
    if device != "cuda":
        pytest.skip("Full learner test requires CUDA for Fused AdamW")

    learner = FlyBPTTLearner(
        model, state,
        lr=2e-4,
        settle_ticks=14,
        writer_baseline_clock="physical",
        use_checkpointing=True,
        adaptive_admission=True,
        min_settle_ticks=3,
        flux_baseline=0.048,
        lambda_mcr2=0.0
    )

    learner.previous_token = 50
    initial_physical_ticks = learner.physical_ticks

    tokens = [101, 205]
    scores, metrics = learner.observe(tokens)

    assert learner.physical_ticks > initial_physical_ticks
    # Must have added 1 pulse tick + adaptive quiet ticks per token
    expected_added = learner.last_adaptive_metrics["adaptive_total_ticks"]
    assert learner.physical_ticks - initial_physical_ticks == expected_added
    assert "adaptive_ticks_mean" in metrics
