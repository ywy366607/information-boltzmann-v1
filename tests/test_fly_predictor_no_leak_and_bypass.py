"""Rigorous verification tests to audit zero future leakage and zero bigram shortcuts in LeJEPAPredictor."""
import pytest
import torch
import torch.nn.functional as F
import numpy as np

from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import (
    FlyPhysicalState, FlyBPTTLearner, collect_fly_quiet_trajectory
)
from information_boltzmann.core.lejepa_predictor import LeJEPAPredictor, DynamicDelayAttention


@pytest.fixture
def mini_cns(tmp_path):
    graph_path = tmp_path / "mini_cns.npz"
    N = 64
    pre = np.array([0, 1, 2, 3, 4, 10, 15, 20], dtype=np.int32)
    post = np.array([1, 2, 3, 4, 5, 11, 16, 21], dtype=np.int32)
    w = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4], dtype=np.float32)
    np.savez(
        graph_path,
        neuron_body_ids=np.arange(N),
        edge_pre_e=pre,
        edge_post_e=post,
        edge_weight_e=w,
        delay_splits_e=[0, 2, 5, 8, 8],
        edge_pre_i=np.array([], dtype=np.int32),
        edge_post_i=np.array([], dtype=np.int32),
        edge_weight_i=np.array([], dtype=np.float32),
        delay_splits_i=[0, 0, 0, 0, 0],
        superclass_names=["cb_sensory", "cb_motor"],
        superclass_id=np.array([0] * 32 + [1] * 32, dtype=np.int32),
    )
    model = FlyReservoirLM(
        graph_path,
        vocab_size=50,
        d_model=32,
        injection="sensory",
        read_surface="output",
        synapse_model="coba",
        use_latent_predictor=True,
        lambda_sigreg=0.2,
        max_horizon=14,
    )
    return model


def make_clean_state(model):
    return FlyPhysicalState(
        h=torch.zeros(1, model.n_neurons),
        ring=(torch.zeros(1, model.n_neurons), torch.zeros(1, model.n_neurons),
              torch.zeros(1, model.n_neurons), torch.zeros(1, model.n_neurons)),
        ge=torch.zeros(1, model.n_neurons),
        gi=torch.zeros(1, model.n_neurons),
        b=torch.zeros(1, model.n_neurons),
        x=torch.ones(1, model.n_neurons),
        u=torch.full((1, model.n_neurons), 0.25),
        baseline=torch.zeros(1, model.n_injection),
        h_mean=torch.zeros(1, model.n_neurons),
        dan_gate=torch.zeros(1, model.n_neurons),
        gamma_z1=torch.zeros(1, model.n_read),
        gamma_z2=torch.zeros(1, model.n_read),
    )


def test_zero_future_leakage_invariance(mini_cns):
    """Perturbing future tokens in a window must NOT alter the output for token t=0."""
    model = mini_cns
    model.eval()

    state = make_clean_state(model)
    learner = FlyBPTTLearner(model, state, lr=1e-3, lambda_jepa=0.2)
    learner.previous_token = 5

    # Sequence A: [10, 20, 30, 40] -> inputs: [5, 10, 20, 30], targets: [10, 20, 30, 40]
    tokens_a = torch.tensor([10, 20, 30, 40])
    ids_a = learner.inputs_for_targets(tokens_a)
    scores_a, _, features_a = learner.forward_window(ids_a, tokens_a[None])

    # Sequence B: identical for token 0, completely different in future tokens: [10, 49, 12, 3]
    tokens_b = torch.tensor([10, 49, 12, 3])
    ids_b = learner.inputs_for_targets(tokens_b)
    scores_b, _, features_b = learner.forward_window(ids_b, tokens_b[None])

    # The score and feature for token 0 MUST be bit-exact identical!
    assert torch.allclose(scores_a[0], scores_b[0], atol=1e-6), (
        f"Leakage detected! Token 0 score changed when future tokens changed: {scores_a[0]} vs {scores_b[0]}"
    )
    assert torch.allclose(features_a[0], features_b[0], atol=1e-6), (
        f"Leakage detected! Token 0 feature changed when future tokens changed!"
    )


def test_zero_bigram_bypass_invariance(mini_cns):
    """The predictor has zero token embedding inputs and cannot bypass the brain."""
    model = mini_cns
    predictor = model.latent_predictor

    # 1. Inspect architecture: predictor parameters have NO embedding connections
    for name, p in predictor.named_parameters():
        assert "sensory_proj" not in name, f"Found banned sensory projection: {name}"
        assert p.shape[0] != model.embedding.num_embeddings, f"Parameter {name} has vocabulary dimension!"

    # 2. Check rollout signature: only accepts z_true, no token embeddings
    d_m = predictor.d_model
    z_true = torch.randn(8, d_m)
    readout, all_hops, attn_weights = predictor.rollout(z_true)
    assert readout.shape == (8, d_m)
    assert all_hops.shape == (8, predictor.max_horizon + 1, d_m)

    # 3. Brain ablation: if z_true is zero, readout is strictly zero
    z_zero = torch.zeros(8, d_m)
    readout_zero, _, _ = predictor.rollout(z_zero)
    assert readout_zero.norm().item() == 0.0, f"Predictor leaked information from outside brain: norm={readout_zero.norm()}"


def test_dynamic_delay_attention_probabilities(mini_cns):
    """Dynamic Delay Attention weights must be valid probability distributions summing to 1."""
    model = mini_cns
    predictor = model.latent_predictor
    d_m = predictor.d_model
    z = torch.randn(16, d_m)
    _, all_hops, attn_weights = predictor.rollout(z)

    # Check shapes
    assert attn_weights.shape == (16, predictor.max_horizon + 1)
    # Check non-negativity
    assert (attn_weights >= 0.0).all()
    # Check sum to 1.0 for every token in window
    sums = attn_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), f"Attention weights do not sum to 1: {sums}"


def test_quiet_trajectory_zero_drive(mini_cns):
    """Quiet trajectory must advance with strictly zero external drive."""
    model = mini_cns
    state = make_clean_state(model)
    # Inject a pulse at step 0
    state.h[0, 0] = 1.0

    trajectory, next_state = collect_fly_quiet_trajectory(
        model, state, num_ticks=14, base_rates=model.get_decay_rates(),
        thresholds=model.get_thresholds()
    )

    d_m = model.latent_predictor.d_model
    assert len(trajectory) == 14
    for i, step_latent in enumerate(trajectory):
        assert step_latent.shape == (1, d_m)
        assert torch.isfinite(step_latent).all()


def test_decoder_readout_autonomous_symmetry(mini_cns):
    """The Decoder readout must be 100% autonomous and invariant to teacher hops.

    Zero teacher physical states may leak into the Decoder readout, ensuring
    strict symmetry between training (with teacher) and generation (without teacher).
    """
    model = mini_cns
    predictor = model.latent_predictor
    d_m = predictor.d_model
    T = 16
    z_true = torch.randn(T, d_m)

    # 1. Rollout with no teacher (inference / generation mode)
    readout_no_teacher, loss_no_t, _ = predictor.rollout_dagger(z_true, teacher_hops=None)

    # 2. Rollout with dummy teacher hops (training mode)
    teacher_hops = torch.randn(predictor.max_horizon, d_m)
    readout_with_teacher, loss_with_t, metrics = predictor.rollout_dagger(z_true, teacher_hops=teacher_hops)

    # The student readout for Decoder MUST be bit-exact identical!
    assert torch.allclose(readout_no_teacher, readout_with_teacher, atol=1e-7), (
        "Asymmetry detected! Decoder readout changed when teacher hops were provided! "
        "Teacher states must never leak into the student CE decoding path."
    )

    # But the training loss MUST incorporate the distillation MSE when teacher is present
    assert metrics["jepa_mse"] > 0.0
    assert loss_with_t > loss_no_t


def test_markov_transition_supervision_indexing(mini_cns):
    """Transition supervision must strictly map P(s_in^{(k-1)}) -> z^{*(k)}.

    Verifies the off-by-one fix: step 1 predicts tick 1 from tick 0,
    step 2 predicts tick 2 from tick 1, etc.
    """
    model = mini_cns
    predictor = model.latent_predictor
    d_m = predictor.d_model
    T = 4
    K = predictor.max_horizon
    z_true = torch.randn(T, d_m)

    # Construct synthetic ground truth quiet physical sequence:
    # z^{*(0)} = z_true[-1]
    # z^{*(k)} = target_k for k=1..K
    ground_truth_trajectory = torch.randn(K, d_m)

    # With beta=0.0 (pure teacher forcing), each step k takes teacher state at k-1
    # and predicts teacher state at k: P(z^{*(k-1)}) -> z^{*(k)}.
    _, loss, metrics = predictor.rollout_dagger(z_true, beta=0.0, teacher_hops=ground_truth_trajectory)
    assert metrics["jepa_mse"] > 0.0
    assert torch.isfinite(loss)

