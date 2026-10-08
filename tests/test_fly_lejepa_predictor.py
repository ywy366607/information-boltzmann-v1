"""Unit tests for LeJEPA Latent Predictor integration with FlyBPTTLearner."""
import pytest
import torch
import torch.nn.functional as F

from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState, FlyBPTTLearner


@pytest.fixture
def mini_fly_model(tmp_path):
    import numpy as np
    graph_path = tmp_path / "mini_fly.npz"
    N = 100
    # Create minimal synthetic connectome file for unit testing
    pre = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    post = np.array([50, 2, 3, 4, 5], dtype=np.int32)
    w = np.array([1.5, 1.5, 1.5, 1.5, 1.5], dtype=np.float32)
    np.savez(
        graph_path,
        neuron_body_ids=np.arange(N),
        edge_pre_e=pre,
        edge_post_e=post,
        edge_weight_e=w,
        delay_splits_e=[0, 2, 4, 5, 5],
        edge_pre_i=np.array([], dtype=np.int32),
        edge_post_i=np.array([], dtype=np.int32),
        edge_weight_i=np.array([], dtype=np.float32),
        delay_splits_i=[0, 0, 0, 0, 0],
        superclass_names=["cb_sensory", "cb_motor"],
        superclass_id=np.array([0] * 50 + [1] * 50, dtype=np.int32),
    )
    model = FlyReservoirLM(
        graph_path,
        vocab_size=100,
        d_model=32,
        injection="sensory",
        read_surface="all",
        synapse_model="coba",
        use_latent_predictor=True,
        lambda_sigreg=0.2,
    )
    return model


def test_fly_model_with_latent_predictor(mini_fly_model):
    model = mini_fly_model
    assert model.use_latent_predictor is True
    assert model.latent_predictor is not None
    assert model.latent_predictor.d_model == 32
    assert model.latent_predictor.d_emb == 32


def test_fly_bptt_learner_with_latent_predictor(mini_fly_model):
    model = mini_fly_model
    state = FlyPhysicalState(
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
    learner = FlyBPTTLearner(model, state, lr=1e-3, lambda_jepa=0.2)
    assert learner.lambda_jepa == 0.2
    assert any("latent_predictor" in name for name in learner.adam_names)

    # Test observe step
    tokens = torch.tensor([5, 12, 42, 7])
    learner.previous_token = 1
    scores, grad_info = learner.observe(tokens)
    assert len(scores) == 4
    assert all(torch.isfinite(torch.tensor(s)) for s in scores)
    assert "predictor_grad_norm" in grad_info
    assert grad_info["predictor_grad_norm"] > 0.0
    assert "jepa_mse" in grad_info
    assert "jepa_sigreg" in grad_info
