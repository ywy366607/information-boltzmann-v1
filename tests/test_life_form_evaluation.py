"""Tests for Life-Form Tri-Pillar Biological Evaluation Suite & Thermodynamic Entropy Auditor."""
import pytest
import numpy as np
import torch

from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.life_form_evaluation import (
    PrequentialStreamTracker,
    ThermodynamicEntropyAuditor,
    StreamingConnectomeLearner,
    EnvironmentalShockEvaluator,
    ContinuousLifecycleSavingsEvaluator,
    ReAdaptationResult,
    SavingsResult,
)


def test_prequential_stream_tracker():
    """Verifies PrequentialStreamTracker cumulative, EMA, and windowed NLL tracking."""
    tracker = PrequentialStreamTracker(window_size=5, ema_alpha=0.1)

    losses = [8.0, 7.5, 7.0, 6.5, 6.0, 5.5]
    for loss in losses:
        tracker.update(loss)

    assert tracker.total_tokens == 6
    assert np.isclose(tracker.mean_cumulative_nll, np.mean(losses))
    assert np.isclose(tracker.window_mean_nll, np.mean(losses[1:]))
    assert tracker.ema_nll < 7.5

    summary = tracker.summary()
    assert summary["total_tokens"] == 6.0
    assert "ema_nll" in summary
    assert "window_mean_nll" in summary


def test_thermodynamic_entropy_auditor():
    """Verifies ThermodynamicEntropyAuditor tracks inflow, dissipation, and SNR."""
    auditor = ThermodynamicEntropyAuditor(window_size=50, d_model=16)

    # Simulate 40 steps of noisy early phase then coherent late phase
    torch.manual_seed(42)
    for t in range(40):
        nll = 8.5 - 0.05 * t  # Decreasing surprise
        biophysics = {"g_total": torch.tensor([1.2])}
        # In early phase, z has high noise; in late phase, z has strong coherent signal
        signal = torch.sin(torch.tensor([float(t) / 5.0])) * torch.ones(16)
        noise = torch.randn(16) * (0.8 if t < 20 else 0.1)
        z = signal + noise
        auditor.record_step(nll=nll, biophysics=biophysics, z_vector=z, weight_norm_sq=10.0)

    summary = auditor.summary()
    assert summary.total_steps == 40
    assert summary.cumulative_entropy_inflow > 0.0
    assert summary.cumulative_entropy_dissipated > 0.0
    assert summary.mean_inflow_rate > 0.0
    assert summary.mean_dissipation_rate > 0.0
    assert summary.spectral_entropy > 0.0
    assert summary.spectral_entropy <= summary.max_possible_spectral_entropy
    assert summary.signal_power > 0.0
    assert summary.noise_power > 0.0
    # The transition from high noise to low noise should yield positive delta SNR
    assert summary.delta_snr_db > 0.0, "Noise expulsion should produce positive delta SNR"


@pytest.fixture
def tiny_fly_model(tmp_path):
    n = 20
    rng = np.random.default_rng(42)
    pre = rng.integers(0, n, 40).astype(np.int32)
    post = rng.integers(0, n, 40).astype(np.int32)
    weight = np.abs(rng.standard_normal(40)).astype(np.float32)
    nt_sign = np.array([1] * 20 + [-1] * 20, dtype=np.int8)
    superclass_id = np.array([0] * 7 + [1] * 7 + [2] * 6, dtype=np.int8)

    path = tmp_path / "tiny_eval_model.npz"
    np.savez_compressed(
        path,
        edge_pre=pre, edge_post=post, edge_weight=weight,
        edge_delay=np.ones_like(pre, dtype=np.int32),
        neuron_body_ids=np.arange(n, dtype=np.int64),
        nt_sign=nt_sign,
        superclass_id=superclass_id,
        superclass_names=np.array(["sensory", "interneuron", "output"]),
        meta=np.array("{}"),
    )

    model = FlyReservoirLM(
        path, vocab_size=32, d_model=16,
        learnable_time_constants=True,
        learnable_thresholds=True,
        learnable_conductance_gains=True,
        use_alif=True,
        use_stp=True,
        synapse_model="coba",
        threshold=0.08,
    )
    return model


def test_streaming_connectome_learner(tiny_fly_model):
    """Verifies that StreamingConnectomeLearner performs valid steps, updates weights, and forks."""
    model = tiny_fly_model
    learner = StreamingConnectomeLearner(model=model, lr=1e-3, grad_accum_tokens=2)

    tok_in = torch.tensor([5])
    tok_tgt = torch.tensor([12])

    w_before = model.output_read.weight.clone()
    res = learner.step(tok_in, tok_tgt, learn=True)

    assert "loss" in res
    assert np.isfinite(res["loss"])
    assert "step_entropy" in res
    assert "biophysics" in res
    assert res["z_latent"].shape == (1, 16)

    # Step again to trigger optimizer accumulation
    tok_in2 = torch.tensor([12])
    tok_tgt2 = torch.tensor([7])
    learner.step(tok_in2, tok_tgt2, learn=True)
    w_after = model.output_read.weight

    assert not torch.allclose(w_before, w_after), "Weights should update after grad_accum_tokens steps"

    # Verify fork
    forked = learner.fork()
    assert torch.allclose(forked.h, learner.h)
    assert forked.h.data_ptr() != learner.h.data_ptr()


def test_environmental_shock_evaluator_unified(tiny_fly_model):
    """Verifies that plastic shock evaluation invokes the unified learner correctly."""
    model = tiny_fly_model
    learner = StreamingConnectomeLearner(model=model, lr=1e-3, grad_accum_tokens=2)

    novel_tokens = torch.randint(0, 32, (30,))
    result = EnvironmentalShockEvaluator.evaluate_shock(learner, novel_tokens)

    assert isinstance(result, ReAdaptationResult)
    assert result.n_tokens == 29
    assert len(result.trajectory) == 29
    assert np.isfinite(result.shock_surprise)
    assert np.isfinite(result.plateau_surprise)
    assert result.half_life_tokens >= 0.0
    assert result.elasticity_score >= 0.0
    assert result.thermodynamics.total_steps == 29


def test_continuous_lifecycle_savings_unbroken(tiny_fly_model):
    """Verifies true unbroken A -> B -> A continuous lifetime savings without state resets."""
    model = tiny_fly_model
    learner = StreamingConnectomeLearner(model=model, lr=1e-3, grad_accum_tokens=2)

    tokens_a = torch.randint(0, 32, (15,))
    tokens_b = torch.randint(0, 32, (25,))

    result = ContinuousLifecycleSavingsEvaluator.evaluate_unbroken_lifecycle(
        learner=learner,
        tokens_a=tokens_a,
        tokens_b=tokens_b,
    )

    assert isinstance(result, SavingsResult)
    assert result.n_tokens == 14
    assert result.intervening_tokens == 24
    assert len(result.initial_trajectory) == 14
    assert len(result.relearn_trajectory) == 14
    assert np.isfinite(result.nll_initial_mean)
    assert np.isfinite(result.nll_relearn_mean)
    assert np.isfinite(result.savings_ratio)
    assert "delta_snr_db" in result.thermodynamic_shift


def test_four_pillar_evaluator(tiny_fly_model):
    """Verifies that FourPillarEvaluator computes all 4 pillars and ledgers cleanly."""
    from information_boltzmann.core.life_form_evaluation import FourPillarEvaluator, FourPillarEvaluationResult
    model = tiny_fly_model
    learner = StreamingConnectomeLearner(model=model, lr=1e-3, grad_accum_tokens=2)

    val_tokens = torch.randint(0, 32, (100,))
    result = FourPillarEvaluator.evaluate(
        learner=learner,
        val_tokens=val_tokens,
        cursor=0,
        shock_tokens=20,
        ebbinghaus_seq_a_tokens=10,
        ebbinghaus_intervene_tokens=15,
        ftle_steps=10,
    )

    assert isinstance(result, FourPillarEvaluationResult)
    # Pillar 1
    assert "total_tokens" in result.pillar_1_prequential
    # Pillar 2
    assert result.pillar_2_shock.n_tokens == 20
    assert np.isfinite(result.pillar_2_shock.shock_surprise)
    # Pillar 3
    assert result.pillar_3_ebbinghaus.n_tokens == 10
    assert np.isfinite(result.pillar_3_ebbinghaus.savings_ratio)
    # Pillar 4: Energy Ledger
    assert result.energy_ledger.cumulative_spikes >= 0
    assert result.energy_ledger.mean_spikes_per_token >= 0.0
    assert np.isfinite(result.energy_ledger.information_efficiency_nats_per_100k_spikes)
    # Pillar 4: Information Ledger
    assert result.information_ledger.effective_rank >= 1.0
    assert result.information_ledger.participation_ratio >= 1.0
    assert result.information_ledger.temporal_roughness >= 0.0
    # Pillar 4: Criticality & Reverberation
    assert np.isfinite(result.criticality.branching_ratio)
    assert np.isfinite(result.criticality.finite_time_lyapunov_exponent)
    assert result.criticality.regime in [
        "Subcritical Reverberating (Biologically Optimal Echo, Wilting & Priesemann 2018)",
        "Exact Critical Boundary (Edge-of-Chaos)",
        "Strongly Damped / Over-Dissipative",
        "Supercritical / Epileptiform Runaway",
    ]
    # Pillar 4: Convergence
    assert np.isfinite(result.convergence.prequential_loss_mean)
    assert np.isfinite(result.convergence.trend_slope_beta)
    assert np.isfinite(result.convergence.drift_to_noise_ratio_dnr)

    # Test dictionary export
    res_dict = result.to_dict()
    assert "pillar_1_prequential_tracker" in res_dict
    assert "energy_ledger" in res_dict
    assert "information_ledger" in res_dict
    assert "criticality_reverberation" in res_dict
    assert "convergence_audit" in res_dict

