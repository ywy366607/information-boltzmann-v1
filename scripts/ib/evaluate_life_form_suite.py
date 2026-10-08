"""CLI entry point for Life-Form Tri-Pillar Biological Evaluation Suite & Thermodynamic Entropy Auditor.

Evaluates trained living connectome checkpoints on:
  1. Pillar 1: Non-Repeating Continuous Stream Prequential Surprise
  2. Pillar 2: Plastic Environmental Shock & Re-adaptation Half-Life (t_{1/2})
     - Evaluates homeostatic elasticity via unified StreamingConnectomeLearner
       with plasticity ACTIVE.
  3. Pillar 3: Unbroken Continuous Lifecycle Ebbinghaus Savings (A -> B -> A)
     - True physical lifetime experience across N_intervene intervening tokens.
     - Compares initial encounter vs review curves, savings ratio, and acceleration.
  4. Thermodynamic Entropy & Noise Expulsion Dynamics:
     - Measures accumulated inflow entropy S_in vs dissipated entropy S_diss.
     - Quantifies net entropy balance Delta S and spectral representation entropy.
     - Analyzes Signal Power, Noise Power, and SNR shift Delta SNR_dB, determining
       whether the open system expels noise.

Usage:
  python scripts/ib/evaluate_life_form_suite.py \
      --checkpoint results/q8_fly_infinite_stream_owt/best.pt \
      --data data/ib_owt_gpt2 \
      --output results/published/q8_fly_infinite_stream_life_form_evaluation.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.life_form_evaluation import (
    ContinuousLifecycleSavingsEvaluator,
    EnvironmentalShockEvaluator,
    PrequentialStreamTracker,
    StreamingConnectomeLearner,
    ThermodynamicEntropyAuditor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("results/q8_fly_infinite_stream_owt/best.pt"))
    parser.add_argument("--graph", type=Path,
                        default=Path("data/malecns_v1/fly_reservoir_coba.npz"))
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/published/q8_fly_infinite_stream_life_form_evaluation.json"))
    parser.add_argument("--warm-tokens", type=int, default=1000,
                        help="Tokens to warm-in living physical dynamics")
    parser.add_argument("--shock-tokens", type=int, default=1000,
                        help="Sequence length for plastic environmental shock evaluation")
    parser.add_argument("--relearn-tokens", type=int, default=500,
                        help="Sequence length for Sequence A in Ebbinghaus savings")
    parser.add_argument("--intervening-tokens", type=int, default=3000,
                        help="Actual physical tokens in intervening stream B (A -> B -> A)")
    parser.add_argument("--d-model", type=int, default=768,
                        help="Dimensionality of readout/embedding (auto-detected from checkpoint if present)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-accum-tokens", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model = args.d_model
    saved = None
    if args.checkpoint.exists():
        saved = torch.load(args.checkpoint, map_location="cpu")
        if "config" in saved and "d_model" in saved["config"]:
            d_model = saved["config"]["d_model"]
        elif "model" in saved and "output_read.weight" in saved["model"]:
            d_model = saved["model"]["output_read.weight"].shape[0]

    print(f"Loading MaleCNS connectome from {args.graph} on {device} (d_model={d_model})...")
    model = FlyReservoirLM(
        args.graph,
        vocab_size=50257,
        d_model=d_model,
        injection="topographic",
        read_surface="output",
        synapse_model="coba",
        use_alif=True,
        use_stp=True,
    ).to(device)

    if saved is not None:
        print(f"Loading checkpoint weights from {args.checkpoint}...")
        graph_keys = {
            "edge_index", "edge_pre", "edge_post", "edge_weight", "delay_splits",
            "edge_pre_e", "edge_post_e", "edge_weight_e", "delay_splits_e",
            "edge_pre_i", "edge_post_i", "edge_weight_i", "delay_splits_i",
            "dan_edge_pre", "dan_edge_post", "dan_edge_weight", "lambda_0"
        }
        model_dict = {}
        for k, v in saved["model"].items():
            if k in graph_keys:
                continue
            if k == "output_read.weight" and v.shape != model.output_read.weight.shape:
                if hasattr(model, "read_indices") and v.shape[-1] == model.n_neurons:
                    v = v[:, model.read_indices.cpu()]
            model_dict[k] = v
        model.load_state_dict(model_dict, strict=False)
        val_nll_val = saved.get("best_validation_nll", saved.get("best_live_nll", "N/A"))
        val_str = f"{val_nll_val:.4f}" if isinstance(val_nll_val, (int, float)) else str(val_nll_val)
        print(f"Checkpoint loaded (Stream Tokens: {saved.get('tokens_streamed', 'N/A')}, Val NLL: {val_str})")

    # Instantiate unified online learner
    learner = StreamingConnectomeLearner(
        model=model,
        lr=args.lr,
        grad_accum_tokens=args.grad_accum_tokens,
    )

    train_data = np.load(args.data / "train.npy", mmap_mode="r")
    val_data = np.load(args.data / "validation.npy", mmap_mode="r")

    print("\n================================================================================")
    print("PHASE 0: CONTINUOUS WARM-IN & THERMODYNAMIC CALIBRATION")
    print("================================================================================")
    print(f"Streaming {args.warm_tokens} tokens from training stream to establish dynamic equilibrium...")
    warm_slice = torch.from_numpy(np.array(train_data[2000:2000 + args.warm_tokens + 1], dtype=np.int64)).to(device)
    for t in range(args.warm_tokens):
        learner.step(warm_slice[t:t + 1], warm_slice[t + 1:t + 2], learn=True)

    calib_thermo = learner.auditor.summary()
    print(f"  >>> Dynamic State Established: EMA Loss = {learner.tracker.ema_nll:.4f}")
    print(f"  >>> Initial SNR:                {calib_thermo.snr_db:.2f} dB")
    print(f"  >>> Spectral Entropy:           {calib_thermo.spectral_entropy:.4f} / {calib_thermo.max_possible_spectral_entropy:.4f} nats")
    print(f"  >>> Mean Dissipation Rate:      {calib_thermo.mean_dissipation_rate:.4f} nats/step")

    print("\n================================================================================")
    print("PILLAR 2: PLASTIC ENVIRONMENTAL SHOCK & RE-ADAPTATION HALF-LIFE (t_{1/2})")
    print("================================================================================")
    print(f"Forking living organism into novel validation stream ({args.shock_tokens} tokens) with Plasticity ON...")
    shock_branch = learner.fork()
    shock_tokens = torch.from_numpy(np.array(val_data[:args.shock_tokens + 1], dtype=np.int64)).to(device)

    shock_result = EnvironmentalShockEvaluator.evaluate_shock(
        learner=shock_branch,
        tokens=shock_tokens,
    )

    print(f"  >>> Shock Surprise Peak (S_init):    {shock_result.shock_surprise:.4f} NLL")
    print(f"  >>> Observed Tail Mean (S_end):      {shock_result.plateau_surprise:.4f} NLL")
    print(f"  >>> Dynamic Recovery (Delta S):      {shock_result.delta_shock:.4f} NLL")
    print(f"  >>> Re-adaptation Half-Life (t_1/2):  {shock_result.half_life_tokens:.1f} tokens")
    print(f"  >>> Homeostatic Elasticity Score:    {shock_result.elasticity_score:.4f}")
    print(f"  >>> Post-Shock Representation SNR:   {shock_result.thermodynamics.snr_db:.2f} dB")

    print("\n================================================================================")
    print("PILLAR 3: UNBROKEN CONTINUOUS LIFECYCLE EBBINGHAUS SAVINGS (A -> B -> A)")
    print("================================================================================")
    print(f"Executing unbroken physical lifetime trajectory:")
    print(f"  Sequence A: {args.relearn_tokens} tokens (initial learning)")
    print(f"  Stream B:   {args.intervening_tokens} tokens (intervening continuous experience)")
    print(f"  Sequence A: {args.relearn_tokens} tokens (review relearning)")
    print(f"  Zero resets, zero artificial contexts; all synaptic plasticity active throughout.")

    tokens_a = torch.from_numpy(np.array(train_data[50000:50000 + args.relearn_tokens + 1], dtype=np.int64)).to(device)
    tokens_b = torch.from_numpy(np.array(train_data[60000:60000 + args.intervening_tokens + 1], dtype=np.int64)).to(device)

    savings_result = ContinuousLifecycleSavingsEvaluator.evaluate_unbroken_lifecycle(
        learner=learner,
        tokens_a=tokens_a,
        tokens_b=tokens_b,
    )

    print(f"  >>> Actual Intervening Tokens:       {savings_result.intervening_tokens} tokens")
    print(f"  >>> Initial Encounter Mean NLL:      {savings_result.nll_initial_mean:.4f}")
    print(f"  >>> Relearning Encounter Mean NLL:   {savings_result.nll_relearn_mean:.4f}")
    print(f"  >>> Immediate Recall Drop:           {savings_result.retention_immediate_drop:.4f} NLL")
    print(f"  >>> Ebbinghaus Savings Ratio:        {savings_result.savings_ratio * 100:.2f}%")
    print(f"  >>> Relearning Acceleration Factor:  {savings_result.acceleration_factor:.2f}x faster")

    print("\n================================================================================")
    print("INFORMATION LEDGER (Representation Geometry & Surprise)")
    print("================================================================================")
    full_thermo = learner.auditor.summary()
    info = full_thermo.information
    print(f"  >>> Total Evaluated Tokens:          {info.total_tokens}")
    print(f"  >>> Cumulative Surprise:             {info.cumulative_surprise_nats:.2f} nats")
    print(f"  >>> Mean Surprise:                   {info.mean_surprise_nats:.4f} nats/token")
    print(f"  >>> Effective Manifold Rank (R_eff): {info.effective_rank:.2f} (Roy & Vetterli 2007)")
    print(f"  >>> Participation Ratio (D_PR):      {info.participation_ratio:.2f}")
    print(f"  >>> Spectral Entropy:                {info.spectral_entropy:.4f} / {info.max_spectral_entropy:.4f} nats ({info.spectral_entropy_ratio*100:.1f}%)")
    print(f"  >>> Spatial Manifold Variance:       {info.spatial_variance:.6f}")
    print(f"  >>> Temporal Roughness (Velocity):   {info.temporal_roughness:.6f}")
    print(f"  >>> State Autocorrelation Proxy:     {info.state_autocorrelation_proxy:.4f}")

    print("\n================================================================================")
    print("ENERGY & METABOLIC LEDGER (Biophysical Dissipation)")
    print("================================================================================")
    energy = full_thermo.energy
    print(f"  >>> Cumulative Action Potentials:    {energy.cumulative_spikes} spikes")
    print(f"  >>> Mean Firing Rate Density:        {energy.mean_firing_rate_density*100:.3f}% ({energy.mean_spikes_per_token:.1f} spikes/token)")
    print(f"  >>> Cumulative Conductance Contraction: {energy.cumulative_conductance:.2f}")
    print(f"  >>> Mean Membrane Conductance g_tot:    {energy.mean_conductance_per_token:.4f}")
    print(f"  >>> Cumulative Adaptation Load b:       {energy.cumulative_adaptation_load:.2f}")
    print(f"  >>> Information Efficiency:             {energy.information_efficiency_nats_per_100k_spikes:.4f} nats per 100k spikes")

    # Output artifact
    report = {
        "architecture": "FlyReservoir-MaleCNS-Topographic-COBA-ALIF-STP",
        "eval_paradigm": "Tri-Pillar Biological Life-Form Suite (Unbroken Continuum)",
        "warm_tokens": args.warm_tokens,
        "pillar_1_prequential_tracker": learner.tracker.summary(),
        "pillar_2_environmental_shock": {
            "generalization": shock_result.generalization,
            "legacy_tail_statistics": True,
            "n_tokens": shock_result.n_tokens,
            "shock_surprise_initial": shock_result.shock_surprise,
            "plateau_surprise": shock_result.plateau_surprise,
            "delta_shock": shock_result.delta_shock,
            "half_life_tokens": shock_result.half_life_tokens,
            "elasticity_score": shock_result.elasticity_score,
            "post_shock_snr_db": shock_result.thermodynamics.snr_db,
        },
        "pillar_3_unbroken_lifecycle_savings": {
            "n_tokens": savings_result.n_tokens,
            "actual_intervening_tokens": savings_result.intervening_tokens,
            "nll_initial_mean": savings_result.nll_initial_mean,
            "nll_relearn_mean": savings_result.nll_relearn_mean,
            "initial_opening_nll": savings_result.initial_opening_nll,
            "relearn_opening_nll": savings_result.relearn_opening_nll,
            "retention_immediate_drop": savings_result.retention_immediate_drop,
            "savings_ratio": savings_result.savings_ratio,
            "acceleration_factor": savings_result.acceleration_factor,
            "thermodynamic_shift": savings_result.thermodynamic_shift,
        },
        "information_ledger": {
            "total_tokens": info.total_tokens,
            "cumulative_surprise_nats": info.cumulative_surprise_nats,
            "mean_surprise_nats": info.mean_surprise_nats,
            "effective_rank": info.effective_rank,
            "participation_ratio": info.participation_ratio,
            "spectral_entropy_nats": info.spectral_entropy,
            "max_spectral_entropy_nats": info.max_spectral_entropy,
            "spectral_entropy_ratio": info.spectral_entropy_ratio,
            "spatial_variance": info.spatial_variance,
            "temporal_roughness": info.temporal_roughness,
            "state_autocorrelation_proxy": info.state_autocorrelation_proxy,
        },
        "energy_ledger": {
            "total_tokens": energy.total_tokens,
            "cumulative_spikes": energy.cumulative_spikes,
            "mean_spikes_per_token": energy.mean_spikes_per_token,
            "mean_firing_rate_density": energy.mean_firing_rate_density,
            "cumulative_conductance": energy.cumulative_conductance,
            "mean_conductance_per_token": energy.mean_conductance_per_token,
            "cumulative_adaptation_load": energy.cumulative_adaptation_load,
            "mean_adaptation_offset": energy.mean_adaptation_offset,
            "information_efficiency_nats_per_100k_spikes": energy.information_efficiency_nats_per_100k_spikes,
        },
        "legacy_thermodynamic_summary": {
            "total_steps": full_thermo.total_steps,
            "cumulative_entropy_inflow_nats": full_thermo.cumulative_entropy_inflow,
            "cumulative_entropy_dissipated_nats": full_thermo.cumulative_entropy_dissipated,
            "net_entropy_accumulated_nats": full_thermo.net_entropy_accumulated,
            "entropy_balance_ratio": full_thermo.entropy_balance_ratio,
            "mean_inflow_rate_nats_per_step": full_thermo.mean_inflow_rate,
            "mean_dissipation_rate_nats_per_step": full_thermo.mean_dissipation_rate,
            "signal_power": full_thermo.signal_power,
            "noise_power": full_thermo.noise_power,
            "snr_db": full_thermo.snr_db,
            "initial_snr_db": full_thermo.initial_snr_db,
            "delta_snr_db": full_thermo.delta_snr_db,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nSaved Tri-Pillar Biological Evaluation Report to {args.output}")


if __name__ == "__main__":
    main()
