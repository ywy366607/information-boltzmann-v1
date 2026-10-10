"""Frozen causal attribution ablation audit for 3D Plastic Wave Medium at 86k tokens.

Evaluates 6 causal conditions on held-out real-world OpenWebText validation segments:
1. Full (Baseline)
2. No Transport (skip spatial wave propagation, B = 0)
3. No Collision (skip nonlinear scattering layers)
4. No Both (simultaneous ablation of transport and collision)
5. Homogeneous Material (flatten spatial Fourier modes, CV -> 0, preserving DC mean)
6. No Temporal Probes (instantaneous physical readout only, omitting complex resonator bank)
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D


def evaluate_condition(model, segments, condition, event_duration, device="cuda"):
    """Evaluates NLL across segments under a specific ablation condition."""
    saved_coeffs = model.medium.material.coefficients.data.clone()
    saved_temporal = model.temporal_readout
    original_advance = model.medium.advance

    # Configure condition
    disable_trans = (condition in ("no_transport", "no_both"))
    disable_coll = (condition in ("no_collision", "no_both"))
    disable_temp = (condition == "no_temporal_probes")
    homogeneous = (condition == "homogeneous_material")

    if homogeneous:
        homo_coeffs = torch.zeros_like(saved_coeffs)
        homo_coeffs[0] = saved_coeffs[0]
        model.medium.material.coefficients.data.copy_(homo_coeffs)

    if disable_temp:
        model.temporal_readout = None

    def patched_advance(state, duration, **kwargs):
        if disable_trans:
            kwargs["transport"] = False
        if disable_coll:
            kwargs["collision"] = False
        return original_advance(state, duration, **kwargs)

    model.medium.advance = patched_advance

    segment_nlls = []
    total_tokens = 0
    all_nlls = []

    table = F.normalize(model.source.embedding.weight, dim=-1)
    prepared = model.medium.prepare_evolution()

    try:
        with torch.no_grad():
            for seg in segments:
                # Fresh initial belief per independent evaluation segment
                belief = model.initial_belief(1)
                carry_token = 50256  # standard EOS/BOS token
                seg_losses = []

                for target_token in seg:
                    input_id = torch.tensor([carry_token], device=device, dtype=torch.long)
                    target = torch.tensor([target_token], device=device, dtype=torch.long)

                    duration = model.event_time(belief, event_duration)
                    written, _ = model.assimilate(belief, input_id, token_features=table, diagnostics=False)
                    outgoing, _ = model.advance_interval(written, duration, prepared=prepared, diagnostics=False)
                    feature, _ = model.read(outgoing, decode=False, prepared=prepared, diagnostics=False)
                    logits = model.decode(feature)

                    loss = F.cross_entropy(logits, target).item()
                    seg_losses.append(loss)
                    all_nlls.append(loss)

                    # Advance state
                    belief = outgoing
                    carry_token = target_token

                segment_nlls.append(float(np.mean(seg_losses)))
    finally:
        # Restore original state and methods
        model.medium.material.coefficients.data.copy_(saved_coeffs)
        model.temporal_readout = saved_temporal
        model.medium.advance = original_advance

    mean_nll = float(np.mean(all_nlls))
    std_nll = float(np.std(all_nlls))
    return {
        "mean_nll": mean_nll,
        "std_nll": std_nll,
        "segment_nlls": segment_nlls,
        "token_count": len(all_nlls)
    }


def main():
    parser = argparse.ArgumentParser(description="Audit causal attribution of 3D medium operators at 86k tokens")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("results/medium_d768_streaming_pathway_8x8x8_160k/last.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2/validation.npy"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/published/medium_causal_attribution_86k_20261009.json"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--num-segments", type=int, default=4)
    parser.add_argument("--segment-spacing", type=int, default=2048)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {args.checkpoint}...")
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False, mmap=True)
    cfg = saved["config"]
    constructor = cfg["constructor"]
    event_duration = float(cfg.get("event_duration", 0.03327237442135811))

    model = PlasticMediumPorts3D(**constructor).to(args.device).eval()
    model.load_state_dict(saved["model"])
    print(f"Model loaded successfully on {args.device}.")

    val_data = np.load(args.data, mmap_mode="r")
    segments = []
    for s in range(args.num_segments):
        start = s * args.segment_spacing
        seg = val_data[start : start + args.horizon].tolist()
        segments.append(seg)
    print(f"Prepared {len(segments)} validation segments of length {args.horizon} from {args.data}.")

    conditions = [
        ("full", "Full Baseline (All Operators Active)"),
        ("no_transport", "No Transport (Bypass Wave Propagation)"),
        ("no_collision", "No Collision (Bypass Nonlinear Scattering)"),
        ("no_both", "No Both (Simultaneous Transport & Collision Ablation)"),
        ("homogeneous_material", "Homogeneous Material (Flatten Spatial Fourier Modes, CV=0)"),
        ("no_temporal_probes", "No Temporal Probes (Instantaneous Field Readout Only)")
    ]

    results = {}
    for cond_key, cond_desc in conditions:
        print(f"Evaluating condition: {cond_desc}...")
        res = evaluate_condition(model, segments, cond_key, event_duration, device=args.device)
        results[cond_key] = res
        print(f"  -> Mean NLL: {res['mean_nll']:.4f} (std: {res['std_nll']:.4f})")

    full_nll = results["full"]["mean_nll"]
    delta_trans = results["no_transport"]["mean_nll"] - full_nll
    delta_coll = results["no_collision"]["mean_nll"] - full_nll
    delta_joint = results["no_both"]["mean_nll"] - full_nll
    delta_material = results["homogeneous_material"]["mean_nll"] - full_nll
    delta_temporal = results["no_temporal_probes"]["mean_nll"] - full_nll

    # Synergy ratio S = Delta_joint / (Delta_trans + Delta_coll)
    synergy_sum = delta_trans + delta_coll
    synergy_ratio = delta_joint / synergy_sum if synergy_sum > 0 else 1.0

    summary = {
        "title": "Generation 1 (86k Tokens) Causal Operator Attribution & Synergy Audit",
        "checkpoint": str(args.checkpoint),
        "step": saved["step"],
        "fresh_training_tokens": saved["cursor"] - 1,
        "segments_evaluated": len(segments),
        "tokens_per_segment": args.horizon,
        "total_tokens_scored": len(segments) * args.horizon,
        "event_duration": event_duration,
        "conditions": results,
        "causal_deltas": {
            "delta_nll_transport": delta_trans,
            "delta_nll_collision": delta_coll,
            "delta_nll_joint": delta_joint,
            "delta_nll_material_anisotropy": delta_material,
            "delta_nll_temporal_probes": delta_temporal,
            "synergy_ratio": synergy_ratio,
            "synergy_interpretation": (
                f"S = {synergy_ratio:.3f} > 1.0 confirms positive super-additive coupling between 3D wave "
                "routing and nonlinear collision dynamics."
                if synergy_ratio > 1.0 else
                f"S = {synergy_ratio:.3f} <= 1.0 indicates sub-additive or independent operator contributions."
            )
        }
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("CAUSAL ATTRIBUTION RESULTS:")
    print(f"  Full Baseline:        {full_nll:.4f} nats")
    print(f"  No Transport:         {results['no_transport']['mean_nll']:.4f} nats (Delta: {delta_trans:+.4f})")
    print(f"  No Collision:         {results['no_collision']['mean_nll']:.4f} nats (Delta: {delta_coll:+.4f})")
    print(f"  No Both:              {results['no_both']['mean_nll']:.4f} nats (Delta: {delta_joint:+.4f})")
    print(f"  Synergy Ratio (S):    {synergy_ratio:.4f}")
    print(f"  Homogeneous Material: {results['homogeneous_material']['mean_nll']:.4f} nats (Delta: {delta_material:+.4f})")
    print(f"  No Temporal Probes:   {results['no_temporal_probes']['mean_nll']:.4f} nats (Delta: {delta_temporal:+.4f})")
    print(f"Saved audit report to {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
