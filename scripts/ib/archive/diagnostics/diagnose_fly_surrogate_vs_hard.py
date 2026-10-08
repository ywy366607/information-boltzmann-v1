"""Mechanism Diagnostic: Surrogate Gradient vs. Hard Forward Dynamics.

Executes the preregistered mechanism check on the 1.5M fly connectome checkpoint:
1. Inherits full physical state (including 2-stage Gamma traces z1, z2).
2. Runs 3 consecutive 32-token OWT windows from the restored train cursor.
3. Compares:
   - Actual saved-Adam displacement (Full, Head-only, Body-only)
   - Pure negative Surrogate Gradient direction for Body (norm-matched to Adam displacement)
   - Step fractions (0.125, 0.25, 0.5)
   - Subcomponents (edges, writer, cells)
4. For each condition, measures:
   - Hard forward fit NLL
   - Fixed-spike smooth branch fit NLL
   - Hard forward following window NLL
   - Discrete spike flipping count (branch divergence)
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core import fly_reservoir as reservoir
from information_boltzmann.core.fly_bptt_learning import (
    FlyBPTTLearner,
    FlyPhysicalState,
)


def clone_state(state: FlyPhysicalState) -> FlyPhysicalState:
    return FlyPhysicalState(**{
        k: tuple(t.detach().clone() for t in v) if k == "ring"
        else v.detach().clone() for k, v in state.state_dict().items()
    })


@contextmanager
def observe_gradient_instrument():
    """Captures unclipped gradients at the moment of clip_grad_norm_."""
    original = torch.nn.utils.clip_grad_norm_
    captured = {}

    def instrument(parameters, *args, **kwargs):
        parameters = list(parameters)
        norm = original(parameters, *args, **kwargs)
        for parameter in parameters:
            if parameter.grad is not None:
                captured[id(parameter)] = parameter.grad.detach().clone()
        return norm

    torch.nn.utils.clip_grad_norm_ = instrument
    try:
        yield captured
    finally:
        torch.nn.utils.clip_grad_norm_ = original


@contextmanager
def spike_instrument(record=None):
    """Record actual hard binary spikes or fix them for a smooth-branch replay."""
    original = reservoir.SpikeFn
    spikes = []
    cursor = [0]

    class Instrument:
        @staticmethod
        def apply(margin):
            index = cursor[0]
            cursor[0] += 1
            if record is None:
                spike = original.apply(margin)
                spikes.append(spike.detach().bool().clone())
                return spike
            if index >= len(record):
                raise ValueError("Fixed branch has fewer ticks than replay")
            return record[index].to(margin.dtype)

    reservoir.SpikeFn = Instrument
    try:
        yield spikes
        if record is not None and cursor[0] != len(record):
            raise ValueError(f"Fixed branch tick count mismatch: expected {len(record)}, got {cursor[0]}")
    finally:
        reservoir.SpikeFn = original


def branch_distance(a, b):
    if len(a) != len(b):
        raise ValueError("Branch lengths differ")
    per_tick = [int((x != y).sum().item()) for x, y in zip(a, b)]
    changed = sum(per_tick)
    total = sum(x.numel() for x in a)
    return {
        "changed_spikes": changed,
        "total_decisions": total,
        "changed_fraction": changed / max(total, 1),
    }


def load_checkpoint(checkpoint_path: Path, device: str = "cuda"):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg, life = saved["config"], saved["learner"]

    p = life["physical"]
    for k in ("h", "ring", "ge", "gi", "b", "x", "u", "baseline", "h_mean", "gamma_z1", "gamma_z2"):
        if k not in p:
            raise ValueError(f"Missing required physical state: {k}")

    model = reservoir.FlyReservoirLM(
        ROOT / cfg["graph"],
        vocab_size=50257,
        d_model=cfg["d_model"],
        injection="topographic",
        read_surface="output",
        synapse_model="coba",
        use_alif=True,
        use_stp=True,
        decoder_bias=cfg["decoder_bias"],
        read_centering=life.get("read_centering", cfg.get("read_centering", False)),
        use_read_gamma_trace=life.get("use_read_gamma_trace", cfg.get("use_read_gamma_trace", True)),
    ).to(device)

    with torch.no_grad():
        for name in ("edge_weight_e", "edge_weight_i"):
            if name in saved["model"]:
                getattr(model, name).copy_(saved["model"][name].to(device))
        model.load_state_dict(
            {k: v.to(device) for k, v in saved["model"].items() if k not in ("edge_weight_e", "edge_weight_i")},
            strict=False,
        )

    model.dan_plastic_lr = life.get("dan_plastic_lr", 0.0)

    physical = FlyPhysicalState(**{
        k: tuple(t.to(device) for t in v) if k == "ring" else v.to(device)
        for k, v in life["physical"].items()
    })

    learner = FlyBPTTLearner(
        model, physical,
        adam_names=life["adam_names"],
        lr=cfg["lr"],
        lr_synapse=cfg["lr_synapse"],
        lr_sensory=cfg["lr_sensory"],
        lr_decoder=cfg.get("lr_decoder"),
        plasticity_optimizer=life["plasticity_optimizer_kind"],
        settle_ticks=life.get("settle_ticks", 0),
        writer_baseline_clock=life.get("writer_baseline_clock", "input"),
        learn_stp=life.get("learn_stp", False),
    )
    learner.load_adam_state(life["optimizer"])
    learner.sgd.load_state_dict(life["sgd"])
    learner.load_edge_signs(life)

    for key in ("events", "updates", "previous_token", "ema", "physical_ticks"):
        if key in life:
            setattr(learner, key, life[key])
    learner.latent_window.copy_(life["latent_window"].to(device))

    return saved, learner


def snapshot_parameters(named: dict[str, nn.Parameter]) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in named.items()}


def restore_parameters(named: dict[str, nn.Parameter], snapshot: dict[str, torch.Tensor],
                       selected: set[str] | None = None, fraction: float = 1.0,
                       origin: dict[str, torch.Tensor] | None = None):
    with torch.no_grad():
        for n, p in named.items():
            use_snapshot = selected is None or n in selected
            if not use_snapshot and origin is None:
                continue
            src = snapshot[n] if use_snapshot else origin[n]
            if use_snapshot and origin is not None and fraction != 1.0:
                p.copy_(origin[n] + fraction * (src - origin[n]))
            else:
                p.copy_(src)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/active_training/q8_fly_bptt32_gamma_2333_wide115_100k/last.pt"))
    parser.add_argument("--windows", type=int, default=3, help="Number of consecutive windows (default 3)")
    parser.add_argument("--output", type=Path,
                        default=Path("results/published/fly_surrogate_vs_hard_1500k.json"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading 1.5M checkpoint from {args.checkpoint} on {args.device}...")
    saved, learner = load_checkpoint(args.checkpoint, device=args.device)
    cfg, model = saved["config"], learner.model
    cursor, width = saved["train_cursor"], cfg["window"]
    print(f"Restored train cursor: {cursor}, window width: {width}")

    train_data = np.load(ROOT / cfg["data"] / "train.npy", mmap_mode="r")
    if int(train_data[cursor]) != learner.previous_token:
        print(f"Warning: previous token ({learner.previous_token}) != train_data[{cursor}] ({train_data[cursor]})")

    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    head_keys = {"output_read.weight", "read_norm.weight", "decoder.weight", "decoder.bias", "logit_read_gamma"} & named.keys()
    body_keys = named.keys() - head_keys
    edge_keys = set(learner.edge_names) & named.keys()
    writer_keys = {n for n in body_keys if n.startswith("topographic_writer.")}
    cell_keys = body_keys - edge_keys - writer_keys

    groups = {
        "full": set(named.keys()),
        "head": head_keys,
        "body": body_keys,
        "edges": edge_keys,
        "writer": writer_keys,
        "cells": cell_keys,
    }

    print(f"\nParameter Partition:")
    print(f"  Head parameters ({len(head_keys)}): {sorted(head_keys)}")
    print(f"  Body parameters ({len(body_keys)}): {len(body_keys)} tensors")
    print(f"    - Edges ({len(edge_keys)}): {sorted(edge_keys)}")
    print(f"    - Writer ({len(writer_keys)}): {sorted(writer_keys)}")
    print(f"    - Cells ({len(cell_keys)}): {sorted(cell_keys)}")

    def forward_eval(state: FlyPhysicalState, ids: torch.Tensor, targets: torch.Tensor, fixed_spikes=None):
        learner.state = state
        with torch.no_grad(), spike_instrument(fixed_spikes) as rec:
            scores, terminal, _ = learner.forward_window(ids, targets[None])
        return scores.detach().cpu().tolist(), clone_state(terminal), rec

    results = {
        "checkpoint": str(args.checkpoint),
        "bptt_train_tokens": saved.get("bptt_train_tokens", 1500000),
        "windows": [],
    }

    print("\n" + "=" * 70)
    print(f"STARTING PREREGISTERED MECHANISM DIAGNOSIS ({args.windows} CONSECUTIVE WINDOWS)")
    print("=" * 70)

    for it in range(args.windows):
        t0 = time.perf_counter()
        offset = cursor + it * width
        labels = torch.as_tensor(np.array(train_data[offset+1:offset+width+1], dtype=np.int64), device=args.device)
        fresh = torch.as_tensor(np.array(train_data[offset+width+1:offset+2*width+1], dtype=np.int64), device=args.device)
        ids = torch.cat((labels.new_tensor([learner.previous_token]), labels[:-1]))[None]
        fresh_ids = torch.cat((labels[-1:], fresh[:-1]))[None]

        s0 = clone_state(learner.state)
        old_params = snapshot_parameters(named)

        # Baseline evaluations on old weights
        base_fit_scores, s1, ref_fit_spikes = forward_eval(s0, ids, labels)
        base_follow_scores, _, ref_follow_spikes = forward_eval(s1, fresh_ids, fresh)
        base_fit_nll = float(np.mean(base_fit_scores))
        base_follow_nll = float(np.mean(base_follow_scores))

        print(f"\n--- Window {it+1}/{args.windows} (Tokens {offset+1}..{offset+width}) ---", flush=True)
        print(f"  Base Fit NLL    : {base_fit_nll:.4f}", flush=True)
        print(f"  Base Follow NLL : {base_follow_nll:.4f}", flush=True)

        window_entry = {
            "window": it + 1,
            "base_fit_nll": base_fit_nll,
            "base_follow_nll": base_follow_nll,
            "controls": {},
        }

        try:
            # Live Observe: execute real Adam update and capture raw gradients
            print("  [DEBUG 1] Starting observe...", flush=True)
            learner.state = s0
            learner.runner = None  # Ensure eager execution without graph conflicts
            with observe_gradient_instrument() as captured:
                scores_obs, metrics = learner.observe(labels)
            print(f"  [DEBUG 2] Observe completed. Grad norm: {metrics['grad_norm_before_clip']:.4f}", flush=True)
            live_s1 = clone_state(learner.state)
            new_params = snapshot_parameters(named)
            print("  [DEBUG 3] Snapshots taken.", flush=True)
        except Exception as ex:
            import traceback
            print("  [ERROR in Observe]:", ex, flush=True)
            traceback.print_exc()
            sys.exit(1)

        try:
            print("  [DEBUG 4] Reconstructing raw grads...", flush=True)
            clip_factor = min(1.0, learner.max_grad_norm / (metrics["grad_norm_before_clip"] + 1e-6))
            raw_grads = {}
            for p_id, p_name in {id(p): n for n, p in named.items()}.items():
                if p_id in captured:
                    raw_grads[p_name] = (captured[p_id] / clip_factor).detach().cpu()

            print("  [DEBUG 5] Measuring body displacement...", flush=True)
            body_disp_norm_sq = sum((new_params[n] - old_params[n]).norm()**2 for n in body_keys)
            body_disp_norm = float(body_disp_norm_sq**0.5)

            print("  [DEBUG 6] Constructing pure SG direction...", flush=True)
            body_grad_norm_sq = sum(raw_grads[n].norm()**2 for n in body_keys if n in raw_grads)
            body_grad_norm = float(body_grad_norm_sq**0.5)

            pure_sg_params = {n: t.clone() for n, t in old_params.items()}
            if body_grad_norm > 1e-9:
                step_scale = body_disp_norm / body_grad_norm
                for n in body_keys:
                    if n in raw_grads:
                        pure_sg_params[n] = old_params[n] - step_scale * raw_grads[n]

            print("  [DEBUG 7] Copying pure SG to model & clamping...", flush=True)
            torch.cuda.empty_cache()
            with torch.no_grad():
                for n, p in named.items():
                    p.copy_(pure_sg_params[n])
                learner.clamp_edges()
                pure_sg_clamped = snapshot_parameters(named)

            print("  [DEBUG 8] Evaluating Full Adam...", flush=True)
            # A. Full Adam
            restore_parameters(named, new_params)
            f_fit, _, f_fit_spk = forward_eval(s0, ids, labels)
            f_fol, _, f_fol_spk = forward_eval(s1, fresh_ids, fresh)
            f_fix, _, _ = forward_eval(s0, ids, labels, fixed_spikes=ref_fit_spikes)
            print("  [DEBUG 9] Full Adam evaluated successfully.", flush=True)
            window_entry["grad_norm_before_clip"] = metrics["grad_norm_before_clip"]
            window_entry["body_disp_norm"] = body_disp_norm
        except Exception as ex:
            import traceback
            print("  [ERROR in Post-Observe]:", ex, flush=True)
            traceback.print_exc()
            sys.exit(1)
        window_entry["controls"]["full_adam"] = {
            "fit_nll": float(np.mean(f_fit)),
            "following_nll": float(np.mean(f_fol)),
            "fixed_branch_nll": float(np.mean(f_fix)),
            "delta_fit": float(np.mean(f_fit)) - base_fit_nll,
            "delta_following": float(np.mean(f_fol)) - base_follow_nll,
            "spikes_changed": branch_distance(ref_fit_spikes, f_fit_spk)["changed_spikes"],
        }

        # B. Head-only Adam
        restore_parameters(named, new_params, selected=head_keys, origin=old_params)
        h_fit, _, h_fit_spk = forward_eval(s0, ids, labels)
        h_fol, _, h_fol_spk = forward_eval(s1, fresh_ids, fresh)
        h_fix, _, _ = forward_eval(s0, ids, labels, fixed_spikes=ref_fit_spikes)
        window_entry["controls"]["head_only_adam"] = {
            "fit_nll": float(np.mean(h_fit)),
            "following_nll": float(np.mean(h_fol)),
            "fixed_branch_nll": float(np.mean(h_fix)),
            "delta_fit": float(np.mean(h_fit)) - base_fit_nll,
            "delta_following": float(np.mean(h_fol)) - base_follow_nll,
            "spikes_changed": branch_distance(ref_fit_spikes, h_fit_spk)["changed_spikes"],
        }

        # C. Body-only Adam
        restore_parameters(named, new_params, selected=body_keys, origin=old_params)
        b_fit, _, b_fit_spk = forward_eval(s0, ids, labels)
        b_fol, _, b_fol_spk = forward_eval(s1, fresh_ids, fresh)
        b_fix, _, _ = forward_eval(s0, ids, labels, fixed_spikes=ref_fit_spikes)
        window_entry["controls"]["body_only_adam"] = {
            "fit_nll": float(np.mean(b_fit)),
            "following_nll": float(np.mean(b_fol)),
            "fixed_branch_nll": float(np.mean(b_fix)),
            "delta_fit": float(np.mean(b_fit)) - base_fit_nll,
            "delta_following": float(np.mean(b_fol)) - base_follow_nll,
            "spikes_changed": branch_distance(ref_fit_spikes, b_fit_spk)["changed_spikes"],
        }

        # D. Body Pure Negative SG Direction
        restore_parameters(named, pure_sg_clamped, selected=body_keys, origin=old_params)
        sg_fit, _, sg_fit_spk = forward_eval(s0, ids, labels)
        sg_fol, _, sg_fol_spk = forward_eval(s1, fresh_ids, fresh)
        sg_fix, _, _ = forward_eval(s0, ids, labels, fixed_spikes=ref_fit_spikes)
        window_entry["controls"]["body_pure_sg"] = {
            "fit_nll": float(np.mean(sg_fit)),
            "following_nll": float(np.mean(sg_fol)),
            "fixed_branch_nll": float(np.mean(sg_fix)),
            "delta_fit": float(np.mean(sg_fit)) - base_fit_nll,
            "delta_following": float(np.mean(sg_fol)) - base_follow_nll,
            "spikes_changed": branch_distance(ref_fit_spikes, sg_fit_spk)["changed_spikes"],
        }

        # E. Step Fractions on Body Adam (0.125, 0.25, 0.5)
        window_entry["body_fractions"] = []
        for frac in [0.125, 0.25, 0.5]:
            restore_parameters(named, new_params, selected=body_keys, fraction=frac, origin=old_params)
            bf_fit, _, bf_fit_spk = forward_eval(s0, ids, labels)
            bf_fix, _, _ = forward_eval(s0, ids, labels, fixed_spikes=ref_fit_spikes)
            window_entry["body_fractions"].append({
                "fraction": frac,
                "hard_fit_nll": float(np.mean(bf_fit)),
                "delta_hard_fit": float(np.mean(bf_fit)) - base_fit_nll,
                "fixed_fit_nll": float(np.mean(bf_fix)),
                "delta_fixed_fit": float(np.mean(bf_fix)) - base_fit_nll,
                "spikes_changed": branch_distance(ref_fit_spikes, bf_fit_spk)["changed_spikes"],
            })

        # F. Subcomponents of Body Adam (edges, writer, cells)
        window_entry["body_subcomponents"] = {}
        for sub_name, sub_keys in [("edges", edge_keys), ("writer", writer_keys), ("cells", cell_keys)]:
            if not sub_keys:
                continue
            restore_parameters(named, new_params, selected=sub_keys, origin=old_params)
            sub_fit, _, sub_fit_spk = forward_eval(s0, ids, labels)
            sub_fol, _, _ = forward_eval(s1, fresh_ids, fresh)
            sub_fix, _, _ = forward_eval(s0, ids, labels, fixed_spikes=ref_fit_spikes)
            window_entry["body_subcomponents"][sub_name] = {
                "fit_nll": float(np.mean(sub_fit)),
                "following_nll": float(np.mean(sub_fol)),
                "fixed_branch_nll": float(np.mean(sub_fix)),
                "delta_fit": float(np.mean(sub_fit)) - base_fit_nll,
                "delta_following": float(np.mean(sub_fol)) - base_follow_nll,
                "spikes_changed": branch_distance(ref_fit_spikes, sub_fit_spk)["changed_spikes"],
            }

        # Print summary for this window
        print(f"  [Full Adam]      : Fit NLL = {window_entry['controls']['full_adam']['fit_nll']:.4f} ({window_entry['controls']['full_adam']['delta_fit']:+.4f}) | Follow NLL = {window_entry['controls']['full_adam']['following_nll']:.4f} ({window_entry['controls']['full_adam']['delta_following']:+.4f}) | Spikes Changed = {window_entry['controls']['full_adam']['spikes_changed']}")
        print(f"  [Head-only Adam] : Fit NLL = {window_entry['controls']['head_only_adam']['fit_nll']:.4f} ({window_entry['controls']['head_only_adam']['delta_fit']:+.4f}) | Follow NLL = {window_entry['controls']['head_only_adam']['following_nll']:.4f} ({window_entry['controls']['head_only_adam']['delta_following']:+.4f}) | Spikes Changed = {window_entry['controls']['head_only_adam']['spikes_changed']}")
        print(f"  [Body-only Adam] : Fit NLL = {window_entry['controls']['body_only_adam']['fit_nll']:.4f} ({window_entry['controls']['body_only_adam']['delta_fit']:+.4f}) | Follow NLL = {window_entry['controls']['body_only_adam']['following_nll']:.4f} ({window_entry['controls']['body_only_adam']['delta_following']:+.4f}) | Spikes Changed = {window_entry['controls']['body_only_adam']['spikes_changed']}")
        print(f"  [Body Pure SG]   : Fit NLL = {window_entry['controls']['body_pure_sg']['fit_nll']:.4f} ({window_entry['controls']['body_pure_sg']['delta_fit']:+.4f}) | Follow NLL = {window_entry['controls']['body_pure_sg']['following_nll']:.4f} ({window_entry['controls']['body_pure_sg']['delta_following']:+.4f}) | Spikes Changed = {window_entry['controls']['body_pure_sg']['spikes_changed']}")
        for frac_entry in window_entry["body_fractions"]:
            print(f"    Body Frac {frac_entry['fraction']:.3f} : Hard Delta = {frac_entry['delta_hard_fit']:+.6f} | Fixed Smooth Delta = {frac_entry['delta_fixed_fit']:+.6f} | Spikes = {frac_entry['spikes_changed']}")

        # Advance live individual state with live weights
        restore_parameters(named, new_params)
        learner.state = live_s1

        window_entry["elapsed_sec"] = time.perf_counter() - t0
        results["windows"].append(window_entry)

        # Explicit cleanup to release all non-live tensors from VRAM
        del old_params, new_params, pure_sg_clamped, raw_grads, captured
        del s0, s1, live_s1, ref_fit_spikes, ref_follow_spikes
        gc.collect()
        torch.cuda.empty_cache()

        # Check memory
        peak_mib = torch.cuda.max_memory_allocated() / 2**20
        print(f"  Window {it+1} elapsed: {window_entry['elapsed_sec']:.2f}s | Peak VRAM: {peak_mib:.1f} MiB")
        if peak_mib >= 3900:
            raise RuntimeError(f"Peak VRAM exceeded 3900 MiB limit: {peak_mib:.1f} MiB")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nDiagnosis complete! Full results saved to {args.output}")


if __name__ == "__main__":
    main()
