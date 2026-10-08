"""Preregistered Mechanism Diagnosis: Surrogate Gradient vs. Hard Forward Dynamics.

Runs 3 consecutive 32-token OWT windows on the 1.5M fly connectome checkpoint.
Evaluates:
- Base Fit / Base Follow NLL
- Full Adam (hard, follow, fixed-spike)
- Head-only Adam
- Body-only Adam
- Body Pure Negative SG direction (norm-matched to Adam displacement, with edge clamps)
- Body Adam Fractions (0.125, 0.25, 0.5)
- Body Adam Subcomponents (edges, writer, cells)
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:64"

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


def snapshot_parameters(named: dict[str, nn.Parameter]) -> dict[str, torch.Tensor]:
    torch.cuda.synchronize()
    snaps = {n: p.detach().cpu().clone() for n, p in named.items()}
    return snaps


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
                val = origin[n] + fraction * (src - origin[n])
                p.copy_(val)
            else:
                p.copy_(src)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/active_training/q8_fly_bptt32_gamma_2333_wide115_100k/last.pt"))
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=Path("results/published/fly_surrogate_vs_hard_1500k.json"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading 1.5M checkpoint from {args.checkpoint} on {args.device}...", flush=True)
    saved, learner = load_checkpoint(args.checkpoint, device=args.device)
    cfg, model = saved["config"], learner.model
    cursor, width = saved["train_cursor"], cfg["window"]
    print(f"Restored train cursor: {cursor}, window width: {width}", flush=True)

    train_data = np.load(ROOT / cfg["data"] / "train.npy", mmap_mode="r")
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}

    head_keys = {"output_read.weight", "read_norm.weight", "decoder.weight", "decoder.bias", "logit_read_gamma"} & named.keys()
    body_keys = named.keys() - head_keys
    edge_keys = set(learner.edge_names) & named.keys()
    writer_keys = {n for n in body_keys if n.startswith("topographic_writer.")}
    cell_keys = body_keys - edge_keys - writer_keys

    print(f"Parameter Partition: Head={len(head_keys)}, Body={len(body_keys)} (Edges={len(edge_keys)}, Writer={len(writer_keys)}, Cells={len(cell_keys)})", flush=True)

    def forward_eval(state, ids, targets, fixed_spikes=None, return_terminal=False):
        learner.state = state
        with torch.no_grad(), spike_instrument(fixed_spikes) as rec:
            scores, terminal, _ = learner.forward_window(ids, targets[None])
        term = clone_state(terminal) if return_terminal else None
        return scores.detach().cpu().tolist(), term, rec

    results = {
        "checkpoint": str(args.checkpoint),
        "bptt_train_tokens": saved.get("bptt_train_tokens", 1500000),
        "windows": [],
    }

    print("\n" + "=" * 70)
    print(f"STARTING 3-WINDOW PREREGISTERED MECHANISM DIAGNOSIS")
    print("=" * 70, flush=True)

    for it in range(args.windows):
        t0 = time.perf_counter()
        offset = cursor + it * width
        labels = torch.as_tensor(np.array(train_data[offset+1:offset+width+1], dtype=np.int64), device=args.device)
        fresh = torch.as_tensor(np.array(train_data[offset+width+1:offset+2*width+1], dtype=np.int64), device=args.device)
        ids = torch.cat((labels.new_tensor([learner.previous_token]), labels[:-1]))[None]
        fresh_ids = torch.cat((labels[-1:], fresh[:-1]))[None]

        s0 = clone_state(learner.state)
        old_params = snapshot_parameters(named)

        # Baseline evaluations on old parameters
        base_fit, s1, ref_fit_spk = forward_eval(s0, ids, labels, return_terminal=True)
        base_fol, _, ref_fol_spk = forward_eval(s1, fresh_ids, fresh)
        base_fit_nll = float(np.mean(base_fit))
        base_follow_nll = float(np.mean(base_fol))

        print(f"\n--- Window {it+1}/{args.windows} (Tokens {offset+1}..{offset+width}) ---", flush=True)
        print(f"  Base Fit NLL    : {base_fit_nll:.4f}", flush=True)
        print(f"  Base Follow NLL : {base_follow_nll:.4f}", flush=True)

        # Live Observe: execute real Adam update and capture unclipped raw gradients
        learner.state = s0
        learner.runner = None
        gc.collect()
        torch.cuda.empty_cache()
        with observe_gradient_instrument() as captured:
            scores_obs, metrics = learner.observe(labels)
        torch.cuda.synchronize()
        live_s1 = clone_state(learner.state)
        new_params = snapshot_parameters(named)

        # Reconstruct unclipped raw gradient on CPU
        clip_factor = min(1.0, learner.max_grad_norm / (metrics["grad_norm_before_clip"] + 1e-6))
        raw_grads = {}
        for p_id, p_name in {id(p): n for n, p in named.items()}.items():
            if p_id in captured:
                raw_grads[p_name] = (captured[p_id] / clip_factor).detach().cpu()
        del captured
        torch.cuda.empty_cache()

        # Measure Body Adam Displacement Norm
        body_disp_norm_sq = sum((new_params[n] - old_params[n]).norm().item()**2 for n in body_keys)
        body_disp_norm = float(body_disp_norm_sq**0.5)

        # Construct Pure Negative SG Direction for Body
        body_grad_norm_sq = sum(raw_grads[n].norm().item()**2 for n in body_keys if n in raw_grads)
        body_grad_norm = float(body_grad_norm_sq**0.5)

        pure_sg_params = {n: t.clone() for n, t in old_params.items() if n in body_keys}
        if body_grad_norm > 1e-9:
            step_scale = body_disp_norm / body_grad_norm
            for n in body_keys:
                if n in raw_grads:
                    pure_sg_params[n] = old_params[n] - step_scale * raw_grads[n]

        # Apply edge clamps to pure_sg_params
        torch.cuda.empty_cache()
        with torch.no_grad():
            for n in body_keys:
                named[n].copy_(pure_sg_params[n])
            learner.clamp_edges()
            torch.cuda.synchronize()
            pure_sg_clamped = {n: named[n].detach().cpu().clone() for n in body_keys}

        window_entry = {
            "window": it + 1,
            "base_fit_nll": base_fit_nll,
            "base_follow_nll": base_follow_nll,
            "grad_norm_before_clip": metrics["grad_norm_before_clip"],
            "body_disp_norm": body_disp_norm,
            "body_grad_norm": body_grad_norm,
            "controls": {},
        }

        # Control A: Full Adam
        restore_parameters(named, new_params)
        f_fit, _, f_fit_spk = forward_eval(s0, ids, labels)
        f_fol, _, _ = forward_eval(s1, fresh_ids, fresh)
        f_fix, _, _ = forward_eval(s0, ids, labels, fixed_spikes=ref_fit_spk)
        f_spikes = branch_distance(ref_fit_spk, f_fit_spk)["changed_spikes"]
        del f_fit_spk
        window_entry["controls"]["full_adam"] = {
            "fit_nll": float(np.mean(f_fit)),
            "following_nll": float(np.mean(f_fol)),
            "fixed_branch_nll": float(np.mean(f_fix)),
            "delta_fit": float(np.mean(f_fit)) - base_fit_nll,
            "delta_following": float(np.mean(f_fol)) - base_follow_nll,
            "spikes_changed": f_spikes,
        }

        # Control B: Head-only Adam
        restore_parameters(named, new_params, selected=head_keys, origin=old_params)
        h_fit, _, h_fit_spk = forward_eval(s0, ids, labels)
        h_fol, _, _ = forward_eval(s1, fresh_ids, fresh)
        h_fix, _, _ = forward_eval(s0, ids, labels, fixed_spikes=ref_fit_spk)
        h_spikes = branch_distance(ref_fit_spk, h_fit_spk)["changed_spikes"]
        del h_fit_spk
        window_entry["controls"]["head_only_adam"] = {
            "fit_nll": float(np.mean(h_fit)),
            "following_nll": float(np.mean(h_fol)),
            "fixed_branch_nll": float(np.mean(h_fix)),
            "delta_fit": float(np.mean(h_fit)) - base_fit_nll,
            "delta_following": float(np.mean(h_fol)) - base_follow_nll,
            "spikes_changed": h_spikes,
        }

        # Control C: Body-only Adam
        restore_parameters(named, new_params, selected=body_keys, origin=old_params)
        b_fit, _, b_fit_spk = forward_eval(s0, ids, labels)
        b_fol, _, _ = forward_eval(s1, fresh_ids, fresh)
        b_fix, _, _ = forward_eval(s0, ids, labels, fixed_spikes=ref_fit_spk)
        b_spikes = branch_distance(ref_fit_spk, b_fit_spk)["changed_spikes"]
        del b_fit_spk
        window_entry["controls"]["body_only_adam"] = {
            "fit_nll": float(np.mean(b_fit)),
            "following_nll": float(np.mean(b_fol)),
            "fixed_branch_nll": float(np.mean(b_fix)),
            "delta_fit": float(np.mean(b_fit)) - base_fit_nll,
            "delta_following": float(np.mean(b_fol)) - base_follow_nll,
            "spikes_changed": b_spikes,
        }

        # Control D: Body Pure Negative SG Direction
        restore_parameters(named, pure_sg_clamped, selected=body_keys, origin=old_params)
        sg_fit, _, sg_fit_spk = forward_eval(s0, ids, labels)
        sg_fol, _, _ = forward_eval(s1, fresh_ids, fresh)
        sg_fix, _, _ = forward_eval(s0, ids, labels, fixed_spikes=ref_fit_spk)
        sg_spikes = branch_distance(ref_fit_spk, sg_fit_spk)["changed_spikes"]
        del sg_fit_spk
        window_entry["controls"]["body_pure_sg"] = {
            "fit_nll": float(np.mean(sg_fit)),
            "following_nll": float(np.mean(sg_fol)),
            "fixed_branch_nll": float(np.mean(sg_fix)),
            "delta_fit": float(np.mean(sg_fit)) - base_fit_nll,
            "delta_following": float(np.mean(sg_fol)) - base_follow_nll,
            "spikes_changed": sg_spikes,
        }

        # Control E: Body Fractions (0.125, 0.25, 0.5)
        window_entry["body_fractions"] = []
        for frac in [0.125, 0.25, 0.5]:
            restore_parameters(named, new_params, selected=body_keys, fraction=frac, origin=old_params)
            bf_fit, _, bf_fit_spk = forward_eval(s0, ids, labels)
            bf_fix, _, _ = forward_eval(s0, ids, labels, fixed_spikes=ref_fit_spk)
            bf_spikes = branch_distance(ref_fit_spk, bf_fit_spk)["changed_spikes"]
            del bf_fit_spk
            window_entry["body_fractions"].append({
                "fraction": frac,
                "delta_hard_fit": float(np.mean(bf_fit)) - base_fit_nll,
                "delta_fixed_fit": float(np.mean(bf_fix)) - base_fit_nll,
                "spikes_changed": bf_spikes,
            })

        # Control F: Body Subcomponents
        window_entry["body_subcomponents"] = {}
        for sub_name, sub_keys in [("edges", edge_keys), ("writer", writer_keys), ("cells", cell_keys)]:
            if not sub_keys:
                continue
            restore_parameters(named, new_params, selected=sub_keys, origin=old_params)
            sub_fit, _, sub_fit_spk = forward_eval(s0, ids, labels)
            sub_fol, _, _ = forward_eval(s1, fresh_ids, fresh)
            sub_fix, _, _ = forward_eval(s0, ids, labels, fixed_spikes=ref_fit_spk)
            sub_spikes = branch_distance(ref_fit_spk, sub_fit_spk)["changed_spikes"]
            del sub_fit_spk
            window_entry["body_subcomponents"][sub_name] = {
                "delta_fit": float(np.mean(sub_fit)) - base_fit_nll,
                "delta_following": float(np.mean(sub_fol)) - base_follow_nll,
                "delta_fixed_fit": float(np.mean(sub_fix)) - base_fit_nll,
                "spikes_changed": sub_spikes,
            }

        print(f"  [Full Adam]      : Fit Delta = {window_entry['controls']['full_adam']['delta_fit']:+.4f} | Follow Delta = {window_entry['controls']['full_adam']['delta_following']:+.4f} | Spikes = {window_entry['controls']['full_adam']['spikes_changed']}", flush=True)
        print(f"  [Head-only Adam] : Fit Delta = {window_entry['controls']['head_only_adam']['delta_fit']:+.4f} | Follow Delta = {window_entry['controls']['head_only_adam']['delta_following']:+.4f} | Spikes = {window_entry['controls']['head_only_adam']['spikes_changed']}", flush=True)
        print(f"  [Body-only Adam] : Fit Delta = {window_entry['controls']['body_only_adam']['delta_fit']:+.4f} | Follow Delta = {window_entry['controls']['body_only_adam']['delta_following']:+.4f} | Spikes = {window_entry['controls']['body_only_adam']['spikes_changed']}", flush=True)
        print(f"  [Body Pure SG]   : Fit Delta = {window_entry['controls']['body_pure_sg']['delta_fit']:+.4f} | Follow Delta = {window_entry['controls']['body_pure_sg']['delta_following']:+.4f} | Spikes = {window_entry['controls']['body_pure_sg']['spikes_changed']}", flush=True)
        for frac_entry in window_entry["body_fractions"]:
            print(f"    Body Frac {frac_entry['fraction']:.3f} : Hard Delta = {frac_entry['delta_hard_fit']:+.6f} | Fixed Smooth Delta = {frac_entry['delta_fixed_fit']:+.6f} | Spikes = {frac_entry['spikes_changed']}", flush=True)
        for sub_name, sub_data in window_entry["body_subcomponents"].items():
            print(f"    Subcomponent {sub_name:<6}: Hard Delta = {sub_data['delta_fit']:+.4f} | Follow Delta = {sub_data['delta_following']:+.4f} | Fixed Delta = {sub_data['delta_fixed_fit']:+.4f} | Spikes = {sub_data['spikes_changed']}", flush=True)

        # Advance live state with live weights
        restore_parameters(named, new_params)
        learner.state = live_s1

        window_entry["elapsed_sec"] = time.perf_counter() - t0
        results["windows"].append(window_entry)

        # Memory cleanup
        del old_params, new_params, pure_sg_params, pure_sg_clamped, raw_grads
        del s0, s1, live_s1, ref_fit_spk, ref_fol_spk
        gc.collect()
        torch.cuda.empty_cache()

        peak_mib = torch.cuda.max_memory_allocated() / 2**20
        print(f"  Window {it+1} elapsed: {window_entry['elapsed_sec']:.2f}s | Peak VRAM: {peak_mib:.1f} MiB", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nDiagnosis complete! Full results saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
