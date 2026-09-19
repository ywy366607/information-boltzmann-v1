"""Measure parameter gradient convergence across BPTT horizons H in {16, 32, 64, 128, 256, 512}.

Evaluates:
- delta_H = ||grad_theta(H) - grad_theta(H_ref)|| / ||grad_theta(H_ref)||
- c_H = cos(grad_theta(H), grad_theta(H_ref))

This directly determines the calibrated effective BPTT horizon H*(epsilon).
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.cbim_torus3d import CBIMTorus3D


def flatten_grads(parameters):
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
        else:
            grads.append(torch.zeros_like(p).flatten())
    return torch.cat(grads)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("results/cbim_three_clock_w2_8x8x4_k3_3000/BBest.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2/validation.npy"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/published/cbim_parameter_gradient_convergence.json"))
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--ref-h", type=int, default=256)
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...", flush=True)
    saved = torch.load(args.checkpoint, map_location="cuda")
    cfg = saved["config"]

    model = CBIMTorus3D(
        shape=tuple(cfg["shape"]),
        velocities=cfg["velocities"],
        content_dim=cfg["content_dim"],
        v2_coordinate_components=True,
        readout_type=cfg.get("readout_type", "kernel_r1"),
        write_type=cfg.get("write_type", "w2_impedance"),
        micro_steps=cfg.get("micro_steps", 3),
        adaptive_clock=cfg.get("adaptive_clock", True),
        continuous_velocities=cfg.get("continuous_velocities", True),
        dissipation_type="unified",
        dissipation_rank=cfg.get("dissipation_rank", 4),
        three_clock=cfg.get("three_clock", True),
        tau_mem=cfg.get("tau_mem", 3.0),
        nu_s_init=cfg.get("nu_s_init", 0.020),
        decouple_source_feedback=True
    ).cuda()
    model.load_state_dict(saved["model"])
    model.eval()

    val_data = np.load(args.data, mmap_mode="r")
    mature_state_base = saved["state"].detach().cuda()

    horizons = [16, 32, 64, 128, args.ref_h]
    H_ref = args.ref_h
    start_offset = 8192

    # 1. Warm up state over warmup tokens
    state = mature_state_base.clone()
    with torch.no_grad():
        for t in range(args.warmup):
            inp = torch.as_tensor([val_data[start_offset + t]], dtype=torch.long, device="cuda")
            _, state, _ = model.step(state, inp, micro_steps=3)

    # 2. Run target sequence of length H_ref
    seq_start = start_offset + args.warmup
    full_ids = torch.as_tensor(val_data[seq_start:seq_start + H_ref], dtype=torch.long, device="cuda")[None]
    full_targets = torch.as_tensor(val_data[seq_start + 1:seq_start + H_ref + 1], dtype=torch.long, device="cuda")[None]

    # Compute reference gradient at H_ref
    model.zero_grad(set_to_none=True)
    loss_ref, _, _ = model(full_ids, full_targets, state.clone())
    loss_ref.backward()
    grad_ref = flatten_grads(model.parameters())
    norm_ref = float(grad_ref.norm().item())

    results = {}
    print("\n" + "=" * 95)
    print(f"   PARAMETER GRADIENT CONVERGENCE vs H_ref={H_ref} (Full Model Weights)")
    print("=" * 95)
    print(f"{'Horizon H':<12} | {'Loss':<10} | {'Grad Norm':<14} | {'Rel Error (delta_H)':<22} | {'Cosine (c_H)':<16} | {'Convergence'}")
    print("-" * 95)

    for H in horizons:
        model.zero_grad(set_to_none=True)
        h_ids = full_ids[:, :H]
        h_targets = full_targets[:, :H]
        loss_h, _, _ = model(h_ids, h_targets, state.clone())
        loss_h.backward()
        grad_h = flatten_grads(model.parameters())
        norm_h = float(grad_h.norm().item())

        dot = float(torch.dot(grad_h, grad_ref).item())
        cos_sim = dot / (norm_h * norm_ref + 1e-12)
        diff_norm = float((grad_h - grad_ref).norm().item())
        rel_err = diff_norm / (norm_ref + 1e-12)

        is_converged = ">= 99% Align" if cos_sim >= 0.99 else (">= 95% Align" if cos_sim >= 0.95 else "Deviating")
        print(f"{H:<12d} | {loss_h.item():<10.4f} | {norm_h:<14.4f} | {rel_err * 100:<20.2f}% | {cos_sim:<16.4f} | {is_converged}")

        results[str(H)] = {
            "loss": float(loss_h.item()),
            "grad_norm": norm_h,
            "rel_error": rel_err,
            "cosine_similarity": cos_sim,
        }

    print("=" * 95)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved convergence report to {args.output}")


if __name__ == "__main__":
    main()
