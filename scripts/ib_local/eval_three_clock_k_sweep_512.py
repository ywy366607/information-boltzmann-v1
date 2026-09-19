"""Dense evaluation of internal pondering depth K in [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512].

Evaluates:
- Model: CBIM Three-Clock v1 BPTT-128 Champion (trained at K=3, 7.1496 nats)
- Strict Rule: NEVER reset state to 0; inherit mature physical state from training.
- 1. Site 3 (high-causality site) evaluation across all K.
- 2. 512-token continuous stream evaluation across all K (NLL, Speed tok/s, Energy E).
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
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.cbim_torus3d import CBIMTorus3D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("results/cbim_three_clock_bptt128_8x8x4_k3_3000/BBest.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2/validation.npy"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/published/cbim_three_clock_k_sweep_512.json"))
    args = parser.parse_args()

    print(f"Loading champion checkpoint from {args.checkpoint}...", flush=True)
    saved = torch.load(args.checkpoint, map_location="cuda")
    cfg = saved["config"]

    model = CBIMTorus3D(
        shape=tuple(cfg["shape"]),
        velocities=cfg["velocities"],
        content_dim=cfg["content_dim"],
        v2_coordinate_components=True,
        readout_type=cfg.get("readout_type", "kernel_r1"),
        write_type=cfg.get("write_type", "w2_impedance"),
        micro_steps=3,
        adaptive_clock=True,
        continuous_velocities=True,
        dissipation_type="unified",
        dissipation_rank=4,
        three_clock=True,
        tau_mem=3.0,
        nu_s_init=0.020,
        decouple_source_feedback=True
    ).cuda()
    model.load_state_dict(saved["model"])
    model.eval()

    val_data = np.load(args.data, mmap_mode="r")
    mature_state_base = saved["state"].detach().cuda()

    horizons_k = [1, 2, 3, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 1024]

    # -------------------------------------------------------------
    # Experiment 1: Site 3 (High-Causality Stream) Evaluation
    # -------------------------------------------------------------
    site3_start = 8192 + 3 * 4096  # 20480
    warmup_len = 256
    eval_len = 128

    print("\n" + "=" * 90)
    print("   EXPERIMENT 1: ZERO-SHOT PONDERING DEPTH K SWEEP ON SITE 3 (HIGH-CAUSALITY)")
    print("=" * 90)
    print(f"{'Pondering K':<14} | {'Site 3 NLL':<14} | {'Delta vs K=3':<16} | {'Speed (tok/s)':<14} | {'Mean Energy E'}")
    print("-" * 90)

    # First warm up the shared 256-token history at native K=3
    shared_state = mature_state_base.clone()
    with torch.no_grad():
        for t in range(warmup_len):
            inp = torch.as_tensor([val_data[site3_start + t]], dtype=torch.long, device="cuda")
            _, shared_state, _ = model.step(shared_state, inp, micro_steps=3)

    site3_results = {}
    k3_ref_nll = None

    for k in horizons_k:
        state = shared_state.clone()
        total_loss = 0.0
        energies = []
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            for t in range(warmup_len, warmup_len + eval_len):
                inp = torch.as_tensor([val_data[site3_start + t]], dtype=torch.long, device="cuda")
                tgt = torch.as_tensor([val_data[site3_start + t + 1]], dtype=torch.long, device="cuda")
                logits, state, diag = model.step(state, inp, micro_steps=k)
                loss = F.cross_entropy(logits, tgt).item()
                total_loss += loss
                energies.append(float(diag["energy"]))

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        speed = eval_len / elapsed
        mean_nll = total_loss / eval_len
        mean_e = float(np.mean(energies))

        if k == 3:
            k3_ref_nll = mean_nll
        delta = (mean_nll - k3_ref_nll) if k3_ref_nll is not None else 0.0

        site3_results[k] = {
            "nll": mean_nll,
            "delta_vs_k3": delta,
            "speed": speed,
            "energy": mean_e,
        }
        print(f"K = {k:<10d} | {mean_nll:<14.4f} | {delta:<+16.4f} | {speed:<14.1f} | {mean_e:.6f}", flush=True)

    # -------------------------------------------------------------
    # Experiment 2: Long Continuous Stream (256 Tokens) Evaluation
    # -------------------------------------------------------------
    stream_start = 8192
    long_eval_len = 256

    print("\n" + "=" * 90)
    print("   EXPERIMENT 2: 512-TOKEN CONTINUOUS STREAM PONDERING DEPTH EVALUATION")
    print("=" * 90)
    print(f"{'Pondering K':<14} | {'Stream NLL':<14} | {'Delta vs K=3':<16} | {'Speed (tok/s)':<14} | {'Mean Energy E'}")
    print("-" * 90)

    # Warm up shared 256 tokens at stream_start
    stream_shared_state = mature_state_base.clone()
    with torch.no_grad():
        for t in range(warmup_len):
            inp = torch.as_tensor([val_data[stream_start + t]], dtype=torch.long, device="cuda")
            _, stream_shared_state, _ = model.step(stream_shared_state, inp, micro_steps=3)

    stream_results = {}
    k3_stream_nll = None

    for k in horizons_k:
        state = stream_shared_state.clone()
        total_loss = 0.0
        energies = []
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            for t in range(warmup_len, warmup_len + long_eval_len):
                inp = torch.as_tensor([val_data[stream_start + t]], dtype=torch.long, device="cuda")
                tgt = torch.as_tensor([val_data[stream_start + t + 1]], dtype=torch.long, device="cuda")
                logits, state, diag = model.step(state, inp, micro_steps=k)
                loss = F.cross_entropy(logits, tgt).item()
                total_loss += loss
                energies.append(float(diag["energy"]))

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        speed = long_eval_len / elapsed
        mean_nll = total_loss / long_eval_len
        mean_e = float(np.mean(energies))

        if k == 3:
            k3_stream_nll = mean_nll
        delta = (mean_nll - k3_stream_nll) if k3_stream_nll is not None else 0.0

        stream_results[k] = {
            "nll": mean_nll,
            "delta_vs_k3": delta,
            "speed": speed,
            "energy": mean_e,
        }
        print(f"K = {k:<10d} | {mean_nll:<14.4f} | {delta:<+16.4f} | {speed:<14.1f} | {mean_e:.6f}", flush=True)

    print("=" * 90)
    report = {
        "checkpoint": str(args.checkpoint),
        "site3_sweep": site3_results,
        "continuous_stream_512": stream_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved full K-sweep report to {args.output}")


if __name__ == "__main__":
    main()
