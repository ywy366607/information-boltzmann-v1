"""Sequential 3000-Step End-to-End Training Suite for Readout Arms B, C, D.

Arms:
- Arm B: DynamicLinearReadout (Token-conditioned dynamic query + linear attention)
- Arm C: CharacteristicKernelReadout R=1 (Gaussian kernel in R^134 with [r_h, s_h, e_h] readings)
- Arm D: CharacteristicKernelReadout R=2 (Gaussian kernel with 2-round recurrent controller)

Physical field: Pure T^3 Torus (8, 8, 4), D3Q8 x 16 channels = d128.
Write operator: Original FullRankTorusWrite (strictly preserved, no impedance matching).
Budget: 3000 updates x 128 tokens = 384,000 tokens on OWT GPT-2 BPE, seed 11, lr 3e-4.
"""
from __future__ import annotations

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
from pathlib import Path
import subprocess
import time


ARMS = [
    {
        "name": "arm_b_dynamic_linear",
        "readout_type": "dynamic_linear",
        "output": Path("results/cbim_torus3d_arm_b_dynamic_linear_3000"),
    },
    {
        "name": "arm_c_kernel_r1",
        "readout_type": "kernel_r1",
        "output": Path("results/cbim_torus3d_arm_c_kernel_r1_3000"),
    },
    {
        "name": "arm_d_kernel_r2",
        "readout_type": "kernel_r2",
        "output": Path("results/cbim_torus3d_arm_d_kernel_r2_3000"),
    },
]


def run_arm(arm_info: dict, steps: int = 3000, validate_every: int = 250):
    name = arm_info["name"]
    readout_type = arm_info["readout_type"]
    output = arm_info["output"]
    progress_file = output / "progress.json"

    if progress_file.exists():
        try:
            pdata = json.loads(progress_file.read_text(encoding="utf-8"))
            if pdata.get("status") == "complete" and pdata.get("step", 0) >= steps and (output / "BBest.pt").exists():
                print(f"[{name}] Already complete with {pdata.get('step')} steps (best NLL {pdata.get('best_validation_nll'):.5f}). Skipping.")
                return
        except Exception:
            pass

    cmd = [
        sys.executable,
        "scripts/ib_local/train_cbim_torus3d.py",
        "--shape", "8", "8", "4",
        "--v2-coordinate-components",
        "--readout-type", readout_type,
        "--output", str(output),
        "--steps", str(steps),
        "--validate-every", str(validate_every),
    ]

    if (output / "last.pt").exists():
        cmd.extend(["--resume", str(output / "last.pt")])

    print(f"\n" + "=" * 70)
    print(f"LAUNCHING 3000-STEP END-TO-END TRAINING: {name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)

    t0 = time.perf_counter()
    ret = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0

    if ret.returncode != 0:
        raise RuntimeError(f"Arm {name} failed with exit code {ret.returncode}")

    print(f"[{name}] Completed successfully in {elapsed / 60:.2f} minutes.")


def summarize():
    summary = {
        "baseline_arm_a": {
            "name": "arm_a_baseline",
            "architecture": "CBIM-Torus3D-fullrank-d3q8-v2",
            "readout": "EnergyFactoredTorusReadout (Static 4-Query Linear)",
            "trainable_parameters": 6816709,
            "best_validation_nll": 7.24059,
        },
        "arms": {},
    }

    for arm_info in ARMS:
        name = arm_info["name"]
        output = arm_info["output"]
        prog_file = output / "progress.json"
        cfg_file = output / "config.json"
        if prog_file.exists() and cfg_file.exists():
            prog = json.loads(prog_file.read_text(encoding="utf-8"))
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            summary["arms"][name] = {
                "architecture": cfg.get("architecture"),
                "readout_type": cfg.get("readout_type"),
                "parameters": cfg.get("parameter_count"),
                "status": prog.get("status"),
                "best_validation_nll": prog.get("best_validation_nll"),
                "delta_vs_baseline": (
                    prog.get("best_validation_nll") - 7.24059
                    if prog.get("best_validation_nll") is not None
                    else None
                ),
            }

    summary_file = Path("results/cbim_torus3d_arms_bcd_3000_summary.json")
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, choices=["all", "b", "c", "d"], default="all")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--validate-every", type=int, default=250)
    args = parser.parse_args()

    arm_map = {
        "b": [ARMS[0]],
        "c": [ARMS[1]],
        "d": [ARMS[2]],
        "all": ARMS,
    }

    target_arms = arm_map[args.arm]
    for arm_info in target_arms:
        run_arm(arm_info, steps=args.steps, validate_every=args.validate_every)

    summarize()


if __name__ == "__main__":
    main()
