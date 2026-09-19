"""Sequential 3000-Step End-to-End Training & Evaluation Suite for Write Arms W1, W2, W3.

Controlled Write Operator Ablation:
- W0 (Baseline): Theta_max = 0.30, standard 2-port scattering, original init (~3.6% transmission).
- W1 (Unbounded Angle): 0 < theta < pi/2, standard 2-port unitary scattering, init transmission ~13%.
- W2 (Field-Conditioned Impedance): theta_t = pi/2 * sigmoid(g(x_t, E_local, <f_hat, p_hat>, |f|, |p|)), init ~13%.
- W3 (Bounded Additive Forcing): f' = f + eta_t * p, 0 <= eta_t <= 1.0, contractive bath, BIBO stable.

Fixed across all arms:
- Periodic T^3 (8, 8, 4), 256 nodes, D3Q8 x 16 channels (d=128).
- Static linear readout: EnergyFactoredTorusReadout (Arm A baseline readout strictly preserved).
- Transport, Collision, Cold Bath strictly preserved.
- No resetting persistent field.
- 3000 updates x 128 tokens = 384,000 tokens on OWT GPT-2 BPE, seed 11, lr 3e-4.
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


WRITE_ARMS = [
    {
        "name": "w1_unbounded",
        "write_type": "w1_unbounded",
        "output": Path("results/cbim_torus3d_w1_unbounded_3000"),
        "description": "Unbounded Two-Port Unitary Scattering (0 < theta < pi/2)",
    },
    {
        "name": "w2_impedance",
        "write_type": "w2_impedance",
        "output": Path("results/cbim_torus3d_w2_impedance_3000"),
        "description": "Field-Conditioned Impedance Matching (theta_t = pi/2 * sigma(x_t, E_loc, <f,p>, |f|, |p|))",
    },
    {
        "name": "w3_additive",
        "write_type": "w3_additive",
        "output": Path("results/cbim_torus3d_w3_additive_3000"),
        "description": "Bounded Additive Forcing Control (f' = f + eta_t * p, BIBO stable with contractive bath)",
    },
]


def run_arm(arm_info: dict, target_steps: int = 3000, max_step_increment: int = 1500, validate_every: int = 250):
    name = arm_info["name"]
    write_type = arm_info["write_type"]
    output = arm_info["output"]
    progress_file = output / "progress.json"

    current_step = 0
    if progress_file.exists():
        try:
            pdata = json.loads(progress_file.read_text(encoding="utf-8"))
            current_step = pdata.get("step", 0)
            if pdata.get("status") == "complete" and current_step >= target_steps and (output / "BBest.pt").exists():
                print(f"[{name}] Already complete with {current_step} steps (best NLL {pdata.get('best_validation_nll'):.5f}). Skipping.")
                return
        except Exception:
            pass

    while current_step < target_steps:
        next_steps = min(current_step + max_step_increment, target_steps)
        cmd = [
            sys.executable,
            "scripts/ib_local/train_cbim_torus3d.py",
            "--shape", "8", "8", "4",
            "--v2-coordinate-components",
            "--readout-type", "baseline",
            "--write-type", write_type,
            "--output", str(output),
            "--steps", str(next_steps),
            "--validate-every", str(validate_every),
        ]
        if (output / "last.pt").exists():
            cmd.extend(["--resume", str(output / "last.pt")])

        print(f"\n" + "=" * 70)
        print(f"[{name}] Chunk training: step {current_step} -> {next_steps} (target: {target_steps})")
        print(f"Command: {' '.join(cmd)}")
        print("=" * 70)

        t0 = time.perf_counter()
        ret = subprocess.run(cmd)
        elapsed = time.perf_counter() - t0

        if ret.returncode != 0:
            raise RuntimeError(f"Arm {name} failed at step {current_step} with exit code {ret.returncode}")

        if progress_file.exists():
            pdata = json.loads(progress_file.read_text(encoding="utf-8"))
            current_step = pdata.get("step", next_steps)
        else:
            current_step = next_steps

        print(f"[{name}] Finished chunk to step {current_step} in {elapsed / 60:.2f} minutes.")


def run_causality(checkpoint_path: Path, output_path: Path):
    cmd = [
        sys.executable,
        "scripts/ib_local/measure_cbim_unified_causality.py",
        "--checkpoint", str(checkpoint_path),
        "--output", str(output_path),
    ]
    print(f"\nRunning causality evaluation: {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError(f"Causality measurement failed for {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, choices=["all", "w1", "w2", "w3"], default="all")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--chunk", type=int, default=1500)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--causality-only", action="store_true")
    args = parser.parse_args()

    arm_map = {
        "w1": [WRITE_ARMS[0]],
        "w2": [WRITE_ARMS[1]],
        "w3": [WRITE_ARMS[2]],
        "all": WRITE_ARMS,
    }

    target_arms = arm_map[args.arm]

    if not args.causality_only:
        for arm_info in target_arms:
            run_arm(arm_info, target_steps=args.steps, max_step_increment=args.chunk, validate_every=args.validate_every)

    # Run causality evaluation on BBest.pt
    for arm_info in target_arms:
        bbest = arm_info["output"] / "BBest.pt"
        if bbest.exists():
            causal_out = arm_info["output"] / "causality_report.json"
            run_causality(bbest, causal_out)


if __name__ == "__main__":
    main()
