"""Orchestrate 6000-step training and causality comparison for Run 1 vs Run 3."""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import json
import shutil
import subprocess
from pathlib import Path
import numpy as np


def run_cmd(cmd, desc):
    print(f"\n{'='*70}\n[START] {desc}\nCommand: {' '.join(cmd)}\n{'='*70}", flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")
    print(f"\n[DONE] {desc}\n", flush=True)


def main():
    base_dir = Path(repo_root)
    r1_3000 = base_dir / "results" / "cbim_torus3d_w2_16ch_k3_adaptive_xavier_3000"
    r1_6000 = base_dir / "results" / "cbim_torus3d_w2_16ch_k3_adaptive_xavier_6000"
    r3_3000 = base_dir / "results" / "cbim_torus3d_w2_16ch_k3_adaptive_continuous_q8_3000"
    r3_6000 = base_dir / "results" / "cbim_torus3d_w2_16ch_k3_adaptive_continuous_q8_6000"
    pub_dir = base_dir / "results" / "published"
    pub_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Run 1: D3Q8 Baseline 3000 -> 6000 Steps
    # -------------------------------------------------------------
    print("\n" + "#" * 70)
    print("# STEP 1: RUN 1 (D3Q8) TRAINING TO 6000 STEPS")
    print("#" * 70, flush=True)

    r1_6000.mkdir(parents=True, exist_ok=True)
    if not (r1_6000 / "metrics.jsonl").exists() and (r1_3000 / "metrics.jsonl").exists():
        shutil.copy2(r1_3000 / "metrics.jsonl", r1_6000 / "metrics.jsonl")
        print(f"Preserved metrics.jsonl from {r1_3000.name} -> {r1_6000.name}")

    cmd_r1_train = [
        sys.executable, str(base_dir / "scripts" / "ib_local" / "train_cbim_torus3d.py"),
        "--output", str(r1_6000),
        "--resume", str(r1_3000 / "last.pt"),
        "--steps", "6000",
        "--tokens", "128",
        "--shape", "8", "8", "4",
        "--velocities", "8",
        "--content-dim", "16",
        "--collision-layers", "2",
        "--v2-coordinate-components",
        "--readout-type", "kernel_r1",
        "--write-type", "w2_impedance",
        "--micro-steps", "3",
        "--adaptive-clock",
        "--chunk-tokens", "128",
        "--validate-every", "250",
    ]
    run_cmd(cmd_r1_train, "Run 1 (D3Q8) Train 3000 -> 6000 Steps")

    cmd_r1_causal = [
        sys.executable, str(base_dir / "scripts" / "ib_local" / "measure_cbim_unified_causality.py"),
        "--checkpoint", str(r1_6000 / "BBest.pt"),
        "--output", str(pub_dir / "cbim_torus3d_w2_16ch_k3_adaptive_xavier_causality_6000.json"),
    ]
    run_cmd(cmd_r1_causal, "Run 1 (D3Q8) Causality Intervention Evaluation at Step 6000")

    # -------------------------------------------------------------
    # 2. Run 3: Continuous Q8 on S^2 3000 -> 6000 Steps
    # -------------------------------------------------------------
    print("\n" + "#" * 70)
    print("# STEP 2: RUN 3 (CONTINUOUS Q8 ON S^2) TRAINING TO 6000 STEPS")
    print("#" * 70, flush=True)

    r3_6000.mkdir(parents=True, exist_ok=True)
    if not (r3_6000 / "metrics.jsonl").exists() and (r3_3000 / "metrics.jsonl").exists():
        shutil.copy2(r3_3000 / "metrics.jsonl", r3_6000 / "metrics.jsonl")
        print(f"Preserved metrics.jsonl from {r3_3000.name} -> {r3_6000.name}")

    cmd_r3_train = [
        sys.executable, str(base_dir / "scripts" / "ib_local" / "train_cbim_torus3d.py"),
        "--output", str(r3_6000),
        "--resume", str(r3_3000 / "last.pt"),
        "--steps", "6000",
        "--tokens", "128",
        "--shape", "4", "4", "4",
        "--velocities", "8",
        "--content-dim", "16",
        "--collision-layers", "2",
        "--v2-coordinate-components",
        "--readout-type", "dynamic_linear",
        "--write-type", "w2_impedance",
        "--micro-steps", "3",
        "--adaptive-clock",
        "--continuous-velocities",
        "--chunk-tokens", "128",
        "--validate-every", "250",
    ]
    run_cmd(cmd_r3_train, "Run 3 (Continuous Q8) Train 3000 -> 6000 Steps")

    cmd_r3_causal = [
        sys.executable, str(base_dir / "scripts" / "ib_local" / "measure_cbim_unified_causality.py"),
        "--checkpoint", str(r3_6000 / "BBest.pt"),
        "--output", str(pub_dir / "cbim_continuous_q8_causality_6000.json"),
    ]
    run_cmd(cmd_r3_causal, "Run 3 (Continuous Q8) Causality Intervention Evaluation at Step 6000")

    # -------------------------------------------------------------
    # 3. Comparative Synthesis & Summary Report
    # -------------------------------------------------------------
    print("\n" + "#" * 70)
    print("# STEP 3: COMPARATIVE SYNTHESIS & REPORT")
    print("#" * 70, flush=True)

    c1 = json.loads((pub_dir / "cbim_torus3d_w2_16ch_k3_adaptive_xavier_causality_6000.json").read_text(encoding="utf-8"))
    c3 = json.loads((pub_dir / "cbim_continuous_q8_causality_6000.json").read_text(encoding="utf-8"))

    summary = {
        "steps": 6000,
        "run1_d3q8": {
            "mean_full_nll": c1["mean_full_nll"],
            "mean_collision_delta_nll": c1["mean_collision_delta_nll"],
            "mean_transport_delta_nll": c1["mean_transport_delta_nll"],
            "mean_joint_delta_nll": c1["mean_joint_delta_nll"],
            "diagnostics": c1.get("physical_diagnostics", {}),
        },
        "run3_continuous_q8": {
            "mean_full_nll": c3["mean_full_nll"],
            "mean_collision_delta_nll": c3["mean_collision_delta_nll"],
            "mean_transport_delta_nll": c3["mean_transport_delta_nll"],
            "mean_joint_delta_nll": c3["mean_joint_delta_nll"],
            "diagnostics": c3.get("physical_diagnostics", {}),
        },
    }
    summary_path = pub_dir / "cbim_comparison_6000_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary saved to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
