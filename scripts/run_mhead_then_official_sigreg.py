#!/usr/bin/env python3
"""1) 2000-step multi-head prior ablation (same readout+λ as readout_l01).
   2) Pick the best of {1-head readout_l01, 4-head mhead} × {jepa, bayes}.
   3) Re-run that arm with official Epps-Pulley SIGReg and live S (no pred detach).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TRIAD = ROOT / "scripts" / "train_triad_2000steps.py"
ONEHEAD = ROOT / "results" / "published" / "readout_l01_2000step_table.json"
MHEAD = ROOT / "results" / "published" / "mhead_2000step_table.json"


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def arm_score(table: dict, arm: str) -> float:
    if arm not in table:
        return float("-inf")
    block = table[arm]
    if "mean_acc" in block:
        return float(block["mean_acc"])
    runs = block.get("runs") or []
    if not runs:
        return float("-inf")
    return float(sum(r["final_acc"] for r in runs) / len(runs) * 100.0)


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    run([
        PY, str(TRIAD),
        "--arms", "v0_jepa,v1_bayes",
        "--steps", "2000",
        "--runs", "3",
        "--device", "cuda",
        "--prior-loss-coef", "0.1",
        "--detach-pred-target",
        "--sigreg-coef", "0.0",
        "--tag", "mhead",
        "--out", str(MHEAD),
    ])

    one = load(ONEHEAD)
    four = load(MHEAD)
    candidates = [
        ("v0_jepa", "1head", arm_score(one, "v0_jepa")),
        ("v1_bayes", "1head", arm_score(one, "v1_bayes")),
        ("v0_jepa", "4head", arm_score(four, "v0_jepa")),
        ("v1_bayes", "4head", arm_score(four, "v1_bayes")),
    ]
    print("\n=== ablation scores (mean acc %) ===", flush=True)
    for arm, heads, acc in candidates:
        print(f"  {arm:10s} {heads}: {acc:.2f}", flush=True)
    best_arm, best_heads, best_acc = max(candidates, key=lambda t: t[2])
    print(f"\nBEST: {best_arm} {best_heads}  {best_acc:.2f}%", flush=True)
    print("Next: official SIGReg + no pred detach on that arm only.", flush=True)

    out = ROOT / "results" / "published" / f"official_sig_{best_arm}_2000step_table.json"
    run([
        PY, str(TRIAD),
        "--arms", best_arm,
        "--steps", "2000",
        "--runs", "3",
        "--device", "cuda",
        "--prior-loss-coef", "0.1",
        "--no-detach-pred-target",
        "--sigreg-coef", "0.05",
        "--tag", f"osig_{best_heads}",
        "--out", str(out),
    ])
    print(f"\nWrote official-SIGReg table to {out}", flush=True)


if __name__ == "__main__":
    main()
