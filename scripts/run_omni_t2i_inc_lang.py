#!/usr/bin/env python3
"""t2i: increment write + language only on top-k slices.

Classification default stays write A / all slices.
Does not overwrite omni_d256_t2i_best.pt or omni_d256_t2i_f2_best.pt.

  python scripts/run_omni_t2i_inc_lang.py --device cuda
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_omni_probe import main as train_main

PUB_F2 = ROOT / "results" / "published" / "omni_t2i_f2_2000step_table.json"
PUB_OLD = ROOT / "results" / "published" / "omni_t2i_2000step_table.json"
OUT = ROOT / "results" / "published" / "omni_t2i_inc_lang_2000step_table.json"
CMP = ROOT / "results" / "published" / "omni_t2i_inc_lang_vs_f2.json"
FIG = ROOT / "present" / "figs"


def _t2i(table_path: Path) -> dict:
    if not table_path.exists():
        return {}
    t = json.loads(table_path.read_text(encoding="utf-8"))
    return (t.get("final") or {}).get("t2i") or {}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--s-lang-topk", type=int, default=8)
    args = ap.parse_args()
    os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")

    print(
        f"=== t2i  increment + lang topk={args.s_lang_topk}  "
        f"F2-λ=0.1  d=256 M=64  seed=42 ===",
        flush=True,
    )
    sys.argv = [
        "train_omni_probe.py",
        "--device", args.device,
        "--steps", str(args.steps),
        "--d-model", "256",
        "--n-slices", "64",
        "--n-heads", "8",
        "--mix", "t2i",
        "--vfe-coef", "0.1",
        "--deslice-write", "increment",
        "--s-lang-topk", str(args.s_lang_topk),
        "--tag", "t2i_inc_lang",
        "--out", str(OUT),
    ]
    train_main()

    new = _t2i(OUT)
    f2 = _t2i(PUB_F2)
    old = _t2i(PUB_OLD)
    cmp = {
        "note": "Increment write + top-k slices take language Δ. Product default unchanged.",
        "s_lang_topk": args.s_lang_topk,
        "deslice_write": "increment",
        "old_champb_mismatch_or_flood": old,
        "f2_write_A": f2,
        "inc_lang": new,
    }
    CMP.write_text(json.dumps(cmp, indent=2), encoding="utf-8")

    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0f172a")
    labs = ["PSNR", "Stroke", "BG", "Ink×20", "Flood×20"]
    def row(rec):
        if not rec:
            return [0, 0, 0, 0, 0]
        return [
            float(rec.get("psnr") or 0),
            float(rec.get("stroke_psnr") or 0),
            float(rec.get("bg_psnr") or 0),
            float(rec.get("ink") or 0) * 20.0,
            float(rec.get("flood") or 0) * 20.0,
        ]
    x = np.arange(5)
    ax.bar(x - 0.2, row(f2), 0.4, label="F2 write A", color="#2dd4bf")
    ax.bar(x + 0.2, row(new), 0.4, label=f"inc + lang k={args.s_lang_topk}", color="#c084fc")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, color="#e2e8f0")
    ax.tick_params(colors="#94a3b8")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
    ax.set_title("t2i  ·  write A vs increment+lang-topk", color="white", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "omni_t2i_inc_lang_vs_f2_bars.png", facecolor=fig.get_facecolor())
    plt.close()
    print(f"saved {CMP}", flush=True)


if __name__ == "__main__":
    main()
