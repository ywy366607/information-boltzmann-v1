#!/usr/bin/env python3
"""Phase 1.1: line-recon sweep over recur_T ∈ {1,2,4}.

Uses shipped line_recon path + ARMS slice_loc_nogumbel_recur{1,2,4}.
Matched budget across cells; defaults are 4GB-friendly (res=32, modest steps).

Example:
  python scripts/run_recur_recon_sweep.py --res 32 --steps 150 --rgb --amp
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse line_recon runners
from scripts import line_recon as LR  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--probe_every", type=int, default=50)
    ap.add_argument("--eval_n", type=int, default=128)
    ap.add_argument("--eval_batch", type=int, default=16)
    ap.add_argument("--hard_frac", type=float, default=0.35)
    ap.add_argument("--hard_tile", type=int, default=16)
    ap.add_argument("--k_min", type=int, default=5)
    ap.add_argument("--k_max", type=int, default=10)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--slice_num", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pos_weight", type=float, default=40.0)
    ap.add_argument("--dice_w", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--rgb", action="store_true", default=True)
    ap.add_argument("--no_rgb", action="store_true")
    ap.add_argument("--patch_decoder", default="unpatchify")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument(
        "--arms",
        default="slice_loc_nogumbel_recur1,slice_loc_nogumbel_recur2,slice_loc_nogumbel_recur4",
    )
    ap.add_argument("--ckpt_dir", default="checkpoints/line_recon_recur")
    ap.add_argument("--out", default="results/published/recur_recon_table.json")
    ap.add_argument("--conclusion", default="results/published/recur_recon_conclusion.md")
    args = ap.parse_args()
    if args.no_rgb:
        args.rgb = False

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"recur recon sweep | res={args.res} steps={args.steps} B={args.batch} "
        f"arms={args.arms} rgb={args.rgb} device={device}",
        flush=True,
    )

    results = []
    t0 = time.time()
    for name in args.arms.split(","):
        name = name.strip()
        if not name:
            continue
        print(f"\n=== {name} ===", flush=True)
        try:
            row = LR.run_arm(name, args, device)
            results.append({"status": "ok", **row})
        except Exception as e:
            print(f"  FAIL {e}", flush=True)
            results.append({"status": "error", "arm": name, "error": str(e)[:300]})
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # summary table by recur_T
    ok = [r for r in results if r.get("status") == "ok"]
    by_recur = {}
    for r in ok:
        rt = int(r.get("recur_T", 1))
        by_recur[rt] = r

    payload = {
        "task": "line_recon_recur_T_sweep",
        "res": args.res,
        "batch": args.batch,
        "steps": args.steps,
        "rgb": bool(args.rgb),
        "hard_frac": args.hard_frac,
        "seed": args.seed,
        "probe_every": args.probe_every,
        "arms": results,
        "seconds_total": time.time() - t0,
        "comparison": {
            str(k): {
                "dice": v["final"]["dice"],
                "iou": v["final"]["iou"],
                "recall": v["final"]["recall"],
                "step": v["final"]["step"],
            }
            for k, v in sorted(by_recur.items())
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# Line recon recurrence sweep (`recur_T`)",
        "",
        f"- Task: 1px polyline mask recon; rgb={args.rgb} res={args.res} "
        f"steps={args.steps} batch={args.batch} hard_frac={args.hard_frac}",
        f"- Arms: `{args.arms}` (same family as `slice_loc_nogumbel`)",
        f"- Seed={args.seed}; matched budget across cells",
        "",
        "| recur_T | arm | dice | iou | recall | final_step | status |",
        "|---------|-----|------|-----|--------|------------|--------|",
    ]
    for r in results:
        if r.get("status") != "ok":
            lines.append(
                f"| — | {r.get('arm')} | — | — | — | — | {r.get('error')} |"
            )
        else:
            f = r["final"]
            lines.append(
                f"| {r.get('recur_T', 1)} | {r['arm']} | {f['dice']:.3f} | "
                f"{f['iou']:.3f} | {f['recall']:.3f} | {f['step']} | ok |"
            )
    lines.extend(["", "## Relative recurrence", ""])
    if len(by_recur) >= 2:
        base = by_recur.get(1) or ok[0]
        d1 = base["final"]["dice"]
        for rt in sorted(by_recur):
            d = by_recur[rt]["final"]["dice"]
            lines.append(f"- recur_T={rt}: dice={d:.3f} (Δ vs T=1: {d - d1:+.3f})")
        best = max(ok, key=lambda x: x["final"]["dice"])
        lines.append(
            f"- Best: **{best['arm']}** recur_T={best.get('recur_T')} "
            f"dice={best['final']['dice']:.3f}"
        )
        # early history: first probe after 0
        lines.append("")
        lines.append("### Learning speed (history dice)")
        for rt in sorted(by_recur):
            hist = by_recur[rt].get("history") or []
            pts = ", ".join(f"s{h['step']}={h['dice']:.3f}" for h in hist[:6])
            lines.append(f"- recur_T={rt}: {pts}")
    else:
        lines.append("- Not enough successful cells for comparison.")
    lines.append("")
    lines.append(
        "Note: this is Phase-1 recurrence relative gain (encoder), "
        "not frozen-LM free-gen OCR."
    )
    Path(args.conclusion).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"json → {args.out}", flush=True)


if __name__ == "__main__":
    main()
