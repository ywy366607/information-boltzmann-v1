#!/usr/bin/env python3
"""Unified residual graph + OT flow matching on t2i (same knobs as other ports).

  python scripts/run_omni_unified_fm.py --device cuda
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.omni_tasks import GRID_PLACES
from scripts.train_omni_probe import main as train_omni_main

PUB = ROOT / "results" / "published"
OUT = PUB / "omni_unified_fm_t2i_2000step_table.json"
CMP = PUB / "omni_unified_fm_t2i_vs_l1.json"
OLD = PUB / "omni_unified_t2i_2000step_table.json"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--flow-steps", type=int, default=8)
    ap.add_argument("--ports", type=str, default="t2i")
    ap.add_argument("--init", type=str, default="")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--flow-t", type=str, default="uniform")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--mix", type=str, default="")
    ap.add_argument("--t2i-hint-frac", type=float, default=0.0)
    ap.add_argument("--fm-pred", type=str, default="x", choices=["x", "v"])
    ap.add_argument("--fm-x0", type=str, default="pair", choices=["pair", "noise"])
    ap.add_argument("--flow-method", type=str, default="heun", choices=["heun", "euler"])
    ap.add_argument("--t2i-canvas", type=str, default="paper", choices=["paper", "black"])
    ap.add_argument("--t2i-stroke-px", type=int, default=1)
    ap.add_argument(
        "--t2i-place", type=str, default="random",
        choices=["random", "center", "grid", *GRID_PLACES],
    )
    ap.add_argument("--t2i-digit", type=int, default=None)
    ap.add_argument("--t2i-color", type=str, default="")
    ap.add_argument("--prior-write", type=float, default=None)
    ap.add_argument("--f-gen", action="store_true")
    ap.add_argument(
        "--gen-recipe", choices=["core", "active_f2", "vfe"],
        default="active_f2",
    )
    ap.add_argument("--f-steps", type=int, default=1,
                    help="Legacy compatibility; native generation is one stack pass.")
    ap.add_argument("--f-halt-eps", type=float, default=0.03)
    ap.add_argument("--fm-signed", action="store_true")
    ap.add_argument("--cfg", type=float, default=1.0)
    args = ap.parse_args()
    os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")

    ports = [p.strip() for p in (args.mix or args.ports).split(",") if p.strip()]
    mix = ",".join(ports)
    results = {}
    tag = args.tag or ("unified_fm_" + "_".join(ports))
    out_path = PUB / f"omni_{tag}_{args.steps}step_table.json"
    print(
        f"=== UNIFIED FM  mix={mix}  pred={args.fm_pred}  recipe={args.gen_recipe}  "
        f"{args.flow_method}{args.flow_steps} t={args.flow_t} "
        f"x0={args.fm_x0} canvas={args.t2i_canvas} hint={args.t2i_hint_frac} ===",
        flush=True,
    )
    argv = [
        "train_omni_probe.py",
        "--device", args.device,
        "--steps", str(args.steps),
        "--d-model", "256",
        "--n-slices", "64",
        "--n-heads", "8",
        "--mix", mix,
        "--vfe-coef", "0.1",
        "--deslice-write", "increment",
        "--s-lang-topk", "0",
        "--flow-steps", str(args.flow_steps),
        "--flow-t", args.flow_t,
        "--t2i-hint-frac", str(args.t2i_hint_frac),
        "--fm-pred", args.fm_pred,
        "--fm-x0", args.fm_x0,
        "--flow-method", args.flow_method,
        "--t2i-canvas", args.t2i_canvas,
        "--t2i-stroke-px", str(args.t2i_stroke_px),
        "--t2i-place", args.t2i_place,
        "--gen-recipe", args.gen_recipe,
        "--cfg", str(args.cfg),
        "--tag", tag,
        "--out", str(out_path),
    ]
    if args.prior_write is not None:
        argv.extend(["--prior-write", str(args.prior_write)])
    if args.lr is not None:
        argv.extend(["--lr", str(args.lr)])
    if args.fm_signed:
        argv.append("--fm-signed")
    if args.f_gen:
        argv.extend([
            "--f-gen", "--f-steps", str(args.f_steps),
            "--f-halt-eps", str(args.f_halt_eps),
        ])
    else:
        argv.append("--flow-match")
    if args.t2i_digit is not None:
        argv.extend(["--t2i-digit", str(args.t2i_digit)])
    if args.t2i_color:
        argv.extend(["--t2i-color", args.t2i_color])
    if args.init:
        argv.extend(["--init", args.init])
    sys.argv = argv
    train_omni_main()
    table = json.loads(out_path.read_text(encoding="utf-8"))
    results = table.get("final") or {}

    ft = PUB / "omni_unified_fm_t2i_ft_2000step_table.json"
    prev = {}
    if ft.exists():
        prev = (json.loads(ft.read_text(encoding="utf-8")).get("final") or {}).get("t2i") or {}
    cmp_path = PUB / f"omni_{tag}_vs_prev.json"
    cmp_path.write_text(json.dumps({
        "note": "Unified FM. Mix/hint are data only; eval t2i has no gray hint.",
        "mix": ports,
        "t2i_hint_frac": args.t2i_hint_frac,
        "t2i_canvas": args.t2i_canvas,
        "fm_x0": args.fm_x0,
        "prev_pure_t2i_ft": prev,
        "this_run": results,
    }, indent=2), encoding="utf-8")
    print(f"saved {cmp_path}", flush=True)
    for p, rec in results.items():
        if p in ("t2t", "i2t", "it2t"):
            print(f"  FM {p}: acc={rec.get('acc',0)*100:.1f}", flush=True)
        else:
            print(
                f"  FM {p}: psnr={rec.get('psnr',0):.2f} bg={rec.get('bg_psnr',0):.2f} "
                f"str={rec.get('stroke_psnr',0):.1f} flood={rec.get('flood',0)*100:.1f} "
                f"ink={rec.get('ink',0)*100:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
