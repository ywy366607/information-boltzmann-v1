#!/usr/bin/env python3
"""Train each I/O port separately on the *same* unified residual graph.

  write = increment (Transolver skip)   F2 λ=0.1   no lang-topk   no write A
  Ports differ by data + loss only. Old F2 / ChampB ckpts are not overwritten.

  python scripts/run_omni_unified_ports.py --device cuda
  python scripts/run_omni_unified_ports.py --device cuda --skip-triad --ports recon,t2i
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.unified_arch import OMNI_PORTS, UNIFIED_KNOBS, UNIFIED_OMNI_SIZE
from scripts.train_omni_probe import main as train_omni_main
from scripts.train_triad_2000steps import train_single_run

PUB = ROOT / "results" / "published"
SUMMARY = PUB / "omni_unified_ports.json"


def _train_triad(device: str, steps: int) -> dict:
    print("=== UNIFIED triad  increment+F2  d=128 M=32  seed=42 ===", flush=True)
    res = train_single_run(
        arm="v1_bayes",
        run_idx=0,
        seed=42,
        steps=steps,
        device=device,
        s_update=UNIFIED_KNOBS["s_update"],
        use_stiefel=UNIFIED_KNOBS["use_stiefel"],
        deslice_topk=UNIFIED_KNOBS["deslice_topk"],
        gate_on=UNIFIED_KNOBS["gate_on"],
        deslice_write=UNIFIED_KNOBS["deslice_write"],
        gate_h_local=UNIFIED_KNOBS["gate_h_local"],
        vfe_coef=UNIFIED_KNOBS["vfe_coef"],
        s_kalman_update=UNIFIED_KNOBS["s_kalman_update"],
        ckpt_tag="unified_inc",
    )
    out = {
        "kind": "triad",
        "acc": res["final_acc"],
        "tasks": res["final_tasks"],
        "loss": res["final_loss"],
        "checkpoint": res.get("checkpoint"),
        "deslice_write": "increment",
        "vfe_coef": 0.1,
    }
    (PUB / "unified_inc_triad_2000step_table.json").write_text(
        json.dumps({"v1_bayes": {"note": "Unified residual write. One seed.", "runs": [res]}}, indent=2),
        encoding="utf-8",
    )
    print(
        f"  TRIAD acc={out['acc']*100:.1f} ocr={out['tasks'].get('ocr',0)*100:.1f} "
        f"kinks={out['tasks'].get('kinks',0)*100:.1f}",
        flush=True,
    )
    return out


def _train_port(port: str, device: str, steps: int) -> dict:
    out_path = PUB / f"omni_unified_{port}_2000step_table.json"
    print(f"=== UNIFIED omni/{port}  increment+F2  d=256 M=64 ===", flush=True)
    sys.argv = [
        "train_omni_probe.py",
        "--device", device,
        "--steps", str(steps),
        "--d-model", str(UNIFIED_OMNI_SIZE["d_model"]),
        "--n-slices", str(UNIFIED_OMNI_SIZE["n_slices"]),
        "--n-heads", str(UNIFIED_OMNI_SIZE["n_heads"]),
        "--mix", port,
        "--vfe-coef", str(UNIFIED_KNOBS["vfe_coef"]),
        "--deslice-write", UNIFIED_KNOBS["deslice_write"],
        "--s-lang-topk", "0",
        "--tag", f"unified_{port}",
        "--out", str(out_path),
    ]
    train_omni_main()
    table = json.loads(out_path.read_text(encoding="utf-8"))
    rec = (table.get("final") or {}).get(port) or {}
    rec = {
        "kind": port,
        "acc": rec.get("acc", 0.0),
        "psnr": rec.get("psnr", 0.0),
        "stroke_psnr": rec.get("stroke_psnr", 0.0),
        "bg_psnr": rec.get("bg_psnr", 0.0),
        "ink": rec.get("ink", 0.0),
        "flood": rec.get("flood", 0.0),
        "checkpoint": table.get("ckpt"),
        "deslice_write": table.get("deslice_write"),
        "vfe_coef": table.get("vfe_coef"),
    }
    return rec


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--ports", type=str, default=",".join(OMNI_PORTS))
    ap.add_argument("--skip-triad", action="store_true")
    ap.add_argument("--only-triad", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")

    ports = [p.strip() for p in args.ports.split(",") if p.strip()]
    for p in ports:
        if p not in OMNI_PORTS:
            raise SystemExit(f"unknown port {p!r}; choose from {OMNI_PORTS}")

    summary = {
        "arch": "unified residual increment + F2",
        "knobs": UNIFIED_KNOBS,
        "note": (
            "Same graph for every port. Separate training only. "
            "Does not overwrite F2 / ChampB / old omni ckpts."
        ),
        "ports": {},
    }
    if SUMMARY.exists():
        try:
            summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
            summary.setdefault("ports", {})
        except Exception:
            pass

    if not args.skip_triad:
        summary["ports"]["triad"] = _train_triad(args.device, args.steps)
        SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.only_triad:
        print(f"saved {SUMMARY}", flush=True)
        return

    for port in ports:
        summary["ports"][port] = _train_port(port, args.device, args.steps)
        SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nUNIFIED PORT SUMMARY", flush=True)
    for k, rec in summary["ports"].items():
        if k == "triad":
            t = rec.get("tasks") or {}
            print(
                f"  {k:6s} acc={rec.get('acc',0)*100:5.1f}  "
                f"ocr={t.get('ocr',0)*100:5.1f}  kinks={t.get('kinks',0)*100:5.1f}",
                flush=True,
            )
        elif k in ("t2t", "i2t", "it2t"):
            print(f"  {k:6s} acc={rec.get('acc',0)*100:5.1f}", flush=True)
        else:
            print(
                f"  {k:6s} psnr={rec.get('psnr',0):5.2f}  bg={rec.get('bg_psnr',0):5.2f}  "
                f"str={rec.get('stroke_psnr',0):5.1f}  flood={rec.get('flood',0)*100:5.1f}",
                flush=True,
            )
    print(f"saved {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
