#!/usr/bin/env python3
"""Same z, same prompt, vary t. If x_θ barely moves, time is unused."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")

from fine_grain.omni_model import DualStreamOmni
from fine_grain.omni_tasks import one_sample
from fine_grain.unified_arch import UNIFIED_OMNI_SIZE

CKPTS = {
    "paper_xpred": ROOT / "checkpoints" / "omni_d256_unified_jit_t2i_best.pt",
    "noise_xpred": ROOT / "checkpoints" / "omni_d256_unified_jit_noise_best.pt",
}
TS = (0.05, 0.25, 0.50, 0.75, 0.95)
OUT = ROOT / "results" / "published" / "jit_time_sensitivity.json"


def _rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).pow(2).mean().sqrt())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.reshape(-1)
    y = b.reshape(-1)
    return float(torch.nn.functional.cosine_similarity(x, y, dim=0))


@torch.no_grad()
def probe_one(ckpt: Path, device: torch.device) -> dict:
    model = DualStreamOmni.unified(**UNIFIED_OMNI_SIZE, fm_pred="x").to(device)
    raw = torch.load(ckpt, map_location="cpu")
    missing = model.load_state_dict(raw, strict=False)
    model.eval()
    rng = np.random.default_rng(0)
    s = one_sample(rng, 32, "t2i", t2i_canvas="paper")
    paper = s["image"].to(device)
    tgt = s["target_rgb"].to(device)
    prompt = [s["prompt"]]
    need = [True]

    stem = model.mot_stack.time_cond.net[-1]
    layer_norms = []
    for layer in model.mot_stack.layers:
        last = layer.time_mod.net[-1]
        layer_norms.append({
            "w": float(last.weight.detach().abs().mean()),
            "b": float(last.bias.detach().abs().mean()),
        })

    rec = {
        "ckpt": str(ckpt),
        "missing": int(len(missing.missing_keys)),
        "missing_time_mod": sum(1 for k in missing.missing_keys if "time_mod" in k),
        "stem_time_w": float(stem.weight.detach().abs().mean()),
        "stem_time_b": float(stem.bias.detach().abs().mean()),
        "layer_time_mod": layer_norms,
        "prompt": s["prompt"],
        "starts": {},
    }

    starts = {
        "paper": paper,
        "noise": torch.randn_like(paper),
        "mid_paper": 0.5 * paper + 0.5 * tgt,
        "mid_noise": 0.5 * torch.randn_like(paper) + 0.5 * tgt,
    }
    for name, z in starts.items():
        xs = {}
        for tv in TS:
            t = torch.full((1,), float(tv), device=device)
            xs[tv] = model(z, prompt, need_pix=need, t=t)["x_pred"]
        pairs = {}
        for i, t0 in enumerate(TS):
            for t1 in TS[i + 1 :]:
                pairs[f"{t0:.2f}_vs_{t1:.2f}"] = {
                    "rms": _rms(xs[t0], xs[t1]),
                    "cos": _cos(xs[t0], xs[t1]),
                    "max": float((xs[t0] - xs[t1]).abs().max()),
                }
        rec["starts"][name] = {
            "vs_pairs": pairs,
            "mean_rms": float(np.mean([p["rms"] for p in pairs.values()])),
            "min_cos": float(np.min([p["cos"] for p in pairs.values()])),
            "x_mean_by_t": {f"{tv:.2f}": float(xs[tv].mean()) for tv in TS},
        }
    return rec


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for name, ckpt in CKPTS.items():
        if not ckpt.exists():
            out[name] = {"error": f"missing {ckpt}"}
            print(f"skip {name}: no ckpt", flush=True)
            continue
        rec = probe_one(ckpt, device)
        out[name] = rec
        print(f"=== {name}  stem|w|={rec['stem_time_w']:.4e}  "
              f"missing_time_mod={rec['missing_time_mod']} ===", flush=True)
        for sn, sr in rec["starts"].items():
            print(
                f"  {sn:10s} mean_rms={sr['mean_rms']:.5f}  min_cos={sr['min_cos']:.4f}  "
                f"x_mean={sr['x_mean_by_t']}",
                flush=True,
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
