#!/usr/bin/env python3
"""What each recur loop looks at vs F2's four unshared layers.

  python scripts/probe_recur_look.py --device cuda
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.bayesian_surprise import spatial_surprise_map
from fine_grain.vlm_data import make_vqa_batch
from scripts.run_v0_surprise_eval import DualStreamVQAModel

RECUR_CKPT = ROOT / "checkpoints" / "v1_bayes_2000step_recur_k4_delta_run1_best.pt"
F2_CKPT = ROOT / "checkpoints" / "v1_bayes_2000step_f2_vfe_run1_best.pt"
OUT_JSON = ROOT / "results" / "published" / "recur_look_probe.json"
FIG = ROOT / "present" / "figs"


def _load(ckpt: Path, share: bool, device: torch.device) -> DualStreamVQAModel:
    m = DualStreamVQAModel(
        d_model=128, n_slices=32, n_layers=4, res=32,
        surprise_mode="v1_bayes", surprise_beta=1.5,
        s_update="rms_dir", use_stiefel=True, deslice_topk=2, n_heads=4,
        gate_on="u", deslice_write="absolute", gate_h_local=False,
        share_layers=share, n_loops=4,
    ).to(device)
    raw = torch.load(ckpt, map_location="cpu")
    miss = m.load_state_dict(raw, strict=False)
    print(f"  load {ckpt.name} miss={len(miss.missing_keys)} extra={len(miss.unexpected_keys)}", flush=True)
    m.eval()
    return m


def _hook_snaps(model: DualStreamVQAModel):
    snaps = []

    def hook(mod, _inp, _out):
        w = getattr(mod, "last_w", None)
        u = getattr(mod, "last_u", None)
        g = getattr(mod, "last_gate", None)
        x = getattr(mod, "last_X", None)
        if w is None:
            return
        snaps.append({
            "w": w.detach().cpu(),
            "u": None if u is None else u.detach().cpu(),
            "gate": None if g is None else g.detach().cpu(),
            "X": None if x is None else x.detach().cpu(),
        })

    handles = [ly.register_forward_hook(hook) for ly in model.mot_stack.layers]
    return snaps, handles


@torch.no_grad()
def _forward_snaps(model, img, prompt, device):
    snaps, handles = _hook_snaps(model)
    try:
        model(img.to(device), [prompt])
    finally:
        for h in handles:
            h.remove()
    return snaps


def _u_map(snap, res: int) -> np.ndarray:
    if snap["u"] is None:
        return np.zeros((res, res), np.float32)
    m = spatial_surprise_map(snap["u"], snap["w"], res=res)
    return m[0, 0].numpy()


def _x_map(snap, res: int) -> np.ndarray:
    x = snap["X"]
    if x is None:
        return np.zeros((res, res), np.float32)
    return x[0].norm(dim=-1).view(res, res).numpy()


def _top_mask(arr: np.ndarray, frac: float = 0.15) -> np.ndarray:
    flat = arr.reshape(-1)
    k = max(1, int(round(frac * flat.size)))
    thr = np.partition(flat, -k)[-k]
    return arr >= thr


def _stats_for_snaps(snaps, res: int, stroke: np.ndarray | None) -> dict:
    umaps = [_u_map(s, res) for s in snaps]
    xmaps = [_x_map(s, res) for s in snaps]
    w_flat = [s["w"][0].reshape(-1).float() for s in snaps]
    out = {
        "n": len(snaps),
        "mean_u": [float(s["u"].mean()) if s["u"] is not None else 0.0 for s in snaps],
        "mean_gate": [float(s["gate"].mean()) if s["gate"] is not None else 1.0 for s in snaps],
        "x_rms": [float(m.mean()) for m in xmaps],
        "w_cos": [],
        "u_top_jaccard": [],
        "u_new_frac": [],
        "stroke_u_mass": [],
    }
    prev_top = None
    union = np.zeros((res, res), dtype=bool)
    for i, um in enumerate(umaps):
        top = _top_mask(um, 0.15)
        if prev_top is None:
            out["u_top_jaccard"].append(1.0)
            out["u_new_frac"].append(1.0)
        else:
            inter = np.logical_and(top, prev_top).sum()
            uni = np.logical_or(top, prev_top).sum()
            out["u_top_jaccard"].append(float(inter / max(1, uni)))
            new = np.logical_and(top, np.logical_not(prev_top)).sum()
            out["u_new_frac"].append(float(new / max(1, top.sum())))
        if i > 0:
            a, b = w_flat[i - 1], w_flat[i]
            out["w_cos"].append(float(torch.nn.functional.cosine_similarity(a, b, dim=0)))
        union |= top
        prev_top = top
        if stroke is not None:
            ssum = float(um[stroke].sum())
            tsum = float(um.sum()) + 1e-8
            out["stroke_u_mass"].append(ssum / tsum)
    out["union_coverage"] = float(union.mean())
    return out, umaps, xmaps


def _stroke_mask(img: torch.Tensor, res: int) -> np.ndarray:
    # 1px ink: not near-black background
    rgb = img[0].cpu().numpy()
    lum = rgb.mean(axis=0)
    return lum > 0.15


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")
    device = torch.device(args.device)

    print("loading models", flush=True)
    recur = _load(RECUR_CKPT, share=True, device=device)
    f2 = _load(F2_CKPT, share=False, device=device)

    rng = np.random.default_rng(42)
    batches = {
        "kinks": make_vqa_batch(rng, batch=1, res=32, mix=["kinks"]),
        "ocr": make_vqa_batch(rng, batch=1, res=32, mix=["ocr"]),
        "color": make_vqa_batch(rng, batch=1, res=32, mix=["color"]),
    }

    report = {}
    fig, axes = plt.subplots(3, 9, figsize=(18, 6.2), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    for r, task in enumerate(["kinks", "ocr", "color"]):
        b = batches[task]
        img = b["image"]
        prompt = b["prompt"][0]
        stroke = _stroke_mask(img, 32)
        rec_snaps = _forward_snaps(recur, img, prompt, device)
        f2_snaps = _forward_snaps(f2, img, prompt, device)
        rec_st, rec_u, rec_x = _stats_for_snaps(rec_snaps, 32, stroke)
        f2_st, f2_u, _ = _stats_for_snaps(f2_snaps, 32, stroke)
        rec_pred = recur(img.to(device), [prompt])["logits"].argmax(-1).item()
        f2_pred = f2(img.to(device), [prompt])["logits"].argmax(-1).item()
        tgt = b["answer"][0]
        report[task] = {
            "prompt": prompt,
            "answer": tgt,
            "recur_pred": recur.answers[rec_pred] if rec_pred < len(recur.answers) else str(rec_pred),
            "f2_pred": f2.answers[f2_pred] if f2_pred < len(f2.answers) else str(f2_pred),
            "recur": rec_st,
            "f2": f2_st,
        }
        vis = img[0].permute(1, 2, 0).cpu().numpy()
        ax0 = axes[r, 0]
        ax0.imshow(np.clip(vis, 0, 1))
        ax0.set_title(f"{task}\n{tgt}", color="#e2e8f0", fontsize=8)
        ax0.axis("off")
        ax0.set_facecolor("#0f172a")
        for k in range(4):
            ax = axes[r, 1 + k]
            ax.imshow(rec_u[k], cmap="magma")
            ax.set_title(f"R0 L{k} U", color="#c4b5fd", fontsize=8)
            ax.axis("off")
        for k in range(4):
            ax = axes[r, 5 + k]
            ax.imshow(f2_u[k], cmap="magma")
            ax.set_title(f"F2 L{k} U", color="#7dd3fc", fontsize=8)
            ax.axis("off")

    FIG.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Where Bayes looks  ·  R0 shared loops vs F2 unshared layers", color="white", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "recur_look_vs_f2.png", facecolor=fig.get_facecolor())
    plt.close()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: {kk: report[k][kk] for kk in ("answer", "recur_pred", "f2_pred", "recur", "f2") if kk != "prompt"} for k in report}, indent=2))
    print(f"saved {OUT_JSON} and {FIG / 'recur_look_vs_f2.png'}", flush=True)


if __name__ == "__main__":
    main()
