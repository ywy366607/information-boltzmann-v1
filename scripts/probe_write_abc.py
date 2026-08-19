#!/usr/bin/env python3
"""A/B/C write-operator probe on Champion B weights (no H/Local gate).

  A: X += D(S)              absolute broadcast
  B: X += D(S+ − S)         time increment (P0)
  C: X += D(S+ − Read(W))   workspace consistency (W ≠ evidence)

  python scripts/probe_write_abc.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.vlm_data import make_vqa_batch
from scripts.probe_deslice_memory import (
    CKPTS,
    _cos,
    apply_knobs,
    load_model,
    task_eval,
)

FIG = ROOT / "present" / "figs"
OUT = ROOT / "results" / "published" / "write_abc_probe.json"
ABC = [
    ("A_absolute", "absolute"),
    ("B_increment", "increment"),
    ("C_workspace", "workspace"),
]


@torch.no_grad()
def abc_trace(model, img, prompts, device) -> dict:
    tok_ids, mask = model.tokenize(prompts, device)
    text_emb = model.embed(tok_ids)
    E = model.mot_stack.encode_X(img)
    X, H = E, model.mot_stack.text_in(text_emb)
    W = torch.zeros_like(X)
    rows = []
    for i, layer in enumerate(model.mot_stack.layers):
        S_in, _ = layer.read(X)
        S_w0, _ = layer.read(W)
        X2, H, tr = layer(X, H, text_mask=mask, prompt_mask=mask, layer_idx=i, W=W)
        S_re, _ = layer.read(X2)
        dW = getattr(layer, "last_dW", X2.new_zeros(X2.shape))
        if str(layer.deslice_write) in ("workspace", "ws", "consistency", "consist"):
            W = W + dW
        S_w1, _ = layer.read(W)
        rows.append({
            "layer": i,
            "U": tr.surprise_u,
            "x_delta": tr.x_delta,
            "rms_X": float(X2.pow(2).mean().sqrt()),
            "C_ws": float(getattr(layer, "last_C", 0.0)),
            "dS_time": float((layer.last_S_write - layer.last_S).pow(2).mean().sqrt()),
            "dS_ws": float((layer.last_S_write - S_w0).pow(2).mean().sqrt()),
            "cos_reread_Swrite": _cos(S_re, layer.last_S_write),
            "cos_SW_Sin": _cos(S_w0, S_in),
            "cos_SWafter_Swrite": _cos(S_w1, layer.last_S_write),
        })
        X = X2
    return rows


@torch.no_grad()
def freeze_s_identity(model, img, prompts, device) -> dict:
    tok_ids, mask = model.tokenize(prompts, device)
    text_emb = model.embed(tok_ids)
    X = model.mot_stack.encode_X(img)
    H = model.mot_stack.text_in(text_emb)
    W = torch.zeros_like(X)
    z = torch.zeros(img.shape[0], model.mot_stack.n_slices, 1, device=device)
    rows = []
    for i, layer in enumerate(model.mot_stack.layers):
        S0, _ = layer.read(X)
        X2, H2, _ = layer(X, H, text_mask=mask, force_gate=z, layer_idx=i, W=W)
        S1, _ = layer.read(X2)
        dW = getattr(layer, "last_dW", X2.new_zeros(()))
        if str(layer.deslice_write) in ("workspace", "ws", "consistency", "consist"):
            W = W + dW
        rows.append({
            "layer": i,
            "dx": float((X2 - X).norm(dim=-1).mean()),
            "d_read": float((S1 - S0).norm(dim=-1).mean()),
            "cos_read": _cos(S1, S0),
            "C": float(getattr(layer, "last_C", 0.0)),
        })
        X, H = X2, H2
    return rows


def render(all_rows: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    colors = {"A_absolute": "#fbbf24", "B_increment": "#fb7185", "C_workspace": "#2dd4bf"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    tasks = ["ocr", "kinks", "color"]
    for ax, ck in zip(axes, ("champ_b", "p0_id")):
        if ck not in all_rows:
            continue
        ax.set_facecolor("#0f172a")
        x = np.arange(3)
        w = 0.24
        for i, (tag, _) in enumerate(ABC):
            ev = all_rows[ck][tag]["eval"]
            ax.bar(x + (i - 1) * w, [ev[t] * 100 for t in tasks], w, label=tag, color=colors[tag])
        ax.set_xticks(x)
        ax.set_xticklabels(["OCR", "Kinks", "Color"], color="#e2e8f0")
        ax.set_ylim(0, 115)
        ax.set_title(ck + "  (ungated H/Local)", color="white", fontweight="bold")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, axis="y", ls=":", alpha=0.3)
        if ck == "champ_b":
            ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
    fig.suptitle("Write operators A / B / C   eval-only, same weights", color="white", fontweight="bold")
    fig.tight_layout()
    p = FIG / "write_abc_tasks.png"
    fig.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  saved {p}", flush=True)

    if "champ_b" not in all_rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.7), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    getters = [
        ("C = ||S+ − Read(W)||", lambda r: r["C_ws"]),
        ("||S+ − S_read|| time", lambda r: r["dS_time"]),
        ("cos(Read(W'), S+)", lambda r: r["cos_SWafter_Swrite"]),
    ]
    for ax, (title, fn) in zip(axes, getters):
        ax.set_facecolor("#0f172a")
        for tag, _ in ABC:
            ys = [fn(r) for r in all_rows["champ_b"][tag]["trace_kinks"]]
            ax.plot(range(len(ys)), ys, "o-", color=colors[tag], label=tag, lw=2)
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xticks([0, 1, 2, 3])
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, ls=":", alpha=0.3)
    axes[0].legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=7)
    fig.suptitle("Champion B  ·  kinks  ·  consistency vs time increment", color="white", fontweight="bold")
    fig.tight_layout()
    p2 = FIG / "write_abc_consistency.png"
    fig.savefig(p2, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  saved {p2}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batches", type=int, default=20)
    args = ap.parse_args()
    os.environ.setdefault("ML_CACHE_ROOT", r"D:\\ml_cache")
    device = torch.device(args.device)

    rng = np.random.default_rng(7)
    bk = make_vqa_batch(rng, batch=16, res=32, mix=["kinks"])
    all_rows = {}
    for ck_name, ck_path in CKPTS.items():
        if not ck_path.exists():
            print(f"SKIP {ck_name}", flush=True)
            continue
        print(f"\n=== {ck_name} ===", flush=True)
        model = load_model(ck_path, device)
        all_rows[ck_name] = {}
        img, pr = bk["image"].to(device), bk["prompt"]
        for tag, write in ABC:
            apply_knobs(model, write, gated=False)
            ev = task_eval(model, device, n_batches=args.batches)
            tr = abc_trace(model, img, pr, device)
            ident = freeze_s_identity(model, img, pr, device)
            all_rows[ck_name][tag] = {"write": write, "eval": ev, "trace_kinks": tr, "identity": ident}
            print(
                f"  {tag:14s} acc={ev['acc']*100:5.1f} ocr={ev['ocr']*100:5.1f} "
                f"kinks={ev['kinks']*100:5.1f}  C_L3={tr[-1]['C_ws']:.3f}  "
                f"cosW={tr[-1]['cos_SWafter_Swrite']:.3f}  "
                f"g0 dRead={ident[-1]['d_read']:.3f}",
                flush=True,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    OUT.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"saved {OUT}", flush=True)
    render(all_rows)


if __name__ == "__main__":
    main()
