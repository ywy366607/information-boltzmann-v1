#!/usr/bin/env python3
"""Smoke: L-layer 4-step cross-modal loop trains on digit probe (no LLM).

Prints per-layer trace norms so you can see each of the 4 steps is live.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.cross_modal_slice_loop import CrossModalSliceFrontend  # noqa: E402
from fine_grain.ocr_1px import N_OCR, make_ocr_1px  # noqa: E402


class Probe(nn.Module):
    def __init__(self, fe, d):
        super().__init__()
        self.fe = fe
        self.head = nn.Linear(d, N_OCR)

    def forward(self, img):
        out = self.fe(img)
        return self.head(out.tokens.mean(1)), out.meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/published/xmodal_loop_smoke.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    d = 64
    fe = CrossModalSliceFrontend(
        d_llm=d, res=args.res, T=args.T, n_layers=args.n_layers,
        shared_layer=True, projector="linear", writeback=True,
    )
    model = Probe(fe, d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(0)
    hist = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        img, lab, _ = make_ocr_1px(rng, n=args.batch, res=args.res, hard_frac=0.3)
        img, lab = img.to(device), lab.to(device)
        logits, meta = model(img)
        loss = F.cross_entropy(logits, lab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 20 == 0 or step in (1, args.steps):
            acc = (logits.argmax(-1) == lab).float().mean().item()
            traces = meta.get("layer_traces") or []
            hist.append({
                "step": step,
                "loss": float(loss.item()),
                "train_acc": float(acc),
                "layer_traces": traces,
            })
            tr_s = ", ".join(
                f"L{t['layer']}:xΔ={t['x_delta_norm']:.3f}/HΔ={t['h_delta_norm']:.3f}/Hent={t['read_entropy']:.2f}"
                for t in traces
            )
            print(f"step {step:3d} loss={loss.item():.4f} acc={acc:.3f} | {tr_s}", flush=True)

    # held-out
    model.eval()
    hit = tot = 0
    rng_v = np.random.default_rng(90_001)
    with torch.no_grad():
        for _ in range(16):
            img, lab, _ = make_ocr_1px(rng_v, n=32, res=args.res, hard_frac=0.3)
            img, lab = img.to(device), lab.to(device)
            logits, _ = model(img)
            hit += int((logits.argmax(-1) == lab).sum())
            tot += lab.numel()
    acc = hit / max(tot, 1)
    payload = {
        "task": "xmodal_4step_loop_smoke",
        "n_layers": args.n_layers,
        "steps": args.steps,
        "res": args.res,
        "T": args.T,
        "val_acc": acc,
        "chance": 1.0 / N_OCR,
        "history": hist,
        "contract": [
            "1 X→Slice→interact→deslice→X",
            "2 H dynamic read slices",
            "3 text state update",
            "4 optional deslice writeback H→X",
        ],
        "seconds": time.time() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"val_acc={acc:.3f} (chance={1/N_OCR:.2f}) → {args.out}", flush=True)


if __name__ == "__main__":
    main()
