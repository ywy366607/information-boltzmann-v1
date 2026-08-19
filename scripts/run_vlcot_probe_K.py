#!/usr/bin/env python3
"""Phase 1.2: closed-set digit probe sweep over latent_steps K ∈ {1,2,4}.

No LLM. Mean-pool (or last) tokens → Linear(10). Compares VisualLatentCoTSlice
K steps vs static PatchFrontend baseline.

Example:
  python scripts/run_vlcot_probe_K.py --steps 400 --res 32
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

from fine_grain.frontends import PatchFrontend  # noqa: E402
from fine_grain.ocr_1px import N_OCR, make_ocr_1px  # noqa: E402
from fine_grain.visual_latent_cot import VisualLatentCoTSlice  # noqa: E402


class DigitProbe(nn.Module):
    def __init__(self, frontend: nn.Module, d_feat: int, n_cls: int = N_OCR):
        super().__init__()
        self.frontend = frontend
        self.head = nn.Linear(d_feat, n_cls)

    def forward(self, img: torch.Tensor):
        out = self.frontend(img)
        pooled = out.tokens.mean(dim=1)
        return self.head(pooled), out.meta


def train_eval(kind: str, K: int, args, device):
    torch.manual_seed(args.seed)
    d_feat = args.d_feat
    if kind == "A":
        fe = PatchFrontend(
            d_feat, res=args.res, patch=args.patch, dim=args.dim,
            depth=args.depth, T=args.T, projector="linear",
        )
        label = "A_patch"
    else:
        fe = VisualLatentCoTSlice(
            d_feat, res=args.res, T=args.T, dim=args.dim, depth=args.depth,
            deslice_topk=args.topk, projector="linear",
            latent_steps=K, beta=args.beta, gamma=args.gamma,
        )
        label = f"B_vlcot_K{K}"
    model = DigitProbe(fe, d_feat).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(args.seed + 3)
    hist = []
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        img, lab, _ = make_ocr_1px(
            rng, n=args.batch, res=args.res, hard_frac=args.hard_frac,
        )
        img, lab = img.to(device), lab.to(device)
        logits, meta = model(img)
        loss = F.cross_entropy(logits, lab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % args.log_every == 0 or step in (1, args.steps):
            acc = (logits.argmax(-1) == lab).float().mean().item()
            hist.append({
                "step": step,
                "loss": float(loss.item()),
                "train_acc": float(acc),
            })
            print(
                f"  [{label}] step {step:4d} loss={loss.item():.4f} "
                f"train_acc={acc:.3f}",
                flush=True,
            )

    # held-out eval
    model.eval()
    rng_v = np.random.default_rng(args.val_seed)
    hit = tot = 0
    with torch.no_grad():
        left = args.probe_n
        while left > 0:
            b = min(args.batch, left)
            img, lab, _ = make_ocr_1px(
                rng_v, n=b, res=args.res, hard_frac=args.hard_frac,
            )
            img, lab = img.to(device), lab.to(device)
            logits, _ = model(img)
            hit += int((logits.argmax(-1) == lab).sum().item())
            tot += b
            left -= b
    model.train()
    acc = hit / max(tot, 1)
    return {
        "status": "ok",
        "kind": label,
        "base": kind,
        "latent_steps": K if kind != "A" else 1,
        "T": args.T,
        "final_loss": hist[-1]["loss"] if hist else float("nan"),
        "acc_digit": acc,
        "n_eval": tot,
        "history": hist,
        "seconds": time.time() - t0,
        "chance": 1.0 / N_OCR,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--d_feat", type=int, default=64)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--K_list", type=int, nargs="*", default=[1, 2, 4])
    ap.add_argument("--include_patch", action="store_true", default=True)
    ap.add_argument("--no_patch", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--probe_n", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hard_frac", type=float, default=0.35)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_seed", type=int, default=90_001)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/published/vlcot_probe_K_table.json")
    ap.add_argument("--conclusion", default="results/published/vlcot_probe_K_conclusion.md")
    args = ap.parse_args()
    if args.no_patch:
        args.include_patch = False

    device = torch.device(args.device)
    print(
        f"digit probe K-sweep | K={args.K_list} T={args.T} steps={args.steps} "
        f"device={device}",
        flush=True,
    )

    rows = []
    if args.include_patch:
        print("\n=== A patch baseline ===", flush=True)
        try:
            rows.append(train_eval("A", 1, args, device))
        except Exception as e:
            rows.append({"status": "error", "kind": "A_patch", "error": str(e)[:200]})

    for K in args.K_list:
        print(f"\n=== B_vlcot K={K} ===", flush=True)
        try:
            rows.append(train_eval("B", K, args, device))
        except Exception as e:
            rows.append({
                "status": "error", "kind": f"B_vlcot_K{K}",
                "latent_steps": K, "error": str(e)[:200],
            })
            if device.type == "cuda":
                torch.cuda.empty_cache()

    ok = [r for r in rows if r.get("status") == "ok"]
    payload = {
        "task": "digit_closed_set_probe_latent_K",
        "res": args.res,
        "T": args.T,
        "steps": args.steps,
        "batch": args.batch,
        "hard_frac": args.hard_frac,
        "seed": args.seed,
        "val_seed": args.val_seed,
        "chance": 1.0 / N_OCR,
        "table": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# Digit closed-set probe: latent steps K sweep (no LLM)",
        "",
        f"- Task: 1px stick digits 0–9; hard_frac={args.hard_frac}; res={args.res}",
        f"- Readout: mean-pool tokens → Linear({N_OCR}); **no free-gen**",
        f"- steps={args.steps} batch={args.batch} T={args.T} chance={1/N_OCR:.2f}",
        "",
        "| kind | K | acc_digit | final_loss | seconds | status |",
        "|------|---|-----------|------------|---------|--------|",
    ]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(
                f"| {r.get('kind')} | {r.get('latent_steps', '—')} | — | — | — | "
                f"{r.get('error')} |"
            )
        else:
            lines.append(
                f"| {r['kind']} | {r['latent_steps']} | {r['acc_digit']:.3f} | "
                f"{r['final_loss']:.3f} | {r['seconds']:.1f} | ok |"
            )
    lines.extend(["", "## Relative K", ""])
    b_rows = [r for r in ok if str(r["kind"]).startswith("B_vlcot")]
    if b_rows:
        by_k = {int(r["latent_steps"]): r for r in b_rows}
        if 1 in by_k:
            a1 = by_k[1]["acc_digit"]
            for k in sorted(by_k):
                acc = by_k[k]["acc_digit"]
                lines.append(f"- K={k}: acc={acc:.3f} (Δ vs K=1: {acc - a1:+.3f})")
        best = max(b_rows, key=lambda x: x["acc_digit"])
        lines.append(
            f"- Best B: **K={best['latent_steps']}** acc={best['acc_digit']:.3f}"
        )
    a = next((r for r in ok if r["kind"] == "A_patch"), None)
    if a and b_rows:
        best_b = max(b_rows, key=lambda x: x["acc_digit"])
        lines.append(
            f"- Best B vs A: {best_b['acc_digit']:.3f} vs {a['acc_digit']:.3f} "
            f"(Δ={best_b['acc_digit']-a['acc_digit']:+.3f})"
        )
    lines.append("")
    lines.append(
        "Phase-1 closed-set probe only; not frozen-LM free-gen OCR."
    )
    Path(args.conclusion).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"json → {args.out}", flush=True)


if __name__ == "__main__":
    main()
