#!/usr/bin/env python3
"""Did the language-prior readout actually un-collapse?

Measures H_ctx / S_hat effective rank at init and after a short train,
plus whether CE+pred_loss sends a non-zero gradient into Q.

Usage:
  python scripts/probe_prior_readout.py --steps 80 --device cpu
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.vlm_data import make_vqa_batch
from scripts.run_v0_surprise_eval import DualStreamVQAModel, eval_model


def erank(tokens: torch.Tensor) -> float:
    t = tokens.detach().float()
    if t.ndim == 3:
        t = t.reshape(-1, t.shape[-1]) if t.shape[0] == 1 else t[0]
    s = torch.linalg.svdvals(t)
    p = (s * s).clamp_min(1e-12)
    p = p / p.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def pairwise_cos(tokens: torch.Tensor) -> float:
    t = tokens.detach().float()
    if t.ndim == 3:
        t = t[0]
    u = F.normalize(t, dim=-1)
    sim = u @ u.T
    m = sim.shape[0]
    eye = torch.eye(m, dtype=torch.bool, device=sim.device)
    return float(sim.masked_select(~eye).abs().mean())


@torch.no_grad()
def readout_stats(model: DualStreamVQAModel, imgs, prompts) -> dict:
    model.eval()
    _ = model(imgs, prompts)
    rows = []
    for i, layer in enumerate(model.mot_stack.layers):
        g = layer.surprise_gate
        if not hasattr(g, "slice_queries"):
            continue
        hctx = getattr(g, "last_H_ctx", None)
        row = {"layer": i, "Q_erank": erank(g.slice_queries[0])}
        if hctx is not None:
            row["H_ctx_erank"] = erank(hctx)
            row["H_ctx_cos"] = pairwise_cos(hctx)
        sh = None
        if getattr(layer, "last_u", None) is not None:
            pass
        # S_hat / mu_p live on last forward via surprise meta; re-read if stored
        # NativeMoTLayer does not keep S_hat; pull from gate buffer if present
        rows.append(row)
        # fill S_hat from a second cheap call on stored H_ctx
        if hctx is not None and hasattr(g, "lang_to_prior"):
            sh = g.lang_to_prior(hctx)
            row["S_hat_erank"] = erank(sh)
            row["S_hat_cos"] = pairwise_cos(sh)
        elif hctx is not None and hasattr(g, "prior_head"):
            mu_p, _ = g.prior_head(hctx).chunk(2, dim=-1)
            row["S_hat_erank"] = erank(mu_p)
            row["S_hat_cos"] = pairwise_cos(mu_p)
    return {"layers": rows}


def grad_q_norm(model: DualStreamVQAModel, imgs, prompts, targets) -> float:
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(imgs, prompts)
    loss = model.task_loss(out, targets)
    loss.backward()
    norms = []
    for layer in model.mot_stack.layers:
        g = layer.surprise_gate
        if hasattr(g, "slice_queries") and g.slice_queries.grad is not None:
            norms.append(float(g.slice_queries.grad.norm()))
    model.zero_grad(set_to_none=True)
    return float(np.mean(norms)) if norms else 0.0


def train_arm(mode: str, steps: int, device: torch.device, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rng_val = np.random.default_rng(seed + 99)
    model = DualStreamVQAModel(
        d_model=128, n_slices=32, n_layers=4, res=32,
        surprise_mode=mode, surprise_beta=1.5, prior_loss_coef=0.1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    b0 = make_vqa_batch(rng, batch=8, res=32, mix=["ocr", "kinks", "color"])
    imgs0 = b0["image"].to(device)
    tgt0 = torch.tensor([model.ans_to_idx[a] for a in b0["answer"]], device=device)
    init = readout_stats(model, imgs0, b0["prompt"])
    g0 = grad_q_norm(model, imgs0, b0["prompt"], tgt0)

    for step in range(1, steps + 1):
        model.train()
        b = make_vqa_batch(rng, batch=16, res=32, mix=["ocr", "kinks", "color"])
        imgs = b["image"].to(device)
        targets = torch.tensor([model.ans_to_idx[a] for a in b["answer"]], device=device)
        opt.zero_grad()
        out = model(imgs, b["prompt"])
        loss = model.task_loss(out, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step == steps or step % 20 == 0:
            print(
                f"  [{mode}] step {step:3d}/{steps}  loss={float(loss):.3f}  "
                f"pred={float(out['pred_loss']):.3f}",
                flush=True,
            )

    after = readout_stats(model, imgs0, b0["prompt"])
    g1 = grad_q_norm(model, imgs0, b0["prompt"], tgt0)
    val = eval_model(model, rng_val, val_batches=6, batch_size=16, res=32, device=device)
    return {
        "mode": mode,
        "steps": steps,
        "init": init,
        "after": after,
        "grad_Q_init": g0,
        "grad_Q_after": g1,
        "val_acc": val["acc"],
        "val_loss": val["loss"],
        "val_tasks": val["task_accs"],
    }


def summarize(rep: dict):
    print(f"\n== {rep['mode']}  {rep['steps']} steps  val_acc={rep['val_acc']*100:.1f}% ==")
    print(f"  ||grad Q||  init={rep['grad_Q_init']:.3e}  after={rep['grad_Q_after']:.3e}")
    print(f"  {'L':>2}  {'Hctx0':>6} {'Hctx1':>6}  {'Shat0':>6} {'Shat1':>6}  {'cos0':>5} {'cos1':>5}")
    for a, b in zip(rep["init"]["layers"], rep["after"]["layers"]):
        print(
            f"  {a['layer']:>2}  "
            f"{a.get('H_ctx_erank', float('nan')):6.2f} {b.get('H_ctx_erank', float('nan')):6.2f}  "
            f"{a.get('S_hat_erank', float('nan')):6.2f} {b.get('S_hat_erank', float('nan')):6.2f}  "
            f"{a.get('H_ctx_cos', float('nan')):5.2f} {b.get('H_ctx_cos', float('nan')):5.2f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/published/prior_readout_probe.json")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"device={device} steps={args.steps}", flush=True)

    reports = []
    for mode in ("v0_jepa", "v1_bayes"):
        print(f"\n--- training {mode} ---", flush=True)
        rep = train_arm(mode, args.steps, device)
        summarize(rep)
        reports.append(rep)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
