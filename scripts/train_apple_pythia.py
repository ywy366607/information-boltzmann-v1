#!/usr/bin/env python3
"""Pythia language backbone + real apple photos. t2i learnability probe.

Language meets vision only in MoT. Pythia is frozen; H is its last hidden
state, projected by text_in. Same residual-read / S0 graph as digits.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.apple_data import load_apple_pairs, sample_batch
from fine_grain.flow_match import interpolate, sample_t
from fine_grain.omni_model import DualStreamOmni
from scripts.train_omni_probe import f_generate


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res, steps, batch, lr = 64, 400, 4, 2e-4
    store = load_apple_pairs(res=res, max_n=96)
    n = store["image"].shape[0]
    print(f"[apple-pythia] n={n} device={device}", flush=True)

    model = DualStreamOmni.unified(
        d_model=256, n_slices=64, n_heads=8, n_layers=4, res=res,
        fm_pred="x", fm_signed=True, language="pythia", lm_device="cpu",
    )
    print(f"  language={model.lm_note}", flush=True)
    print(f"  d_llm={model.d_llm} n_par_train={sum(p.numel() for p in trainable(model))}", flush=True)
    model.to(device)
    if model.lm is not None:
        model.lm.to(device)
    opt = torch.optim.AdamW(trainable(model), lr=lr, weight_decay=1e-4)

    rng = np.random.default_rng(0)
    t0 = time.time()
    best = float("inf")
    ckpt = ROOT / "checkpoints" / "omni_apple_pythia_best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, steps + 1):
        model.train()
        if model.lm is not None:
            model.lm.eval()
        idx = rng.integers(0, n, size=batch)
        b = sample_batch(store, idx, signed=True)
        x1 = b["target_rgb"].to(device)
        noise = torch.randn_like(x1)
        t = sample_t(batch, device, "uniform")
        xt = interpolate(noise, x1, t)
        b["target_rgb"] = x1.detach().cpu()
        b["t"] = t
        opt.zero_grad()
        out = model(xt, b["prompt"], need_pix=b["need_pix"], t=t)
        loss, meta = model.omni_loss(out, b, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable(model), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1 or step == steps:
            print(
                f"  step {step:4d}/{steps} loss={float(loss):.3f} "
                f"pix={meta.get('pix', 0):.3f} s0={meta.get('s0_acc', 0):.3f} "
                f"dt={time.time()-t0:.0f}s",
                flush=True,
            )
            if float(loss) < best:
                best = float(loss)
                cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()
                       if not k.startswith("lm.")}
                torch.save(cpu, ckpt)

    # generate
    model.eval()
    prompts = ["an apple", "a red apple", "a green apple", "a photo of an apple"]
    x0 = torch.randn(len(prompts), 3, res, res, device=device)
    with torch.no_grad():
        gen = f_generate(model, x0, prompts, [True] * len(prompts), n_steps=8, halt_eps=0.03)
    rgb = ((gen + 1) * 0.5).clamp(0, 1)
    real = ((store["image"][:4] + 1) * 0.5).clamp(0, 1) if store["image"].min() < 0 else store["image"][:4]
    # store is [0,1]
    real = store["image"][:4]

    fig, axes = plt.subplots(2, 4, figsize=(10, 5), facecolor="#0f172a")
    for i in range(4):
        axes[0, i].imshow(real[i].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[0, i].set_title("real apple", color="#e2e8f0", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(rgb[i].permute(1, 2, 0).detach().cpu().numpy())
        axes[1, i].set_title(prompts[i], color="#e2e8f0", fontsize=8)
        axes[1, i].axis("off")
        axes[0, i].set_facecolor("#0f172a")
        axes[1, i].set_facecolor("#0f172a")
    fig.suptitle("Pythia + apple photos  t2i", color="white")
    out = ROOT / "present" / "figs" / "apple_pythia_gallery.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=140)
    plt.close()
    print(f"saved {out}  best_loss={best:.3f} ckpt={ckpt}", flush=True)


if __name__ == "__main__":
    main()
