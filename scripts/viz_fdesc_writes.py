#!/usr/bin/env python3
"""Show what F-descent actually paints: digit grid + per-write unroll."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.ocr_1px import render_digit_mask
from fine_grain.omni_model import DualStreamOmni
from fine_grain.omni_tasks import _paint
from fine_grain.tasks import SIGNAL
from fine_grain.vlm_data import COLORS
from scripts.train_omni_probe import f_generate, to_unit


def target_digit(d: int, color: str, res: int = 32):
    box = min(16, res - 2)
    y0 = x0 = (res - box) // 2
    pm = render_digit_mask(str(d), res, box, y0, x0, jitter=0.0)
    stroke = torch.from_numpy(pm.astype(np.float32)).view(1, res, res)
    paper = torch.zeros(1, 3, res, res)
    ink = SIGNAL[{"red": 0, "green": 1, "blue": 2, "yellow": 3}[color]]
    return _paint(paper, stroke, ink)[0].permute(1, 2, 0).numpy()


def load_model(ckpt: Path, device: torch.device) -> DualStreamOmni:
    model = DualStreamOmni(
        d_model=256, n_slices=64, n_layers=4, res=32, n_heads=8,
        surprise_mode="v1_bayes", s_update="rms_dir",
        prior_loss_coef=0.1, use_stiefel=True, deslice_topk=2,
        gate_on="u", deslice_write="increment", vfe_coef=0.1,
        s_lang_topk=0, fm_pred="x", fm_signed=True,
        prior_write=1.0, prior_write_by_t=False,
    ).to(device)
    raw = torch.load(ckpt, map_location="cpu")
    missing = model.load_state_dict(raw, strict=False)
    print(f"loaded {ckpt.name} missing={len(missing.missing_keys)}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def f_trace(model, x0, prompt, n_steps=8, cfg=2.0, halt_eps=0.0):
    """Return [K+1, 3, H, W] unit-rgb: noise then each write."""
    frames = [to_unit(x0).clamp(0, 1).cpu()]
    x = x0
    B = x.shape[0]
    ones = torch.ones(B, device=x.device, dtype=x.dtype)
    null = [""]
    for _ in range(n_steps):
        xc = model(x, [prompt], need_pix=[True], t=ones)["x_pred"]
        if abs(float(cfg) - 1.0) > 1e-6:
            xu = model(x, null, need_pix=[True], t=ones)["x_pred"]
            xc = xu + float(cfg) * (xc - xu)
        x = xc
        frames.append(to_unit(x).clamp(0, 1).cpu())
    return torch.cat(frames, 0)


def show_im(ax, im, title=None, title_color="#e2e8f0"):
    ax.imshow(np.clip(im, 0, 1), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#0f172a")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, color=title_color, fontsize=9, pad=3)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = ROOT / "checkpoints" / "omni_d256_unified_fdesc_center_4k_best.pt"
    model = load_model(ckpt, device)
    out_dir = ROOT / "present" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    res, cfg, tmax = 32, 2.0, 8

    # --- 10 digits x 4 colors ---
    fig, axes = plt.subplots(4, 10, figsize=(14.5, 6.2), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    for r, color in enumerate(COLORS):
        for c, d in enumerate(range(10)):
            prompt = f"Draw digit {d} with a thin {color} stroke blank image"
            torch.manual_seed(1000 + r * 10 + c)
            x0 = torch.randn(1, 3, res, res, device=device)
            z = f_generate(
                model, x0, [prompt], [True], n_steps=tmax, cfg=cfg, halt_eps=0.0,
            )
            rgb = to_unit(z).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy()
            title = f"{d}" if r == 0 else None
            show_im(axes[r, c], rgb, title)
            if c == 0:
                axes[r, c].set_ylabel(color, color="#fbbf24", fontsize=10)
    fig.suptitle(
        "F-descent 4k best  ·  10 digits × 4 colors  ·  K=8 from noise",
        color="white", fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    grid_path = out_dir / "omni_fdesc_4k_digit_grid.png"
    fig.savefig(grid_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"saved {grid_path}", flush=True)

    # --- unroll writes for 6 prompts ---
    specs = [
        (6, "yellow"), (8, "blue"), (7, "green"),
        (3, "yellow"), (0, "red"), (4, "red"),
    ]
    fig, axes = plt.subplots(len(specs), tmax + 2, figsize=(16.2, 8.6), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    for r, (d, color) in enumerate(specs):
        prompt = f"Draw digit {d} with a thin {color} stroke blank image"
        tgt = target_digit(d, color, res)
        torch.manual_seed(7 + r)
        x0 = torch.randn(1, 3, res, res, device=device)
        frames = f_trace(model, x0, prompt, n_steps=tmax, cfg=cfg)
        for k in range(tmax + 1):
            im = frames[k].permute(1, 2, 0).numpy()
            title = "noise" if (r == 0 and k == 0) else (f"F{k}" if r == 0 and k else None)
            show_im(axes[r, k], im, title)
        show_im(axes[r, -1], tgt, "target" if r == 0 else None, title_color="#94a3b8")
        axes[r, 0].set_ylabel(f"{color} {d}", color="#fbbf24", fontsize=10)
    fig.suptitle(
        "Each column is one F write  ·  canvas after k steps  ·  last col = target",
        color="white", fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    unroll_path = out_dir / "omni_fdesc_4k_unroll.png"
    fig.savefig(unroll_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"saved {unroll_path}", flush=True)


if __name__ == "__main__":
    main()
