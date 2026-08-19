#!/usr/bin/env python3
"""Halo of weak ink vs Bayes VFE maps (U, gap, σ_q) on the 4k F-descent ckpt."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.bayesian_surprise import compute_point_vfe
from fine_grain.ocr_1px import render_digit_mask
from fine_grain.omni_model import DualStreamOmni
from fine_grain.omni_tasks import _paint
from fine_grain.tasks import SIGNAL
from scripts.train_omni_probe import to_unit
from scripts.viz_fdesc_writes import load_model


def target_digit(d: int, color: str, res: int = 32):
    box = min(16, res - 2)
    y0 = x0 = (res - box) // 2
    pm = render_digit_mask(str(d), res, box, y0, x0, jitter=0.0)
    stroke = torch.from_numpy(pm.astype(np.float32)).view(1, res, res)
    paper = torch.zeros(1, 3, res, res)
    ink = SIGNAL[{"red": 0, "green": 1, "blue": 2, "yellow": 3}[color]]
    return _paint(paper, stroke, ink)[0], stroke[0]


def point_map(field: torch.Tensor | None, res: int) -> np.ndarray | None:
    """Desliced per-point field [B,N] → [H,W]."""
    if field is None:
        return None
    return field[0].detach().float().view(res, res).cpu().numpy()


def chroma_maps(rgb: torch.Tensor):
    """rgb [3,H,W] unit. chroma, luma, weak-halo (tinted but not full ink)."""
    c = (rgb.max(0).values - rgb.min(0).values)
    lum = rgb.max(0).values
    halo = ((c > 0.08) & (c < 0.45) & (lum > 0.08)).float()
    return c.cpu().numpy(), lum.cpu().numpy(), halo.cpu().numpy()


def layer_maps(layer, res: int) -> dict:
    return {
        "U": point_map(getattr(layer, "last_U_x", None), res),
        "gap": point_map(getattr(layer, "last_gap_x", None), res),
        "Uσ": point_map(getattr(layer, "last_usig_x", None), res),
        "σq": point_map(getattr(layer, "last_sigq_x", None), res),
    }


def point_decode_maps(model, tgt_t: torch.Tensor, res: int) -> dict:
    """True pixel Bayes: p(rgb|X)=N(μ,σ²) vs language prior field."""
    X = model.mot_stack._last_X
    mu_q, lv_q = model.decode_gauss(X)
    sig = torch.exp(0.5 * lv_q).mean(-1)[0].view(res, res).detach().cpu().numpy()
    out = {"σ_rgb": sig}
    Xp = getattr(model.mot_stack, "_last_X_prior", None)
    if Xp is None:
        return out
    mu_p, lv_p = model.decode_gauss(Xp)
    y = tgt_t.reshape(3, -1).transpose(0, 1).unsqueeze(0).to(mu_q.device)
    vfe = compute_point_vfe(mu_p[:1], lv_p[:1], mu_q[:1], lv_q[:1], y)
    out["F_n"] = vfe["F"][0, :, 0].view(res, res).detach().cpu().numpy()
    out["gap_n"] = vfe["gap"][0, :, 0].view(res, res).detach().cpu().numpy()
    return out


def region_mean(arr, mask) -> float:
    m = mask > 0.5
    if m.sum() < 1:
        return 0.0
    return float(arr[m].mean())


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = ROOT / "checkpoints" / "omni_d256_unified_fdesc_center_4k_best.pt"
    model = load_model(ckpt, device)
    res, cfg, tmax = 32, 2.0, 8
    specs = [
        (7, "yellow"), (7, "green"), (8, "blue"),
        (9, "yellow"), (0, "red"),
    ]
    n = len(specs)
    cols = ["pred", "|err|", "weak chroma", "F_n", "gap_n", "σ(rgb|X)"]
    fig, axes = plt.subplots(n, len(cols), figsize=(15.6, 2.15 * n), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    stats = []

    for r, (d, color) in enumerate(specs):
        tgt_t, stroke = target_digit(d, color, res)
        tgt = tgt_t.permute(1, 2, 0).numpy()
        torch.manual_seed(11 + r)
        x = torch.randn(1, 3, res, res, device=device)
        prompt = f"Draw digit {d} with a thin {color} stroke blank image"
        ones = torch.ones(1, device=device)
        with torch.no_grad():
            for _ in range(tmax):
                xc = model(x, [prompt], need_pix=[True], t=ones)["x_pred"]
                xu = model(x, [""], need_pix=[True], t=ones)["x_pred"]
                x = xu + float(cfg) * (xc - xu)
            # Stamp Bayes on the finished canvas with the real prompt (not CFG null).
            model(x, [prompt], need_pix=[True], t=ones)
        rgb = to_unit(x).clamp(0, 1)[0]
        err = (rgb.cpu() - tgt_t).abs().mean(0).numpy()
        c, lum, halo = chroma_maps(rgb.cpu())
        st = stroke.cpu().numpy()
        bg = 1.0 - st
        pm = point_decode_maps(model, tgt_t * 2.0 - 1.0, res)
        rec = {
            "name": f"{color} {d}",
            "F_stroke": region_mean(pm.get("F_n", np.zeros_like(st)), st),
            "F_halo": region_mean(pm.get("F_n", np.zeros_like(st)), halo * bg),
            "F_bg": region_mean(pm.get("F_n", np.zeros_like(st)), bg * (halo < 0.5)),
            "sig_stroke": region_mean(pm.get("σ_rgb", np.zeros_like(st)), st),
            "sig_halo": region_mean(pm.get("σ_rgb", np.zeros_like(st)), halo * bg),
            "halo_frac": float((halo * bg).mean()),
        }
        stats.append(rec)
        imgs = [
            ("pred", np.clip(rgb.cpu().permute(1, 2, 0).numpy(), 0, 1), None),
            ("|err|", err, "magma"),
            ("weak chroma", halo, "cividis"),
            ("F_n", pm.get("F_n"), "inferno"),
            ("gap_n", pm.get("gap_n"), "inferno"),
            ("σ(rgb|X)", pm.get("σ_rgb"), "inferno"),
        ]
        for c_i, (title, im, cmap) in enumerate(imgs):
            ax = axes[r, c_i]
            if cmap is None:
                ax.imshow(im, interpolation="nearest")
            else:
                ax.imshow(im, cmap=cmap, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("#0f172a")
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(title, color="#e2e8f0", fontsize=10, pad=3)
            if c_i == 0:
                ax.set_ylabel(f"{color} {d}", color="#fbbf24", fontsize=10)

    fig.suptitle(
        "Pixel Bayes  ·  F_n / gap_n / σ(rgb|X)  on the canvas  ·  "
        "slices were only the workspace",
        color="white", fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = ROOT / "present" / "figs" / "omni_fdesc_4k_uncertainty.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"saved {path}", flush=True)
    print(f"{'case':<12} {'F_stroke':>8} {'F_halo':>8} {'F_bg':>8} {'σ_h':>8} {'halo%':>7}")
    for s in stats:
        print(
            f"{s['name']:<12} {s['F_stroke']:8.3f} {s['F_halo']:8.3f} {s['F_bg']:8.3f} "
            f"{s['sig_halo']:8.3f} {s['halo_frac']*100:6.1f}%"
        )


if __name__ == "__main__":
    main()
