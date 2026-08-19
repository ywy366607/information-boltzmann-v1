#!/usr/bin/env python3
"""Point-F ckpt: σ/F maps + whether unit-energy ink actually stuck."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.ocr_1px import render_digit_mask
from fine_grain.omni_tasks import _paint, equal_energy_ink
from fine_grain.tasks import SIGNAL
from fine_grain.vlm_data import COLORS
from scripts.train_omni_probe import f_generate, to_unit
from scripts.viz_fdesc_uncertainty import chroma_maps, point_decode_maps, region_mean
from scripts.viz_fdesc_writes import load_model


def target_unit(d: int, color: str, res: int = 32):
    box = min(16, res - 2)
    y0 = x0 = (res - box) // 2
    pm = render_digit_mask(str(d), res, box, y0, x0, jitter=0.0)
    stroke = torch.from_numpy(pm.astype(np.float32)).view(1, res, res)
    paper = torch.zeros(1, 3, res, res)
    return _paint(paper, stroke, equal_energy_ink(color))[0], stroke[0]


def ink_l2(rgb: torch.Tensor) -> float:
    """Mean L2 of chromatic pixels in unit RGB."""
    c = rgb.max(0).values - rgb.min(0).values
    lum = rgb.max(0).values
    m = (c > 0.15) & (lum > 0.15)
    if int(m.sum()) < 3:
        return 0.0
    v = rgb[:, m].reshape(3, -1)
    return float(v.norm(dim=0).mean())


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_pf = ROOT / "checkpoints" / "omni_d256_unified_fdesc_pointf_best.pt"
    ckpt_4k = ROOT / "checkpoints" / "omni_d256_unified_fdesc_center_4k_best.pt"
    pf = load_model(ckpt_pf, device)
    old = load_model(ckpt_4k, device)
    w = pf.pix_logσ[-1].weight.detach().abs().mean().item()
    print(f"pix_logσ |W| mean={w:.4f}  (0 ⇒ σ unused)", flush=True)

    res, tmax, cfg = 32, 8, 2.0
    out_dir = ROOT / "present" / "figs"

    # --- digit grid ---
    fig, axes = plt.subplots(4, 10, figsize=(14.5, 6.2), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    for r, color in enumerate(COLORS):
        for c, d in enumerate(range(10)):
            torch.manual_seed(2000 + r * 10 + c)
            z = f_generate(
                pf, torch.randn(1, 3, res, res, device=device),
                [f"Draw digit {d} with a thin {color} stroke blank image"],
                [True], n_steps=tmax, cfg=cfg, halt_eps=0.0,
            )
            rgb = to_unit(z).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy()
            ax = axes[r, c]
            ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("#0f172a")
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(str(d), color="#e2e8f0", fontsize=9, pad=3)
            if c == 0:
                ax.set_ylabel(color, color="#fbbf24", fontsize=10)
    fig.suptitle("point F  ·  10×4  ·  trained on L2-unit ink", color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p_grid = out_dir / "omni_fdesc_pointf_digit_grid.png"
    fig.savefig(p_grid, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"saved {p_grid}", flush=True)

    # --- σ / F maps ---
    specs = [(7, "yellow"), (7, "green"), (8, "blue"), (9, "yellow"), (0, "red")]
    fig, axes = plt.subplots(len(specs), 6, figsize=(14.8, 2.15 * len(specs)), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    print(f"{'case':<12} {'F_str':>7} {'F_halo':>7} {'F_bg':>7} {'σ_str':>7} {'σ_halo':>7} {'σ_bg':>7}")
    for r, (d, color) in enumerate(specs):
        tgt, stroke = target_unit(d, color, res)
        torch.manual_seed(21 + r)
        x = torch.randn(1, 3, res, res, device=device)
        prompt = f"Draw digit {d} with a thin {color} stroke blank image"
        ones = torch.ones(1, device=device)
        with torch.no_grad():
            for _ in range(tmax):
                xc = pf(x, [prompt], need_pix=[True], t=ones)["x_pred"]
                xu = pf(x, [""], need_pix=[True], t=ones)["x_pred"]
                x = xu + float(cfg) * (xc - xu)
            pf(x, [prompt], need_pix=[True], t=ones)
        rgb = to_unit(x).clamp(0, 1)[0]
        err = (rgb.cpu() - tgt).abs().mean(0).numpy()
        _, _, halo = chroma_maps(rgb.cpu())
        st = stroke.cpu().numpy()
        bg = 1.0 - st
        pm = point_decode_maps(pf, tgt * 2.0 - 1.0, res)
        print(
            f"{color+' '+str(d):<12} "
            f"{region_mean(pm.get('F_n', st*0), st):7.3f} "
            f"{region_mean(pm.get('F_n', st*0), halo*bg):7.3f} "
            f"{region_mean(pm.get('F_n', st*0), bg*(halo<0.5)):7.3f} "
            f"{region_mean(pm.get('σ_rgb', st*0), st):7.3f} "
            f"{region_mean(pm.get('σ_rgb', st*0), halo*bg):7.3f} "
            f"{region_mean(pm.get('σ_rgb', st*0), bg*(halo<0.5)):7.3f}"
        )
        imgs = [
            ("pred", np.clip(rgb.cpu().permute(1, 2, 0).numpy(), 0, 1), None),
            ("|err|", err, "magma"),
            ("halo", halo, "cividis"),
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
    fig.suptitle("point F ckpt  ·  F and σ on pixels  ·  σ should light the halo if it learned", color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p_unc = out_dir / "omni_fdesc_pointf_uncertainty.png"
    fig.savefig(p_unc, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"saved {p_unc}", flush=True)

    # --- color energy: 4k vs pointf vs target ---
    print("\n=== chromatic-pixel L2  (unit RGB; target yellow should be 1.00 not 1.41) ===")
    print(f"{'color':<8} {'target':>7} {'4k':>7} {'pointF':>7} {'4k/tgt':>7} {'pf/tgt':>7}")
    for color in COLORS:
        tgt_e = float(np.linalg.norm(equal_energy_ink(color)))
        loud = float(np.linalg.norm(SIGNAL[{"red": 0, "green": 1, "blue": 2, "yellow": 3}[color]]))
        e4, ep = [], []
        for d in range(10):
            prompt = f"Draw digit {d} with a thin {color} stroke blank image"
            torch.manual_seed(3000 + d)
            x0 = torch.randn(1, 3, res, res, device=device)
            z4 = f_generate(old, x0.clone(), [prompt], [True], n_steps=tmax, cfg=cfg)
            torch.manual_seed(3000 + d)
            zp = f_generate(pf, x0.clone(), [prompt], [True], n_steps=tmax, cfg=cfg)
            e4.append(ink_l2(to_unit(z4).clamp(0, 1)[0].cpu()))
            ep.append(ink_l2(to_unit(zp).clamp(0, 1)[0].cpu()))
        a4, ap = float(np.mean(e4)), float(np.mean(ep))
        print(f"{color:<8} {tgt_e:7.3f} {a4:7.3f} {ap:7.3f} {a4/max(tgt_e,1e-6):7.2f} {ap/max(tgt_e,1e-6):7.2f}  (old SIGNAL L2={loud:.2f})")


if __name__ == "__main__":
    main()
