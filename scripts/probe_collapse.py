#!/usr/bin/env python3
"""Where does the net fail to tell 1px from black, 8 from 9, or everything from one vector?"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.ocr_1px import render_digit_mask
from fine_grain.omni_tasks import _paint, equal_energy_ink
from scripts.train_omni_probe import to_signed
from scripts.viz_fdesc_writes import load_model


def canvas(d: int | None, color: str, res: int = 32, flood: bool = False):
    paper = torch.zeros(1, 3, res, res)
    if flood:
        ink = torch.tensor(equal_energy_ink(color)).view(1, 3, 1, 1)
        return ink.expand_as(paper).contiguous(), torch.ones(1, res, res)
    if d is None:
        return paper, torch.zeros(1, res, res)
    box = min(16, res - 2)
    y0 = x0 = (res - box) // 2
    pm = render_digit_mask(str(d), res, box, y0, x0, jitter=0.0)
    stroke = torch.from_numpy(pm.astype(np.float32)).view(1, res, res)
    return _paint(paper, stroke, equal_energy_ink(color)), stroke


def cos(a, b):
    a = F.normalize(a.flatten(1), dim=-1)
    b = F.normalize(b.flatten(1), dim=-1)
    return float((a * b).sum(-1).mean())


def mean_pool(x, mask=None):
    if mask is None:
        return x.mean(dim=1)
    m = mask.reshape(x.shape[0], -1, 1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp_min(1.0)


@torch.no_grad()
def pack(model, img_u, prompt, device):
    img = to_signed(img_u).to(device)
    t1 = torch.ones(1, device=device)
    X = model.mot_stack.encode_X(img, t=t1)
    out = model(img, [prompt], need_pix=[True], t=t1)
    layer = model.mot_stack.layers[-1]
    S, w = layer.read(X)
    mu = getattr(layer, "last_mu_p", None)
    return {
        "X": X,
        "S": S,
        "w": w,
        "mu_p": None if mu is None else mu.detach(),
        "x_pred": out["x_pred"],
    }


def focus(w, stroke):
    """Mass of each slice on the stroke; max slice's stroke-share."""
    st = stroke.reshape(1, -1).to(w.device)
    mass = w.sum(1).clamp_min(1e-8)  # [B,M]
    on = (w * st.unsqueeze(-1)).sum(1) / mass
    return float(on.max()), float(on.mean())


def report(name, model, device):
    print(f"\n======== {name} ========")
    items = {
        "black": canvas(None, "green"),
        "g7": canvas(7, "green"),
        "g8": canvas(8, "green"),
        "g9": canvas(9, "green"),
        "g1": canvas(1, "green"),
        "y7": canvas(7, "yellow"),
        "flood": canvas(7, "green", flood=True),
    }
    prompts = {
        "black": "Draw digit 7 with a thin green stroke blank image",
        "g7": "Draw digit 7 with a thin green stroke blank image",
        "g8": "Draw digit 8 with a thin green stroke blank image",
        "g9": "Draw digit 9 with a thin green stroke blank image",
        "g1": "Draw digit 1 with a thin green stroke blank image",
        "y7": "Draw digit 7 with a thin yellow stroke blank image",
        "flood": "Draw digit 7 with a thin green stroke blank image",
    }
    rec = {}
    for k, (img, st) in items.items():
        rec[k] = pack(model, img, prompts[k], device)
        rec[k]["stroke"] = st

    # stem: 1px vs black
    Xb, X7 = rec["black"]["X"], rec["g7"]["X"]
    st = rec["g7"]["stroke"].to(device)
    print(f"  stem ||X_g7 - X_black|| mean         {float((X7-Xb).norm(dim=-1).mean()):.4f}")
    print(f"  stem ||X_g7 - X_black|| ON stroke    {float(((X7-Xb).norm(dim=-1)*st.reshape(1,-1)).sum()/st.sum().clamp_min(1)):.4f}")
    print(f"  stem ||X_g7 - X_black|| OFF stroke   {float(((X7-Xb).norm(dim=-1)*(1-st.reshape(1,-1))).sum()/(1-st).sum().clamp_min(1)):.4f}")

    mx, meanf = focus(rec["g7"]["w"], st)
    print(f"  SliceRead: max slice's mass on 7     {mx:.3f}   mean over slices {meanf:.3f}")
    # chance: stroke frac ~ 0.04
    print(f"  stroke pixel fraction                {float(st.mean()):.4f}")

    print("  cosine S (last layer, flattened):")
    pairs = [("black", "g7"), ("g7", "g8"), ("g7", "g9"), ("g8", "g9"), ("g7", "g1"), ("g7", "y7"), ("g7", "flood"), ("black", "flood")]
    for a, b in pairs:
        print(f"    S[{a:5s}] vs S[{b:5s}]  {cos(rec[a]['S'], rec[b]['S']):+.3f}")

    print("  cosine μ_p (language prior, last layer):")
    for a, b in [("g7", "g8"), ("g7", "g9"), ("g7", "y7"), ("g7", "g1")]:
        pa, pb = rec[a]["mu_p"], rec[b]["mu_p"]
        if pa is None:
            print("    (no mu_p)")
            break
        print(f"    μp[{a}] vs μp[{b}]  {cos(pa, pb):+.3f}")

    # within-image: S diversity
    S = rec["g7"]["S"][0]
    cmat = F.normalize(S, dim=-1) @ F.normalize(S, dim=-1).T
    off = cmat - torch.eye(cmat.shape[0], device=cmat.device)
    print(f"  S_g7 pairwise cosine mean (off-diag) {float(off.sum()/(off.numel()-cmat.shape[0])):+.3f}")
    S0 = rec["black"]["S"][0]
    c0 = F.normalize(S0, dim=-1) @ F.normalize(S0, dim=-1).T
    off0 = c0 - torch.eye(c0.shape[0], device=c0.device)
    print(f"  S_black pairwise cosine mean         {float(off0.sum()/(off0.numel()-c0.shape[0])):+.3f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = [
        ("g7_sy (only green 7)", ROOT / "checkpoints" / "omni_d256_unified_g7_sy_best.pt"),
        ("4k mixed F-desc", ROOT / "checkpoints" / "omni_d256_unified_fdesc_center_4k_best.pt"),
    ]
    for name, path in ckpts:
        if not path.exists():
            print("missing", path)
            continue
        m = load_model(path, device)
        report(name, m, device)


if __name__ == "__main__":
    main()
