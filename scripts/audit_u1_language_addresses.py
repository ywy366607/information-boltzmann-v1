#!/usr/bin/env python3
"""Read-only audit of the 64px language-address U1 candidate.

This intentionally evaluates the checkpoint that owns the new address prior,
rather than reusing the legacy 32px active-F2 rendering.  It records whether
language changes the actual SliceRead assignments and whether those assignments
become spatially useful enough to draw a sparse glyph.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.omni_model import DualStreamOmni
from fine_grain.pythia_bridge import capability_champion_kwargs
from scripts.train_pythia_capabilities import eval_static_t2i, make_cycle_eval_bank


def effective_rank(w: torch.Tensor) -> float:
    """Effective rank of an [N, M] assignment matrix."""
    values = torch.linalg.svdvals(w.float())
    energy = values.square().clamp_min(1e-12)
    p = energy / energy.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def address_stats(w: torch.Tensor, side: int) -> dict:
    """Quantify spatial locality; uniform global support has radius near one."""
    # w is [N, M] and each point distributes one unit across slots.
    n, m = w.shape
    grid = torch.linspace(-1.0, 1.0, side, device=w.device, dtype=w.dtype)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    xy = torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=-1)
    mass = w.sum(0).clamp_min(1e-8)
    centres = (w.transpose(0, 1) @ xy) / mass[:, None]
    distance = torch.cdist(xy[None], centres[None]).squeeze(0)
    radius = (w * distance.square()).sum(0).div(mass).sqrt()
    entropy = -(w.clamp_min(1e-8) * w.clamp_min(1e-8).log()).sum(-1).mean()
    return {
        "normalized_row_entropy": float(entropy / math.log(m)),
        "effective_rank": effective_rank(w),
        "mean_radius": float(radius.mean()),
        "uniform_radius": float(torch.cdist(xy[None], xy.mean(0, keepdim=True)[None]).squeeze(0).square().mean().sqrt()),
        "mean_peak_assignment": float(w.max(-1).values.mean()),
    }


def build(path: Path, device: torch.device) -> DualStreamOmni:
    model = DualStreamOmni(**capability_champion_kwargs(
        res=64, n_slices=64, language="pythia", lm_device=str(device),
        pixel_loss_mode="gaussian_nll", s0_acc_coef=0.0,
        deslice_write_sharpening=True, use_lang_address=True,
    )).to(device)
    if model.lm is not None:
        model.lm.to(device)
    raw = torch.load(path, map_location=device)
    state = raw.get("state_dict", raw)
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [name for name in unexpected if not name.startswith("lm.")]
    if bad:
        raise RuntimeError(f"unexpected non-LM checkpoint keys: {bad[:5]}")
    # The checkpoint intentionally excludes frozen Pythia weights.
    allowed_missing = [name for name in missing if name.startswith("lm.")]
    if len(allowed_missing) != len(missing):
        raise RuntimeError(f"missing non-LM checkpoint keys: {missing[:5]}")
    return model.eval()


@torch.no_grad()
def run_prompt(model: DualStreamOmni, prompt: str, device: torch.device):
    captured = []
    original = []
    for layer in model.mot_stack.layers:
        reader = layer.read
        original.append(reader.forward)
        def hooked(*args, _reader=reader, _forward=reader.forward, **kwargs):
            s, w = _forward(*args, **kwargs)
            captured.append((w.detach().cpu(), _reader.last_w_prior.detach().cpu()))
            return s, w
        reader.forward = hooked
    try:
        blank = torch.zeros(1, 3, 64, 64, device=device)
        out = model(blank, [prompt], image_precision=0.0)
    finally:
        for layer, forward in zip(model.mot_stack.layers, original):
            layer.read.forward = forward
    return out["rgb"][0].detach().cpu(), captured


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default=str(ROOT / "checkpoints" / "_unified_u1_candidate.pt"))
    parser.add_argument("--out", default=str(ROOT / "results" / "published" / "u1_language_address_audit.json"))
    parser.add_argument("--figure", default=str(ROOT / "present" / "figs" / "u1_language_address_audit.png"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = build(Path(args.candidate), device)
    prompts = [
        "Draw digit 7 with a thin red stroke blank image",
        "Draw digit 0 with a thin blue stroke blank image",
    ]
    outputs, assignments = zip(*(run_prompt(model, prompt, device) for prompt in prompts))
    # One captured pair per layer and prompt; assignment is the post-combined W.
    per_layer = []
    for layer_idx in range(len(model.mot_stack.layers)):
        w0, p0 = assignments[0][layer_idx]
        w1, p1 = assignments[1][layer_idx]
        per_layer.append({
            "layer": layer_idx,
            "actual": address_stats(w0[0], 64),
            "prior": address_stats(p0[0], 64),
            "actual_prompt_relative_difference": float(
                torch.norm(w0 - w1) / torch.norm(w0).clamp_min(1e-8)
            ),
            "prior_prompt_relative_difference": float(
                torch.norm(p0 - p1) / torch.norm(p0).clamp_min(1e-8)
            ),
        })
    static = eval_static_t2i(model, make_cycle_eval_bank(64, "text_to_both"), device, chunk=8)
    record = {
        "schema": "u1-language-address-reload-audit",
        "candidate": str(Path(args.candidate).resolve()),
        "prompts": prompts,
        "layers": per_layer,
        "static_t2i": static,
        "interpretation": (
            "Prompt sensitivity only establishes a control path. Sparse drawing requires "
            "localized, high-rank actual assignments and nontrivial digit/paired IoU."
        ),
    }
    Path(args.out).write_text(json.dumps(record, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(3, 4, figsize=(12, 8), dpi=160)
    fig.suptitle("U1 candidate reload: actual output and layer-3 Slice assignments", fontweight="bold")
    for col, (prompt, image, captures) in enumerate(zip(prompts, outputs, assignments)):
        axes[0, col * 2].imshow(image.permute(1, 2, 0).clamp(0, 1))
        axes[0, col * 2].set_title(prompt.replace(" blank image", ""), fontsize=8)
        axes[0, col * 2].axis("off")
        w, prior = captures[-1]
        axes[0, col * 2 + 1].imshow(w[0, :, 0].reshape(64, 64), cmap="magma")
        axes[0, col * 2 + 1].set_title("layer-3 actual W, slice 0", fontsize=8)
        axes[0, col * 2 + 1].axis("off")
        for row, layer_idx in enumerate((0, 3), start=1):
            w, prior = captures[layer_idx]
            axes[row, col * 2].imshow(prior[0, :, 0].reshape(64, 64), cmap="viridis")
            axes[row, col * 2].set_title(f"L{layer_idx} language prior, slice 0", fontsize=8)
            axes[row, col * 2].axis("off")
            axes[row, col * 2 + 1].imshow(w[0, :, 0].reshape(64, 64), cmap="magma")
            axes[row, col * 2 + 1].set_title(f"L{layer_idx} actual W, slice 0", fontsize=8)
            axes[row, col * 2 + 1].axis("off")
    fig.tight_layout()
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, bbox_inches="tight")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
