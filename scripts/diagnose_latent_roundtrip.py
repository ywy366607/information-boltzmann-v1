#!/usr/bin/env python3
"""Latent read-write round-trip fidelity of the Slice canvas.

N103 localized sequential-composition failure at the latent read: the
set-point write X += proj(mu_p - S(X)) accumulates across passes only if
S(X) faithfully reads back content previously written by Deslice. The B3
reconstruction gate does not cover this: it proves pixel-level decode of a
stem-encoded (perception-manifold) field, not slice-space read-back of
deslice-written content.

This diagnostic measures the round trip directly, with zero training:
  X_rt = X0 + W(read(X))       (write the read content onto a blank canvas)
  RT error = pixel distance between decode(X_rt) and decode(X)
measured per layer, on both the perception manifold (stem-encoded images)
and the generation manifold (fields produced by the champion's own prior
write), with and without prescribed write sharpening.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.omni_model import DualStreamOmni
from fine_grain.pythia_bridge import load_visual_champion
from fine_grain.capability_tasks import _scene
from fine_grain.vlm_data import COLORS, OCR_DIGITS
from fine_grain.omni_tasks import GRID_PLACES

DEFAULT_INIT = (
    ROOT / "checkpoints" /
    "omni_d64_northstar_omni_active_f2_grid_best.pt"
)


@torch.no_grad()
def decode(model, X: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(model.decode_field(X)).clamp(0.0, 1.0)


@torch.no_grad()
def roundtrip_error(model, X: torch.Tensor, layer_index: int) -> dict:
    """Write read(X) onto a blank canvas; measure content survival."""
    layer = model.mot_stack.layers[layer_index]
    target = decode(model, X)
    blank = torch.zeros_like(X)
    S, w = layer.read(X)
    X_rt = blank + layer.deslice.write_delta(S, w)
    got = decode(model, X_rt)
    mse = (got - target).pow(2).mean(dim=(1, 2, 3))
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    # Energy accounting: how much field variance does W(read()) recover?
    var_total = float(X.var())
    var_rt = float(X_rt.var())
    return {
        "rgb_mse": float(mse.mean()),
        "rgb_psnr": float(psnr.mean()),
        "field_variance_recovered": var_rt / max(var_total, 1e-8),
    }


@torch.no_grad()
def perception_fields(model, samples, device):
    """Encode the digit images through the stem (perception manifold)."""
    imgs = torch.cat([
        _scene(s["digit"], s["color"], s["place"], model.res)[0]
        for s in samples
    ], dim=0).to(device)
    pi = torch.ones(imgs.shape[0], device=device)
    return model.mot_stack.encode_X(imgs, image_precision=pi)


@torch.no_grad()
def generation_fields(model, prompts, device, res):
    """Fields produced by the champion's own prior write (generation manifold)."""
    zeros = torch.zeros(len(prompts), 3, res, res, device=device)
    out = model(
        zeros, prompts, pi_x=1.0, t=torch.zeros(len(prompts), device=device),
        image_precision=torch.zeros(len(prompts), device=device),
        text_precision=torch.ones(len(prompts), device=device),
    )
    return out["belief_mu"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default=str(DEFAULT_INIT))
    parser.add_argument("--out", default=str(
        ROOT / "results" / "published" / "latent_roundtrip_diagnostic.json",
    ))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=90)
    parser.add_argument(
        "--write-gammas", default="1,4,8",
        help="Prescribed write-sharpening doses to sweep.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    base_kwargs = dict(
        d_model=64, n_slices=16, n_layers=4, n_heads=4, res=16,
        surprise_mode="v1_bayes", s_update="raw",
        prior_loss_coef=0.1, sigreg_coef=0.0,
        use_stiefel=False, deslice_topk=0,
        use_null_slice=False, use_residual_read=False,
        gate_on="u", deslice_write="increment", gate_h_local=False,
        vfe_coef=0.1, prior_write=1.0, prior_write_by_t=False,
        pixel_loss_mode="balanced_bce",
        spatial_prompt_vocab=True, capability_vocab=True,
        use_modal_precision=True, use_target_time=True,
        use_target_time_adaln=False, use_horizon_tokens=False,
        gate_action_by_horizon=True, history_size=2, action_dim=2,
        use_action_adaln=False, use_action_tokens=False,
        use_action_rel_bias=False, use_action_transport=False,
        use_action_slice_transition=False, use_active_gdn2=False,
        use_goal_adaln=False, seg_classes=2, seg_loss_coef=1.0,
        s0_acc_coef=0.0,
    )

    bank = []
    for place in GRID_PLACES:
        for digit in OCR_DIGITS:
            bank.append({
                "digit": str(digit), "color": "red", "place": str(place),
                "prompt": (
                    f"Draw digit {digit} with a thin red stroke "
                    f"at {place.replace('_', ' ')}"
                ),
            })
    bank = bank[: int(args.limit)]
    prompts = [s["prompt"] for s in bank]

    gammas = [float(v) for v in str(args.write_gammas).split(",")]
    results = {}
    for gamma in gammas:
        model = DualStreamOmni(
            **base_kwargs,
            deslice_write_sharpening=gamma != 1.0,
        ).to(device)
        load_visual_champion(model, Path(args.init), skip_language_interface=False)
        if gamma != 1.0:
            for layer in model.mot_stack.layers:
                layer.deslice.write_gamma_raw.data.fill_(float(math.log(gamma)))
                layer.deslice.write_gamma_raw.requires_grad_(False)
        model.eval()

        X_perc = perception_fields(model, bank, device)
        X_gen = generation_fields(model, prompts, device, int(model.res))
        # Anchor: decode the field directly (no read-write cycle). The B3
        # reconstruction gate lives here; composition needs the round trip.
        imgs = torch.cat([
            _scene(s["digit"], s["color"], s["place"], model.res)[0]
            for s in bank
        ], dim=0).to(device)
        anchor_mse = (decode(model, X_perc) - imgs).pow(2).mean()
        cell = {
            "anchor_decode_psnr": float(
                -10.0 * torch.log10(anchor_mse.clamp_min(1e-12))
            ),
            "perception": {},
            "generation": {},
        }
        for layer_index in range(len(model.mot_stack.layers)):
            cell["perception"][f"layer{layer_index}"] = roundtrip_error(
                model, X_perc, layer_index,
            )
            cell["generation"][f"layer{layer_index}"] = roundtrip_error(
                model, X_gen, layer_index,
            )
        results[f"gamma_{gamma:g}"] = cell
        print(f"gamma={gamma:g} done", flush=True)

    record = {
        "schema": "latent-roundtrip-fidelity",
        "init": str(args.init),
        "note": (
            "Zero-training operator diagnostic: X_rt = blank + W(read(X)); "
            "error is decode distance to decode(X). Composition across "
            "sequential passes requires this to be faithful."
        ),
        "n_samples": len(bank),
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    for gamma in gammas:
        cell = results[f"gamma_{gamma:g}"]
        for manifold in ("perception", "generation"):
            line = " ".join(
                f"L{i}={cell[manifold][f'layer{i}']['rgb_psnr']:.1f}dB"
                for i in range(4)
            )
            print(f"gamma={gamma:g} {manifold:10s} roundtrip {line}", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
