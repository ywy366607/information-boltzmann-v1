#!/usr/bin/env python3
"""Locate geometry/identity conflicts in the existing ten-digit energy loss.

This is an audit of one X--Slice--H graph, not a new loss or head.  It compares
the same-observation geometry gradient with the registered ten-by-ten paired
digit likelihood, separately on stroke pixels shared by multiple digit
templates and on digit-unique pixels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.capability_tasks import collate_capability
from fine_grain.omni_model import DualStreamOmni
from fine_grain.pythia_bridge import capability_champion_kwargs
from scripts.train_pythia_capabilities import (
    _forward,
    _observation_energy,
    make_static_bank,
    digit_residual_likelihood,
    paired_digit_likelihood,
    sample_digit_group,
)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    if a.numel() == 0 or b.numel() == 0:
        return None
    denom = a.norm() * b.norm()
    if float(denom) == 0.0:
        return None
    return float((a.flatten() @ b.flatten() / denom).detach())


def _region_stats(
    geometry: torch.Tensor,
    identity: torch.Tensor,
    region: torch.Tensor,
) -> dict:
    # Gradients are RGB [B,3,H,W], while a region is [B,H,W].
    mask = region.unsqueeze(1).expand_as(geometry)
    g = geometry.masked_select(mask)
    c = identity.masked_select(mask)
    opposing = (g * c < 0).to(torch.float32)
    return {
        "entries": int(g.numel()),
        "geometry_norm": float(g.norm()),
        "identity_norm": float(c.norm()),
        "cosine": _cosine(g, c),
        "opposing_coordinate_fraction": float(opposing.mean()) if opposing.numel() else None,
    }


def _parameter_groups(model: DualStreamOmni) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "f2_prior": [], "language_to_vision": [], "slice_write": [], "other": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".surprise_gate.prior_" in name or ".surprise_gate.slice_queries" in name:
            groups["f2_prior"].append((name, parameter))
        elif ".mot.Wk_t" in name or ".mot.Wv_t" in name:
            groups["language_to_vision"].append((name, parameter))
        elif ".deslice." in name or ".read." in name:
            groups["slice_write"].append((name, parameter))
        else:
            groups["other"].append((name, parameter))
    return groups


def _parameter_stats(
    geometry: torch.Tensor,
    identity: torch.Tensor,
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]],
) -> dict:
    params = [parameter for rows in groups.values() for _, parameter in rows]
    gg = torch.autograd.grad(geometry, params, retain_graph=True, allow_unused=True)
    gi = torch.autograd.grad(identity, params, retain_graph=True, allow_unused=True)
    by_param = {id(parameter): (a, b) for parameter, a, b in zip(params, gg, gi)}
    report = {}
    for label, rows in groups.items():
        a = [by_param[id(parameter)][0].flatten() for _, parameter in rows if by_param[id(parameter)][0] is not None]
        b = [by_param[id(parameter)][1].flatten() for _, parameter in rows if by_param[id(parameter)][1] is not None]
        if not a or not b:
            report[label] = {"n_tensors": len(rows), "cosine": None}
            continue
        av, bv = torch.cat(a), torch.cat(b)
        report[label] = {
            "n_tensors": len(rows),
            "geometry_norm": float(av.norm()),
            "identity_norm": float(bv.norm()),
            "cosine": _cosine(av, bv),
        }
    return report


def _response_rank(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    """Measure prompt-conditioned residual capacity without a classifier."""
    pred = prediction.flatten(1)
    gold = target.flatten(1)
    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    gold_centered = gold - gold.mean(dim=0, keepdim=True)

    def spectrum(value: torch.Tensor) -> dict:
        singular = torch.linalg.svdvals(value)
        energy = singular.square()
        p = energy / energy.sum().clamp_min(1e-12)
        return {
            "singular_values": [float(x) for x in singular.detach().cpu()],
            "effective_rank": float(1.0 / p.square().sum()),
            "rank_eps": int((singular > singular.max() * 1e-3).sum()),
        }

    denom = pred_centered.norm() * gold_centered.norm()
    return {
        "prediction": spectrum(pred_centered),
        "target": spectrum(gold_centered),
        "centered_rgb_cosine": float(
            (pred_centered.flatten() @ gold_centered.flatten()) / denom.clamp_min(1e-12)
        ),
        "centered_rgb_relative_mse": float(
            (pred_centered - gold_centered).square().mean() /
            gold_centered.square().mean().clamp_min(1e-12)
        ),
    }


def _batch_spectrum(value: torch.Tensor) -> dict:
    """Prompt-to-state rank after flattening all non-batch dimensions."""
    flat = value.detach().float().flatten(1)
    centered = flat - flat.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(1e-12)
    return {
        "effective_rank": float(1.0 / probability.square().sum()),
        "rank_eps": int((singular > singular.max() * 1e-3).sum()),
        "rms": float(flat.square().mean().sqrt()),
    }


def _masked_token_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] < value.shape[1]:
        # The exact North-Star graph may append horizon/action control tokens
        # after the lexical trace mask is captured.  They are deliberately
        # excluded here: this audit measures the ten digit prompts only.
        mask = torch.nn.functional.pad(mask, (0, value.shape[1] - mask.shape[1]))
    elif mask.shape[1] > value.shape[1]:
        mask = mask[:, : value.shape[1]]
    weight = mask.to(value.dtype).unsqueeze(-1)
    return (value * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def _condition_path_spectra(model: DualStreamOmni) -> dict:
    """Locate where ten prompt conditions lose independent directions."""
    stack = model.mot_stack
    mask = stack._last_trace_text_mask
    if mask is None:
        return {}
    report = {
        "frozen_pythia": _batch_spectrum(_masked_token_mean(stack._last_text_evidence, mask)),
        "text_in": _batch_spectrum(_masked_token_mean(stack._last_H_stem, mask)),
    }
    for index, state in enumerate(stack._last_step_H):
        report[f"joint_layer_{index}"] = _batch_spectrum(_masked_token_mean(state, mask))
    for index, field in enumerate(stack._last_X_steps):
        report[f"field_after_layer_{index - 1}"] = _batch_spectrum(field)
    for index, layer in enumerate(stack.layers):
        h_ctx = getattr(layer, "last_H_ctx", None)
        if h_ctx is not None:
            report[f"f2_context_{index}"] = _batch_spectrum(h_ctx)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "checkpoints" / "_static32_m16_proportional_t2i_candidate.pt"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "results" / "published" / "static32_m16_contrast_geometry_diagnosis.json"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lm-device", default=None)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--n-slices", type=int, default=16)
    parser.add_argument("--glyph-box", type=int, default=12)
    parser.add_argument("--glyph-stroke-px", type=int, default=2)
    parser.add_argument("--normalized-glyph-layout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-dilation", type=int, default=1)
    parser.add_argument("--contrast-coef", type=float, default=10.0)
    parser.add_argument(
        "--identity-mode", choices=("pair", "difference", "residual"), default="pair",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    raw = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(raw, dict) and isinstance(raw.get("config"), dict):
        # Target-resolution scratch checkpoints already carry the exact graph.
        # Loading that graph directly avoids accidentally diagnosing a Pythia
        # bridge or mesh variant that was never used during training.
        model = DualStreamOmni(**raw["config"]).to(device)
        model.load_state_dict(raw["state_dict"])
    else:
        model = DualStreamOmni(**capability_champion_kwargs(
            res=args.resolution, n_slices=args.n_slices,
            local_dilation=args.local_dilation,
            language="pythia", lm_device=args.lm_device or args.device,
        )).to(device)
        if model.lm is not None:
            model.lm.to(device)
        model.load_visual_champion(args.checkpoint, skip_language_interface=False)
    model.set_optimization_phase("language")
    model.train()

    bank = make_static_bank(
        args.resolution, "text_to_both", glyph_box=args.glyph_box,
        glyph_stroke_px=args.glyph_stroke_px,
        normalized_layout=args.normalized_glyph_layout,
    )
    samples = sample_digit_group(bank, np.random.default_rng(20260905))
    batch = collate_capability(samples)
    # The graph already exposes a detached trace for diagnostics; enabling it
    # does not alter the forward update or trainable parameters.
    model.mot_stack.set_record_field_trace(True)
    out = _forward(model, batch)
    geometry_per_sample = _observation_energy(out, batch, seg_weight=0.25)
    geometry = geometry_per_sample.mean()
    if args.identity_mode == "residual":
        identity, identity_meta = digit_residual_likelihood(out, batch, seg_weight=0.25)
    else:
        identity, identity_meta = paired_digit_likelihood(
            out, batch, temperature=0.1, seg_weight=0.25,
            difference_only=args.identity_mode == "difference",
        )

    rgb_geometry = torch.autograd.grad(geometry, out["rgb"], retain_graph=True)[0]
    rgb_identity = torch.autograd.grad(identity, out["rgb"], retain_graph=True)[0]
    strokes = batch["stroke"].to(device) > 0.5
    if strokes.dim() == 4:
        strokes = strokes[:, 0]
    multiplicity = strokes.sum(dim=0)
    shared = strokes & (multiplicity.unsqueeze(0) > 1)
    unique = strokes & (multiplicity.unsqueeze(0) == 1)
    foreground = strokes
    all_points = torch.ones_like(foreground, dtype=torch.bool)
    coefficient = float(args.contrast_coef)
    param_stats = _parameter_stats(geometry, identity, _parameter_groups(model))

    report = {
        "schema": "static32-m16-geometry-identity-gradient-audit-v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "graph": {"resolution": args.resolution, "n_slices": args.n_slices,
                  "glyph_box": args.glyph_box, "glyph_stroke_px": args.glyph_stroke_px,
                  "normalized_layout": bool(args.normalized_glyph_layout),
                  "local_dilation": int(args.local_dilation),
                  "language_backend": getattr(model, "language", None),
                  "language_backbone_frozen": getattr(model, "lm", None) is not None,
                  "new_head": False, "recurrence": False,
                  "identity_mode": args.identity_mode},
        "losses": {"geometry": float(geometry.detach()), "identity": float(identity.detach()),
                   "contrast_coef_screened": coefficient,
                   "scaled_identity_to_geometry_norm_ratio": float(
                       coefficient * rgb_identity.norm() / rgb_geometry.norm().clamp_min(1e-12)
                   )},
        "template_overlap": {
            "foreground_pixels": int(foreground.sum()),
            "shared_foreground_fraction": float(shared.sum() / foreground.sum().clamp_min(1)),
            "unique_foreground_fraction": float(unique.sum() / foreground.sum().clamp_min(1)),
        },
        "rgb_gradient_regions": {
            "shared_strokes": _region_stats(rgb_geometry, rgb_identity, shared),
            "unique_strokes": _region_stats(rgb_geometry, rgb_identity, unique),
            "all_foreground": _region_stats(rgb_geometry, rgb_identity, foreground),
            "full_canvas": _region_stats(rgb_geometry, rgb_identity, all_points),
        },
        "parameter_gradient_groups": param_stats,
        "prompt_conditioned_response": _response_rank(
            out["rgb"].detach(), batch["target_rgb"].to(device),
        ),
        "condition_path_spectra": _condition_path_spectra(model),
        "identity_meta": identity_meta,
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
