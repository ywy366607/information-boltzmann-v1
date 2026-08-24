#!/usr/bin/env python3
"""Causal probe for the NORTH_STAR native Slice generation contract.

Uses one fixed bank and one native X-Slice-H stack pass. It compares matched
prompts, cyclically shuffled prompts, and an explicit pi_x=0 field clamp, then
records point-write and Slice-assignment diagnostics. It does not train or
select a checkpoint.
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

from fine_grain.gen_metrics import (
    background_flood_rate,
    gen_free_scores,
    ink_centroid_error,
    paired_ink_iou,
)
from fine_grain.models import coords
from fine_grain.omni_model import DualStreamOmni
from fine_grain.omni_tasks import GRID_PLACES, one_sample


def to_signed(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def to_unit(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def pixel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).flatten(1).pow(2).mean(1).sqrt().mean())


def score_bank(rgb: torch.Tensor, samples: list[dict]) -> dict:
    rec = {
        "psnr": 0.0,
        "digit_iou": 0.0,
        "digit_top1": 0.0,
        "color_acc": 0.0,
        "ink_frac": 0.0,
        "flood": 0.0,
        "paired_iou": 0.0,
        "centroid_error": 0.0,
    }
    for i, sample in enumerate(samples):
        pred = rgb[i : i + 1]
        tgt = sample["target_rgb"]
        stroke = sample["stroke"].unsqueeze(1)
        mse = (pred - tgt).pow(2).mean().clamp_min(1e-8)
        rec["psnr"] += float(-10.0 * torch.log10(mse))
        free = gen_free_scores(pred, sample["digit"], sample["color"])
        for key in ("digit_iou", "digit_top1", "color_acc", "ink_frac"):
            rec[key] += float(free[key])
        rec["flood"] += background_flood_rate(pred, tgt, stroke)
        rec["paired_iou"] += paired_ink_iou(
            pred, stroke, sample["color"],
        )
        rec["centroid_error"] += ink_centroid_error(
            pred, stroke, sample["color"],
        )
    return {key: value / len(samples) for key, value in rec.items()}


def assignment_diagnostics(model: DualStreamOmni) -> list[dict]:
    out = []
    xy = coords(model.res, next(model.parameters()).device)[0]
    for i, layer in enumerate(model.mot_stack.layers):
        w = layer.last_w
        if w is None:
            continue
        wd = w.float()
        ent = -(wd.clamp_min(1e-8) * wd.clamp_min(1e-8).log()).sum(-1)
        ent_norm = ent / max(math.log(wd.shape[-1]), 1e-8)
        mass = wd.sum(dim=1).clamp_min(1e-8)
        centroid = torch.einsum("bnm,nc->bmc", wd, xy.float()) / mass.unsqueeze(-1)
        spread = centroid.std(dim=1).norm(dim=-1).mean()
        null = getattr(layer, "last_null", None)
        out.append({
            "layer": i,
            "point_assignment_entropy_norm": float(ent_norm.mean()),
            "slice_mass_cv": float(mass.std(dim=1).div(mass.mean(dim=1).clamp_min(1e-8)).mean()),
            "slice_centroid_spread": float(spread),
            "null_mass_mean": None if null is None else float(null.float().mean()),
        })
    return out


def f2_diagnostics(model: DualStreamOmni) -> list[dict]:
    """Per-layer F2 state; layer-specific coordinates are not compared as a descent curve."""
    out = []
    for i, layer in enumerate(model.mot_stack.layers):
        F = getattr(layer, "last_F", None)
        gap = getattr(layer, "last_gap", None)
        U = getattr(layer, "last_u", None)
        S = getattr(layer, "last_S", None)
        mu_p = getattr(layer, "last_mu_p", None)
        if F is None:
            continue
        out.append({
            "layer": i,
            "F_mean": float(F.float().mean()),
            "gap_mean": None if gap is None else float(gap.float().mean()),
            "U_mean": None if U is None else float(U.float().mean()),
            "prior_error_mean": (
                None if S is None or mu_p is None
                else float((S.float() - mu_p.detach().float()).pow(2).mean())
            ),
        })
    return out


def common_trajectory_diagnostics(
    model: DualStreamOmni,
    samples: list[dict],
) -> dict:
    """Score X_0..X_K with one frozen layer-0 F2 coordinate and RGB head."""
    trace = model.mot_stack.common_f2_anchor_trace(anchor_layer=0)
    energy = trace["energy"]
    delta = trace["delta"]
    terminal_delta = trace["terminal_delta"]
    targets = torch.cat([sample["target_rgb"] for sample in samples], 0)
    steps = []
    for k, X_k in enumerate(model.mot_stack._last_X_steps):
        rgb_k = model.decode_rgb(X_k).clamp(0.0, 1.0)
        mse_per_sample = (rgb_k - targets).flatten(1).pow(2).mean(1)
        steps.append({
            "step": k,
            "action_energy_mean": float(energy[:, k].mean()),
            "action_energy_median": float(energy[:, k].median()),
            "rgb_mse_mean": float(mse_per_sample.mean()),
            "rgb_mse_median": float(mse_per_sample.median()),
            "scores": score_bank(rgb_k, samples),
        })
    median_delta = delta.median(dim=0).values
    nonincrease_fraction = (delta <= 0.0).float().mean(dim=0)
    initial, final = steps[0], steps[-1]
    monotone = bool((median_delta <= 0.0).all())
    terminal_lower = bool(float(terminal_delta.median()) < 0.0)
    output_improves = bool(
        final["rgb_mse_mean"] <= initial["rgb_mse_mean"]
        and final["scores"]["digit_top1"] >= 0.95
        and final["scores"]["color_acc"] >= 0.95
        and final["scores"]["paired_iou"] >= 0.95
    )
    t0 = torch.zeros(
        targets.shape[0], device=targets.device, dtype=targets.dtype,
    )
    X_terminal = model.mot_stack._last_X_steps[-1]
    X_clean_stem = model.mot_stack.encode_X(targets, t=t0)
    terminal_rgb = model.decode_rgb(X_terminal).clamp(0.0, 1.0)
    X_terminal_reencoded = model.mot_stack.encode_X(terminal_rgb, t=t0)
    clean_stem_energy = model.mot_stack.common_f2_anchor_energy(X_clean_stem)
    reencoded_energy = model.mot_stack.common_f2_anchor_energy(
        X_terminal_reencoded,
    )

    def field_rms(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a - b).flatten(1).pow(2).mean(1).sqrt().mean())

    return {
        "definition": (
            "A_k = -log p_anchor(S_anchor(X_k)|H_0); anchor is the frozen "
            "layer-0 SliceRead and F2 language prior reused for every step."
        ),
        "anchor_layer": 0,
        "prompt_state": "fixed initial H_0",
        "writes_to_field": False,
        "steps": steps,
        "transition_median_delta": [float(x) for x in median_delta],
        "transition_nonincrease_fraction": [
            float(x) for x in nonincrease_fraction
        ],
        "terminal_delta_mean": float(terminal_delta.mean()),
        "terminal_delta_median": float(terminal_delta.median()),
        "terminal_lower_fraction": float((terminal_delta < 0.0).float().mean()),
        "coordinate_controls": {
            "clean_target_stem_energy_mean": float(clean_stem_energy.mean()),
            "clean_target_stem_energy_median": float(clean_stem_energy.median()),
            "terminal_rgb_reencoded_energy_mean": float(reencoded_energy.mean()),
            "terminal_rgb_reencoded_energy_median": float(reencoded_energy.median()),
            "terminal_vs_clean_stem_field_rms": field_rms(
                X_terminal, X_clean_stem,
            ),
            "terminal_vs_reencoded_field_rms": field_rms(
                X_terminal, X_terminal_reencoded,
            ),
            "decode_clean_target_stem_scores": score_bank(
                model.decode_rgb(X_clean_stem).clamp(0.0, 1.0), samples,
            ),
            "decode_terminal_reencoded_scores": score_bank(
                model.decode_rgb(X_terminal_reencoded).clamp(0.0, 1.0), samples,
            ),
            "note": (
                "These controls distinguish true prior mismatch from a latent "
                "gauge/decoder-null mismatch; none is fed back into generation."
            ),
        },
        "gate_verdicts": {
            "median_nonincrease_each_transition": monotone,
            "median_terminal_lower": terminal_lower,
            "output_improves": output_improves,
            "joint_common_action_descent": bool(
                monotone and terminal_lower and output_improves
            ),
        },
    }


@torch.no_grad()
def run(args) -> dict:
    device = torch.device(args.device)
    if args.recipe in ("core", "active_f2"):
        active_f2 = args.recipe == "active_f2"
        model = DualStreamOmni(
            d_model=args.d_model,
            n_slices=args.n_slices,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            res=args.res,
            surprise_mode="v1_bayes" if active_f2 else "baseline",
            s_update="raw",
            prior_loss_coef=0.1 if active_f2 else 0.0,
            sigreg_coef=0.0,
            use_stiefel=False,
            deslice_topk=0,
            deslice_write="increment",
            gate_h_local=False,
            vfe_coef=0.1 if active_f2 else 0.0,
            use_null_slice=False,
            use_residual_read=False,
            fm_pred="x",
            fm_signed=False,
            prior_write=1.0 if active_f2 else 0.0,
            prior_write_by_t=False,
            pixel_loss_mode="balanced_bce",
            spatial_prompt_vocab=args.place == "grid" or args.place in GRID_PLACES,
        ).to(device).eval()
    else:
        model = DualStreamOmni.unified(
            d_model=args.d_model,
            n_slices=args.n_slices,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            res=args.res,
            fm_pred="x",
            fm_signed=True,
            prior_write=0.0,
            spatial_prompt_vocab=args.place == "grid" or args.place in GRID_PLACES,
        ).to(device).eval()
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=True)

    placements = GRID_PLACES if args.place == "grid" else (args.place,)
    samples = []
    for repeat in range(args.repeats):
        for place_index, place in enumerate(placements):
            for digit in range(10):
                samples.append(one_sample(
                    np.random.default_rng(
                        1000 + 1000 * repeat + 10 * place_index + digit
                    ),
                    args.res,
                    "t2i",
                    t2i_canvas="black",
                    t2i_place=place,
                    t2i_digit=digit,
                    t2i_color=args.color,
                ))

    x0 = torch.cat([sample["image"] for sample in samples], 0).to(device)
    unit_chart = args.recipe in ("core", "active_f2")
    if not unit_chart:
        x0 = to_signed(x0)
    prompts = [sample["prompt"] for sample in samples]
    digit_shuffled = [
        one_sample(
            np.random.default_rng(0), args.res, "t2i",
            t2i_canvas="black", t2i_place=sample["placement"],
            t2i_digit=(int(sample["digit"]) + 1) % 10,
            t2i_color=args.color,
        )["prompt"]
        for sample in samples
    ]
    place_shuffled = None
    if args.place == "grid":
        place_shuffled = []
        for sample in samples:
            index = GRID_PLACES.index(sample["placement"])
            row, col = divmod(index, 3)
            wrong_place = GRID_PLACES[((row + 1) % 3) * 3 + (col + 1) % 3]
            place_shuffled.append(one_sample(
                np.random.default_rng(0), args.res, "t2i",
                t2i_canvas="black", t2i_place=wrong_place,
                t2i_digit=int(sample["digit"]), t2i_color=args.color,
            )["prompt"])
    need_pix = [True] * len(samples)
    t0 = torch.zeros(len(samples), device=device)

    device_samples = []
    for sample in samples:
        copy = dict(sample)
        copy["target_rgb"] = copy["target_rgb"].to(device)
        copy["stroke"] = copy["stroke"].to(device)
        device_samples.append(copy)

    model.mot_stack.set_record_field_trace(args.recipe == "active_f2")
    matched_out = model(x0, prompts, pi_x=1.0, need_pix=need_pix, t=t0)
    matched = matched_out["x_pred"].clamp(0.0, 1.0)
    if not unit_chart:
        matched = to_unit(matched_out["x_pred"])
    assignment = assignment_diagnostics(model)
    f2_state = f2_diagnostics(model)
    matched_field_rms = pixel_rms(matched_out["X"], model.mot_stack._last_X_stem)
    common_trajectory = (
        common_trajectory_diagnostics(model, device_samples)
        if args.recipe == "active_f2"
        else None
    )
    model.mot_stack.set_record_field_trace(False)

    shuffled_out = model(x0, digit_shuffled, pi_x=1.0, need_pix=need_pix, t=t0)
    shuffled_rgb = shuffled_out["x_pred"].clamp(0.0, 1.0)
    if not unit_chart:
        shuffled_rgb = to_unit(shuffled_out["x_pred"])
    place_shuffled_rgb = None
    if place_shuffled is not None:
        place_out = model(x0, place_shuffled, pi_x=1.0, need_pix=need_pix, t=t0)
        place_shuffled_rgb = place_out["x_pred"].clamp(0.0, 1.0)
        if not unit_chart:
            place_shuffled_rgb = to_unit(place_out["x_pred"])
    prior_off_rgb = None
    if args.recipe == "active_f2":
        saved_stack_pw = model.mot_stack.prior_write
        saved_layer_pw = [layer.prior_write for layer in model.mot_stack.layers]
        model.mot_stack.prior_write = 0.0
        for layer in model.mot_stack.layers:
            layer.prior_write = 0.0
        prior_off = model(x0, prompts, pi_x=1.0, need_pix=need_pix, t=t0)
        prior_off_rgb = prior_off["x_pred"].clamp(0.0, 1.0)
        model.mot_stack.prior_write = saved_stack_pw
        for layer, value in zip(model.mot_stack.layers, saved_layer_pw):
            layer.prior_write = value
    clamped_out = model(x0, prompts, pi_x=0.0, need_pix=need_pix, t=t0)
    clamped_rgb = clamped_out["x_pred"].clamp(0.0, 1.0)
    if not unit_chart:
        clamped_rgb = to_unit(clamped_out["x_pred"])

    matched_metrics = score_bank(matched, device_samples)
    shuffled_metrics = score_bank(shuffled_rgb, device_samples)
    place_shuffled_metrics = (
        None if place_shuffled_rgb is None
        else score_bank(place_shuffled_rgb, device_samples)
    )
    prior_off_metrics = (
        None if prior_off_rgb is None
        else score_bank(prior_off_rgb, device_samples)
    )
    clamped_metrics = score_bank(clamped_rgb, device_samples)

    return {
        "protocol": "docs/NORTH_STAR.md#9-竞争假设与关键实验",
        "checkpoint": str(Path(args.ckpt).resolve()),
        "recipe": args.recipe,
        "bank": {
            "n": len(samples),
            "digits": list(range(10)),
            "color": args.color,
            "placement": args.place,
            "placement_values": list(placements),
            "source": "coordinate black field",
            "seed_rule": "1000 + 1000*repeat + 10*place_index + digit",
        },
        "matched": matched_metrics,
        "shuffled_prompt": shuffled_metrics,
        "position_shuffled_prompt": place_shuffled_metrics,
        "prior_action_off": prior_off_metrics,
        "field_clamped_pi0": clamped_metrics,
        "causal_effects": {
            "matched_vs_shuffled_rgb_rms": pixel_rms(matched, shuffled_rgb),
            "matched_vs_position_shuffled_rgb_rms": (
                None if place_shuffled_rgb is None
                else pixel_rms(matched, place_shuffled_rgb)
            ),
            "matched_vs_prior_action_off_rgb_rms": (
                None if prior_off_rgb is None
                else pixel_rms(matched, prior_off_rgb)
            ),
            "matched_vs_clamped_rgb_rms": pixel_rms(matched, clamped_rgb),
            "matched_latent_field_write_rms": matched_field_rms,
        },
        "assignment": assignment,
        "f2_state": f2_state,
        "f2_state_note": (
            "Each non-shared layer has its own prior head/query coordinates; "
            "these F values establish F2 form but are not a cross-layer descent curve."
        ),
        "common_action_trajectory": common_trajectory,
        "decision_rules": {
            "prompt_causal": "matched digit_top1 must exceed shuffled by >=0.50",
            "write_causal": "matched digit_top1 must exceed pi_x=0 by >=0.50",
            "capacity": "matched digit_top1 >=0.95 before scaling",
            "position_causal": (
                "grid only: matched paired_iou must exceed position-shuffled by >=0.50"
            ),
            "grid_capacity": "matched paired_iou >=0.85 and centroid_error <=0.08",
            "prior_action_causal": (
                "active_f2 only: disabling Deslice(mu_p-S) must reduce digit_top1 "
                "or paired_iou by >=0.50"
            ),
            "common_action_descent": (
                "active_f2 only: one fixed anchor has median delta <=0 at every "
                "transition, median terminal delta <0, and RGB task quality improves"
            ),
        },
        "gate_verdicts": {
            "prompt_causal": bool(
                matched_metrics["digit_top1"]
                - shuffled_metrics["digit_top1"]
                >= 0.50
            ),
            "write_causal": bool(
                matched_metrics["digit_top1"]
                - clamped_metrics["digit_top1"]
                >= 0.50
            ),
            "capacity": bool(
                matched_metrics["digit_top1"] >= 0.95
                and matched_metrics["color_acc"] >= 0.95
                and matched_metrics["flood"] <= 0.10
                and (
                    args.place != "grid"
                    or (
                        matched_metrics["paired_iou"] >= 0.85
                        and matched_metrics["centroid_error"] <= 0.08
                    )
                )
            ),
            "position_causal": (
                None if place_shuffled_metrics is None
                else bool(
                    matched_metrics["paired_iou"]
                    - place_shuffled_metrics["paired_iou"]
                    >= 0.50
                )
            ),
            "prior_action_causal": (
                None if prior_off_metrics is None
                else bool(
                    matched_metrics["digit_top1"]
                    - prior_off_metrics["digit_top1"] >= 0.50
                    or matched_metrics["paired_iou"]
                    - prior_off_metrics["paired_iou"] >= 0.50
                )
            ),
            "common_action_descent": (
                None if common_trajectory is None
                else common_trajectory["gate_verdicts"][
                    "joint_common_action_descent"
                ]
            ),
        },
        "review": "not independently reviewed",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default="checkpoints/omni_d64_northstar_omni_active_f2_grid_best.pt",
    )
    ap.add_argument("--out", default="results/published/northstar_generation_probe.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--res", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-slices", type=int, default=16)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--color", default="green")
    ap.add_argument(
        "--place", choices=["center", "grid", *GRID_PLACES], default="center",
    )
    ap.add_argument(
        "--recipe", choices=["core", "active_f2", "vfe"], default="active_f2",
    )
    args = ap.parse_args()

    result = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
