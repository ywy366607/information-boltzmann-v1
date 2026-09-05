#!/usr/bin/env python3
"""Bind frozen Pythia to the proven static North-Star capability graph.

This B2 bridge starts from the one-checkpoint toy capability champion, not the
generation-only champion. A generation-first warmup may adapt the shared RGB
likelihood head, but stem/SliceRead/Deslice/segmentation stay fixed and the RGB
head is frozen again for refinement. The first admission gate is joint T2I
plus current-frame reconstruction/segmentation. Editing remains blocked until
that identity-action prerequisite passes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.capability_tasks import capability_sample, collate_capability
from fine_grain.gen_metrics import (
    background_flood_rate,
    gen_free_scores,
    ink_centroid_error,
    paired_ink_iou,
)
from fine_grain.omni_model import DualStreamOmni, balanced_observation_bce
from fine_grain.omni_tasks import GRID_PLACES
from fine_grain.pythia_bridge import (
    CAPABILITY_CHAMPION_PATH,
    capability_champion_kwargs,
    param_groups,
)
from fine_grain.vlm_data import COLORS, OCR_DIGITS
from scripts.train_pythia_generation import (
    digit_shuffle_prompts,
    source_shuffle_images,
    t2i_gate,
)


PROTECTED_CHECKPOINTS = {
    ROOT / "checkpoints" / "northstar_slice_capability_best.pt",
    ROOT / "checkpoints" / "omni_d64_northstar_omni_active_f2_grid_best.pt",
    ROOT / "checkpoints" / "omni_d64_pythia_language_best.pt",
    ROOT / "checkpoints" / "omni_d64_pythia_named_edit_spatial_best.pt",
    ROOT / "checkpoints" / "omni_d64_pythia_capability_b2_best.pt",
    ROOT / "checkpoints" / "omni_d64_pythia_capability_b3_edit_best.pt",
}


def make_static_bank(
    res: int,
    case: str,
    *,
    glyph_box: int | None = None,
    glyph_stroke_px: int = 1,
    normalized_layout: bool = False,
) -> list[dict]:
    """Balanced digit/color/address bank in one explicit boundary condition."""
    if case not in ("text_to_both", "image_to_current", "image_text_edit"):
        raise ValueError(f"B2 static bank does not admit {case!r}")
    seeds = {
        "text_to_both": 1103,
        "image_to_current": 2207,
        "image_text_edit": 3301,
    }
    rng = np.random.default_rng(seeds[case])
    bank = [
        capability_sample(
            rng, res, case, digit=digit, color=color, place=place,
            glyph_box=glyph_box, glyph_stroke_px=glyph_stroke_px,
            normalized_layout=normalized_layout,
        )
        for place in GRID_PLACES
        for digit in OCR_DIGITS
        for color in COLORS
    ]
    if case == "text_to_both":
        # The fixed OCR evaluator confuses thin 6/8; preserve the proven T2I
        # rehearsal correction without changing the evaluation distribution.
        sixes = [sample for sample in bank if str(sample["digit"]) == "6"]
        bank.extend(sixes * 3)
    return bank


def make_cycle_eval_bank(
    res: int,
    case: str,
    *,
    glyph_box: int | None = None,
    glyph_stroke_px: int = 1,
    normalized_layout: bool = False,
) -> list[dict]:
    """One color-cycled sample for every digit/address on the same scene API."""
    bank = make_static_bank(
        res, case, glyph_box=glyph_box, glyph_stroke_px=glyph_stroke_px,
        normalized_layout=normalized_layout,
    )
    return [
        sample for sample in bank
        if sample["target_color"] == COLORS[
            (
                OCR_DIGITS.index(sample["digit"])
                + GRID_PLACES.index(sample["source_place"])
            )
            % len(COLORS)
        ]
    ]


def initialize_text_in_from_toy(
    model: DualStreamOmni,
    checkpoint: str | Path,
    ridge: float = 1e-3,
) -> dict:
    """Align frozen-Pythia lexical anchors to the champion's MoT H chart.

    The target is the champion's *post-text_in* word state. This initializes
    the existing 512->d interface in closed form; it adds no encoder, lookup
    bypass, or trainable language model.
    """
    if model.lm is None or model.lm_tok is None:
        raise RuntimeError("lexical alignment requires a frozen LM and tokenizer")
    raw = torch.load(checkpoint, map_location="cpu")
    state = raw.get("state_dict", raw)
    required = (
        "embed.weight",
        "mot_stack.text_in.weight",
        "mot_stack.text_in.bias",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise KeyError(f"toy capability checkpoint misses {missing}")
    toy_embed = state["embed.weight"].float()
    toy_weight = state["mot_stack.text_in.weight"].float()
    toy_bias = state["mot_stack.text_in.bias"].float()
    table = model.lm.get_input_embeddings().weight
    xs, ys = [], []
    for word, row in model.vocab.items():
        if word == "<pad>" or row >= toy_embed.shape[0]:
            continue
        target = toy_weight @ toy_embed[row] + toy_bias
        for form in (word, " " + word):
            token_ids = model.lm_tok(
                form, add_special_tokens=False,
            )["input_ids"]
            for token_id in token_ids:
                xs.append(table[int(token_id)].detach().float().cpu())
                ys.append(target)
    x = torch.stack(xs)
    y = torch.stack(ys)
    xa = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
    regularizer = torch.eye(xa.shape[1]) * float(ridge)
    regularizer[-1, -1] = 0.0
    theta = torch.linalg.solve(
        xa.T @ xa + regularizer,
        xa.T @ y,
    )
    with torch.no_grad():
        model.mot_stack.text_in.weight.copy_(
            theta[:-1].T.to(
                device=model.mot_stack.text_in.weight.device,
                dtype=model.mot_stack.text_in.weight.dtype,
            ),
        )
        model.mot_stack.text_in.bias.copy_(
            theta[-1].to(
                device=model.mot_stack.text_in.bias.device,
                dtype=model.mot_stack.text_in.bias.dtype,
            ),
        )
    return {
        "checkpoint": str(checkpoint),
        "anchors": len(xs),
        "ridge": float(ridge),
        "fit_mse": float((xa @ theta - y).pow(2).mean()),
    }


def _balanced_seg_nll(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    loss = F.cross_entropy(logits, target.long(), reduction="none")
    fg = target > 0
    bg = ~fg

    def pane(mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(loss.dtype)
        return (loss * weight).sum(dim=(1, 2)) / weight.sum(
            dim=(1, 2),
        ).clamp_min(1.0)

    return 0.5 * pane(fg) + 0.5 * pane(bg)


def _observation_energy(out: dict, batch: dict, seg_weight: float) -> torch.Tensor:
    target_rgb = batch["target_rgb"].to(out["rgb"].device)
    rgb = balanced_observation_bce(
        out["rgb"], target_rgb, signed=False, reduction="none",
    )
    if out.get("seg_logits") is None:
        return rgb
    target_seg = batch["target_seg"].to(out["seg_logits"].device)
    return rgb + float(seg_weight) * _balanced_seg_nll(
        out["seg_logits"], target_seg,
    )


def paired_digit_likelihood(
    out: dict,
    batch: dict,
    temperature: float = 0.1,
    seg_weight: float = 0.25,
    difference_only: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Match ten prompts to targets through existing RGB/seg likelihoods.

    ``difference_only`` keeps the matched observation energy unchanged but
    evaluates mismatched targets only on their symmetric-difference pixels.
    It prevents common digit strokes from becoming false negatives while
    retaining a plain pairwise observation-energy contrast.
    """
    digits = list(batch["digit"])
    if len(digits) != 10 or set(digits) != set(OCR_DIGITS):
        zero = out["rgb"].sum() * 0.0
        return zero, {"digit_group_active": False}
    if len(set(batch["target_color"])) != 1 or len(set(batch["source_place"])) != 1:
        raise ValueError("digit contrast requires one color and one address")
    order = torch.tensor(
        [digits.index(digit) for digit in OCR_DIGITS],
        device=out["rgb"].device,
    )
    pred_rgb = out["rgb"].index_select(0, order).clamp(0.0, 1.0)
    target_rgb = batch["target_rgb"].to(pred_rgb.device).index_select(0, order)
    pred_seg = out["seg_logits"].index_select(0, order)
    target_seg = batch["target_seg"].to(pred_rgb.device).index_select(0, order)
    diagonal = _observation_energy(
        {"rgb": pred_rgb, "seg_logits": pred_seg},
        {"target_rgb": target_rgb, "target_seg": target_seg}, seg_weight,
    )
    columns = []
    for column in range(10):
        target_rgb_column = target_rgb[column : column + 1].expand_as(pred_rgb)
        seg_target = target_seg[column : column + 1].expand(
            pred_seg.shape[0], -1, -1,
        )
        if difference_only:
            difference = (
                (target_rgb - target_rgb_column).abs().amax(dim=1) > 1e-5
            )
            rgb_error = torch.nn.functional.binary_cross_entropy(
                pred_rgb.clamp(1e-5, 1.0 - 1e-5), target_rgb_column,
                reduction="none",
            ).mean(dim=1)
            seg_error = torch.nn.functional.cross_entropy(
                pred_seg, seg_target.long(), reduction="none",
            )
            weight = difference.to(rgb_error.dtype)
            count = weight.flatten(1).sum(dim=1).clamp_min(1.0)
            energy = ((rgb_error + float(seg_weight) * seg_error) * weight).flatten(1).sum(dim=1) / count
            energy[column] = diagonal[column]
        else:
            rgb = balanced_observation_bce(
                pred_rgb, target_rgb_column, reduction="none",
            )
            seg = _balanced_seg_nll(pred_seg, seg_target)
            energy = rgb + float(seg_weight) * seg
        columns.append(energy)
    energy = torch.stack(columns, dim=1)
    labels = torch.arange(10, device=energy.device)
    logits = -energy / float(temperature)
    loss = 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )
    offdiag = energy.masked_select(
        ~torch.eye(10, dtype=torch.bool, device=energy.device),
    ).mean()
    diagonal = energy.diagonal().mean()
    return loss, {
        "digit_group_active": True,
        "digit_group_contrast": float(loss.detach()),
        "digit_energy_diagonal": float(diagonal.detach()),
        "digit_energy_offdiagonal": float(offdiag.detach()),
        "digit_energy_gap": float((offdiag - diagonal).detach()),
        "digit_group_difference_only": bool(difference_only),
    }


def digit_residual_likelihood(
    out: dict,
    batch: dict,
    seg_weight: float = 0.25,
) -> tuple[torch.Tensor, dict]:
    """Match each prompt's observed glyph residual around its group mean.

    This is a grouped RGB/seg observation likelihood, not a digit classifier:
    common strokes cancel from both target and prediction residuals, while
    prompt-specific strokes retain their signed target signal.
    """
    digits = list(batch["digit"])
    if len(digits) != 10 or set(digits) != set(OCR_DIGITS):
        zero = out["rgb"].sum() * 0.0
        return zero, {"digit_residual_active": False}
    if len(set(batch["target_color"])) != 1 or len(set(batch["source_place"])) != 1:
        raise ValueError("digit residual requires one color and one address")
    order = torch.tensor([digits.index(digit) for digit in OCR_DIGITS], device=out["rgb"].device)
    pred_rgb = out["rgb"].index_select(0, order).clamp(0.0, 1.0)
    target_rgb = batch["target_rgb"].to(pred_rgb.device).index_select(0, order)
    pred_seg = torch.softmax(out["seg_logits"].index_select(0, order), dim=1)[:, 1]
    target_seg = batch["target_seg"].to(pred_rgb.device).index_select(0, order).to(pred_seg.dtype)
    union = target_rgb.abs().amax(dim=(0, 1), keepdim=True) > 1e-5
    weight = union.to(pred_rgb.dtype)
    count = weight.sum().clamp_min(1.0)
    rgb_residual = pred_rgb - pred_rgb.mean(dim=0, keepdim=True)
    target_residual = target_rgb - target_rgb.mean(dim=0, keepdim=True)
    rgb = ((rgb_residual - target_residual).square() * weight).sum() / (count * pred_rgb.shape[0] * pred_rgb.shape[1])
    seg_residual = pred_seg - pred_seg.mean(dim=0, keepdim=True)
    target_seg_residual = target_seg - target_seg.mean(dim=0, keepdim=True)
    seg = ((seg_residual - target_seg_residual).square() * weight[:, 0]).sum() / (count * pred_seg.shape[0])
    loss = rgb + float(seg_weight) * seg
    return loss, {
        "digit_residual_active": True,
        "digit_residual_rgb": float(rgb.detach()),
        "digit_residual_seg": float(seg.detach()),
        "digit_residual_union_pixels": int(union.sum()),
    }


def _forward(model: DualStreamOmni, batch: dict, image=None, prompts=None) -> dict:
    device = next(model.parameters()).device
    n = len(batch["prompt"])
    return model(
        batch["image"].to(device) if image is None else image,
        list(batch["prompt"]) if prompts is None else list(prompts),
        t=torch.zeros(n, device=device),
        image_precision=batch["image_precision"].to(device),
        text_precision=batch["text_precision"].to(device),
        target_time=batch["target_time"].to(device),
        task_id=(
            batch["task_id"].to(device)
            if bool(getattr(model.mot_stack, "use_task_tokens", False))
            and "task_id" in batch
            else None
        ),
    )


def static_one_step(
    model: DualStreamOmni,
    samples: list[dict],
    bank: list[dict],
    rng: np.random.Generator,
    *,
    digit_shuffle_coef: float = 1.0,
    current_shuffle_coef: float = 0.1,
    shuffle_margin: float = 0.05,
    seg_energy_weight: float = 0.25,
    digit_group_coef: float = 0.1,
    digit_group_temperature: float = 0.1,
    digit_group_difference_only: bool = False,
    digit_residual_coef: float = 0.0,
    edit_shuffle_coef: float = 0.1,
    foreground_bce_coef: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """One boundary-conditioned update using existing RGB/seg likelihoods."""
    case = str(samples[0]["case"])
    if any(str(sample["case"]) != case for sample in samples):
        raise ValueError("static_one_step expects one boundary per batch")
    batch = collate_capability(samples)
    n = len(samples)
    batch["need_text"] = [False] * n
    batch["target_text_precision"] = torch.zeros(n)
    if case == "image_to_current":
        # North-Star boundary: current perception observes the image, not a
        # task label. The old pi_text=1 reconstruction prompt let T2I prior
        # updates rewrite an otherwise valid source posterior.
        batch["prompt"] = [""] * n
        batch["text_precision"] = torch.zeros(n)
    out = _forward(model, batch)
    device = out["rgb"].device
    loss, meta = model.omni_loss(out, batch, device)
    # Gaussian RGB likelihood is intentionally valid for natural images, but
    # its point-average can be background-dominated for a sparse 64px digit.
    # This optional *same-observation* pane term is a curriculum weight, not
    # an auxiliary head or a second target.  Default zero preserves every
    # registered historical protocol.
    if float(foreground_bce_coef) != 0.0:
        foreground = balanced_observation_bce(
            out["rgb"], batch["target_rgb"].to(device), signed=False,
        )
        loss = loss + float(foreground_bce_coef) * foreground
        meta["foreground_bce"] = float(foreground.detach())
        meta["foreground_bce_coef"] = float(foreground_bce_coef)

    if case == "text_to_both":
        coef = float(digit_shuffle_coef)
        tag = "digit_shuffle"
        shuffled_prompts = digit_shuffle_prompts(samples, batch["prompt"])
        out_s = _forward(model, batch, prompts=shuffled_prompts) if coef else None
    elif case in ("image_to_current", "image_text_edit"):
        coef = float(
            current_shuffle_coef
            if case == "image_to_current"
            else edit_shuffle_coef
        )
        tag = "source_shuffle"
        shuffled_image = source_shuffle_images(samples, bank, rng, device)
        out_s = _forward(model, batch, image=shuffled_image) if coef else None
    else:
        raise ValueError(f"unsupported static case {case!r}")

    if coef:
        matched = _observation_energy(out, batch, seg_energy_weight)
        shuffled = _observation_energy(out_s, batch, seg_energy_weight)
        contrast = torch.relu(
            matched - shuffled + float(shuffle_margin),
        ).mean()
        loss = loss + coef * contrast
        meta[tag] = float(contrast.detach())
        meta["matched_energy"] = float(matched.mean().detach())
        meta["shuffled_energy"] = float(shuffled.mean().detach())
        meta["shuffle_coef_used"] = coef
    if case == "text_to_both" and float(digit_group_coef) != 0.0:
        group, group_meta = paired_digit_likelihood(
            out, batch, temperature=digit_group_temperature,
            seg_weight=seg_energy_weight,
            difference_only=digit_group_difference_only,
        )
        if group_meta.get("digit_group_active"):
            loss = loss + float(digit_group_coef) * group
        meta.update(group_meta)
    if case == "text_to_both" and float(digit_residual_coef) != 0.0:
        residual, residual_meta = digit_residual_likelihood(
            out, batch, seg_weight=seg_energy_weight,
        )
        if residual_meta.get("digit_residual_active"):
            loss = loss + float(digit_residual_coef) * residual
        meta.update(residual_meta)
    meta["case"] = case
    meta["image_precision"] = float(batch["image_precision"][0])
    meta["text_precision"] = float(batch["text_precision"][0])
    return loss, meta


def sample_digit_group(bank: list[dict], rng: np.random.Generator) -> list[dict]:
    """Select all ten digits at one color/address for identified contrast."""
    place = str(rng.choice(GRID_PLACES))
    color = str(rng.choice(COLORS))
    selected = []
    for digit in OCR_DIGITS:
        matches = [
            sample for sample in bank
            if sample["source_place"] == place
            and sample["target_color"] == color
            and sample["digit"] == digit
        ]
        if not matches:
            raise RuntimeError(f"missing digit group {place}/{color}/{digit}")
        selected.append(matches[0])
    return selected


def requires_digit_group(case: str, digit_group_coef: float, digit_residual_coef: float) -> bool:
    """Whether a T2I update must carry the ten prompt-conditioned examples.

    Both the legacy contrast energy and the residual observation likelihood
    compare predictions across all digits.  Sampling an ordinary minibatch
    silently disables either objective, so keep this decision explicit and
    independently testable.
    """
    return (
        str(case) == "text_to_both"
        and (float(digit_group_coef) != 0.0 or float(digit_residual_coef) != 0.0)
    )


def _seg_iou(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=0) > 0
    gold = target.to(pred.device) > 0
    union = (pred | gold).sum().clamp_min(1)
    return float((pred & gold).sum() / union)


@torch.no_grad()
def eval_static_t2i(
    model: DualStreamOmni,
    samples: list[dict],
    device: torch.device,
    chunk: int = 30,
) -> dict:
    """Strict T2I gate on the same capability scene generator as all ports."""
    model.eval()
    pred, digit_shuffled, color_shuffled = [], [], []
    for start in range(0, len(samples), int(chunk)):
        part = samples[start : start + int(chunk)]
        batch = collate_capability(part)
        pred.append(_forward(model, batch)["rgb"].clamp(0.0, 1.0).cpu())
        digit_prompts = digit_shuffle_prompts(part, batch["prompt"])
        digit_shuffled.append(
            _forward(model, batch, prompts=digit_prompts)["rgb"]
            .clamp(0.0, 1.0).cpu()
        )
        color_prompts = []
        for sample in part:
            color = sample["target_color"]
            other = COLORS[(COLORS.index(color) + 1) % len(COLORS)]
            color_prompts.append(sample["prompt"].replace(color, other, 1))
        color_shuffled.append(
            _forward(model, batch, prompts=color_prompts)["rgb"]
            .clamp(0.0, 1.0).cpu()
        )
    pred = torch.cat(pred)
    digit_shuffled = torch.cat(digit_shuffled)
    color_shuffled = torch.cat(color_shuffled)
    target = torch.cat([sample["target_rgb"] for sample in samples])
    stroke = torch.cat([sample["stroke"] for sample in samples])
    rows, digit_rows = [], []
    ious, centroid = [], []
    for i, sample in enumerate(samples):
        digit, color = sample["digit"], sample["target_color"]
        rows.append(gen_free_scores(pred[i : i + 1], digit, color))
        digit_rows.append(
            gen_free_scores(digit_shuffled[i : i + 1], digit, color),
        )
        ious.append(paired_ink_iou(pred[i : i + 1], stroke[i : i + 1], color))
        centroid.append(
            ink_centroid_error(pred[i : i + 1], stroke[i : i + 1], color),
        )
    n = max(1, len(samples))
    return {
        "n": len(samples),
        "psnr": float(
            -10.0 * math.log10(max(float((pred - target).pow(2).mean()), 1e-8))
        ),
        "digit_top1": float(sum(row["digit_top1"] for row in rows) / n),
        "digit_iou": float(sum(row["digit_iou"] for row in rows) / n),
        "color_acc": float(sum(row["color_acc"] for row in rows) / n),
        "paired_iou": float(sum(ious) / n),
        "centroid_error": float(sum(centroid) / n),
        "flood": background_flood_rate(pred, target, stroke),
        "digit_shuffled_top1": float(
            sum(row["digit_top1"] for row in digit_rows) / n
        ),
        "color_shuffled_acc": float(
            sum(
                gen_free_scores(
                    color_shuffled[i : i + 1],
                    samples[i]["digit"],
                    samples[i]["target_color"],
                )["color_acc"]
                for i in range(len(samples))
            )
            / n
        ),
        "bce": float(balanced_observation_bce(pred, target)),
    }


@torch.no_grad()
def eval_current(
    model: DualStreamOmni,
    samples: list[dict],
    device: torch.device,
    chunk: int = 30,
) -> dict:
    """Evaluate identity-action reconstruction and causal source dependence."""
    model.eval()
    pred, shuffled_pred = [], []
    seg, shuffled_seg = [], []
    rng = np.random.default_rng(991)
    for start in range(0, len(samples), int(chunk)):
        part = samples[start : start + int(chunk)]
        batch = collate_capability(part)
        batch["prompt"] = [""] * len(part)
        batch["text_precision"] = torch.zeros(len(part))
        out = _forward(model, batch)
        shuf = source_shuffle_images(part, samples, rng, device)
        out_s = _forward(model, batch, image=shuf)
        pred.append(out["rgb"].clamp(0.0, 1.0).cpu())
        shuffled_pred.append(out_s["rgb"].clamp(0.0, 1.0).cpu())
        seg.append(out["seg_logits"].cpu())
        shuffled_seg.append(out_s["seg_logits"].cpu())
    pred = torch.cat(pred)
    shuffled_pred = torch.cat(shuffled_pred)
    seg = torch.cat(seg)
    shuffled_seg = torch.cat(shuffled_seg)
    target = torch.cat([sample["target_rgb"] for sample in samples])
    stroke = torch.cat([sample["stroke"] for sample in samples])
    rows, shuffled_rows = [], []
    ious, shuffled_ious, seg_ious, shuffled_seg_ious = [], [], [], []
    for i, sample in enumerate(samples):
        digit, color = sample["digit"], sample["target_color"]
        rows.append(gen_free_scores(pred[i : i + 1], digit, color))
        shuffled_rows.append(
            gen_free_scores(shuffled_pred[i : i + 1], digit, color),
        )
        ious.append(paired_ink_iou(pred[i : i + 1], stroke[i : i + 1], color))
        shuffled_ious.append(
            paired_ink_iou(
                shuffled_pred[i : i + 1], stroke[i : i + 1], color,
            ),
        )
        seg_ious.append(_seg_iou(seg[i], sample["target_seg"]))
        shuffled_seg_ious.append(
            _seg_iou(shuffled_seg[i], sample["target_seg"]),
        )
    n = max(1, len(samples))
    digit = sum(row["digit_top1"] for row in rows) / n
    shuffled_digit = sum(row["digit_top1"] for row in shuffled_rows) / n
    iou = sum(ious) / n
    shuffled_iou = sum(shuffled_ious) / n
    seg_iou = sum(seg_ious) / n
    shuffled_seg_iou = sum(shuffled_seg_ious) / n
    return {
        "n": len(samples),
        "psnr": float(
            -10.0 * math.log10(max(float((pred - target).pow(2).mean()), 1e-8))
        ),
        "digit_top1": float(digit),
        "color_acc": float(sum(row["color_acc"] for row in rows) / n),
        "paired_iou": float(iou),
        "seg_iou": float(seg_iou),
        "flood": background_flood_rate(pred, target, stroke),
        "source_shuffled_digit_top1": float(shuffled_digit),
        "source_shuffled_paired_iou": float(shuffled_iou),
        "source_shuffled_seg_iou": float(shuffled_seg_iou),
        "source_digit_gap": float(digit - shuffled_digit),
        "source_iou_gap": float(iou - shuffled_iou),
        "source_seg_gap": float(seg_iou - shuffled_seg_iou),
    }


def current_gate(metrics: dict) -> bool:
    return bool(
        metrics["digit_top1"] >= 0.85
        and metrics["color_acc"] >= 0.90
        and metrics["paired_iou"] >= 0.85
        and metrics["seg_iou"] >= 0.90
        and metrics["flood"] <= 0.10
        and metrics["source_iou_gap"] >= 0.30
    )


@torch.no_grad()
def eval_edit_static(
    model: DualStreamOmni,
    samples: list[dict],
    device: torch.device,
    chunk: int = 30,
) -> dict:
    """Official next-color IT2I on the unified capability scene family."""
    model.eval()
    pred, shuffled_pred, seg = [], [], []
    rng = np.random.default_rng(1771)
    for start in range(0, len(samples), int(chunk)):
        part = samples[start : start + int(chunk)]
        batch = collate_capability(part)
        out = _forward(model, batch)
        shuffled = source_shuffle_images(part, samples, rng, device)
        out_s = _forward(model, batch, image=shuffled)
        pred.append(out["rgb"].clamp(0.0, 1.0).cpu())
        shuffled_pred.append(out_s["rgb"].clamp(0.0, 1.0).cpu())
        seg.append(out["seg_logits"].cpu())
    pred = torch.cat(pred)
    shuffled_pred = torch.cat(shuffled_pred)
    seg = torch.cat(seg)
    target = torch.cat([sample["target_rgb"] for sample in samples])
    stroke = torch.cat([sample["stroke"] for sample in samples])
    rows, shuffled_rows, ious, shuffled_ious, seg_ious = [], [], [], [], []
    for i, sample in enumerate(samples):
        digit, color = sample["digit"], sample["target_color"]
        rows.append(gen_free_scores(pred[i : i + 1], digit, color))
        shuffled_rows.append(
            gen_free_scores(shuffled_pred[i : i + 1], digit, color),
        )
        ious.append(paired_ink_iou(pred[i : i + 1], stroke[i : i + 1], color))
        shuffled_ious.append(
            paired_ink_iou(
                shuffled_pred[i : i + 1], stroke[i : i + 1], color,
            ),
        )
        seg_ious.append(_seg_iou(seg[i], sample["target_seg"]))
    n = max(1, len(samples))
    digit = sum(row["digit_top1"] for row in rows) / n
    shuffled_digit = sum(row["digit_top1"] for row in shuffled_rows) / n
    iou = sum(ious) / n
    shuffled_iou = sum(shuffled_ious) / n
    return {
        "n": len(samples),
        "style": "next-color",
        "digit_top1": float(digit),
        "color_acc": float(sum(row["color_acc"] for row in rows) / n),
        "paired_iou": float(iou),
        "seg_iou": float(sum(seg_ious) / n),
        "flood": background_flood_rate(pred, target, stroke),
        "source_shuffled_digit_top1": float(shuffled_digit),
        "source_shuffled_paired_iou": float(shuffled_iou),
        "source_digit_gap": float(digit - shuffled_digit),
        "source_iou_gap": float(iou - shuffled_iou),
        "psnr": float(
            -10.0 * math.log10(max(float((pred - target).pow(2).mean()), 1e-8))
        ),
    }


def edit_gate(metrics: dict) -> bool:
    return bool(
        metrics["digit_top1"] >= 0.75
        and metrics["color_acc"] >= 0.90
        and metrics["paired_iou"] >= 0.65
        and metrics["seg_iou"] >= 0.95
        and metrics["flood"] <= 0.10
        and metrics["source_digit_gap"] >= 0.30
    )


def _score(t2i: dict, current: dict, edit: dict | None = None) -> float:
    score = float(
        t2i["digit_top1"] + t2i["color_acc"] + t2i["paired_iou"]
        - t2i["flood"] + current["digit_top1"] + current["color_acc"]
        + current["paired_iou"] + current["seg_iou"] - current["flood"]
        + current["source_iou_gap"]
    )
    if edit is not None:
        score += float(
            edit["digit_top1"] + edit["color_acc"] + edit["paired_iou"]
            + edit["seg_iou"] - edit["flood"] + edit["source_digit_gap"]
        )
    return score


def _save(model: DualStreamOmni, path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    runtime_config = capability_champion_kwargs(
        res=int(model.res),
        n_slices=int(model.mot_stack.n_slices),
        local_dilation=int(getattr(model.mot_stack, "local_dilation", 1)),
    )
    torch.save(
        {
            "state_dict": model.non_lm_state_dict(),
            "config": runtime_config,
            "language": model.language_meta(),
            "schema": (
                "pythia-capability-b3-edit"
                if payload.get("edit") is not None
                else "pythia-capability-b2"
            ),
            **payload,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default=str(CAPABILITY_CHAMPION_PATH))
    parser.add_argument(
        "--ckpt",
        default=str(
            ROOT / "checkpoints" / "omni_d64_pythia_capability_trial_best.pt"
        ),
    )
    parser.add_argument(
        "--candidate-ckpt",
        default=str(
            ROOT / "checkpoints" / "omni_d64_pythia_capability_b2_candidate.pt"
        ),
        help="Best partial state; never presented as an admitted champion.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "results" / "published" / "pythia_capability_b2.json"),
    )
    parser.add_argument("--language", default="pythia")
    parser.add_argument("--lm-device", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--resolution", type=int, default=16,
        help="Registered field resolution; changing it requires a separately audited candidate.",
    )
    parser.add_argument(
        "--n-slices", type=int, default=16,
        help="Transient Slice count for this candidate's single X--Slice--H graph.",
    )
    parser.add_argument(
        "--local-dilation", type=int, default=1,
        help=(
            "Physical mesh spacing of the loaded local stencil; 2 preserves "
            "a 16px-trained receptive field on a 32px proportional field."
        ),
    )
    parser.add_argument(
        "--glyph-scale", type=float, default=None,
        help="Optional fraction of resolution for a registered scale curriculum.",
    )
    parser.add_argument(
        "--glyph-stroke-scale", type=float, default=None,
        help="Optional fraction of resolution for proportional glyph stroke width.",
    )
    parser.add_argument(
        "--normalized-glyph-layout", action="store_true",
        help="Keep named grid addresses proportional across curriculum resolutions.",
    )
    parser.add_argument(
        "--t2i-warmup-steps", type=int, default=100,
        help="Generation-first language binding before capability rehearsal.",
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="Alternating T2I/current refinement steps after T2I warmup.",
    )
    parser.add_argument(
        "--edit-steps", type=int, default=0,
        help="Official next-color phase after the T2I/current prerequisite.",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--interface-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-interface-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-visual-lr", type=float, default=1e-5)
    parser.add_argument(
        "--warmup-phase", choices=("language", "language_rgb"),
        default="language_rgb",
    )
    parser.add_argument(
        "--warmup-current-every", type=int, default=4,
        help="Use one missing-text current batch every N warmup steps; 0 disables.",
    )
    parser.add_argument("--digit-shuffle-coef", type=float, default=1.0)
    parser.add_argument("--current-shuffle-coef", type=float, default=0.1)
    parser.add_argument("--edit-shuffle-coef", type=float, default=0.1)
    parser.add_argument("--edit-interface-lr", type=float, default=5e-5)
    parser.add_argument("--shuffle-margin", type=float, default=0.05)
    parser.add_argument("--seg-energy-weight", type=float, default=0.25)
    parser.add_argument("--digit-group-coef", type=float, default=0.1)
    parser.add_argument("--digit-group-temperature", type=float, default=0.1)
    parser.add_argument(
        "--digit-group-difference-only", action="store_true",
        help="Contrast digit targets only where their observed RGB/seg fields differ.",
    )
    parser.add_argument(
        "--digit-residual-coef", type=float, default=0.0,
        help="Grouped prompt-specific RGB/seg residual likelihood; default preserves history.",
    )
    parser.add_argument("--load-language", action="store_true")
    parser.add_argument(
        "--lexical-ridge", action=argparse.BooleanOptionalAction, default=True,
        help="Initialize Pythia text_in in the toy champion's proven H chart.",
    )
    parser.add_argument("--lexical-ridge-coef", type=float, default=1e-3)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    init_path = Path(args.init)
    ckpt_path = Path(args.ckpt)
    candidate_path = Path(args.candidate_ckpt)
    protected = {path.resolve() for path in PROTECTED_CHECKPOINTS}
    if not args.eval_only and ckpt_path.resolve() in protected:
        raise SystemExit(f"refusing to overwrite protected checkpoint {ckpt_path}")
    if not args.eval_only and candidate_path.resolve() in protected:
        raise SystemExit(
            f"refusing to overwrite protected checkpoint {candidate_path}"
        )
    if not init_path.exists():
        raise SystemExit(f"missing initialization checkpoint {init_path}")

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = torch.device(args.device)
    lm_device = args.lm_device or args.device
    model = DualStreamOmni(
        **capability_champion_kwargs(
            language=args.language, lm_device=lm_device,
            res=int(args.resolution), n_slices=int(args.n_slices),
            local_dilation=int(args.local_dilation),
        )
    ).to(device)
    report = model.load_visual_champion(
        init_path, skip_language_interface=not args.load_language,
    )
    lexical_init = None
    if args.lexical_ridge and not args.load_language:
        lexical_init = initialize_text_in_from_toy(
            model, init_path, ridge=args.lexical_ridge_coef,
        )
    model.set_optimization_phase(args.warmup_phase)
    warmup_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    def make_optimizer(interface_lr: float, visual_lr: float):
        return torch.optim.AdamW(
            param_groups(
                model, interface_lr=interface_lr, visual_lr=visual_lr,
            ),
            weight_decay=1e-4,
        )

    optimizer = make_optimizer(
        args.warmup_interface_lr, args.warmup_visual_lr,
    )
    glyph_box = (
        None if args.glyph_scale is None
        else int(round(float(args.glyph_scale) * model.res))
    )
    glyph_stroke_px = (
        1 if args.glyph_stroke_scale is None
        else max(1, int(round(float(args.glyph_stroke_scale) * model.res)))
    )
    bank_kw = {
        "glyph_box": glyph_box,
        "glyph_stroke_px": glyph_stroke_px,
        "normalized_layout": bool(args.normalized_glyph_layout),
    }
    t2i_train = make_static_bank(model.res, "text_to_both", **bank_kw)
    current_train = make_static_bank(model.res, "image_to_current", **bank_kw)
    edit_train = make_static_bank(model.res, "image_text_edit", **bank_kw)
    t2i_eval = make_cycle_eval_bank(model.res, "text_to_both", **bank_kw)
    current_eval = make_cycle_eval_bank(model.res, "image_to_current", **bank_kw)
    edit_eval = make_cycle_eval_bank(model.res, "image_text_edit", **bank_kw)
    history: list[dict] = []
    best_score = float("-inf")
    best_partial_score = float("-inf")
    best_step = 0
    best_partial_step = 0
    did_save = False
    started = time.time()

    use_edit_gate = int(args.edit_steps) > 0

    def evaluate(step: int, meta: dict | None = None) -> tuple[dict, dict, dict | None]:
        nonlocal best_score, best_partial_score
        nonlocal best_step, best_partial_step, did_save
        t2i = eval_static_t2i(model, t2i_eval, device, chunk=args.batch)
        current = eval_current(model, current_eval, device, chunk=args.batch)
        edit = (
            eval_edit_static(model, edit_eval, device, chunk=args.batch)
            if use_edit_gate else None
        )
        gates = {"t2i": t2i_gate(t2i), "current": current_gate(current)}
        if edit is not None:
            gates["edit"] = edit_gate(edit)
        score = _score(t2i, current, edit)
        row = {
            "step": step,
            "meta": meta or {},
            "t2i": t2i,
            "current": current,
            "edit": edit,
            "gates": gates,
            "score": score,
            "elapsed_sec": time.time() - started,
        }
        history.append(row)
        if score > best_partial_score and not args.eval_only:
            best_partial_score, best_partial_step = score, step
            _save(
                model, candidate_path,
                {
                    "step": step,
                    "t2i": t2i,
                    "current": current,
                    "edit": edit,
                    "admitted": False,
                    "candidate_only": True,
                },
            )
        if all(gates.values()) and score > best_score and not args.eval_only:
            best_score, best_step, did_save = score, step, True
            _save(
                model, ckpt_path,
                {"step": step, "t2i": t2i, "current": current, "edit": edit},
            )
        edit_text = "" if edit is None else (
            f" edit={edit['color_acc']:.3f}/{edit['paired_iou']:.3f}"
        )
        gate_text = "".join(str(int(value)) for value in gates.values())
        print(
            f"step={step:4d} t2i={t2i['digit_top1']:.3f}/"
            f"{t2i['paired_iou']:.3f} current={current['digit_top1']:.3f}/"
            f"{current['paired_iou']:.3f} seg={current['seg_iou']:.3f} "
            f"source_gap={current['source_iou_gap']:.3f} "
            f"{edit_text} gates={gate_text}",
            flush=True,
        )
        return t2i, current, edit

    evaluate(0)
    if not args.eval_only:
        warmup = max(0, int(args.t2i_warmup_steps))
        refinement = max(0, int(args.steps))
        edit_steps = max(0, int(args.edit_steps))
        static_steps = warmup + refinement
        total_steps = static_steps + edit_steps
        for step in range(1, total_steps + 1):
            if step == warmup + 1 and refinement > 0:
                model.set_optimization_phase("language")
                optimizer = make_optimizer(args.interface_lr, args.interface_lr)
            if step == static_steps + 1 and edit_steps > 0:
                model.set_optimization_phase("language")
                optimizer = make_optimizer(
                    args.edit_interface_lr, args.edit_interface_lr,
                )
            in_warmup = step <= warmup
            refine_step = step - warmup
            in_edit = step > static_steps
            if in_edit:
                edit_step = step - static_steps
                cycle = (
                    "image_text_edit", "image_text_edit",
                    "text_to_both", "image_to_current",
                )
                case = cycle[(edit_step - 1) % len(cycle)]
            elif in_warmup:
                current_every = max(0, int(args.warmup_current_every))
                case = (
                    "image_to_current"
                    if current_every and step % current_every == 0
                    else "text_to_both"
                )
            else:
                case = (
                    "text_to_both" if refine_step % 2 == 1
                    else "image_to_current"
                )
            if case == "text_to_both":
                bank = t2i_train
            elif case == "image_to_current":
                bank = current_train
            else:
                bank = edit_train
            if requires_digit_group(
                case, args.digit_group_coef, args.digit_residual_coef,
            ):
                samples = sample_digit_group(bank, rng)
            else:
                indices = rng.choice(
                    len(bank), size=min(args.batch, len(bank)), replace=False,
                )
                samples = [bank[int(index)] for index in indices]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, meta = static_one_step(
                model, samples, bank, rng,
                digit_shuffle_coef=args.digit_shuffle_coef,
                current_shuffle_coef=args.current_shuffle_coef,
                shuffle_margin=args.shuffle_margin,
                seg_energy_weight=args.seg_energy_weight,
                digit_group_coef=args.digit_group_coef,
                digit_group_temperature=args.digit_group_temperature,
                digit_group_difference_only=args.digit_group_difference_only,
                digit_residual_coef=args.digit_residual_coef,
                edit_shuffle_coef=args.edit_shuffle_coef,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()
            meta["loss"] = float(loss.detach())
            if step % args.eval_every == 0 or step == total_steps:
                evaluate(step, meta)

    terminal = history[-1]
    if did_save:
        model.load_visual_champion(ckpt_path, skip_language_interface=False)
        selected_t2i = eval_static_t2i(
            model, t2i_eval, device, chunk=args.batch,
        )
        selected_current = eval_current(
            model, current_eval, device, chunk=args.batch,
        )
        selected_edit = (
            eval_edit_static(model, edit_eval, device, chunk=args.batch)
            if use_edit_gate else None
        )
        final = {
            "t2i": selected_t2i,
            "current": selected_current,
            "edit": selected_edit,
            "gates": {
                "t2i": t2i_gate(selected_t2i),
                "current": current_gate(selected_current),
                **(
                    {"edit": edit_gate(selected_edit)}
                    if selected_edit is not None else {}
                ),
            },
        }
    else:
        final = terminal
    record = {
        "schema": (
            "pythia-capability-b3-edit" if use_edit_gate
            else "pythia-capability-b2"
        ),
        "claim_scope": (
            "Frozen-Pythia binding to the proven static Slice capability chart; "
            "editing is admitted only after T2I and identity reconstruction pass."
        ),
        "init": str(init_path),
        "init_report": report,
        "lexical_init": lexical_init,
        "checkpoint": str(init_path if args.eval_only else ckpt_path),
        "candidate_checkpoint": str(candidate_path),
        "did_save": did_save,
        "best_step": best_step,
        "best_partial_step": best_partial_step,
        "best_partial_score": (
            None if best_partial_score == float("-inf") else best_partial_score
        ),
        "best_score": None if best_score == float("-inf") else best_score,
        "language": model.language_meta(),
        "run": {
            "warmup_phase": args.warmup_phase,
            "refinement_phase": "language",
            "resolution": int(args.resolution),
            "n_slices": int(args.n_slices),
            "local_dilation": int(args.local_dilation),
            "glyph_box": glyph_box,
            "glyph_stroke_px": glyph_stroke_px,
            "normalized_glyph_layout": bool(args.normalized_glyph_layout),
            "t2i_warmup_steps": int(args.t2i_warmup_steps),
            "refinement_steps": int(args.steps),
            "edit_steps": int(args.edit_steps),
            "schedule": (
                "generation-first text_to_both warmup, then deterministic "
                "alternating text_to_both,image_to_current"
                + (
                    ", then next-edit,next-edit,T2I,current"
                    if use_edit_gate else ""
                )
            ),
            "warmup_trainable": warmup_trainable,
            "refinement_trainable": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "stem_read_deslice_seg_frozen": True,
            "rgb_head_warmup_only": args.warmup_phase == "language_rgb",
            "interface_lr": args.interface_lr,
            "warmup_interface_lr": args.warmup_interface_lr,
            "warmup_visual_lr": args.warmup_visual_lr,
            "warmup_current_every": args.warmup_current_every,
            "digit_shuffle_coef": args.digit_shuffle_coef,
            "current_shuffle_coef": args.current_shuffle_coef,
            "edit_shuffle_coef": args.edit_shuffle_coef,
            "edit_interface_lr": args.edit_interface_lr,
            "shuffle_margin": args.shuffle_margin,
            "seg_energy_weight": args.seg_energy_weight,
            "digit_group_coef": args.digit_group_coef,
            "digit_group_temperature": args.digit_group_temperature,
            "digit_group_difference_only": bool(args.digit_group_difference_only),
            "digit_residual_coef": args.digit_residual_coef,
            "gdn2": False,
            "scene_generator": "capability_sample/grid_digit_mask for every port",
        },
        "gates": {
            "t2i": "official pythia T2I gate",
            "current": {
                "digit_top1": 0.85,
                "color_acc": 0.90,
                "paired_iou": 0.85,
                "seg_iou": 0.90,
                "flood_max": 0.10,
                "source_iou_gap": 0.30,
            },
            "edit": (
                {
                    "style": "Change the stroke to the next color",
                    "digit_top1": 0.75,
                    "color_acc": 0.90,
                    "paired_iou": 0.65,
                    "seg_iou": 0.95,
                    "flood_max": 0.10,
                    "source_digit_gap": 0.30,
                }
                if use_edit_gate else None
            ),
        },
        "final": {
            "t2i": final["t2i"],
            "current": final["current"],
            "edit": final.get("edit"),
        },
        "final_gates": final["gates"],
        "terminal_step": terminal["step"],
        "terminal": {
            "t2i": terminal["t2i"],
            "current": terminal["current"],
            "edit": terminal.get("edit"),
            "gates": terminal["gates"],
        },
        "history": history,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {out_path}; did_save={did_save}", flush=True)


if __name__ == "__main__":
    main()
