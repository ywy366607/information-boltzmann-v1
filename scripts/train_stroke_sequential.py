#!/usr/bin/env python3
"""Stroke-sequential generation on the persistent latent canvas.

Human painters write sequentially: each stroke lands on a canvas that stays
visible, and bandwidth accumulates over passes instead of being spent in one
shot. This script tests whether the native Slice graph already expresses that
protocol. The persistent field X is the canvas: step k writes the k-th
sub-stroke of the digit through the SAME Slice-MoT-Deslice path, conditioned
on (a) the accumulated canvas via ``x_init`` and (b) the step index through
the existing ``t_coord`` input coordinate. Sub-stroke supervision comes from
the closed digit polyline templates: each polyline is split into short
segments, trailing no-op steps train write suppression, and the final
cumulative mask must reproduce the registered digit.

This is the H1-corrected sequential form: no RGB round-trip, every write is a
latent Deslice increment onto one persistent X.
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

from fine_grain.capability_tasks import _scene
from fine_grain.gen_metrics import (
    background_flood_rate,
    gen_free_scores,
    ink_centroid_error,
    paired_ink_iou,
)
from fine_grain.native_mot import NativeMoTStack  # noqa: F401  (import check)
from fine_grain.omni_model import DualStreamOmni, balanced_observation_bce
from fine_grain.omni_tasks import GRID_PLACES, _paint, equal_energy_ink
from fine_grain.ocr_1px import _DIGIT_STROKES, _bresenham
from fine_grain.pythia_bridge import load_visual_champion, set_optimization_phase
from fine_grain.vlm_data import COLORS, OCR_DIGITS
from scripts.train_pythia_capabilities import PROTECTED_CHECKPOINTS
from scripts.train_pythia_generation import t2i_gate

DEFAULT_INIT = (
    ROOT / "checkpoints" /
    "omni_d64_northstar_omni_active_f2_grid_best.pt"
)
STROKE_SPLIT = 2  # unit-square edges per sub-stroke
K_MAX = 8  # trailing no-op steps pad every digit to the same schedule


def sub_strokes(digit: str) -> list[list[tuple[int, int]]]:
    """Split a digit's polylines into short sub-strokes (unit-square coords)."""
    subs: list[list[tuple[int, int]]] = []
    for polyline in _DIGIT_STROKES[str(digit)]:
        for start in range(0, len(polyline) - 1, STROKE_SPLIT):
            chunk = list(polyline[start : start + STROKE_SPLIT + 1])
            if len(chunk) >= 2:
                subs.append(chunk)
    return subs


def digit_schedule(digit: str) -> list[list[tuple[float, float]]]:
    """Per-step polyline list; empty list = no-op step."""
    subs = sub_strokes(str(digit))
    padded = list(subs) + [[] for _ in range(K_MAX - len(subs))]
    return padded[:K_MAX]


def draw_substroke(
    mask: np.ndarray, polyline: list[tuple[float, float]],
    res: int, y0: int, x0: int, box: int,
) -> None:
    pts = []
    for ux, uy in polyline:
        x = int(round(x0 + ux * (box - 1)))
        y = int(round(y0 + uy * (box - 1)))
        pts.append((y, x))
    for i in range(len(pts) - 1):
        for y, x in _bresenham(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]):
            if 0 <= y < res and 0 <= x < res:
                mask[y, x] = True


def scene_geometry(res: int, place: str) -> tuple[int, int, int]:
    box = max(4, min(16, (int(res) + 2) // 3, int(res) - 2))
    row, col = str(place).split("_")
    rows = {"top": 1, "middle": (int(res) - box) // 2, "bottom": int(res) - box - 1}
    cols = {"left": 1, "center": (int(res) - box) // 2, "right": int(res) - box - 1}
    return box, int(rows[row]), int(cols[col])


def partial_scene(
    digit: str, color: str, res: int, place: str, k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cumulative scene after k writes: rgb [1,3,R,R], mask [1,R,R]."""
    box, y0, x0 = scene_geometry(res, place)
    schedule = digit_schedule(digit)
    msk = np.zeros((int(res), int(res)), dtype=bool)
    for polyline in schedule[: k + 1]:
        if polyline:
            draw_substroke(msk, polyline, int(res), y0, x0, box)
    mask = torch.from_numpy(msk.astype(np.float32)).view(1, int(res), int(res))
    blank = torch.zeros(1, 3, int(res), int(res), dtype=torch.float32)
    rgb = _paint(blank, mask, equal_energy_ink(str(color)))
    return rgb, mask


def make_bank(res: int) -> list[dict]:
    """Deterministic grid bank: place x digit x color with stroke schedules."""
    bank = []
    for pi, place in enumerate(GRID_PLACES):
        for di, digit in enumerate(OCR_DIGITS):
            for ci, color in enumerate(COLORS):
                rgb, stroke = _scene(digit, color, place, res)
                bank.append({
                    "digit": str(digit),
                    "color": str(color),
                    "place": str(place),
                    "prompt": (
                        f"Draw digit {digit} with a thin {color} stroke "
                        f"at {place.replace('_', ' ')}"
                    ),
                    "target_rgb": rgb,
                    "stroke": stroke,
                    "n_strokes": len(sub_strokes(digit)),
                    "seed": 7000 + 100 * pi + 10 * di + ci,
                })
    return bank


def forward_step(model, prompts, x_init, t_value, device, res):
    """One latent write through the same graph; returns the written field."""
    B = len(prompts)
    zeros_img = torch.zeros(B, 3, res, res, device=device)
    t = torch.full((B,), float(t_value), device=device)
    out = model(
        zeros_img, prompts,
        pi_x=1.0,
        t=t,
        image_precision=torch.zeros(B, device=device),
        text_precision=torch.ones(B, device=device),
        x_init=x_init,
    )
    return out["belief_mu"]


@torch.no_grad()
def sequential_generate(model, samples, device, res, *, ablate_t=False, ablate_canvas=False):
    """Run the K-step schedule; return per-step decoded fields."""
    fields = None
    per_step = []
    prompts = [str(sample["prompt"]) for sample in samples]
    for k in range(K_MAX):
        t_value = 0.0 if ablate_t else k / (K_MAX - 1)
        x_init = fields
        if ablate_canvas and k == K_MAX - 1:
            x_init = torch.zeros_like(fields)
        fields = forward_step(model, prompts, x_init, t_value, device, res)
        rgb = torch.sigmoid(model.decode_field(fields)).clamp(0.0, 1.0)
        per_step.append(rgb.cpu())
    return per_step


def sequential_loss(model, samples, device, res):
    """K-step accumulated-field loss with per-step partial supervision."""
    prompts = [str(sample["prompt"]) for sample in samples]
    targets = []
    for sample in samples:
        rgb_k, _ = partial_scene(
            sample["digit"], sample["color"], res, sample["place"], K_MAX - 1,
        )
        targets.append(rgb_k)
    fields = None
    total = None
    meta_steps = []
    for k in range(K_MAX):
        t_value = k / (K_MAX - 1)
        fields = forward_step(model, prompts, fields, t_value, device, res)
        rgb = torch.sigmoid(model.decode_field(fields)).clamp(1e-5, 1.0 - 1e-5)
        part = torch.cat([
            partial_scene(
                sample["digit"], sample["color"], res, sample["place"], k,
            )[0]
            for sample in samples
        ], dim=0).to(device)
        loss_k = balanced_observation_bce(rgb, part, signed=False, reduction="none")
        total = loss_k.sum() if total is None else total + loss_k.sum()
        meta_steps.append(float(loss_k.mean().detach()))
    return total / len(samples), meta_steps


@torch.no_grad()
def evaluate_sequential(model, bank, device, res, limit=90):
    """Final-field gate metrics plus per-step and causality controls."""
    model.eval()
    chosen = bank[: min(int(limit), len(bank))]
    per_step = sequential_generate(model, chosen, device, res)
    final = per_step[-1]
    rows = []
    for i, sample in enumerate(chosen):
        rows.append(gen_free_scores(final[i : i + 1], sample["digit"], sample["color"]))
    numeric = {
        key for key, value in rows[0].items()
        if isinstance(value, (int, float, bool))
    }
    metrics = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in numeric
    }
    stroke = torch.cat([sample["stroke"] for sample in chosen])
    iou = [
        paired_ink_iou(final[i : i + 1], stroke[i : i + 1], sample["color"])
        for i in range(len(chosen))
    ]
    metrics["paired_iou"] = float(np.mean(iou))
    metrics["centroid_error"] = float(np.mean([
        ink_centroid_error(final[i : i + 1], stroke[i : i + 1], sample["color"])
        for i in range(len(chosen))
    ]))
    target = torch.cat([sample["target_rgb"] for sample in chosen])
    metrics["flood"] = float(background_flood_rate(final, target, stroke))
    metrics["gate"] = t2i_gate(metrics)

    # Per-step fidelity: decoded field vs the cumulative partial target.
    step_iou = []
    for k, rgb_k in enumerate(per_step):
        ious = []
        for i, sample in enumerate(chosen):
            _, mask_k = partial_scene(
                sample["digit"], sample["color"], res, sample["place"], k,
            )
            ious.append(paired_ink_iou(
                rgb_k[i : i + 1], mask_k, sample["color"],
            ))
        step_iou.append(float(np.mean(ious)))
    metrics["per_step_iou"] = step_iou

    # Canvas causality: at the final step, a blank canvas must yield only the
    # last sub-stroke, not the memorized full digit.
    blank_final = sequential_generate(
        model, chosen, device, res, ablate_canvas=True,
    )[-1]
    blank_iou = []
    for i, sample in enumerate(chosen):
        _, mask_last = partial_scene(
            sample["digit"], sample["color"], res, sample["place"], K_MAX - 1,
        )
        blank_iou.append(paired_ink_iou(
            blank_final[i : i + 1], mask_last, sample["color"],
        ))
    metrics["blank_canvas_final_iou"] = float(np.mean(blank_iou))

    # Step-condition causality: constant t=0 must degrade the schedule.
    flat_t = sequential_generate(model, chosen, device, res, ablate_t=True)[-1]
    flat_rows = [
        gen_free_scores(flat_t[i : i + 1], sample["digit"], sample["color"])
        for i, sample in enumerate(chosen)
    ]
    metrics["ablate_t_digit_top1"] = float(np.mean(
        [row["digit_top1"] for row in flat_rows],
    ))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default=str(DEFAULT_INIT))
    parser.add_argument("--candidate", default=str(
        ROOT / "checkpoints" / "omni_d64_stroke_sequential_candidate.pt",
    ))
    parser.add_argument("--out", default=str(
        ROOT / "results" / "published" / "stroke_sequential_gate.json",
    ))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--res", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-slices", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-limit", type=int, default=90)
    parser.add_argument(
        "--write-gamma", type=float, default=0.0,
        help="Optional prescribed write-assignment sharpening dose.",
    )
    parser.add_argument(
        "--prior-step-condition", action="store_true",
        help=(
            "Opt-in step-conditioned prior content: the language prior head "
            "receives the write-step coordinate through a zero-init "
            "projection, so the set-point target varies per stroke."
        ),
    )
    parser.add_argument(
        "--prior-increment", action="store_true",
        help=(
            "Painter write semantics: the language prior emits a slice-space "
            "increment written additively (deslice_write='prior_increment'); "
            "the canvas read-back is never subtracted, so strokes accumulate."
        ),
    )
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    init_path = Path(args.init)
    candidate_path = Path(args.candidate)
    protected = {Path(p).resolve() for p in PROTECTED_CHECKPOINTS}
    if candidate_path.resolve() in protected or candidate_path.resolve() == init_path.resolve():
        raise SystemExit("refusing to overwrite a protected or input checkpoint")
    if not init_path.is_file():
        raise FileNotFoundError(init_path)

    device = torch.device(args.device)
    torch.manual_seed(0)
    model = DualStreamOmni(
        d_model=args.d_model, n_slices=args.n_slices, n_layers=4,
        n_heads=args.n_heads, res=args.res,
        surprise_mode="v1_bayes", s_update="raw",
        prior_loss_coef=0.1, sigreg_coef=0.0,
        use_stiefel=False, deslice_topk=0,
        use_null_slice=False, use_residual_read=False,
        gate_on="u", gate_h_local=False,
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
        deslice_write_sharpening=float(args.write_gamma) > 0.0,
        prior_step_condition=bool(args.prior_step_condition),
        deslice_write=(
            "prior_increment" if args.prior_increment else "increment"
        ),
    ).to(device)
    report = load_visual_champion(model, init_path, skip_language_interface=False)
    if float(args.write_gamma) > 0.0:
        for layer in model.mot_stack.layers:
            layer.deslice.write_gamma_raw.data.fill_(float(math.log(args.write_gamma)))
            layer.deslice.write_gamma_raw.requires_grad_(False)
    trainable = set_optimization_phase(model, "generation_write")
    assert any("t_coord" in name for name in trainable), (
        "step condition must be trainable for the sequential schedule"
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=float(args.lr),
    )
    rng = np.random.default_rng(0)
    bank = make_bank(int(args.res))
    res = int(args.res)

    started = time.time()
    history = []
    best_score = float("-inf")

    def evaluate(step: int):
        nonlocal best_score
        metrics = evaluate_sequential(
            model, bank, device, res, limit=args.eval_limit,
        )
        metrics["step"] = int(step)
        history.append(metrics)
        score = metrics["paired_iou"] + metrics.get("digit_top1", 0.0)
        print(
            f"step={step:5d} digit={metrics.get('digit_top1', 0):.3f} "
            f"color={metrics.get('color_acc', 0):.3f} "
            f"iou={metrics['paired_iou']:.3f} gate={int(metrics['gate'])} "
            f"blank_iou={metrics['blank_canvas_final_iou']:.3f} "
            f"abl_t={metrics['ablate_t_digit_top1']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            if not args.eval_only:
                candidate_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": model.state_dict(),
                    "step": step,
                    "metrics": metrics,
                    "candidate_only": True,
                }, candidate_path)
        return metrics

    evaluate(0)
    if not args.eval_only:
        for step in range(1, int(args.steps) + 1):
            idx = rng.choice(len(bank), size=int(args.batch), replace=False)
            samples = [bank[int(i)] for i in idx]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, _ = sequential_loss(model, samples, device, res)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()
            if step % int(args.eval_every) == 0 or step == int(args.steps):
                evaluate(step)

    final = history[-1]
    record = {
        "schema": "stroke-sequential-canvas-gate",
        "init": str(init_path),
        "init_report": report,
        "candidate": str(candidate_path),
        "run": {
            "steps": int(args.steps),
            "batch": int(args.batch),
            "lr": float(args.lr),
            "k_max": K_MAX,
            "stroke_split": STROKE_SPLIT,
            "write_gamma": float(args.write_gamma),
            "prior_step_condition": bool(args.prior_step_condition),
            "prior_increment": bool(args.prior_increment),
            "trainable": trainable,
            "n_trainable": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "x_init_persistent_canvas": True,
            "rgb_round_trip": False,
            "lm_generate_called": False,
        },
        "final": final,
        "history": history,
        "elapsed_sec": time.time() - started,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
