#!/usr/bin/env python3
"""Bind frozen Pythia embeddings to the generation champion's language readers.

Visual stem / SliceRead / Deslice stay frozen unless an explicit later phase
unfreezes them. Named-edit batches train a source-shuffle hinge (same target
color, different digit/place) with an independent --edit-shuffle-coef.
T2I forwards image_precision=0; named edit forwards image_precision=1; both
use text_precision=1. Does not overwrite published checkpoints.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.capability_tasks import capability_sample
from fine_grain.gen_metrics import (
    background_flood_rate,
    gen_free_scores,
    ink_centroid_error,
    paired_ink_iou,
)
from fine_grain.omni_model import DualStreamOmni, balanced_observation_bce
from fine_grain.omni_tasks import GRID_PLACES, one_sample
from fine_grain.pythia_bridge import (
    GENERATION_CHAMPION_PATH,
    generation_champion_kwargs,
    param_groups,
)
from fine_grain.vlm_data import COLORS


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def make_t2i_bank(res: int, colors=None):
    colors = list(colors or COLORS)
    samples = []
    for pi, place in enumerate(GRID_PLACES):
        for digit in range(10):
            for ci, color in enumerate(colors):
                samples.append(
                    one_sample(
                        np.random.default_rng(4000 + 200 * pi + 20 * digit + ci),
                        res,
                        "t2i",
                        t2i_canvas="black",
                        t2i_place=place,
                        t2i_digit=digit,
                        t2i_color=color,
                    )
                )
    # 6 is scored as 8 on the eval bank; keep it over-represented.
    sixes = [s for s in samples if str(s["digit"]) == "6"]
    samples.extend(sixes * 3)
    return samples


def make_eval_bank(res: int):
    """90-cell grid, colors cycle so color_acc is not free."""
    samples = []
    for pi, place in enumerate(GRID_PLACES):
        for digit in range(10):
            color = COLORS[(pi + digit) % len(COLORS)]
            samples.append(
                one_sample(
                    np.random.default_rng(9000 + 100 * pi + digit),
                    res,
                    "t2i",
                    t2i_canvas="black",
                    t2i_place=place,
                    t2i_digit=digit,
                    t2i_color=color,
                )
            )
    return samples


def digit_shuffle_prompts(samples, prompts):
    out = []
    for s, p in zip(samples, prompts):
        d = str(s["digit"])
        if d == "6":
            other = "8"
        elif d == "8":
            other = "6"
        else:
            other = str((int(d) + 1) % 10)
        out.append(p.replace(f"digit {d}", f"digit {other}", 1))
    return out


def make_edit_bank(res: int, n: int = 72, seed: int = 7, style: str = "named"):
    """IT2I bank. style=named|next — never mix; they are different gates."""
    rng = np.random.default_rng(seed)
    style = str(style).lower()
    if style not in ("named", "next"):
        raise ValueError("edit style must be 'named' or 'next', not a mix")
    samples = []
    for _ in range(n):
        sample = capability_sample(rng, res, "image_text_edit")
        if style == "named":
            sample = dict(sample)
            sample["prompt"] = f"Change the stroke to {sample['target_color']}"
            sample["edit_kind"] = "named"
        else:
            sample["edit_kind"] = "next"
        samples.append(sample)
    return samples


def make_edit_eval(res: int, n: int = 36, seed: int = 123, style: str = "named"):
    return make_edit_bank(res, n=n, seed=seed, style=style)


def tensors_from(samples, device):
    image = torch.cat([s["image"] if s["image"].dim() == 4 else s["image"].unsqueeze(0)
                       for s in samples]).to(device)
    target = torch.cat([s["target_rgb"] if s["target_rgb"].dim() == 4 else s["target_rgb"].unsqueeze(0)
                        for s in samples]).to(device)
    stroke = torch.cat([
        s["stroke"] if s["stroke"].dim() == 3 else s["stroke"].unsqueeze(0)
        for s in samples
    ]).to(device)
    prompts = [s["prompt"] for s in samples]
    return image, target, stroke, prompts


def _run_rgb(model, image, prompts, image_precision, text_precision):
    n = len(prompts)
    B = image.shape[0]
    img_pi = image.new_full((B,), float(image_precision))
    txt_pi = image.new_full((B,), float(text_precision))
    return model(
        image, prompts, need_pix=[True] * n,
        image_precision=img_pi, text_precision=txt_pi,
    )["rgb"].clamp(0, 1)


@torch.no_grad()
def eval_t2i(model, samples, device):
    model.eval()
    image, target, stroke, prompts = tensors_from(samples, device)
    pred = _run_rgb(model, image, prompts, image_precision=0.0, text_precision=1.0)
    rows = [
        gen_free_scores(pred[i : i + 1], samples[i]["digit"], samples[i]["color"])
        for i in range(len(samples))
    ]
    n = len(samples)
    digit_shuffled_prompts = []
    for s in samples:
        other = str((int(s["digit"]) + 1) % 10)
        digit_shuffled_prompts.append(s["prompt"].replace(f"digit {s['digit']}", f"digit {other}"))
    shuffled = _run_rgb(
        model, image, digit_shuffled_prompts, image_precision=0.0, text_precision=1.0,
    )
    sh_rows = [
        gen_free_scores(shuffled[i : i + 1], samples[i]["digit"], samples[i]["color"])
        for i in range(n)
    ]
    color_shuffled_prompts = []
    for s in samples:
        cur = str(s["color"])
        nxt = COLORS[(list(COLORS).index(cur) + 1) % len(COLORS)]
        color_shuffled_prompts.append(s["prompt"].replace(cur, nxt, 1))
    cpred = _run_rgb(
        model, image, color_shuffled_prompts, image_precision=0.0, text_precision=1.0,
    )
    return {
        "n": n,
        "psnr": float(-10.0 * torch.log10((pred - target).pow(2).mean().clamp_min(1e-8))),
        "digit_top1": float(sum(r["digit_top1"] for r in rows) / n),
        "digit_iou": float(sum(r["digit_iou"] for r in rows) / n),
        "color_acc": float(sum(r["color_acc"] for r in rows) / n),
        "paired_iou": float(
            sum(
                paired_ink_iou(pred[i : i + 1], stroke[i : i + 1], samples[i]["color"])
                for i in range(n)
            )
            / n
        ),
        "centroid_error": float(
            sum(
                ink_centroid_error(pred[i : i + 1], stroke[i : i + 1], samples[i]["color"])
                for i in range(n)
            )
            / n
        ),
        "flood": background_flood_rate(pred, target, stroke),
        "digit_shuffled_top1": float(sum(r["digit_top1"] for r in sh_rows) / n),
        "color_shuffled_acc": float(
            sum(
                gen_free_scores(
                    cpred[i : i + 1], samples[i]["digit"], samples[i]["color"],
                )["color_acc"]
                for i in range(n)
            )
            / n
        ),
        "bce": float(balanced_observation_bce(pred, target)),
    }


@torch.no_grad()
def eval_edit(model, samples, device):
    model.eval()
    image, target, stroke, prompts = tensors_from(samples, device)
    pred = _run_rgb(model, image, prompts, image_precision=1.0, text_precision=1.0)
    n = len(samples)
    rows = [
        gen_free_scores(pred[i : i + 1], samples[i]["digit"], samples[i]["target_color"])
        for i in range(n)
    ]
    color_acc = float(sum(r["color_acc"] for r in rows) / n)
    digit_top1 = float(sum(r["digit_top1"] for r in rows) / n)
    iou = float(
        sum(
            paired_ink_iou(pred[i : i + 1], stroke[i : i + 1], samples[i]["target_color"])
            for i in range(n)
        )
        / n
    )
    shuf_img = source_shuffle_images(
        samples, samples, np.random.default_rng(0), device,
    )
    pred_s = _run_rgb(model, shuf_img, prompts, image_precision=1.0, text_precision=1.0)
    sh_rows = [
        gen_free_scores(pred_s[i : i + 1], samples[i]["digit"], samples[i]["target_color"])
        for i in range(n)
    ]
    return {
        "n": n,
        "style": samples[0].get("edit_kind", "unknown") if samples else "unknown",
        "color_acc": color_acc,
        "digit_top1": digit_top1,
        "source_shuffled_digit_top1": float(sum(r["digit_top1"] for r in sh_rows) / n),
        "source_shuffled_color_acc": float(sum(r["color_acc"] for r in sh_rows) / n),
        "paired_iou": iou,
        "psnr": float(-10.0 * torch.log10((pred - target).pow(2).mean().clamp_min(1e-8))),
        "flood": background_flood_rate(pred, target, stroke),
        "score": 0.5 * (color_acc + iou),
    }


def t2i_gate(metrics: dict) -> bool:
    return bool(
        metrics["digit_top1"] >= 0.95
        and metrics["color_acc"] >= 0.95
        and metrics["flood"] <= 0.10
        and metrics["paired_iou"] >= 0.85
        and metrics["centroid_error"] <= 0.08
        and metrics["digit_top1"] - metrics["digit_shuffled_top1"] >= 0.50
        and metrics["color_acc"] - metrics.get("color_shuffled_acc", 1.0) >= 0.50
    )


def edit_gate(metrics: dict) -> bool:
    geom = float(metrics.get("digit_top1", 0.0)) - float(
        metrics.get("source_shuffled_digit_top1", 1.0)
    )
    return bool(
        metrics["score"] >= 0.80
        and metrics["color_acc"] >= 0.80
        and geom >= 0.30
    )


def _is_edit_sample(sample) -> bool:
    return bool(sample.get("edit_kind") or sample.get("case") == "image_text_edit")


def source_shuffle_images(batch, bank, rng, device):
    """Same target color, different digit or place — geometry control, not hue.

    Prefer also matching ``source_color`` so the hinge cannot be solved by
    swapping palettes. Never fall back to a different target color.
    """
    bank = list(bank or batch)
    imgs = []
    for sample in batch:
        color = sample.get("target_color") or sample.get("color")
        src_color = sample.get("source_color")
        digit = str(sample.get("digit"))
        place = str(sample.get("source_place") or sample.get("placement") or "")
        prefer, cands = [], []
        for other in bank:
            od = str(other.get("digit"))
            op = str(other.get("source_place") or other.get("placement") or "")
            oc = other.get("target_color") or other.get("color")
            if oc != color or (od == digit and op == place):
                continue
            cands.append(other)
            if src_color is None or other.get("source_color") == src_color:
                prefer.append(other)
        pool = prefer or cands
        pick = pool[int(rng.integers(0, len(pool)))] if pool else sample
        image = pick["image"]
        if image.dim() == 3:
            image = image.unsqueeze(0)
        imgs.append(image)
    return torch.cat(imgs, 0).to(device=device)


def one_step(
    model, samples, device, batch_size, rng,
    shuffle_coef=1.0, edit_shuffle_coef=0.1, shuffle_margin=0.05,
):
    idx = rng.choice(len(samples), size=min(batch_size, len(samples)), replace=False)
    batch = [samples[int(i)] for i in idx]
    image, target, stroke, prompts = tensors_from(batch, device)
    need_pix = [True] * len(batch)
    zeros = torch.zeros(len(batch), device=device)
    is_edit = any(_is_edit_sample(s) for s in batch)
    img_pi = torch.ones(len(batch), device=device) if is_edit else torch.zeros(
        len(batch), device=device,
    )
    txt_pi = torch.ones(len(batch), device=device)
    out = model(
        image, prompts, need_pix=need_pix, t=zeros,
        image_precision=img_pi, text_precision=txt_pi,
    )
    fake = {
        "need_text": [False] * len(batch),
        "need_pix": need_pix,
        "need_seg": [False] * len(batch),
        "target_rgb": target.cpu(),
        "stroke": stroke.cpu(),
        "t": zeros,
        "target_image_precision": torch.ones(len(batch)),
        "target_text_precision": torch.zeros(len(batch)),
        "target_seg_precision": torch.zeros(len(batch)),
    }
    loss, meta = model.omni_loss(out, fake, device)
    meta["image_precision"] = float(img_pi[0])
    meta["text_precision"] = float(txt_pi[0])
    meta["is_edit"] = bool(is_edit)
    coef = float(edit_shuffle_coef if is_edit else shuffle_coef)
    if coef != 0.0:
        if is_edit:
            shuf_img = source_shuffle_images(batch, samples, rng, device)
            out_s = model(
                shuf_img, prompts, need_pix=need_pix, t=zeros,
                image_precision=img_pi, text_precision=txt_pi,
            )
            tag = "source_shuffle"
        else:
            shuffled = digit_shuffle_prompts(batch, prompts)
            out_s = model(
                image, shuffled, need_pix=need_pix, t=zeros,
                image_precision=img_pi, text_precision=txt_pi,
            )
            tag = "digit_shuffle"
        bce_m = balanced_observation_bce(
            out["rgb"], target, signed=False, reduction="none",
        )
        bce_s = balanced_observation_bce(
            out_s["rgb"], target, signed=False, reduction="none",
        )
        causal = torch.relu(bce_m - bce_s + float(shuffle_margin)).mean()
        loss = loss + coef * causal
        meta[tag] = float(causal.detach())
        meta["bce_matched"] = float(bce_m.mean().detach())
        meta["bce_shuffled"] = float(bce_s.mean().detach())
        meta["shuffle_coef_used"] = coef
    return loss, meta


def save_ckpt(model, path: Path, extra: dict):
    payload = {
        "state_dict": model.non_lm_state_dict(),
        "language": model.language_meta(),
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(GENERATION_CHAMPION_PATH))
    ap.add_argument(
        "--ckpt",
        default=str(ROOT / "checkpoints" / "omni_d64_pythia_named_edit_best.pt"),
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "results" / "published" / "pythia_named_edit.json"),
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lm-device", default=None)
    ap.add_argument("--language", default="pythia")
    ap.add_argument("--steps-language", type=int, default=400)
    ap.add_argument("--steps-joint", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--interface-lr", type=float, default=1e-3)
    ap.add_argument("--visual-lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--edit-ratio", type=float, default=0.0)
    ap.add_argument(
        "--edit-style",
        choices=("named", "next"),
        default="named",
        help="Named-color and next-color are different gates; never mix.",
    )
    ap.add_argument("--shuffle-coef", type=float, default=1.0)
    ap.add_argument(
        "--edit-shuffle-coef",
        type=float,
        default=0.1,
        help="Source-image shuffle hinge on named-edit batches. 0 disables.",
    )
    ap.add_argument("--shuffle-margin", type=float, default=0.05)
    ap.add_argument(
        "--opt-phase",
        default="auto",
        help="language | language_rgb | edit_spatial | edit_read | token_interface | auto",
    )
    ap.add_argument(
        "--load-language",
        action="store_true",
        help="Load text_in/out and language readers from --init (resume).",
    )
    ap.add_argument(
        "--decode-head",
        action="store_true",
        help="Also train pix_head while keeping stem/Deslice frozen.",
    )
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="Load --init, evaluate, write --out JSON. Do not train or reload --ckpt.",
    )
    ap.add_argument(
        "--modal-precision",
        action="store_true",
        help="Enable zero-init image/text precision coords (identity at load).",
    )
    args = ap.parse_args()

    protected = {
        ROOT / "checkpoints" / "v1_bayes_2000step_upd_rms_run1_best.pt",
        ROOT / "checkpoints" / "v1_bayes_2000step_f2_vfe_run1_best.pt",
        ROOT / "checkpoints" / "omni_d256_unified_fdesc_center_4k_best.pt",
        ROOT / "checkpoints" / "omni_d256_unified_t2i_s0band_best.pt",
        ROOT / "checkpoints" / "omni_d64_northstar_omni_active_f2_grid_best.pt",
        ROOT / "checkpoints" / "northstar_slice_capability_best.pt",
        ROOT / "checkpoints" / "omni_d64_pythia_language_best.pt",
        ROOT / "checkpoints" / "omni_d64_pythia_named_edit_spatial_best.pt",
    }
    ckpt_path = Path(args.ckpt)
    init_path = Path(args.init)
    if not args.eval_only and ckpt_path.resolve() in {p.resolve() for p in protected}:
        raise SystemExit(f"refusing to overwrite published checkpoint {ckpt_path}")

    device = torch.device(args.device)
    lm_device = args.lm_device or args.device
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    model = DualStreamOmni(
        **generation_champion_kwargs(
            language=args.language,
            lm_device=lm_device,
            use_modal_precision=bool(args.modal_precision),
        )
    ).to(device)
    if model.lm is not None:
        model.lm.to(device)
    init_path = Path(args.init)
    report = model.load_visual_champion(
        init_path, skip_language_interface=not args.load_language,
    )
    print(
        f"[pythia-gen] init={init_path} loaded={report['loaded']} "
        f"skipped={report['n_skipped']} language={model.lm_note} d_llm={model.d_llm}",
        flush=True,
    )

    t2i_bank = make_t2i_bank(model.res)
    eval_bank = make_eval_bank(model.res)
    edit_bank = (
        make_edit_bank(model.res, style=args.edit_style) if args.edit_ratio > 0 else []
    )
    edit_eval = (
        make_edit_eval(model.res, style=args.edit_style) if args.edit_ratio > 0 else []
    )
    print(
        f"  t2i_bank={len(t2i_bank)} eval={len(eval_bank)} "
        f"edit_bank={len(edit_bank)} device={device}",
        flush=True,
    )

    def t2i_score(metrics: dict) -> float:
        return (
            float(metrics.get("digit_top1", 0.0))
            + float(metrics.get("color_acc", 0.0))
            + float(metrics.get("paired_iou", 0.0))
            - float(metrics.get("flood", 0.0))
            + (
                float(metrics.get("digit_top1", 0.0))
                - float(metrics.get("digit_shuffled_top1", 0.0))
            )
        )

    history = []
    best_score = float("-inf")
    best_step = 0
    init_raw = torch.load(init_path, map_location="cpu")
    if isinstance(init_raw, dict) and isinstance(init_raw.get("t2i"), dict):
        best_score = t2i_score(init_raw["t2i"])
        print(
            f"  resume floor score={best_score:.3f} "
            f"top1={init_raw['t2i'].get('digit_top1')}",
            flush=True,
        )
    t0 = time.time()
    global_step = 0
    did_save = False
    n_edit_steps = 0
    n_t2i_steps = 0

    if args.eval_only:
        t2i = eval_t2i(model, eval_bank, device)
        edit = {"score": 0.0, "color_acc": 0.0, "n": 0}
        payload = {
            "schema": "pythia-eval-only",
            "init": str(init_path),
            "checkpoint": str(init_path),
            "eval_only": True,
            "modal_precision": bool(args.modal_precision),
            "language": model.language_meta(),
            "final": {"t2i": t2i, "edit": edit},
            "gates": {"t2i": t2i_gate(t2i), "edit": False},
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"final": payload["final"], "gates": payload["gates"]}, indent=2))
        print(f"eval-only from {init_path} -> {out}", flush=True)
        return

    def run_phase(phase: str, steps: int, interface_lr: float, visual_lr: float):
        nonlocal best_score, best_step, global_step, did_save, n_edit_steps, n_t2i_steps
        model.set_optimization_phase(phase)
        n_train = sum(p.numel() for p in trainable(model))
        opt = torch.optim.AdamW(
            param_groups(model, interface_lr=interface_lr, visual_lr=visual_lr),
            weight_decay=1e-4,
        )
        print(
            f"  phase={phase} steps={steps} n_train={n_train} "
            f"interface_lr={interface_lr} visual_lr={visual_lr}",
            flush=True,
        )
        for local in range(1, steps + 1):
            global_step += 1
            model.train()
            use_edit = bool(edit_bank) and rng.random() < float(args.edit_ratio)
            if use_edit:
                n_edit_steps += 1
            else:
                n_t2i_steps += 1
            bank = edit_bank if use_edit else t2i_bank
            opt.zero_grad(set_to_none=True)
            loss, meta = one_step(
                model, bank, device, args.batch, rng,
                shuffle_coef=args.shuffle_coef,
                edit_shuffle_coef=args.edit_shuffle_coef,
                shuffle_margin=args.shuffle_margin,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable(model), 1.0)
            opt.step()
            if local == 1 or local % args.eval_every == 0 or local == steps:
                t2i = eval_t2i(model, eval_bank, device)
                edit = eval_edit(model, edit_eval, device) if edit_eval else {
                    "score": 0.0, "color_acc": 0.0, "n": 0,
                }
                score = t2i_score(t2i) + float(edit.get("score") or 0.0)
                t2i_hold = t2i_gate(t2i)
                record = {
                    "phase": phase,
                    "step": global_step,
                    "loss": float(loss.detach()),
                    "pix": meta.get("pix", 0.0),
                    "source_shuffle": meta.get("source_shuffle"),
                    "digit_shuffle": meta.get("digit_shuffle"),
                    "image_precision": meta.get("image_precision"),
                    "text_precision": meta.get("text_precision"),
                    "t2i": t2i,
                    "edit": edit,
                    "t2i_gate": t2i_gate(t2i),
                    "edit_gate": edit_gate(edit),
                    "dt": time.time() - t0,
                }
                history.append(record)
                print(
                    f"  {phase} {local:4d}/{steps} loss={record['loss']:.3f} "
                    f"top1={t2i['digit_top1']:.3f} color={t2i['color_acc']:.3f} "
                    f"iou={t2i['paired_iou']:.3f} flood={t2i['flood']:.3f} "
                    f"shuff={t2i['digit_shuffled_top1']:.3f} "
                    f"cshuff={t2i.get('color_shuffled_acc', 0):.3f} "
                    f"edit={edit['score']:.3f} ecol={edit.get('color_acc', 0):.3f} "
                    f"edig={edit.get('digit_top1', 0):.3f} "
                    f"eshuf={edit.get('source_shuffled_digit_top1', 0):.3f} "
                    f"gates={int(record['t2i_gate'])}{int(record['edit_gate'])} "
                    f"dt={record['dt']:.0f}s",
                    flush=True,
                )
                if t2i_hold and score > best_score:
                    best_score = score
                    best_step = global_step
                    did_save = True
                    save_ckpt(model, ckpt_path, {
                        "step": global_step,
                        "phase": phase,
                        "t2i": t2i,
                        "edit": edit,
                        "init": report,
                    })

    if args.opt_phase != "auto":
        phase = args.opt_phase
    elif args.decode_head:
        phase = "language_rgb"
    elif args.edit_ratio > 0:
        phase = "edit_spatial"
    else:
        phase = "language"
    run_phase(phase, args.steps_language, args.interface_lr, args.visual_lr)
    if args.steps_joint > 0:
        run_phase("joint", args.steps_joint, args.interface_lr, args.visual_lr)

    if did_save:
        raw = torch.load(ckpt_path, map_location="cpu")
        current = model.state_dict()
        for k, v in raw["state_dict"].items():
            if k in current and current[k].shape == v.shape:
                current[k] = v
        model.load_state_dict(current)
    final_t2i = eval_t2i(model, eval_bank, device)
    final_edit = (
        eval_edit(model, edit_eval, device) if edit_eval
        else {"score": 0.0, "color_acc": 0.0, "n": 0}
    )
    payload = {
        "schema": "pythia-named-edit",
        "init": str(init_path),
        "checkpoint": str(ckpt_path) if did_save else None,
        "did_save": did_save,
        "run": {
            "edit_ratio": args.edit_ratio,
            "edit_style": args.edit_style,
            "edit_shuffle_coef": args.edit_shuffle_coef,
            "shuffle_coef": args.shuffle_coef,
            "shuffle_margin": args.shuffle_margin,
            "modal_precision": bool(args.modal_precision),
            "opt_phase": phase,
            "decode_head": bool(args.decode_head),
            "load_language": bool(args.load_language),
            "interface_lr": args.interface_lr,
            "visual_lr": args.visual_lr,
            "batch": args.batch,
            "steps_language": args.steps_language,
            "steps_joint": args.steps_joint,
            "eval_every": args.eval_every,
            "t2i_image_precision": 0.0,
            "edit_image_precision": 1.0,
            "text_precision": 1.0,
            "source_shuffle_trained": bool(args.edit_ratio > 0 and args.edit_shuffle_coef != 0.0),
            "n_edit_steps": n_edit_steps,
            "n_t2i_steps": n_t2i_steps,
        },
        "language": model.language_meta(),
        "best_step": best_step,
        "final": {"t2i": final_t2i, "edit": final_edit},
        "gates": {
            "t2i": t2i_gate(final_t2i),
            "edit": edit_gate(final_edit),
        },
        "history": history,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"final": payload["final"], "gates": payload["gates"], "did_save": did_save}, indent=2))
    if did_save:
        print(f"saved {out} and {ckpt_path}", flush=True)
    else:
        print(f"wrote {out}; no checkpoint saved (T2I official gate not held)", flush=True)


if __name__ == "__main__":
    main()
