#!/usr/bin/env python3
"""Deterministic capacity gate for native Slice text-to-image generation.

This is not a second generator.  It trains ``DualStreamOmni`` on one fixed
full-resolution point field trajectory:

    black X0 -> SliceRead -> MoT(S, H) -> Deslice -> X -> RGB

The deliberately plain objective answers one question before VFE/active-
inference mechanisms are restored: can the native graph express ten centered
digit observations and make their identity prompt-causal?
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

from fine_grain.gen_metrics import background_flood_rate, gen_free_scores
from fine_grain.omni_model import DualStreamOmni, balanced_observation_bce
from fine_grain.omni_tasks import one_sample


def make_bank(
    res: int, digits: list[int], colors: list[str], device: torch.device,
):
    samples = []
    for color_idx, color in enumerate(colors):
        for digit in digits:
            sample = one_sample(
                np.random.default_rng(1000 + 100 * color_idx + digit),
                res,
                "t2i",
                t2i_canvas="black",
                t2i_place="center",
                t2i_digit=digit,
                t2i_color=color,
            )
            samples.append(sample)
    source = torch.cat([s["image"] for s in samples]).to(device)
    target = torch.cat([s["target_rgb"] for s in samples]).to(device)
    stroke = torch.cat([s["stroke"] for s in samples]).to(device)
    prompts = [s["prompt"] for s in samples]
    return samples, source, target, stroke, prompts


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = (pred - target).pow(2).mean().clamp_min(1e-12)
    return float(-10.0 * torch.log10(mse))


@torch.no_grad()
def evaluate(
    model: DualStreamOmni,
    samples,
    source: torch.Tensor,
    target: torch.Tensor,
    stroke: torch.Tensor,
    prompts: list[str],
):
    model.eval()
    matched = model(source, prompts, need_pix=[True] * len(prompts))["rgb"]
    assignment = []
    for i, layer in enumerate(model.mot_stack.layers):
        w = layer.last_w
        assignment.append({
            "layer": i,
            "entropy_norm": float(
                (-(w.clamp_min(1e-8) * w.clamp_min(1e-8).log()).sum(-1).mean())
                / np.log(max(2, w.shape[-1]))
            ),
            "mass_cv": float(
                w.sum(1).std(dim=-1).mean() / w.sum(1).mean(dim=-1).mean().clamp_min(1e-8)
            ),
        })
    prompt_for = {(s["digit"], s["color"]): s["prompt"] for s in samples}
    colors = list(dict.fromkeys(s["color"] for s in samples))
    digit_shuffled_prompts = [
        prompt_for.get((str((int(s["digit"]) + 1) % 10), s["color"]), "")
        for s in samples
    ]
    color_shuffled_prompts = [
        prompt_for.get(
            (s["digit"], colors[(colors.index(s["color"]) + 1) % len(colors)]),
            "",
        )
        if len(colors) > 1 else ""
        for s in samples
    ]
    digit_shuffled = model(
        source, digit_shuffled_prompts, need_pix=[True] * len(prompts),
    )["rgb"]
    color_shuffled = model(
        source, color_shuffled_prompts, need_pix=[True] * len(prompts),
    )["rgb"]

    matched_scores = [
        gen_free_scores(matched[i : i + 1], samples[i]["digit"], samples[i]["color"])
        for i in range(len(samples))
    ]
    digit_shuffled_scores = [
        gen_free_scores(
            digit_shuffled[i : i + 1], samples[i]["digit"], samples[i]["color"],
        )
        for i in range(len(samples))
    ]
    color_shuffled_scores = [
        gen_free_scores(
            color_shuffled[i : i + 1], samples[i]["digit"], samples[i]["color"],
        )
        for i in range(len(samples))
    ]

    def mean(key: str, rows) -> float:
        return float(sum(float(r[key]) for r in rows) / len(rows))

    return {
        "loss": float(balanced_observation_bce(matched, target)),
        "psnr": psnr(matched, target),
        "digit_top1": mean("digit_top1", matched_scores),
        "digit_iou": mean("digit_iou", matched_scores),
        "ink_frac": mean("ink_frac", matched_scores),
        "color_acc": mean("color_acc", matched_scores),
        "flood": background_flood_rate(matched, target, stroke),
        "digit_shuffled_top1": mean("digit_top1", digit_shuffled_scores),
        "color_shuffled_acc": mean("color_acc", color_shuffled_scores),
        "matched_vs_digit_shuffle_rms": float(
            (matched - digit_shuffled).pow(2).mean().sqrt()
        ),
        "matched_vs_color_shuffle_rms": float(
            (matched - color_shuffled).pow(2).mean().sqrt()
        ),
        "assignment": assignment,
        "pred": matched.detach().cpu(),
    }


def render_gallery(path: Path, target: torch.Tensor, pred: torch.Tensor, labels: list[str]):
    import matplotlib.pyplot as plt

    n = len(labels)
    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 3.2), squeeze=False)
    for i, label in enumerate(labels):
        axes[0, i].imshow(target[i].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[0, i].set_title(f"target {label}")
        axes[1, i].imshow(pred[i].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[1, i].set_title(f"generated {label}")
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--res", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-slices", type=int, default=16)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--digit", type=int, default=None)
    ap.add_argument(
        "--color", default="green",
        help="red/green/blue/yellow, or 'all' for the four-color gate",
    )
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="northstar_digit_capacity")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    digits = [args.digit] if args.digit is not None else list(range(10))
    colors = (
        ["red", "green", "blue", "yellow"]
        if args.color.lower() == "all" else [args.color]
    )
    samples, source, target, stroke, prompts = make_bank(
        args.res, digits, colors, device,
    )
    model = DualStreamOmni(
        d_model=args.d_model,
        n_slices=args.n_slices,
        n_layers=args.n_layers,
        res=args.res,
        n_heads=args.n_heads,
        surprise_mode="baseline",
        s_update="raw",
        prior_loss_coef=0.0,
        sigreg_coef=0.0,
        use_stiefel=False,
        deslice_topk=0,
        deslice_write="increment",
        gate_h_local=False,
        vfe_coef=0.0,
        use_null_slice=False,
        use_residual_read=False,
        s0_acc_coef=0.0,
        fm_signed=False,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best = None
    best_state = None
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(source, prompts, need_pix=[True] * len(prompts))["rgb"]
        loss = balanced_observation_bce(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, samples, source, target, stroke, prompts)
            record = {k: v for k, v in metrics.items() if k != "pred"}
            record["step"] = step
            history.append(record)
            score = metrics["digit_top1"] + metrics["digit_iou"] - metrics["flood"]
            if best is None or score > best:
                best = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"step={step:4d} loss={metrics['loss']:.4f} psnr={metrics['psnr']:.2f} "
                f"top1={metrics['digit_top1']:.2f} dshuffle={metrics['digit_shuffled_top1']:.2f} "
                f"color={metrics['color_acc']:.2f} cshuffle={metrics['color_shuffled_acc']:.2f} "
                f"iou={metrics['digit_iou']:.3f} flood={metrics['flood']:.3f} "
                f"prompt_rms={metrics['matched_vs_digit_shuffle_rms']:.4f}",
                flush=True,
            )
            if (
                len(digits) == 10
                and metrics["digit_top1"] >= 0.95
                and metrics["color_acc"] >= 0.95
                and metrics["flood"] <= 0.10
                and metrics["digit_top1"] - metrics["digit_shuffled_top1"] >= 0.50
                and (
                    len(colors) == 1
                    or metrics["color_acc"] - metrics["color_shuffled_acc"] >= 0.50
                )
            ):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    final = evaluate(model, samples, source, target, stroke, prompts)
    checkpoint = ROOT / "checkpoints" / f"{args.tag}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    gallery = ROOT / "present" / "figs" / f"{args.tag}.png"
    labels = [f"{s['digit']}-{s['color'][0]}" for s in samples]
    render_gallery(gallery, target.detach().cpu(), final.pop("pred"), labels)
    report = {
        "task": "native Slice fixed-center digit capacity",
        "config": vars(args),
        "digits": digits,
        "colors": colors,
        "graph": "black X0 -> SliceRead -> MoT(S,H) -> Deslice -> X -> RGB",
        "mechanisms_disabled": [
            "shared recurrence", "VFE gate", "Stiefel", "null slice",
            "residual read", "external generator",
        ],
        "history": history,
        "final": final,
        "checkpoint": str(checkpoint),
        "gallery": str(gallery),
        "review": "not independently reviewed",
    }
    output = ROOT / "results" / "published" / f"{args.tag}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2), flush=True)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
