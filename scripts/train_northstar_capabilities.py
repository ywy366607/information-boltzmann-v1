"""Train and audit one native Slice checkpoint across the North-Star matrix."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.capability_tasks import (
    CAPABILITY_CASES,
    WORLD_ACTIONS_WITH_NOOP,
    collate_capability,
    fixed_capability_bank,
    make_capability_batch,
    make_counterfactual_future_batch,
)
from scripts.train_pythia_capabilities import (
    digit_residual_likelihood,
    make_static_bank,
    paired_digit_likelihood,
    sample_digit_group,
)
from fine_grain.gen_metrics import (
    free_color_acc,
    ink_centroid_error,
    paired_ink_iou,
    digit_shift_scores,
)
from fine_grain.omni_model import DualStreamOmni


DEFAULT_INIT = ROOT / "checkpoints" / "omni_d64_northstar_omni_active_f2_grid_best.pt"


def model_config(args) -> Dict:
    return {
        "d_model": args.d_model,
        "n_slices": args.n_slices,
        "n_layers": 4,
        "n_heads": args.n_heads,
        "res": args.res,
        "surprise_mode": "v1_bayes",
        "s_update": "raw",
        "prior_loss_coef": 0.1,
        "sigreg_coef": 0.0,
        "use_stiefel": False,
        "deslice_topk": 0,
        "use_null_slice": False,
        "use_residual_read": False,
        "gate_on": "u",
        "deslice_write": "increment",
        "gate_h_local": False,
        "vfe_coef": 0.1,
        "prior_write": 1.0,
        "prior_write_by_t": False,
        "pixel_loss_mode": "balanced_bce",
        "spatial_prompt_vocab": True,
        "capability_vocab": True,
        "use_modal_precision": True,
        "use_target_time": True,
        "use_target_time_adaln": bool(args.target_time_adaln),
        "use_horizon_tokens": False,
        "gate_action_by_horizon": True,
        "history_size": 2,
        "action_dim": 2,
        # Global AdaLN action/goal paths failed the causal audit. The action is
        # instead a masked MoT token that interacts with spatial Slice tokens.
        "use_action_adaln": False,
        "use_action_tokens": bool(args.action_tokens),
        "use_task_tokens": bool(args.task_tokens),
        "n_task_tokens": len(CAPABILITY_CASES),
        "control_prefix_attention": bool(args.control_prefix_attention),
        "use_attention_sink": bool(args.attention_sink),
        "head_balance_coef": float(args.head_balance_coef),
        "gaussian_head_layout": args.gaussian_head_layout,
        "use_action_rel_bias": bool(args.action_rel_bias),
        "use_action_transport": bool(args.action_transport),
        "use_action_slice_transition": bool(args.action_slice_transition),
        "use_active_gdn2": bool(args.active_gdn2),
        "use_active_gdn2_history_transport": bool(
            args.active_gdn2_history_transport
        ),
        "active_gdn2_initial_trust": float(args.active_gdn2_initial_trust),
        "use_goal_adaln": False,
        "seg_classes": 2,
        "seg_loss_coef": 1.0,
        "deep_visual_likelihood_coef": args.deep_visual_likelihood_coef,
        "transition_loss_coef": args.transition_loss_coef,
        "transition_posterior_loss_coef": args.transition_posterior_loss_coef,
        "transition_detach_q": True,
        "causal_memory_loss_coef": args.causal_memory_loss_coef,
        "s0_acc_coef": 0.0,
    }


def load_compatible(model: torch.nn.Module, path: Path) -> Dict:
    """Load the proven generator while admitting only explicit new interfaces."""
    raw = torch.load(path, map_location="cpu")
    source_layout = raw.get("config", {}).get("gaussian_head_layout", "legacy") if isinstance(raw, dict) else "legacy"
    target_layout = getattr(model.mot_stack, "gaussian_head_layout", "legacy")
    if source_layout != target_layout:
        raise ValueError(
            f"Gaussian layout mismatch: checkpoint={source_layout}, model={target_layout}. "
            "Use the checkpoint's declared layout or train the corrected layout from scratch."
        )
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    current = model.state_dict()
    loaded, partial, skipped = [], [], []
    for key, value in state.items():
        if key not in current:
            skipped.append(key)
            continue
        if current[key].shape == value.shape:
            current[key] = value
            loaded.append(key)
        elif key == "embed.weight" and current[key].ndim == value.ndim == 2:
            rows = min(current[key].shape[0], value.shape[0])
            cols = min(current[key].shape[1], value.shape[1])
            current[key][:rows, :cols] = value[:rows, :cols]
            partial.append(f"{key}:{rows}x{cols}")
        else:
            skipped.append(key)
    model.load_state_dict(current)
    return {"loaded": len(loaded), "partial": partial, "skipped": skipped}


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = float((pred - target).pow(2).mean())
    return float(-10.0 * math.log10(max(mse, 1e-8)))


def _seg_iou(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=0) > 0
    target = target.to(device=pred.device) > 0
    union = (pred | target).sum().clamp_min(1)
    return float((pred & target).sum() / union)


def _checkpoint_score(
    result: Dict,
    action_ablated: Dict | None = None,
    action_min_gain: float = 0.0,
    transition_ablated: Dict | None = None,
) -> tuple[float, float, bool]:
    """Select dynamics only after the already-proven abilities clear floors."""
    floors = {
        "text_to_both": 0.85,
        "image_to_current": 0.75,
        "image_text_edit": 0.80,
    }
    margins = [result[case]["score"] - floor for case, floor in floors.items()]
    future = result["image_to_future"]
    future_primary = 0.5 * (future["paired_iou"] + future["seg_iou"])
    action_margins = []
    if action_ablated is not None:
        # Do not let an average hide a harmful action path: both transported
        # position and dense mask must improve over action_precision=0.
        action_margins = [
            future["paired_iou"] - action_ablated["paired_iou"],
            future["seg_iou"] - action_ablated["seg_iou"],
        ]
    admitted = min(margins) >= 0.0
    if action_margins:
        admitted = admitted and min(action_margins) > float(action_min_gain) + 1e-12
    transition_margins = []
    if transition_ablated is not None:
        transition_margins = [
            future["paired_iou"] - transition_ablated["paired_iou"],
            future["seg_iou"] - transition_ablated["seg_iou"],
        ]
        admitted = admitted and min(transition_margins) > 0.0
    # Failed candidates sort below every admitted candidate.  Once admitted,
    # selection is entirely about the quantities that must move in the future.
    rejection_margins = margins + action_margins + transition_margins
    score = future_primary if admitted else -1.0 + min(rejection_margins)
    return float(score), float(future_primary), admitted


def paired_action_likelihood(
    out: Dict[str, torch.Tensor],
    batch: Dict,
    temperature: float = 0.1,
    seg_weight: float = 0.1,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Contrast matched interventions through the existing observation model.

    Rows are predictions conditioned on actions and columns are targets under
    paired counterfactual actions for the identical state and history. This is
    an observation-energy term, not a task or action classifier.
    """
    if "counterfactual_group" not in batch:
        zero = out["rgb"].sum() * 0.0
        return zero, {
            "action_contrast": 0.0,
            "action_energy_diagonal": 0.0,
            "action_energy_offdiagonal": 0.0,
            "action_energy_gap": 0.0,
        }
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    device = out["rgb"].device
    groups = batch["counterfactual_group"].to(device)
    action_index = batch["counterfactual_index"].to(device)
    target_rgb = batch["target_rgb"].to(device)
    target_seg = batch["target_seg"].to(device)
    losses, diagonals, offdiagonals = [], [], []
    for group_id in torch.unique(groups, sorted=True):
        indices = torch.nonzero(groups == group_id, as_tuple=False).flatten()
        order = torch.argsort(action_index.index_select(0, indices))
        indices = indices.index_select(0, order)
        if indices.numel() < 2:
            raise ValueError("counterfactual groups need at least two actions")
        pred_rgb = out["rgb"].index_select(0, indices).clamp(0.0, 1.0)
        gold_rgb = target_rgb.index_select(0, indices)
        rgb_energy = (pred_rgb[:, None] - gold_rgb[None, :]).square().mean(
            dim=(2, 3, 4),
        )

        pred_seg = out["seg_logits"].index_select(0, indices).log_softmax(dim=1)
        gold_seg = target_seg.index_select(0, indices)
        one_hot = F.one_hot(gold_seg, num_classes=pred_seg.shape[1]).permute(
            0, 3, 1, 2,
        ).to(dtype=pred_seg.dtype)
        seg_energy = -(pred_seg[:, None] * one_hot[None, :]).sum(dim=2).mean(
            dim=(2, 3),
        )
        energy = rgb_energy + float(seg_weight) * seg_energy
        labels = torch.arange(indices.numel(), device=device)
        logits = -energy / float(temperature)
        losses.append(0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)))
        mask = ~torch.eye(indices.numel(), dtype=torch.bool, device=device)
        diagonals.append(energy.diagonal().mean())
        offdiagonals.append(energy.masked_select(mask).mean())
    loss = torch.stack(losses).mean()
    diagonal = torch.stack(diagonals).mean()
    offdiagonal = torch.stack(offdiagonals).mean()
    return loss, {
        "action_contrast": float(loss.detach()),
        "action_energy_diagonal": float(diagonal.detach()),
        "action_energy_offdiagonal": float(offdiagonal.detach()),
        "action_energy_gap": float((offdiagonal - diagonal).detach()),
    }


def _case_score(rec: Dict) -> float:
    keys = ("text_acc", "seg_iou", "color_acc", "digit_top1", "paired_iou")
    return float(sum(float(rec[k]) for k in keys) / len(keys))


def _language_boundary(
    batch: Dict,
    device: torch.device,
    future_language_evidence: bool,
):
    """Return semantic text evidence, excluding vacuous future task labels."""
    prompts = list(batch["prompt"])
    text_pi = batch["text_precision"].to(device).clone()
    if not future_language_evidence:
        for index, kind in enumerate(batch["kind"]):
            if kind == "image_to_future":
                prompts[index] = ""
                text_pi[index] = 0.0
    return prompts, text_pi


@torch.no_grad()
def evaluate_samples(
    model: DualStreamOmni,
    samples: List[Dict],
    device: torch.device,
    chunk: int = 24,
    ablate: str = "",
    future_language_evidence: bool = True,
) -> Dict:
    model.eval()
    totals = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    for start in range(0, len(samples), int(chunk)):
        part = samples[start : start + int(chunk)]
        batch = collate_capability(part)
        image_pi = batch["image_precision"].to(device)
        prompts, text_pi = _language_boundary(
            batch, device, future_language_evidence,
        )
        horizon = batch["target_time"].to(device)
        history_pi = batch["history_precision"].to(device)
        action_pi = batch["action_precision"].to(device)
        task_id = batch["task_id"].to(device)
        if ablate == "image":
            image_pi.zero_()
        elif ablate == "text":
            text_pi.zero_()
            prompts = [""] * len(part)
        elif ablate == "goal":
            if bool(getattr(model.mot_stack, "use_task_tokens", False)):
                task_id.fill_(-1)
            else:
                text_pi.zero_()
                prompts = [""] * len(part)
        elif ablate == "horizon":
            horizon.zero_()
        elif ablate == "history":
            history_pi.zero_()
        elif ablate == "action":
            action_pi.zero_()
        if ablate == "slice_transition":
            for layer in model.mot_stack.layers:
                layer.disable_action_transition = True
        if ablate == "causal_memory":
            model.mot_stack.disable_active_gdn2 = True
        out = model(
            batch["image"].to(device),
            prompts,
            t=torch.zeros(len(part), device=device),
            image_precision=image_pi,
            text_precision=text_pi,
            target_time=horizon,
            history_images=batch["history_images"].to(device),
            history_precision=history_pi,
            action=batch["action"].to(device),
            action_precision=action_pi,
            task_id=task_id,
        )
        if ablate == "slice_transition":
            for layer in model.mot_stack.layers:
                layer.disable_action_transition = False
        if ablate == "causal_memory":
            model.mot_stack.disable_active_gdn2 = False
        rgb = out["rgb"].clamp(0.0, 1.0)
        pred_ans = out["logits"].argmax(dim=-1)
        for i, sample in enumerate(part):
            case = sample["case"]
            rec = totals[case]
            counts[case] += 1
            gold = model.ans_to_idx[sample["answer"]]
            rec["text_acc"] += float(int(pred_ans[i]) == gold)
            target = sample["target_rgb"].to(device)
            stroke = sample["stroke"].to(device)
            color = sample["target_color"]
            digit = sample["digit"]
            pred_i = rgb[i : i + 1]
            rec["psnr"] += _psnr(pred_i, target)
            rec["color_acc"] += free_color_acc(pred_i, color)
            rec["paired_iou"] += paired_ink_iou(pred_i, stroke, color)
            rec["centroid_error"] += ink_centroid_error(pred_i, stroke, color)
            rec["digit_top1"] += digit_shift_scores(pred_i, digit, color)["digit_top1"]
            rec["seg_iou"] += _seg_iou(out["seg_logits"][i], sample["target_seg"])
    result = {}
    for case, rec in totals.items():
        n = max(1, counts[case])
        result[case] = {key: float(value / n) for key, value in rec.items()}
        result[case]["n"] = n
        result[case]["score"] = _case_score(result[case])
    result["macro_score"] = float(
        sum(result[c]["score"] for c in counts) / max(1, len(counts))
    )
    return result


def audit_causal_boundaries(
    model, bank, device, chunk, future_language_evidence=True,
) -> Dict:
    interventions = {
        "text_to_both": "text",
        "image_to_current": "image",
        "image_text_edit": "text",
        "image_to_future": "horizon",
    }
    report = {}
    for case, intervention in interventions.items():
        samples = [s for s in bank if s["case"] == case]
        intact = evaluate_samples(
            model, samples, device, chunk=chunk,
            future_language_evidence=future_language_evidence,
        )[case]
        ablated = evaluate_samples(
            model, samples, device, chunk=chunk, ablate=intervention,
            future_language_evidence=future_language_evidence,
        )[case]
        report[case] = {
            "intervention": intervention,
            "intact": intact,
            "ablated": ablated,
            "score_drop": intact["score"] - ablated["score"],
        }
        if case == "image_to_future":
            # Horizon changes the future address, while digit/color are
            # invariants of this toy transition. Audit the affected quantities
            # instead of hiding transport gains in an all-attribute average.
            intact_primary = 0.5 * (intact["paired_iou"] + intact["seg_iou"])
            ablated_primary = 0.5 * (ablated["paired_iou"] + ablated["seg_iou"])
            report[case].update({
                "primary_metric": "mean(target_paired_iou,target_seg_iou)",
                "intact_primary": intact_primary,
                "ablated_primary": ablated_primary,
                "primary_drop": intact_primary - ablated_primary,
            })
        else:
            report[case].update({
                "primary_metric": "case_score",
                "intact_primary": intact["score"],
                "ablated_primary": ablated["score"],
                "primary_drop": intact["score"] - ablated["score"],
            })
    future_samples = [s for s in bank if s["case"] == "image_to_future"]
    intact = report["image_to_future"]["intact"]
    intact_primary = 0.5 * (intact["paired_iou"] + intact["seg_iou"])
    interventions = ["history", "action"]
    if bool(getattr(model.mot_stack, "use_action_slice_transition", False)):
        interventions.append("slice_transition")
    if bool(getattr(model.mot_stack, "use_active_gdn2", False)):
        interventions.append("causal_memory")
    for intervention in interventions:
        ablated = evaluate_samples(
            model, future_samples, device, chunk=chunk, ablate=intervention,
            future_language_evidence=future_language_evidence,
        )["image_to_future"]
        ablated_primary = 0.5 * (
            ablated["paired_iou"] + ablated["seg_iou"]
        )
        report[f"image_to_future_{intervention}"] = {
            "intervention": intervention,
            "intact": intact,
            "ablated": ablated,
            "primary_metric": "mean(target_paired_iou,target_seg_iou)",
            "intact_primary": intact_primary,
            "ablated_primary": ablated_primary,
            "primary_drop": intact_primary - ablated_primary,
            "paired_iou_drop": intact["paired_iou"] - ablated["paired_iou"],
            "seg_iou_drop": intact["seg_iou"] - ablated["seg_iou"],
        }
    for case in ("image_to_current", "image_to_future"):
        samples = [s for s in bank if s["case"] == case]
        intact = report[case]["intact"]
        ablated = evaluate_samples(
            model, samples, device, chunk=chunk, ablate="goal",
            future_language_evidence=future_language_evidence,
        )[case]
        if case == "image_to_future":
            intact_primary = 0.5 * (intact["paired_iou"] + intact["seg_iou"])
            ablated_primary = 0.5 * (
                ablated["paired_iou"] + ablated["seg_iou"]
            )
            metric = "mean(target_paired_iou,target_seg_iou)"
        else:
            intact_primary = intact["score"]
            ablated_primary = ablated["score"]
            metric = "case_score"
        report[f"{case}_goal"] = {
            "intervention": (
                "task_token" if bool(getattr(model.mot_stack, "use_task_tokens", False))
                else "language_goal"
            ),
            "intact": intact,
            "ablated": ablated,
            "primary_metric": metric,
            "intact_primary": intact_primary,
            "ablated_primary": ablated_primary,
            "primary_drop": intact_primary - ablated_primary,
        }
    if bool(getattr(model.mot_stack, "use_task_tokens", False)):
        # Report the semantic mode separately from lexical content for every
        # port. Generation/editing still need words after the mode is known.
        for case in CAPABILITY_CASES:
            if case in ("image_to_current", "image_to_future"):
                report[f"{case}_mode"] = dict(report[f"{case}_goal"])
                continue
            samples = [s for s in bank if s["case"] == case]
            intact = report[case]["intact"]
            ablated = evaluate_samples(
                model, samples, device, chunk=chunk, ablate="goal",
                future_language_evidence=future_language_evidence,
            )[case]
            report[f"{case}_mode"] = {
                "intervention": "task_token",
                "intact": intact,
                "ablated": ablated,
                "primary_metric": "case_score",
                "intact_primary": intact["score"],
                "ablated_primary": ablated["score"],
                "primary_drop": intact["score"] - ablated["score"],
            }
    return report


@torch.no_grad()
def render_gallery(
    model, bank, device, path: Path, future_language_evidence=True,
) -> None:
    model.eval()
    samples = []
    for case in CAPABILITY_CASES:
        samples.append(next(s for s in bank if s["case"] == case and s["digit"] == "7"))
    batch = collate_capability(samples)
    prompts, text_pi = _language_boundary(
        batch, device, future_language_evidence,
    )
    out = model(
        batch["image"].to(device),
        prompts,
        t=torch.zeros(len(samples), device=device),
        image_precision=batch["image_precision"].to(device),
        text_precision=text_pi,
        target_time=batch["target_time"].to(device),
        history_images=batch["history_images"].to(device),
        history_precision=batch["history_precision"].to(device),
        action=batch["action"].to(device),
        action_precision=batch["action_precision"].to(device),
        task_id=batch["task_id"].to(device),
    )
    pred = out["rgb"].clamp(0, 1).cpu()
    seg = out["seg_logits"].argmax(dim=1).float().cpu()
    fig, axes = plt.subplots(4, 4, figsize=(9, 9), dpi=150)
    titles = ("boundary input", "target field", "predicted field", "predicted mask")
    for row, sample in enumerate(samples):
        images = (
            sample["image"][0].permute(1, 2, 0),
            sample["target_rgb"][0].permute(1, 2, 0),
            pred[row].permute(1, 2, 0),
            seg[row],
        )
        for col, image in enumerate(images):
            axes[row, col].imshow(image, cmap="gray" if col == 3 else None, vmin=0, vmax=1)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(titles[col])
        axes[row, 0].set_ylabel(sample["case"], fontsize=9)
    fig.suptitle("One Slice-MoT checkpoint · all North-Star boundaries", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--res", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-slices", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--control-prefix-attention", action="store_true",
                        help="Make observed control tokens visible to causal text without answer leakage.")
    parser.add_argument("--attention-sink", action="store_true",
                        help="Independent zero-value attention sink per head, separate from task tokens.")
    parser.add_argument("--head-balance-coef", type=float, default=0.0)
    parser.add_argument("--gaussian-head-layout", choices=("legacy", "per_head"), default="legacy",
                        help="Versioned Gaussian parameter packing; per_head repairs mean/variance head separation.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Shared graph learning rate.")
    parser.add_argument(
        "--head-lr", type=float, default=1e-3,
        help="New likelihood heads and boundary-coordinate learning rate.",
    )
    parser.add_argument(
        "--time-lr", type=float, default=5e-4,
        help="Temporal coordinate, embedding, and zero-AdaLN learning rate.",
    )
    parser.add_argument(
        "--lr-schedule", choices=("constant", "cosine"), default="constant",
        help="Learning-rate schedule; cosine scales every parameter group proportionally.",
    )
    parser.add_argument(
        "--min-lr-ratio", type=float, default=0.05,
        help="Final/base learning-rate ratio for --lr-schedule cosine.",
    )
    parser.add_argument(
        "--static-only", action="store_true",
        help="Train/select generation, current reconstruction/segmentation, and editing only.",
    )
    parser.add_argument(
        "--static-selection",
        choices=("capability_floor", "generation", "macro"),
        default="capability_floor",
        help=(
            "Checkpoint rule for --static-only. 'generation' is for an explicit "
            "T2I refinement phase and selects text_to_both score directly."
        ),
    )
    parser.add_argument(
        "--generation-warmup-steps", type=int, default=0,
        help="From-scratch curriculum: train text_to_both alone for the first N steps.",
    )
    parser.add_argument(
        "--optimization-phase",
        choices=(
            "joint", "language", "language_rgb", "generation_write",
            "generation_capacity",
        ),
        default="joint",
        help="Optional same-graph refinement phase after loading a target-resolution checkpoint.",
    )
    parser.add_argument("--digit-contrast-coef", type=float, default=0.0)
    parser.add_argument(
        "--digit-residual-coef", type=float, default=0.0,
        help=(
            "Weight of the group-centred RGB/seg observation likelihood. "
            "This gives prompt-specific glyph residuals a signed target."
        ),
    )
    parser.add_argument("--digit-contrast-temperature", type=float, default=0.1)
    parser.add_argument(
        "--digit-contrast-difference-only", action="store_true",
        help="Score only target-template symmetric differences in the ten-digit likelihood.",
    )
    parser.add_argument(
        "--digit-group-every", type=int, default=4,
        help="Once enabled, use one complete ten-digit group every N steps.",
    )
    parser.add_argument(
        "--digit-group-start", type=int, default=1,
        help="First optimization step eligible for a grouped digit likelihood.",
    )
    parser.add_argument(
        "--deep-visual-likelihood-coef", type=float, default=0.0,
        help=(
            "Apply the shared RGB observation likelihood to intermediate layer "
            "fields so later inference updates cannot erase earlier structure."
        ),
    )
    parser.add_argument(
        "--target-time-adaln",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Inject the prediction horizon through per-layer zero-AdaLN.",
    )
    parser.add_argument(
        "--task-tokens",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append one learned semantic-intention token (generation, current, "
            "edit, or future) to the shared H workspace."
        ),
    )
    parser.add_argument(
        "--temporal-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze the proven static trunk/readouts and train only transition interfaces.",
    )
    parser.add_argument(
        "--causal-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze every parameter except the active GDN-2 causal memory.",
    )
    parser.add_argument(
        "--causal-projection-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Freeze the graph and GDN-2 belief; train only its prior residual "
            "projection to test whether the stored atlas is already useful."
        ),
    )
    parser.add_argument(
        "--pcgrad-static",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Project future gradients that conflict with static capability gradients.",
    )
    parser.add_argument(
        "--action-contrast-coef",
        type=float,
        default=0.0,
        help="Weight of paired counterfactual RGB+segmentation observation energy.",
    )
    parser.add_argument(
        "--action-transport",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Learn action-conditioned Deslice write assignments on fixed X addresses.",
    )
    parser.add_argument(
        "--action-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Expose action as an MoT control token. Disable with active GDN-2 "
            "to test one authoritative physical action path."
        ),
    )
    parser.add_argument(
        "--action-rel-bias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Bias transient Slice assignment by action. Disable with action "
            "tokens when testing GDN-2 as the sole physical action path."
        ),
    )
    parser.add_argument(
        "--action-slice-transition",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply one normalized action-conditioned Slice belief transition.",
    )
    parser.add_argument(
        "--active-gdn2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use persistent active-inference GDN-2 Slice memory.",
    )
    parser.add_argument(
        "--active-gdn2-history-transport",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Transport stored history before correction using inferred velocity; "
            "kept off because the v21 correspondence ablation regressed."
        ),
    )
    parser.add_argument(
        "--active-gdn2-initial-trust",
        type=float,
        default=0.0,
        help=(
            "Initial signed trust in the closed-coordinate dynamic prior. "
            "Use 0 for checkpoint identity; the audited physical warm start is 0.1."
        ),
    )
    parser.add_argument(
        "--future-language-evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Treat the literal 'Predict next frame' label as observed language. "
            "Disable to let history, horizon, and action define world prediction."
        ),
    )
    parser.add_argument(
        "--action-contrast-temp",
        type=float,
        default=0.1,
        help="Temperature for paired counterfactual observation energies.",
    )
    parser.add_argument(
        "--action-min-gain",
        type=float,
        default=0.0,
        help="Required intact-minus-zero-action gain for both dense future metrics.",
    )
    parser.add_argument(
        "--case-repeats",
        default="2,1,1,4",
        help=(
            "Nonnegative sampling repeats for text_to_both,image_to_current,"
            "image_text_edit,image_to_future; at least one must be positive."
        ),
    )
    parser.add_argument(
        "--transition-loss-coef",
        type=float,
        default=0.5,
        help="Same-graph heteroscedastic KL(q_future || p_future).",
    )
    parser.add_argument(
        "--transition-posterior-loss-coef",
        type=float,
        default=1.0,
        help="Observation likelihood anchoring q(s_future | o_future).",
    )
    parser.add_argument(
        "--causal-memory-loss-coef",
        type=float,
        default=0.25,
        help="Slice-chart KL(q_future || p_future) for active GDN-2.",
    )
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-per-case", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--init", default=str(DEFAULT_INIT))
    parser.add_argument(
        "--out",
        default=str(ROOT / "results" / "published" / "northstar_capability_closure.json"),
    )
    parser.add_argument(
        "--ckpt",
        default=str(ROOT / "checkpoints" / "northstar_slice_capability_best.pt"),
    )
    parser.add_argument(
        "--gallery",
        default=str(ROOT / "present" / "figs" / "northstar_capability_closure.png"),
    )
    args = parser.parse_args()
    if args.action_contrast_coef > 0.0 and not args.pcgrad_static:
        raise ValueError("paired action likelihood currently requires --pcgrad-static")
    if args.action_transport and args.action_slice_transition:
        raise ValueError("assignment transport and Slice transition are competing ablations")
    if args.active_gdn2 and args.action_slice_transition:
        raise ValueError("active GDN-2 replaces transient Slice transition")
    if args.causal_only and not args.active_gdn2:
        raise ValueError("--causal-only requires --active-gdn2")
    if args.causal_projection_only and not args.active_gdn2:
        raise ValueError("--causal-projection-only requires --active-gdn2")
    if args.causal_only and args.causal_projection_only:
        raise ValueError("causal-only and causal-projection-only are competing ablations")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must lie in [0, 1]")
    if not 0 <= args.generation_warmup_steps <= args.steps:
        raise ValueError("--generation-warmup-steps must lie in [0, steps]")
    if args.digit_contrast_coef < 0.0:
        raise ValueError("--digit-contrast-coef must be nonnegative")
    if args.digit_residual_coef < 0.0:
        raise ValueError("--digit-residual-coef must be nonnegative")
    if (args.digit_contrast_coef > 0.0 or args.digit_residual_coef > 0.0) and args.digit_group_every < 1:
        raise ValueError("--digit-group-every must be positive when a grouped digit likelihood is enabled")
    if args.digit_group_start < 1:
        raise ValueError("--digit-group-start must be positive")
    if args.deep_visual_likelihood_coef < 0.0:
        raise ValueError("--deep-visual-likelihood-coef must be nonnegative")

    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    device = torch.device(args.device)
    config = model_config(args)
    model = DualStreamOmni(**config).to(device)
    init_report = None
    init_path = Path(args.init) if args.init else None
    if init_path and init_path.exists():
        init_report = load_compatible(model, init_path)
        print(f"initialized from {init_path} ({init_report})", flush=True)
    if args.optimization_phase != "joint":
        model.set_optimization_phase(args.optimization_phase)

    repeats = [int(x.strip()) for x in args.case_repeats.split(",")]
    if (
        len(repeats) != len(CAPABILITY_CASES)
        or any(x < 0 for x in repeats)
        or not any(x > 0 for x in repeats)
    ):
        raise ValueError(
            f"--case-repeats needs {len(CAPABILITY_CASES)} nonnegative integers "
            "with at least one positive value; "
            f"got {args.case_repeats!r}"
        )
    train_cases = tuple(
        case for case, repeat in zip(CAPABILITY_CASES, repeats)
        for _ in range(repeat)
    )

    fast, temporal, shared = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if args.causal_only and not name.startswith("mot_stack.active_gdn2."):
            param.requires_grad_(False)
            continue
        if args.causal_projection_only and name != (
            "mot_stack.active_gdn2.prior_channel_gain"
        ):
            param.requires_grad_(False)
            continue
        is_temporal = (
            "target_time_coord" in name
            or "mot_stack.t_embed" in name
            or "history_stem" in name
            or "action_" in name
            or "active_gdn2" in name
            or ".ada_x." in name
            or ".ada_s." in name
            or ".ada_write." in name
        )
        transition_trainable = (
            "action_" in name or "active_gdn2" in name
            or name.startswith("belief_logvar_head.")
        )
        if args.temporal_only and not transition_trainable:
            param.requires_grad_(False)
            continue
        is_new_interface = (
            name.startswith("head.")
            or name.startswith("seg_head.")
            or name.startswith("belief_logvar_head.")
            or "precision_coord" in name
            or "mot_stack.y_to_c" in name
        )
        if is_temporal:
            temporal.append(param)
        elif is_new_interface:
            fast.append(param)
        else:
            shared.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": shared, "lr": args.lr},
            {"params": fast, "lr": args.head_lr},
            {"params": temporal, "lr": args.time_lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = None
    if args.lr_schedule == "cosine":
        floor = float(args.min_lr_ratio)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: floor + (1.0 - floor) * 0.5 * (
                1.0 + math.cos(math.pi * min(step, args.steps) / max(1, args.steps))
            ),
        )
    bank = fixed_capability_bank(args.res)
    digit_bank = make_static_bank(args.res, "text_to_both")
    quick = []
    eval_cases = CAPABILITY_CASES[:3] if args.static_only else CAPABILITY_CASES
    for case in eval_cases:
        case_bank = [s for s in bank if s["case"] == case]
        ids = np.linspace(0, len(case_bank) - 1, args.eval_per_case, dtype=int)
        quick.extend([case_bank[int(i)] for i in ids])
    ckpt_path = Path(args.ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = float("-inf")
    best_step = 0
    started = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        # Keep the already-proven generator as a rehearsal anchor while the
        # missing boundaries learn; all cases remain online from step one.
        def batch_loss(batch, with_future, with_digit_contrast=False):
            n = len(batch["prompt"])
            zeros = torch.zeros(n, device=device)
            batch["t"] = zeros
            call = (
                model.forward_with_future_posterior if with_future else model
            )
            prompts, text_pi = _language_boundary(
                batch, device, args.future_language_evidence,
            )
            call_args = [batch["image"].to(device), prompts]
            if with_future:
                call_args.append(batch["target_rgb"].to(device))
            out = call(
                *call_args,
                t=zeros,
                image_precision=batch["image_precision"].to(device),
                text_precision=text_pi,
                target_time=batch["target_time"].to(device),
                history_images=batch["history_images"].to(device),
                history_precision=batch["history_precision"].to(device),
                action=batch["action"].to(device),
                action_precision=batch["action_precision"].to(device),
                task_id=batch["task_id"].to(device),
            )
            loss, meta = model.omni_loss(out, batch, device)
            if with_digit_contrast:
                if args.digit_contrast_coef > 0.0:
                    digit_loss, digit_meta = paired_digit_likelihood(
                        out, batch,
                        temperature=args.digit_contrast_temperature,
                        difference_only=args.digit_contrast_difference_only,
                    )
                    loss = loss + float(args.digit_contrast_coef) * digit_loss
                    meta.update(digit_meta)
                if args.digit_residual_coef > 0.0:
                    residual_loss, residual_meta = digit_residual_likelihood(
                        out, batch,
                    )
                    loss = loss + float(args.digit_residual_coef) * residual_loss
                    meta.update(residual_meta)
                    x_steps = out.get("X_steps") or []
                    if args.deep_visual_likelihood_coef > 0.0 and len(x_steps) > 1:
                        deep_residuals = []
                        for x_step in x_steps[:-1]:
                            mu_step, _ = model.decode_gauss(x_step)
                            rgb_step = torch.sigmoid(model._pts_to_img(mu_step))
                            seg_step = model.seg_head(x_step)
                            seg_step = seg_step.transpose(1, 2).reshape(
                                x_step.shape[0], model.seg_classes,
                                model.res, model.res,
                            )
                            step_residual, _ = digit_residual_likelihood(
                                {"rgb": rgb_step, "seg_logits": seg_step}, batch,
                            )
                            deep_residuals.append(step_residual)
                        deep_residual = torch.stack(deep_residuals).mean()
                        weight = (
                            float(args.digit_residual_coef)
                            * float(args.deep_visual_likelihood_coef)
                        )
                        loss = loss + weight * deep_residual
                        meta["deep_digit_residual"] = float(deep_residual.detach())
                        meta["n_deep_digit_steps"] = len(deep_residuals)
            if args.action_contrast_coef > 0.0:
                contrast, contrast_meta = paired_action_likelihood(
                    out, batch, temperature=args.action_contrast_temp,
                )
                loss = loss + float(args.action_contrast_coef) * contrast
                meta.update(contrast_meta)
            return loss, meta

        optimizer.zero_grad(set_to_none=True)
        if args.pcgrad_static:
            if args.action_contrast_coef > 0.0:
                n_groups = max(1, (args.batch // 2) // len(WORLD_ACTIONS_WITH_NOOP))
                n_future = n_groups * len(WORLD_ACTIONS_WITH_NOOP)
                n_static = max(1, args.batch - n_future)
            else:
                n_static = max(1, args.batch // 2)
                n_future = max(1, args.batch - n_static)
            static_batch = make_capability_batch(
                rng, n_static, args.res, cases=CAPABILITY_CASES[:3],
            )
            if args.action_contrast_coef > 0.0:
                future_batch = make_counterfactual_future_batch(
                    rng, n_groups, args.res,
                )
            else:
                future_batch = make_capability_batch(
                    rng, n_future, args.res, cases=("image_to_future",),
                )
            static_loss, static_meta = batch_loss(static_batch, False)
            future_loss, future_meta = batch_loss(future_batch, True)
            params = [p for p in model.parameters() if p.requires_grad]
            gs = torch.autograd.grad(
                static_loss, params, retain_graph=False, allow_unused=True,
            )
            gf = torch.autograd.grad(
                future_loss, params, retain_graph=False, allow_unused=True,
            )
            dot = static_loss.new_zeros(())
            norm = static_loss.new_zeros(())
            for a, b in zip(gs, gf):
                if a is not None and b is not None:
                    dot = dot + (a * b).sum()
                    norm = norm + a.square().sum()
            coef = torch.clamp(-dot / norm.clamp_min(1e-12), min=0.0)
            for param, a, b in zip(params, gs, gf):
                if a is None:
                    grad = b
                elif b is None:
                    grad = a
                else:
                    grad = a + b + coef * a
                param.grad = None if grad is None else grad.detach()
            loss = static_loss + future_loss
            meta = {
                "static": static_meta,
                "future": future_meta,
                "pcgrad_dot": float(dot.detach()),
                "pcgrad_projection": float(coef.detach()),
            }
        else:
            active_cases = (
                ("text_to_both",)
                if step <= args.generation_warmup_steps
                else train_cases
            )
            use_digit_group = bool(
                args.digit_contrast_coef > 0.0 or args.digit_residual_coef > 0.0
            ) and step >= args.digit_group_start and step % args.digit_group_every == 0
            if use_digit_group:
                batch = collate_capability(sample_digit_group(digit_bank, rng))
            else:
                batch = make_capability_batch(
                    rng, args.batch, args.res, cases=active_cases,
                )
            loss, meta = batch_loss(
                batch, not args.static_only and not use_digit_group,
                with_digit_contrast=use_digit_group,
            )
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate_samples(
                model, quick, device, chunk=args.batch,
                future_language_evidence=args.future_language_evidence,
            )
            quick_future = [s for s in quick if s["case"] == "image_to_future"]
            action_ablated = None
            transition_ablated = None
            transition_ablation_name = None
            if args.static_only:
                floors = {
                    "text_to_both": 0.85,
                    "image_to_current": 0.75,
                    "image_text_edit": 0.80,
                }
                margins = [ev[case]["score"] - floor for case, floor in floors.items()]
                admitted = min(margins) >= 0.0
                if args.static_selection == "generation":
                    score = float(ev["text_to_both"]["score"])
                elif args.static_selection == "macro":
                    score = float(ev["macro_score"])
                else:
                    score = float(ev["macro_score"] if admitted else -1.0 + min(margins))
                future_primary = float("nan")
            else:
                action_ablated = evaluate_samples(
                    model, quick_future, device, chunk=args.batch, ablate="action",
                    future_language_evidence=args.future_language_evidence,
                )["image_to_future"]
            if not args.static_only and args.action_slice_transition:
                transition_ablated = evaluate_samples(
                    model, quick_future, device, chunk=args.batch,
                    ablate="slice_transition",
                    future_language_evidence=args.future_language_evidence,
                )["image_to_future"]
                transition_ablation_name = "slice_transition"
            elif not args.static_only and args.active_gdn2:
                transition_ablated = evaluate_samples(
                    model, quick_future, device, chunk=args.batch,
                    ablate="causal_memory",
                    future_language_evidence=args.future_language_evidence,
                )["image_to_future"]
                transition_ablation_name = "causal_memory"
            if not args.static_only:
                score, future_primary, admitted = _checkpoint_score(
                    ev, action_ablated, action_min_gain=args.action_min_gain,
                    transition_ablated=transition_ablated,
                )
            history.append({
                "step": step,
                "loss": float(loss.detach()),
                "meta": meta,
                "eval": ev,
                "checkpoint_score": score,
                "future_primary": future_primary,
                "admitted": admitted,
                "action_ablated": action_ablated,
                "transition_ablation": transition_ablation_name,
                "transition_ablated": transition_ablated,
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
            })
            if score > best_score:
                best_score, best_step = score, step
                torch.save(
                    {
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "config": config,
                        "schema": "northstar-capability-v1",
                    },
                    ckpt_path,
                )
            bits = " ".join(f"{case}={ev[case]['score']:.3f}" for case in eval_cases)
            print(
                f"step {step:4d}/{args.steps} loss={float(loss.detach()):.4f} "
                f"macro={ev['macro_score']:.3f} future_primary={future_primary:.3f} "
                f"admitted={admitted} {bits}",
                flush=True,
            )

    best = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(best["state_dict"])
    model.to(device)
    full = evaluate_samples(
        model, bank, device, chunk=args.batch,
        future_language_evidence=args.future_language_evidence,
    )
    causal = audit_causal_boundaries(
        model, bank, device, args.batch,
        future_language_evidence=args.future_language_evidence,
    )
    gallery = Path(args.gallery)
    render_gallery(
        model, bank, device, gallery,
        future_language_evidence=args.future_language_evidence,
    )
    record = {
        "schema": "northstar-capability-v1",
        "claim_scope": (
            f"Candidate evaluation on a deterministic {args.res}x{args.res} colored-digit microbenchmark. "
            "Metrics and admission fields determine capability closure. Text output is one categorical token."
        ),
        "architecture": (
            "one full-resolution X + H coevolution graph; transient fixed Slice; "
            "single terminal text/RGB/segmentation likelihood heads"
        ),
        "active_inference": {
            "recipe": "active_f2",
            "likelihoods": ["categorical text", "Gaussian/BCE RGB", "categorical point mask"],
            "boundary_coordinates": [
                "image_precision", "text_precision", "action_precision", "target_time"
            ],
            "history_size": 2,
            "prior_write": 1.0,
            "vfe_coef": 0.1,
            "transition_kl_coef": args.transition_loss_coef,
            "transition_posterior_loss_coef": args.transition_posterior_loss_coef,
            "causal_memory_loss_coef": args.causal_memory_loss_coef,
            "transition_detach_q": True,
            "paired_action_likelihood": {
                "coefficient": args.action_contrast_coef,
                "temperature": args.action_contrast_temp,
                "rgb_model": "unit-variance Gaussian MSE",
                "segmentation_model": "categorical NLL weighted by 0.1",
            },
        },
        "config": config,
        "steps": args.steps,
        "generation_warmup_steps": args.generation_warmup_steps,
        "optimization_phase": args.optimization_phase,
        "digit_contrast_coef": args.digit_contrast_coef,
        "digit_residual_coef": args.digit_residual_coef,
        "digit_contrast_temperature": args.digit_contrast_temperature,
        "digit_contrast_difference_only": bool(args.digit_contrast_difference_only),
        "digit_group_start": args.digit_group_start,
        "deep_visual_likelihood_coef": args.deep_visual_likelihood_coef,
        "static_only": bool(args.static_only),
        "task_tokens": bool(args.task_tokens),
        "static_selection": args.static_selection,
        "lr_schedule": args.lr_schedule,
        "min_lr_ratio": args.min_lr_ratio,
        "case_repeats": dict(zip(CAPABILITY_CASES, repeats)),
        "temporal_only": bool(args.temporal_only),
        "causal_only": bool(args.causal_only),
        "causal_projection_only": bool(args.causal_projection_only),
        "pcgrad_static": bool(args.pcgrad_static),
        "action_min_gain": args.action_min_gain,
        "future_language_evidence": bool(args.future_language_evidence),
        "best_step": best_step,
        "checkpoint_selection": (
            "maximize text_to_both score during an explicit static generation refinement"
            if args.static_only and args.static_selection == "generation"
            else "maximize static generation/current/edit macro score"
            if args.static_only and args.static_selection == "macro"
            else (
                "maximize static macro score after generation/current/edit score floors "
                "0.85/0.75/0.80"
                if args.static_only
                else (
                    "maximize mean(future paired-IoU, segmentation-IoU) after "
                    "generation/current/edit score floors 0.85/0.75/0.80 and after "
                    "intact action conditioning beats action_precision=0 on both "
                    "future paired-IoU and segmentation-IoU by more than "
                    f"{args.action_min_gain:.6f}; when enabled, the Slice transition "
                    "or active causal memory must also beat its own ablation on both metrics"
                )
            )
        ),
        "best_quick_score": best_score,
        "elapsed_sec": time.time() - started,
        "init": str(init_path) if init_path else None,
        "init_report": init_report,
        "checkpoint": str(ckpt_path.resolve()),
        "gallery": str(gallery.resolve()),
        "fixed_bank": {"samples": len(bank), "per_case": 90},
        "final": full,
        "causal_boundary_audit": causal,
        "history": history,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"final": full, "causal": causal}, indent=2), flush=True)
    print(f"saved {out_path} and {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
