"""Native-resolution static capacity training with a lazy bank and accumulation.

The reference checkpoint supplies configuration only, never pretrained weights.
Every RGB target is rasterized at the requested resolution; no output upscaling.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.capability_tasks import CAPABILITY_CASES, capability_sample, collate_capability
from fine_grain.gen_metrics import (
    background_flood_rate, free_color_acc, paired_digit_scores, paired_ink_iou,
)
from fine_grain.omni_model import DualStreamOmni
from fine_grain.omni_tasks import GRID_PLACES
from fine_grain.vlm_data import COLORS
from scripts.audit_attention_write import forward_batch

CASES = CAPABILITY_CASES[:3]


@dataclass(frozen=True)
class Raster:
    res: int = 256
    box: int = 88
    stroke_px: int = 7
    normalized_layout: bool = False

    def sample(self, descriptor):
        case, digit, color, place = descriptor
        # Static action is deterministic and gated off at tau=0.
        return capability_sample(
            np.random.default_rng(91), self.res, case, digit=digit,
            color=color, place=place, glyph_box=self.box,
            glyph_stroke_px=self.stroke_px, normalized_layout=self.normalized_layout,
        )


def scene_bank():
    """Store only descriptors: 360 scenes, not gigabytes of rendered tensors."""
    return list(itertools.product(map(str, range(10)), list(COLORS), GRID_PLACES))


def evaluation_bank(scenes, count=0):
    # Deterministic small panel; zero means the complete training bank.
    if count:
        order = np.random.default_rng(31415).permutation(len(scenes))[:count]
        scenes = [scenes[int(i)] for i in order]
    return [(case, *scene) for case in CASES for scene in scenes]


class BalancedStream:
    """Each mode traverses every scene before reshuffling; no mode is starved."""

    def __init__(self, scenes, seed):
        self.scenes = scenes
        self.rng = np.random.default_rng(seed)
        self.queues = {case: [] for case in CASES}
        self.joint_index = 0

    def take(self, count, generation_only=False):
        result = []
        for _ in range(count):
            case = CASES[0] if generation_only else CASES[self.joint_index % 3]
            if not generation_only:
                self.joint_index += 1
            if not self.queues[case]:
                self.queues[case] = self.rng.permutation(len(self.scenes)).tolist()
            scene = self.scenes[self.queues[case].pop()]
            result.append((case, *scene))
        return result


@torch.no_grad()
def evaluate(model, descriptors, raster, device, microbatch):
    model.eval()
    rows = {case: [] for case in CASES}
    for start in range(0, len(descriptors), microbatch):
        samples = [raster.sample(d) for d in descriptors[start:start + microbatch]]
        out = forward_batch(model, samples, device)
        # Metrics use the exact target-resolution renderer on CPU.
        rgb = out["rgb"].detach().cpu()
        segmentation = out["seg_logits"].argmax(1).cpu() > 0
        labels = out["logits"].argmax(-1).cpu()
        for i, sample in enumerate(samples):
            mask, gold = segmentation[i], sample["target_seg"] > 0
            identity = paired_digit_scores(
                rgb[i], sample["digit"], sample["target_color"], sample["target_place"],
                box=raster.box, stroke_px=raster.stroke_px,
                normalized_layout=raster.normalized_layout,
            )["paired_digit_top1"]
            rows[sample["case"]].append({
                "paired_digit_top1": identity,
                "paired_iou": paired_ink_iou(rgb[i], sample["stroke"], sample["target_color"]),
                "text_acc": float(labels[i] == model.ans_to_idx[sample["answer"]]),
                "seg_iou": float((mask & gold).sum() / (mask | gold).sum().clamp_min(1)),
                "color_acc": free_color_acc(rgb[i:i + 1], sample["target_color"]),
                "flood": background_flood_rate(rgb[i:i + 1], sample["target_rgb"], sample["stroke"]),
            })
        del out
    return {case: {"n": len(values), **{
        key: float(np.mean([row[key] for row in values])) for key in values[0]
    }} for case, values in rows.items() if values}


def selection_score(metrics):
    # Weakest capability matters; color/geometry are not replaced by identity.
    return min(np.mean([m[k] for k in (
        "paired_digit_top1", "paired_iou", "text_acc", "seg_iou", "color_acc",
    )]) - m["flood"] for m in metrics.values())


def is_joint_candidate(step, warmup_steps, score, best_score):
    """Never select a generation-only warmup as the shared-task candidate."""
    return step > warmup_steps and score > best_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-from", default="checkpoints/gaussian_layout_per_head_finish.pt")
    parser.add_argument("--init", default=None,
                        help="Continue matching-resolution weights; explicitly starts a new optimizer/schedule")
    parser.add_argument("--audit-only", action="store_true",
                        help="Reload --init and evaluate the complete bank without training")
    parser.add_argument("--res", type=int, default=256)
    parser.add_argument("--glyph-box", type=int, default=88)
    parser.add_argument("--stroke-px", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-scenes", type=int, default=30)
    parser.add_argument("--tag", default="static256_scratch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.audit_only and not args.init:
        parser.error("audit-only requires --init")
    if min(args.steps, args.microbatch, args.accumulation, args.eval_every) < 1:
        parser.error("steps, batch, accumulation and evaluation interval must be positive")
    if not 0 <= args.warmup_steps < args.steps:
        parser.error("warmup must leave at least one joint-training step")
    if not 2 <= args.glyph_box <= args.res - 2 or args.stroke_px < 1 or args.stroke_px % 2 != 1:
        parser.error("glyph box must fit and stroke width must be positive and odd")
    if not 0 <= args.eval_scenes <= 360:
        parser.error("eval-scenes must be between 0 (full bank) and 360")
    if Path(args.tag).name != args.tag or args.tag in (".", ".."):
        parser.error("tag must be a filename, not a path")
    checkpoint = ROOT / "checkpoints" / f"{args.tag}_best.pt"
    last_path = checkpoint.with_name(f"{args.tag}_last.pt")
    out_path = ROOT / "results" / "published" / f"{args.tag}.json"
    if any(p.exists() for p in (checkpoint, last_path, out_path)):
        parser.error("output already exists; use a new tag to preserve previous runs")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    reference = torch.load(ROOT / args.config_from, map_location="cpu")
    config = {**reference["config"], "res": args.res, "control_prefix_attention": True,
              "gaussian_head_layout": "per_head", "use_attention_sink": False,
              "deep_visual_likelihood_coef": 0.0}
    del reference
    model = DualStreamOmni(**config).to(device)
    raster = Raster(args.res, args.glyph_box, args.stroke_px)
    if args.init:
        initial = torch.load(ROOT / args.init, map_location="cpu")
        if initial["config"] != config or initial.get("raster") != asdict(raster):
            parser.error("continuation requires identical graph and raster, not resolution transfer")
        model.load_state_dict(initial["state_dict"])
        initial_step = initial.get("step")
        del initial
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scenes = scene_bank()
    stream = BalancedStream(scenes, args.seed)
    quick_bank = evaluation_bank(scenes, args.eval_scenes)
    report = {
        "scope": "native-resolution synthetic fixed-bank overfit; not generalization, 1px OCR, natural images, Pythia, or future prediction",
        "initialization": ("same-resolution checkpoint continuation; new optimizer/schedule"
                           if args.init else "random weights; reference config only"),
        "config": config, "raster": asdict(raster), "training": vars(args),
        "scene_count": len(scenes), "sample_count": 3 * len(scenes),
        "effective_batch": args.microbatch * args.accumulation,
        "status": "running", "history": [], "seen_by_case": {case: 0 for case in CASES},
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    started = time.perf_counter()
    if args.audit_only:
        report.update({"status": "audit_complete_candidate_only", "checkpoint": args.init,
                       "selected_step": initial_step, "training": None,
                       "initialization": "independent checkpoint reload; no training"})
        report["reloaded_full_metrics"] = evaluate(
            model, evaluation_bank(scenes), raster, device, args.microbatch,
        )
        report["seconds"] = time.perf_counter() - started
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return
    best = -float("inf")
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        descriptors = stream.take(args.microbatch * args.accumulation, step <= args.warmup_steps)
        total_loss = 0.0
        for offset in range(0, len(descriptors), args.microbatch):
            chunk = descriptors[offset:offset + args.microbatch]
            samples = [raster.sample(d) for d in chunk]
            for case, *_ in chunk:
                report["seen_by_case"][case] += 1
            batch = collate_capability(samples)
            batch["t"] = torch.zeros(len(samples), device=device)
            out = forward_batch(model, samples, device)
            loss, _ = model.omni_loss(out, batch, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            (loss / args.accumulation).backward()
            total_loss += float(loss.detach()) / args.accumulation
            del out, loss
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        rate = args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.step()
        if step == 1 or step % 20 == 0:
            print(json.dumps({"step": step, "loss": total_loss, "lr": rate,
                              "seconds": time.perf_counter() - started}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, quick_bank, raster, device, args.microbatch)
            row = {"step": step, "loss": total_loss, "lr": rate,
                   "gradient_norm": float(norm), "quick_metrics": metrics}
            report["history"].append(row)
            score = selection_score(metrics)
            saved = {"config": config, "state_dict": model.state_dict(), "step": step,
                     "raster": asdict(raster), "scope": report["scope"],
                     "training": vars(args), "optimizer": optimizer.state_dict()}
            torch.save(saved, last_path)
            # A generation-only checkpoint cannot be a joint candidate.
            if is_joint_candidate(step, args.warmup_steps, score, best):
                best = score
                torch.save(saved, checkpoint)
            report["seconds"] = time.perf_counter() - started
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)
    # Selection uses a small panel; the final claim uses the entire saved bank.
    saved = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(saved["state_dict"])
    report["selected_step"] = saved["step"]
    report["checkpoint"] = str(checkpoint.relative_to(ROOT))
    del saved
    report["reloaded_full_metrics"] = evaluate(model, evaluation_bank(scenes), raster, device, args.microbatch)
    report["unique_scenes_seen_by_case"] = {
        case: min(count, len(scenes)) for case, count in report["seen_by_case"].items()
    }
    report["status"] = "completed_budget_candidate_only"
    report["seconds"] = time.perf_counter() - started
    if device.type == "cuda":
        report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
