"""Equal-weight native 64/256 training and joint per-example admission checks."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fine_grain.omni_model import DualStreamOmni
from fine_grain.real_capacity import (
    add_zero_horizon_controls, bank_metadata, build_real_bank,
    collate_real_capacity, forward_real_capacity,
)
from fine_grain.unified_capacity import extend_unified_bank
from fine_grain.spatial_likelihood import stratified_rgb_weights
from scripts.train_real256_capacity import evaluate, load_graph_weights, sha256, write_gallery


def capability_failures(metrics, required_ids=None):
    """Check every example, not a macro average that hides a failed task."""
    rows = metrics["samples"]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("admission requires nonempty, unique sample results")
    failed = []
    if required_ids is not None:
        failed += [f"{key}:missing_result" for key in sorted(set(required_ids) - {r["id"] for r in rows})]
    for row in rows:
        task = row["task"]
        checks = {}
        if task == "t2i" and row.get("family") == "generation1px":
            checks = {"psnr30": row["psnr"] >= 30, "rgb_stroke": row["rgb_stroke_iou"] >= .9,
                      "seg_stroke": row["seg_iou"] >= .95,
                      "prompt_used": row["prompt_stroke_iou_gap"] > .1}
        elif task == "t2i":
            checks = {"psnr20": row["psnr"] >= 20, "edge_content": row["edge_correlation"] >= .5,
                      "prompt_used": row["control_mse_gap"] > .001}
        elif task == "it2i":
            checks = {"psnr23": row["psnr"] >= 23, "edge_content": row["edge_correlation"] >= .6,
                      "instruction_used": row["control_mse_gap"] > .001}
        elif task in ("i2t", "it2t", "t2t"):
            checks = {"full_caption": row.get("decoded", {}).get("exact", False),
                      "eos": row.get("decoded", {}).get("eos", False)}
            if task != "t2t":
                checks["image_used"] = row["image_shuffle_nll_gap"] > .3
            if task != "i2t":
                checks["question_used"] = row["text_shuffle_nll_gap"] > .3
            if row.get("need_pix", True):
                checks["reconstruction"] = row["psnr"] >= 25
        elif task == "reconstruction":
            checks = {"psnr25": row["psnr"] >= 25, "edge_content": row["edge_correlation"] >= .7}
        elif task == "segmentation":
            checks = {"foreground_iou": row["seg_iou"] >= .85, "reconstruction": row["psnr"] >= 25}
        elif task == "future":
            checks = {"foreground_iou": row["seg_iou"] >= .85, "psnr25": row["psnr"] >= 25}
            if row["horizon"] > 0:
                checks.update(beats_copy=row["improvement_over_copy"] > 0,
                              predicts_change=row["changed_region_mse"] < row["changed_region_copy_mse"],
                              horizon_used=row["zero_horizon_mse_gap"] > 0)
        else:
            raise ValueError(f"no capability checks registered for {task!r}")
        failed += [f'{row["id"]}:{name}' for name, passed in checks.items() if not passed]
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", default="checkpoints/real256_joint.pt")
    parser.add_argument("--sharegpt-manifest", required=True)
    parser.add_argument("--davis-manifest", required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--steps", type=int, default=3200)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--tag", default="mixed_native_64_256")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--full-bank", action="store_true",
                        help="Also train authored real-image QA, pure text QA, native noisy OCR and 1px generation")
    parser.add_argument("--resume-training", action="store_true",
                        help="Restore optimizer, sampling position and schedule from --init; keep the original total steps")
    parser.add_argument("--rgb-measure", choices=("uniform", "stratified"), default="uniform")
    parser.add_argument("--stop-after-updates", type=int, default=None,
                        help="Bound this trial without restarting the original learning-rate schedule")
    args = parser.parse_args()
    if sorted(args.resolutions) != [64, 256]:
        parser.error("this goal requires both native 64 and native 256, exactly once each")
    if args.steps < 1 or args.eval_every < 1:
        parser.error("steps and eval interval must be positive")
    if Path(args.tag).name != args.tag or args.tag in (".", ".."):
        parser.error("tag must be a filename")
    path = ROOT / "checkpoints" / f"{args.tag}.pt"
    output = ROOT / "results" / "published" / f"{args.tag}.json"
    if path.exists() or output.exists():
        parser.error("use a new tag; existing run files are protected")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    initial = torch.load(ROOT / args.init, map_location="cpu")
    start_step = int(initial["step"]) if args.resume_training else 0
    if args.stop_after_updates is not None and args.stop_after_updates < 1:
        parser.error("stop-after-updates must be positive")
    end_step = min(args.steps, start_step + args.stop_after_updates) if args.stop_after_updates else args.steps
    if args.resume_training:
        for key in ("steps", "lr", "full_bank", "resolutions"):
            if initial.get("training", {}).get(key) != getattr(args, key):
                parser.error(f"resume must retain original {key}")
        if start_step >= args.steps or "optimizer" not in initial or "seen" not in initial:
            parser.error("resume requires an unfinished checkpoint with optimizer and sampling state")
    config = initial["config"]
    model = DualStreamOmni(**config).to(args.device)
    load_graph_weights(model, initial["state_dict"])
    initial_step = initial["step"]
    banks = {r: add_zero_horizon_controls(build_real_bank(args.sharegpt_manifest, args.davis_manifest, r))
             for r in args.resolutions}
    if args.full_bank:
        banks = {r: extend_unified_bank(bank, r) for r, bank in banks.items()}
    spatial_weights = {}
    if args.rgb_measure == "stratified":
        for res, bank in banks.items():
            for sample in bank:
                if sample["need_pix"]:
                    foreground = sample["target_seg"][None] if sample["need_seg"] else None
                    spatial_weights[(res, sample["id"])] = stratified_rgb_weights(sample["target_rgb"][None], foreground)
    group_ids = defaultdict(list)
    for i, sample in enumerate(banks[64]):
        # Keep natural-image replay from being diluted by ten OCR/glyph cases.
        group_ids[(sample["task"], sample.get("family", "real"))].append(i)
    tasks = list(group_ids)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    if args.resume_training:
        optimizer.load_state_dict(initial["optimizer"])
    report = {"status": "training", "goal": "one champion shared across native 64x64 and 256x256",
              "training": vars(args), "config": config, "init_step": initial_step,
              "init_sha256": sha256(ROOT / args.init),
              "initialization": ("restored graph, optimizer, sample position and original cosine schedule"
                                 if args.resume_training else "same graph weights; new optimizer and cosine schedule"),
              "resolution_weighting": "mean of two native-resolution losses before one optimizer step",
              "sampling": "equal task-family groups; uniform example rotation within each group",
              "data": bank_metadata(banks[256]), "history": [],
              "scope": ("42 fixed examples: real bank plus explicitly authored QA and synthetic 1px; not generalization"
                        if args.full_bank else "real fixed-bank capacity; remaining text QA and 1px retention gates still required for the full north star"),
              "seen": {str(r): dict.fromkeys([s["id"] for s in banks[r]], 0) for r in args.resolutions}}
    if args.resume_training:
        if any(set(initial["seen"][r]) != set(rows) for r, rows in report["seen"].items()):
            parser.error("resume bank IDs differ from saved checkpoint")
        report["seen"] = initial["seen"]
        previous = ROOT / "results" / "published" / f'{initial["training"]["tag"]}.json'
        if previous.exists():
            prior_report = json.loads(previous.read_text(encoding="utf-8"))
            report["history"] = [e for e in prior_report["history"] if e["step"] <= start_step]
        report["resumed_from_step"] = start_step
        report["rng_restored"] = "rng_state" in initial
        if "rng_state" in initial:
            torch.set_rng_state(initial["rng_state"])
            if args.device.startswith("cuda") and initial.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(initial["cuda_rng_state"])
    del initial
    output.parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    report["current_step"] = start_step
    report["checkpoint_step"] = None
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    started = time.perf_counter()
    for step in range(start_step + 1, end_step + 1):
        task = tasks[(step - 1) % len(tasks)]
        indices = group_ids[task]
        index = indices[((step - 1) // len(tasks)) % len(indices)]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        # Each loss is normalized inside its own native grid; no padding or
        # resizing to the larger image, and no resolution-private parameters.
        for res in args.resolutions:
            sample = banks[res][index]
            batch = collate_real_capacity(model.lm_tok, [sample], args.device)
            weight = spatial_weights.get((res, sample["id"]))
            if weight is not None:
                batch["rgb_likelihood_weight"] = weight.to(args.device)
            out = forward_real_capacity(model, batch, with_posterior=True)
            loss, _ = model.omni_loss(out, batch, torch.device(args.device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"step {step}, res {res}, task {task}")
            (loss / len(args.resolutions)).backward()
            total += float(loss.detach()) / len(args.resolutions)
            report["seen"][str(res)][sample["id"]] += 1
            del out, loss
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1,
                                      error_if_nonfinite=True)
        rate = args.lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps)))
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.step()
        if any(p.grad is not None or p.requires_grad for p in model.lm.parameters()):
            raise RuntimeError("Pythia must remain frozen")
        if step == 1 or step % 50 == 0:
            report["current_step"] = step
            report["seconds"] = time.perf_counter() - started
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps({"step": step, "task": task, "mean_loss": total,
                              "lr": rate, "seconds": time.perf_counter() - started}), flush=True)
        if step % args.eval_every == 0 or step == end_step:
            entry = {"step": step, "resolutions": {}}
            for res in args.resolutions:
                metrics, displays = evaluate(model, banks[res], args.device)
                failures = capability_failures(metrics, [s["id"] for s in banks[res]])
                entry["resolutions"][str(res)] = {"metrics": metrics, "failures": failures}
                write_gallery(ROOT / "present" / f"{args.tag}_{res}.html", displays, metrics, step)
                print(json.dumps({"eval_step": step, "res": res, "failed_checks": len(failures),
                                  "summary": metrics["summary"]}), flush=True)
            report["history"].append(entry)
            report["seconds"] = time.perf_counter() - started
            payload = {"config": config, "state_dict": model.non_lm_state_dict(), "step": step,
                       "optimizer": optimizer.state_dict(), "training": vars(args),
                       "language": model.language_meta(), "seen": report["seen"], "data": report["data"],
                       "rng_state": torch.get_rng_state(),
                       "cuda_rng_state": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else None}
            torch.save(payload, path)
            report["checkpoint_step"] = step
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    del model, optimizer, payload, displays
    if args.device.startswith("cuda"):
        report["peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
        torch.cuda.empty_cache()
    saved = torch.load(path, map_location="cpu")
    model = DualStreamOmni(**saved["config"]).to(args.device)
    load_graph_weights(model, saved["state_dict"])
    report["reloaded"] = {}
    for res in args.resolutions:
        metrics, displays = evaluate(model, banks[res], args.device)
        report["reloaded"][str(res)] = {
            "metrics": metrics, "failures": capability_failures(metrics, [s["id"] for s in banks[res]])}
        write_gallery(ROOT / "present" / f"{args.tag}_{res}.html", displays, metrics, end_step)
    report["status"] = "completed_budget_candidate_only"
    report["checkpoint"] = str(path.relative_to(ROOT))
    report["checkpoint_sha256"] = sha256(path)
    report["seconds"] = time.perf_counter() - started
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "failed_checks": {
        r: len(v["failures"]) for r, v in report["reloaded"].items()}}), flush=True)


if __name__ == "__main__":
    main()
