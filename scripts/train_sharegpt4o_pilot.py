#!/usr/bin/env python3
"""Train a candidate-only real-data pilot on the North-Star graph.

Pythia remains frozen and supplies only embeddings/causal likelihoods.  Image
generation and editing are terminal likelihoods of the same persistent
X-Slice-H graph.  This script never overwrites a protected synthetic champion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.omni_model import DualStreamOmni
from fine_grain.pythia_bridge import (
    capability_champion_kwargs,
    load_visual_champion,
    param_groups,
    set_optimization_phase,
)
from fine_grain.sharegpt4o_data import (
    collate_real_multimodal,
    counterfactual_real_samples,
    materialize_record,
)
from scripts.train_pythia_capabilities import (
    PROTECTED_CHECKPOINTS,
    current_gate,
    edit_gate,
    eval_current,
    eval_edit_static,
    eval_static_t2i,
    make_cycle_eval_bank,
)
from scripts.train_pythia_generation import t2i_gate


DEFAULT_INIT = ROOT / "checkpoints" / "omni_d64_pythia_capability_b3_edit_best.pt"
DEFAULT_CANDIDATE = ROOT / "checkpoints" / "omni_d64_pythia_sharegpt4o_candidate.pt"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def forward_batch(model, batch: dict) -> dict:
    return model.forward_tokens(
        batch["image"],
        batch["input_ids"],
        batch["attention_mask"],
        labels=batch["labels"],
        visual_prompt_mask=batch["visual_prompt_mask"],
        image_precision=batch["image_precision"],
        text_precision=batch["text_precision"],
    )


def terminal_nll(out: dict, batch: dict) -> torch.Tensor:
    """Per-row terminal likelihood used by the causal shuffle gate."""
    rows = []
    rgb = out["rgb"]
    lv = out["rgb_lv"].clamp(-6.0, 3.0)
    rgb_nll = 0.5 * (
        lv + (batch["target_rgb"] - rgb).pow(2) * torch.exp(-lv)
    ).flatten(1).mean(dim=1)
    token_nll = out["token_nll"]
    for index, (need_pix, need_text) in enumerate(
        zip(batch["need_pix"], batch["need_text"])
    ):
        if need_pix:
            rows.append(rgb_nll[index])
        elif need_text:
            rows.append(token_nll[index])
        else:
            raise RuntimeError("sample has no terminal likelihood")
    return torch.stack(rows)


def reconstruction_views(samples: list[dict]) -> list[dict]:
    """Turn real observed images into zero-text-precision reconstruction rows."""
    rows = []
    for sample in samples:
        if float(sample["image_precision"]) == 0.0:
            continue
        row = deepcopy(sample)
        row["id"] = str(row["id"]) + "-reconstruct"
        row["task"] = "reconstruct"
        row["prompt"] = ""
        row["answer"] = ""
        row["text_missing"] = True
        row["target_rgb"] = row["image"].clone()
        row["need_pix"] = True
        row["need_text"] = False
        row["target_image_precision"] = 1.0
        row["target_text_precision"] = 0.0
        rows.append(row)
    return rows


@torch.no_grad()
def static_audit(model, device: torch.device, batch_size: int) -> dict:
    banks = {
        case: make_cycle_eval_bank(model.res, case)
        for case in ("text_to_both", "image_to_current", "image_text_edit")
    }
    metrics = {
        "t2i": eval_static_t2i(
            model, banks["text_to_both"], device, chunk=batch_size,
        ),
        "current": eval_current(
            model, banks["image_to_current"], device, chunk=batch_size,
        ),
        "edit": eval_edit_static(
            model, banks["image_text_edit"], device, chunk=batch_size,
        ),
    }
    return {
        "metrics": metrics,
        "gates": {
            "t2i": t2i_gate(metrics["t2i"]),
            "current": current_gate(metrics["current"]),
            "edit": edit_gate(metrics["edit"]),
        },
    }


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    groups: dict[str, list[dict]],
    device,
    batch_size: int,
    limit: int | None = None,
    max_answer_tokens: int | None = None,
    append_eos: bool = True,
) -> dict:
    model.eval()
    result = {}
    for task, samples in groups.items():
        if not samples:
            continue
        if limit is not None and len(samples) > int(limit):
            indices = np.linspace(0, len(samples) - 1, int(limit), dtype=int)
            samples = [samples[int(index)] for index in indices]
        matched_values, shuffled_values = [], []
        matched_mse_values, shuffled_mse_values = [], []
        for start in range(0, len(samples), batch_size):
            part = samples[start : start + batch_size]
            if len(part) < 2:
                continue
            batch = move_batch(collate_real_multimodal(
                tokenizer, part, max_answer_tokens=max_answer_tokens,
                append_eos=append_eos,
            ), device)
            matched_out = forward_batch(model, batch)
            matched = terminal_nll(matched_out, batch)
            controls = counterfactual_real_samples(part)
            control_batch = move_batch(
                collate_real_multimodal(
                    tokenizer, controls, max_answer_tokens=max_answer_tokens,
                    append_eos=append_eos,
                ), device,
            )
            shuffled_out = forward_batch(model, control_batch)
            shuffled = terminal_nll(shuffled_out, control_batch)
            matched_values.append(matched.cpu())
            shuffled_values.append(shuffled.cpu())
            if all(bool(value) for value in batch["need_pix"]):
                matched_mse_values.append(
                    (matched_out["rgb"] - batch["target_rgb"])
                    .pow(2).flatten(1).mean(dim=1).cpu()
                )
                shuffled_mse_values.append(
                    (shuffled_out["rgb"] - control_batch["target_rgb"])
                    .pow(2).flatten(1).mean(dim=1).cpu()
                )
        if not matched_values:
            continue
        matched = torch.cat(matched_values)
        shuffled = torch.cat(shuffled_values)
        gap = shuffled - matched
        result[task] = {
            "n": int(matched.numel()),
            "nll": float(matched.mean()),
            "shuffled_nll": float(shuffled.mean()),
            "mean_causal_gap": float(gap.mean()),
            "positive_gap_fraction": float((gap > 0).float().mean()),
        }
        if matched_mse_values:
            mse = torch.cat(matched_mse_values).mean().item()
            shuffled_mse = torch.cat(shuffled_mse_values).mean().item()
            result[task].update({
                "rgb_mse": float(mse),
                "rgb_psnr": float(-10.0 * math.log10(max(mse, 1e-12))),
                "shuffled_rgb_mse": float(shuffled_mse),
                "rgb_mse_causal_gap": float(shuffled_mse - mse),
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default=r"D:\ml_cache\sharegpt4o\pilot_manifest.json",
    )
    parser.add_argument("--init", default=str(DEFAULT_INIT))
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument(
        "--out", default=str(ROOT / "results" / "sharegpt4o_real_pilot.json"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lm-device", default=None)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--phase",
        choices=(
            "auto", "token_reader", "rgb_likelihood", "language_rgb",
            "edit_spatial", "edit_read",
        ),
        default="auto",
        help=(
            "Use a registered narrow phase. Safe auto maps image terminals to "
            "rgb_likelihood and text terminals to token_reader; joint is forbidden."
        ),
    )
    parser.add_argument("--shuffle-coef", type=float, default=0.25)
    parser.add_argument("--shuffle-margin", type=float, default=0.02)
    parser.add_argument(
        "--tasks", default="reconstruct,t2i,it2i,i2t,it2t",
        help="Comma-separated registered real-data tasks to rehearse.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--eval-limit", type=int, default=64,
        help="Deterministic examples evaluated per task; use 0 for all.",
    )
    parser.add_argument(
        "--holdout-fraction", type=float, default=0.2,
        help="Deterministic per-port holdout used only for real-data evaluation.",
    )
    parser.add_argument(
        "--max-answer-tokens", type=int, default=64,
        help="Supervise the visually grounded answer prefix; 0 keeps all available tokens.",
    )
    parser.add_argument(
        "--answer-eos", action=argparse.BooleanOptionalAction, default=True,
        help="Supervise answer termination; disable only to reproduce legacy runs.",
    )
    parser.add_argument(
        "--skip-static-audit", action="store_true",
        help="Skip synthetic gates during a memory smoke; audit the saved candidate at res=16.",
    )
    args = parser.parse_args()

    init_path = Path(args.init).resolve()
    candidate_path = Path(args.candidate).resolve()
    protected = {Path(path).resolve() for path in PROTECTED_CHECKPOINTS}
    protected.add(DEFAULT_INIT.resolve())
    if candidate_path in protected:
        raise SystemExit(f"refusing to overwrite protected checkpoint {candidate_path}")
    if not init_path.is_file():
        raise FileNotFoundError(init_path)
    init_sha256 = file_sha256(init_path)

    raw = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    samples = [
        materialize_record(row, int(args.resolution))
        for row in raw.get("records", [])
    ]
    if not samples:
        raise RuntimeError("pilot manifest contains no materialized samples")
    all_groups: dict[str, list[dict]] = {}
    for sample in samples:
        all_groups.setdefault(str(sample["task"]), []).append(sample)
    holdout_fraction = float(args.holdout_fraction)
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("--holdout-fraction must be in [0, 1)")
    train_groups: dict[str, list[dict]] = {}
    eval_groups: dict[str, list[dict]] = {}
    split_rng = np.random.default_rng(20260824)
    for task in sorted(all_groups):
        bank = all_groups[task]
        if holdout_fraction == 0.0 or len(bank) < 3:
            train_groups[task] = list(bank)
            eval_groups[task] = list(bank)
            continue
        order = split_rng.permutation(len(bank))
        n_eval = max(2, int(round(len(bank) * holdout_fraction)))
        eval_index = set(int(index) for index in order[:n_eval])
        train_groups[task] = [
            sample for index, sample in enumerate(bank) if index not in eval_index
        ]
        eval_groups[task] = [
            sample for index, sample in enumerate(bank) if index in eval_index
        ]
    train_recon = reconstruction_views([
        sample for bank in train_groups.values() for sample in bank
    ])
    eval_recon = reconstruction_views([
        sample for bank in eval_groups.values() for sample in bank
    ])
    if train_recon:
        train_groups["reconstruct"] = train_recon
    if eval_recon:
        eval_groups["reconstruct"] = eval_recon

    device = torch.device(args.device)
    lm_device = args.lm_device or args.device
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = DualStreamOmni(**capability_champion_kwargs(
        res=int(args.resolution), language="pythia", lm_device=lm_device,
        pixel_loss_mode="gaussian_nll", s0_acc_coef=0.0,
    )).to(device)
    if model.lm is not None:
        model.lm.to(device)
    load_report = load_visual_champion(
        model, init_path, skip_language_interface=False,
    )
    if args.phase == "auto":
        image_names = set(set_optimization_phase(model, "rgb_likelihood"))
        text_names = set(set_optimization_phase(model, "token_reader"))
        trainable_set = image_names | text_names
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in trainable_set)
        trainable = sorted(trainable_set)
    else:
        image_names = text_names = set()
        trainable = set_optimization_phase(model, args.phase)
        trainable_set = set(trainable)
    if any(parameter.requires_grad for parameter in model.lm.parameters()):
        raise RuntimeError("Pythia must remain frozen")
    optimizer = torch.optim.AdamW(
        param_groups(model, interface_lr=args.lr, visual_lr=args.lr),
        weight_decay=1e-4,
    )
    tokenizer = model.lm_tok
    eval_limit = None if int(args.eval_limit) <= 0 else int(args.eval_limit)
    max_answer_tokens = (
        None if int(args.max_answer_tokens) <= 0 else int(args.max_answer_tokens)
    )
    print(
        "real-data ports: "
        + ", ".join(f"{task}={len(bank)}" for task, bank in all_groups.items())
        + "; train="
        + ", ".join(f"{task}:{len(bank)}" for task, bank in train_groups.items())
        + "; heldout="
        + ", ".join(f"{task}:{len(bank)}" for task, bank in eval_groups.items()),
        flush=True,
    )
    static_before = None if args.skip_static_audit else static_audit(
        model, device, max(2, args.batch),
    )
    before = evaluate(
        model, tokenizer, eval_groups, device, max(2, args.batch), eval_limit,
        max_answer_tokens, bool(args.answer_eos),
    )
    history = []
    started = time.time()
    requested_tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unknown = set(requested_tasks) - {"reconstruct", "t2i", "it2i", "i2t", "it2t"}
    if unknown:
        raise ValueError(f"unknown --tasks entries: {sorted(unknown)}")
    task_cycle = [task for task in requested_tasks if task in train_groups]
    if not task_cycle:
        raise RuntimeError("no supported real tasks in pilot")

    if not args.eval_only:
        for step in range(1, int(args.steps) + 1):
            task = task_cycle[(step - 1) % len(task_cycle)]
            if args.phase == "auto":
                active_phase = (
                    "token_reader" if task in ("i2t", "it2t")
                    else "rgb_likelihood"
                )
                set_optimization_phase(model, active_phase)
            else:
                active_phase = args.phase
            bank = train_groups[task]
            replace = len(bank) < int(args.batch)
            indices = rng.choice(len(bank), size=int(args.batch), replace=replace)
            part = [bank[int(index)] for index in indices]
            batch = move_batch(collate_real_multimodal(
                tokenizer, part, max_answer_tokens=max_answer_tokens,
                append_eos=bool(args.answer_eos),
            ), device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            out = forward_batch(model, batch)
            matched = terminal_nll(out, batch)
            base_loss, meta = model.omni_loss(out, batch, device)
            loss = base_loss
            causal = matched.new_zeros(())
            if len(part) >= 2 and task != "reconstruct":
                controls = counterfactual_real_samples(part)
                control_batch = move_batch(
                    collate_real_multimodal(
                        tokenizer, controls, max_answer_tokens=max_answer_tokens,
                        append_eos=bool(args.answer_eos),
                    ), device,
                )
                control_out = forward_batch(model, control_batch)
                shuffled = terminal_nll(control_out, control_batch)
                causal = torch.relu(
                    matched - shuffled + float(args.shuffle_margin)
                ).mean()
                loss = loss + float(args.shuffle_coef) * causal
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()
            history.append({
                "step": step,
                "task": task,
                "phase": active_phase,
                "loss": float(loss.detach()),
                "terminal_nll": float(matched.mean().detach()),
                "causal_hinge": float(causal.detach()),
                "meta": meta,
            })
            if step == 1 or step % 5 == 0 or step == int(args.steps):
                print(
                    f"step={step:4d} task={task:11s} "
                    f"loss={float(loss.detach()):.4f} "
                    f"nll={float(matched.mean().detach()):.4f} "
                    f"causal={float(causal.detach()):.4f}",
                    flush=True,
                )

        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.non_lm_state_dict(),
            "language": model.language_meta(),
            "candidate_only": True,
            "dataset": "ShareGPT-4o-Image/OpenGVLab-ShareGPT-4o",
            "resolution": int(args.resolution),
            "steps": int(args.steps),
        }, candidate_path)

    after = evaluate(
        model, tokenizer, eval_groups, device, max(2, args.batch), eval_limit,
        max_answer_tokens, bool(args.answer_eos),
    )
    static_after = None if args.skip_static_audit else static_audit(
        model, device, max(2, args.batch),
    )
    final_init_sha256 = file_sha256(init_path)
    if final_init_sha256 != init_sha256:
        raise RuntimeError("protected initialization checkpoint changed during the run")
    record = {
        "schema": "sharegpt4o-real-northstar-pilot-v1",
        "candidate_only": True,
        "admitted": False,
        "init": str(init_path),
        "init_sha256": init_sha256,
        "candidate": str(candidate_path),
        "load_report": load_report,
        "language": model.language_meta(),
        "run": {
            "resolution": int(args.resolution),
            "steps": int(args.steps),
            "batch": int(args.batch),
            "eval_limit_per_task": eval_limit,
            "holdout_fraction": holdout_fraction,
            "max_answer_tokens": max_answer_tokens,
            "answer_eos": bool(args.answer_eos),
            "lr": float(args.lr),
            "phase": args.phase,
            "tasks": task_cycle,
            "pixel_likelihood": "heteroscedastic_gaussian",
            "pythia_frozen": all(
                not parameter.requires_grad for parameter in model.lm.parameters()
            ),
            "lm_generate_called": False,
            "gdn2": False,
            "trainable_count": sum(
                parameter.numel() for name, parameter in model.named_parameters()
                if name in trainable_set
            ),
            "trainable_names": trainable,
        },
        "data": {
            "total": {task: len(bank) for task, bank in all_groups.items()},
            "train": {task: len(bank) for task, bank in train_groups.items()},
            "heldout": {task: len(bank) for task, bank in eval_groups.items()},
        },
        "before": before,
        "after": after,
        "static_before": static_before,
        "static_after": static_after,
        "history": history,
        "elapsed_sec": time.time() - started,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {output}; candidate_only=True", flush=True)


if __name__ == "__main__":
    main()
