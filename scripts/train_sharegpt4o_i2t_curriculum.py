#!/usr/bin/env python3
"""Progressive real-image I2T overfit gate for the North-Star graph.

This is a capacity/training-interface diagnostic, not a generalization claim.
Pythia stays frozen; every decoded token reruns the complete X-Slice-H graph.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.omni_model import DualStreamOmni
from fine_grain.ocr_1px import make_ocr_1px
from fine_grain.pythia_bridge import (
    capability_champion_kwargs,
    load_visual_champion,
    set_optimization_phase,
)
from fine_grain.sharegpt4o_data import (
    collate_real_multimodal,
    materialize_record,
)
from scripts.train_pythia_capabilities import PROTECTED_CHECKPOINTS


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_answer(text: str, max_words: int = 12) -> str:
    """Deterministically keep one concise, natural reference sentence."""
    text = re.sub(r"```.*?```", " ", str(text), flags=re.DOTALL)
    text = re.sub(r"[*#_`]+", " ", text)
    text = " ".join(text.split())
    if not text:
        raise ValueError("empty I2T answer")
    match = re.match(r"^(.*?[.!?])([\"'”’])?(?:\s+|$)", text)
    sentence = (
        match.group(1) + (match.group(2) or "") if match else text
    )
    words = sentence.split()[: max(1, int(max_words))]
    answer = " ".join(words).strip()
    if not re.search(r"[.!?][\"'”’]*$", answer):
        answer += "."
    return answer


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def make_fixed_ocr_samples(resolution: int, seed: int) -> list[dict]:
    """One deterministic hard 1px noisy image for every digit."""
    images, labels, masks = make_ocr_1px(
        np.random.default_rng(int(seed) + 991), np.arange(10),
        res=int(resolution), hard_frac=1.0, hard_box=14,
    )
    samples = []
    for index in range(10):
        image = images[index]
        samples.append({
            "id": f"ocr1px-{index}",
            "dataset": "synthetic-ocr1px-fixed",
            "source_group": "ocr_1px",
            "task": "i2t",
            "prompt": "",
            "answer": str(int(labels[index].item())),
            "image": image,
            "target_rgb": image.clone(),
            "image_precision": 1.0,
            "text_missing": True,
            "need_pix": False,
            "need_text": True,
            "target_image_precision": 0.0,
            "target_text_precision": 1.0,
            "target_seg": masks[index],
        })
    return samples


def forward_batch(model, batch: dict) -> dict:
    return model.forward_tokens(
        batch["image"], batch["input_ids"], batch["attention_mask"],
        labels=batch["labels"],
        visual_prompt_mask=batch["visual_prompt_mask"],
        image_precision=batch["image_precision"],
        text_precision=batch["text_precision"],
    )


@torch.no_grad()
def teacher_metrics(model, tokenizer, samples, device, batch_size, max_tokens):
    model.eval()
    nlls, correct, totals = [], 0, 0
    for start in range(0, len(samples), batch_size):
        part = samples[start : start + batch_size]
        batch = move_batch(collate_real_multimodal(
            tokenizer, part, max_answer_tokens=max_tokens, append_eos=True,
        ), device)
        out = forward_batch(model, batch)
        labels = batch["labels"]
        n_vis = int(out["n_vis_tokens"])
        full = torch.cat([
            labels.new_full((labels.shape[0], n_vis), -100), labels,
        ], dim=1)
        prediction = out["token_logits"][:, :-1].argmax(dim=-1)
        target = full[:, 1:]
        valid = target.ne(-100)
        correct += int(((prediction == target) & valid).sum().item())
        totals += int(valid.sum().item())
        nlls.append(out["token_nll"].detach().cpu())
    return {
        "nll": float(torch.cat(nlls).mean()),
        "token_accuracy": float(correct / max(1, totals)),
        "tokens": totals,
    }


def target_ids(tokenizer, answer: str, max_tokens: int) -> list[int]:
    content = list(tokenizer.encode(" " + answer, add_special_tokens=False))
    content = content[: max(0, int(max_tokens) - 1)]
    return content + [int(tokenizer.eos_token_id)]


@torch.no_grad()
def graph_decode(
    model, tokenizer, samples, device, batch_size, max_tokens,
) -> list[dict]:
    """Batched greedy decode; each token reruns the complete graph."""
    model.eval()
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    if bos is None or tokenizer.eos_token_id is None:
        raise ValueError("tokenizer requires BOS/EOS for graph decode")
    rows = []
    for start in range(0, len(samples), batch_size):
        part = samples[start : start + batch_size]
        generated = [[] for _ in part]
        done = [False] * len(part)
        images = torch.stack([sample["image"] for sample in part]).to(device)
        for _ in range(int(max_tokens)):
            ids = torch.tensor(
                [[int(bos)] + row for row in generated], device=device,
            )
            out = model.forward_tokens(
                images, ids, torch.ones_like(ids), labels=None,
                visual_prompt_mask=torch.zeros_like(ids, dtype=torch.bool),
                image_precision=torch.ones(len(part), device=device),
                text_precision=torch.tensor(
                    [[0.0] + [1.0] * len(row) for row in generated],
                    device=device,
                ),
            )
            position = int(out["n_vis_tokens"]) + ids.shape[1] - 1
            next_ids = out["token_logits"][:, position].argmax(dim=-1).tolist()
            for index, token_id in enumerate(next_ids):
                token_id = int(token_id)
                generated[index].append(
                    int(tokenizer.eos_token_id) if done[index] else token_id
                )
                done[index] = done[index] or token_id == tokenizer.eos_token_id
            if all(done):
                break
        for sample, token_row in zip(part, generated):
            if tokenizer.eos_token_id in token_row:
                end = token_row.index(tokenizer.eos_token_id) + 1
                token_row = token_row[:end]
            expected = target_ids(tokenizer, sample["answer"], max_tokens)
            rows.append({
                "id": str(sample["id"]),
                "expected": str(sample["answer"]),
                "generated": tokenizer.decode(
                    token_row, skip_special_tokens=True,
                ).strip(),
                "token_exact": token_row == expected,
                "stopped": bool(token_row and token_row[-1] == tokenizer.eos_token_id),
            })
    return rows


def evaluate_overfit(model, tokenizer, samples, device, batch_size, max_tokens):
    teacher = teacher_metrics(
        model, tokenizer, samples, device, batch_size, max_tokens,
    )
    decoded = graph_decode(
        model, tokenizer, samples, device, batch_size, max_tokens,
    )
    rotated = [dict(sample) for sample in samples]
    for index, row in enumerate(rotated):
        source = samples[(index + 1) % len(samples)]
        row["image"] = source["image"].clone()
        row["answer"] = source["answer"]
    swapped = graph_decode(
        model, tokenizer, rotated, device, batch_size, max_tokens,
    )
    groups = {}
    for sample, row in zip(samples, decoded):
        group = str(sample.get("source_group", "real_i2t"))
        groups.setdefault(group, []).append(row)
    group_metrics = {
        group: {
            "n": len(rows),
            "graph_greedy_exact": sum(row["token_exact"] for row in rows),
            "stopped": sum(row["stopped"] for row in rows),
        }
        for group, rows in groups.items()
    }
    return {
        **teacher,
        "graph_greedy_exact": sum(row["token_exact"] for row in decoded),
        "graph_greedy_rate": float(
            sum(row["token_exact"] for row in decoded) / len(decoded)
        ),
        "stopped_rate": float(sum(row["stopped"] for row in decoded) / len(decoded)),
        "rotated_exact": sum(row["token_exact"] for row in swapped),
        "rotated_rate": float(
            sum(row["token_exact"] for row in swapped) / len(swapped)
        ),
        "examples": decoded[:8],
        "errors": [row for row in decoded if not row["token_exact"]],
        "rotated_errors": [row for row in swapped if not row["token_exact"]],
        "groups": group_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default=r"D:\ml_cache\sharegpt4o\pilot_manifest.json",
    )
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--accumulate", type=int, default=1,
        help="Micro-batches accumulated before one optimizer update.",
    )
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--max-words", type=int, default=12)
    parser.add_argument("--max-answer-tokens", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument(
        "--ocr-one-per-digit", action="store_true",
        help="Append ten fixed hard noisy 1px OCR samples to the real I2T bank.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--stop-on-pass", action=argparse.BooleanOptionalAction, default=True,
        help="Stop once original and rotated graph-greedy decode are both exact.",
    )
    parser.add_argument(
        "--resume-optimizer", action=argparse.BooleanOptionalAction, default=True,
        help="Restore Adam state when the input curriculum checkpoint contains it.",
    )
    parser.add_argument(
        "--hard-ids", default="",
        help="Comma-separated sample IDs for explicit hard-example replay.",
    )
    parser.add_argument(
        "--hard-per-batch", type=int, default=0,
        help="Number of hard examples placed in every micro-batch.",
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lm-device", default="cuda")
    parser.add_argument("--init", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    init_path = Path(args.init).resolve()
    candidate_path = Path(args.candidate).resolve()
    protected = {Path(path).resolve() for path in PROTECTED_CHECKPOINTS}
    if candidate_path in protected or candidate_path == init_path:
        raise SystemExit("refusing to overwrite protected or input checkpoint")
    init_hash = file_sha256(init_path)
    raw = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    records = [row for row in raw.get("records", []) if row.get("task") == "i2t"]
    if not 1 <= int(args.count) <= len(records):
        raise ValueError(f"--count must be in [1, {len(records)}]")
    order = np.random.default_rng(int(args.seed)).permutation(len(records))
    selected = []
    for index in order[: int(args.count)]:
        row = dict(records[int(index)])
        row["source_group"] = "real_i2t"
        row["answer"] = compact_answer(row["answer"], int(args.max_words))
        selected.append(materialize_record(row, int(args.resolution)))
    if bool(args.ocr_one_per_digit):
        selected.extend(make_fixed_ocr_samples(int(args.resolution), int(args.seed)))

    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    model = DualStreamOmni(**capability_champion_kwargs(
        res=int(args.resolution), language="pythia", lm_device=args.lm_device,
        pixel_loss_mode="gaussian_nll", s0_acc_coef=0.0,
    )).to(device)
    if model.lm is not None:
        model.lm.to(device)
    load_report = load_visual_champion(
        model, init_path, skip_language_interface=False,
    )
    trainable = set_optimization_phase(model, "token_reader")
    if any(parameter.requires_grad for parameter in model.lm.parameters()):
        raise RuntimeError("Pythia must remain frozen")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.lr), weight_decay=0.0,
    )
    if bool(args.resume_optimizer):
        raw_init = torch.load(init_path, map_location=device)
        optimizer_state = (
            raw_init.get("optimizer_state_dict")
            if isinstance(raw_init, dict) else None
        )
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            for group in optimizer.param_groups:
                group["lr"] = float(args.lr)
    rng = np.random.default_rng(int(args.seed) + int(args.count))
    requested_hard = {
        value.strip() for value in str(args.hard_ids).split(",") if value.strip()
    }
    hard_indices = [
        index for index, sample in enumerate(selected)
        if str(sample["id"]) in requested_hard
    ]
    if requested_hard != {str(selected[index]["id"]) for index in hard_indices}:
        raise ValueError("--hard-ids contains an ID outside the selected bank")
    hard_per_batch = int(args.hard_per_batch)
    if not 0 <= hard_per_batch < int(args.batch):
        raise ValueError("--hard-per-batch must be in [0, batch)")
    if hard_per_batch and not hard_indices:
        raise ValueError("--hard-per-batch requires --hard-ids")
    regular_indices = [
        index for index in range(len(selected)) if index not in set(hard_indices)
    ]
    if not regular_indices:
        regular_indices = list(range(len(selected)))
    epoch_order = rng.permutation(regular_indices).tolist()
    epoch_cursor = 0
    hard_cursor = 0

    def next_indices() -> list[int]:
        nonlocal epoch_order, epoch_cursor, hard_cursor
        indices = []
        for _ in range(hard_per_batch):
            indices.append(hard_indices[hard_cursor % len(hard_indices)])
            hard_cursor += 1
        while len(indices) < int(args.batch):
            available = len(epoch_order) - epoch_cursor
            take = min(int(args.batch) - len(indices), available)
            indices.extend(epoch_order[epoch_cursor : epoch_cursor + take])
            epoch_cursor += take
            if epoch_cursor == len(epoch_order):
                epoch_order = rng.permutation(regular_indices).tolist()
                epoch_cursor = 0
        return [int(index) for index in indices]

    history = []
    started = time.time()
    completed_step = 0
    best_score = None
    best_step = 0
    best_metrics = None
    best_state = None
    best_optimizer = None
    for step in range(int(args.steps) + 1):
        completed_step = step
        if step == 0 or step % int(args.eval_every) == 0 or step == int(args.steps):
            metrics = evaluate_overfit(
                model, model.lm_tok, selected, device, int(args.eval_batch),
                int(args.max_answer_tokens),
            )
            metrics["step"] = step
            history.append(metrics)
            score = (
                int(metrics["graph_greedy_exact"])
                + int(metrics["rotated_exact"]),
                float(metrics["token_accuracy"]),
                -float(metrics["nll"]),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_step = step
                best_metrics = copy.deepcopy(metrics)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.non_lm_state_dict().items()
                }
                best_optimizer = copy.deepcopy(optimizer.state_dict())
                for state in best_optimizer.get("state", {}).values():
                    for key, value in list(state.items()):
                        if torch.is_tensor(value):
                            state[key] = value.detach().cpu().clone()
            print(
                f"step={step:5d} nll={metrics['nll']:.4f} "
                f"token={metrics['token_accuracy']:.3f} "
                f"greedy={metrics['graph_greedy_exact']}/{len(selected)} "
                f"rotated={metrics['rotated_exact']}/{len(selected)}",
                flush=True,
            )
            if (
                bool(args.stop_on_pass)
                and step > 0
                and metrics["graph_greedy_exact"] == len(selected)
                and metrics["rotated_exact"] == len(selected)
            ):
                break
        if step == int(args.steps):
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(int(args.accumulate)):
            part = [selected[index] for index in next_indices()]
            batch = move_batch(collate_real_multimodal(
                model.lm_tok, part,
                max_answer_tokens=int(args.max_answer_tokens),
                append_eos=True,
            ), device)
            out = forward_batch(model, batch)
            loss = out["token_nll"].mean() / int(args.accumulate)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
        )
        optimizer.step()

    if file_sha256(init_path) != init_hash:
        raise RuntimeError("input checkpoint changed during curriculum run")
    if best_state is None or best_optimizer is None or best_metrics is None:
        raise RuntimeError("curriculum produced no evaluated state")
    model.load_state_dict(best_state, strict=False)
    record = {
        "schema": "sharegpt4o-i2t-curriculum-v1",
        "candidate_only": True,
        "admitted": False,
        "count": len(selected),
        "real_i2t_count": int(args.count),
        "ocr_1px_count": 10 if bool(args.ocr_one_per_digit) else 0,
        "steps": int(completed_step),
        "requested_steps": int(args.steps),
        "batch": int(args.batch),
        "accumulate": int(args.accumulate),
        "effective_batch": int(args.batch) * int(args.accumulate),
        "hard_ids": sorted(requested_hard),
        "hard_per_batch": hard_per_batch,
        "resolution": int(args.resolution),
        "max_words": int(args.max_words),
        "max_answer_tokens": int(args.max_answer_tokens),
        "answer_eos": True,
        "phase": "token_reader",
        "pythia_frozen": True,
        "lm_generate_called": False,
        "init": str(init_path),
        "init_sha256": init_hash,
        "candidate": str(candidate_path),
        "load_report": load_report,
        "trainable_count": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "trainable_names": trainable,
        "elapsed_seconds": time.time() - started,
        "selected_step": int(best_step),
        "selected_metrics": best_metrics,
        "history": history,
    }
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.non_lm_state_dict(),
        "language": model.language_meta(),
        "curriculum": {key: record[key] for key in (
            "schema", "count", "steps", "max_words", "max_answer_tokens",
        )},
        "candidate_only": True,
        "optimizer_state_dict": best_optimizer,
    }, candidate_path)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"wrote {args.out}; checkpoint={candidate_path}", flush=True)


if __name__ == "__main__":
    main()
