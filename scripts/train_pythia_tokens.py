#!/usr/bin/env python3
"""Stage C/D: causal token NLL and graph-native decode on the B3 checkpoint.

Only terminal text/Slice-to-LM interfaces train. The frozen Pythia decoder is
never called through ``generate``. Static RGB/segmentation dynamics therefore
remain the B3 graph, while I2T and next-color IT2T must beat image shuffles.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.omni_model import DualStreamOmni
from fine_grain.pythia_bridge import capability_champion_kwargs, param_groups
from fine_grain.token_tasks import (
    TOKEN_CASES,
    answer_class_nll,
    answer_token_accuracy,
    collate_token_capabilities,
    counterfactual_token_samples,
    graph_greedy_decode,
    move_token_batch,
)
from scripts.train_pythia_capabilities import (
    PROTECTED_CHECKPOINTS,
    current_gate,
    edit_gate,
    eval_current,
    eval_edit_static,
    eval_static_t2i,
    make_cycle_eval_bank,
    make_static_bank,
)
from scripts.train_pythia_generation import t2i_gate
from fine_grain.omni_tasks import GRID_PLACES
from fine_grain.vlm_data import COLORS, OCR_DIGITS


DEFAULT_INIT = ROOT / "checkpoints" / "omni_d64_pythia_capability_b3_edit_best.pt"
DEFAULT_CKPT = ROOT / "checkpoints" / "omni_d64_pythia_token_best.pt"


def initialize_terminal_token_chart(
    model,
    tokenizer,
    samples: list[dict],
    device: torch.device,
    *,
    prototype_steps: int = 120,
    prototype_lr: float = 0.05,
    ridge: float = 1e-2,
    chunk: int = 32,
) -> dict:
    """Align fixed-atlas X features to frozen-Pythia next-token messages.

    The optimized digit messages are discarded after a ridge initialization
    of ``terminal_atlas_to_text``; no class table or inference bypass remains.
    """
    connector = model.mot_stack.terminal_atlas_to_text
    if connector is None:
        raise RuntimeError("terminal token chart requires terminal_token_atlas")
    with torch.no_grad():
        existing = float(connector.weight.detach().norm())
    if existing != 0.0:
        return {"applied": False, "reason": "connector already initialized", "norm": existing}
    answer_words = list(OCR_DIGITS)
    target_ids = []
    for word in answer_words:
        ids = tokenizer.encode(" " + str(word), add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"token chart requires one-token answer {word!r}: {ids}")
        target_ids.append(int(ids[0]))
    target = torch.tensor(target_ids, dtype=torch.long, device=device)
    lm_dtype = model.lm.get_input_embeddings().weight.dtype
    prototypes = torch.nn.Parameter(torch.zeros(
        len(answer_words), model.d_llm, device=device, dtype=torch.float32,
    ))
    optimizer = torch.optim.Adam([prototypes], lr=float(prototype_lr))
    visual = torch.zeros(
        len(answer_words), model.mot_stack.n_slices, model.d_llm,
        device=device, dtype=lm_dtype,
    )
    attention = torch.ones(
        len(answer_words), model.mot_stack.n_slices + 1,
        device=device, dtype=torch.long,
    )
    model.lm.eval()
    for _ in range(int(prototype_steps)):
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.cat([visual, prototypes.to(lm_dtype).unsqueeze(1)], dim=1)
        logits = model.lm(
            inputs_embeds=inputs, attention_mask=attention, use_cache=False,
        ).logits[:, -1].float()
        loss = F.cross_entropy(logits, target) + 1e-5 * prototypes.pow(2).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        inputs = torch.cat([visual, prototypes.to(lm_dtype).unsqueeze(1)], dim=1)
        proto_logits = model.lm(
            inputs_embeds=inputs, attention_mask=attention, use_cache=False,
        ).logits[:, -1].float()
        proto_acc = float(proto_logits.argmax(dim=-1).eq(target).float().mean())
        proto_nll = float(F.cross_entropy(proto_logits, target))

    features, target_messages = [], []
    model.eval()
    for start in range(0, len(samples), int(chunk)):
        part = samples[start : start + int(chunk)]
        images = torch.cat([sample["image"] for sample in part], dim=0).to(device)
        batch = len(part)
        bos = int(tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id)
        ids = torch.full((batch, 1), bos, dtype=torch.long, device=device)
        emb = model.lm.get_input_embeddings()(ids)
        with torch.no_grad():
            X, _, _, _ = model.mot_stack.forward_native(
                img=images,
                text_emb=emb.float(),
                text_mask=torch.ones(batch, 1, dtype=torch.bool, device=device),
                prompt_mask=torch.zeros(batch, 1, dtype=torch.bool, device=device),
                image_precision=torch.ones(batch, device=device),
                text_precision=torch.zeros(batch, 1, device=device),
            )
            side = int(round(model.mot_stack.n_slices ** 0.5))
            atlas = F.adaptive_avg_pool2d(
                X.transpose(1, 2).reshape(
                    batch, model.d_model, model.res, model.res,
                ),
                (side, side),
            ).flatten(1)
        features.append(atlas.cpu().float())
        index = torch.tensor(
            [answer_words.index(str(sample["answer"])) for sample in part],
            device=device,
        )
        target_messages.append(prototypes.detach().index_select(0, index).cpu())
    x = torch.cat(features)
    y = torch.cat(target_messages)
    xa = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
    # Dual ridge is stable here because examples (360) < atlas width (1024).
    gram = xa @ xa.T
    solve = torch.linalg.solve(
        gram + float(ridge) * torch.eye(gram.shape[0]), y,
    )
    theta = xa.T @ solve
    with torch.no_grad():
        connector.weight.copy_(theta[:-1].T.to(
            device=connector.weight.device, dtype=connector.weight.dtype,
        ))
        connector.bias.copy_(theta[-1].to(
            device=connector.bias.device, dtype=connector.bias.dtype,
        ))
    return {
        "applied": True,
        "prototype_steps": int(prototype_steps),
        "prototype_lr": float(prototype_lr),
        "prototype_accuracy": proto_acc,
        "prototype_nll": proto_nll,
        "ridge": float(ridge),
        "fit_mse": float((xa @ theta - y).pow(2).mean()),
        "samples": len(samples),
    }


def _forward_token_batch(model, batch: dict, device: torch.device) -> dict:
    batch = move_token_batch(batch, device)
    out = model.forward_tokens(
        batch["image"],
        batch["input_ids"],
        batch["attention_mask"],
        labels=batch["labels"],
        visual_prompt_mask=batch["visual_prompt_mask"],
        image_precision=batch["image_precision"],
        text_precision=batch["text_precision"],
    )
    return out


def _prompt_counterfactual(samples: list[dict]) -> list[dict]:
    changed = []
    for sample in samples:
        item = dict(sample)
        digit = str(sample["digit"])
        other = str((int(digit) + 1) % 10)
        item["prompt"] = str(sample["prompt"]).replace(
            f"digit {digit}", f"digit {other}", 1,
        )
        changed.append(item)
    return changed


def sample_identified_group(
    bank: list[dict], case: str, rng: np.random.Generator,
) -> list[dict]:
    """Hold nuisance factors fixed while covering the answer classes."""
    place = str(rng.choice(GRID_PLACES))
    if case in ("text_to_both", "image_to_current"):
        color = str(rng.choice(COLORS))
        group = []
        for digit in OCR_DIGITS:
            matches = [
                sample for sample in bank
                if sample["source_place"] == place
                and sample["source_color"] == color
                and sample["digit"] == digit
            ]
            group.append(matches[0])
        return group
    if case == "image_text_edit":
        digit = str(rng.choice(OCR_DIGITS))
        group = []
        for color in COLORS:
            matches = [
                sample for sample in bank
                if sample["source_place"] == place
                and sample["source_color"] == color
                and sample["digit"] == digit
            ]
            group.append(matches[0])
        return group
    raise ValueError(case)


@torch.no_grad()
def evaluate_token_case(
    model,
    tokenizer,
    samples: list[dict],
    control_bank: list[dict],
    device: torch.device,
    chunk: int,
) -> dict:
    model.eval()
    matched_nll, shuffled_nll, accuracy, records = [], [], [], []
    for start in range(0, len(samples), int(chunk)):
        part = samples[start : start + int(chunk)]
        batch = move_token_batch(
            collate_token_capabilities(tokenizer, part), device,
        )
        out = _forward_token_batch(model, batch, device)
        matched_nll.append(out["token_nll"].detach().cpu())
        part_accuracy = answer_token_accuracy(out, batch).detach().cpu()
        accuracy.append(part_accuracy)
        if part[0]["case"] == "text_to_both":
            controls = _prompt_counterfactual(part)
        else:
            controls = counterfactual_token_samples(part, control_bank)
        control_batch = move_token_batch(
            collate_token_capabilities(tokenizer, controls), device,
        )
        control_out = _forward_token_batch(model, control_batch, device)
        part_matched = out["token_nll"].detach().cpu()
        part_shuffled = control_out["token_nll"].detach().cpu()
        shuffled_nll.append(part_shuffled)
        part_gap = part_shuffled - part_matched
        for position, sample in enumerate(part):
            records.append({
                "digit": str(sample["digit"]),
                "place": str(sample["source_place"]),
                "color": str(sample["source_color"]),
                "answer": str(sample["answer"]),
                "accuracy": float(part_accuracy[position]),
                "matched_nll": float(part_matched[position]),
                "gap": float(part_gap[position]),
            })
    matched = torch.cat(matched_nll)
    shuffled = torch.cat(shuffled_nll)
    gap = shuffled - matched
    acc = torch.cat(accuracy)
    return {
        "n": int(matched.numel()),
        "nll": float(matched.mean()),
        "shuffled_nll": float(shuffled.mean()),
        "mean_gap": float(gap.mean()),
        "median_gap": float(gap.median()),
        "token_accuracy": float(acc.mean()),
        "records": records,
    }


def hard_cells_from_records(records: list[dict]) -> list[tuple[str, str, str]]:
    """Deterministic sorted cells whose teacher-forced answer was wrong."""
    cells = {
        (str(record["digit"]), str(record["place"]), str(record["color"]))
        for record in records
        if float(record["accuracy"]) < 1.0
    }
    return sorted(cells)


def pick_hard_samples(
    bank: list[dict], cells: list[tuple[str, str, str]], count: int, cursor: int,
) -> list[dict]:
    """Round-robin train-bank rows matching failing cells, cursor-resumable.

    The fixed audit bank is a registered subset of the training bank, so
    replaying failing cells stays inside the finite-bank capability protocol.
    """
    if count <= 0 or not cells:
        return []
    picked = []
    for offset in range(int(count)):
        digit, place, color = cells[(int(cursor) + offset) % len(cells)]
        matches = [
            sample for sample in bank
            if str(sample["digit"]) == digit
            and str(sample["source_place"]) == place
            and str(sample["source_color"]) == color
        ]
        if not matches:
            raise ValueError(f"train bank lacks hard cell {digit}/{place}/{color}")
        picked.append(matches[0])
    return picked


def decode_summaries(samples: list[dict], rows: list[dict]) -> tuple[list[dict], dict]:
    """Compact per-case decode table plus expected-to-pred confusion counts."""
    results = [
        {
            "digit": str(sample["digit"]),
            "place": str(sample["source_place"]),
            "color": str(sample["source_color"]),
            "expected": row["expected"],
            "text": row["text"],
            "exact": bool(row["exact"]),
        }
        for sample, row in zip(samples, rows)
    ]
    confusion: dict[str, dict[str, int]] = {}
    for item in results:
        slot = confusion.setdefault(str(item["expected"]), {})
        slot[str(item["text"])] = slot.get(str(item["text"]), 0) + 1
    return results, confusion


def token_gate(metrics: dict, *, require_image: bool) -> bool:
    return bool(
        metrics["token_accuracy"] >= 0.80
        and (
            not require_image
            or metrics["median_gap"] >= 0.50
        )
    )


@torch.no_grad()
def evaluate_graph_decode(
    model,
    tokenizer,
    samples: list[dict],
    limit: int,
) -> dict:
    chosen = samples if int(limit) < 0 else samples[: min(int(limit), len(samples))]
    rows = [graph_greedy_decode(model, tokenizer, sample) for sample in chosen]
    results, confusion = decode_summaries(chosen, rows)
    return {
        "n": len(rows),
        "exact": float(sum(row["exact"] for row in rows) / max(1, len(rows))),
        "all_steps_rerun_graph": all(
            all(step["graph_rerun"] for step in row["steps"]) for row in rows
        ),
        "examples": rows[:8],
        "results": results,
        "confusion": confusion,
    }


def _save(model, path: Path, payload: dict, optimizer=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "state_dict": model.non_lm_state_dict(),
        "language": model.language_meta(),
        **payload,
    }
    if optimizer is not None:
        raw = optimizer.state_dict()
        # state_dict() entries alias the live optimizer state; copy out
        # per-entry so the training optimizer stays on its device.
        state = {
            index: {
                key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                for key, value in entry.items()
            }
            for index, entry in raw["state"].items()
        }
        record["optimizer_state_dict"] = {
            "state": state,
            "param_groups": raw["param_groups"],
        }
    torch.save(record, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default=str(DEFAULT_INIT))
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument(
        "--candidate",
        default=str(ROOT / "checkpoints" / "omni_d64_pythia_token_candidate.pt"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "results" / "published" / "pythia_token_nll.json"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lm-device", default=None)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--initial-proj-trust", type=float, default=0.0,
        help="Open the read-only visual-token residual if its saved gate is exactly zero.",
    )
    parser.add_argument(
        "--prototype-init", action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument("--prototype-steps", type=int, default=120)
    parser.add_argument("--prototype-lr", type=float, default=0.05)
    parser.add_argument("--prototype-ridge", type=float, default=1e-2)
    parser.add_argument(
        "--phase", choices=("token_interface", "token_reader"),
        default="token_reader",
    )
    parser.add_argument(
        "--terminal-atlas", action=argparse.BooleanOptionalAction, default=False,
        help="Experimental fixed 4x4 terminal Slice likelihood interface.",
    )
    parser.add_argument("--shuffle-coef", type=float, default=1.0)
    parser.add_argument("--shuffle-margin", type=float, default=0.5)
    parser.add_argument(
        "--class-contrast-coef", type=float, default=1.0,
        help="Answer-set NLL on the same frozen Pythia logits; adds no head.",
    )
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument(
        "--it2t-start", type=int, default=300,
        help="First next-color step; earlier steps alternate T2T and I2T only.",
    )
    parser.add_argument(
        "--i2t-warmup", type=int, default=100,
        help="Initial image-only batches that force the terminal Slice reader to form.",
    )
    parser.add_argument(
        "--case-cycle",
        choices=("default", "i2t_heavy"),
        default="default",
        help=(
            "Post-it2t rehearsal cycle: default = 2xIT2T,T2T,I2T; "
            "i2t_heavy = 2xI2T,T2T,IT2T."
        ),
    )
    parser.add_argument(
        "--hard-per-batch", type=int, default=0,
        help=(
            "Replay this many train-bank rows from failing audited cells in "
            "every step of the matching case (0 disables)."
        ),
    )
    parser.add_argument(
        "--resume-optimizer", action="store_true",
        help="Restore Adam state from the init checkpoint when present.",
    )
    parser.add_argument("--decode-limit", type=int, default=30,
                        help="Greedy decode samples per case; -1 decodes the full bank.")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    init_path = Path(args.init)
    ckpt_path = Path(args.ckpt)
    candidate_path = Path(args.candidate)
    protected = {path.resolve() for path in PROTECTED_CHECKPOINTS}
    protected.add(DEFAULT_INIT.resolve())
    if not args.eval_only and ckpt_path.resolve() in protected:
        raise SystemExit(f"refusing to overwrite protected checkpoint {ckpt_path}")
    if not init_path.is_file():
        raise FileNotFoundError(init_path)

    device = torch.device(args.device)
    lm_device = args.lm_device or args.device
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = DualStreamOmni(**capability_champion_kwargs(
        language="pythia", lm_device=lm_device,
        terminal_token_atlas=bool(args.terminal_atlas),
    )).to(device)
    if model.lm is not None:
        model.lm.to(device)
    report = model.load_visual_champion(
        init_path, skip_language_interface=False,
    )
    initial_proj_gate = float(model.mot_stack.proj_gate.detach())
    if initial_proj_gate == 0.0 and float(args.initial_proj_trust) != 0.0:
        with torch.no_grad():
            model.mot_stack.proj_gate.fill_(float(args.initial_proj_trust))
    trainable = model.set_optimization_phase(args.phase)
    optimizer = torch.optim.AdamW(
        param_groups(model, interface_lr=args.lr, visual_lr=args.lr),
        weight_decay=1e-4,
    )
    resumed_optimizer = False
    if args.resume_optimizer:
        raw_init = torch.load(init_path, map_location=device)
        optimizer_state = (
            raw_init.get("optimizer_state_dict")
            if isinstance(raw_init, dict) else None
        )
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            for group in optimizer.param_groups:
                group["lr"] = float(args.lr)
            resumed_optimizer = True
    tokenizer = model.lm_tok
    train_banks = {
        case: make_static_bank(model.res, case) for case in TOKEN_CASES
    }
    eval_banks = {
        case: make_cycle_eval_bank(model.res, case) for case in TOKEN_CASES
    }
    chart_init = None
    if args.prototype_init:
        chart_init = initialize_terminal_token_chart(
            model,
            tokenizer,
            train_banks["image_to_current"],
            device,
            prototype_steps=args.prototype_steps,
            prototype_lr=args.prototype_lr,
            ridge=args.prototype_ridge,
            chunk=args.batch,
        )

    # Terminal token interfaces do not feed X, but verify the protected B3
    # checkpoint before admitting any token result.
    static_before = {
        "t2i": eval_static_t2i(
            model, eval_banks["text_to_both"], device, chunk=args.batch,
        ),
        "current": eval_current(
            model, eval_banks["image_to_current"], device, chunk=args.batch,
        ),
        "edit": eval_edit_static(
            model, eval_banks["image_text_edit"], device, chunk=args.batch,
        ),
    }
    static_gates = {
        "t2i": t2i_gate(static_before["t2i"]),
        "current": current_gate(static_before["current"]),
        "edit": edit_gate(static_before["edit"]),
    }
    if not all(static_gates.values()):
        raise RuntimeError(f"B3 initialization failed static gates: {static_gates}")

    started = time.time()
    history = []
    best_score = float("-inf")
    best_step = 0
    admitted = False
    hard_cells = {case: [] for case in TOKEN_CASES}
    hard_cursor = {case: 0 for case in TOKEN_CASES}
    case_cycle = (
        ("image_to_current", "image_to_current", "text_to_both", "image_text_edit")
        if args.case_cycle == "i2t_heavy"
        else ("image_text_edit", "image_text_edit", "text_to_both", "image_to_current")
    )

    def evaluate(step: int, train_meta: dict | None = None):
        nonlocal best_score, best_step, admitted
        token = {
            case: evaluate_token_case(
                model, tokenizer, eval_banks[case], train_banks[case],
                device, args.batch,
            )
            for case in TOKEN_CASES
        }
        for case in TOKEN_CASES:
            hard_cells[case] = hard_cells_from_records(token[case]["records"])
        gates = {
            "t2t": token_gate(token["text_to_both"], require_image=False),
            "i2t": token_gate(token["image_to_current"], require_image=True),
            "it2t": token_gate(token["image_text_edit"], require_image=True),
        }
        # Once the causal gap clears its 0.5-nat gate, extra margin must not
        # outrank actual token accuracy when selecting a continuation state.
        cap_gap = lambda value: max(0.0, min(0.5, float(value)))
        if step <= int(args.i2t_warmup):
            score = token["image_to_current"]["token_accuracy"]
            score += cap_gap(token["image_to_current"]["median_gap"])
        elif step <= int(args.it2t_start):
            score = token["text_to_both"]["token_accuracy"]
            score += token["image_to_current"]["token_accuracy"]
            score += cap_gap(token["image_to_current"]["median_gap"])
        else:
            score = sum(value["token_accuracy"] for value in token.values())
            score += cap_gap(token["image_to_current"]["median_gap"])
            score += cap_gap(token["image_text_edit"]["median_gap"])
        row = {
            "step": int(step),
            "train": train_meta or {},
            "token": token,
            "gates": gates,
            "score": float(score),
            "gate_values": {
                "proj_gate": float(model.mot_stack.proj_gate.detach()),
                "text_out_gate": float(model.mot_stack.text_out_gate.detach()),
            },
            "hard_cells": {
                case: len(hard_cells[case]) for case in TOKEN_CASES
            },
            "elapsed_sec": time.time() - started,
        }
        history.append(row)
        if not args.eval_only and score > best_score:
            best_score, best_step = float(score), int(step)
            _save(model, candidate_path, {
                "step": step, "token": token, "gates": gates,
                "candidate_only": True,
            }, optimizer)
        if not args.eval_only and all(gates.values()) and score > best_score - 1e-12:
            admitted = True
            _save(model, ckpt_path, {
                "step": step, "token": token, "gates": gates,
                "static_gates": static_gates,
            }, optimizer)
        print(
            f"step={step:4d} "
            + " ".join(
                f"{case}={token[case]['token_accuracy']:.3f}/"
                f"{token[case]['median_gap']:.3f}"
                for case in TOKEN_CASES
            )
            + f" gates={''.join(str(int(x)) for x in gates.values())}",
            flush=True,
        )
        return row

    evaluate(0)
    if not args.eval_only:
        for step in range(1, int(args.steps) + 1):
            if step <= int(args.i2t_warmup):
                case = "image_to_current"
            elif step <= int(args.it2t_start):
                case = ("text_to_both", "image_to_current")[(step - 1) % 2]
            else:
                # Add IT2T only after the basic language/image readers have a
                # chance to form; the cycle preset sets rehearsal pressure.
                cycle = case_cycle
                case = cycle[(step - int(args.it2t_start) - 1) % len(cycle)]
            bank = train_banks[case]
            samples = sample_identified_group(bank, case, rng)
            hard_rows = pick_hard_samples(
                bank, hard_cells[case], int(args.hard_per_batch),
                hard_cursor[case],
            )
            if hard_rows:
                hard_cursor[case] = (
                    hard_cursor[case] + len(hard_rows)
                ) % max(1, len(hard_cells[case]))
            batch = move_token_batch(
                collate_token_capabilities(tokenizer, samples + hard_rows), device,
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            out = _forward_token_batch(model, batch, device)
            nll = out["token_nll"]
            class_nll = answer_class_nll(
                out, batch, rows=range(len(samples)),
            )
            loss = nll.mean() + float(args.class_contrast_coef) * class_nll
            meta = {
                "case": case,
                "hard_rows": len(hard_rows),
                "matched_nll": float(nll.mean().detach()),
                "class_nll": float(class_nll.detach()),
            }
            if case != "text_to_both":
                controls = counterfactual_token_samples(samples + hard_rows, bank)
                control_batch = move_token_batch(
                    collate_token_capabilities(tokenizer, controls), device,
                )
                control_out = _forward_token_batch(model, control_batch, device)
                shuffled = control_out["token_nll"]
                causal = torch.relu(
                    nll - shuffled + float(args.shuffle_margin)
                ).mean()
                loss = loss + float(args.shuffle_coef) * causal
                meta.update({
                    "shuffled_nll": float(shuffled.mean().detach()),
                    "causal_hinge": float(causal.detach()),
                })
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                1.0,
            )
            optimizer.step()
            meta["loss"] = float(loss.detach())
            if step % int(args.eval_every) == 0 or step == int(args.steps):
                evaluate(step, meta)

    selected = init_path
    if admitted and ckpt_path.is_file():
        selected = ckpt_path
    elif not args.eval_only and candidate_path.is_file():
        selected = candidate_path
    if selected != init_path:
        model.load_visual_champion(selected, skip_language_interface=False)

    final_token = {
        case: evaluate_token_case(
            model, tokenizer, eval_banks[case], train_banks[case],
            device, args.batch,
        )
        for case in TOKEN_CASES
    }
    decode = {
        case: evaluate_graph_decode(
            model, tokenizer, eval_banks[case], args.decode_limit,
        )
        for case in TOKEN_CASES
    }
    decode_gates = {
        case: value["exact"] >= 0.80 and value["all_steps_rerun_graph"]
        for case, value in decode.items()
    }
    final_gates = {
        "t2t": token_gate(final_token["text_to_both"], require_image=False),
        "i2t": token_gate(final_token["image_to_current"], require_image=True),
        "it2t": token_gate(final_token["image_text_edit"], require_image=True),
        "decode": all(decode_gates.values()),
        "static": all(static_gates.values()),
    }
    # A provisional teacher-forcing checkpoint is not called the champion
    # until real graph decode also passes.
    final_admitted = bool(all(final_gates.values()) and selected == ckpt_path)
    record = {
        "schema": "pythia-stage-cd-token-nll-graph-decode",
        "init": str(init_path),
        "init_report": report,
        "selected_checkpoint": str(selected),
        "champion_checkpoint": str(ckpt_path),
        "admitted": final_admitted,
        "best_step": best_step,
        "best_score": None if best_score == float("-inf") else best_score,
        "language": model.language_meta(),
        "terminal_chart_init": chart_init,
        "run": {
            "phase": args.phase,
            "terminal_atlas": bool(args.terminal_atlas),
            "n_trainable": sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "trainable_names": trainable,
            "steps": int(args.steps),
            "batch": int(args.batch),
            "lr": float(args.lr),
            "shuffle_margin_nat_per_token": float(args.shuffle_margin),
            "shuffle_coef": float(args.shuffle_coef),
            "class_contrast_coef": float(args.class_contrast_coef),
            "initial_proj_gate_saved": initial_proj_gate,
            "initial_proj_trust": float(args.initial_proj_trust),
            "it2t_start": int(args.it2t_start),
            "i2t_warmup": int(args.i2t_warmup),
            "case_cycle": args.case_cycle,
            "hard_per_batch": int(args.hard_per_batch),
            "resumed_optimizer": resumed_optimizer,
            "identified_groups": {
                "t2t_i2t": "ten digits at fixed color/address",
                "it2t": "four source colors at fixed digit/address",
            },
            "pythia_frozen": all(
                not parameter.requires_grad for parameter in model.lm.parameters()
            ),
            "lm_generate_called": False,
            "gdn2": False,
        },
        "static_before": static_before,
        "static_gates": static_gates,
        "token": final_token,
        "decode": decode,
        "decode_gates": decode_gates,
        "final_gates": final_gates,
        "history": history,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {out_path}; admitted={final_admitted}", flush=True)


if __name__ == "__main__":
    main()
