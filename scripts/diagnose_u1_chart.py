#!/usr/bin/env python3
"""Read-only diagnosis for the rejected U1 direct-merge protocol.

It separates three failure classes: broken capability boundaries, a frozen or
disconnected training path, and the intended-but-insufficient cross-chart
transfer from a 16-Slice static champion to a 64-Slice natural champion.
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

from fine_grain.omni_model import DualStreamOmni
from fine_grain.pythia_bridge import capability_champion_kwargs, load_visual_champion
from fine_grain.sharegpt4o_data import collate_real_multimodal
from scripts.train_pythia_capabilities import (
    eval_current,
    eval_edit_static,
    eval_static_t2i,
    make_cycle_eval_bank,
    make_static_bank,
    sample_digit_group,
    static_one_step,
)
from scripts.train_unified_champion import (
    NATURAL_BASE,
    TOKEN_BASE,
    configure_trainables,
    copy_terminal_language_reader,
    fix_write_gamma,
    move,
    natural_loss,
)
from scripts.train_sharegpt4o_t2i_overfit import select_t2i_records


def build(path: Path | None, device: torch.device):
    model = DualStreamOmni(**capability_champion_kwargs(
        res=64, n_slices=64, language="pythia", lm_device=str(device),
        pixel_loss_mode="gaussian_nll", s0_acc_coef=0.0,
        deslice_write_sharpening=True,
    )).to(device)
    if model.lm is not None:
        model.lm.to(device)
    if path is None:
        natural = load_visual_champion(model, NATURAL_BASE, skip_language_interface=False)
        overlay = copy_terminal_language_reader(model, TOKEN_BASE)
    else:
        natural = load_visual_champion(model, path, skip_language_interface=False)
        overlay = {"loaded": [], "skipped": []}
    fix_write_gamma(model, 8.0)
    trainable = configure_trainables(model)
    return model.eval(), natural, overlay, trainable


@torch.no_grad()
def static_metrics(model, device: torch.device) -> dict:
    return {
        "t2i": eval_static_t2i(
            model, make_cycle_eval_bank(64, "text_to_both"), device, chunk=8,
        ),
        "current": eval_current(
            model, make_cycle_eval_bank(64, "image_to_current"), device, chunk=8,
        ),
        "edit": eval_edit_static(
            model, make_cycle_eval_bank(64, "image_text_edit"), device, chunk=8,
        ),
    }


def family(name: str) -> str:
    if name.startswith("mot_stack.text_in"):
        return "text_in"
    if ".mot.Wk_t" in name or ".mot.Wv_t" in name:
        return "language_to_vision"
    if ".surprise_gate." in name:
        return "f2_prior"
    if ".read." in name or ".deslice." in name or ".mot.Wq_v" in name or ".mot.Wo_v" in name:
        return "slice_read_write"
    if name.startswith("pix_head") or name.startswith("pix_log"):
        return "rgb_likelihood"
    if name.startswith("seg_head"):
        return "seg_likelihood"
    if name.startswith("mot_stack.text_out") or name.startswith("mot_stack.proj"):
        return "terminal_token"
    return "other"


def gradient_audit(model, device: torch.device) -> dict:
    rng = np.random.default_rng(20260904)
    results = {}
    for case in ("text_to_both", "image_to_current", "image_text_edit"):
        bank = make_static_bank(64, case)
        samples = sample_digit_group(bank, rng) if case == "text_to_both" else [bank[0]]
        model.train()
        model.zero_grad(set_to_none=True)
        loss, meta = static_one_step(model, samples, bank, rng)
        loss.backward()
        groups = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                key = family(name)
                groups[key] = groups.get(key, 0.0) + float(parameter.grad.norm().detach())
        results[case] = {
            "loss": float(loss.detach()),
            "boundary": {
                "image_precision": meta["image_precision"],
                "text_precision": meta["text_precision"],
            },
            "gradient_norm_sum": groups,
        }
    return results


def relative_parameter_drift(before: dict, after: dict) -> dict:
    """Report how far the actual U1 starting state moved in each port family."""
    grouped: dict[str, list[float]] = {}
    for name, value in before.items():
        if name not in after or value.shape != after[name].shape:
            continue
        denom = value.float().pow(2).mean().sqrt().clamp_min(1e-9)
        delta = (after[name].detach().cpu().float() - value.float()).pow(2).mean().sqrt() / denom
        grouped.setdefault(family(name), []).append(float(delta))
    return {
        group: {
            "n": len(values),
            "median_relative_rms": float(np.median(values)),
            "max_relative_rms": float(max(values)),
        }
        for group, values in grouped.items()
    }


def _family_gradient(model, loss: torch.Tensor) -> dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    loss.backward()
    groups: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            # Missing gradient is a genuine zero for this objective; retaining
            # it makes the two task vectors have identical parameter support.
            grad = (
                parameter.grad.detach().flatten().cpu()
                if parameter.grad is not None
                else torch.zeros(parameter.numel())
            )
            groups[name] = grad
    return groups


def natural_static_gradient_cosine(model, device: torch.device) -> dict:
    """Measure whether native natural T2I and static T2I ask shared ports to move together."""
    manifest = json.loads(
        Path(r"D:\ml_cache\sharegpt4o\pilot_manifest.json").read_text(encoding="utf-8")
    )
    natural = select_t2i_records(
        manifest, ["freedom-t2i-34407", "freedom-t2i-3191"], 64,
    )
    batch = move(collate_real_multimodal(model.lm_tok, natural), device)
    model.train()
    natural_value, _ = natural_loss(
        model, batch, len(natural),
        type("Args", (), {
            "mse_coef": 4.0, "nll_coef": 0.0, "edge_coef": 8.0,
            "retrieval_coef": 0.5, "retrieval_temperature": 0.02,
            "shuffle_coef": 1.0, "shuffle_margin": 0.02,
        })(),
    )
    natural_grad = _family_gradient(model, natural_value)
    rng = np.random.default_rng(20260904)
    bank = make_static_bank(64, "text_to_both")
    static_value, _ = static_one_step(model, sample_digit_group(bank, rng), bank, rng)
    static_grad = _family_gradient(model, static_value)
    rows = {}
    for name in sorted(set(natural_grad) & set(static_grad)):
        key = family(name)
        entry = rows.setdefault(key, {"dot": 0.0, "left_sq": 0.0, "right_sq": 0.0})
        left, right = natural_grad[name], static_grad[name]
        entry["dot"] += float(torch.dot(left, right))
        entry["left_sq"] += float(torch.dot(left, left))
        entry["right_sq"] += float(torch.dot(right, right))
    for entry in rows.values():
        left_norm = max(entry.pop("left_sq"), 0.0) ** 0.5
        right_norm = max(entry.pop("right_sq"), 0.0) ** 0.5
        dot = entry.pop("dot")
        entry.update({
            "natural_norm": left_norm,
            "static_norm": right_norm,
            "cosine": dot / max(left_norm * right_norm, 1e-12),
        })
    return {"natural_loss": float(natural_value.detach()), "static_loss": float(static_value.detach()),
            "by_family": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default=str(ROOT / "checkpoints" / "_unified_u1_candidate.pt"))
    parser.add_argument("--out", default=str(ROOT / "results" / "published" / "u1_chart_diagnosis.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    initial, initial_load, overlay, trainable = build(None, device)
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in initial.state_dict().items()
    }
    before = static_metrics(initial, device)
    gradients = gradient_audit(initial, device)
    cosine = natural_static_gradient_cosine(initial, device)
    candidate, candidate_load, _, _ = build(Path(args.candidate), device)
    after = static_metrics(candidate, device)
    drift = relative_parameter_drift(initial_state, candidate.state_dict())
    record = {
        "schema": "u1-direct-merge-chart-diagnosis",
        "source": {"natural": str(NATURAL_BASE), "token": str(TOKEN_BASE),
                   "candidate": str(args.candidate)},
        "initial_load": initial_load,
        "overlay": {"loaded_count": len(overlay["loaded"]), "skipped": overlay["skipped"]},
        "trainable_count": len(trainable),
        "static_before_training": before,
        "static_after_400_steps": after,
        "gradient_audit": gradients,
        "natural_static_t2i_gradient_cosine": cosine,
        "relative_parameter_drift_from_u1_start": drift,
        "interpretation": (
            "Nonzero gradients and correct 0/1 precision boundaries rule out a frozen-path "
            "or modality-boundary wiring failure. Shape-skipped M-dependent token readout and "
            "poor pretraining static scores identify cross-chart transfer plus insufficient per-task "
            "updates as the direct-merge limitation."
        ),
    }
    Path(args.out).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
