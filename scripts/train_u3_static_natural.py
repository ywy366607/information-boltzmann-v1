#!/usr/bin/env python3
"""U3: 64px static-T2I chart closure under a natural-generation constraint.

The loss is deliberately still one X--Slice--H graph.  It pairs a
foreground-balanced *view of the same RGB observation* with ten-way
observation-energy contrast, then projects only static gradients that oppose
the natural T2I gradient.  This is a candidate-only diagnosis, not a new
architecture or an admission shortcut.
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

from fine_grain.omni_model import DualStreamOmni
from fine_grain.pythia_bridge import capability_champion_kwargs, param_groups
from fine_grain.sharegpt4o_data import collate_real_multimodal
from scripts.train_pythia_capabilities import (
    current_gate,
    edit_gate,
    eval_current,
    eval_edit_static,
    eval_static_t2i,
    make_cycle_eval_bank,
    make_static_bank,
    sample_digit_group,
    static_one_step,
    t2i_gate,
)
from scripts.train_sharegpt4o_t2i_overfit import evaluate as evaluate_natural
from scripts.train_unified_champion import (
    NATURAL_BASE,
    TOKEN_BASE,
    configure_trainables,
    copy_terminal_language_reader,
    file_sha256,
    fix_write_gamma,
    load_visual_champion,
    move,
    natural_loss,
    save_candidate,
    save_gallery,
    select_t2i_records,
)


def _grads(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> list[torch.Tensor | None]:
    return list(torch.autograd.grad(loss, parameters, allow_unused=True))


def merge_static_with_natural(
    natural: list[torch.Tensor | None], static: list[torch.Tensor | None],
    parameters: list[torch.nn.Parameter], static_coef: float,
) -> dict:
    """Preserve the natural first-order descent direction parameter-by-parameter."""
    conflict, dot_sum, natural_sq, static_sq = 0, 0.0, 0.0, 0.0
    for parameter, gn, gs in zip(parameters, natural, static):
        if gn is None and gs is None:
            parameter.grad = None
            continue
        gn = torch.zeros_like(parameter) if gn is None else gn
        gs = torch.zeros_like(parameter) if gs is None else gs
        dot = (gn * gs).sum()
        gnsq = gn.square().sum().clamp_min(1e-12)
        if dot < 0:
            gs = gs - dot / gnsq * gn
            conflict += 1
        parameter.grad = gn + float(static_coef) * gs
        dot_sum += float(dot.detach())
        natural_sq += float(gn.square().sum().detach())
        static_sq += float(gs.square().sum().detach())
    return {
        "conflicting_tensors": conflict,
        "raw_dot": dot_sum,
        "natural_norm": natural_sq ** 0.5,
        "projected_static_norm": static_sq ** 0.5,
    }


@torch.no_grad()
def static_report(model, device, batch: int) -> dict:
    rows = {
        "t2i": eval_static_t2i(model, make_cycle_eval_bank(64, "text_to_both"), device, chunk=batch),
        "current": eval_current(model, make_cycle_eval_bank(64, "image_to_current"), device, chunk=batch),
        "edit": eval_edit_static(model, make_cycle_eval_bank(64, "image_text_edit"), device, chunk=batch),
    }
    return {"metrics": rows, "gates": {"t2i": t2i_gate(rows["t2i"]),
                                           "current": current_gate(rows["current"]),
                                           "edit": edit_gate(rows["edit"])}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=r"D:\ml_cache\sharegpt4o\pilot_manifest.json")
    parser.add_argument("--candidate", default=str(ROOT / "checkpoints" / "_u3_static_natural_candidate.pt"))
    parser.add_argument("--out", default=str(ROOT / "results" / "published" / "u3_static_natural.json"))
    parser.add_argument("--gallery", default=str(ROOT / "present" / "figs" / "u3_static_natural.png"))
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--interface-lr", type=float, default=3e-5)
    parser.add_argument("--visual-lr", type=float, default=3e-6)
    parser.add_argument("--static-coef", type=float, default=1.0)
    parser.add_argument("--foreground-bce-coef", type=float, default=4.0)
    parser.add_argument("--digit-group-coef", type=float, default=4.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lm-device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(20260904)
    rng = np.random.default_rng(20260904)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    natural_rows = select_t2i_records(
        manifest, ["freedom-t2i-34407", "freedom-t2i-3191"], 64,
    )
    model = DualStreamOmni(**capability_champion_kwargs(
        res=64, n_slices=64, language="pythia", lm_device=args.lm_device or args.device,
        pixel_loss_mode="gaussian_nll", s0_acc_coef=0.0, deslice_write_sharpening=True,
    )).to(device)
    if model.lm is not None:
        model.lm.to(device)
    natural_load = load_visual_champion(model, NATURAL_BASE, skip_language_interface=False)
    token_overlay = copy_terminal_language_reader(model, TOKEN_BASE)
    fix_write_gamma(model, 8.0)
    trainable_names = configure_trainables(model, include_language_reader=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        param_groups(model, interface_lr=args.interface_lr, visual_lr=args.visual_lr), weight_decay=0.0,
    )
    natural_batch = move(collate_real_multimodal(model.lm_tok, natural_rows), device)
    static_bank = make_static_bank(64, "text_to_both")
    history, started = [], time.time()
    for step in range(1, args.steps + 1):
        model.train()
        natural_value, natural_meta = natural_loss(
            model, natural_batch, len(natural_rows),
            type("NaturalArgs", (), {
                "mse_coef": 4.0, "nll_coef": 0.0, "edge_coef": 8.0,
                "retrieval_coef": 0.5, "retrieval_temperature": 0.02,
                "shuffle_coef": 1.0, "shuffle_margin": 0.02,
            })(),
        )
        natural_grad = _grads(natural_value, parameters)
        samples = sample_digit_group(static_bank, rng)
        static_value, static_meta = static_one_step(
            model, samples, static_bank, rng,
            digit_group_coef=args.digit_group_coef,
            foreground_bce_coef=args.foreground_bce_coef,
        )
        static_grad = _grads(static_value, parameters)
        optimizer.zero_grad(set_to_none=True)
        steering = merge_static_with_natural(
            natural_grad, static_grad, parameters, args.static_coef,
        )
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        row = {"step": step, "natural_loss": float(natural_value.detach()),
               "static_loss": float(static_value.detach()), "natural": natural_meta,
               "static": static_meta, "steering": steering,
               "elapsed_sec": time.time() - started}
        history.append(row)
        if step % args.eval_every == 0 or step == args.steps:
            natural_metrics, _ = evaluate_natural(model, model.lm_tok, natural_rows, device)
            static = static_report(model, device, args.batch)
            row["natural_gate"] = natural_metrics
            row["static_report"] = static
            save_candidate(model, Path(args.candidate), {
                "schema": "u3-foreground-static-with-natural-gradient-constraint",
                "admitted": False, "candidate_only": True, "step": step,
                "natural": natural_metrics, "static": static, "history": history,
            })
            print(f"step={step:04d} natural={natural_metrics['passed']} "
                  f"edge={natural_metrics['edge_correlation']:.3f} "
                  f"static_digit={static['metrics']['t2i']['digit_top1']:.3f}", flush=True)

    natural_metrics, prediction = evaluate_natural(model, model.lm_tok, natural_rows, device)
    static = static_report(model, device, args.batch)
    record = {
        "schema": "u3-foreground-static-with-natural-gradient-constraint",
        "admitted": False, "candidate_only": True,
        "sources": {"natural": {"path": str(NATURAL_BASE), "sha256": file_sha256(NATURAL_BASE)},
                    "token": {"path": str(TOKEN_BASE), "sha256": file_sha256(TOKEN_BASE)},
        },
        "natural_load": natural_load, "token_overlay": token_overlay,
        "run": {"resolution": 64, "n_slices": 64, "write_gamma": 8.0,
                "foreground_bce_coef": args.foreground_bce_coef,
                "digit_group_coef": args.digit_group_coef,
                "natural_first_order_constraint": "project conflicting static gradient per parameter",
                "pythia_frozen": True, "lm_generate_called": False,
                "trainable_names": trainable_names},
        "natural": natural_metrics, "static": static, "history": history,
    }
    save_candidate(model, Path(args.candidate), record)
    save_gallery(natural_rows, prediction, Path(args.gallery))
    Path(args.out).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {args.out}; candidate_only=true", flush=True)


if __name__ == "__main__":
    main()
