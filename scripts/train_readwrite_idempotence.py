#!/usr/bin/env python3
"""Latent round-trip idempotence training gate (N104 follow-up).

The Transolver lesson applied to our stack: slice read assignments become
content-faithful only when a loss demands it, and no objective ever asked
W(read(X)) to carry canvas content. This gate trains ONLY the per-layer
SliceRead and Deslice projection with the registered idempotence objective

    W_l(read_l(X))  ~=  X - X_blank

on two field manifolds (stem-encoded perception fields and frozen
champion-generated fields), then audits whether round-trip fidelity rises
WITHOUT regressing the protected static T2I / current / edit gates.
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
from fine_grain.pythia_bridge import load_visual_champion
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
from scripts.train_pythia_capabilities import (
    make_static_bank,
    static_one_step,
)
from fine_grain.capability_tasks import capability_sample
from fine_grain.omni_tasks import GRID_PLACES
from fine_grain.vlm_data import COLORS, OCR_DIGITS

DEFAULT_INIT = (
    ROOT / "checkpoints" /
    "omni_d64_pythia_capability_b3_edit_best.pt"
)


@torch.no_grad()
def decode(model, X: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(model.decode_field(X)).clamp(0.0, 1.0)


@torch.no_grad()
def roundtrip_psnr(model, X: torch.Tensor) -> dict:
    """Per-layer W(read(X)) round-trip PSNR against decode(X)."""
    target = decode(model, X)
    blank = torch.zeros_like(X)
    out = {}
    for index, layer in enumerate(model.mot_stack.layers):
        S, w = layer.read(X)
        X_rt = blank + layer.deslice.write_delta(S, w)
        mse = (decode(model, X_rt) - target).pow(2).mean()
        out[f"layer{index}"] = float(
            -10.0 * torch.log10(mse.clamp_min(1e-12))
        )
    return out


def idempotence_loss(model, X: torch.Tensor, X_blank: torch.Tensor) -> dict:
    """Summed per-layer || W(read(X)) - (X - X_blank) ||^2."""
    content = X - X_blank
    total = X.new_zeros(())
    per_layer = {}
    for index, layer in enumerate(model.mot_stack.layers):
        S, w = layer.read(X)
        delta = layer.deslice.write_delta(S, w)
        loss_l = (delta - content).pow(2).mean()
        per_layer[f"layer{index}"] = float(loss_l.detach())
        total = total + loss_l
    return total / len(model.mot_stack.layers), per_layer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default=str(DEFAULT_INIT))
    parser.add_argument("--candidate", default=str(
        ROOT / "checkpoints" / "omni_d64_readwrite_idempotence_candidate.pt",
    ))
    parser.add_argument("--out", default=str(
        ROOT / "results" / "published" / "readwrite_idempotence_gate.json",
    ))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--gen-frac", type=float, default=0.25,
                        help="Fraction of each batch drawn from frozen generation fields.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-limit", type=int, default=90)
    parser.add_argument(
        "--rehearsal-frac", type=float, default=0.0,
        help=(
            "Fraction of training steps that take a static capability "
            "rehearsal update (T2I/current/edit) alongside idempotence."
        ),
    )
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    init_path = Path(args.init)
    candidate_path = Path(args.candidate)
    protected = {Path(p).resolve() for p in PROTECTED_CHECKPOINTS}
    if candidate_path.resolve() in protected or candidate_path.resolve() == init_path.resolve():
        raise SystemExit("refusing to overwrite a protected or input checkpoint")
    if not init_path.is_file():
        raise FileNotFoundError(init_path)

    device = torch.device(args.device)
    lm_device = args.device
    torch.manual_seed(0)
    from fine_grain.pythia_bridge import capability_champion_kwargs
    model = DualStreamOmni(**capability_champion_kwargs(
        language="pythia", lm_device=lm_device,
    )).to(device)
    if model.lm is not None:
        model.lm.to(device)
    load_report = load_visual_champion(model, init_path, skip_language_interface=False)
    trainable = model.set_optimization_phase("readwrite_idempotence")
    assert trainable, "idempotence phase opened nothing"
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=float(args.lr),
    )
    res = int(model.res)

    # Perception fields: stem is frozen, encode the whole bank once.
    from fine_grain.capability_tasks import _scene
    from fine_grain.omni_tasks import GRID_PLACES
    from fine_grain.vlm_data import COLORS, OCR_DIGITS
    scenes = []
    for place in GRID_PLACES:
        for digit in OCR_DIGITS:
            for color in COLORS:
                rgb, _ = _scene(digit, color, place, res)
                scenes.append((str(digit), str(color), str(place), rgb))
    imgs = torch.cat([s[3] for s in scenes], dim=0).to(device)
    with torch.no_grad():
        perc_fields = model.mot_stack.encode_X(
            imgs, image_precision=torch.ones(imgs.shape[0], device=device),
        )
    prompts_all = [
        f"Draw digit {s[0]} with a thin {s[1]} stroke at {s[2].replace('_', ' ')}"
        for s in scenes
    ]

    # Generation fields: frozen snapshot from the initial champion.
    with torch.no_grad():
        zeros = torch.zeros(len(prompts_all), 3, res, res, device=device)
        gen_fields = model(
            zeros, prompts_all, pi_x=1.0,
            t=torch.zeros(len(prompts_all), device=device),
            image_precision=torch.zeros(len(prompts_all), device=device),
            text_precision=torch.ones(len(prompts_all), device=device),
        )["belief_mu"].detach()
    with torch.no_grad():
        blank_img = torch.zeros(1, 3, res, res, device=device)
        X_blank = model.mot_stack.encode_X(
            blank_img, image_precision=torch.ones(1, device=device),
        )

    static_banks = {
        case: make_cycle_eval_bank(res, case)
        for case in ("text_to_both", "image_to_current", "image_text_edit")
    }

    def static_gates():
        t2i = eval_static_t2i(
            model, static_banks["text_to_both"], device, chunk=30,
        )
        current = eval_current(
            model, static_banks["image_to_current"], device, chunk=30,
        )
        edit = eval_edit_static(
            model, static_banks["image_text_edit"], device, chunk=30,
        )
        gates = {
            "t2i": t2i_gate(t2i),
            "current": current_gate(current),
            "edit": edit_gate(edit),
        }
        return {"t2i": t2i, "current": current, "edit": edit}, gates

    started = time.time()
    history = []
    best_score = float("-inf")

    def evaluate(step: int):
        nonlocal best_score
        rt_perc = roundtrip_psnr(model, perc_fields[:256])
        rt_gen = roundtrip_psnr(model, gen_fields[:256])
        statics, gates = static_gates()
        rt_min = min(
            min(rt_perc.values()), min(rt_gen.values()),
        )
        row = {
            "step": int(step),
            "rt_perception": rt_perc,
            "rt_generation": rt_gen,
            "rt_min": rt_min,
            "static": {
                k: {
                    "digit_top1": statics[k].get("digit_top1"),
                    "paired_iou": statics[k].get("paired_iou"),
                }
                for k in statics
            },
            "gates": gates,
            "all_static_held": all(gates.values()),
        }
        history.append(row)
        print(
            f"step={step:5d} rt_min={rt_min:.2f}dB "
            f"perc=[{rt_perc['layer0']:.1f} {rt_perc['layer1']:.1f} "
            f"{rt_perc['layer2']:.1f} {rt_perc['layer3']:.1f}] "
            f"gen=[{rt_gen['layer0']:.1f} {rt_gen['layer1']:.1f} "
            f"{rt_gen['layer2']:.1f} {rt_gen['layer3']:.1f}] "
            f"gates={''.join(str(int(v)) for v in gates.values())}",
            flush=True,
        )
        score = rt_min + 10.0 * sum(gates.values())
        if score > best_score:
            best_score = score
            if not args.eval_only:
                candidate_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": model.state_dict(),
                    "step": step,
                    "metrics": row,
                    "candidate_only": True,
                }, candidate_path)
        return row

    rehearsal_banks = {}
    if float(args.rehearsal_frac) > 0.0:
        for case in ("text_to_both", "image_to_current", "image_text_edit"):
            rehearsal_banks[case] = make_static_bank(res, case)

    evaluate(0)
    if not args.eval_only:
        rng = np.random.default_rng(0)
        n_gen = max(1, int(args.batch * float(args.gen_frac)))
        n_perc = int(args.batch) - n_gen
        for step in range(1, int(args.steps) + 1):
            perc_idx = rng.choice(len(perc_fields), size=n_perc, replace=False)
            gen_idx = rng.choice(len(gen_fields), size=n_gen, replace=False)
            X = torch.cat([
                perc_fields[torch.from_numpy(perc_idx).to(device)],
                gen_fields[torch.from_numpy(gen_idx).to(device)],
            ], dim=0)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, _ = idempotence_loss(model, X, X_blank)
            if rehearsal_banks and rng.random() < float(args.rehearsal_frac):
                case = ("text_to_both", "image_to_current", "image_text_edit")[
                    int(rng.integers(0, 3))
                ]
                group_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
                if case == "text_to_both":
                    digit = str(rng.choice(OCR_DIGITS))
                    color = str(rng.choice(list(COLORS)))
                    place = str(rng.choice(GRID_PLACES))
                    samples = [capability_sample(
                        group_rng, res, case, digit, color, place,
                    )]
                else:
                    digit = str(rng.choice(OCR_DIGITS))
                    place = str(rng.choice(GRID_PLACES))
                    color = str(rng.choice(list(COLORS)))
                    samples = [capability_sample(
                        group_rng, res, case, digit, color, place,
                    )]
                replay_loss, _ = static_one_step(
                    model, samples, rehearsal_banks[case], rng,
                )
                loss = loss + replay_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()
            if step % int(args.eval_every) == 0 or step == int(args.steps):
                evaluate(step)

    record = {
        "schema": "readwrite-idempotence-gate",
        "init": str(init_path),
        "init_report": load_report,
        "candidate": str(candidate_path),
        "run": {
            "steps": int(args.steps),
            "batch": int(args.batch),
            "lr": float(args.lr),
            "gen_frac": float(args.gen_frac),
            "rehearsal_frac": float(args.rehearsal_frac),
            "objective": "sum_l || W_l(read_l(X)) - (X - X_blank) ||^2",
            "trainable": trainable,
            "n_trainable": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        },
        "history": history,
        "final": history[-1],
        "elapsed_sec": time.time() - started,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    final = history[-1]
    print(
        f"wrote {out_path}; rt_min {history[0]['rt_min']:.2f} -> "
        f"{final['rt_min']:.2f} dB; static_held={final['all_static_held']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
