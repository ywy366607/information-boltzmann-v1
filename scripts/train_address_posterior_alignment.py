#!/usr/bin/env python3
"""Focused, same-graph T2I repair for prompt-conditioned Slice addresses.

Training observes the target image only through an amortized address posterior
q(W|target,H).  The deployed branch remains the usual blank-field forward
p(W|blank,H); its RGB likelihood and Deslice path are unchanged.  The KL is
therefore a training-time active-inference correction, not a target-image
shortcut at inference.
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
from fine_grain.pythia_bridge import capability_champion_kwargs
from fine_grain.spatial_address import (
    compute_address_compactness_loss,
    compute_address_diversity_loss,
    compute_address_kl_loss,
)
from scripts.train_pythia_capabilities import (
    _forward,
    collate_capability,
    eval_static_t2i,
    make_cycle_eval_bank,
    make_static_bank,
    sample_digit_group,
    static_one_step,
)
from scripts.train_unified_champion import configure_trainables, fix_write_gamma, save_candidate


def build(path: Path, device: torch.device) -> DualStreamOmni:
    model = DualStreamOmni(**capability_champion_kwargs(
        res=64, n_slices=64, language="pythia", lm_device=str(device),
        pixel_loss_mode="gaussian_nll", s0_acc_coef=0.0,
        deslice_write_sharpening=True, use_lang_address=True,
    )).to(device)
    if model.lm is not None:
        model.lm.to(device)
    raw = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(raw.get("state_dict", raw), strict=False)
    if unexpected or any(not name.startswith("lm.") for name in missing):
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:4]}, unexpected={unexpected[:4]}")
    fix_write_gamma(model, 8.0)
    configure_trainables(model, include_language_reader=True)
    return model


@torch.no_grad()
def target_address_posterior(model: DualStreamOmni, batch: dict) -> list[torch.Tensor]:
    """q(W|o,H), obtained by observing o in the same full-resolution graph."""
    device = next(model.parameters()).device
    n = len(batch["prompt"])
    target = batch["target_rgb"].to(device)
    task = batch.get("task_id")
    out = model(
        target, list(batch["prompt"]), t=torch.zeros(n, device=device),
        image_precision=torch.ones(n, device=device),
        text_precision=batch["text_precision"].to(device),
        target_time=batch["target_time"].to(device),
        task_id=task.to(device) if task is not None else None,
    )
    return [w.detach() for w in out["address_w"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(ROOT / "checkpoints" / "_unified_u1_candidate.pt"))
    parser.add_argument("--candidate", default=str(ROOT / "checkpoints" / "_u1_address_alignment_candidate.pt"))
    parser.add_argument("--out", default=str(ROOT / "results" / "published" / "u1_address_alignment.json"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--address-kl", type=float, default=1.0)
    parser.add_argument("--compactness", type=float, default=0.15)
    parser.add_argument("--diversity", type=float, default=0.02)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    source, candidate = Path(args.source).resolve(), Path(args.candidate).resolve()
    if source == candidate:
        raise ValueError("candidate must not overwrite source")
    device = torch.device(args.device)
    torch.manual_seed(20260906)
    rng = np.random.default_rng(20260906)
    model = build(source, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=0.0)
    bank = make_static_bank(64, "text_to_both")
    history = []
    for step in range(1, int(args.steps) + 1):
        samples = sample_digit_group(bank, rng)
        optimizer.zero_grad(set_to_none=True)
        # Existing T2I RGB/mask likelihood and digit prompt counterfactual.
        task_loss, meta = static_one_step(
            model, samples, bank, rng, digit_group_coef=0.1,
            foreground_bce_coef=0.5,
        )
        batch = collate_capability(samples)
        posterior = target_address_posterior(model, batch)
        prior_out = _forward(model, batch)
        prior = prior_out["address_w"]
        # Align the *deployed* Deslice address, not the pre-sharpening read
        # assignment.  Gamma/top-k/mass preservation can radically alter W;
        # comparing q(W_read) to p(W_read) would otherwise optimize an object
        # that never writes to X. q stays detached, so targets are training
        # evidence only and no gradient can create an inference shortcut.
        posterior_write = [
            layer.deslice._write_w(q) for q, layer in zip(posterior, model.mot_stack.layers)
        ]
        prior_write = [
            layer.deslice._write_w(p) for p, layer in zip(prior, model.mot_stack.layers)
        ]
        kl = sum(compute_address_kl_loss(q, p) for q, p in zip(posterior_write, prior_write)) / len(prior)
        compact = sum(compute_address_compactness_loss(p)["address_compactness_loss"] for p in prior_write) / len(prior)
        diversity = sum(compute_address_diversity_loss(p)["address_diversity_loss"] for p in prior_write) / len(prior)
        loss = task_loss + float(args.address_kl) * kl + float(args.compactness) * compact + float(args.diversity) * diversity
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        row = {
            "step": step, "loss": float(loss.detach()), "task_loss": float(task_loss.detach()),
            "address_kl": float(kl.detach()), "compactness": float(compact.detach()),
            "diversity": float(diversity.detach()),
        }
        if step % int(args.eval_every) == 0 or step == int(args.steps):
            metrics = eval_static_t2i(model.eval(), make_cycle_eval_bank(64, "text_to_both"), device, chunk=8)
            row["t2i"] = metrics
            model.train()
            print(json.dumps(row), flush=True)
        history.append(row)
    final = eval_static_t2i(model.eval(), make_cycle_eval_bank(64, "text_to_both"), device, chunk=8)
    record = {
        "schema": "u1-address-posterior-alignment-candidate",
        "admitted": False,
        "source": str(source),
        "run": {"steps": args.steps, "lr": args.lr, "address_kl": args.address_kl,
                "compactness": args.compactness, "diversity": args.diversity,
                "frozen_pythia": True, "same_slice_deslice_graph": True,
                "address_space": "deployed_post_gamma_write_weights"},
        "history": history, "final_t2i": final,
        "decision_rule": "Promote neither this candidate nor recurrence unless reload improves paired IoU and address locality without a prompt-control collapse.",
    }
    save_candidate(model, candidate, record)
    Path(args.out).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"final_t2i": final, "candidate": str(candidate)}, indent=2))


if __name__ == "__main__":
    main()
