#!/usr/bin/env python3
"""Pre-registered H1--H4 probe from docs/ADDRESS_FAILURE_HARNESS.md.

This is diagnostic only: the oracle sees the target to measure whether the
already-deployed Slice/Deslice write maps span a sparse stroke.  It never
produces a generative sample or saves a trainable model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_u1_language_addresses import address_stats, build
from scripts.train_pythia_capabilities import _forward, collate_capability, make_static_bank


def binary_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = pred.amax(dim=-1) > 0.2
    y = target.amax(dim=-1) > 0.2
    return float((p & y).sum() / (p | y).sum().clamp_min(1))


def oracle(write_maps: list[torch.Tensor], target: torch.Tensor) -> dict:
    """Least-squares RGB fit through frozen address columns, for capacity only."""
    design = torch.cat([w[0].float() for w in write_maps], dim=-1)
    y = target[0].permute(1, 2, 0).reshape(-1, 3).float()
    coeff = torch.linalg.lstsq(design, y).solution
    pred = (design @ coeff).clamp(0.0, 1.0)
    return {
        "columns": int(design.shape[1]),
        "mse": float((pred - y).square().mean()),
        "stroke_iou": binary_iou(pred, y),
    }


@torch.no_grad()
def target_forward(model, batch: dict):
    device = next(model.parameters()).device
    target = batch["target_rgb"].to(device)
    count = target.shape[0]
    return model(
        target, list(batch["prompt"]), t=torch.zeros(count, device=device),
        image_precision=torch.ones(count, device=device),
        text_precision=batch["text_precision"].to(device),
        target_time=batch["target_time"].to(device),
        task_id=batch.get("task_id", None).to(device) if batch.get("task_id", None) is not None else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default=str(ROOT / "checkpoints" / "_u1_address_alignment_candidate.pt"))
    parser.add_argument("--out", default=str(ROOT / "results" / "published" / "address_teaching_signal_probe.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = build(Path(args.candidate), device)
    sample = next(
        row for row in make_static_bank(64, "text_to_both")
        if row["digit"] == "7" and row["target_color"] == "red" and row["source_place"] == "middle_center"
    )
    batch = collate_capability([sample])
    blank = _forward(model, batch)
    observed = target_forward(model, batch)
    target = batch["target_rgb"].to(device)
    stroke = target.amax(dim=1).reshape(-1) > 0.2
    layers = []
    pre_blank, post_blank = [], []
    for index, (w_blank, w_obs, layer) in enumerate(zip(
        blank["address_w"], observed["address_w"], model.mot_stack.layers,
    )):
        wb, wo = w_blank.detach(), w_obs.detach()
        wb_write = layer.deslice._write_w(wb)
        wo_write = layer.deslice._write_w(wo)
        point_delta = (wo - wb).norm(dim=-1)[0]
        fg = point_delta[stroke].mean()
        bg = point_delta[~stroke].mean().clamp_min(1e-8)
        layers.append({
            "layer": index,
            "pre_blank": address_stats(wb[0], 64),
            "post_blank": address_stats(wb_write[0], 64),
            "target_blank_relative_difference": float(torch.norm(wo - wb) / torch.norm(wb).clamp_min(1e-8)),
            "foreground_to_background_assignment_change": float(fg / bg),
        })
        pre_blank.append(wb)
        post_blank.append(wb_write)
    result = {
        "schema": "address-teaching-signal-h1-h4-probe",
        "prediction": "P-20260906-01",
        "candidate": str(Path(args.candidate).resolve()),
        "sample": {"digit": sample["digit"], "color": sample["target_color"], "place": sample["source_place"]},
        "layers": layers,
        "pre_write_oracle": oracle(pre_blank, target),
        "post_write_oracle": oracle(post_blank, target),
        "free_pixel_oracle": {"mse": 0.0, "stroke_iou": 1.0},
        "interpretation_rule": "H1: tiny target/blank change and fg/bg ratio near 1. H2: low post-write oracle IoU. H3: high post-write oracle and selective target response. H4: post-write sharply improves locality/oracle over pre-write.",
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
