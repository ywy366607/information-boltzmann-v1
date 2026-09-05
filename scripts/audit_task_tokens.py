#!/usr/bin/env python3
"""Audit whether semantic task TTokens are functionally distinguished."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.capability_tasks import CAPABILITY_CASES, CAPABILITY_TASK_IDS, fixed_capability_bank
from fine_grain.omni_model import DualStreamOmni
from scripts.train_northstar_capabilities import evaluate_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "checkpoints" / "northstar_static32_unified_task_best.pt"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "results" / "published" / "northstar_task_token_audit.json"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk", type=int, default=24)
    args = parser.parse_args()

    raw = torch.load(args.checkpoint, map_location="cpu")
    model = DualStreamOmni(**raw["config"])
    model.load_state_dict(raw["state_dict"])
    device = torch.device(args.device)
    model.to(device).eval()
    if not bool(getattr(model.mot_stack, "use_task_tokens", False)):
        raise RuntimeError("checkpoint does not enable task tokens")

    weight = model.mot_stack.task_embed.weight.detach().float().cpu()
    normalized = F.normalize(weight, dim=-1)
    cosine = normalized @ normalized.T
    distance = torch.cdist(weight, weight)

    bank = fixed_capability_bank(model.res)
    score_matrix: dict[str, dict[str, float]] = {}
    metric_matrix: dict[str, dict[str, dict]] = {}
    margins: dict[str, float] = {}
    for source_case in CAPABILITY_CASES:
        samples = [sample for sample in bank if sample["case"] == source_case]
        score_matrix[source_case] = {}
        metric_matrix[source_case] = {}
        for token_case in CAPABILITY_CASES:
            token_id = CAPABILITY_TASK_IDS[token_case]
            swapped = [{**sample, "task_id": token_id} for sample in samples]
            metrics = evaluate_samples(
                model, swapped, device, chunk=args.chunk,
            )[source_case]
            score_matrix[source_case][token_case] = float(metrics["score"])
            metric_matrix[source_case][token_case] = metrics
        correct = score_matrix[source_case][source_case]
        best_wrong = max(
            value for case, value in score_matrix[source_case].items()
            if case != source_case
        )
        margins[source_case] = float(correct - best_wrong)

    names = list(CAPABILITY_CASES)
    report = {
        "schema": "northstar-task-token-swap-audit-v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "task_ids": CAPABILITY_TASK_IDS,
        "embedding_norms": {
            name: float(weight[index].norm()) for index, name in enumerate(names)
        },
        "embedding_cosine": {
            row: {column: float(cosine[i, j]) for j, column in enumerate(names)}
            for i, row in enumerate(names)
        },
        "embedding_distance": {
            row: {column: float(distance[i, j]) for j, column in enumerate(names)}
            for i, row in enumerate(names)
        },
        "score_matrix": score_matrix,
        "correct_minus_best_wrong": margins,
        "functionally_identified": {
            case: margin > 0.0 for case, margin in margins.items()
        },
        "metrics": metric_matrix,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "embedding_cosine": report["embedding_cosine"],
        "score_matrix": score_matrix,
        "correct_minus_best_wrong": margins,
        "functionally_identified": report["functionally_identified"],
    }, indent=2), flush=True)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
