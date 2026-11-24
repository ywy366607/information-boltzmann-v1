"""Read recorded response shadows; compare residual transport with factual copying.

Uses event vectors only. It restores no body and performs no world interaction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def event(path: Path, cycle: int) -> dict:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("cycle") == cycle and row.get("causal_program", {}).get(
                "counterfactual_response_shadow"
            ):
                return row
    raise ValueError(f"No response shadow at cycle {cycle}: {path}")


def mse(left: list, right: list) -> float:
    if not left or len(left) != len(right):
        raise ValueError("Matched nonempty vectors required")
    return sum((a - b) ** 2 for a, b in zip(left, right)) / len(left)


def audit(paths: list[Path], cycle: int) -> dict:
    rows = [event(path, cycle) for path in paths]
    signatures = [row["shared_unknown"]["signature"] for row in rows]
    if not signatures[0] or signatures[0] != signatures[1]:
        raise ValueError("Different question signatures")
    if rows[0]["joint_language_search"]["candidates"] != rows[1][
        "joint_language_search"
    ]["candidates"]:
        raise ValueError("Different candidate menus")
    shadows = [row["causal_program"]["counterfactual_response_shadow"] for row in rows]
    chosen = [shadow["chosen"] for shadow in shadows]
    if chosen[0] == chosen[1]:
        raise ValueError("Two distinct actions required")
    alternatives = [
        next(item for item in shadows[i]["alternatives"]
             if item["option"] == chosen[1 - i])
        for i in range(2)
    ]
    observed = [shadow["observed"] for shadow in shadows]
    residual = [[t - p for t, p in zip(item["transported"], item["prior"])]
                for item in alternatives]
    factual_mean = [[y - e for y, e in zip(observed[i], residual[i])]
                    for i in range(2)]
    agreement = [max(abs(p - q) for p, q in zip(
        alternatives[i]["prior"], factual_mean[1 - i])) for i in range(2)]
    if max(agreement) > 2e-6:
        raise ValueError("Frozen predictions differ between branches")
    dimension = len(observed[0])
    if any(len(v) != dimension for v in observed + residual + factual_mean):
        raise ValueError("Inconsistent vector dimensions")
    a, b = [sum(x * x for x in vector) / dimension for vector in residual]
    c = sum(x * y for x, y in zip(*residual)) / dimension
    checks = []
    for i in range(2):
        target = observed[1 - i]
        contrast = [p - q for p, q in zip(
            alternatives[i]["prior"], factual_mean[i])]
        realized = [y - x for x, y in zip(observed[i], target)]
        transport_error = mse(alternatives[i]["transported"], target)
        contrast_error = mse(contrast, realized)
        if abs(transport_error - contrast_error) > 1e-12:
            raise ValueError("Transport/contrast identity reconstruction failed")
        checks.append(dict(
            observed=rows[i]["choice"], alternative=rows[1 - i]["choice"],
            mse=dict(prior=mse(alternatives[i]["prior"], target),
                     copy_factual=mse(observed[i], target),
                     full_residual=transport_error,
                     predicted_vs_realized_contrast=contrast_error),
        ))
    return dict(
        cycle=cycle,
        scope="same recorded question/menu and frozen means; post-observation inference",
        inputs=[dict(path=str(path.resolve()),
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                for path in paths],
        world_starts=[row["world"]["start"] for row in rows],
        question_signature=signatures[0], dimension=dimension,
        prior_agreement_max=agreement,
        residual_cosine=c / math.sqrt(a * b) if a * b > 0 else None,
        checks=checks, world_calls=0, optimizer_updates=0,
        interpretation="One action pair; reverse transport errors share one algebraic contrast.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path, nargs=2)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.events, args.cycle)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                      allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
