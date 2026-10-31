"""Audit a no-new-state C/S teaching direction on real OpenWebText bytes.

The candidate comes from ``CGBColumnLanguageBody.contact_teaching_direction``:
the already learned language-to-C value map projects the arrived innovation
into the column width, then contracts it with the old routed C/S trace.  This
program never applies that direction.  It compares it to the exact gradient
of the *next* byte score through the actual two-event H/C/S transition.

The retained graph is only the two events used to produce one numerical
reference.  The online body remains detached between audited pairs; this is
an admission measurement, not BPTT or an alternative training algorithm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core.cgb_column_body import CGBColumnLanguageBody
from run_cgb_column_language_admission import (  # same registered data contract
    BYTE_OFFSET,
    CudaGraphValueCredit,
    load_split,
)


def _cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return (first * second).sum() / (first.norm() * second.norm()).clamp_min(1e-12)


def _summary(cosines: list[float], normalized_dots: list[float], positive: int) -> dict:
    return {
        "mean_cosine": sum(cosines) / len(cosines),
        "positive_dot_fraction": positive / len(cosines),
        "mean_exact_normalized_dot": sum(normalized_dots) / len(normalized_dots),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "ib_owt_10000")
    parser.add_argument("--warmup-bytes", type=int, default=100_000)
    parser.add_argument("--audit-pairs", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--active-columns", type=int, default=8)
    parser.add_argument("--contact-rank", type=int, default=4)
    parser.add_argument("--head-step", type=float, default=0.01)
    parser.add_argument("--value-step", type=float, default=0.001)
    parser.add_argument("--graph-block", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "published" / "cgb_column_value_teaching_alignment_owt_20261101.json",
    )
    args = parser.parse_args()
    if min(args.warmup_bytes, args.audit_pairs, args.d_model, args.grid, args.active_columns,
           args.contact_rank, args.graph_block) < 1:
        raise ValueError("all count arguments must be positive")
    if args.head_step < 0 or args.value_step < 0:
        raise ValueError("learning step sizes must be nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")

    train, train_source = load_split(args.data_root, "train", args.warmup_bytes)
    validation, validation_source = load_split(args.data_root, "validation", max(args.audit_pairs * 4, 4096))
    torch.manual_seed(args.seed)
    body = CGBColumnLanguageBody(
        args.d_model, (args.grid, args.grid), active_columns=args.active_columns,
        contact_rank=args.contact_rank,
    ).to(device).eval()

    # This uses exactly the established G14 + direct C-value learner to make
    # the candidate's existing value map meaningful.  The candidate itself is
    # frozen throughout the audit below.
    runner = CudaGraphValueCredit(
        body, body.initial_state(1, device=device), args.head_step,
        args.value_step, args.graph_block,
    )
    warmup = runner.phase(train, report_every=max(args.warmup_bytes, 1))
    state = runner.state()

    cosines: list[float] = []
    normalized_dots: list[float] = []
    null_cosines: list[float] = []
    null_normalized_dots: list[float] = []
    positive = null_positive = zero_reference = skipped = 0
    null_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    value_permutation = torch.randperm(256, generator=null_generator, device=device)
    byte_values = body.event_value(body.byte_embedding.weight).detach()
    index = 0
    while index + 1 < len(validation) and len(cosines) < args.audit_pairs:
        current, following = int(validation[index]), int(validation[index + 1])
        if not BYTE_OFFSET <= current < BYTE_OFFSET + 256:
            state = body.advance_time(state, 1.0)
            index += 1
            skipped += 1
            continue
        current_token = torch.tensor([current - BYTE_OFFSET], device=device, dtype=torch.long)
        prepared = body.step(state, current_token, elapsed=0.0, source="external")
        if not BYTE_OFFSET <= following < BYTE_OFFSET + 256:
            state = body.detach_state(prepared.state)
            index += 1
            skipped += 1
            continue
        following_token = torch.tensor([following - BYTE_OFFSET], device=device, dtype=torch.long)
        candidate = body.contact_teaching_direction(state, prepared, current_token)
        # Preserve every tensor shape, norm source, state factor and target;
        # only break the learned byte-to-value association.  This is the
        # appropriate null for the claim that the existing C value path, not
        # an arbitrary vector, provides the teaching direction.
        receipt = body.contact_teaching_receipt(state, prepared, current_token)
        null_teaching = receipt.innovation @ byte_values.index_select(0, value_permutation)
        null_candidate = torch.einsum(
            "bd,ba,baqsr->dqsr",
            null_teaching, receipt.route_weight, receipt.trace,
        ).flatten(start_dim=1)
        next_prepared = body.step(prepared.state, following_token, elapsed=0.0, source="external")
        next_score = F.log_softmax(next_prepared.logits, dim=-1)[0, following_token.item()]
        exact, = torch.autograd.grad(next_score, body.contact_to_activity.weight)
        candidate_norm, null_norm, exact_norm = candidate.norm(), null_candidate.norm(), exact.norm()
        if candidate_norm <= 1e-12 or null_norm <= 1e-12 or exact_norm <= 1e-12:
            zero_reference += 1
        else:
            cosine = _cosine(candidate, exact)
            dot = (candidate * exact).sum()
            null_cosine = _cosine(null_candidate, exact)
            null_dot = (null_candidate * exact).sum()
            cosines.append(float(cosine.item()))
            normalized_dots.append(float((dot / exact_norm).item()))
            null_cosines.append(float(null_cosine.item()))
            null_normalized_dots.append(float((null_dot / exact_norm).item()))
            positive += int(dot.item() > 0.0)
            null_positive += int(null_dot.item() > 0.0)
        state = body.detach_state(next_prepared.state)
        index += 2

    if not cosines:
        raise RuntimeError("no nonzero candidate/reference pairs were available")
    report = {
        "status": "value_path_contact_teaching_alignment_audit",
        "claim_scope": (
            "Real-OWT numerical admission audit of a no-new-state candidate direction "
            "against the exact one-next-byte contact gradient."
        ),
        "not_claimed": [
            "positive alignment beyond the measured stream", "an online S learning rule",
            "delayed action credit", "BPTT-free training for all parameters",
            "language-model capability", "life closure",
        ],
        "candidate": {
            "teaching": "(arrived - predicted) @ event_value(byte_embedding)",
            "eligibility": "old C/S trace at the post-arrival write route",
            "new_persistent_state": False,
            "new_parameters": False,
            "parameter_update_applied": False,
        },
        "reference": {
            "quantity": "exact gradient of next-byte log likelihood w.r.t. existing S-to-H contact weights",
            "graph": "two causal events per independently detached pair",
            "used_online": False,
        },
        "protocol": {
            "warmup": "continuous real OWT G14 head plus exact direct C-value credit",
            "warmup_metrics": warmup,
            "audit_pairs_requested": args.audit_pairs,
            "audit_pairs_nonzero": len(cosines),
            "zero_reference_pairs": zero_reference,
            "special_or_terminal_events": skipped,
            "pair_stride": 2,
        },
        "alignment": {
            "candidate": _summary(cosines, normalized_dots, positive),
            "byte_value_permutation_null": {
                **_summary(null_cosines, null_normalized_dots, null_positive),
                "permutation_seed": args.seed + 1,
                "control": "same receipt and exact gradient; learned byte-to-C value rows permuted",
            },
        },
        "model": {
            "runtime": "CGBColumnLanguageBody",
            "d_model": args.d_model,
            "grid_shape": [args.grid, args.grid],
            "active_columns": args.active_columns,
            "contact_rank": args.contact_rank,
            "parameter_numbers": sum(parameter.numel() for parameter in body.parameters()),
            "persistent_state_bytes_fp32": body.persistent_state_bytes(state),
            "device": str(device),
            "seed": args.seed,
        },
        "source": {"train": train_source, "validation": validation_source},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "alignment": report["alignment"],
                      "pairs": len(cosines)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
