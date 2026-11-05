"""Audit whether the existing local S proxy points toward future language value.

The S update in :mod:`run_cgb_column_language_admission` is deliberately a
local proxy: a one-event returned contact signal times a VJP of the current
shared S pool.  It is not a claim about a total future gradient.  This script
keeps that distinction executable.

Starting from a saved continuous language individual, it advances a reference
body with further S slow updates disabled.  At each declared real OWT anchor it
forms the existing proxy from the just-arrived byte, lets all ordinary current
event updates happen, and branches the complete post-event body three ways:
``+delta``, ``-delta``, and zero.  The three bodies then receive the same future
bytes while H, C, route, value, head, and contact learning remain enabled; only
additional S slow updates are held off in every arm.  Thus this is a local
directional audit, not a new learner, a hyperparameter search, or a capability
benchmark.

The resulting future code lengths instantiate the external empirical audit in
``GenerationFutureValue``.  They are not reward, food, reserve, or a tensor
stored in the organism.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.cgb_column_body import ColumnBodyState
from scripts.run_cgb_column_language_admission import (
    BYTE_OFFSET,
    BYTE_VOCAB,
    CudaGraphValueCredit,
    load_split,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def proxy_direction_for_byte(engine: CudaGraphValueCredit, byte_value: int) -> torch.Tensor:
    """Read the current S proxy without changing state, receipt, or parameters."""
    if not 0 <= byte_value < BYTE_VOCAB:
        raise ValueError("proxy byte must be a UTF-8 byte value")
    if not engine.record_s_return:
        raise ValueError("checkpoint must retain the one-event S return receipt")
    body = engine.body
    state = ColumnBodyState(engine.activity, engine.natural, engine.precision,
                            engine.causal_token, engine.participation)
    token = torch.full((1,), byte_value, device=engine.activity.device, dtype=torch.long)
    read_route = body._state_route(state)
    prepared = body._step_tensors(
        engine.activity, engine.natural, engine.precision, engine.causal_token,
        engine.participation, token, engine.elapsed_zero, engine.external_source,
    )
    (_, _, _, _, _, features, write_route, _, _, _, response_gate, contact_pre) = prepared
    logits = body.readout(features)
    teaching = body._s_contact_return_from_tensors(
        logits, read_route, engine.previous_contact_index,
        engine.previous_contact_gate, engine.previous_contact_unique,
        engine.previous_s_flux, engine.previous_s_return_weight,
        token, engine.has_previous_contact,
    )
    direction = body._s_contact_direction_from_tensors(
        engine.participation, engine.precision, contact_pre, response_gate,
        write_route, engine.external_source, teaching,
    )
    return direction[body.d_model + 2:].detach()


@torch.no_grad()
def apply_relative_s_delta(engine: CudaGraphValueCredit, direction: torch.Tensor,
                           signed_relative_l2: float) -> dict:
    """Apply one signed, fixed-relative-L2 S displacement and report its size."""
    if not math.isfinite(signed_relative_l2):
        raise ValueError("relative perturbation must be finite")
    s_rows = engine.body.reciprocal_contact[engine.body.d_model + 2:]
    if direction.shape != s_rows.shape:
        raise ValueError("proxy direction shape does not match the S rows")
    direction_norm = float(direction.norm().item())
    parameter_norm = float(s_rows.norm().item())
    if direction_norm == 0.0 or signed_relative_l2 == 0.0:
        return {"direction_l2": direction_norm, "s_parameter_l2_before": parameter_norm,
                "delta_l2": 0.0}
    delta = direction * (signed_relative_l2 * parameter_norm / direction_norm)
    s_rows.add_(delta)
    return {"direction_l2": direction_norm, "s_parameter_l2_before": parameter_norm,
            "delta_l2": float(delta.norm().item())}


def lexical_index(raw: np.ndarray, start: int, stop: int) -> int:
    """Return the first raw position carrying a byte; separators are elapsed time."""
    for index in range(start, stop):
        if BYTE_OFFSET <= int(raw[index]) < BYTE_OFFSET + BYTE_VOCAB:
            return index
    raise ValueError("no lexical byte found before the requested source limit")


def clone_from_state(state: dict, device: torch.device) -> CudaGraphValueCredit:
    """Make a one-event-graph audit arm without expanding graph memory by horizon.

    The live individual uses a 128-event CUDA graph for throughput.  A three-way
    finite perturbation needs no such graph: its fixed one-event graph has the
    identical ordered transition and prevents audit branch count from consuming
    the whole GPU merely by duplicating captured unrolls.
    """
    audit_state = dict(state)
    audit_state["profile"] = dict(state["profile"], block_size=1)
    return CudaGraphValueCredit.from_checkpoint(audit_state, device=device, s_contact_step=0.0)


@torch.no_grad()
def score_branch(state: dict, direction: torch.Tensor, signed_relative_l2: float,
                 future: np.ndarray, device: torch.device) -> tuple[dict, dict]:
    """Run one disposable branch, keeping only its scalar result on the GPU."""
    branch = clone_from_state(state, device)
    delta = apply_relative_s_delta(branch, direction, signed_relative_l2)
    result = branch.phase(future, report_every=len(future) + 1)
    del branch
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return result, delta


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"count": int(len(array)), "min": float(array.min()),
            "median": float(np.median(array)), "mean": float(array.mean()),
            "max": float(array.max()),
            "negative": int((array < 0).sum()), "positive": int((array > 0).sum()),
            "zero": int((array == 0).sum())}


@torch.no_grad()
def run_audit(checkpoint: dict, raw: np.ndarray, *, arm: str, device: torch.device,
              raw_start: int, anchors: int, anchor_stride: int, horizon: int,
              relative_l2: float) -> dict:
    if arm not in checkpoint["engines"]:
        raise ValueError(f"checkpoint has no {arm!r} arm")
    if anchors < 1 or anchor_stride < 1 or horizon < 1 or relative_l2 <= 0:
        raise ValueError("anchors, stride, horizon, and relative L2 must be positive")
    if raw_start < 0 or raw_start >= len(raw):
        raise ValueError("raw start lies outside the loaded source")

    reference = clone_from_state(checkpoint["engines"][arm], device)
    cursor = raw_start
    records: list[dict] = []
    for anchor_number in range(anchors):
        candidate_start = raw_start + anchor_number * anchor_stride
        if candidate_start >= len(raw):
            raise ValueError("source limit does not cover all requested anchors")
        if candidate_start > cursor:
            reference.phase(raw[cursor:candidate_start], report_every=candidate_start - cursor + 1)
        cursor = lexical_index(raw, candidate_start, len(raw))
        probe = int(raw[cursor]) - BYTE_OFFSET
        source_lifetime_before = source_state_after_lifetime(reference)
        direction = proxy_direction_for_byte(reference, probe)
        if source_lifetime_before != source_state_after_lifetime(reference):
            raise AssertionError("proxy direction unexpectedly changed the reference lifetime")
        # The byte first becomes an ordinary received event.  The subsequent
        # perturbation therefore cannot alter its already-issued score.
        reference.phase(raw[cursor:cursor + 1], report_every=2)
        post_event = reference.checkpoint_state()
        future_stop = min(cursor + 1 + horizon, len(raw))
        future = raw[cursor + 1:future_stop]
        if len(future) < horizon:
            raise ValueError("source limit did not cover a complete future horizon")
        # These are intentionally sequential.  A branch is an evaluation
        # continuation, not a concurrent body, and sequential reuse preserves
        # the exact outcome while bounding CUDA graph memory at one branch.
        result_plus, plus_delta = score_branch(post_event, direction, relative_l2, future, device)
        result_minus, minus_delta = score_branch(post_event, direction, -relative_l2, future, device)
        result_zero, zero_delta = score_branch(post_event, direction, 0.0, future, device)
        if not (result_plus["scored_bytes"] == result_minus["scored_bytes"] ==
                result_zero["scored_bytes"]):
            raise AssertionError("paired branches scored different byte counts")
        records.append({
            "anchor": anchor_number, "probe_raw_offset": cursor,
            "probe_byte": probe, "future_raw": [cursor + 1, future_stop],
            "scored_bytes": result_zero["scored_bytes"],
            "proxy": plus_delta,
            "minus_delta_l2": minus_delta["delta_l2"], "zero_delta_l2": zero_delta["delta_l2"],
            "future_nll_nats": {"plus": result_plus["mean_nll_nats"],
                                 "zero": result_zero["mean_nll_nats"],
                                 "minus": result_minus["mean_nll_nats"]},
            "plus_minus_zero_nats": result_plus["mean_nll_nats"] - result_zero["mean_nll_nats"],
            "minus_minus_zero_nats": result_minus["mean_nll_nats"] - result_zero["mean_nll_nats"],
        })
        # Keep the reference on its single S-held trajectory.  Branches are
        # deliberately disposable measurements and never feed later anchors.
        cursor += 1
    plus_deltas = [record["plus_minus_zero_nats"] for record in records]
    minus_deltas = [record["minus_minus_zero_nats"] for record in records]
    directional = [record["plus_minus_zero_nats"] - record["minus_minus_zero_nats"]
                   for record in records]
    return {"anchors": records,
            "summary": {"plus_minus_zero_nats": summarize(plus_deltas),
                        "minus_minus_zero_nats": summarize(minus_deltas),
                        "plus_minus_minus_nats": summarize(directional),
                        "plus_better_than_zero": int(sum(value < 0 for value in plus_deltas)),
                        "minus_better_than_zero": int(sum(value < 0 for value in minus_deltas)),
                        "plus_better_than_minus": int(sum(value < 0 for value in directional))}}


def source_state_after_lifetime(engine: CudaGraphValueCredit) -> dict:
    """Small immutable view used to assert that proxy extraction is read-only."""
    return {"scored_lifetime": engine.scored_lifetime, "special_lifetime": engine.special_lifetime,
            "update_calls": dict(engine.update_calls), "advanced_elapsed": engine.advanced_elapsed}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "results/cgb_column_language_living_20261101/latest.pt")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/ib_owt_10000")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/published/cgb_column_s_proxy_future_value_owt_20261101.json")
    parser.add_argument("--arm", choices=("s_on", "s_off"), default="s_on")
    parser.add_argument("--raw-start", type=int, default=None)
    parser.add_argument("--anchors", type=int, default=16)
    parser.add_argument("--anchor-stride", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=2048)
    parser.add_argument("--relative-l2", type=float, default=1e-4)
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("experiment", {}).get("next_raw_cursor") is None:
        raise ValueError("checkpoint lacks the continuous raw source cursor")
    raw_start = (checkpoint["experiment"]["next_raw_cursor"]
                 if args.raw_start is None else args.raw_start)
    source_limit = raw_start + (args.anchors - 1) * args.anchor_stride + 1 + args.horizon
    raw, provenance = load_split(args.data_root.resolve(), "train", source_limit)
    if len(raw) < source_limit:
        raise ValueError("registered OWT train split is too short for the declared audit")
    expected = checkpoint["experiment"].get("provenance", {}).get("train", {})
    if expected and provenance["revision"] != expected.get("revision"):
        raise ValueError("checkpoint and audit source revisions differ")
    device = torch.device(args.device)
    audit = run_audit(checkpoint, raw, arm=args.arm, device=device, raw_start=raw_start,
                      anchors=args.anchors, anchor_stride=args.anchor_stride,
                      horizon=args.horizon, relative_l2=args.relative_l2)
    result = {
        "kind": "local_s_proxy_empirical_future_value_audit",
        "scope": (
            "A post-event finite perturbation audit of the existing one-event S proxy. "
            "All branches retain H/C/route/value/contact/head learning, but later S proxy writes "
            "are disabled in every branch. This does not evaluate the live update order, BPTT, "
            "an exact future gradient, long-horizon credit, or language capability."
        ),
        "theorem_contract": {
            "name": "GenerationFutureValue.observed_continuation_improvement_is_empirically_paid",
            "use": "future language code length is an external audit only, never a body state or reward account",
        },
        "checkpoint": {"path": str(args.checkpoint.relative_to(ROOT)), "sha256": sha256(args.checkpoint),
                       "arm": args.arm, "raw_start": raw_start,
                       "checkpoint_s_contact_step": checkpoint["engines"][args.arm]["profile"]["s_contact_step"]},
        "source": provenance,
        "protocol": {"anchors": args.anchors, "anchor_stride_raw": args.anchor_stride,
                     "future_horizon_raw": args.horizon, "relative_s_l2": args.relative_l2,
                     "all_later_s_slow_updates": "disabled_in_plus_minus_zero",
                     "ordinary_other_learning": "enabled_in_plus_minus_zero"},
        **audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
