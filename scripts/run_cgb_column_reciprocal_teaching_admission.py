"""Real-stream admission for the column body's reciprocal teaching splice.

This is intentionally a paired online comparison on the same OWT byte stream,
not a synthetic delayed-credit task.  Both bodies use the exact G14 output
update.  Only one settles the post-arrival S teaching phase and its matched
reciprocal H-to-S/S-to-H update.  The report keeps full-path alignment and
language benefit separate: splice contraction alone is not an alignment claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from information_boltzmann.core.cgb_column_body import CGBColumnLanguageBody, ColumnBodyState
from run_cgb_column_language_admission import (
    BYTE_OFFSET, BYTE_VOCAB, ROOT, CudaGraphBlock, bpb, load_split,
)


@torch.no_grad()
def run_phase(
    body: CGBColumnLanguageBody,
    state: ColumnBodyState,
    raw_tokens: np.ndarray,
    *,
    head_step: float,
    teaching_step: float | None,
    report_every: int,
) -> tuple[ColumnBodyState, dict]:
    """Run a continuing prequential byte stream without a reverse-time graph."""
    total_nll = 0.0
    total_alignment = 0.0
    alignment_count = scored = skipped = 0
    progress: list[dict[str, float | int]] = []
    started = time.monotonic()
    for raw in raw_tokens:
        value = int(raw)
        if not BYTE_OFFSET <= value < BYTE_OFFSET + BYTE_VOCAB:
            state = body.advance_time(state, 1.0)
            skipped += 1
            continue
        target = torch.tensor([value - BYTE_OFFSET], device=state.activity.device, dtype=torch.long)
        prepared = body.step(state, target, elapsed=0.0, source="external")
        total_nll += float(F.cross_entropy(prepared.logits, target, reduction="sum"))
        if teaching_step is not None:
            # This compares the live local direction to the exact immediate
            # G14 feature direction before either parameter family is changed.
            total_alignment += float(body.teaching_alignment_to_g14(prepared, target).sum())
            alignment_count += target.shape[0]
        body.update_readout_from_arrival(prepared, target, step_size=head_step)
        state = (body.settle_reciprocal_teaching_from_arrival(
            prepared, target, step_size=teaching_step,
        ) if teaching_step is not None else prepared.state)
        state = body.detach_state(state)
        scored += 1
        if scored % report_every == 0:
            item: dict[str, float | int] = {
                "scored_bytes": scored,
                "bits_per_byte": bpb(total_nll, scored),
            }
            if alignment_count:
                item["mean_teaching_to_g14_cosine"] = total_alignment / alignment_count
            progress.append(item)
    seconds = time.monotonic() - started
    return state, {
        "scored_bytes": scored,
        "special_tokens_elapsed_without_lexical_write": skipped,
        "bits_per_byte": bpb(total_nll, scored),
        "mean_nll_nats": total_nll / max(scored, 1),
        "mean_teaching_to_g14_cosine": (
            total_alignment / alignment_count if alignment_count else None
        ),
        "seconds": seconds,
        "bytes_per_second": scored / max(seconds, 1e-9),
        "progress": progress,
    }


class CudaOnlineTeaching:
    """CUDA-graph execution of ordered, per-byte online teaching updates.

    Capture removes Python/kernel-launch gaps only.  It does not turn the
    recurrence into a batch update: every byte still scores old H/C, writes
    C/S, updates the head, then optionally settles reciprocal teaching before
    the next byte in the captured block.
    """

    def __init__(self, body: CGBColumnLanguageBody, state: ColumnBodyState, *,
                 head_step: float, teaching_step: float | None, block_size: int) -> None:
        if block_size < 1:
            raise ValueError("CUDA graph block size must be positive")
        self.body = body
        self.head_step = float(head_step)
        self.teaching_step = teaching_step
        self.block_size = int(block_size)
        self.activity, self.natural, self.precision, self.participation = (
            value.detach().clone() for value in (
                state.activity, state.natural, state.precision, state.participation,
            )
        )
        device = self.activity.device
        self.elapsed_zero = torch.zeros(1, device=device)
        self.external_source = torch.zeros(1, device=device, dtype=torch.long)
        self.one_hot_table = torch.eye(BYTE_VOCAB, device=device)
        self.total_nll = torch.zeros(1, device=device)
        self.total_alignment = torch.zeros(1, device=device)
        self.alignment_count = 0
        with torch.no_grad():
            probe = torch.zeros(1, device=device, dtype=torch.long)
            body._step_tensors(self.activity, self.natural, self.precision,
                               self.participation, probe, self.elapsed_zero,
                               self.external_source)
        self.graphs = {1: self._capture_block(1)}
        if self.block_size > 1:
            self.graphs[self.block_size] = self._capture_block(self.block_size)

    def _mutable_tensors(self) -> tuple[torch.Tensor, ...]:
        tensors = [
            self.activity, self.natural, self.precision, self.participation,
            self.body.readout.weight, self.body.readout.bias,
            self.total_nll, self.total_alignment,
        ]
        if self.teaching_step is not None:
            tensors.extend((self.body.teaching_to_activity.weight,
                            self.body.activity_to_teaching.weight))
        return tuple(tensors)

    @torch.no_grad()
    def _capture_block(self, length: int) -> CudaGraphBlock:
        device = self.activity.device
        tokens = torch.zeros(length, device=device, dtype=torch.long)
        targets = torch.zeros(length, BYTE_VOCAB, device=device)
        targets[:, 0] = 1
        snapshots = tuple(value.clone() for value in self._mutable_tensors())
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for index in range(length):
                target = targets[index:index + 1]
                activity, natural, precision, participation, features, route, _ = self.body._step_tensors(
                    self.activity, self.natural, self.precision, self.participation,
                    tokens[index:index + 1], self.elapsed_zero, self.external_source,
                )
                logits = self.body.readout(features)
                self.total_nll.add_(torch.logsumexp(logits, dim=-1) - (logits * target).sum(dim=-1))
                innovation = target - torch.softmax(logits, dim=-1)
                if self.teaching_step is not None:
                    state = ColumnBodyState(activity, natural, precision, participation)
                    mass = route.sum(dim=(1, 2)).clamp_min(torch.finfo(activity.dtype).eps)
                    local_activity = (route.unsqueeze(-1) * activity).sum(dim=(1, 2)) / mass.unsqueeze(-1)
                    contact = torch.tanh(
                        self.body.activity_to_teaching(local_activity) * self.body.error_to_teaching(innovation)
                    )
                    teaching = self.body.teaching_to_activity(contact)
                    exact = innovation @ self.body.readout.weight
                    self.total_alignment.add_(F.cosine_similarity(teaching, exact, dim=-1, eps=1e-8))
                    local = torch.cat((activity, natural / precision, self.body.demand(state)), dim=-1)
                    participation = self.body._reciprocal_teaching_update(
                        participation, local, route, contact,
                    )
                    release = self.teaching_step * self.body._teaching_release(state, route)
                    shared = (local_activity.transpose(0, 1) @ contact) / local_activity.shape[0]
                    factor = 1.0 - release
                    self.body.teaching_to_activity.weight.mul_(factor).add_(shared, alpha=self.teaching_step)
                    self.body.activity_to_teaching.weight.mul_(factor).add_(shared.transpose(0, 1),
                                                                            alpha=self.teaching_step)
                self.body.readout.weight.add_(innovation.transpose(0, 1) @ features, alpha=self.head_step)
                self.body.readout.bias.add_(innovation.sum(dim=0), alpha=self.head_step)
                self.activity.copy_(activity)
                self.natural.copy_(natural)
                self.precision.copy_(precision)
                self.participation.copy_(participation)
        torch.cuda.synchronize(device)
        for mutable, snapshot in zip(self._mutable_tensors(), snapshots):
            mutable.copy_(snapshot)
        torch.cuda.synchronize(device)
        return CudaGraphBlock(graph, tokens, targets)

    @torch.no_grad()
    def _replay(self, byte_values: np.ndarray) -> None:
        block = self.graphs.get(len(byte_values))
        if block is None:
            raise ValueError("only captured CUDA graph lengths may be replayed")
        block.tokens.copy_(torch.as_tensor(byte_values, device=self.activity.device, dtype=torch.long))
        block.targets.copy_(self.one_hot_table.index_select(0, block.tokens))
        block.graph.replay()

    @torch.no_grad()
    def advance_empty_time(self) -> None:
        mean = self.natural / self.precision
        process = F.softplus(self.body.process_variance_raw).to(dtype=self.precision.dtype)
        next_precision = self.precision / (1.0 + process * self.precision)
        self.natural.copy_(next_precision * mean)
        self.precision.copy_(next_precision)

    @torch.no_grad()
    def phase(self, raw_tokens: np.ndarray, report_every: int) -> dict:
        self.total_nll.zero_()
        self.total_alignment.zero_()
        self.alignment_count = scored = skipped = 0
        progress: list[dict[str, float | int]] = []
        started = time.monotonic()
        index = 0
        while index < len(raw_tokens):
            if not BYTE_OFFSET <= int(raw_tokens[index]) < BYTE_OFFSET + BYTE_VOCAB:
                self.advance_empty_time()
                skipped += 1
                index += 1
                continue
            run_start = index
            index += 1
            while index < len(raw_tokens) and BYTE_OFFSET <= int(raw_tokens[index]) < BYTE_OFFSET + BYTE_VOCAB:
                index += 1
            values = np.asarray(raw_tokens[run_start:index], dtype=np.int64) - BYTE_OFFSET
            cursor = 0
            while cursor < len(values):
                until_report = report_every - (scored % report_every)
                width = min(self.block_size, len(values) - cursor, until_report)
                if width == self.block_size:
                    self._replay(values[cursor:cursor + width])
                else:
                    for value in values[cursor:cursor + width]:
                        self._replay(np.asarray([value], dtype=np.int64))
                cursor += width
                scored += width
                if self.teaching_step is not None:
                    self.alignment_count += width
                if scored % report_every == 0:
                    torch.cuda.synchronize(self.activity.device)
                    item: dict[str, float | int] = {
                        "scored_bytes": scored,
                        "bits_per_byte": bpb(float(self.total_nll.item()), scored),
                    }
                    if self.alignment_count:
                        item["mean_teaching_to_g14_cosine"] = float(self.total_alignment.item()) / self.alignment_count
                    progress.append(item)
        torch.cuda.synchronize(self.activity.device)
        seconds = time.monotonic() - started
        total = float(self.total_nll.item())
        return {
            "scored_bytes": scored,
            "special_tokens_elapsed_without_lexical_write": skipped,
            "bits_per_byte": bpb(total, scored),
            "mean_nll_nats": total / max(scored, 1),
            "mean_teaching_to_g14_cosine": (
                float(self.total_alignment.item()) / self.alignment_count if self.alignment_count else None
            ),
            "seconds": seconds, "bytes_per_second": scored / max(seconds, 1e-9), "progress": progress,
        }

    def state(self) -> ColumnBodyState:
        return ColumnBodyState(*(value.detach().clone() for value in (
            self.activity, self.natural, self.precision, self.participation,
        )))


def make_body(args: argparse.Namespace, device: torch.device) -> CGBColumnLanguageBody:
    return CGBColumnLanguageBody(
        args.d_model, (args.grid, args.grid), active_columns=args.active_columns,
        contact_rank=args.contact_rank,
    ).to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "ib_owt_10000")
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--active-columns", type=int, default=8)
    parser.add_argument(
        "--contact-rank", type=int, default=4,
        help="four basis coordinates give 8r=32 typed-contact directions for D=32; rank1 is a low-rank ablation",
    )
    parser.add_argument("--head-step", type=float, default=0.01)
    parser.add_argument("--teaching-step", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report-every", type=int, default=10_000)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "published" /
                        "cgb_column_reciprocal_teaching_owt100k_20261031.json")
    args = parser.parse_args()
    if (args.limit < 1 or args.report_every < 1 or args.contact_rank < 1 or
            args.head_step < 0 or args.teaching_step < 0):
        raise ValueError("limits/rank must be positive and update scales nonnegative")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    train, train_source = load_split(args.data_root, "train", args.limit)
    validation, validation_source = load_split(args.data_root, "validation", args.limit)

    # Re-seeding produces identical legacy H/C/S maps for both arms.  Their
    # only state/parameter difference after the first byte is the admitted
    # post-arrival teaching settlement.
    torch.manual_seed(args.seed)
    frozen_teaching = make_body(args, device)
    torch.manual_seed(args.seed)
    reciprocal_teaching = make_body(args, device)
    baseline_state = frozen_teaching.initial_state(1, device=device)
    reciprocal_state = reciprocal_teaching.initial_state(1, device=device)
    if device.type == "cuda":
        baseline_runner = CudaOnlineTeaching(
            frozen_teaching, baseline_state, head_step=args.head_step,
            teaching_step=None, block_size=128,
        )
        reciprocal_runner = CudaOnlineTeaching(
            reciprocal_teaching, reciprocal_state, head_step=args.head_step,
            teaching_step=args.teaching_step, block_size=128,
        )
        baseline_train = baseline_runner.phase(train, args.report_every)
        reciprocal_train = reciprocal_runner.phase(train, args.report_every)
        baseline_valid = baseline_runner.phase(validation, args.report_every)
        reciprocal_valid = reciprocal_runner.phase(validation, args.report_every)
        baseline_state, reciprocal_state = baseline_runner.state(), reciprocal_runner.state()
    else:
        baseline_state, baseline_train = run_phase(
            frozen_teaching, baseline_state, train, head_step=args.head_step,
            teaching_step=None, report_every=args.report_every,
        )
        reciprocal_state, reciprocal_train = run_phase(
            reciprocal_teaching, reciprocal_state, train, head_step=args.head_step,
            teaching_step=args.teaching_step, report_every=args.report_every,
        )
        baseline_state, baseline_valid = run_phase(
            frozen_teaching, baseline_state, validation, head_step=args.head_step,
            teaching_step=None, report_every=args.report_every,
        )
        reciprocal_state, reciprocal_valid = run_phase(
            reciprocal_teaching, reciprocal_state, validation, head_step=args.head_step,
            teaching_step=args.teaching_step, report_every=args.report_every,
        )
    report = {
        "status": "reciprocal_teaching_admission",
        "claim_scope": "Paired real-OWT prequential comparison of an H/C/S column body with or without post-arrival reciprocal teaching settlement.",
        "not_claimed": [
            "full feedback-path alignment", "delayed action credit", "active language acquisition",
            "language-model capability", "life closure",
        ],
        "mechanism": {
            "persistent_state": "one H/C/S body; S has arrival and signed teaching types",
            "third_factor": "arrived byte observed-minus-predicted residual",
            "eligibility": "local H-to-S map read at routed columns",
            "reciprocity": "H-to-S and S-to-H maps receive the same local outer product and scaled common release",
            "causal_order": "score old H/C; arrive byte; G14 head update; optionally settle teaching S; next event reads teaching S modulatively",
        },
        "protocol": {
            "data": "same continuous OWT train then validation stream for both arms",
            "autograd_history": "absent", "bptt": "absent",
            "baseline": "same output G14 update, teaching settlement omitted",
            "alignment": "cosine between S-to-H local teaching direction and exact immediate affine G14 feature score before updates",
        },
        "model": {
            "d_model": args.d_model, "grid_shape": [args.grid, args.grid],
            "active_columns": args.active_columns, "contact_rank": args.contact_rank,
            "head_step": args.head_step, "teaching_step": args.teaching_step,
            "seed": args.seed, "device": str(device),
            "baseline_parameters": sum(p.numel() for p in frozen_teaching.parameters()),
            "reciprocal_parameters": sum(p.numel() for p in reciprocal_teaching.parameters()),
            "baseline_state_bytes_fp32": frozen_teaching.persistent_state_bytes(baseline_state),
            "reciprocal_state_bytes_fp32": reciprocal_teaching.persistent_state_bytes(reciprocal_state),
        },
        "source": {"train": train_source, "validation": validation_source},
        "arms": {
            "g14_without_teaching_settlement": {
                "train_first_pass": baseline_train,
                "validation_continuing_first_pass": baseline_valid,
            },
            "g14_with_reciprocal_teaching_settlement": {
                "train_first_pass": reciprocal_train,
                "validation_continuing_first_pass": reciprocal_valid,
            },
        },
        "effects": {
            "train_bpb_delta_teaching_minus_baseline": reciprocal_train["bits_per_byte"] - baseline_train["bits_per_byte"],
            "validation_bpb_delta_teaching_minus_baseline": reciprocal_valid["bits_per_byte"] - baseline_valid["bits_per_byte"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "effects": report["effects"],
                      "validation_bpb": {name: arm["validation_continuing_first_pass"]["bits_per_byte"]
                                         for name, arm in report["arms"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
