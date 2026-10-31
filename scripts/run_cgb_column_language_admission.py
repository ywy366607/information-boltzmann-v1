"""Run a causal continuous-language admission for CGBColumnLanguageBody.

This is a real OpenWebText UTF-8 byte stream experiment.  It evaluates only
whether the new common-state body and its Lean-admitted G14 affine readout can
learn prequentially.  It makes no active-world, delayed-credit, or complete
language-capability claim.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.cgb_column_body import CGBColumnLanguageBody, ColumnBodyState

BYTE_OFFSET = 4
BYTE_VOCAB = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(root: Path, split: str, limit: int) -> tuple[np.ndarray, dict]:
    manifest_path = root / "manifest.json"
    tokenizer_path = root / "tokenizer.json"
    token_path = root / f"{split}.npy"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    if manifest.get("source") != "openwebtext" or tokenizer.get("type") != "utf8_bytes":
        raise ValueError("expected the registered OpenWebText UTF-8 byte source")
    values = np.load(token_path, mmap_mode="r")
    values = values[:min(limit, len(values))]
    return values, {
        "source": manifest["source"], "revision": manifest["revision"],
        "split": split, "path": str(token_path.relative_to(ROOT)),
        "tokens_available": int(len(np.load(token_path, mmap_mode="r"))),
        "tokens_presented": int(len(values)), "manifest_sha256": sha256(manifest_path),
        "tokenizer_sha256": sha256(tokenizer_path), "token_file_sha256": sha256(token_path),
        "split_rule": manifest["split_rule"],
    }


def bpb(nll: float, bytes_scored: int) -> float:
    return nll / (math.log(2.0) * max(bytes_scored, 1))


@dataclass
class PairHeads:
    """Two causal readout arms over exactly one body trajectory.

    The current admission updates only the G14 affine head.  H/C/S dynamics
    do not read that head, so frozen and locally updated heads provably share
    the same body trajectory.  Computing that trajectory twice would be a
    hardware artefact, not an experimental control.
    """

    frozen_weight: torch.Tensor
    frozen_bias: torch.Tensor
    learned_weight: torch.Tensor
    learned_bias: torch.Tensor
    bias_only: torch.Tensor


@dataclass
class CudaGraphBlock:
    """One fixed-length, unrolled causal execution graph and its inputs."""

    graph: torch.cuda.CUDAGraph
    tokens: torch.Tensor
    targets: torch.Tensor


class CudaGraphPair:
    """Fixed-shape online execution with exact, unrolled causal blocks.

    A block is not an associative scan: its individual steps remain ordered,
    so H/C/S and the G14 head are updated after every byte.  Capturing an
    entire block simply removes host launch overhead between those steps.
    """

    def __init__(self, body: CGBColumnLanguageBody, state: ColumnBodyState,
                 step_size: float, block_size: int) -> None:
        if block_size < 1:
            raise ValueError("CUDA graph block size must be positive")
        self.body = body
        self.step_size = float(step_size)
        self.block_size = int(block_size)
        self.activity, self.natural, self.precision, self.participation = (
            value.detach().clone() for value in (
                state.activity, state.natural, state.precision, state.participation,
            )
        )
        device = self.activity.device
        self.elapsed_zero = torch.zeros(1, device=device)
        self.external_source = torch.zeros(1, device=device, dtype=torch.long)
        self.total_nll = torch.zeros(3, device=device)
        self.one_hot_table = torch.eye(BYTE_VOCAB, device=device)
        readout = body.readout
        self.heads = PairHeads(*(parameter.detach().clone() for parameter in (
            readout.weight, readout.bias, readout.weight, readout.bias, readout.bias,
        )))

        # Materialise allocations before capture. Capturing executes the graph
        # once, so `_capture_block` snapshots and restores every mutable state
        # tensor afterwards; capture can never become a phantom language event.
        with torch.no_grad():
            probe = torch.zeros(1, device=device, dtype=torch.long)
            body._step_tensors(self.activity, self.natural, self.precision,
                               self.participation, probe, self.elapsed_zero,
                               self.external_source)
        self.graphs = {1: self._capture_block(1)}
        if self.block_size > 1:
            self.graphs[self.block_size] = self._capture_block(self.block_size)

    def _mutable_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.activity, self.natural, self.precision, self.participation,
                self.heads.frozen_weight, self.heads.frozen_bias,
                self.heads.learned_weight, self.heads.learned_bias,
                self.heads.bias_only, self.total_nll)

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
                (activity, natural, precision, participation,
                 features, _, _) = self.body._step_tensors(
                    self.activity, self.natural, self.precision, self.participation,
                    tokens[index:index + 1], self.elapsed_zero, self.external_source,
                )
                frozen_logits = F.linear(features, self.heads.frozen_weight, self.heads.frozen_bias)
                learned_logits = F.linear(features, self.heads.learned_weight, self.heads.learned_bias)
                bias_only_logits = self.heads.bias_only.unsqueeze(0)
                frozen_nll = torch.logsumexp(frozen_logits, dim=-1) - (frozen_logits * target).sum(dim=-1)
                learned_nll = torch.logsumexp(learned_logits, dim=-1) - (learned_logits * target).sum(dim=-1)
                bias_only_nll = torch.logsumexp(bias_only_logits, dim=-1) - (bias_only_logits * target).sum(dim=-1)
                self.total_nll.add_(torch.cat((frozen_nll, bias_only_nll, learned_nll)))
                innovation = target - torch.softmax(learned_logits, dim=-1)
                self.heads.learned_weight.add_(innovation.transpose(0, 1) @ features, alpha=self.step_size)
                self.heads.learned_bias.add_(innovation.sum(dim=0), alpha=self.step_size)
                self.heads.bias_only.add_(
                    (target - torch.softmax(bias_only_logits, dim=-1)).sum(dim=0),
                    alpha=self.step_size,
                )
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
        values = torch.as_tensor(byte_values, device=self.activity.device, dtype=torch.long)
        block.tokens.copy_(values)
        block.targets.copy_(self.one_hot_table.index_select(0, block.tokens))
        block.graph.replay()

    @torch.no_grad()
    def advance_empty_time(self) -> None:
        """Exact C release for a separator, preserving H/S and graph buffers."""
        mean = self.natural / self.precision
        process = F.softplus(self.body.process_variance_raw).to(dtype=self.precision.dtype)
        next_precision = self.precision / (1.0 + process * self.precision)
        self.natural.copy_(next_precision * mean)
        self.precision.copy_(next_precision)

    @torch.no_grad()
    def phase(self, raw_tokens: np.ndarray, report_every: int) -> tuple[dict, dict, dict]:
        self.total_nll.zero_()
        scored = skipped = 0
        progress: list[dict] = []
        started = time.monotonic()
        index = 0
        while index < len(raw_tokens):
            value = int(raw_tokens[index])
            if not BYTE_OFFSET <= value < BYTE_OFFSET + BYTE_VOCAB:
                self.advance_empty_time()
                skipped += 1
                index += 1
                continue
            run_start = index
            index += 1
            while index < len(raw_tokens) and BYTE_OFFSET <= int(raw_tokens[index]) < BYTE_OFFSET + BYTE_VOCAB:
                index += 1
            bytes_in_run = np.asarray(raw_tokens[run_start:index], dtype=np.int64) - BYTE_OFFSET
            cursor = 0
            while cursor < len(bytes_in_run):
                until_report = report_every - (scored % report_every)
                block = min(self.block_size, len(bytes_in_run) - cursor, until_report)
                if block == self.block_size:
                    self._replay(bytes_in_run[cursor:cursor + block])
                else:
                    for byte in bytes_in_run[cursor:cursor + block]:
                        self._replay(np.asarray([byte], dtype=np.int64))
                cursor += block
                scored += block
                if scored % report_every == 0:
                    torch.cuda.synchronize(self.activity.device)
                    totals = self.total_nll.cpu().tolist()
                    progress.append({
                        "scored_bytes": scored,
                        "frozen_readout_bits_per_byte": bpb(totals[0], scored),
                        "bias_only_g14_bits_per_byte": bpb(totals[1], scored),
                        "local_g14_readout_bits_per_byte": bpb(totals[2], scored),
                    })
        torch.cuda.synchronize(self.activity.device)
        elapsed = time.monotonic() - started
        totals = self.total_nll.cpu().tolist()
        common = {
            "scored_bytes": scored,
            "special_tokens_elapsed_without_lexical_write": skipped,
            "seconds": elapsed,
            "bytes_per_second": scored / max(elapsed, 1e-9),
            "progress": progress,
        }
        return ({**common, "bits_per_byte": bpb(totals[0], scored),
                 "mean_nll_nats": totals[0] / max(scored, 1)},
                {**common, "bits_per_byte": bpb(totals[1], scored),
                 "mean_nll_nats": totals[1] / max(scored, 1)},
                {**common, "bits_per_byte": bpb(totals[2], scored),
                 "mean_nll_nats": totals[2] / max(scored, 1)})

    def state(self) -> ColumnBodyState:
        return ColumnBodyState(*(value.detach().clone() for value in (
            self.activity, self.natural, self.precision, self.participation,
        )))


@torch.no_grad()
def eager_pair_phase(body: CGBColumnLanguageBody, state: ColumnBodyState, heads: PairHeads,
                     raw_tokens: np.ndarray, *, step_size: float, report_every: int) -> tuple[ColumnBodyState, dict, dict, dict]:
    """CPU fallback with the same one-body/two-head semantics as CUDA Graph."""
    totals = torch.zeros(3, device=state.activity.device)
    scored = skipped = 0
    progress: list[dict] = []
    started = time.monotonic()
    for raw in raw_tokens:
        value = int(raw)
        if not BYTE_OFFSET <= value < BYTE_OFFSET + BYTE_VOCAB:
            state = body.advance_time(state, 1.0)
            skipped += 1
            continue
        target = torch.tensor([value - BYTE_OFFSET], device=state.activity.device, dtype=torch.long)
        prepared = body.step(state, target, elapsed=0.0, source="external")
        frozen_logits = F.linear(prepared.features, heads.frozen_weight, heads.frozen_bias)
        learned_logits = F.linear(prepared.features, heads.learned_weight, heads.learned_bias)
        bias_only_logits = heads.bias_only.unsqueeze(0)
        target_one_hot = F.one_hot(target, BYTE_VOCAB).to(dtype=prepared.features.dtype)
        totals.add_(torch.stack((
            torch.logsumexp(frozen_logits, dim=-1)[0] - (frozen_logits * target_one_hot).sum(),
            torch.logsumexp(bias_only_logits, dim=-1)[0] - (bias_only_logits * target_one_hot).sum(),
            torch.logsumexp(learned_logits, dim=-1)[0] - (learned_logits * target_one_hot).sum(),
        )))
        innovation = target_one_hot - torch.softmax(learned_logits, dim=-1)
        heads.learned_weight.add_(innovation.transpose(0, 1) @ prepared.features, alpha=step_size)
        heads.learned_bias.add_(innovation.sum(dim=0), alpha=step_size)
        heads.bias_only.add_((target_one_hot - torch.softmax(bias_only_logits, dim=-1)).sum(dim=0),
                             alpha=step_size)
        state = body.detach_state(prepared.state)
        scored += 1
        if scored % report_every == 0:
            values = totals.tolist()
            progress.append({"scored_bytes": scored,
                             "frozen_readout_bits_per_byte": bpb(values[0], scored),
                             "bias_only_g14_bits_per_byte": bpb(values[1], scored),
                             "local_g14_readout_bits_per_byte": bpb(values[2], scored)})
    elapsed = time.monotonic() - started
    values = totals.tolist()
    common = {"scored_bytes": scored, "special_tokens_elapsed_without_lexical_write": skipped,
              "seconds": elapsed, "bytes_per_second": scored / max(elapsed, 1e-9), "progress": progress}
    return (state,
            {**common, "bits_per_byte": bpb(values[0], scored),
             "mean_nll_nats": values[0] / max(scored, 1)},
            {**common, "bits_per_byte": bpb(values[1], scored),
             "mean_nll_nats": values[1] / max(scored, 1)},
            {**common, "bits_per_byte": bpb(values[2], scored),
             "mean_nll_nats": values[2] / max(scored, 1)})


def run_pair(template: CGBColumnLanguageBody, train: np.ndarray, validation: np.ndarray, *,
             step_size: float, report_every: int, device: torch.device, graph_block: int):
    body = template.to(device).eval()
    initial_state = body.initial_state(1, device=device)
    if device.type == "cuda":
        runner = CudaGraphPair(body, initial_state, step_size, graph_block)
        frozen_train, bias_train, learned_train = runner.phase(train, report_every)
        frozen_validation, bias_validation, learned_validation = runner.phase(validation, report_every)
        final_state = runner.state()
        execution = f"cuda_graph_block{graph_block}_shared_body"
    else:
        readout = body.readout
        heads = PairHeads(*(parameter.detach().clone() for parameter in (
            readout.weight, readout.bias, readout.weight, readout.bias, readout.bias,
        )))
        final_state, frozen_train, bias_train, learned_train = eager_pair_phase(
            body, initial_state, heads, train, step_size=step_size, report_every=report_every)
        final_state, frozen_validation, bias_validation, learned_validation = eager_pair_phase(
            body, final_state, heads, validation, step_size=step_size, report_every=report_every)
        execution = "eager_shared_body"
    common = {"parameter_numbers": sum(parameter.numel() for parameter in body.parameters()),
              "persistent_state_bytes_fp32": body.persistent_state_bytes(final_state),
              "shared_body_trajectory": True, "execution": execution}
    return [
        {"name": "frozen_readout", "head_step_size": 0.0, **common,
         "train_first_pass": frozen_train, "validation_continuing_first_pass": frozen_validation},
        {"name": "bias_only_g14", "head_step_size": step_size, **common,
         "train_first_pass": bias_train, "validation_continuing_first_pass": bias_validation},
        {"name": "local_g14_readout", "head_step_size": step_size, **common,
         "train_first_pass": learned_train, "validation_continuing_first_pass": learned_validation},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "ib_owt_10000")
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--active-columns", type=int, default=8)
    parser.add_argument("--head-step", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report-every", type=int, default=10_000)
    parser.add_argument("--graph-block", type=int, default=128,
                        help="ordered byte steps unrolled into each CUDA Graph replay")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "published" / "cgb_column_g14_owt100k_20261031.json")
    args = parser.parse_args()
    if args.limit < 1 or args.report_every < 1 or args.graph_block < 1 or args.head_step < 0:
        raise ValueError("limits and step sizes must be nonnegative, with positive presentation/report limits")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    train, train_source = load_split(args.data_root, "train", args.limit)
    validation, validation_source = load_split(args.data_root, "validation", args.limit)
    torch.manual_seed(args.seed)
    template = CGBColumnLanguageBody(args.d_model, (args.grid, args.grid),
                                     active_columns=args.active_columns).to(device)
    arms = run_pair(template, train, validation, step_size=args.head_step,
                    report_every=args.report_every, device=device, graph_block=args.graph_block)
    by_name = {arm["name"]: arm for arm in arms}
    frozen = by_name["frozen_readout"]
    bias_only = by_name["bias_only_g14"]
    learned = by_name["local_g14_readout"]
    report = {
        "status": "column_body_language_admission",
        "claim_scope": "Real OWT UTF-8 byte prequential comparison of frozen, bias-only G14, and H/C/S-feature G14 readouts on one continuous H/C/S column body.",
        "not_claimed": ["full body parameter learning", "delayed three-factor action credit",
                          "active language acquisition", "general language-model capability", "life closure"],
        "protocol": {"prediction": "score before target write", "state": "one H/C/S trajectory across train then validation; frozen/G14 readout heads do not feed back into H/C/S",
                     "special_tokens": "advance empty time without lexical write", "autograd_history": "absent",
                     "online_update": "G14 affine readout only; matched bias-only control", "action": "absent",
                     "execution": arms[0]["execution"]},
        "model": {"runtime": "CGBColumnLanguageBody", "d_model": args.d_model,
                  "grid_shape": [args.grid, args.grid], "active_columns": args.active_columns,
                  "device": str(device), "seed": args.seed, "cuda_graph_block": args.graph_block},
        "source": {"train": train_source, "validation": validation_source},
        "arms": arms,
        "effect_of_g14": {"train_bpb_delta_vs_frozen": learned["train_first_pass"]["bits_per_byte"] - frozen["train_first_pass"]["bits_per_byte"],
                          "validation_bpb_delta_vs_frozen": learned["validation_continuing_first_pass"]["bits_per_byte"] - frozen["validation_continuing_first_pass"]["bits_per_byte"],
                          "train_bpb_delta_vs_bias_only": learned["train_first_pass"]["bits_per_byte"] - bias_only["train_first_pass"]["bits_per_byte"],
                          "validation_bpb_delta_vs_bias_only": learned["validation_continuing_first_pass"]["bits_per_byte"] - bias_only["validation_continuing_first_pass"]["bits_per_byte"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "effect_of_g14": report["effect_of_g14"],
                      "validation_bpb": {arm["name"]: arm["validation_continuing_first_pass"]["bits_per_byte"] for arm in arms}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
