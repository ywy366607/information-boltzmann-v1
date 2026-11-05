"""Run a causal continuous-language admission for CGBColumnLanguageBody.

This is a real OpenWebText UTF-8 byte stream experiment.  It evaluates only
whether the new common-state body and its Lean-admitted G14 affine readout can
learn prequentially.  It makes no active-world, delayed-credit, or complete
language-capability claim.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, replace
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
from information_boltzmann.core.cgb_column_body import CGBColumnLanguageBody, ColumnBodyState, ColumnBodyStep

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
        self.activity, self.natural, self.precision, self.causal_token, self.participation = (
            value.detach().clone() for value in (
                state.activity, state.natural, state.precision, state.causal_token, state.participation,
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
            body._step_tensors(self.activity, self.natural, self.precision, self.causal_token,
                               self.participation, probe, self.elapsed_zero,
                               self.external_source)
        self.graphs = {1: self._capture_block(1)}
        if self.block_size > 1:
            self.graphs[self.block_size] = self._capture_block(self.block_size)

    def _mutable_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.activity, self.natural, self.precision, self.causal_token, self.participation,
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
                (activity, natural, precision, causal_token, participation,
                 features, _, _, _, _, _, _) = self.body._step_tensors(
                    self.activity, self.natural, self.precision, self.causal_token, self.participation,
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
                self.causal_token.copy_(causal_token)
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
            self.activity, self.natural, self.precision, self.causal_token, self.participation,
        )))


class CudaGraphValueCredit:
    """One H/C/S trajectory with admitted direct C and fixed-route credit."""

    def __init__(self, body: CGBColumnLanguageBody, state: ColumnBodyState,
                 head_step: float, value_step: float, route_step: float,
                 contact_step: float, block_size: int, *, s_contact_step: float = 0.0,
                 record_s_return: bool = False, collect_diagnostics: bool = False) -> None:
        if block_size < 1:
            raise ValueError("graph block size must be positive")
        self.body, self.head_step = body, float(head_step)
        self.value_step, self.route_step = float(value_step), float(route_step)
        self.contact_step = float(contact_step)
        self.s_contact_step = float(s_contact_step)
        self.record_s_return = bool(record_s_return or self.s_contact_step > 0)
        # A one-event flux receipt is also needed by the admitted dynamic
        # event-drive direction.  It remains a fixed pipeline buffer, not H/C/S.
        self.record_contact_return = bool(self.record_s_return or self.value_step > 0)
        self.collect_diagnostics = bool(collect_diagnostics)
        self.block_size = int(block_size)
        self.scored_lifetime = self.special_lifetime = 0
        # Enabled update calls/opportunities, including zero directions (e.g.
        # the birth event has no previous receipt); not nonzero-update counts.
        self.update_calls = dict.fromkeys(("head", "value", "route", "contact", "s_contact"), 0)
        self.advanced_elapsed = 0.0
        self.activity, self.natural, self.precision, self.causal_token, self.participation = (
            value.detach().clone() for value in (
                state.activity, state.natural, state.precision, state.causal_token, state.participation,
            )
        )
        device = self.activity.device
        self.elapsed_zero = self.activity.new_zeros(1)
        self.external_source = torch.zeros(1, device=device, dtype=torch.long)
        self.total_nll = self.activity.new_zeros(1)
        self.one_hot_table = torch.eye(BYTE_VOCAB, device=device, dtype=self.activity.dtype)
        # External observation totals, never body state or an eligibility page.
        self.diagnostic_totals = self.activity.new_zeros(6) if self.collect_diagnostics else None
        # One issued local receipt is a fixed O(1) pipeline buffer, not part of
        # the organism state.  It is consumed by the immediately next scored
        # byte, then overwritten.  Longer delay must use a proved H/C/S carrier.
        self.local_receipt_sites = 5 * body.active_columns
        self.previous_contact_index = torch.zeros(
            1, self.local_receipt_sites, device=device, dtype=torch.long,
        )
        self.previous_contact_pre = self.activity.new_zeros(
            1, self.local_receipt_sites, body.local_contact_input_dim,
        )
        self.previous_contact_gate = torch.zeros(
            1, self.local_receipt_sites, body.d_model, device=device, dtype=torch.bool,
        )
        self.previous_contact_unique = torch.zeros(
            1, self.local_receipt_sites, device=device, dtype=torch.bool,
        )
        self.has_previous_contact = torch.zeros(1, device=device, dtype=torch.bool)
        if self.record_contact_return:
            self.previous_s_flux = self.activity.new_zeros((*self.activity.shape[:-1], 2))
        if self.record_s_return:
            self.previous_s_return_weight = torch.zeros_like(
                body.reciprocal_contact[:body.d_model + 2, body._contact_s_input],
            )
        self.graphs: dict[int, CudaGraphBlock] = {}
        if device.type == "cuda":
            with torch.no_grad():
                probe = torch.zeros(1, device=device, dtype=torch.long)
                probe_step = body._step_tensors(
                    self.activity, self.natural, self.precision, self.causal_token,
                    self.participation, probe, self.elapsed_zero, self.external_source,
                )
                if self.s_contact_step > 0:
                    # Initialise the same bounded S exchange used by a real
                    # one-receipt language return.  It adds no graph or
                    # persistent state page.
                    body.evidence_gated_s_exchange(
                        self.participation, torch.zeros_like(self.participation),
                        self.s_contact_step,
                    )
            self.graphs[1] = self._capture_block(1)
            if self.block_size > 1:
                self.graphs[self.block_size] = self._capture_block(self.block_size)

    def mutable_tensor_map(self) -> dict[str, torch.Tensor]:
        """Stable live-buffer names; ``body.*`` entries alias model parameters."""
        names = (
            "activity", "natural", "precision", "causal_token", "participation",
            "previous_contact_index", "previous_contact_pre", "previous_contact_gate",
            "previous_contact_unique", "has_previous_contact", "total_nll",
        )
        tensors = {name: getattr(self, name) for name in names}
        if self.record_contact_return:
            tensors["previous_s_flux"] = self.previous_s_flux
        if self.record_s_return:
            tensors["previous_s_return_weight"] = self.previous_s_return_weight
        if self.collect_diagnostics:
            tensors["diagnostic_totals"] = self.diagnostic_totals
        parameters = dict(self.body.named_parameters())
        for name in ("byte_embedding.weight", "event_value.weight", "event_likelihood.weight",
                     "event_likelihood.bias", "dynamics_value.weight",
                     "causal_address.weight", "column_keys", "reciprocal_contact",
                     "readout.weight", "readout.bias"):
            tensors[f"body.{name}"] = parameters[name]
        return tensors

    def _mutable(self) -> tuple[torch.Tensor, ...]:
        return tuple(self.mutable_tensor_map().values())

    def _synchronize(self) -> None:
        if self.activity.device.type == "cuda":
            torch.cuda.synchronize(self.activity.device)

    @torch.no_grad()
    def _step_event(self, token: torch.Tensor) -> None:
        """Run one prepared prediction, old receipt, then current assimilation.

        The elapsed prior and prediction contain no knowledge of ``token``.
        Its score first supplies the third factor for the already-issued local
        receipt.  Only then does the current byte enter H/C/S and issue the
        next receipt.  This keeps the shared field that generated an event
        available when its one-boundary credit is consumed, without a copied
        parameter page or a reverse-time graph.
        """
        if token.dtype != torch.long:
            raise ValueError("arriving language byte must be an integer index")
        target = token
        natural, precision = self.body._prepare_carrier_tensors(
            self.natural, self.precision, self.elapsed_zero,
        )
        (features, read_route, read_values, credit_token,
         eligibility) = self.body._predict_tensors(
            self.activity, natural, precision, self.causal_token, self.participation,
        )
        logits = self.body.readout(features)
        target_one_hot = self.one_hot_table.index_select(0, target)
        self.total_nll.add_(torch.logsumexp(logits, dim=-1) - (logits * target_one_hot).sum(dim=-1))

        # Form every event-map direction at the same unmodified body point.
        # The old flux is a one-boundary receipt; C supplies the preceding byte.
        # This common C-write step therefore adds no dynamics-specific rate.
        if self.value_step > 0:
            self.body._update_event_maps_from_tensors(
                logits, natural, precision, self.causal_token, self.participation,
                read_route, self.previous_contact_index, self.previous_contact_gate,
                self.previous_contact_unique, self.previous_s_flux, target,
                self.value_step, self.has_previous_contact,
            )
        # Every current score direction reads the issuing head.  Apply this
        # separate shared-field update only after event maps consumed its old rows.
        self.body._update_reciprocal_h_from_tensors(
            logits, read_route, self.previous_contact_index,
            self.previous_contact_pre, self.previous_contact_gate,
            self.previous_contact_unique, target, self.contact_step,
            self.has_previous_contact,
        )
        self.body._update_route_from_tensors(
            logits, features, read_route, read_values, credit_token,
            target, self.route_step,
        )
        innovation = self.body._language_innovation(logits, target)
        self.body.readout.weight.add_(innovation.transpose(0, 1) @ features,
                                      alpha=self.head_step)
        self.body.readout.bias.add_(innovation.sum(dim=0), alpha=self.head_step)

        # This is the shared-field snapshot used by the current transition.
        # It is one transient receipt for S's following return, not another
        # persistent body coordinate.
        if self.record_s_return:
            s_return_weight = self.body.reciprocal_contact[
                :self.body.d_model + 2, self.body._contact_s_input
            ].detach().clone()
        diagnostics = ({"diagnostic_totals": self.diagnostic_totals}
                       if self.collect_diagnostics else {})
        (activity, natural, precision, participation, write_route, edge_flux,
         contact_gate, contact_pre) = self.body._assimilate_tensors(
            self.activity, natural, precision, self.participation,
            token, self.external_source,
            **diagnostics,
        )
        if self.s_contact_step > 0:
            teaching = self.body._s_contact_return_from_tensors(
                logits, read_route, self.previous_contact_index,
                self.previous_contact_gate, self.previous_contact_unique,
                self.previous_s_flux, self.previous_s_return_weight,
                target, self.has_previous_contact,
            )
            evidence_participation = self.body.evidence_gated_s_exchange(
                participation, teaching, self.s_contact_step,
            )
            if self.collect_diagnostics:
                self.diagnostic_totals[4].add_(teaching.square().sum())
                self.diagnostic_totals[5].add_((evidence_participation - participation).square().sum())
            participation = evidence_participation
        contact_index, packed_pre, packed_gate, packed_unique = self.body._pack_local_receipt(
            contact_pre, contact_gate, write_route,
        )
        self.previous_contact_index.copy_(contact_index)
        self.previous_contact_pre.copy_(packed_pre)
        self.previous_contact_gate.copy_(packed_gate)
        self.previous_contact_unique.copy_(packed_unique)
        if self.record_contact_return:
            self.previous_s_flux.copy_(edge_flux)
        if self.record_s_return:
            self.previous_s_return_weight.copy_(s_return_weight)
        self.has_previous_contact.fill_(True)
        self.activity.copy_(activity)
        self.natural.copy_(natural)
        self.precision.copy_(precision)
        self.causal_token.copy_(token)
        self.participation.copy_(participation)

    @torch.no_grad()
    def _capture_block(self, length: int) -> CudaGraphBlock:
        device = self.activity.device
        tokens = torch.zeros(length, device=device, dtype=torch.long)
        snapshots = tuple(value.clone() for value in self._mutable())
        self._synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for index in range(length):
                self._step_event(tokens[index:index + 1])
        self._synchronize()
        for mutable, snapshot in zip(self._mutable(), snapshots):
            mutable.copy_(snapshot)
        self._synchronize()
        # ``CudaGraphBlock`` is shared with the three-head admission runner,
        # whose target buffer is real.  This online body uses only its arrival
        # buffer: the arrived byte is both scored label and the subsequent
        # H/C/S write.
        return CudaGraphBlock(graph, tokens, torch.empty(0, device=device, dtype=torch.long))

    @torch.no_grad()
    def _replay(self, values: np.ndarray) -> None:
        """Replay arriving bytes under the score-then-assimilate contract."""
        if len(values) == 0:
            return
        tokens = torch.as_tensor(values, device=self.activity.device, dtype=torch.long)
        if self.activity.device.type == "cuda":
            block = self.graphs.get(len(values))
            if block is None:
                raise ValueError("only captured CUDA graph lengths may be replayed")
            block.tokens.copy_(tokens)
            block.graph.replay()
        else:
            for index in range(len(values)):
                self._step_event(tokens[index:index + 1])
        self.scored_lifetime += len(values)
        for name in self.update_calls:
            if getattr(self, f"{name}_step") > 0:
                self.update_calls[name] += len(values)

    @torch.no_grad()
    @torch.no_grad()
    def advance_empty_time(self, elapsed: float = 1.0) -> None:
        """Advance C's process prior; elapsed units belong to the outer clock."""
        elapsed = float(elapsed)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("elapsed must be finite and nonnegative")
        if elapsed == 0:
            return
        mean = self.natural / self.precision
        process = F.softplus(self.body.process_variance_raw).to(dtype=self.precision.dtype)
        next_precision = self.precision / (1.0 + elapsed * process * self.precision)
        self.natural.copy_(next_precision * mean)
        self.precision.copy_(next_precision)
        self.advanced_elapsed += elapsed

    @torch.no_grad()
    def phase(self, raw_tokens: np.ndarray, report_every: int) -> dict:
        if report_every < 1:
            raise ValueError("report_every must be positive")
        self.total_nll.zero_()
        if self.collect_diagnostics:
            self.diagnostic_totals.zero_()
        scored = skipped = 0
        progress: list[dict] = []
        started = time.monotonic()
        index = 0
        while index < len(raw_tokens):
            if not BYTE_OFFSET <= int(raw_tokens[index]) < BYTE_OFFSET + BYTE_VOCAB:
                self.advance_empty_time()
                self.special_lifetime += 1
                skipped += 1; index += 1; continue
            run_start = index; index += 1
            while index < len(raw_tokens) and BYTE_OFFSET <= int(raw_tokens[index]) < BYTE_OFFSET + BYTE_VOCAB:
                index += 1
            tokens = np.asarray(raw_tokens[run_start:index], dtype=np.int64) - BYTE_OFFSET
            cursor = 0
            while cursor < len(tokens):
                width = min(self.block_size, len(tokens) - cursor,
                            report_every - (scored % report_every))
                if width == self.block_size:
                    self._replay(tokens[cursor:cursor + width])
                else:
                    # Only one-step and full-block graphs are captured.  A
                    # report/run boundary is a presentation concern, never a
                    # reason to invent a third transition semantics.
                    for byte in tokens[cursor:cursor + width]:
                        self._replay(np.asarray([byte], dtype=np.int64))
                cursor += width; scored += width
                if scored % report_every == 0:
                    self._synchronize()
                    progress.append({"scored_bytes": scored,
                                     "bits_per_byte": bpb(float(self.total_nll.item()), scored)})
        self._synchronize()
        seconds = time.monotonic() - started
        total = float(self.total_nll.item())
        result = {"scored_bytes": scored, "special_tokens_elapsed_without_lexical_write": skipped,
                  "bits_per_byte": bpb(total, scored), "mean_nll_nats": total / max(scored, 1),
                  "seconds": seconds, "bytes_per_second": scored / max(seconds, 1e-9), "progress": progress}
        if self.collect_diagnostics:
            result["diagnostic_totals"] = self.diagnostic_totals.cpu().tolist()
        return result

    def checkpoint_state(self) -> dict:
        """Tensor/primitive-only engine snapshot; outer driver owns RNG/cursors."""
        self._synchronize()
        return {
            "model_config": {
                "d_model": self.body.d_model, "grid_shape": [self.body.rows, self.body.cols],
                "active_columns": self.body.active_columns, "contact_rank": self.body.contact_rank,
                "vocab_size": self.body.byte_vocab,
                "dtype": str(self.body.column_keys.dtype).removeprefix("torch."),
                "initial_precision": self.body.initial_precision,
            },
            "profile": {
                "head_step": self.head_step, "value_step": self.value_step,
                "route_step": self.route_step, "contact_step": self.contact_step,
                "s_contact_step": self.s_contact_step, "record_s_return": self.record_s_return,
                "collect_diagnostics": self.collect_diagnostics, "block_size": self.block_size,
            },
            "body_state_dict": {name: value.detach().cpu().clone()
                                for name, value in self.body.state_dict().items()},
            "buffers": {name: value.detach().cpu().clone()
                        for name, value in self.mutable_tensor_map().items()
                        if not name.startswith("body.")},
            "lifetime": {
                "scored_lifetime": self.scored_lifetime, "special_lifetime": self.special_lifetime,
                "update_calls": dict(self.update_calls), "advanced_elapsed": self.advanced_elapsed,
            },
        }

    @classmethod
    @torch.no_grad()
    def from_checkpoint(cls, payload: dict, *, device: torch.device | str = "cpu",
                        s_contact_step: float | None = None) -> CudaGraphValueCredit:
        config = payload["model_config"]
        dtype = getattr(torch, config["dtype"])
        body = CGBColumnLanguageBody(
            config["d_model"], tuple(config["grid_shape"]),
            active_columns=config["active_columns"], contact_rank=config["contact_rank"],
            byte_vocab=config["vocab_size"],
        ).to(device=device, dtype=dtype).eval()
        body.initial_precision = config["initial_precision"]
        body.load_state_dict(payload["body_state_dict"])
        buffers = payload["buffers"]
        state = ColumnBodyState(*(buffers[name].to(device=device) for name in (
            "activity", "natural", "precision", "causal_token", "participation",
        )))
        profile = dict(payload["profile"])
        if s_contact_step is not None:
            profile["s_contact_step"] = float(s_contact_step)
        runner = cls(body, state, **profile)
        for name, value in runner.mutable_tensor_map().items():
            if not name.startswith("body."):
                value.copy_(buffers[name])
        lifetime = payload["lifetime"]
        runner.scored_lifetime = lifetime["scored_lifetime"]
        runner.special_lifetime = lifetime["special_lifetime"]
        runner.update_calls = dict(lifetime["update_calls"])
        runner.advanced_elapsed = lifetime["advanced_elapsed"]
        return runner

    def state(self) -> ColumnBodyState:
        return ColumnBodyState(*(value.detach().clone() for value in (
            self.activity, self.natural, self.precision, self.causal_token, self.participation,
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



@torch.no_grad()
def _eager_value_credit_phase(body: CGBColumnLanguageBody, state: ColumnBodyState,
                              raw_tokens: np.ndarray, head_step: float,
                              value_step: float, route_step: float, contact_step: float, report_every: int, *,
                              s_contact_step: float = 0.0,
                              previous: ColumnBodyStep | None = None) -> dict:
    """Independent eager reference for score-then-assimilate prequentiality."""
    total = 0.0; scored = skipped = 0; progress: list[dict] = []; started = time.monotonic()
    for raw in raw_tokens:
        if not BYTE_OFFSET <= int(raw) < BYTE_OFFSET + BYTE_VOCAB:
            state = body.advance_time(state, 1.0); skipped += 1; continue
        arrived = int(raw) - BYTE_OFFSET
        target = torch.tensor([arrived], device=state.activity.device, dtype=torch.long)
        # The eager reference deliberately follows the captured runner's
        # prepare → predict → old-receipt → assimilate ordering.  It is kept
        # separate so capture is never accepted merely by replaying itself.
        elapsed_zero = torch.zeros(1, device=state.activity.device, dtype=state.activity.dtype)
        natural, precision = body._prepare_carrier_tensors(
            state.natural, state.precision, elapsed_zero,
        )
        (features, read_route, read_values, credit_token,
         eligibility) = body._predict_tensors(
            state.activity, natural, precision, state.causal_token, state.participation,
        )
        logits = body.readout(features)
        total += float((torch.logsumexp(logits, dim=-1) - logits[0, target[0]]).item())
        if previous is not None:
            index, pre, gate, unique = body._pack_local_receipt(
                previous.contact_pre, previous.contact_response_gate, previous.route,
            )
            if value_step > 0:
                body._update_event_maps_from_tensors(
                    logits, natural, precision, state.causal_token, state.participation,
                    read_route, index, gate, unique, previous.edge_flux, target, value_step,
                )
            body._update_reciprocal_h_from_tensors(
                logits, read_route, index, pre, gate, unique,
                target, contact_step,
            )
        body._update_route_from_tensors(
            logits, features, read_route, read_values, credit_token, target, route_step,
        )
        innovation = body._language_innovation(logits, target)
        body.readout.weight.add_(innovation.transpose(0, 1) @ features, alpha=head_step)
        body.readout.bias.add_(innovation.sum(dim=0), alpha=head_step)
        return_weight = (
            body.reciprocal_contact[:body.d_model + 2, body._contact_s_input].detach().clone()
            if s_contact_step > 0 else None
        )
        source = body._source_index("external", 1, state.activity)
        (activity, natural, precision, participation, route, flux,
         gate, contact_pre) = body._assimilate_tensors(
            state.activity, natural, precision, state.participation, target, source,
        )
        prepared = ColumnBodyStep(
            state=ColumnBodyState(activity, natural, precision, target.clone(), participation),
            logits=logits, features=features, read_route=read_route, read_values=read_values,
            route=route, edge_flux=flux, credit_token=credit_token,
            value_eligibility=eligibility, contact_response_gate=gate,
            contact_pre=contact_pre, contact_return_weight=return_weight,
        )
        prepared = replace(prepared, state=body.update_s_participation_from_following(
            previous, prepared, target, step_size=s_contact_step,
        ))
        previous = prepared
        state = body.detach_state(prepared.state); scored += 1
        if scored % report_every == 0:
            progress.append({"scored_bytes": scored, "bits_per_byte": bpb(total, scored)})
    elapsed = time.monotonic() - started
    return {"scored_bytes": scored, "special_tokens_elapsed_without_lexical_write": skipped,
            "bits_per_byte": bpb(total, scored), "mean_nll_nats": total / max(scored, 1),
            "seconds": elapsed, "bytes_per_second": scored / max(elapsed, 1e-9),
            "progress": progress, "_state": state, "_previous": previous}

def run_pair(template: CGBColumnLanguageBody, train: np.ndarray, validation: np.ndarray, *,
             step_size: float, value_step: float, route_step: float, contact_step: float, report_every: int,
             device: torch.device, graph_block: int, s_contact_step: float = 0.0):
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
    credit_body = copy.deepcopy(template).to(device).eval()
    credit_initial = credit_body.initial_state(1, device=device)
    if device.type == "cuda":
        credit_runner = CudaGraphValueCredit(credit_body, credit_initial, step_size, value_step,
                                              route_step, contact_step, graph_block,
                                              s_contact_step=s_contact_step)
        credit_train = credit_runner.phase(train, report_every)
        credit_validation = credit_runner.phase(validation, report_every)
        credit_final = credit_runner.state()
        credit_execution = f"cuda_graph_block{graph_block}_independent_event_credit_body"
    else:
        # CPU is a correctness fallback. CUDA Graph is the production route.
        credit_state = credit_initial
        credit_train = _eager_value_credit_phase(credit_body, credit_state, train, step_size,
                                                  value_step, route_step, contact_step, report_every,
                                                  s_contact_step=s_contact_step)
        credit_state = credit_train.pop("_state")
        credit_previous = credit_train.pop("_previous")
        credit_validation = _eager_value_credit_phase(credit_body, credit_state, validation, step_size,
                                                       value_step, route_step, contact_step, report_every,
                                                       s_contact_step=s_contact_step,
                                                       previous=credit_previous)
        credit_final = credit_validation.pop("_state")
        credit_validation.pop("_previous")
        credit_execution = "eager_independent_event_credit_body"
    # The direct reciprocal update retains one issued event only.  Account for
    # it separately from organism H/C/S state so a fixed O(1) pipeline buffer
    # is never misreported as free state or as a long-history eligibility bank.
    one_event_receipt_bytes = (
        5 * credit_body.active_columns * credit_body.local_contact_input_dim * 4
        + 5 * credit_body.active_columns * credit_body.d_model
        + 5 * credit_body.active_columns * 9 + 1
    )
    extra_contact_flow_bytes = 0
    if value_step > 0 or s_contact_step > 0:
        extra_contact_flow_bytes = 4 * 2 * math.prod(credit_initial.activity.shape[:-1])
    extra_s_return_bytes = 0
    if s_contact_step > 0:
        extra_s_return_bytes = 4 * (credit_body.d_model + 2) * 8 * credit_body.contact_rank
    one_event_receipt_bytes += extra_contact_flow_bytes + extra_s_return_bytes
    credit_common = {"parameter_numbers": sum(parameter.numel() for parameter in credit_body.parameters()),
                     "persistent_state_bytes_fp32": credit_body.persistent_state_bytes(credit_final),
                     "one_event_reciprocal_receipt_bytes": one_event_receipt_bytes,
                     "extra_contact_flow_receipt_bytes": extra_contact_flow_bytes,
                     "extra_s_return_bytes": extra_s_return_bytes,
                     "s_contact_vjp_workspace": "temporary, fixed by model size, excluded from one-event receipt and persistent H/C/S bytes",
                     "shared_body_trajectory": False, "execution": credit_execution}

    return [
        {"name": "frozen_readout", "head_step_size": 0.0, **common,
         "train_first_pass": frozen_train, "validation_continuing_first_pass": frozen_validation},
        {"name": "bias_only_g14", "head_step_size": step_size, **common,
         "train_first_pass": bias_train, "validation_continuing_first_pass": bias_validation},
        {"name": "local_g14_readout", "head_step_size": step_size, **common,
         "train_first_pass": learned_train, "validation_continuing_first_pass": learned_validation},
        {"name": "local_g14_event_credit", "head_step_size": step_size,
         "value_step_size": value_step, "route_step_size": route_step,
         "contact_step_size": contact_step, "s_contact_step": s_contact_step, **credit_common,
         "train_first_pass": credit_train, "validation_continuing_first_pass": credit_validation},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "ib_owt_10000")
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--active-columns", type=int, default=8)
    parser.add_argument("--contact-rank", type=int, default=1,
                        help="number of bounded causal basis coordinates in the existing S carrier")
    parser.add_argument("--head-step", type=float, default=0.01)
    parser.add_argument("--value-step", type=float, default=0.001)
    parser.add_argument("--route-step", type=float, default=0.0,
                        help="admitted one-event causal-address/key credit; zero preserves prior comparison")
    parser.add_argument("--contact-step", type=float, default=0.0,
                        help="admitted one-next-byte shared H-contact credit; zero preserves prior comparison")
    parser.add_argument("--s-contact-step", type=float, default=0.0,
                        help="one-receipt conserved S transfer; zero disables its language-return gate")
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report-every", type=int, default=10_000)
    parser.add_argument("--graph-block", type=int, default=128,
                        help="ordered byte steps unrolled into each CUDA Graph replay")
    parser.add_argument("--mute-contact-readout", action="store_true",
                        help="ablate the S-to-H read while retaining the identical reciprocal S write")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "published" / "cgb_column_event_credit_owt100k_20261101.json")
    args = parser.parse_args()
    if (args.limit < 1 or args.report_every < 1 or args.graph_block < 1 or
            args.head_step < 0 or args.value_step < 0 or args.route_step < 0 or args.contact_step < 0 or
            not math.isfinite(args.s_contact_step) or args.s_contact_step < 0):
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
                                     active_columns=args.active_columns,
                                     contact_rank=args.contact_rank).to(device)
    if args.mute_contact_readout:
        # This retains the S carrier and its reciprocal update exactly.  Only
        # its old-state contribution to the succeeding local H rule is muted.
        # It is therefore a causal read ablation, rather than a smaller body.
        template.mute_contact_readout_()
    arms = run_pair(template, train, validation, step_size=args.head_step, value_step=args.value_step,
                    route_step=args.route_step, contact_step=args.contact_step,
                    s_contact_step=args.s_contact_step, report_every=args.report_every, device=device,
                    graph_block=args.graph_block)
    by_name = {arm["name"]: arm for arm in arms}
    frozen = by_name["frozen_readout"]
    bias_only = by_name["bias_only_g14"]
    learned = by_name["local_g14_readout"]
    event_credit = by_name["local_g14_event_credit"]
    report = {
        "status": "column_body_language_admission",
        "claim_scope": "Real OWT UTF-8 byte prequential comparison of frozen, bias-only G14, and H/C/S-feature G14 readouts on one continuous H/C/S column body.",
        "s_contact_step": args.s_contact_step,
        "s_contact_scope": "One-receipt language evidence redistributes only the existing finite S pool: positive return admits from its implicit free partner and negative return releases the indicated factor. It is not BPTT, an arbitrary-horizon gradient, or proof of true future benefit.",
        "not_claimed": ["full body parameter learning", "long-delay three-factor credit",
                          "active language acquisition", "general language-model capability", "life closure",
                          "exact long-horizon S-contact gradients", "proved true future benefit of S transfer"],
        "protocol": {"prediction": "score before target write", "state": "one H/C/S trajectory across train then validation; frozen/G14 readout heads do not feed back into H/C/S",
                     "special_tokens": "advance empty time without lexical write", "autograd_history": "absent",
                     "online_update": "G14 affine readout; the event-credit arm atomically applies exact one-boundary C-value, C-likelihood, bounded H-flow dynamics and their summed byte-embedding directions before its reciprocal H/contact, route and head updates. Optional S transfer changes only existing typed/rank S participation: positive return admits from its implicit local free pool and negative return releases the indicated factor. All directions use the issuing pre-update head; no BPTT or history-sized gradient is retained.", "action": "absent",
                     "execution": arms[0]["execution"]},
        "model": {"runtime": "CGBColumnLanguageBody", "d_model": args.d_model,
                  "grid_shape": [args.grid, args.grid], "active_columns": args.active_columns,
                  "contact_rank": args.contact_rank,
                  "device": str(device), "seed": args.seed, "cuda_graph_block": args.graph_block,
                  "contact_readout": "muted_S_to_H" if args.mute_contact_readout else "enabled_S_to_H"},
        "source": {"train": train_source, "validation": validation_source},
        "arms": arms,
        "effect_of_g14": {"train_bpb_delta_vs_frozen": learned["train_first_pass"]["bits_per_byte"] - frozen["train_first_pass"]["bits_per_byte"],
                          "validation_bpb_delta_vs_frozen": learned["validation_continuing_first_pass"]["bits_per_byte"] - frozen["validation_continuing_first_pass"]["bits_per_byte"],
                          "train_bpb_delta_vs_bias_only": learned["train_first_pass"]["bits_per_byte"] - bias_only["train_first_pass"]["bits_per_byte"],
                          "validation_bpb_delta_vs_bias_only": learned["validation_continuing_first_pass"]["bits_per_byte"] - bias_only["validation_continuing_first_pass"]["bits_per_byte"]},
        "effect_of_event_credit": {"train_bpb_delta_vs_head_only": event_credit["train_first_pass"]["bits_per_byte"] - learned["train_first_pass"]["bits_per_byte"],
                                    "validation_bpb_delta_vs_head_only": event_credit["validation_continuing_first_pass"]["bits_per_byte"] - learned["validation_continuing_first_pass"]["bits_per_byte"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "effect_of_g14": report["effect_of_g14"],
                      "validation_bpb": {arm["name"]: arm["validation_continuing_first_pass"]["bits_per_byte"] for arm in arms}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
