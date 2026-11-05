"""Frozen, paired context evaluation; never updates a training individual.

Reuse the production logits/state transition. CUDA Graph bounds execution
workspace independently of prefix length. This measures context utility, not
online plasticity, semantic task performance, or organism admission.
"""
from __future__ import annotations

import os
import sys

# Direct execution must not let ib_local/types.py shadow the stdlib module.
if sys.path and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.gated_deltanet import GatedDeltaNetLM


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def target_positions(size, samples, max_prefix, scored, exclude):
    """Disjoint maximal windows, spread across the available stream.

    Position t denotes the first *target*, not its preceding input. The first
    target has exactly prefix known bytes; all context arms score identical t's.
    """
    if min(samples, max_prefix, scored) < 1 or exclude < 0:
        raise ValueError("positive samples/prefix/scored and nonnegative exclude required")
    width = max_prefix + scored
    if size - exclude < samples * width:
        raise ValueError("not enough held-out bytes for disjoint windows")
    starts = np.linspace(exclude, size - width, samples, dtype=np.int64)
    return starts + max_prefix


def paired_summary(differences):
    """Region-level paired uncertainty, conditional on this trained seed.

    This does not estimate training-seed uncertainty or make regions iid facts.
    """
    values = np.asarray(differences, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("finite paired region differences required")
    ci = None
    if len(values) > 1:
        rng = np.random.default_rng(11)
        means = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
        ci = np.quantile(means, [.025, .975]).tolist()
    return dict(mean=float(values.mean()), region_differences=values.tolist(),
                bootstrap95=ci, uncertainty="paired stream regions, one trained seed; descriptive")


def aligned_revisit_blocks(config, rows):
    """Recover matched pre-update means without inventing per-byte logs.

    Exclude every log block ending at a phase boundary: its final successor
    can jump to another source interval. Coarse recorded blocks remain coarse.
    """
    phases = {p["name"]: p for p in config["phases"]}
    encounters = {}
    for name in ("A_first", "A_revisit"):
        phase = phases[name]
        intervals = {}
        for row in rows:
            if row.get("kind") != "train" or row.get("phase") != name:
                continue
            stop = row["step"]
            start = stop - row["window_updates"]
            if start < phase["start_update"] or stop >= phase["stop_update"]:
                continue
            first = phase["source_byte_start"] + (start-phase["start_update"])*config["tokens"] + 1
            end = first + (stop-start)*config["tokens"]
            key = (first, end)
            if key in intervals:
                raise ValueError("duplicate recorded target interval")
            intervals[key] = row["window_nll"]
        encounters[name] = intervals
    common = sorted(set(encounters["A_first"]) & set(encounters["A_revisit"]))
    blocks = [dict(source_target_start=a, source_target_stop=b,
                   first_nll=encounters["A_first"][(a,b)], revisit_nll=encounters["A_revisit"][(a,b)])
              for a,b in common]
    count = sum(b-a for a,b in common)
    if not count:
        raise ValueError("no exactly matched recorded revisit intervals")
    first = sum((r["source_target_stop"]-r["source_target_start"])*r["first_nll"] for r in blocks)/count
    revisit = sum((r["source_target_stop"]-r["source_target_start"])*r["revisit_nll"] for r in blocks)/count
    return dict(blocks=blocks, matched_target_bytes=count, first_nll=first,
                revisit_nll=revisit, nll_saving=first-revisit,
                recovery_time=None, resolution="original logged block averages; no per-byte reconstruction",
                interpretation="revisit benefit after further learning; not isolated retention or latch causality")


class FrozenScorer:
    def __init__(self, model, batch, chunk, use_graph=True):
        self.model = model.eval()
        self.batch, self.chunk = batch, chunk
        self.device = next(model.parameters()).device
        self.state = model.initial_state(batch)
        self.ids = torch.zeros(batch, chunk, dtype=torch.long, device=self.device)
        self.targets = torch.zeros_like(self.ids)
        self.graph = None
        if use_graph and self.device.type == "cuda":
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.no_grad(), torch.cuda.stream(stream):
                for _ in range(3):
                    self._forward()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(self.graph):
                self._forward()

    def _forward(self):
        logits, self.next_state, _ = self.model.predict_logits(self.ids, self.state)
        self.nll = F.cross_entropy(logits.flatten(0, 1), self.targets.flatten(),
                                   reduction="none").view_as(self.targets)
        self.correct = logits.argmax(-1).eq(self.targets)

    @torch.no_grad()
    def continuations(self, prompts, targets):
        """Score variable-length byte continuations, with no leading padding.

        All greedy target bytes correct is exact greedy-prefix agreement at the
        gold target length. It does not verify generation of a word delimiter.
        Right padding is only consumed after a row's scored continuation ends.
        """
        if not 0 < len(prompts) == len(targets) <= self.batch:
            raise ValueError("one nonempty prompt/target pair per active batch row")
        if any(not p or not t for p, t in zip(prompts, targets)):
            raise ValueError("empty prompt or continuation")
        lengths = [len(p) + len(t) for p, t in zip(prompts, targets)]
        size = ((max(lengths)-1 + self.chunk-1)//self.chunk)*self.chunk
        values = np.zeros((self.batch, size+1), dtype=np.int64)
        mask = np.zeros((self.batch, size), dtype=bool)
        for row, (prompt, target) in enumerate(zip(prompts, targets)):
            values[row, :len(prompt)+len(target)] = np.frombuffer(prompt+target, dtype=np.uint8)
            mask[row, len(prompt)-1:len(prompt)+len(target)-1] = True
        values = torch.as_tensor(values, device=self.device)
        mask = torch.as_tensor(mask, device=self.device)
        self.state.copy_(self.model.initial_state(self.batch))
        sums = torch.zeros(self.batch, dtype=torch.float64, device=self.device)
        matches = torch.ones(self.batch, dtype=torch.bool, device=self.device)
        for at in range(0, size, self.chunk):
            self.ids.copy_(values[:, at:at+self.chunk])
            self.targets.copy_(values[:, at+1:at+self.chunk+1])
            if self.graph is None:
                self._forward()
            else:
                self.graph.replay()
            self.state.copy_(self.next_state)
            selected = mask[:, at:at+self.chunk]
            sums.add_(self.nll.double().masked_fill(~selected, 0).sum(1))
            matches.logical_and_((self.correct | ~selected).all(1))
        count = len(prompts)
        result = sums[:count].cpu().tolist()
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite continuation scores")
        return result, matches[:count].cpu().tolist()

    @torch.no_grad()
    def score(self, data, positions, prefix, scored):
        if prefix % self.chunk or scored % self.chunk or prefix < self.chunk:
            raise ValueError("prefix and scored must be positive chunk multiples")
        positions = np.asarray(positions)
        if len(positions) != self.batch or (positions < prefix).any() or (positions + scored > len(data)).any():
            raise ValueError("invalid evaluation positions")
        self.state.copy_(self.model.initial_state(self.batch))
        nll = torch.zeros(self.batch, device=self.device, dtype=torch.float64)
        correct = torch.zeros(self.batch, device=self.device, dtype=torch.float64)
        horizons = sorted({min(scored, h) for h in (32, 128, 512, scored)})
        horizon_sums = {h: torch.zeros_like(nll) for h in horizons}
        # x[t-1] predicts target x[t]. Start prefix bytes before target t.
        starts = positions - prefix
        for at in range(0, prefix + scored, self.chunk):
            indices = starts[:, None] + at + np.arange(self.chunk)[None, :]
            # The last prediction is unused; cap its dummy target at array end.
            self.ids.copy_(torch.as_tensor(data[indices], device=self.device))
            self.targets.copy_(torch.as_tensor(data[np.minimum(indices + 1, len(data)-1)], device=self.device))
            if self.graph is None:
                self._forward()
            else:
                self.graph.replay()
            self.state.copy_(self.next_state)
            lo, hi = max(0, prefix-1-at), min(self.chunk, prefix+scored-1-at)
            if hi > lo:
                nll.add_(self.nll[:, lo:hi].double().sum(1))
                correct.add_(self.correct[:, lo:hi].sum(1))
            for horizon in horizons:
                end = min(self.chunk, prefix+horizon-1-at)
                if end > lo:
                    horizon_sums[horizon].add_(self.nll[:, lo:end].double().sum(1))
        scores = (nll / scored).cpu().numpy()
        if not np.isfinite(scores).all() or not bool(torch.isfinite(self.state).all()):
            raise FloatingPointError("nonfinite frozen evaluation")
        return dict(region_nll=scores.tolist(), nll_nats_per_byte=float(scores.mean()),
                    bits_per_byte=float(scores.mean()/math.log(2)),
                    target_horizons={str(h): dict(region_nll=(s/h).cpu().tolist(),
                                                  nll_nats_per_byte=float((s/h).mean().cpu()))
                                     for h, s in horizon_sums.items()},
                    next_byte_accuracy=float(correct.sum().cpu())/(scored*self.batch),
                    scored_bytes=scored*self.batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefixes", type=int, nargs="+", default=[32, 128, 512, 2048, 8192, 32768])
    parser.add_argument("--scored", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--exclude", type=int, default=16416,
                        help="exclude validation prefix previously used for checkpoint selection")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    manifest = json.loads((args.data/"manifest.json").read_text(encoding="utf-8"))
    tokenizer = json.loads((args.data/"tokenizer.json").read_text(encoding="utf-8"))
    source = args.data/"validation.npy"
    source_hash = sha256(source)
    if source_hash != manifest["files"]["validation.npy"] or tokenizer["type"] != "utf8_bytes":
        raise ValueError("source hash or tokenizer mismatch")
    raw = np.load(source, mmap_mode="r")
    data = np.asarray(raw[raw >= tokenizer["offset"]], dtype=np.int64) - tokenizer["offset"]
    if (data < 0).any() or (data > 255).any():
        raise ValueError("invalid byte data")
    positions = target_positions(len(data), args.samples, max(args.prefixes), args.scored, args.exclude)
    snapshot = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GatedDeltaNetLM(**snapshot["config"]["model"]).to(args.device)
    model.load_state_dict(snapshot["model"], strict=True)
    if model.vocab_size != 256:
        raise ValueError("byte evaluator requires 256 output classes")
    scorer = FrozenScorer(model, args.samples, args.chunk)
    started = time.perf_counter()
    rows = []
    for prefix in sorted(set(args.prefixes)):
        result = scorer.score(data, positions, prefix, args.scored)
        rows.append(dict(prefix_bytes=prefix, **result))
        print(json.dumps(rows[-1]), flush=True)
    short = np.asarray(rows[0]["region_nll"])
    for row in rows:
        row["context_gain_over_shortest"] = paired_summary(short - np.asarray(row["region_nll"]))
    report = dict(protocol="frozen_paired_real_stream_context_v1", training=False,
                  checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
                  checkpoint_step=snapshot["step"], checkpoint_events=snapshot["events"],
                  model_config=model.config_dict(), source_revision=manifest["revision"],
                  source_sha256=source_hash, converted_stream_sha256=hashlib.sha256(data.astype(np.uint8).tobytes()).hexdigest(),
                  evaluator_sha256=sha256(__file__), body_sha256=sha256(ROOT/"scripts/ib_local/gated_deltanet.py"),
                  checkpoint_body_sha256=snapshot["config"].get("code_sha256", {}),
                  target_positions=positions.tolist(), scored_bytes_per_region=args.scored,
                  excluded_selection_prefix=args.exclude, execution_chunk=args.chunk,
                  context_is="concatenated OWT validation stream; may cross document boundaries",
                  state_policy="fresh sidecar per region and prefix; recurrent state continuous within each; weights frozen",
                  live_learning_evaluated=False, semantic_downstream_evaluated=False,
                  converged=False, architecture_verdict="undetermined", independent_review=False,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  single_individual_state_bytes=model.initial_state(1).numel()*4,
                  seconds=time.perf_counter()-started, results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
