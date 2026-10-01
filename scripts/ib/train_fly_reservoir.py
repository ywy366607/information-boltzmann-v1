"""Level-0 fly-reservoir language training on the registered OWT protocol.

The MaleCNS v1.0 wiring is frozen (LIF substrate, signed synapses, one event
per token); only the broadcast input projection and the weighted readout are
trained.  The reservoir state persists across chunks and updates and is never
reset within training; evaluation sites warm the state with 256 observed
tokens from zero (echo-state convention - an LIF substrate has no NESS
spectrum to re-sample) and score the next 128.

Performance discipline: transmission is gather + index_add along real
edges (a CSR sparse.mm backward is not CUDA-graph capturable here); the
training chunk (forward+backward) is captured in a CUDA Graph; validation
processes the four sites as a batch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from information_boltzmann.core.fly_reservoir import FlyReservoirLM


def atomic_json(path: Path, payload: dict, *, required: bool = False) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    encoded = json.dumps(payload, allow_nan=False)
    for attempt in range(60):
        try:
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, path)
            return
        except PermissionError:
            time.sleep(0.05 * min(attempt + 1, 4))
    if required:
        raise PermissionError(f"Could not update {path}")


class GraphChunkRunner:
    """Captures one training chunk (forward + backward) in a CUDA Graph."""

    def __init__(self, model: FlyReservoirLM, chunk_tokens: int,
                 optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.chunk = chunk_tokens
        self.ids = torch.zeros(1, chunk_tokens, dtype=torch.long, device="cuda")
        self.targets = torch.zeros_like(self.ids)
        self.h_buf = torch.zeros(1, model.n_neurons, device="cuda")

        original_parameters = [p.detach().clone() for p in model.parameters()]

        def chunk_fn():
            loss, h_next, diag = model.forward_chunk(self.ids, self.targets, self.h_buf)
            loss.backward()
            return loss, h_next, diag

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.optimizer.zero_grad(set_to_none=False)
                chunk_fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.empty_cache()
        with torch.no_grad():
            for parameter, original in zip(model.parameters(), original_parameters):
                parameter.copy_(original)
        self.optimizer.zero_grad(set_to_none=False)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss, self.h_next, self.diag = chunk_fn()

    def replay(self, ids: torch.Tensor, targets: torch.Tensor, h: torch.Tensor):
        self.ids.copy_(ids)
        self.targets.copy_(targets)
        self.h_buf.copy_(h.detach())
        self.graph.replay()
        return self.loss.detach(), self.h_next.detach(), self.diag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    parser.add_argument("--graph", type=Path,
                        default=Path("data/malecns_v1/fly_reservoir_full.npz"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/q8_fly_reservoir_level0_3000"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--chunk-tokens", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--leak", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--injection", choices=("broadcast", "sensory"),
                        default="broadcast",
                        help="broadcast drives every neuron; sensory drives "
                             "only the 15,912 annotated sensory neurons - "
                             "information must then flow through the wiring")
    parser.add_argument("--read-surface", choices=("all", "interneuron"),
                        default="all",
                        help="interneuron: the readout sees only non-injected "
                             "neurons - write and read surfaces are separated")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--site-starts", type=str, default="8192,12288,16384,20480")
    parser.add_argument("--warm-in-tokens", type=int, default=256)
    parser.add_argument("--score-tokens", type=int, default=128)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required for the fly-reservoir trainer")
    if args.tokens % args.chunk_tokens:
        parser.error("tokens must be a multiple of chunk-tokens")

    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "last.pt").exists() and args.resume is None:
        parser.error("Output already contains a run; pass --resume or choose a new directory")

    train = np.load(args.data / "train.npy", mmap_mode="r")
    validation = np.load(args.data / "validation.npy", mmap_mode="r")
    site_starts = tuple(int(s) for s in args.site_starts.split(","))
    n_sites = len(site_starts)

    model = FlyReservoirLM(
        args.graph, vocab_size=50257, d_model=args.d_model,
        leak=args.leak, threshold=args.threshold,
        injection=args.injection,
        read_surface=args.read_surface).cuda()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    config = {
        "architecture": "FlyReservoir-level0-malecns-signed-LIF",
        "graph": str(args.graph),
        "graph_sha256": hashlib.sha256(args.graph.read_bytes()).hexdigest(),
        "neurons": model.n_neurons,
        "edges": int(model.edge_weight.numel()),
        "d_model": args.d_model, "leak": args.leak, "threshold": args.threshold,
        "tokens": args.tokens, "bptt_chunk_tokens": args.chunk_tokens,
        "steps": args.steps, "lr": args.lr, "optimizer": "AdamW",
        "cuda_graph": True,
        "frozen": "all wiring; trained = broadcast input projection, weighted readout, tied embedding",
        "injection": args.injection, "injection_neurons": model.n_injection,
        "read_surface": args.read_surface,
        "state_policy": "reservoir state persists across chunks and updates; never reset",
        "evaluation_protocol": "IB-warm-local-language-v1 reservoir variant: state from zero, 256 warm-in, 128 scored, four sites batched",
        "sites": site_starts,
        "seed": 11,
        "parameter_count_trainable": sum(p.numel() for p in trainable),
    }
    atomic_json(args.output / "config.json", config, required=True)

    step, best = 0, float("inf")

    def save(name: str) -> None:
        temporary = args.output / f"{name}.{os.getpid()}.tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "state": h.detach().clone(), "step": step,
                    "best_validation_nll": best, "config": config}, temporary)
        os.replace(temporary, args.output / name)

    def log(row: dict) -> None:
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    @torch.no_grad()
    def validate() -> float:
        """Four sites processed as one batch; state from zero, warm-in 256,
        score 128."""
        model.eval()
        starts = torch.tensor(site_starts, device="cuda")
        h = torch.zeros(n_sites, model.n_neurons, device="cuda")
        per_site = torch.zeros(n_sites, device="cuda")
        total = args.warm_in_tokens + args.score_tokens
        for offset in range(total):
            tokens = torch.tensor(
                [int(validation[int(s) + offset]) for s in starts], device="cuda")
            h, _ = model.step(h, tokens)
            if offset >= args.warm_in_tokens:
                targets = torch.tensor(
                    [int(validation[int(s) + offset + 1]) for s in starts],
                    device="cuda")
                logits = model.read(h)
                per_site += torch.nn.functional.cross_entropy(
                    logits, targets, reduction="none")
        model.train()
        return float(per_site.mean() / args.score_tokens)

    runner = GraphChunkRunner(model, args.chunk_tokens, optimizer)
    h = torch.zeros(1, model.n_neurons, device="cuda")

    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cuda", weights_only=False)
        for key in ("neurons", "d_model", "leak", "threshold", "tokens"):
            if saved["config"].get(key) != config.get(key):
                parser.error(f"Resume mismatch: {key}")
        graph_keys = {"edge_index", "edge_pre", "edge_post", "edge_weight"}
        model.load_state_dict({k: v for k, v in saved["model"].items()
                               if k not in graph_keys})
        optimizer.load_state_dict(saved["optimizer"])
        h = saved["state"].cuda().clone()
        step, best = int(saved["step"]), float(saved["best_validation_nll"])
    else:
        atomic_json(args.output / "progress.json",
                    {"status": "initializing", "step": 0, "target_steps": args.steps},
                    required=True)
        initial = validate()
        best = initial
        log({"kind": "validation", "step": 0, "validation_nll": initial})
        save("BBest.pt")
        save("last.pt")

    try:
        while step < args.steps:
            offset = step * args.tokens
            ids = torch.from_numpy(
                np.array(train[offset:offset + args.tokens]).astype(np.int64)).cuda()[None]
            targets = torch.from_numpy(
                np.array(train[offset + 1:offset + args.tokens + 1]).astype(np.int64)).cuda()[None]
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=False)
            loss_sum = 0.0
            for begin in range(0, args.tokens, args.chunk_tokens):
                chunk_loss, h, diag = runner.replay(
                    ids[:, begin:begin + args.chunk_tokens],
                    targets[:, begin:begin + args.chunk_tokens], h)
                loss_sum += float(chunk_loss)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            if not math.isfinite(float(grad_norm)):
                raise FloatingPointError("Non-finite gradient norm")
            optimizer.step()
            torch.cuda.synchronize()
            step += 1
            elapsed = time.perf_counter() - started
            if not math.isfinite(loss_sum):
                raise FloatingPointError("Non-finite training loss")
            if step == 1 or step % 10 == 0 or step == args.steps:
                row = {"kind": "train", "step": step, "events": step * args.tokens,
                       "nll": loss_sum / (args.tokens // args.chunk_tokens),
                       "seconds": elapsed,
                       "firing_rate": float(diag["firing_rate"]),
                       "neuron_activity_pr": float(diag["neuron_activity_pr"]),
                       "input_weight_pr": float(diag["input_weight_pr"]),
                       "output_weight_pr": float(diag["output_weight_pr"]),
                       "grad_norm": float(grad_norm)}
                log(row)
                atomic_json(args.output / "progress.json",
                            {"status": "running", "target_steps": args.steps, **row})
            if step % args.validate_every == 0 or step == args.steps:
                score = validate()
                if score < best:
                    best = score
                    save("BBest.pt")
                log({"kind": "validation", "step": step, "validation_nll": score,
                     "best_validation_nll": best})
                save("last.pt")
        atomic_json(args.output / "progress.json",
                    {"status": "complete", "step": step, "target_steps": args.steps,
                     "best_validation_nll": best}, required=True)
    except BaseException as error:
        atomic_json(args.output / "progress.json",
                    {"status": "failed", "step": step, "target_steps": args.steps,
                     "error": str(error)}, required=True)
        raise


if __name__ == "__main__":
    main()
