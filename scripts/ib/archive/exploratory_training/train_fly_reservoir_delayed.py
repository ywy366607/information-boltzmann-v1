"""Biological multi-delay fly-reservoir language training on the registered OWT protocol.

In this model, the fruit fly central nervous system (MaleCNS v1.0, 165,122 neurons,
25.5 million synapses) is augmented with realistic axonal conduction delays derived
from 3D physical distances (0.3 m/s conduction velocity + 0.8 ms synaptic delay):
  - 1 ms delay: 82.6% (21.1M local microcircuits)
  - 2 ms delay: 15.2% (3.89M neuropil projections)
  - 3 ms delay:  1.8% (451k long-range circuits)
  - 4 ms delay:  0.4% (108k descending highways)

The multi-delay ring buffer runs inside fused Triton kernels with zero extra VRAM
allocations and 100% CUDA Graph capture compatibility.
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


class GraphChunkRunnerDelayed:
    """Captures one training chunk (forward + backward) in a CUDA Graph with ring buffer and continuous synaptic conductance or current."""

    def __init__(self, model: FlyReservoirLM, chunk_tokens: int,
                 optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.is_coba = (model.synapse_model == "coba")
        self.use_alif = getattr(model, "use_alif", False)
        self.use_stp = getattr(model, "use_stp", False)
        self.chunk = chunk_tokens
        device = "cuda"
        self.ids = torch.zeros(1, self.chunk, dtype=torch.long, device=device)
        self.targets = torch.zeros_like(self.ids)
        self.h_buf = torch.zeros(1, model.n_neurons, device=device)
        self.ring_bufs = [torch.zeros(1, model.n_neurons, device=device) for _ in range(4)]
        if self.is_coba:
            self.ge_buf = torch.zeros(1, model.n_neurons, device=device)
            self.gi_buf = torch.zeros(1, model.n_neurons, device=device)
        else:
            self.i_syn_buf = torch.zeros(1, model.n_neurons, device=device)
        if self.use_alif:
            self.b_buf = torch.zeros(1, model.n_neurons, device=device)
        if self.use_stp:
            self.x_buf = torch.ones(1, model.n_neurons, device=device)
            self.u_buf = model.get_stp_params()[0].detach().clone().expand(1, model.n_neurons).contiguous()

        original_parameters = [p.detach().clone() for p in model.parameters()]

        def chunk_fn():
            fwd_kwargs = {}
            if self.is_coba:
                fwd_kwargs["ge"] = self.ge_buf
                fwd_kwargs["gi"] = self.gi_buf
            else:
                fwd_kwargs["i_syn"] = self.i_syn_buf
            if self.use_alif:
                fwd_kwargs["b"] = self.b_buf
            if self.use_stp:
                fwd_kwargs["x"] = self.x_buf
                fwd_kwargs["u"] = self.u_buf

            res = model.forward_chunk(self.ids, self.targets, self.h_buf, tuple(self.ring_bufs), **fwd_kwargs)
            loss = res[0]
            loss.backward()
            return res

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=False)
                chunk_fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.empty_cache()
        with torch.no_grad():
            for parameter, original in zip(model.parameters(), original_parameters):
                parameter.copy_(original)
        optimizer.zero_grad(set_to_none=False)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = chunk_fn()

    def replay(self, ids: torch.Tensor, targets: torch.Tensor, h: torch.Tensor,
               ring: tuple[torch.Tensor, ...], *syn_state):
        self.ids.copy_(ids)
        self.targets.copy_(targets)
        self.h_buf.copy_(h.detach())
        for buf, r in zip(self.ring_bufs, ring):
            buf.copy_(r.detach())
        idx = 0
        if self.is_coba:
            self.ge_buf.copy_(syn_state[idx].detach()); idx += 1
            self.gi_buf.copy_(syn_state[idx].detach()); idx += 1
        else:
            self.i_syn_buf.copy_(syn_state[idx].detach()); idx += 1
        if self.use_alif:
            self.b_buf.copy_(syn_state[idx].detach()); idx += 1
        if self.use_stp:
            self.x_buf.copy_(syn_state[idx].detach()); idx += 1
            self.u_buf.copy_(syn_state[idx].detach()); idx += 1

        self.graph.replay()

        loss = self.output[0].detach()
        h_next = self.output[1].detach()
        diag = self.output[2]
        ring_next = tuple(r.detach() for r in self.output[3])
        ret = [loss, h_next, diag, ring_next]
        out_idx = 4
        if self.is_coba:
            ret.append(self.output[out_idx].detach()); out_idx += 1
            ret.append(self.output[out_idx].detach()); out_idx += 1
        else:
            ret.append(self.output[out_idx].detach()); out_idx += 1
        if self.use_alif:
            ret.append(self.output[out_idx].detach()); out_idx += 1
        if self.use_stp:
            ret.append(self.output[out_idx].detach()); out_idx += 1
            ret.append(self.output[out_idx].detach()); out_idx += 1
        return tuple(ret)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    default_graph = (Path("data/malecns_v1/fly_reservoir_coba.npz")
                     if Path("data/malecns_v1/fly_reservoir_coba.npz").exists()
                     else Path("data/malecns_v1/fly_reservoir_delayed.npz"))
    parser.add_argument("--graph", type=Path, default=default_graph)
    parser.add_argument("--synapse-model", choices=("auto", "coba", "cuba"), default="auto")
    parser.add_argument("--output", type=Path,
                        default=Path("results/q8_fly_reservoir_topographic_coba_3000"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--chunk-tokens", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--leak", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--injection", choices=("broadcast", "sensory", "topographic"),
                        default="topographic",
                        help="sensory: drives 15,912 sensory neurons; topographic: biological partitioned W_bio writer")
    parser.add_argument("--read-surface", choices=("all", "interneuron", "output"),
                        default="output",
                        help="output: reads from the 2,333 descending/motor output neurons")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--site-starts", type=str, default="8192,12288,16384,20480")
    parser.add_argument("--warm-in-tokens", type=int, default=256)
    parser.add_argument("--score-tokens", type=int, default=128)
    parser.add_argument("--alif", action="store_true", default=True, help="Enable ALIF slow threshold adaptation")
    parser.add_argument("--no-alif", dest="alif", action="store_false")
    parser.add_argument("--stp", action="store_true", default=True, help="Enable Tsodyks-Markram short-term synaptic plasticity")
    parser.add_argument("--no-stp", dest="stp", action="store_false")
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
        read_surface=args.read_surface,
        synapse_model=args.synapse_model,
        use_alif=args.alif,
        use_stp=args.stp).cuda()
    if not model.has_delays:
        parser.error("Graph does not contain delay_splits; run build_fly_reservoir_delayed_graph.py first")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    if model.synapse_model == "coba":
        tag = ("ALIF-STP" if (args.alif and args.stp)
               else "ALIF" if args.alif
               else "STP" if args.stp
               else "LIF")
        if args.injection == "topographic":
            arch_name = f"FlyReservoir-level0-malecns-topographic-coba-plastic-{tag}"
        else:
            arch_name = f"FlyReservoir-level0-malecns-coba-plastic-{tag}"
    elif args.injection == "topographic":
        arch_name = "FlyReservoir-level0-malecns-topographic-synapse-LIF"
    elif getattr(model, "has_biological_leak", False):
        arch_name = "FlyReservoir-level0-malecns-continuous-synapse-LIF"
    else:
        arch_name = "FlyReservoir-level0-malecns-delayed-LIF"

    is_coba = (model.synapse_model == "coba")
    n_edges = (int(model.edge_weight_e.numel() + model.edge_weight_i.numel())
               if is_coba else int(model.edge_weight.numel()))
    splits_list = list(model.splits_e) if is_coba else list(model.delay_splits)

    config = {
        "architecture": arch_name,
        "synapse_model": model.synapse_model,
        "graph": str(args.graph),
        "graph_sha256": hashlib.sha256(args.graph.read_bytes()).hexdigest(),
        "neurons": model.n_neurons,
        "edges": n_edges,
        "delays": "biological 3D distance delays (1ms, 2ms, 3ms, 4ms)",
        "delay_splits": splits_list,
        "has_biological_leak": bool(getattr(model, "has_biological_leak", False)),
        "has_dopamine": bool(getattr(model, "has_dopamine", False)),
        "dan_edges": int(model.dan_edge_weight.numel()) if getattr(model, "has_dopamine", False) else 0,
        "synaptic_coupling": "COBA_Ohmic_non_linear (I = g_E*g_e*(E_E - V) + g_I*g_i*(E_I - V))" if is_coba else "CUBA_current",
        "surrogate_gradient": "Peak-normalized ATan (d/dx = 1 / (1 + (pi * x)^2))",
        "plasticity": "27-superclass learned timescales, 27-superclass learned thresholds, global learned conductance gains" + (", ALIF slow adaptation (Bellec et al. 2018)" if args.alif else "") + (", STP Tsodyks-Markram (1998) dynamic synapses" if args.stp else ""),
        "adaptive_threshold": "ALIF-27superclass (b_{t+1}=rho*b_t + (1-rho)*s_t, theta_t=theta_0+beta*b_t)" if args.alif else "none",
        "short_term_plasticity": "Tsodyks-Markram-27superclass (u_{t+1}=U_0+(u-U_0)*rho_fac, x_{t+1}=1+(x-1)*rho_rec)" if args.stp else "none",
        "alif": bool(getattr(model, "use_alif", False)),
        "stp": bool(getattr(model, "use_stp", False)),
        "d_model": args.d_model, "leak": "Scheme_B_topological_continuous" if getattr(model, "has_biological_leak", False) else args.leak, "threshold": "27-superclass-learned",
        "tokens": args.tokens, "bptt_chunk_tokens": args.chunk_tokens,
        "steps": args.steps, "lr": args.lr, "optimizer": "AdamW",
        "cuda_graph": True,
        "frozen": "all wiring, axonal delays and dopamine synapses; trained = input projection, weighted readout, tied embedding, superclass time constants, superclass thresholds, conductance gains",
        "injection": args.injection, "injection_neurons": model.n_injection,
        "read_surface": args.read_surface,
        "state_policy": "reservoir state, multi-delay ring buffer and continuous synaptic conductance persist across chunks and updates; never reset",
        "evaluation_protocol": "IB-warm-local-language-v1 reservoir variant: state from zero, 256 warm-in, 128 scored, four sites batched",
        "sites": site_starts,
        "seed": 11,
        "parameter_count_trainable": sum(p.numel() for p in trainable),
    }
    atomic_json(args.output / "config.json", config, required=True)

    step, best = 0, float("inf")

    def save(name: str) -> None:
        temporary = args.output / f"{name}.{os.getpid()}.tmp"
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "state": h.detach().clone(),
                   "spike_ring": [r.detach().clone() for r in ring],
                   "step": step,
                   "best_validation_nll": best, "config": config}
        if is_coba:
            payload["ge"] = ge.detach().clone()
            payload["gi"] = gi.detach().clone()
            if getattr(model, "use_alif", False):
                payload["b"] = b.detach().clone()
        else:
            payload["i_syn"] = i_syn.detach().clone()
            if getattr(model, "use_alif", False):
                payload["b"] = b.detach().clone()
        if getattr(model, "use_stp", False):
            payload["x"] = x.detach().clone()
            payload["u"] = u.detach().clone()
        if getattr(model, "topographic_writer", None) is not None:
            payload["a_adapt"] = model.topographic_writer.a_adapt.detach().clone()
        torch.save(payload, temporary)
        os.replace(temporary, args.output / name)

    def log(row: dict) -> None:
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    @torch.no_grad()
    def validate() -> float:
        """Four sites processed as one batch; state from zero, warm-in 256, score 128."""
        model.eval()
        if getattr(model, "topographic_writer", None) is not None:
            model.topographic_writer.reset_eval()
        starts = torch.tensor(site_starts, device="cuda")
        h_val = torch.zeros(n_sites, model.n_neurons, device="cuda")
        ring_val = tuple(torch.zeros(n_sites, model.n_neurons, device="cuda") for _ in range(4))
        if is_coba:
            ge_val = torch.zeros(n_sites, model.n_neurons, device="cuda")
            gi_val = torch.zeros(n_sites, model.n_neurons, device="cuda")
        else:
            i_syn_val = torch.zeros(n_sites, model.n_neurons, device="cuda")
        if getattr(model, "use_alif", False):
            b_val = torch.zeros(n_sites, model.n_neurons, device="cuda")
        if getattr(model, "use_stp", False):
            x_val = torch.ones(n_sites, model.n_neurons, device="cuda")
            u0_val, _, _, _ = model.get_stp_params()
            u_val = u0_val.detach().expand(n_sites, model.n_neurons).clone()

        per_site = torch.zeros(n_sites, device="cuda")
        total = args.warm_in_tokens + args.score_tokens
        for offset in range(total):
            tokens = torch.tensor(
                [int(validation[int(s) + offset]) for s in starts], device="cuda")
            step_kwargs = {}
            if is_coba:
                step_kwargs["ge"] = ge_val
                step_kwargs["gi"] = gi_val
            else:
                step_kwargs["i_syn"] = i_syn_val
            if getattr(model, "use_alif", False):
                step_kwargs["b"] = b_val
            if getattr(model, "use_stp", False):
                step_kwargs["x"] = x_val
                step_kwargs["u"] = u_val

            res = model.step(h_val, tokens, ring_val, **step_kwargs)
            h_val = res[0]
            ring_val = res[2]
            if is_coba:
                ge_val = res[3]
                gi_val = res[4]
                idx = 5
            else:
                i_syn_val = res[3]
                idx = 4
            if getattr(model, "use_alif", False):
                b_val = res[idx]; idx += 1
            if getattr(model, "use_stp", False):
                x_val = res[idx]; idx += 1
                u_val = res[idx]; idx += 1

            if offset >= args.warm_in_tokens:
                targets = torch.tensor(
                    [int(validation[int(s) + offset + 1]) for s in starts],
                    device="cuda")
                logits = model.read(h_val)
                per_site += torch.nn.functional.cross_entropy(
                    logits, targets, reduction="none")
        model.train()
        return float(per_site.mean() / args.score_tokens)

    runner = GraphChunkRunnerDelayed(model, args.chunk_tokens, optimizer)
    h = torch.zeros(1, model.n_neurons, device="cuda")
    ring = tuple(torch.zeros(1, model.n_neurons, device="cuda") for _ in range(4))
    if is_coba:
        ge = torch.zeros(1, model.n_neurons, device="cuda")
        gi = torch.zeros(1, model.n_neurons, device="cuda")
    else:
        i_syn = torch.zeros(1, model.n_neurons, device="cuda")
    if getattr(model, "use_alif", False):
        b = torch.zeros(1, model.n_neurons, device="cuda")
    if getattr(model, "use_stp", False):
        x = torch.ones(1, model.n_neurons, device="cuda")
        u0_init, _, _, _ = model.get_stp_params()
        u = u0_init.detach().clone().expand(1, model.n_neurons).contiguous()

    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cuda", weights_only=False)
        for key in ("neurons", "d_model", "leak", "threshold", "tokens"):
            if saved["config"].get(key) != config.get(key):
                parser.error(f"Resume mismatch: {key}")
        graph_keys = {"edge_index", "edge_pre", "edge_post", "edge_weight", "delay_splits",
                      "edge_pre_e", "edge_post_e", "edge_weight_e", "delay_splits_e",
                      "edge_pre_i", "edge_post_i", "edge_weight_i", "delay_splits_i",
                      "dan_edge_pre", "dan_edge_post", "dan_edge_weight", "lambda_0"}
        model.load_state_dict({k: v for k, v in saved["model"].items()
                                if k not in graph_keys})
        optimizer.load_state_dict(saved["optimizer"])
        h = saved["state"].cuda().clone()
        if "spike_ring" in saved:
            ring = tuple(r.cuda().clone() for r in saved["spike_ring"])
        if is_coba:
            if "ge" in saved and "gi" in saved:
                ge = saved["ge"].cuda().clone()
                gi = saved["gi"].cuda().clone()
            if getattr(model, "use_alif", False) and "b" in saved:
                b = saved["b"].cuda().clone()
        else:
            if "i_syn" in saved:
                i_syn = saved["i_syn"].cuda().clone()
            if getattr(model, "use_alif", False) and "b" in saved:
                b = saved["b"].cuda().clone()
        if getattr(model, "use_stp", False) and "x" in saved and "u" in saved:
            x = saved["x"].cuda().clone()
            u = saved["u"].cuda().clone()
        if "a_adapt" in saved and getattr(model, "topographic_writer", None) is not None:
            model.topographic_writer.a_adapt.copy_(saved["a_adapt"].cuda())
        step, best = int(saved["step"]), float(saved["best_validation_nll"])
    else:
        atomic_json(args.output / "progress.json",
                    {"status": "initializing", "step": 0, "target_steps": args.steps},
                    required=True)
        initial = validate()
        best = initial
        log({"kind": "validation", "step": 0, "validation_nll": initial})
        save("best.pt")
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
                chunk_ids = ids[:, begin:begin + args.chunk_tokens]
                chunk_targets = targets[:, begin:begin + args.chunk_tokens]
                syn_args = [ge, gi] if is_coba else [i_syn]
                if getattr(model, "use_alif", False):
                    syn_args.append(b)
                if getattr(model, "use_stp", False):
                    syn_args.extend([x, u])

                res = runner.replay(chunk_ids, chunk_targets, h, ring, *syn_args)
                chunk_loss, h, diag, ring = res[0], res[1], res[2], res[3]
                idx = 4
                if is_coba:
                    ge, gi = res[idx], res[idx + 1]; idx += 2
                else:
                    i_syn = res[idx]; idx += 1
                if getattr(model, "use_alif", False):
                    b = res[idx]; idx += 1
                if getattr(model, "use_stp", False):
                    x, u = res[idx], res[idx + 1]; idx += 2
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
                if "tau_m_mean" in diag:
                    row["tau_m_mean"] = float(diag["tau_m_mean"])
                if "threshold_mean" in diag:
                    row["threshold_mean"] = float(diag["threshold_mean"])
                if "g_e" in diag:
                    row["g_e"] = float(diag["g_e"])
                if "g_i" in diag:
                    row["g_i"] = float(diag["g_i"])
                if "b_mean" in diag:
                    row["b_mean"] = float(diag["b_mean"])
                if "b_max" in diag:
                    row["b_max"] = float(diag["b_max"])
                if "beta_mean" in diag:
                    row["beta_mean"] = float(diag["beta_mean"])
                if "tau_a_mean" in diag:
                    row["tau_a_mean"] = float(diag["tau_a_mean"])
                if "x_mean" in diag:
                    row["x_mean"] = float(diag["x_mean"])
                if "u_mean" in diag:
                    row["u_mean"] = float(diag["u_mean"])
                if "u0_mean" in diag:
                    row["u0_mean"] = float(diag["u0_mean"])
                if "tau_fac_mean" in diag:
                    row["tau_fac_mean"] = float(diag["tau_fac_mean"])
                if "tau_rec_mean" in diag:
                    row["tau_rec_mean"] = float(diag["tau_rec_mean"])
                if "pulse_mean" in diag:
                    row["pulse_mean"] = float(diag["pulse_mean"])
                log(row)
                atomic_json(args.output / "progress.json",
                            {"status": "running", "target_steps": args.steps, **row})

            if step % args.validate_every == 0 or step == args.steps:
                val = validate()
                is_best = val < best
                if is_best:
                    best = val
                log({"kind": "validation", "step": step,
                     "validation_nll": val, "best_validation_nll": best})
                save("last.pt")
                if is_best:
                    save("best.pt")
                atomic_json(args.output / "progress.json",
                            {"status": "running", "target_steps": args.steps,
                             "step": step, "validation_nll": val, "best": best})
    finally:
        save("last.pt")
        atomic_json(args.output / "progress.json",
                    {"status": "completed" if step >= args.steps else "interrupted",
                     "step": step, "target_steps": args.steps, "best": best})


if __name__ == "__main__":
    main()
