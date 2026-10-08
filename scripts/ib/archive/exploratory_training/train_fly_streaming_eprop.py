"""Continuous Streaming O(1) E-Prop & Three-Factor Plasticity Trainer.

Trains the MaleCNS v1.0 connectome (165,122 neurons, 25.32M synapses, 4-bin delays)
on continuous streaming text (OpenWebText) with sequence-length-independent O(1) memory.

Breaks the 128-token BPTT barrier to stream 1024, 2048, 4096+ tokens per sequence
without historical autograd graph accumulation or VRAM growth.

Biophysical Credit Assignment:
  1. Factor 1 (Presynaptic): Delay-aware filtered STP pulse z_bar_i,d(t).
  2. Factor 2 (Postsynaptic): COBA driving force * ALIF adaptation sensitivity phi_j(t).
  3. Factor 3 (Neuromodulation): Readout error + Direct Feedback Alignment + Biological 241k Dopamine (DAN) broadcast.
  4. Memory Complexity: Strictly O(1), VRAM < 1.0 GB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.eprop_credit_assignment import (
    EPropEligibilityState,
    EPropCreditAssignment,
    StreamingConnectomeTrainer,
)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    parser.add_argument("--graph", type=Path, default=Path("data/malecns_v1/fly_reservoir_coba.npz"))
    parser.add_argument("--resume", type=Path,
                        default=Path("results/q8_fly_reservoir_topographic_coba_alif_stp_3000/best.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/q8_fly_streaming_eprop_owt"))
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of streaming sequence episodes to process")
    parser.add_argument("--sequence-length", type=int, default=1024,
                        help="Streaming tokens per episode (breaks 128 barrier: 512, 1024, 2048, 4096)")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--lr-readout", type=float, default=1e-3)
    parser.add_argument("--lr-synapse", type=float, default=1e-5)
    parser.add_argument("--dopamine-coupling", type=float, default=0.1)
    parser.add_argument("--fa-coupling", type=float, default=0.1)
    parser.add_argument("--update-synapses", action="store_true", default=True,
                        help="Enable full-brain 25.32M synaptic updates")
    parser.add_argument("--no-synapse-updates", dest="update_synapses", action="store_false")
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--site-starts", type=str, default="8192,12288,16384,20480")
    parser.add_argument("--warm-in-tokens", type=int, default=256)
    parser.add_argument("--score-tokens", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required for streaming connectome training")

    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading MaleCNS connectome from {args.graph}...", flush=True)
    model = FlyReservoirLM(
        args.graph,
        vocab_size=50257,
        d_model=args.d_model,
        injection="topographic",
        read_surface="output",
        synapse_model="coba",
        use_alif=True,
        use_stp=True,
    ).cuda()

    # Load pretrained parameters if specified
    if args.resume is not None and args.resume.exists():
        print(f"Resuming weights from checkpoint {args.resume}...", flush=True)
        saved = torch.load(args.resume, map_location="cpu")
        graph_keys = {
            "edge_index", "edge_pre", "edge_post", "edge_weight", "delay_splits",
            "edge_pre_e", "edge_post_e", "edge_weight_e", "delay_splits_e",
            "edge_pre_i", "edge_post_i", "edge_weight_i", "delay_splits_i",
            "dan_edge_pre", "dan_edge_post", "dan_edge_weight", "lambda_0"
        }
        model.load_state_dict({k: v for k, v in saved["model"].items() if k not in graph_keys}, strict=False)
        print(f"Successfully loaded checkpoint step {saved.get('step', 'N/A')}, Val NLL: {saved.get('best_validation_nll', 'N/A'):.4f}", flush=True)

    trainer = StreamingConnectomeTrainer(
        model=model,
        lr_readout=args.lr_readout,
        lr_synapse=args.lr_synapse,
        weight_decay=1e-4,
        dopamine_coupling=args.dopamine_coupling,
        fa_coupling=args.fa_coupling,
    )

    train_data = np.load(args.data / "train.npy", mmap_mode="r")
    val_data = np.load(args.data / "validation.npy", mmap_mode="r")
    site_starts = tuple(int(s) for s in args.site_starts.split(","))
    n_sites = len(site_starts)

    config = {
        "architecture": "FlyReservoir-Streaming-EProp-O1-MaleCNS-Topographic-COBA-ALIF-STP",
        "neurons": model.n_neurons,
        "edges": int(model.edge_weight_e.numel() + model.edge_weight_i.numel()),
        "dan_edges": int(model.dan_edge_weight.numel()) if getattr(model, "has_dopamine", False) else 0,
        "sequence_length": args.sequence_length,
        "memory_complexity": "Strictly O(1) sequence-length invariant (Closed computational graphs)",
        "credit_assignment": "E-Prop Three-Factor Plasticity (Exact COBA Driving Force + ALIF Adaptation + TM STP + Whole-Brain DAN Broadcast + Direct Feedback Alignment)",
        "dopamine_coupling": args.dopamine_coupling,
        "fa_coupling": args.fa_coupling,
        "lr_readout": args.lr_readout,
        "lr_synapse": args.lr_synapse,
        "steps": args.steps,
    }
    atomic_json(args.output / "config.json", config, required=True)

    def log(row: dict) -> None:
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        if getattr(model, "topographic_writer", None) is not None:
            model.topographic_writer.reset_eval()
        starts = torch.tensor(site_starts, device="cuda")
        h_val = torch.zeros(n_sites, model.n_neurons, device="cuda")
        ring_val = tuple(torch.zeros(n_sites, model.n_neurons, device="cuda") for _ in range(4))
        ge_val = torch.zeros(n_sites, model.n_neurons, device="cuda")
        gi_val = torch.zeros(n_sites, model.n_neurons, device="cuda")
        b_val = torch.zeros(n_sites, model.n_neurons, device="cuda")
        x_val = torch.ones(n_sites, model.n_neurons, device="cuda")
        u0_val, _, _, _ = model.get_stp_params()
        u_val = u0_val.detach().expand(n_sites, model.n_neurons).clone()

        per_site = torch.zeros(n_sites, device="cuda")
        total = args.warm_in_tokens + args.score_tokens
        for offset in range(total):
            tokens = torch.tensor([int(val_data[int(s) + offset]) for s in starts], device="cuda")
            res = model.step(h_val, tokens, ring_val, ge=ge_val, gi=gi_val, b=b_val, x=x_val, u=u_val)
            h_val = res[0].detach()
            ring_val = tuple(r.detach() for r in res[2])
            ge_val = res[3].detach()
            gi_val = res[4].detach()
            b_val = res[5].detach()
            x_val = res[6].detach()
            u_val = res[7].detach()

            if offset >= args.warm_in_tokens:
                targets = torch.tensor([int(val_data[int(s) + offset + 1]) for s in starts], device="cuda")
                logits = model.read(h_val)
                per_site += torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        model.train()
        return float(per_site.mean() / args.score_tokens)

    print("Running initial validation...", flush=True)
    initial_val = validate()
    print(f"Initial Validation NLL: {initial_val:.4f}", flush=True)
    log({"kind": "validation", "step": 0, "validation_nll": initial_val})
    best_val = initial_val

    # Initialize continuous physical states (never reset, strictly lifelong streaming)
    h = torch.zeros(1, model.n_neurons, device="cuda")
    ring = tuple(torch.zeros(1, model.n_neurons, device="cuda") for _ in range(4))
    syn_state = {
        "ge": torch.zeros(1, model.n_neurons, device="cuda"),
        "gi": torch.zeros(1, model.n_neurons, device="cuda"),
        "b": torch.zeros(1, model.n_neurons, device="cuda"),
        "x": torch.ones(1, model.n_neurons, device="cuda"),
        "u": model.get_stp_params()[0].detach().clone().expand(1, model.n_neurons).contiguous(),
    }

    print(f"\n================================================================================")
    print(f"STARTING LIFELONG O(1) STREAMING E-PROP: {args.sequence_length} TOKENS/EPISODE")
    print(f"================================================================================", flush=True)

    step = 0
    token_offset = 0
    while step < args.steps:
        step += 1
        t_start = time.perf_counter()
        episode_losses = []

        # Stream sequence of arbitrary length (e.g. 1024 tokens) strictly forward in time
        for t in range(args.sequence_length):
            tok_in = torch.tensor([int(train_data[token_offset + t])], device="cuda")
            tok_tgt = torch.tensor([int(train_data[token_offset + t + 1])], device="cuda")

            loss_val, h, ring, syn_state = trainer.train_streaming_step(
                input_token=tok_in,
                target_token=tok_tgt,
                h=h,
                spike_ring=ring,
                syn_state=syn_state,
                update_synapses=args.update_synapses,
            )
            episode_losses.append(loss_val)

        token_offset += args.sequence_length
        elapsed = time.perf_counter() - t_start
        tokens_per_sec = args.sequence_length / elapsed
        mean_loss = sum(episode_losses) / len(episode_losses)
        vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)

        # Output progress every step
        print(f"Step {step:4d}/{args.steps} | Tokens: {token_offset:7d} | Stream Loss: {mean_loss:.4f} | "
              f"Speed: {tokens_per_sec:.1f} tok/s | VRAM: {vram_mb:.1f} MB", flush=True)

        atomic_json(args.output / "progress.json", {
            "status": "running",
            "step": step,
            "target_steps": args.steps,
            "stream_tokens_processed": token_offset,
            "mean_stream_loss": mean_loss,
            "vram_mb": vram_mb,
            "tokens_per_sec": tokens_per_sec,
            "best_validation_nll": best_val,
        })

        if step % args.validate_every == 0 or step == args.steps:
            val_nll = validate()
            is_best = val_nll < best_val
            if is_best:
                best_val = val_nll
                torch.save({
                    "model": model.state_dict(),
                    "step": step,
                    "stream_tokens": token_offset,
                    "best_validation_nll": best_val,
                    "config": config,
                }, args.output / "best.pt")
                print(f"  >>> New Best Validation NLL: {best_val:.4f} (Saved to best.pt)", flush=True)

            log({"kind": "validation", "step": step, "stream_tokens": token_offset,
                 "validation_nll": val_nll, "best_validation_nll": best_val})

    # Save final checkpoint
    torch.save({
        "model": model.state_dict(),
        "step": step,
        "stream_tokens": token_offset,
        "final_validation_nll": best_val,
        "config": config,
    }, args.output / "last.pt")

    atomic_json(args.output / "progress.json", {
        "status": "completed",
        "step": step,
        "target_steps": args.steps,
        "stream_tokens_processed": token_offset,
        "best_validation_nll": best_val,
    })
    print("\nTraining successfully completed!", flush=True)


if __name__ == "__main__":
    main()
