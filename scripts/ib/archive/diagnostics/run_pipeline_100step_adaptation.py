"""100-Step Continuous Adaptation of Pipelined Fly Brain on Real OWT Stream.

Protocol:
- Mature individual from last.pt (cursor 1,600,000).
- Pure native motor read: No Gamma (use_read_gamma_trace=False), No timescale fusion.
- Full learner: FlyPipelineLearner with all 25.3M synapses, ALIF, STP, writer, decoder.
- Standard continuous-individual prequential protocol:
  Every window (32 tokens) is scored strictly BEFORE parameter updates on that window.
- Compared against static Unigram (Decoder Bias) baseline on identical targets.
- 100 consecutive updates (3,200 tokens total).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:64"

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core import fly_reservoir as reservoir
from information_boltzmann.core.fly_bptt_learning import (
    FlyPhysicalState,
    FlyBPTTLearner,
)
from information_boltzmann.core.fly_pipeline import (
    FlyPipelineLearner,
)
from scripts.ib.run_full_diagnosis import load_checkpoint, clone_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/active_training/q8_fly_bptt32_gamma_2333_wide115_100k/last.pt"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path,
                        default=Path("results/published/fly_pipeline_100step_adaptation.json"))
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint} on {args.device}...", flush=True)
    saved, learner_old = load_checkpoint(args.checkpoint, device=args.device)
    model = learner_old.model
    s_init = clone_state(learner_old.state)

    # Disable Gamma for pure native motor pipelined execution
    model.use_read_gamma_trace = False

    data = np.load(ROOT / saved["config"]["data"] / "train.npy", mmap_mode="r")
    cursor = int(saved["train_cursor"])

    W = 32
    n_updates = args.steps
    total_tokens = W * n_updates

    # Instantiate FlyPipelineLearner
    adam_names = [n for n in saved["learner"]["adam_names"] if n != "logit_read_gamma"]
    learner = FlyPipelineLearner(
        model, s_init,
        adam_names=adam_names,
        lr=2e-4,
        lr_decoder=2e-4,
        lr_synapse=2e-4,
        lr_sensory=2e-4,
        plasticity_optimizer="adamw",
        settle_ticks=0,
        writer_baseline_clock="input",
    )

    print(f"Starting 100-step continuous adaptation at cursor {cursor}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
    print("=" * 85)
    print(f"{'Update':<8} | {'Tokens':<12} | {'Prequential NLL':<16} | {'Unigram NLL':<12} | {'Delta vs Uni':<13} | {'Beat Uni?':<9} | {'Speed':<10}")
    print("-" * 85)

    bias_logits = model.decoder.bias.unsqueeze(0).expand(W, -1)

    window_history = []
    wins_against_unigram = 0
    t_start = time.perf_counter()

    for u in range(n_updates):
        cur_pos = cursor + u * W
        window_tokens = np.array(data[cur_pos:cur_pos + W], dtype=np.int64)

        t0_win = time.perf_counter()
        # Observe: prequential scoring happens inside before backward & step
        scores, last_grad = learner.observe(window_tokens)
        t1_win = time.perf_counter()

        mean_nll = float(np.mean(scores))
        u_loss = F.cross_entropy(bias_logits, torch.as_tensor(window_tokens, device=args.device)).item()
        delta_uni = mean_nll - u_loss

        is_win = delta_uni < 0
        if is_win:
            wins_against_unigram += 1

        win_sec = t1_win - t0_win
        speed_tok_s = W / max(win_sec, 1e-5)

        entry = {
            "update": u + 1,
            "cursor": cur_pos + W,
            "prequential_nll": mean_nll,
            "unigram_nll": u_loss,
            "delta_vs_unigram": delta_uni,
            "beat_unigram": is_win,
            "seconds": win_sec,
            "field_energy": float(learner.state.h.square().mean().item()),
            "vram_reserved_mib": torch.cuda.memory_reserved() / 2**20 if args.device == "cuda" else 0.0,
        }
        window_history.append(entry)

        # Print progress every 10 updates or for the first 3
        if u < 5 or (u + 1) % 10 == 0 or (u + 1) == n_updates:
            beat_str = "YES (-)" if is_win else "no (+)"
            print(f"#{u+1:<7d} | {cur_pos+W:<12d} | {mean_nll:<16.4f} | {u_loss:<12.4f} | {delta_uni:<+13.4f} | {beat_str:<9} | {speed_tok_s:<6.1f} tok/s", flush=True)

    t_total = time.perf_counter() - t_start
    print("-" * 85)

    all_prequential = [w["prequential_nll"] for w in window_history]
    all_unigram = [w["unigram_nll"] for w in window_history]
    all_deltas = [w["delta_vs_unigram"] for w in window_history]

    mean_first10 = np.mean(all_prequential[:10])
    mean_last20 = np.mean(all_prequential[-20:])
    mean_last20_uni = np.mean(all_unigram[-20:])
    mean_last20_delta = np.mean(all_deltas[-20:])

    print(f"\nFinal Adaptation Summary (100 Updates = 3,200 tokens):")
    print(f"  Total Elapsed Time           : {t_total:.1f}s ({total_tokens / t_total:.1f} tokens/s)")
    print(f"  Peak VRAM Reserved           : {torch.cuda.max_memory_reserved() / 2**20:.1f} MiB")
    print(f"  Initial 10-Window Mean NLL   : {mean_first10:.4f} nats")
    print(f"  Final 20-Window Mean NLL     : {mean_last20:.4f} nats (Drop: {mean_last20 - mean_first10:+.4f})")
    print(f"  Final 20-Window Unigram Mean : {mean_last20_uni:.4f} nats")
    print(f"  Final 20-Window vs Unigram   : {mean_last20_delta:+.4f} nats")
    print(f"  Total Windows Beating Unigram: {wins_against_unigram}/{n_updates} ({wins_against_unigram / n_updates * 100:.1f}%)")

    results = {
        "checkpoint": str(args.checkpoint),
        "cursor_start": cursor,
        "cursor_end": cursor + total_tokens,
        "n_updates": n_updates,
        "tokens_processed": total_tokens,
        "elapsed_seconds": t_total,
        "initial_10_mean_nll": float(mean_first10),
        "final_20_mean_nll": float(mean_last20),
        "final_20_mean_unigram": float(mean_last20_uni),
        "final_20_mean_delta": float(mean_last20_delta),
        "windows_beating_unigram": wins_against_unigram,
        "fraction_beating_unigram": float(wins_against_unigram / n_updates),
        "history": window_history,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed metrics saved to {args.output} successfully!", flush=True)


if __name__ == "__main__":
    main()
