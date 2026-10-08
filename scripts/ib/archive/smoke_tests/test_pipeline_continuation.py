"""Quick Test: Continuation Adaptation vs. Scratch Training.

Tests whether the mature 1.5M checkpoint can rapidly adapt to the pipelined interface
in just 10 online BPTT updates (320 tokens), tracking prequential NLL progression.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}...")

    ckpt_path = Path("artifacts/active_training/q8_fly_bptt32_gamma_2333_wide115_100k/last.pt")
    saved, learner_old = load_checkpoint(ckpt_path, device=device)
    model = learner_old.model
    s_init = clone_state(learner_old.state)

    # Disable Gamma for pure motor pipelined test
    model.use_read_gamma_trace = False

    data = np.load(ROOT / saved["config"]["data"] / "train.npy", mmap_mode="r")
    cursor = int(saved["train_cursor"])

    W = 32
    n_updates = 10
    total_tokens = W * n_updates
    tokens = np.array(data[cursor:cursor + total_tokens], dtype=np.int64)

    # Instantiate FlyPipelineLearner continuing with current model & physical state
    learner = FlyPipelineLearner(
        model, s_init,
        adam_names=list(saved["learner"]["adam_names"]),
        lr=2e-4,
        lr_decoder=1e-3, # allow readout/decoder to adapt slightly faster
        lr_synapse=0.0,  # freeze synapses for first 10 steps to isolate readout adaptation
        lr_sensory=0.0,
        settle_ticks=0,
        writer_baseline_clock="input",
    )

    print("\n" + "=" * 70)
    print("CONTINUATION TEST: 10 Online Updates (320 tokens) on Mature Individual")
    print("=" * 70)

    bias_logits = model.decoder.bias.unsqueeze(0).expand(W, -1)

    prequential_scores = []
    unigram_scores = []

    for u in range(n_updates):
        window_tokens = tokens[u * W:(u + 1) * W]
        # Prequential score and update inside learner.observe:
        scores, last_grad = learner.observe(window_tokens)
        mean_score = float(np.mean(scores))
        prequential_scores.append(mean_score)

        # Unigram score on this same window
        u_loss = F.cross_entropy(bias_logits, torch.as_tensor(window_tokens, device=device)).item()
        unigram_scores.append(u_loss)

        diff = mean_score - u_loss
        print(f"  Update {u+1:2d}/10 | Prequential NLL: {mean_score:.4f} | Unigram: {u_loss:.4f} | Delta: {diff:+.4f} nats")

    print("\nSummary of Continuation Adaptation:")
    print(f"  Initial Window 1 NLL: {prequential_scores[0]:.4f}")
    print(f"  Final Window 10 NLL : {prequential_scores[-1]:.4f} (Drop: {prequential_scores[-1] - prequential_scores[0]:+.4f} nats)")
    print(f"  Mean vs Unigram (Last 5 windows): {np.mean(np.array(prequential_scores[5:]) - np.array(unigram_scores[5:])):+.4f} nats")


if __name__ == "__main__":
    main()
