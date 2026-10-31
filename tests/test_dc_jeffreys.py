import sys, os, math
import torch
import torch.nn.functional as F
import numpy as np

# Test Jeffreys / Laplace Dirichlet continuity vs Exponential Decay
vocab_size = 50257
train_tokens = np.load("data/ib_owt_gpt2/train.npy", mmap_mode="r")

# Method 1: Previous exponential decay
e_old = torch.ones(vocab_size) / vocab_size
nll_old = []

# Method 2: Jeffreys continuous conservation (Count + alpha) / (N + alpha*V)
counts = torch.zeros(vocab_size)
alpha = 0.5 # Jeffreys invariant prior
nll_jeffreys = []

for step in range(20000):
    n_tok = int(train_tokens[step + 1])

    # Old
    phi_old = torch.log(e_old / e_old.sum() + 1e-12)
    nll_old.append(-phi_old[n_tok].item())
    e_old[n_tok] += 0.001
    if step % 200 == 0:
        e_old /= e_old.sum()

    # Jeffreys
    N = step + 1
    p_j = (counts[n_tok] + alpha) / (N + alpha * vocab_size)
    nll_jeffreys.append(-math.log(p_j))
    counts[n_tok] += 1

print(f"Step 0-1000:")
print(f"  Old NLL:      {np.mean(nll_old[:1000]):.3f} nats")
print(f"  Jeffreys NLL: {np.mean(nll_jeffreys[:1000]):.3f} nats")

print(f"Step 8000-9000:")
print(f"  Old NLL:      {np.mean(nll_old[8000:9000]):.3f} nats")
print(f"  Jeffreys NLL: {np.mean(nll_jeffreys[8000:9000]):.3f} nats")

print(f"Step 17000-18000 (where Old spiked to 12.5 nats):")
print(f"  Old NLL:      {np.mean(nll_old[17000:18000]):.3f} nats")
print(f"  Jeffreys NLL: {np.mean(nll_jeffreys[17000:18000]):.3f} nats")
