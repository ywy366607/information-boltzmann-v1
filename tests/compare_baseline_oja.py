import sys, os, time, math
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Comparing Baseline CFE vs AC-Oja CFE on device: {device}")

from fine_grain.information_boltzmann.owt_cfe import OWTConstructalEngine
from tests.inspect_ac_oja import ACOjaConstructalEngine

train_tokens = np.load("data/ib_owt_gpt2/train.npy", mmap_mode="r")
num_tokens = 5000

# 1. Run Baseline (Frozen Embeddings)
print("\n--- Running Baseline CFE (Frozen Random Embeddings) ---")
engine_base = OWTConstructalEngine(vocab_size=50257, dim=64, init_slots=384, max_slots=768, pipe_lr=4.0, J_crit=3.2, device=device)
base_surps = []
t0 = time.time()
for step in range(num_tokens):
    c_tok = int(train_tokens[step])
    n_tok = int(train_tokens[step + 1])
    res = engine_base.step_stream(c_tok, n_tok)
    base_surps.append(res["surprisal_nats"])
    if step % 1000 == 0 or step == num_tokens - 1:
        print(f"Base Step {step:4d} | NLL (last 200): {np.mean(base_surps[-200:]):.3f} n | Slots: {int(res['num_slots'])}")
t_base = time.time() - t0

# 2. Run AC-Oja Co-flow (Continuous Manifold Sculpting)
print("\n--- Running AC-Oja CFE (Continuous Manifold Sculpting) ---")
engine_oja = ACOjaConstructalEngine(vocab_size=50257, dim=64, init_slots=384, max_slots=768, pipe_lr=4.0, J_crit=3.2, eta_oja=0.06, device=device)
oja_surps = []
t0 = time.time()
for step in range(num_tokens):
    c_tok = int(train_tokens[step])
    n_tok = int(train_tokens[step + 1])
    res = engine_oja.step_stream(c_tok, n_tok)
    oja_surps.append(res["surprisal_nats"])
    if step % 1000 == 0 or step == num_tokens - 1:
        print(f"Oja  Step {step:4d} | NLL (last 200): {np.mean(oja_surps[-200:]):.3f} n | Slots: {int(res['num_slots'])}")
t_oja = time.time() - t0

print("\n" + "="*70)
print(f"Baseline Final NLL (last 500): {np.mean(base_surps[-500:]):.3f} nats | Time: {t_base:.2f}s ({t_base/num_tokens*1000:.2f} ms/token)")
print(f"AC-Oja   Final NLL (last 500): {np.mean(oja_surps[-500:]):.3f} nats | Time: {t_oja:.2f}s ({t_oja/num_tokens*1000:.2f} ms/token)")
print(f"Net NLL Difference: {np.mean(base_surps[-500:]) - np.mean(oja_surps[-500:]):+.3f} nats")
print("="*70)
