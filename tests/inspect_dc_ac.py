import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn.functional as F
import numpy as np

# Inspect the DC vs AC contribution
from tests.test_hierarchical_cfe import HierarchicalConstructalEngine

device = torch.device("cuda")
train_tokens = np.load("data/ib_owt_gpt2/train.npy", mmap_mode="r")
engine = HierarchicalConstructalEngine(vocab_size=50257, dim=64, init_slots=384, max_slots=768, device=device)

# Step 1000 tokens
dc_nlls = []
total_nlls = []
betas = []
z_ac_norms = []

for step in range(1500):
    c_tok = int(train_tokens[step])
    n_tok = int(train_tokens[step + 1])

    # Pre-predict
    logits, act, beta_star, k_in, full_act = engine.prequential_predict(c_tok)
    target = torch.tensor([n_tok], device=device)

    phi_0 = torch.log(engine.e_ground / engine.e_ground.sum() + 1e-12)
    dc_nll = F.cross_entropy(phi_0.unsqueeze(0), target).item()
    tot_nll = F.cross_entropy(logits.unsqueeze(0), target).item()

    dc_nlls.append(dc_nll)
    total_nlls.append(tot_nll)
    betas.append(beta_star)

    z_ac = torch.matmul(full_act, engine.W_out)
    z_ac_norms.append(z_ac.norm().item())

    engine.step_stream(c_tok, n_tok)

print("Step 500-1000:")
print(f"DC-only NLL: {np.mean(dc_nlls[-500:]):.3f} nats")
print(f"Total NLL:   {np.mean(total_nlls[-500:]):.3f} nats")
print(f"Avg beta*:   {np.mean(betas[-500:]):.3f}")
print(f"Avg ||z_ac||: {np.mean(z_ac_norms[-500:]):.3f}")
