import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn.functional as F
import numpy as np

# Test different tubule plasticity rates and gains
from tests.test_hierarchical_cfe import HierarchicalConstructalEngine

device = torch.device("cuda")
train_tokens = np.load("data/ib_owt_gpt2/train.npy", mmap_mode="r")

for lr_pipe in [1.0, 3.0, 5.0, 8.0]:
    engine = HierarchicalConstructalEngine(vocab_size=50257, dim=64, init_slots=384, max_slots=768, device=device)
    dc_nlls = []
    tot_nlls = []

    for step in range(1200):
        c_tok = int(train_tokens[step])
        n_tok = int(train_tokens[step + 1])

        logits, act, beta_star, k_in, full_act = engine.prequential_predict(c_tok)
        target = torch.tensor([n_tok], device=device)

        phi_0 = torch.log(engine.e_ground / engine.e_ground.sum() + 1e-12)
        dc_nll = F.cross_entropy(phi_0.unsqueeze(0), target).item()
        tot_nll = F.cross_entropy(logits.unsqueeze(0), target).item()

        dc_nlls.append(dc_nll)
        tot_nlls.append(tot_nll)

        # Step with customized lr_pipe
        engine.tokens_processed += 1
        act_idx = torch.where(engine.active_mask)[0]

        # DC update
        engine.e_ground[n_tok] += 0.001
        if engine.tokens_processed % 200 == 0:
            engine.e_ground /= engine.e_ground.sum()

        # Variational impedance
        normalized_res = (tot_nll / 10.825) ** 2
        engine.residual_var = 0.98 * engine.residual_var + 0.02 * normalized_res

        # Tubule update with lr_pipe
        engine.W_out[:, n_tok] += lr_pipe * full_act
        if engine.tokens_processed % 5 == 0:
            engine.W_out.mul_(1.0 - 0.001)

        # Update traces
        g_act = engine.gamma[act_idx].unsqueeze(1)
        engine.trace[act_idx] = g_act * engine.trace[act_idx] + (1.0 - g_act) * k_in

    print(f"lr_pipe = {lr_pipe:4.1f} | Final 200 tokens: DC NLL = {np.mean(dc_nlls[-200:]):.3f} | Total NLL = {np.mean(tot_nlls[-200:]):.3f} | AC Gain = {np.mean(dc_nlls[-200:]) - np.mean(tot_nlls[-200:]):.3f} nats")
