import sys, os, time, math
import torch
import torch.nn.functional as F
import numpy as np

# Empirical Information Theory Staircase on OpenWebText
train_tokens = np.load("data/ib_owt_gpt2/train.npy", mmap_mode="r")
vocab_size = 50257
num_tokens = 5000

print("="*80)
print("自然语言信息论真实阶梯实测 (Empirical Entropy Staircase on OpenWebText)")
print("="*80)

# Level 0: Uniform baseline
h0 = math.log(vocab_size)
print(f"阶梯 0 (完全均匀盲猜): {h0:.3f} nats ({h0/math.log(2):.3f} bits)")

# Level 1: Online Unigram (Jeffreys Prior)
unigram_counts = torch.zeros(vocab_size)
alpha_u = 0.5
nll_l1 = []

# Level 2: Online Bigram (Direct Transition Conductance)
# Using sparse dictionary for (w_t, w_{t+1}) counts
bigram_counts = {}
context_counts = {}
alpha_b = 0.1
nll_l2 = []

# Level 3: Online Trigram ((w_{t-1}, w_t) -> w_{t+1})
trigram_counts = {}
trigram_ctx_counts = {}
alpha_t = 0.05
nll_l3 = []

prev2 = -1
prev1 = -1

for step in range(num_tokens):
    c_tok = int(train_tokens[step])
    n_tok = int(train_tokens[step + 1])
    N = step + 1

    # --- Level 1: Unigram ---
    p1 = (unigram_counts[n_tok].item() + alpha_u) / (N + alpha_u * vocab_size)
    nll_l1.append(-math.log(p1))
    unigram_counts[n_tok] += 1.0

    # --- Level 2: Direct Bigram ---
    # P(w_{t+1} | w_t) = (C(w_t, w_{t+1}) + alpha * P(w_{t+1})) / (C(w_t) + alpha)
    c_u = context_counts.get(c_tok, 0)
    c_uv = bigram_counts.get((c_tok, n_tok), 0)
    p2 = (c_uv + alpha_b * p1) / (c_u + alpha_b)
    nll_l2.append(-math.log(p2))
    context_counts[c_tok] = c_u + 1
    bigram_counts[(c_tok, n_tok)] = c_uv + 1

    # --- Level 3: Trigram ---
    if prev1 >= 0:
        ctx2 = (prev1, c_tok)
        c_ctx2 = trigram_ctx_counts.get(ctx2, 0)
        c_ctx2_next = trigram_counts.get((prev1, c_tok, n_tok), 0)
        p3 = (c_ctx2_next + alpha_t * p2) / (c_ctx2 + alpha_t)
        nll_l3.append(-math.log(p3))
        trigram_ctx_counts[ctx2] = c_ctx2 + 1
        trigram_counts[(prev1, c_tok, n_tok)] = c_ctx2_next + 1
    else:
        nll_l3.append(nll_l2[-1])

    prev2 = prev1
    prev1 = c_tok

print(f"\n连续冲刷 {num_tokens} 词元后的实测收敛水平 (最后 500 词元均值):")
print(f"  • 阶梯 1 (纯单字词频 / Unigram):     {np.mean(nll_l1[-500:]):.3f} nats ({np.mean(nll_l1[-500:])/math.log(2):.3f} bits)")
print(f"  • 阶梯 2 (直接因果转移 / Bigram):    {np.mean(nll_l2[-500:]):.3f} nats ({np.mean(nll_l2[-500:])/math.log(2):.3f} bits)")
print(f"  • 阶梯 3 (双词上下文短语 / Trigram):  {np.mean(nll_l3[-500:]):.3f} nats ({np.mean(nll_l3[-500:])/math.log(2):.3f} bits)")
print("="*80)
