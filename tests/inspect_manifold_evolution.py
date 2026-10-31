import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn.functional as F
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")
from tests.test_oja_cfe import OjaConstructalEngine

device = torch.device("cuda")
train_tokens = np.load("data/ib_owt_gpt2/train.npy", mmap_mode="r")

engine = OjaConstructalEngine(vocab_size=50257, dim=64, init_slots=384, max_slots=768, pipe_lr=4.0, J_crit=3.2, eta_oja=0.10, device=device)

# Pair of words to check
words = ["doctor", "hospital", "medical", "patients", "Clinton", "Sanders", "the", "a"]
token_ids = {w: enc.encode(" " + w)[0] for w in words}

print("Initial Cosine Similarities (Random Gaussian Embeddings):")
for w1 in ["doctor", "Clinton", "the"]:
    id1 = token_ids[w1]
    for w2 in words:
        if w1 != w2:
            id2 = token_ids[w2]
            sim = torch.dot(engine.token_sig(id1), engine.token_sig(id2)).item()
            print(f"  cos('{w1}', '{w2}') = {sim:+.4f}")
    print()

# Run 4000 tokens
print("Running 4000 tokens of continuous stream...")
for step in range(4000):
    engine.step_stream(int(train_tokens[step]), int(train_tokens[step + 1]))

print("\nEvolved Cosine Similarities (After 4000 tokens of Oja Co-flow):")
for w1 in ["doctor", "Clinton", "the"]:
    id1 = token_ids[w1]
    for w2 in words:
        if w1 != w2:
            id2 = token_ids[w2]
            sim = torch.dot(engine.token_sig(id1), engine.token_sig(id2)).item()
            print(f"  cos('{w1}', '{w2}') = {sim:+.4f}")
    print()
