import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn.functional as F
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")
from fine_grain.information_boltzmann.owt_cfe import OWTConstructalEngine

device = torch.device("cuda")
train_tokens = np.load("data/ib_owt_gpt2/train.npy", mmap_mode="r")

class ACOjaConstructalEngine(OWTConstructalEngine):
    def __init__(self, *args, eta_oja: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta_oja = eta_oja
        # 慢衰减 DC 全局共模场 (tau ~ 200) 与 快衰减局域动量场 (tau ~ 10)
        self.register_buffer("h_dc", torch.zeros(self.dim, device=self.device))
        self.register_buffer("h_ac", torch.zeros(self.dim, device=self.device))

    @torch.no_grad()
    def step_stream(self, current_token: int, next_token: int):
        metrics = super().step_stream(current_token, next_token)

        k_in = self.token_sig(current_token)

        # 1. 动量连续积累: DC 追踪全局均值 (alpha=0.995), AC 追踪局部语境 (alpha=0.85)
        self.h_dc = 0.995 * self.h_dc + 0.005 * k_in
        self.h_ac = 0.85 * self.h_ac + 0.15 * k_in

        # 2. 纯 AC 波动量 delta_h = h_ac - h_dc (剔除共模漂移)
        if self.tokens_processed > 20:
            delta_h = self.h_ac - self.h_dc
            delta_norm = F.normalize(delta_h + 1e-8, p=2, dim=-1)

            # Oja 切空间投影
            proj = torch.dot(k_in, delta_norm)
            delta_e = self.eta_oja * (delta_norm - proj * k_in)

            # 更新该词元嵌入
            self.embedding.weight[current_token] = F.normalize(
                self.embedding.weight[current_token] + delta_e, p=2, dim=-1
            )

        return metrics

engine = ACOjaConstructalEngine(vocab_size=50257, dim=64, init_slots=384, max_slots=768, pipe_lr=4.0, J_crit=3.2, eta_oja=0.08, device=device)

words = ["doctor", "hospital", "medical", "patients", "Clinton", "Sanders", "the", "a"]
token_ids = {w: enc.encode(" " + w)[0] for w in words}

print("Running 4000 tokens of AC Fluctuation Oja Co-flow...")
for step in range(4000):
    engine.step_stream(int(train_tokens[step]), int(train_tokens[step + 1]))

print("\nEvolved AC Cosine Similarities (Zero DC Drift):")
for w1 in ["doctor", "Clinton", "the"]:
    id1 = token_ids[w1]
    for w2 in words:
        if w1 != w2:
            id2 = token_ids[w2]
            sim = torch.dot(engine.token_sig(id1), engine.token_sig(id2)).item()
            print(f"  cos('{w1}', '{w2}') = {sim:+.4f}")
    print()
