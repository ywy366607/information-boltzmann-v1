import sys, os, time, math
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing Oja Co-flow CFE on device: {device}")

from fine_grain.information_boltzmann.owt_cfe import OWTConstructalEngine

class OjaConstructalEngine(OWTConstructalEngine):
    def __init__(self, *args, eta_oja: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta_oja = eta_oja
        # 全局连续上下文动量场
        self.register_buffer("h_context", torch.zeros(self.dim, device=self.device))

    @torch.no_grad()
    def step_stream(self, current_token: int, next_token: int):
        # 1. 执行第一性原理前向预测与更新
        metrics = super().step_stream(current_token, next_token)

        # 2. 奥贾超球面切空间局域共流收缩 (Oja's Co-flow Contraction)
        k_in = self.token_sig(current_token) # [dim]

        # 当已有足够上下文动量时，将词元坐标向共流方向投影收缩
        if self.tokens_processed > 1:
            h_norm = F.normalize(self.h_context + 1e-8, p=2, dim=-1)
            # 切空间投影: delta_e = eta * (h - <e, h> e)
            proj = torch.dot(k_in, h_norm)
            delta_e = self.eta_oja * (h_norm - proj * k_in)
            # 原地更新并归一化该词元的嵌入坐标
            new_e = F.normalize(self.embedding.weight[current_token] + delta_e, p=2, dim=-1)
            self.embedding.weight[current_token] = new_e

        # 3. 连续上下文动量更新 (惯性衰减率 alpha = 0.85)
        self.h_context = 0.85 * self.h_context + 0.15 * k_in

        return metrics

if __name__ == "__main__":
    train_tokens = np.load("data/ib_owt_gpt2/train.npy", mmap_mode="r")
    engine = OjaConstructalEngine(
        vocab_size=50257,
        dim=64,
        init_slots=384,
        max_slots=768,
        pipe_lr=4.0,
        J_crit=3.2,
        eta_oja=0.08,
        device=device
    )
    print(f"Initialized Oja CFE on GPU, testing 2000 tokens...")
    t0 = time.time()
    surps = []
    for step in range(2000):
        c_tok = int(train_tokens[step])
        n_tok = int(train_tokens[step + 1])
        res = engine.step_stream(c_tok, n_tok)
        surps.append(res["surprisal_nats"])
        if step % 500 == 0 or step == 1999:
            print(f"Step {step:4d} | Slots: {int(res['num_slots']):3d} | NLL (last 100): {np.mean(surps[-100:]):.3f} n ({np.mean(surps[-100:])/math.log(2):.3f} b) | beta*: {res['beta_star']:.2f}")

    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"Completed 2000 steps in {dt:.2f}s ({dt/2000*1000:.2f} ms/step)")
