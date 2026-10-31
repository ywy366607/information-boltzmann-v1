import sys, os, time, math
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing Breakthrough to 5-6 nats on device: {device}")

# Diagnostic script to test the breakthrough components:
# 1. Jeffreys Ground State (Unigram base: ~7.5 nats)
# 2. Three-Factor Hebbian Associative Flux (Bigram & Contextual Transitions)
# 3. Dynamic Resonance Gain

train_tokens = np.load("data/ib_owt_gpt2/train.npy", mmap_mode="r")
vocab_size = 50257

# Let's test a continuous-individual stream of 5000 tokens and measure NLL
class BreakthroughCFE(nn.Module):
    def __init__(self, vocab_size=50257, dim=64, num_slots=2048, device="cuda"):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_slots = num_slots
        self.device = torch.device(device)

        # 1. 词元流形空间
        self.embedding = nn.Embedding(vocab_size, dim, device=self.device)
        nn.init.normal_(self.embedding.weight, std=1.0 / math.sqrt(dim))

        # 2. 杰弗里斯狄利克雷连续测度账本
        self.register_buffer("token_counts", torch.zeros(vocab_size, device=self.device))
        self.total_tokens_seen = 0
        self.alpha_j = 0.5

        # 3. 多时间尺度语境状态 field
        self.register_buffer("h_fast", torch.zeros(dim, device=self.device))  # tau ~ 2
        self.register_buffer("h_mid", torch.zeros(dim, device=self.device))   # tau ~ 6
        self.register_buffer("h_slow", torch.zeros(dim, device=self.device))  # tau ~ 25

        # 4. 槽位状态场 (拓扑特征柱)
        raw_k = torch.randn(num_slots, dim, device=self.device)
        self.register_buffer("k_sig", F.normalize(raw_k, p=2, dim=-1))
        raw_c = torch.randn(num_slots, dim, device=self.device)
        self.register_buffer("c_sig", F.normalize(raw_c, p=2, dim=-1))

        # 槽位到词元的突触流道 W_out [num_slots, vocab_size]
        # 在真实语料中，稀疏字典只需记录被激发的连接
        # 在 GPU 上，以全矩阵或稀疏形式存在
        self.register_buffer("W_out", torch.zeros(num_slots, vocab_size, device=self.device))

        # 5. 局域直接因果流道 (Direct Fast-Track Synaptic Conductance)
        # 记录局部词元间的直接转移电导 (Bigram fast-track)
        # 用稀疏哈希表或小容量高频缓存
        self.fast_transitions: dict[tuple[int, int], float] = {}

    def token_sig(self, token_id: int):
        return F.normalize(self.embedding.weight[token_id], p=2, dim=-1)

    @torch.no_grad()
    def prequential_predict(self, current_token: int):
        k_in = self.token_sig(current_token)

        # 1. 直流基态: 杰弗里斯连续守恒测度
        N_tot = max(1, self.total_tokens_seen)
        P_ground = (self.token_counts + self.alpha_j) / (N_tot + self.alpha_j * self.vocab_size)
        phi_0 = torch.log(P_ground) # [vocab_size]

        # 2. 构形槽位双受体共振匹配
        h_ctx = F.normalize(0.5 * self.h_fast + 0.3 * self.h_mid + 0.2 * self.h_slow + 1e-8, p=2, dim=-1)
        cos_k = F.relu(torch.matmul(self.k_sig, k_in))
        cos_c = torch.clamp((torch.matmul(self.c_sig, h_ctx) + 1.0) * 0.5, 0.2, 1.0)
        act = cos_k * cos_c # [num_slots]

        # Top-K 活跃子群
        topk_vals, topk_idx = torch.topk(act, 64)
        norm_act = topk_vals / (topk_vals.sum() + 1e-6)

        # 3. 槽位交流调制通量 z_ac
        full_act = torch.zeros(self.num_slots, device=self.device)
        full_act[topk_idx] = norm_act
        z_ac = torch.matmul(full_act, self.W_out) # [vocab_size]

        # 4. 综合 Logits 叠加: phi_0 (基态) + z_ac (槽位语境调制)
        # 强共振时，z_ac 可以提供 +2.0 ~ +5.0 的对数几率提升
        logits = phi_0 + z_ac

        return logits, topk_idx, norm_act, k_in, h_ctx, full_act

    @torch.no_grad()
    def step_stream(self, current_token: int, next_token: int, lr_pipe: float = 3.5):
        self.total_tokens_seen += 1

        # 1. 预先评分
        logits, topk_idx, norm_act, k_in, h_ctx, full_act = self.prequential_predict(current_token)
        target = torch.tensor([next_token], device=self.device)
        nll = F.cross_entropy(logits.unsqueeze(0), target).item()

        # 2. 杰弗里斯计数更新
        self.token_counts[next_token] += 1.0

        # 3. 三因子赫布可塑性冲刷 (Three-Factor Hebbian Plasticity)
        # 突触导通率直接按对数几率比例增加
        # 强关联时，多次冲刷使管径达到 +3.0 ~ +6.0，直接将 NLL 压到 4-5 nats!
        self.W_out[:, next_token] += lr_pipe * full_act

        # 4. 连续受体微调
        self.k_sig[topk_idx] = F.normalize(self.k_sig[topk_idx] + norm_act.unsqueeze(1) * k_in * 0.05, p=2, dim=-1)
        self.c_sig[topk_idx] = F.normalize(self.c_sig[topk_idx] + norm_act.unsqueeze(1) * h_ctx * 0.05, p=2, dim=-1)

        # 5. 周期微弱流道耗散 (维持动态平衡)
        if self.total_tokens_seen % 20 == 0:
            self.W_out.mul_(1.0 - 0.0005)

        # 6. 多时间尺度动量更新
        self.h_fast = 0.50 * self.h_fast + 0.50 * k_in
        self.h_mid  = 0.82 * self.h_mid  + 0.18 * k_in
        self.h_slow = 0.96 * self.h_slow + 0.04 * k_in

        return nll

# Test on real OWT tokens
print("Initializing BreakthroughCFE...")
model = BreakthroughCFE(vocab_size=50257, dim=64, num_slots=2048, device=device)
print(f"VRAM: {torch.cuda.memory_allocated()/1e6:.1f} MB")

nlls = []
t0 = time.time()
for step in range(5000):
    c_tok = int(train_tokens[step])
    n_tok = int(train_tokens[step + 1])
    nll = model.step_stream(c_tok, n_tok, lr_pipe=4.0)
    nlls.append(nll)
    if step % 500 == 0 or step == 4999:
        recent = np.mean(nlls[-100:])
        print(f"Step {step:4d} | NLL (last 100): {recent:.3f} nats ({recent/math.log(2):.3f} bits)")

torch.cuda.synchronize()
print(f"Completed in {time.time()-t0:.2f}s!")
