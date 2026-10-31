import sys, os, time, math
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing Log-Odds PMI Relaxation CFE on device: {device}")

class LogOddsConstructalEngine(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        dim: int = 64,
        init_slots: int = 2048,
        max_slots: int = 4096,
        k_sparse: int = 128,
        alpha_j: float = 0.5,
        eta_pmi: float = 0.6,
        eta_geo: float = 0.05,
        device: str = "cuda"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.init_slots = init_slots
        self.max_slots = max_slots
        self.k_sparse = k_sparse
        self.alpha_j = alpha_j
        self.eta_pmi = eta_pmi
        self.eta_geo = eta_geo
        self.device = torch.device(device)

        # 1. 词元流形空间 E in R^{V x D}
        self.embedding = nn.Embedding(vocab_size, dim, device=self.device)
        nn.init.normal_(self.embedding.weight, std=1.0 / math.sqrt(dim))

        # 2. 杰弗里斯狄利克雷连续测度账本
        self.register_buffer("token_counts", torch.zeros(vocab_size, device=self.device))
        self.total_tokens_seen = 0

        # 3. 槽位状态场
        self.register_buffer("active_mask", torch.zeros(max_slots, dtype=torch.bool, device=self.device))
        self.active_mask[:init_slots] = True

        # 多时间尺度连续记忆衰减率 gamma: 快 (0.50) 与 慢 (0.95)
        self.register_buffer("tau_fast", torch.zeros(max_slots, dim, device=self.device))
        self.register_buffer("tau_slow", torch.zeros(max_slots, dim, device=self.device))

        # 槽位受体签名:
        # k_sig: 瞬时词元受体
        # c_fast: 局部短语受体
        # c_slow: 宏观主题门控受体
        raw_k = torch.randn(max_slots, dim, device=self.device)
        self.register_buffer("k_sig", F.normalize(raw_k, p=2, dim=-1))
        raw_cf = torch.randn(max_slots, dim, device=self.device)
        self.register_buffer("c_fast", F.normalize(raw_cf, p=2, dim=-1))
        raw_cs = torch.randn(max_slots, dim, device=self.device)
        self.register_buffer("c_slow", F.normalize(raw_cs, p=2, dim=-1))

        self.register_buffer("age", torch.zeros(max_slots, dtype=torch.long, device=self.device))
        self.register_buffer("theta_M", torch.ones(max_slots, device=self.device) * 0.01)

        # 4. 对数几率互信息管网 W_out: [max_slots, vocab_size]
        self.register_buffer("W_out", torch.zeros(max_slots, vocab_size, device=self.device))

        # 5. 全局时空动量场 (用于慢速语境更新)
        self.register_buffer("g_fast", torch.zeros(dim, device=self.device))
        self.register_buffer("g_slow", torch.zeros(dim, device=self.device))

        self.tokens_processed = 0

    @property
    def num_active(self) -> int:
        return int(self.active_mask.sum().item())

    def token_sig(self, token_id: int) -> torch.Tensor:
        return F.normalize(self.embedding.weight[token_id], p=2, dim=-1)

    @torch.no_grad()
    def prequential_predict(self, current_token: int):
        k_in = self.token_sig(current_token)

        # 1. 直流基态: 杰弗里斯连续守恒测度
        N_total = max(1, self.total_tokens_seen)
        P_ground = (self.token_counts + self.alpha_j) / (N_total + self.alpha_j * self.vocab_size)
        phi_0 = torch.log(P_ground) # [vocab_size]

        # 2. 多尺度乘性相干级联共振匹配
        act_idx = torch.where(self.active_mask)[0]
        k_act = self.k_sig[act_idx]
        cf_act = self.c_fast[act_idx]
        cs_act = self.c_slow[act_idx]

        tf_norm = F.normalize(self.tau_fast[act_idx] + 1e-8, p=2, dim=-1)
        ts_norm = F.normalize(self.tau_slow[act_idx] + 1e-8, p=2, dim=-1)

        # 瞬时匹配 * 短语级相干 * 主题级门控
        cos_k = F.relu(torch.matmul(k_act, k_in))
        cos_cf = torch.clamp((torch.sum(cf_act * tf_norm, dim=-1) + 1.0) * 0.5, 0.1, 1.0)
        cos_cs = torch.clamp((torch.sum(cs_act * ts_norm, dim=-1) + 1.0) * 0.5, 0.1, 1.0)

        # 乘性级联通量
        raw_act = cos_k * cos_cf * cos_cs

        # 稀疏共振子群 (Top-K)
        K_sparse = min(self.k_sparse, len(act_idx))
        topk_vals, topk_pos = torch.topk(raw_act, K_sparse)
        sparse_idx = act_idx[topk_pos]
        sparse_act = topk_vals
        tf_sparse = tf_norm[topk_pos]
        ts_sparse = ts_norm[topk_pos]

        # 3. 交流对数几率调制场: 稀疏矢量 GEMV
        full_act = torch.zeros(self.max_slots, device=self.device)
        if sparse_act.sum() > 1e-6:
            # 强化主导槽位
            sparse_norm = sparse_act / sparse_act.sum()
            full_act[sparse_idx] = sparse_norm
            z_ac = torch.matmul(full_act, self.W_out)
        else:
            z_ac = torch.zeros(self.vocab_size, device=self.device)

        # 4. 全势场叠加: phi_0 (基态) + z_ac (对数几率物理增益)
        logits = phi_0 + z_ac

        return logits, sparse_idx, sparse_act, k_in, tf_sparse, ts_sparse, full_act, phi_0

    @torch.no_grad()
    def step_stream(self, current_token: int, next_token: int):
        self.tokens_processed += 1
        self.total_tokens_seen += 1

        # 1. 预先评分
        logits, sparse_idx, sparse_act, k_in, tf_norm, ts_norm, full_act, phi_0 = self.prequential_predict(current_token)
        target_tensor = torch.tensor([next_token], device=self.device)
        surprisal_nats = F.cross_entropy(logits.unsqueeze(0), target_tensor).item()
        surprisal_bits = surprisal_nats / math.log(2)

        # 2. 杰弗里斯狄利克雷计数更新
        self.token_counts[next_token] += 1.0

        # 3. 第一性原理对数几率 PMI 松弛冲刷 (Log-Odds PMI Relaxation)
        # 该事件的理论本征信息量 (Self-Information)
        I_target = -phi_0[next_token].item() # 约 +6.0 ~ +10.0

        # 对激活槽位驱动其电导向 I_target 连续松弛:
        # Delta W = eta * act_i * (I_target - W_current)
        for rank in range(min(16, len(sparse_idx))):
            s_i = sparse_idx[rank]
            a_i = sparse_act[rank].item()
            if a_i > 0.05:
                w_curr = self.W_out[s_i, next_token].item()
                # 弛豫推进
                self.W_out[s_i, next_token] += self.eta_pmi * a_i * (I_target - w_curr)

        # 4. 槽位受体切空间微调
        self.k_sig[sparse_idx] = F.normalize(self.k_sig[sparse_idx] + sparse_act.unsqueeze(1) * k_in * 0.08, p=2, dim=-1)
        self.c_fast[sparse_idx] = F.normalize(self.c_fast[sparse_idx] + sparse_act.unsqueeze(1) * tf_norm * 0.08, p=2, dim=-1)
        self.c_slow[sparse_idx] = F.normalize(self.c_slow[sparse_idx] + sparse_act.unsqueeze(1) * ts_norm * 0.04, p=2, dim=-1)

        # 5. 测地流流形切空间收缩 (Geodesic Manifold Contraction)
        # 顺带把流床 (Embedding) 冲刷成阻力最小的顺滑形状
        v_next = self.token_sig(next_token)
        proj = torch.dot(k_in, v_next)
        delta_e = self.eta_geo * (v_next - proj * k_in)
        self.embedding.weight[current_token] = F.normalize(self.embedding.weight[current_token] + delta_e, p=2, dim=-1)

        # 6. 多时间尺度时空资格迹演化
        self.tau_fast[sparse_idx] = 0.50 * self.tau_fast[sparse_idx] + 0.50 * k_in
        self.tau_slow[sparse_idx] = 0.95 * self.tau_slow[sparse_idx] + 0.05 * k_in

        # 7. 周期微弱热力学扩散耗散
        if self.tokens_processed % 20 == 0:
            self.W_out.mul_(1.0 - 0.0002)

        return surprisal_nats, surprisal_bits

if __name__ == "__main__":
    train_tokens = np.load("data/ib_owt_gpt2/train.npy", mmap_mode="r")
    engine = LogOddsConstructalEngine(vocab_size=50257, dim=64, init_slots=2048, max_slots=4096, device=device)
    print(f"Allocated VRAM: {torch.cuda.memory_allocated()/1e6:.1f} MB")

    surps = []
    t0 = time.time()
    total_steps = 5000
    print("Step    | NLL (nats) | Surprisal (bits) | 瞬时单步")
    print("-" * 55)
    for step in range(total_steps):
        c_tok = int(train_tokens[step])
        n_tok = int(train_tokens[step + 1])
        nats, bits = engine.step_stream(c_tok, n_tok)
        surps.append(nats)
        if step % 500 == 0 or step == total_steps - 1:
            recent = np.mean(surps[-100:])
            print(f"Step {step:4d} | NLL: {recent:5.2f} n | Surp: {recent/math.log(2):5.2f} b | Current: {nats:5.2f} n")

    torch.cuda.synchronize()
    dt = time.time() - t0
    print("-" * 55)
    print(f"Completed {total_steps} steps in {dt:.2f}s ({dt/total_steps*1000:.2f} ms/step)")
    print(f"Best 100-token window NLL: {min([np.mean(surps[i:i+100]) for i in range(len(surps)-100)]):.3f} nats")
