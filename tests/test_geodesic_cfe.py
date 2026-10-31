import sys, os, time, math
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing Geodesic Semantic Field + Scaled CFE on device: {device}")

class GeodesicConstructalEngine(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        dim: int = 64,
        init_slots: int = 2048,
        max_slots: int = 4096,
        k_sparse: int = 128,
        alpha_j: float = 0.5,
        pipe_lr: float = 4.0,
        J_crit: float = 3.2,
        eta_oja: float = 0.05,
        device: str = "cuda"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.init_slots = init_slots
        self.max_slots = max_slots
        self.k_sparse = k_sparse
        self.alpha_j = alpha_j
        self.pipe_lr = pipe_lr
        self.J_crit = J_crit
        self.eta_oja = eta_oja
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

        gammas = torch.empty(max_slots, device=self.device)
        n_third = max_slots // 3
        gammas[:n_third] = torch.empty(n_third, device=self.device).uniform_(0.45, 0.65)
        gammas[n_third: 2 * n_third] = torch.empty(n_third, device=self.device).uniform_(0.75, 0.88)
        gammas[2 * n_third:] = torch.empty(max_slots - 2 * n_third, device=self.device).uniform_(0.92, 0.98)
        self.register_buffer("gamma", gammas)

        # 槽位能量、相位、内禀频率、不应期
        self.register_buffer("energy", torch.ones(max_slots, device=self.device))
        self.register_buffer("theta", torch.rand(max_slots, device=self.device) * 2 * math.pi)
        self.register_buffer("omega", torch.randn(max_slots, device=self.device) * 0.1)
        self.register_buffer("refractory", torch.zeros(max_slots, device=self.device))

        # 三重受体与输出吸引子 (超球面坐标 S^{D-1}):
        # k_sig: 瞬时输入受体
        # c_sig: 语境历史受体
        # y_sig: 输出语义泛化吸引子 (Geodesic Output Attractor)
        raw_k = torch.randn(max_slots, dim, device=self.device)
        self.register_buffer("k_sig", F.normalize(raw_k, p=2, dim=-1))
        raw_c = torch.randn(max_slots, dim, device=self.device)
        self.register_buffer("c_sig", F.normalize(raw_c, p=2, dim=-1))
        raw_y = torch.randn(max_slots, dim, device=self.device)
        self.register_buffer("y_sig", F.normalize(raw_y, p=2, dim=-1))

        self.register_buffer("trace", torch.zeros(max_slots, dim, device=self.device))
        self.register_buffer("age", torch.zeros(max_slots, dtype=torch.long, device=self.device))
        self.register_buffer("theta_M", torch.ones(max_slots, device=self.device) * 0.01)

        # 4. 精细因果管网 W_out: [max_slots, vocab_size]
        self.register_buffer("W_out", torch.zeros(max_slots, vocab_size, device=self.device))

        # 5. 变分自适应阻抗匹配状态
        self.register_buffer("residual_var", torch.tensor(1.0, device=self.device))

        # 6. Oja 共流场
        self.register_buffer("h_dc", torch.zeros(dim, device=self.device))
        self.register_buffer("h_ac", torch.zeros(dim, device=self.device))

        self.tokens_processed = 0
        self.total_splits = 0
        self.total_merges = 0

    @property
    def num_active(self) -> int:
        return int(self.active_mask.sum().item())

    def token_sig(self, token_id: int) -> torch.Tensor:
        return F.normalize(self.embedding.weight[token_id], p=2, dim=-1)

    @torch.no_grad()
    def prequential_predict(self, current_token: int):
        k_in = self.token_sig(current_token)

        # 1. 双受体共振匹配
        act_idx = torch.where(self.active_mask)[0]
        k_act = self.k_sig[act_idx]
        c_act = self.c_sig[act_idx]
        tr_act = F.normalize(self.trace[act_idx] + 1e-8, p=2, dim=-1)
        r_act = self.refractory[act_idx]

        cos_k = F.relu(torch.matmul(k_act, k_in))
        cos_c = torch.clamp((torch.sum(c_act * tr_act, dim=-1) + 1.0) * 0.5, 0.2, 1.0)
        raw_act = (1.0 - r_act) * cos_k * cos_c

        # 稀疏共振子群 (Top-K)
        K_sparse = min(self.k_sparse, len(act_idx))
        topk_vals, topk_pos = torch.topk(raw_act, K_sparse)
        sparse_idx = act_idx[topk_pos]
        sparse_act = topk_vals

        # 2. 直流基态: 杰弗里斯连续守恒测度
        N_total = max(1, self.total_tokens_seen)
        P_ground = (self.token_counts + self.alpha_j) / (N_total + self.alpha_j * self.vocab_size)
        phi_0 = torch.log(P_ground)

        # 3. 双重交流场:
        # A. 连续超球面测地泛化场: y_field = sum_i act_i * y_sig_i -> E @ y_field
        # B. 离散精细管网通量: full_act @ W_out
        full_act = torch.zeros(self.max_slots, device=self.device)
        if sparse_act.sum() > 1e-6:
            sparse_norm = sparse_act / sparse_act.sum()
            full_act[sparse_idx] = sparse_norm

            # 连续语义泛化场 (零样本几何辐射)
            y_field = torch.matmul(sparse_norm, self.y_sig[sparse_idx]) # [dim]
            # 词表超球面投影归一化内积
            E_norm = F.normalize(self.embedding.weight, p=2, dim=-1)
            z_geo = 3.5 * torch.matmul(E_norm, y_field) # [vocab_size]

            # 精细离散管网
            z_pipe = torch.matmul(full_act, self.W_out) # [vocab_size]
            z_ac = z_geo + z_pipe
        else:
            z_ac = torch.zeros(self.vocab_size, device=self.device)

        # 4. 变分自适应阻抗匹配刚度
        beta_star = float(1.0 / math.sqrt(self.residual_var.item() + 0.04))
        logits = phi_0 + beta_star * z_ac

        return logits, sparse_idx, sparse_act, beta_star, k_in, tr_act[topk_pos], full_act, phi_0

    @torch.no_grad()
    def step_stream(self, current_token: int, next_token: int):
        self.tokens_processed += 1
        self.total_tokens_seen += 1

        # 1. 预先评分
        logits, sparse_idx, sparse_act, beta_star, k_in, tr_sparse, full_act, phi_0 = self.prequential_predict(current_token)
        target_tensor = torch.tensor([next_token], device=self.device)
        surprisal_nats = F.cross_entropy(logits.unsqueeze(0), target_tensor).item()
        surprisal_bits = surprisal_nats / math.log(2)

        # 2. 杰弗里斯狄利克雷计数更新
        self.token_counts[next_token] += 1.0

        # 3. 动态阻抗匹配
        normalized_res = (surprisal_nats / 10.825) ** 2
        self.residual_var = 0.98 * self.residual_var + 0.02 * normalized_res

        # 4. 算子 1 & 2: 锁相与切空间流动
        M_sub = len(sparse_idx)
        k_sub = self.k_sig[sparse_idx]
        c_sub = self.c_sig[sparse_idx]
        y_sub = self.y_sig[sparse_idx]
        theta_sub = self.theta[sparse_idx]
        energy_sub = self.energy[sparse_idx]
        r_sub = self.refractory[sparse_idx]

        K_mat = F.relu(torch.matmul(k_sub, k_sub.T))
        d_theta = theta_sub.unsqueeze(0) - theta_sub.unsqueeze(1)
        W_mat = (1.0 - r_sub).unsqueeze(1) * K_mat * torch.cos(d_theta)

        inv_M = 1.0 / max(1, M_sub)
        kuramoto_force = (W_mat * torch.sin(d_theta) * K_mat).sum(dim=1) * inv_M
        self.theta[sparse_idx] = (theta_sub + self.omega[sparse_idx] + 0.1 * kuramoto_force) % (2 * math.pi)
        self.refractory[sparse_idx] = torch.clamp(0.85 * r_sub + 0.15 * torch.sigmoid(energy_sub - 1.0), 0.0, 1.0)

        # 超球面双流形流动
        flow_k = torch.matmul(W_mat, k_sub) * (0.05 * inv_M) + sparse_act.unsqueeze(1) * k_in * 0.08
        self.k_sig[sparse_idx] = F.normalize(k_sub + flow_k, p=2, dim=-1)

        flow_c = sparse_act.unsqueeze(1) * tr_sparse * 0.08
        self.c_sig[sparse_idx] = F.normalize(c_sub + flow_c, p=2, dim=-1)

        # 5. 超球面测地泛化吸引子更新 (对齐真实目标词的几何向量)
        v_next = self.token_sig(next_token)
        flow_y = sparse_act.unsqueeze(1) * v_next * 0.15
        self.y_sig[sparse_idx] = F.normalize(y_sub + flow_y, p=2, dim=-1)

        # 6. 神经调质可塑性门控 (去甲肾上腺素 NE)
        ne_factor = float(torch.clamp(torch.tensor(surprisal_bits / 10.0), 0.2, 2.0).item())
        effective_lr = self.pipe_lr * ne_factor

        # 默里管径强化 (对数几率物理松弛)
        self.W_out[:, next_token] += effective_lr * full_act

        # BCM 突触滑动阈值与耗散
        self.theta_M[sparse_idx] = 0.99 * self.theta_M[sparse_idx] + 0.01 * (sparse_act ** 2)
        if self.tokens_processed % 10 == 0:
            self.W_out.mul_(1.0 - 0.0002)

        # 7. 多时间尺度连续资格迹演化
        g_act = self.gamma[sparse_idx].unsqueeze(1)
        self.trace[sparse_idx] = g_act * self.trace[sparse_idx] + (1.0 - g_act) * k_in

        # 8. 奥贾超球面共流切空间收缩 (零共模漂移)
        if self.eta_oja > 0.0:
            self.h_dc = 0.995 * self.h_dc + 0.005 * k_in
            self.h_ac = 0.85 * self.h_ac + 0.15 * k_in
            if self.tokens_processed > 20:
                delta_h = self.h_ac - self.h_dc
                delta_norm = F.normalize(delta_h + 1e-8, p=2, dim=-1)
                proj = torch.dot(k_in, delta_norm)
                delta_e = self.eta_oja * (delta_norm - proj * k_in)
                self.embedding.weight[current_token] = F.normalize(
                    self.embedding.weight[current_token] + delta_e, p=2, dim=-1
                )

        return surprisal_nats, surprisal_bits

if __name__ == "__main__":
    train_tokens = np.load("data/ib_owt_gpt2/train.npy", mmap_mode="r")
    engine = GeodesicConstructalEngine(
        vocab_size=50257,
        dim=64,
        init_slots=2048,
        max_slots=4096,
        k_sparse=128,
        pipe_lr=4.0,
        eta_oja=0.05,
        device=device
    )
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
