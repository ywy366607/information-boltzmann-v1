import sys, os, time, math
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing High-Capacity CFE on device: {device}")

class ScaledConstructalEngine(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        dim: int = 64,
        init_slots: int = 2048,
        max_slots: int = 8192,
        alpha_jeffreys: float = 0.5,
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
        self.alpha_j = alpha_jeffreys
        self.pipe_lr = pipe_lr
        self.J_crit = J_crit
        self.eta_oja = eta_oja
        self.device = torch.device(device)

        # 1. 词元嵌入空间
        self.embedding = nn.Embedding(vocab_size, dim, device=self.device)
        nn.init.normal_(self.embedding.weight, std=1.0 / math.sqrt(dim))

        # 2. 连续狄利克雷-杰弗里斯守恒能流账本 (Jeffreys Continuous Ledger)
        self.register_buffer("token_counts", torch.zeros(vocab_size, device=self.device))
        self.total_tokens_seen = 0

        # 3. 槽位状态场 (The State Field Psi)
        self.register_buffer("active_mask", torch.zeros(max_slots, dtype=torch.bool, device=self.device))
        self.active_mask[:init_slots] = True

        # 多时间尺度连续记忆衰减率 gamma: 1/3 快流 (0.50), 1/3 中流 (0.82), 1/3 慢流 (0.96)
        gammas = torch.empty(max_slots, device=self.device)
        n_third = max_slots // 3
        gammas[:n_third] = torch.empty(n_third, device=self.device).uniform_(0.45, 0.65)
        gammas[n_third: 2 * n_third] = torch.empty(n_third, device=self.device).uniform_(0.75, 0.88)
        gammas[2 * n_third:] = torch.empty(max_slots - 2 * n_third, device=self.device).uniform_(0.92, 0.98)
        self.register_buffer("gamma", gammas)

        # 槽位能量 e_i, 相位 theta_i, 内禀频率 omega_i, 不应期 r_i
        self.register_buffer("energy", torch.ones(max_slots, device=self.device))
        self.register_buffer("theta", torch.rand(max_slots, device=self.device) * 2 * math.pi)
        self.register_buffer("omega", torch.randn(max_slots, device=self.device) * 0.1)
        self.register_buffer("refractory", torch.zeros(max_slots, device=self.device))

        # 双受体拓扑签名: k_sig (输入), c_sig (语境)
        raw_k = torch.randn(max_slots, dim, device=self.device)
        self.register_buffer("k_sig", F.normalize(raw_k, p=2, dim=-1))
        raw_c = torch.randn(max_slots, dim, device=self.device)
        self.register_buffer("c_sig", F.normalize(raw_c, p=2, dim=-1))

        self.register_buffer("trace", torch.zeros(max_slots, dim, device=self.device))
        self.register_buffer("age", torch.zeros(max_slots, dtype=torch.long, device=self.device))

        # BCM 突触内稳态滑动阈值: theta_M = E[act_i^2]
        self.register_buffer("theta_M", torch.ones(max_slots, device=self.device) * 0.01)

        # 4. 槽位到词元导通管网 (Physarum Tubules: [8192, 50257] float32 ~ 1.64 GB)
        self.register_buffer("W_out", torch.zeros(max_slots, vocab_size, device=self.device))

        # 5. 变分自适应阻抗匹配状态
        self.register_buffer("residual_var", torch.tensor(1.0, device=self.device))

        # 6. Oja 拓扑共流场 (慢衰减 DC 与 快衰减 AC)
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
        k_in = self.token_sig(current_token) # [dim]

        # 1. 词元与语境双受体匹配 (对所有 active 槽位进行极速超球面内积)
        act_idx = torch.where(self.active_mask)[0]
        k_act = self.k_sig[act_idx]
        c_act = self.c_sig[act_idx]
        tr_act = F.normalize(self.trace[act_idx] + 1e-8, p=2, dim=-1)
        r_act = self.refractory[act_idx]

        cos_k = F.relu(torch.matmul(k_act, k_in))
        cos_c = torch.clamp((torch.sum(c_act * tr_act, dim=-1) + 1.0) * 0.5, 0.2, 1.0)
        raw_act = (1.0 - r_act) * cos_k * cos_c

        # 选取 Top-K 最强共振活跃子群 (Sparse Active Ensemble, K=256)
        # 既保留了全体 8192 个槽位的广袤专业化记忆，又使局域作用力计算严格锁定在毫秒级
        K_sparse = min(256, len(act_idx))
        topk_vals, topk_pos = torch.topk(raw_act, K_sparse)

        # 活跃集索引
        sparse_act_idx = act_idx[topk_pos]
        sparse_act = topk_vals

        # 2. 直流基态: 杰弗里斯连续守恒测度 phi_0 = ln((C(w) + alpha) / (N + alpha*V))
        N_total = max(1, self.total_tokens_seen)
        P_ground = (self.token_counts + self.alpha_j) / (N_total + self.alpha_j * self.vocab_size)
        phi_0 = torch.log(P_ground)

        # 3. 交流构形调制场: 矢量 GEMV
        full_act = torch.zeros(self.max_slots, device=self.device)
        if sparse_act.sum() > 1e-6:
            sparse_norm = sparse_act / sparse_act.sum()
            full_act[sparse_act_idx] = sparse_norm
            z_ac = torch.matmul(full_act, self.W_out)
        else:
            z_ac = torch.zeros(self.vocab_size, device=self.device)

        # 4. 变分自适应阻抗匹配刚度
        beta_star = float(1.0 / math.sqrt(self.residual_var.item() + 0.04))
        logits = phi_0 + beta_star * z_ac

        return logits, sparse_act_idx, sparse_act, beta_star, k_in, full_act

    @torch.no_grad()
    def step_stream(self, current_token: int, next_token: int):
        self.tokens_processed += 1
        self.total_tokens_seen += 1

        # 1. 预先评分
        logits, sparse_idx, sparse_act, beta_star, k_in, full_act = self.prequential_predict(current_token)
        target_tensor = torch.tensor([next_token], device=self.device)
        surprisal_nats = F.cross_entropy(logits.unsqueeze(0), target_tensor).item()
        surprisal_bits = surprisal_nats / math.log(2)

        # 2. 杰弗里斯狄利克雷计数更新 (严格能量守恒，零长流长尾衰减)
        self.token_counts[next_token] += 1.0

        # 3. 动态阻抗匹配
        normalized_res = (surprisal_nats / 10.825) ** 2
        self.residual_var = 0.98 * self.residual_var + 0.02 * normalized_res

        # 4. 算子 1 & 2: 活跃子群上的 Kuramoto 锁相与切空间流动
        M_sub = len(sparse_idx)
        k_sub = self.k_sig[sparse_idx]
        c_sub = self.c_sig[sparse_idx]
        theta_sub = self.theta[sparse_idx]
        energy_sub = self.energy[sparse_idx]
        r_sub = self.refractory[sparse_idx]

        K_mat = F.relu(torch.matmul(k_sub, k_sub.T))
        d_theta = theta_sub.unsqueeze(0) - theta_sub.unsqueeze(1)
        W_mat = (1.0 - r_sub).unsqueeze(1) * K_mat * torch.cos(d_theta)

        inv_M = 1.0 / max(1, M_sub)
        flux_term = (K_mat * torch.sin(d_theta)).sum(dim=1) * inv_M
        J_act = energy_sub * (1.0 - r_sub) * flux_term

        kuramoto_force = (W_mat * torch.sin(d_theta) * K_mat).sum(dim=1) * inv_M
        self.theta[sparse_idx] = (theta_sub + self.omega[sparse_idx] + 0.1 * kuramoto_force) % (2 * math.pi)
        self.refractory[sparse_idx] = torch.clamp(0.85 * r_sub + 0.15 * torch.sigmoid(energy_sub - 1.0), 0.0, 1.0)

        # 超球面双流形流动
        flow_k = torch.matmul(W_mat, k_sub) * (0.05 * inv_M) + sparse_act.unsqueeze(1) * k_in * 0.08
        self.k_sig[sparse_idx] = F.normalize(k_sub + flow_k, p=2, dim=-1)

        tr_norm = F.normalize(self.trace[sparse_idx] + 1e-8, p=2, dim=-1)
        flow_c = sparse_act.unsqueeze(1) * tr_norm * 0.08
        self.c_sig[sparse_idx] = F.normalize(c_sub + flow_c, p=2, dim=-1)

        # 5. 神经调质可塑性门控 (去甲肾上腺素 NE: 惊奇度驱动)
        # 惊奇度越大，可塑性越强；平稳预测时自发冻结流道，防止过拟合
        ne_factor = float(torch.clamp(torch.tensor(surprisal_bits / 10.0), 0.2, 2.0).item())
        effective_lr = self.pipe_lr * ne_factor

        # 默里管径强化 (全矢量极速列更新)
        self.W_out[:, next_token] += effective_lr * full_act

        # BCM 突触滑动阈值与内稳态衰减: theta_M 随激活滑动
        self.theta_M[sparse_idx] = 0.99 * self.theta_M[sparse_idx] + 0.01 * (sparse_act ** 2)
        if self.tokens_processed % 10 == 0:
            # 仅对长期过载的槽位进行温和耗散
            self.W_out.mul_(1.0 - 0.0002)

        # 6. 多时间尺度连续时空资格迹演化
        g_act = self.gamma[sparse_idx].unsqueeze(1)
        self.trace[sparse_idx] = g_act * self.trace[sparse_idx] + (1.0 - g_act) * k_in

        # 7. 奥贾超球面切空间局域共流收缩 (剔除共模漂移)
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

        # 8. 自由能成核相变 (Delta F < 0 <=> J_node > J_crit)
        J_node = sparse_act * surprisal_bits
        splits = 0
        if self.num_active < self.max_slots:
            phase_change_mask = (J_node > self.J_crit)
            if phase_change_mask.any():
                cand_pos = torch.argmax(J_node).item()
                parent_idx = sparse_idx[cand_pos].item()
                free_slots = torch.where(~self.active_mask)[0]
                if len(free_slots) > 0:
                    child_idx = free_slots[0].item()
                    self.active_mask[child_idx] = True
                    self.age[child_idx] = 0
                    self.age[parent_idx] = 0

                    J1 = max(0.01, float(sparse_act[cand_pos].item()))
                    J2 = max(0.01, float(surprisal_bits))
                    J1_23 = J1 ** (2.0 / 3.0)
                    J2_23 = J2 ** (2.0 / 3.0)
                    alpha_parent = J1_23 / (J1_23 + J2_23)
                    alpha_child = J2_23 / (J1_23 + J2_23)

                    self.energy[child_idx] = self.energy[parent_idx] * alpha_child
                    self.energy[parent_idx] *= alpha_parent
                    self.theta[child_idx] = (self.theta[parent_idx] + math.pi) % (2 * math.pi)
                    self.omega[child_idx] = -self.omega[parent_idx]
                    self.refractory[child_idx] = 0.0

                    rand_tangent = torch.randn(self.dim, device=self.device)
                    rand_tangent -= torch.dot(rand_tangent, self.k_sig[parent_idx]) * self.k_sig[parent_idx]
                    rand_tangent = F.normalize(rand_tangent, p=2, dim=-1)
                    delta_angle = 0.55
                    self.k_sig[child_idx] = F.normalize(
                        self.k_sig[parent_idx] * math.cos(delta_angle) + rand_tangent * math.sin(delta_angle), p=2, dim=-1
                    )
                    self.c_sig[child_idx] = F.normalize(0.6 * self.c_sig[parent_idx] + 0.4 * tr_norm[cand_pos], p=2, dim=-1)
                    self.trace[child_idx] = self.trace[parent_idx].clone()
                    self.gamma[child_idx] = self.gamma[parent_idx]

                    self.W_out[child_idx] = self.W_out[parent_idx] * alpha_child
                    self.W_out[parent_idx] *= alpha_parent
                    splits += 1
                    self.total_splits += 1

        self.age[sparse_idx] += 1

        # 9. 拓扑冗余合并 (每 100 步在成熟槽位间检查)
        merges = 0
        if self.tokens_processed % 50 == 0:
            act_all = torch.where(self.active_mask)[0]
            mature_mask = (self.age[act_all] > 60)
            if mature_mask.sum() > self.init_slots:
                mature_idx = act_all[mature_mask]
                # 采样最多 512 个成熟槽位做冗余检查，保证 O(1) 毫秒耗时
                sub_mature = mature_idx[:min(512, len(mature_idx))]
                k_m = self.k_sig[sub_mature]
                sim_m = torch.matmul(k_m, k_m.T)
                sim_m.fill_diagonal_(0.0)
                max_sim, max_p = sim_m.max(dim=1)
                pair_i = torch.argmax(max_sim)
                if max_sim[pair_i] > 0.96:
                    idx_i = sub_mature[pair_i]
                    idx_j = sub_mature[max_p[pair_i]]
                    self.energy[idx_i] += self.energy[idx_j]
                    self.k_sig[idx_i] = F.normalize(self.k_sig[idx_i] + self.k_sig[idx_j], p=2, dim=-1)
                    self.c_sig[idx_i] = F.normalize(self.c_sig[idx_i] + self.c_sig[idx_j], p=2, dim=-1)
                    self.W_out[idx_i] = torch.maximum(self.W_out[idx_i], self.W_out[idx_j])
                    self.active_mask[idx_j] = False
                    merges += 1
                    self.total_merges += 1

        rho = (J_act ** 2) / (energy_sub + 1e-4) + 0.015 * energy_sub + 0.04 * surprisal_bits
        global_resistance = float(rho.sum().item())

        return {
            "surprisal_nats": surprisal_nats,
            "surprisal_bits": surprisal_bits,
            "beta_star": beta_star,
            "resistance": global_resistance,
            "num_slots": float(self.num_active),
            "splits": float(splits),
            "merges": float(merges)
        }

if __name__ == "__main__":
    train_tokens = np.load("data/ib_owt_gpt2/train.npy", mmap_mode="r")
    engine = ScaledConstructalEngine(vocab_size=50257, dim=64, init_slots=2048, max_slots=8192, device=device)
    allocated_mb = torch.cuda.memory_allocated() / 1e6
    print(f"High-Capacity CFE Initialized: {engine.num_active} initial slots (Max {engine.max_slots})")
    print(f"Allocated VRAM: {allocated_mb:.1f} MB (~{allocated_mb/1024:.2f} GB) / 4096 MB")

    t0 = time.time()
    surps = []
    total_steps = 2000
    for step in range(total_steps):
        c_tok = int(train_tokens[step])
        n_tok = int(train_tokens[step + 1])
        res = engine.step_stream(c_tok, n_tok)
        surps.append(res["surprisal_nats"])
        if step % 500 == 0 or step == total_steps - 1:
            recent_nats = np.mean(surps[-100:])
            print(f"Step {step:4d} | Slots: {int(res['num_slots']):4d} | NLL (last 100): {recent_nats:.3f} n ({recent_nats/math.log(2):.3f} b) | beta*: {res['beta_star']:.2f}")

    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"\nCompleted {total_steps} tokens in {dt:.2f}s ({dt/total_steps*1000:.2f} ms/step)")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
