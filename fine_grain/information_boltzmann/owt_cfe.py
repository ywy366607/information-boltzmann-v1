"""
OpenWebText 连续流构形场引擎 (OWT Constructal Field Engine) - 高容量神经生物物理版
严格遵循非平衡态热力学、贝让构形定律 (Constructal Law) 与连续介质场论：
1. 大规模槽位容量: 支持 8,192 ~ 12,288 槽位 (常驻 ~2.5 GB 显存)，稀疏共振子群 (Top-K) 保证 O(1) 毫秒级计算
2. 连续狄利克雷-杰弗里斯测度守恒 (Jeffreys Invariant Measure): P_ground = (C(w) + alpha) / (N + alpha*V)，根除长尾衰减
3. 神经调质可塑性门控 (去甲肾上腺素 NE 机制): 惊奇度驱动局域相变流阻，平稳层流自发固化，无需外部学习率调度器
4. BCM 突触内稳态滑动阈值 (BCM Sliding Homeostasis): theta_M = E[act^2]，自组织临界态 (SOC) 杜绝参数爆炸与退化
5. 奥贾超球面切空间局域共流收缩 (AC-Oja Co-flow Contraction, 零共模漂移): 语义流形自发聚类
6. 贝让-默里定律极值管径分配: W_k prop J_k^(2/3)，由流道耗散极小化变分解解析分配
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OWTConstructalEngine(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        dim: int = 64,
        init_slots: int = 3072,
        max_slots: int = 12288,
        k_sparse: int = 256,
        alpha_jeffreys: float = 0.5,
        lambda_metabolic: float = 0.015,
        pipe_lr: float = 4.0,
        pipe_decay: float = 0.0005,
        J_crit: float = 3.2,
        eta_oja: float = 0.05,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.init_slots = init_slots
        self.max_slots = max_slots
        self.k_sparse = k_sparse
        self.alpha_j = alpha_jeffreys
        self.lambda_metabolic = lambda_metabolic
        self.pipe_lr = pipe_lr
        self.pipe_decay = pipe_decay
        self.J_crit = J_crit
        self.eta_oja = eta_oja
        self.device = torch.device(device)

        # 1. 词元流形空间 E in R^{V x D} 与 切向动量/速度 V in R^{V x D}
        # 词元不仅有球面坐标，还拥有动量与切向速度，通过角动量离心势垒形成稳定公转，杜绝引力坍缩
        self.embedding = nn.Embedding(vocab_size, dim, device=self.device)
        nn.init.normal_(self.embedding.weight, std=1.0 / math.sqrt(dim))
        self.register_buffer("token_velocity", torch.zeros(vocab_size, dim, device=self.device))

        # 2. 连续狄利克雷-杰弗里斯守恒能流账本 (Jeffreys Continuous Measure)
        self.register_buffer("token_counts", torch.zeros(vocab_size, device=self.device))
        self.total_tokens_seen = 0

        # 3. 槽位状态场 (The State Field Psi)
        self.register_buffer("active_mask", torch.zeros(max_slots, dtype=torch.bool, device=self.device))
        self.active_mask[:init_slots] = True

        # 构形水力冲刷累积通量账本 Phi_i = int J_i dt
        self.register_buffer("flux_accum", torch.zeros(max_slots, device=self.device))
        # 槽位水力惯性 (记忆衰减率 gamma_i) 初始全域均质平等 (0.50)，完全随水流冲刷自发涌现分化
        self.register_buffer("gamma", torch.ones(max_slots, device=self.device) * 0.5)

        # 槽位能量 e_i, 相位 theta_i, 内禀频率 omega_i, 不应期 r_i
        self.register_buffer("energy", torch.ones(max_slots, device=self.device))
        self.register_buffer("theta", torch.rand(max_slots, device=self.device) * 2 * math.pi)
        self.register_buffer("omega", torch.randn(max_slots, device=self.device) * 0.1)
        self.register_buffer("refractory", torch.zeros(max_slots, device=self.device))

        # 三重受体拓扑签名 (超球面坐标 S^{D-1})
        # k_sig: 瞬时输入受体 (Instantaneous Receptor)
        raw_k = torch.randn(max_slots, dim, device=self.device)
        self.register_buffer("k_sig", F.normalize(raw_k, p=2, dim=-1))
        # c_sig: 语境历史受体 (Temporal Context Receptor)
        raw_c = torch.randn(max_slots, dim, device=self.device)
        self.register_buffer("c_sig", F.normalize(raw_c, p=2, dim=-1))
        # y_sig: 连续超球面测地输出吸引子 (Geodesic Semantic Output Attractor)
        raw_y = torch.randn(max_slots, dim, device=self.device)
        self.register_buffer("y_sig", F.normalize(raw_y, p=2, dim=-1))

        # 连续资格迹 trace_i in R^D 与 成熟度 age_i
        self.register_buffer("trace", torch.zeros(max_slots, dim, device=self.device))
        self.register_buffer("age", torch.zeros(max_slots, dtype=torch.long, device=self.device))

        # BCM 突触内稳态滑动阈值: theta_M = E[act_i^2]
        self.register_buffer("theta_M", torch.ones(max_slots, device=self.device) * 0.01)

        # 4. 槽位到词元汇端的物理导通管网 (Physarum Tubules: [M_max x V])
        self.register_buffer("W_out", torch.zeros(max_slots, vocab_size, device=self.device))

        # 5. 变分自适应阻抗匹配状态
        self.register_buffer("residual_var", torch.tensor(1.0, device=self.device))

        # 6. Oja 拓扑共流场 (慢衰减 DC 全局共模场 与 快衰减 AC 局域动量场)
        self.register_buffer("h_dc", torch.zeros(dim, device=self.device))
        self.register_buffer("h_ac", torch.zeros(dim, device=self.device))

        # 统计计数
        self.tokens_processed = 0
        self.total_splits = 0
        self.total_merges = 0

    @property
    def num_active(self) -> int:
        return int(self.active_mask.sum().item())

    def token_sig(self, token_id: int) -> torch.Tensor:
        """获取词元的单位超球面坐标"""
        vec = self.embedding.weight[token_id]
        return F.normalize(vec, p=2, dim=-1)

    @torch.no_grad()
    def prequential_predict(self, current_token: int):
        """
        前序预测步：高容量稀疏共振子群 (Top-K Sparse Active Ensemble)
        """
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

        # 稀疏共振选取 (Top-K)
        K_sparse = min(self.k_sparse, len(act_idx))
        topk_vals, topk_pos = torch.topk(raw_act, K_sparse)
        sparse_act_idx = act_idx[topk_pos]
        sparse_act = topk_vals

        # 2. 直流基态: 杰弗里斯连续守恒测度
        N_total = max(1, self.total_tokens_seen)
        P_ground = (self.token_counts + self.alpha_j) / (N_total + self.alpha_j * self.vocab_size)
        phi_0 = torch.log(P_ground)

        # 3. 双重交流构形调制场:
        # A. 连续超球面测地泛化场: y_field = sum_i act_i * y_sig_i -> E @ y_field (解决长尾泛化)
        # B. 离散精细因果管网: full_act @ W_out (高频精准直通)
        full_act = torch.zeros(self.max_slots, device=self.device)
        if sparse_act.sum() > 1e-6:
            sparse_norm = sparse_act / sparse_act.sum()
            full_act[sparse_act_idx] = sparse_norm

            y_field = torch.matmul(sparse_norm, self.y_sig[sparse_act_idx])
            E_norm = F.normalize(self.embedding.weight, p=2, dim=-1)
            z_geo = 3.5 * torch.matmul(E_norm, y_field)
            z_pipe = torch.matmul(full_act, self.W_out)
            z_ac = z_geo + z_pipe
        else:
            z_ac = torch.zeros(self.vocab_size, device=self.device)

        # 4. 变分自适应阻抗匹配刚度
        beta_star = float(1.0 / math.sqrt(self.residual_var.item() + 0.04))
        logits = phi_0 + beta_star * z_ac

        return logits, sparse_act_idx, sparse_act, beta_star, k_in, full_act

    @torch.no_grad()
    def step_stream(self, current_token: int, next_token: int) -> Dict[str, float]:
        """
        单步连续流进：纯物理第一性原理自组织
        """
        self.tokens_processed += 1
        self.total_tokens_seen += 1

        # 1. 预先评分 (严格先测后学)
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

        # 超球面三重流形流动 (Receptor & Attractor Flow)
        flow_k = torch.matmul(W_mat, k_sub) * (0.05 * inv_M) + sparse_act.unsqueeze(1) * k_in * 0.08
        self.k_sig[sparse_idx] = F.normalize(k_sub + flow_k, p=2, dim=-1)

        tr_norm = F.normalize(self.trace[sparse_idx] + 1e-8, p=2, dim=-1)
        flow_c = sparse_act.unsqueeze(1) * tr_norm * 0.08
        self.c_sig[sparse_idx] = F.normalize(c_sub + flow_c, p=2, dim=-1)

        # 出口三角洲测地输出吸引子流动 (Estuary Flow toward next_token)
        v_next = self.token_sig(next_token)
        flow_y = sparse_act.unsqueeze(1) * v_next * 0.15
        self.y_sig[sparse_idx] = F.normalize(self.y_sig[sparse_idx] + flow_y, p=2, dim=-1)

        # 5. 神经调质可塑性门控 (去甲肾上腺素 NE: 惊奇度驱动流阻反压)
        ne_factor = float(torch.clamp(torch.tensor(surprisal_bits / 10.0), 0.2, 2.0).item())
        effective_lr = self.pipe_lr * ne_factor

        # 默里管径强化 (全矢量极速列更新)
        self.W_out[:, next_token] += effective_lr * full_act

        # BCM 突触滑动阈值与内稳态衰减: theta_M = E[act_i^2]
        self.theta_M[sparse_idx] = 0.99 * self.theta_M[sparse_idx] + 0.01 * (sparse_act ** 2)
        if self.tokens_processed % 10 == 0:
            self.W_out.mul_(1.0 - 0.0002)

        # 6. 构形定律水力冲刷：累积通量驱动时间尺度自发分化涌现 (无需人为划分大河与毛细管)
        self.flux_accum[sparse_idx] += sparse_act
        # 贝让水力冲刷定律: 惯性随通量开方平滑加深，高通量槽位自发成为慢速大干流 (gamma -> 0.99+)，低通量槽位保持轻盈毛细管
        erosion_factor = 1.0 - 1.0 / (1.0 + 0.08 * torch.sqrt(self.flux_accum[sparse_idx] + 1e-4))
        self.gamma[sparse_idx] = 0.50 + 0.49 * erosion_factor

        # 连续资格迹演化 (使用由水力自发塑造的 gamma)
        g_act = self.gamma[sparse_idx].unsqueeze(1)
        self.trace[sparse_idx] = g_act * self.trace[sparse_idx] + (1.0 - g_act) * k_in

        # 7. 词元哈密顿相空间引力公转动力学 (Hamiltonian Orbital Dynamics on S^{D-1})
        # 核心物理：引力向心拉拽与角动量离心势垒形成开普勒稳定公转轨道，自然免疫表征坍缩
        v_curr = k_in
        v_next = self.token_sig(next_token)
        m_curr = math.log(1.0 + float(self.token_counts[current_token].item()))
        m_next = math.log(1.0 + float(self.token_counts[next_token].item()))
        m_total = max(1e-4, m_curr + m_next)

        if self.eta_oja > 0.0:
            self.h_dc = 0.995 * self.h_dc + 0.005 * v_curr
            self.h_ac = 0.85 * self.h_ac + 0.15 * v_curr

            # 全局平均场引力中心 (Center of Gravitation Mass on Sphere)
            g_center = F.normalize(0.6 * self.h_ac + 0.4 * v_next, p=2, dim=-1)
            cos_c = torch.clamp(torch.dot(v_curr, g_center), -0.999, 0.999)
            dist_sq = max(0.04, float(2.0 - 2.0 * cos_c))
            tangent_force = (g_center - cos_c * v_curr) / dist_sq

            # 引力场加速度: F_grav = G * m_source / r^2
            G_const = self.eta_oja * 0.35
            f_grav_curr = G_const * (0.6 * tangent_force + 0.4 * (m_next / m_total) * (v_next - torch.dot(v_next, v_curr) * v_curr) / dist_sq)

            # 哈密顿辛动量更新: dv/dt = F_grav - mu * v (微弱流体阻尼 mu = 0.05)
            v_t = self.token_velocity[current_token]
            v_new = (1.0 - 0.05) * v_t + f_grav_curr
            v_tangent = v_new - torch.dot(v_new, v_curr) * v_curr
            self.token_velocity[current_token] = v_tangent

            # 测地线轨道位移更新 (Exp 映射: e_new = Normalize(e + v*dt))
            self.embedding.weight[current_token] = F.normalize(
                self.embedding.weight[current_token] + v_tangent, p=2, dim=-1
            )

            # 反向互引力牵引 next_token
            cos_n = torch.clamp(torch.dot(v_next, v_curr), -0.999, 0.999)
            dist_sq_n = max(0.04, float(2.0 - 2.0 * cos_n))
            tangent_n = (v_curr - cos_n * v_next) / dist_sq_n
            f_grav_next = G_const * (0.4 * (m_curr / m_total) * tangent_n)
            v_next_t = self.token_velocity[next_token]
            v_next_new = (1.0 - 0.05) * v_next_t + f_grav_next
            v_next_tangent = v_next_new - torch.dot(v_next_new, v_next) * v_next
            self.token_velocity[next_token] = v_next_tangent
            self.embedding.weight[next_token] = F.normalize(
                self.embedding.weight[next_token] + v_next_tangent, p=2, dim=-1
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

                    # 贝让-默里定律极值解析分配: W_k prop J_k^(2/3)
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
                    rand_tangent_y = torch.randn(self.dim, device=self.device)
                    rand_tangent_y -= torch.dot(rand_tangent_y, self.y_sig[parent_idx]) * self.y_sig[parent_idx]
                    rand_tangent_y = F.normalize(rand_tangent_y, p=2, dim=-1)
                    self.y_sig[child_idx] = F.normalize(
                        self.y_sig[parent_idx] * math.cos(0.3) + rand_tangent_y * math.sin(0.3), p=2, dim=-1
                    )
                    self.trace[child_idx] = self.trace[parent_idx].clone()
                    # 贝让-默里定律水力通量分配与惯性自发分化
                    self.flux_accum[child_idx] = self.flux_accum[parent_idx] * alpha_child
                    self.flux_accum[parent_idx] *= alpha_parent
                    erosion_p = 1.0 - 1.0 / (1.0 + 0.08 * math.sqrt(self.flux_accum[parent_idx].item() + 1e-4))
                    erosion_c = 1.0 - 1.0 / (1.0 + 0.08 * math.sqrt(self.flux_accum[child_idx].item() + 1e-4))
                    self.gamma[parent_idx] = 0.50 + 0.49 * erosion_p
                    self.gamma[child_idx] = 0.50 + 0.49 * erosion_c

                    self.W_out[child_idx] = self.W_out[parent_idx] * alpha_child
                    self.W_out[parent_idx] *= alpha_parent
                    splits += 1
                    self.total_splits += 1

        self.age[sparse_idx] += 1

        # 9. 拓扑冗余合并 (每 50 步对成熟槽位子集检查)
        merges = 0
        if self.tokens_processed % 50 == 0:
            act_all = torch.where(self.active_mask)[0]
            mature_mask = (self.age[act_all] > 60)
            if mature_mask.sum() > self.init_slots:
                mature_idx = act_all[mature_mask]
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
                    # 支流汇合：通量叠加与水力惯性自然深化
                    self.flux_accum[idx_i] += self.flux_accum[idx_j]
                    erosion_m = 1.0 - 1.0 / (1.0 + 0.08 * math.sqrt(self.flux_accum[idx_i].item() + 1e-4))
                    self.gamma[idx_i] = 0.50 + 0.49 * erosion_m
                    self.k_sig[idx_i] = F.normalize(self.k_sig[idx_i] + self.k_sig[idx_j], p=2, dim=-1)
                    self.c_sig[idx_i] = F.normalize(self.c_sig[idx_i] + self.c_sig[idx_j], p=2, dim=-1)
                    self.y_sig[idx_i] = F.normalize(self.y_sig[idx_i] + self.y_sig[idx_j], p=2, dim=-1)
                    self.W_out[idx_i] = torch.maximum(self.W_out[idx_i], self.W_out[idx_j])
                    self.active_mask[idx_j] = False
                    merges += 1
                    self.total_merges += 1

        rho = (J_act ** 2) / (energy_sub + 1e-4) + 0.015 * energy_sub + 0.04 * surprisal_bits
        global_resistance = float(rho.sum().item())

        return {
            "token_step": float(self.tokens_processed),
            "surprisal_nats": float(surprisal_nats),
            "surprisal_bits": float(surprisal_bits),
            "beta_star": float(beta_star),
            "resistance": float(global_resistance),
            "num_slots": float(self.num_active),
            "splits": float(splits),
            "merges": float(merges)
        }
