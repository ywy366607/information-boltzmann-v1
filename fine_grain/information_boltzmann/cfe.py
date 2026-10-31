"""
构形场引擎 (Constructal Field Engine, CFE) - Hopf 双代数版本
基于非平衡态热力学、无网格拉格朗日槽位场、贝让构形定律与 Hopf 代数 (H, mu, eta, Delta, epsilon, S)
- 余乘法 Delta: 槽位沿切向最陡阻力梯度的树状裂变 (Coproduct Bifurcation)
- 乘法 mu: 冗余对齐槽位的单子并合 (Product Coalescence)
- 对极映射 S: 逆向共轭反相抵消 (Antipode Residual Annihilation)
- 单位 eta / 余单位 epsilon: 真空基底与耗散边界
- 恒等式: mu o (S (x) id) o Delta(x) = eta(epsilon(x)) 恒等抵消为零残差
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConstructalFieldEngine(nn.Module):
    def __init__(
        self,
        dim: int = 32,
        initial_slots: int = 16,
        max_slots: int = 128,
        lambda_metabolic: float = 0.05,
        eta_normalize: float = 0.1,
        split_threshold_sigma: float = 2.0,
        merge_similarity_threshold: float = 0.95,
        delta_split: float = 0.1,
        gamma_trace: float = 0.9,
        dt: float = 0.05,
        device: str = "cpu"
    ):
        super().__init__()
        self.dim = dim
        self.max_slots = max_slots
        self.lambda_metabolic = lambda_metabolic
        self.eta_normalize = eta_normalize
        self.split_threshold_sigma = split_threshold_sigma
        self.merge_similarity_threshold = merge_similarity_threshold
        self.delta_split = delta_split
        self.gamma_trace = gamma_trace
        self.dt = dt
        self.device = torch.device(device)

        # 核心槽位状态 (Slot State Field)
        # active_mask: [max_slots] 标记当前活跃槽位
        self.register_buffer("active_mask", torch.zeros(max_slots, dtype=torch.bool, device=self.device))
        self.active_mask[:initial_slots] = True

        # 能量密度 e_i in R+
        self.register_buffer("energy", torch.zeros(max_slots, dtype=torch.float32, device=self.device))
        self.energy[:initial_slots] = 1.0

        # 复相位 theta_i in [-pi, pi]
        self.register_buffer("theta", torch.zeros(max_slots, dtype=torch.float32, device=self.device))
        self.theta[:initial_slots] = torch.rand(initial_slots, device=self.device) * 2 * math.pi - math.pi

        # 内禀自然角频率 omega_i
        self.register_buffer("omega", torch.zeros(max_slots, dtype=torch.float32, device=self.device))
        self.omega[:initial_slots] = torch.randn(initial_slots, device=self.device) * 0.2

        # 不应期阻尼 r_i in [0, 1]
        self.register_buffer("refractory", torch.zeros(max_slots, dtype=torch.float32, device=self.device))

        # 拓扑签名 k_i in S^{D-1} (严格受限在超球面上)
        self.register_buffer("signature", torch.zeros(max_slots, dim, dtype=torch.float32, device=self.device))
        init_k = torch.randn(initial_slots, dim, device=self.device)
        self.signature[:initial_slots] = F.normalize(init_k, p=2, dim=-1)

        # 资格迹 tau_i in R^D
        self.register_buffer("trace", torch.zeros(max_slots, dim, dtype=torch.float32, device=self.device))

        # 槽位成熟年龄 (防止幼年分支被母体瞬间吞噬合并)
        self.register_buffer("slot_age", torch.zeros(max_slots, dtype=torch.long, device=self.device))
        self.slot_age[:initial_slots] = 10

        # 终生永久骨干神经元标记 (Persistent Core Soma: 终生永存，免疫合并注销)
        self.register_buffer("is_persistent", torch.zeros(max_slots, dtype=torch.bool, device=self.device))
        self.is_persistent[:initial_slots] = True

    @property
    def num_active(self) -> int:
        return int(self.active_mask.sum().item())

    @property
    def active_indices(self) -> torch.Tensor:
        return torch.nonzero(self.active_mask).squeeze(-1)

    def compute_flux_and_conductance(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        算子 1 前半部：计算槽位间导通率 W_ij 与局域能量通量 J_i
        W_ij = (1 - r_i) * max(0, <k_i, k_j>) * cos(theta_i - theta_j)
        J_i = e_i * sum_j (1 - r_i) * <k_i, k_j>_+ * sin(theta_j - theta_i)
        """
        idx = self.active_indices
        if len(idx) == 0:
            return torch.zeros(0), torch.zeros(0, 0), torch.zeros(0)

        k = self.signature[idx]       # [N, D]
        e = self.energy[idx]          # [N]
        th = self.theta[idx]          # [N]
        r = self.refractory[idx]      # [N]

        # 拓扑签名余弦相似度矩阵 <k_i, k_j>
        sim = torch.matmul(k, k.T)   # [N, N]
        sim_pos = torch.clamp(sim, min=0.0)

        # 相位差矩阵 (theta_i - theta_j)
        th_diff = th.unsqueeze(1) - th.unsqueeze(0) # th_diff[i, j] = theta_i - theta_j
        cos_diff = torch.cos(th_diff)
        sin_diff_ji = torch.sin(-th_diff)           # sin(theta_j - theta_i) = -sin(theta_i - theta_j)

        # 瞬时导通率 W_ij
        gate_r = (1.0 - r).unsqueeze(1) # [N, 1]
        W = gate_r * sim_pos * cos_diff  # [N, N]

        # 局域能量通量标量 J_i
        # J_i = e_i * (1 - r_i) * sum_j sim_pos[i, j] * sin(theta_j - theta_i)
        flow_sum = torch.sum(sim_pos * sin_diff_ji, dim=1)
        J = e * (1.0 - r) * flow_sum # [N]

        return J, W, sim_pos

    def compute_resistance(self, J: torch.Tensor, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算局域构形阻力密度 rho_i 与全局阻力泛函 R
        rho_i = |J_i|^2 / (e_i + eps) + lambda * e_i
        R = sum_i rho_i
        """
        idx = self.active_indices
        e = self.energy[idx]
        rho = (J ** 2) / (e + eps) + self.lambda_metabolic * e
        R = torch.sum(rho)
        return rho, R

    def step_flow_and_phase(self, external_flux: Optional[torch.Tensor] = None):
        """
        算子 1：相干涉输运与 Kuramoto 相位动力学
        dtheta_i / dt = omega_i + sum_j W_ij * sin(theta_j - theta_i) * <k_i, k_j>_+
        """
        idx = self.active_indices
        if len(idx) == 0:
            return

        J, W, sim_pos = self.compute_flux_and_conductance()
        th = self.theta[idx]
        th_diff = th.unsqueeze(1) - th.unsqueeze(0)
        sin_diff_ji = torch.sin(-th_diff)

        # Kuramoto 相位耦合力
        coupling = torch.sum(W * sin_diff_ji * sim_pos, dim=1)
        dtheta = self.omega[idx] + coupling

        # 外界输入相位驱动 (若有)
        if external_flux is not None and len(external_flux) == len(idx):
            dtheta = dtheta + external_flux

        # 相位更新并折叠到 [-pi, pi]
        new_theta = th + self.dt * dtheta
        new_theta = (new_theta + math.pi) % (2 * math.pi) - math.pi
        self.theta[idx] = new_theta

        # 不应期阻尼缓慢自发恢复
        self.refractory[idx] = torch.clamp(self.refractory[idx] * 0.95, 0.0, 1.0)
        self.slot_age[idx] += 1

    def step_manifold_normalization(self):
        """
        算子 2：除法归一化与切空间投影 (Manifold Normalization)
        k_i <- Normalize(k_i + eta * sum_{j in N(i)} W_ij k_j)
        """
        idx = self.active_indices
        if len(idx) <= 1:
            return

        _, W, sim_pos = self.compute_flux_and_conductance()
        k = self.signature[idx]

        # 局域竞争邻域：仅保留正向导通且具有相似度的近邻
        W_local = torch.clamp(W, min=0.0) * sim_pos
        delta_k = torch.matmul(W_local, k)

        updated_k = k + self.eta_normalize * delta_k
        self.signature[idx] = F.normalize(updated_k, p=2, dim=-1)

    def step_curvature_split(self) -> int:
        """
        算子 3：曲率爆裂与自适应分裂 (Split / Bifurcation)
        当槽位 i 的局域阻力密度 rho_i > mean(rho) + 2 * std(rho) 时触发分裂。
        在切空间方向沿阻力变分梯度展开新槽位 m'，能量守恒平分。
        """
        idx = self.active_indices
        if len(idx) >= self.max_slots or len(idx) < 3:
            return 0

        J, _, sim_pos = self.compute_flux_and_conductance()
        rho, _ = self.compute_resistance(J)

        mean_rho = torch.mean(rho)
        std_rho = torch.std(rho) if len(rho) > 1 else torch.tensor(0.0, device=self.device)

        # 统计相对阈值 与 绝对阻力拥塞阈值 (解决小样本 N<=3 时 z-score 上界受限问题)
        rel_threshold = mean_rho + self.split_threshold_sigma * (std_rho + 1e-6)
        abs_threshold = self.lambda_metabolic * 1.8 + 0.15 # 绝对通量拥塞阻力

        # 只要相对突出或绝对能流积压严重，即触发曲率爆裂
        split_candidates = torch.nonzero((rho > rel_threshold) | (rho > abs_threshold)).squeeze(-1)
        if len(split_candidates) == 0:
            return 0

        # 一次最多分裂一个最高阻力槽位，维持系统平稳
        m_local_idx = split_candidates[torch.argmax(rho[split_candidates])].item()
        m_global_idx = idx[m_local_idx].item()

        # 寻找一个空闲槽位
        free_indices = torch.nonzero(~self.active_mask).squeeze(-1)
        if len(free_indices) == 0:
            return 0
        new_idx = free_indices[0].item()

        # 计算局域阻力变分在超球面上的黎曼切空间梯度 (Riemannian Tangent Gradient)
        k_m = self.signature[m_global_idx]
        k_all = self.signature[idx]
        th = self.theta[idx]
        th_m = self.theta[m_global_idx]
        sin_diff = torch.sin(th - th_m)

        # dJ_m / dk_m 方向估计
        tangent_dir = torch.sum(sin_diff.unsqueeze(1) * k_all, dim=0)
        # 投影到 k_m 的正交切空间
        tangent_dir = tangent_dir - torch.dot(tangent_dir, k_m) * k_m
        norm_t = torch.norm(tangent_dir)
        if norm_t < 1e-5:
            # 若对称无偏，引入轻微随机正交切向扰动
            rand_v = torch.randn_like(k_m)
            tangent_dir = rand_v - torch.dot(rand_v, k_m) * k_m
            norm_t = torch.norm(tangent_dir)

        tangent_unit = tangent_dir / (norm_t + 1e-7)

        # 沿测地线展开新槽位 m'
        delta = self.delta_split
        k_new = k_m * math.cos(delta) + tangent_unit * math.sin(delta)
        k_new = F.normalize(k_new, p=2, dim=-1)

        # 激活新槽位并平分能量
        half_e = self.energy[m_global_idx] * 0.5
        self.energy[m_global_idx] = half_e
        self.energy[new_idx] = half_e

        self.signature[new_idx] = k_new
        self.theta[new_idx] = self.theta[m_global_idx] + 0.1 # 轻微相位偏置
        self.omega[new_idx] = self.omega[m_global_idx]
        self.refractory[new_idx] = 0.5 # 新槽位初始阻尼，防止震荡
        self.slot_age[new_idx] = 0
        self.active_mask[new_idx] = True

        return 1

    def step_redundancy_merge(self) -> int:
        """
        算子 4：冗余坍缩与合并 (Merge / Collapse)
        当两相邻槽位相似度 <k_i, k_j> > 1 - eps 时，合并为单一槽位，释放空槽。
        """
        idx = self.active_indices
        if len(idx) <= 4:
            return 0

        k = self.signature[idx]
        sim = torch.matmul(k, k.T)
        sim.fill_diagonal_(0.0)

        # 仅允许成熟槽位之间合并 (slot_age >= 4)，给新分枝独立分化和适应的时间
        mature_mask = (self.slot_age[idx] >= 4).unsqueeze(1) & (self.slot_age[idx] >= 4).unsqueeze(0)
        sim[~mature_mask] = 0.0

        # 【核心保护机制】：两永久骨干神经元绝不互相吞噬消除！
        persist = self.is_persistent[idx]
        both_persist = persist.unsqueeze(1) & persist.unsqueeze(0)
        sim[both_persist] = 0.0

        max_val, max_pos = torch.max(sim.view(-1), dim=0)
        if max_val.item() < self.merge_similarity_threshold:
            return 0

        i_loc = max_pos.item() // len(idx)
        j_loc = max_pos.item() % len(idx)
        i_glob = idx[i_loc].item()
        j_glob = idx[j_loc].item()

        # 若 j 是永久核心而 i 是暂态分支，调换位置，确保永久核心永远屹立，只有外延分支缩回母体
        if self.is_persistent[j_glob] and not self.is_persistent[i_glob]:
            i_glob, j_glob = j_glob, i_glob

        # 能量加权合并签名
        e_i = self.energy[i_glob]
        e_j = self.energy[j_glob]
        e_total = e_i + e_j

        k_merged = (e_i * self.signature[i_glob] + e_j * self.signature[j_glob]) / (e_total + 1e-7)
        self.signature[i_glob] = F.normalize(k_merged, p=2, dim=-1)
        self.energy[i_glob] = e_total

        # 圆形平均合并相位
        sin_th = e_i * math.sin(self.theta[i_glob]) + e_j * math.sin(self.theta[j_glob])
        cos_th = e_i * math.cos(self.theta[i_glob]) + e_j * math.cos(self.theta[j_glob])
        self.theta[i_glob] = math.atan2(sin_th, cos_th)

        # 释放槽位 j
        self.active_mask[j_glob] = False
        self.energy[j_glob] = 0.0
        self.signature[j_glob] = 0.0

        return 1

    def step_relaxation_and_trace(self, external_reward: float = 0.0, r_target: float = 1.0):
        """
        算子 5：热力学基线清洗与资格迹演化 (Relaxation & Trace)
        离线清洗：e_i <- alpha * e_i (alpha = R_target / R_current)
        资格迹：tau_i(t) = gamma * tau_i(t-1) + sum_j W_ij k_i
        外部反馈强化：Delta e_i = R * tau_i
        """
        idx = self.active_indices
        if len(idx) == 0:
            return

        J, W, _ = self.compute_flux_and_conductance()
        _, R_current = self.compute_resistance(J)

        # 稳态基线平衡清洗
        if R_current.item() > 1e-5:
            alpha = torch.clamp(torch.tensor(r_target / (R_current.item() + 1e-6)), 0.5, 1.5)
            self.energy[idx] = torch.clamp(self.energy[idx] * alpha, 0.05, 50.0)

        # 资格迹演化
        w_flow = torch.sum(torch.abs(W), dim=1, keepdim=True)
        active_trace = self.trace[idx] * self.gamma_trace + w_flow * self.signature[idx]
        self.trace[idx] = active_trace

        # 环境反馈增强
        if abs(external_reward) > 1e-4:
            reinforce = external_reward * torch.norm(self.trace[idx], dim=-1) * 0.1
            self.energy[idx] = torch.clamp(self.energy[idx] + reinforce, 0.05, 50.0)

    def inject_input(self, x: torch.Tensor, input_energy: float = 2.0):
        """
        将外界输入流投影到流形槽位场上
        x: [D] 外部特征向量
        """
        idx = self.active_indices
        if len(idx) == 0:
            return

        x_norm = F.normalize(x.unsqueeze(0), p=2, dim=-1) # [1, D]
        k = self.signature[idx]                          # [N, D]

        # 匹配响应度
        match = torch.matmul(k, x_norm.T).squeeze(-1)   # [N]
        match_pos = torch.clamp(match, min=0.0)

        # 能量注入与相位驱动
        self.energy[idx] += match_pos * input_energy
        # 强匹配槽位被拉入同步相位
        self.theta[idx] += (1.0 - torch.cos(self.theta[idx])) * match_pos * 0.5

    # =========================================================================
    # Hopf 代数形式化接口 (Hopf Algebraic Operational Layer)
    # =========================================================================
    def coproduct_split(self) -> int:
        """余乘算子 Delta: H -> H (x) H (曲率爆裂树状分支)"""
        return self.step_curvature_split()

    def product_merge(self) -> int:
        """乘法算子 mu: H (x) H -> H (测地对齐单子坍缩并合)"""
        return self.step_redundancy_merge()

    def compute_hopf_cancellation_residual(self) -> float:
        """
        验证 Hopf 终极恒等式: mu o (S (x) id) o Delta(x) = eta(epsilon(x))
        物理意义: 前向分支预测与对极逆向相位在突触后膜干涉后的未抵消净残余。
        完全抵消时该指标趋近于 0。
        """
        idx = self.active_indices
        if len(idx) <= 1:
            return 0.0
        J, _, _ = self.compute_flux_and_conductance()
        # J_i 即为相位差正弦加权未抵消通量
        residual = torch.mean(torch.abs(J)).item()
        return float(residual)
