"""
构形场引擎 (CFE) 无限流自发演化与 3D Hopf 拓扑形态发生仿真
- 引入 Hopf 代数形式化算子 (Delta, mu, S, eta, epsilon)
- 引入 3D 物理空间张力松弛场 (库仑自排斥 Dscam + 弹性导通引力 + 椭球体空间有界势能)
- 彻底解决定向各向异性飞升与纵向失真，形成紧凑美观的 3D 脑状构形树
"""

import sys, os, math
import json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))
from fine_grain.information_boltzmann.cfe import ConstructalFieldEngine

def run_cfe_stream_evolution(total_steps=180, seed=42):
    torch.manual_seed(seed)
    dim = 16
    initial_slots = 6
    max_slots = 72

    engine = ConstructalFieldEngine(
        dim=dim,
        initial_slots=initial_slots,
        max_slots=max_slots,
        lambda_metabolic=0.025,
        split_threshold_sigma=1.0,
        merge_similarity_threshold=0.94,
        delta_split=0.25,
        dt=0.06
    )

    # 3D 空间物理坐标 x_i in R^3
    pos_3d = torch.zeros(max_slots, 3)
    # 初始 6 个种子槽位在双侧对称水平椭球核内排布 (水平 X-Z 平面为底，背腹 Y 为高)
    for i in range(initial_slots):
        angle = (2 * math.pi * i) / initial_slots
        # x: 前后, y: 背腹 (控制在 [-0.2, 0.2]), z: 左右
        pos_3d[i] = torch.tensor([0.35 * math.cos(angle), 0.05 * math.sin(angle * 2), 0.35 * math.sin(angle)])

    history_snapshots = []
    event_logs = []

    print(f"Starting CFE Hopf stream evolution with 3D bounded manifold (Steps: {total_steps})...")

    # 采用平衡对称的输入投影矩阵 (避免前几个维度有方向偏置)
    proj_in = torch.randn(3, dim)
    # 将能量均分到各主轴
    proj_in = F.normalize(proj_in, p=2, dim=-1)

    for step in range(total_steps):
        # 1. 构造多阶段非线性流
        if step < 50:
            stage_name = "阶段 1: 谐波线性加法流 (Linear Harmonic)"
            val_a = math.sin(step * 0.12)
            val_b = math.cos(step * 0.12)
            val_c = val_a + val_b
        elif step < 110:
            stage_name = "阶段 2: 非线性乘积冲击 (Multiplicative Shock)"
            val_a = math.sin(step * 0.16)
            val_b = math.cos(step * 0.16)
            val_c = val_a * val_b * 2.2 # 非线性扰动
        else:
            stage_name = "阶段 3: 多模态混沌流与松弛合并 (Chaotic Relaxation)"
            val_a = math.sin(step * 0.20) * math.cos(step * 0.08)
            val_b = math.cos(step * 0.16) + math.sin(step * 0.05)
            val_c = (val_a ** 2 - val_b ** 2) * 1.4

        raw_input = torch.tensor([val_a, val_b, val_c], dtype=torch.float32)
        x_vec = F.normalize(torch.matmul(raw_input, proj_in), p=2, dim=-1)

        # 2. 注入信号并执行动力学
        engine.inject_input(x_vec, input_energy=2.0)
        engine.step_flow_and_phase()
        engine.step_manifold_normalization()

        # 3. Hopf 余乘分裂 Delta 与 乘法并合 mu
        pre_active = engine.active_mask.clone()
        splits = engine.coproduct_split()

        # 新萌发槽位的 3D 各向同性分支寻路 (向局部低密度区域立体发散)
        if splits > 0:
            newly_active = engine.active_mask & (~pre_active)
            new_idx = torch.nonzero(newly_active).squeeze(-1)[0].item()

            k_new = engine.signature[new_idx]
            sims = torch.matmul(engine.signature, k_new)
            sims[~pre_active] = -1.0
            parent_idx = torch.argmax(sims).item()
            parent_p = pos_3d[parent_idx]

            # 探索周围 12 个等距球面方向，寻找周围已有槽位排斥力最小的方向生长
            active_p = pos_3d[pre_active]
            cand_dirs = F.normalize(torch.randn(16, 3), p=2, dim=-1)
            # 计算各备选方向与现有神经元的最小距离
            best_dir = cand_dirs[0]
            max_dist_to_neighbors = -1.0
            for d in cand_dirs:
                trial_pt = parent_p + d * 0.3
                min_d = torch.min(torch.norm(active_p - trial_pt, dim=-1)).item()
                if min_d > max_dist_to_neighbors:
                    max_dist_to_neighbors = min_d
                    best_dir = d

            pos_3d[new_idx] = parent_p + best_dir * 0.28
            event_msg = f"Step {step}: [Hopf 余乘 Δ] 阻力超标! 槽位 #{parent_idx} 裂变新分支 #{new_idx}"
            event_logs.append(event_msg)

        # Hopf 乘法并合 mu
        merges = engine.product_merge()
        if merges > 0:
            event_logs.append(f"Step {step}: [Hopf 乘法 μ] 相似度超限，两槽位单子坍缩并合")

        engine.step_relaxation_and_trace(r_target=2.5)

        # 4. 3D 物理空间张力场松弛 (Force-directed relaxation with biological envelope)
        # 解决各向异性异常拔高的核心物理机制！
        act_idx = engine.active_indices
        act_p = pos_3d[act_idx]
        N_act = len(act_idx)

        if N_act > 1:
            J, W, _ = engine.compute_flux_and_conductance()

            # A. 槽位间库仑自排斥 (Dscam 分子排斥，防止挤压聚拢)
            diff = act_p.unsqueeze(1) - act_p.unsqueeze(0) # [N, N, 3]
            dist_sq = torch.sum(diff ** 2, dim=-1, keepdim=True) + 0.04
            dist = torch.sqrt(dist_sq)
            repel_force = torch.sum(diff / (dist_sq * dist), dim=1) * 0.012

            # B. 导通通道弹性引力 (高导通 W_ij 维持功能相干微柱)
            W_pos = torch.clamp(W, min=0.0).unsqueeze(-1)
            attract_force = - torch.sum(W_pos * diff, dim=1) * 0.02

            # C. 脑包络椭球体向心阻尼 (Ellipsoid Harmonic Potential: 限制在 X:0.85, Y:0.55, Z:0.75 紧凑颅骨空间)
            # 严格惩罚 Y 轴（背腹高度）超标！
            confinement_force = - torch.stack([
                act_p[:, 0] / 0.85,
                act_p[:, 1] / 0.50, # 强力限制纵向拔高！
                act_p[:, 2] / 0.75
            ], dim=-1) * 0.045

            total_force = repel_force + attract_force + confinement_force
            new_p = act_p + torch.clamp(total_force, -0.06, 0.06)

            # 严格质心归零 (Center of Mass Centering) 防止定向漂移
            new_p = new_p - torch.mean(new_p, dim=0, keepdim=True)

            # 严格包络限制 (限制在半长轴 0.9 的紧凑脑流形内)
            max_r = torch.max(torch.norm(new_p, dim=-1)).item()
            if max_r > 0.85:
                new_p = new_p * (0.85 / max_r)

            pos_3d[act_idx] = new_p

        # 5. 统计 Hopf 终极恒等式残差与相干物理量
        hopf_residual = engine.compute_hopf_cancellation_residual()
        _, R = engine.compute_resistance(J)
        th_active = engine.theta[act_idx]
        order_param = torch.abs(torch.mean(torch.exp(1j * th_active))).item()

        # 记录关键帧
        active_nodes = []
        for idx_val in act_idx.tolist():
            p = pos_3d[idx_val]
            e = float(engine.energy[idx_val].item())
            th = float(engine.theta[idx_val].item())
            r = float(engine.refractory[idx_val].item())
            hue = int(((th + math.pi) / (2 * math.pi)) * 360)

            is_p = bool(engine.is_persistent[idx_val].item())
            active_nodes.append({
                'id': idx_val,
                'x': float(p[0]),
                'y': float(p[1]),
                'z': float(p[2]),
                'energy': e,
                'theta': th,
                'refractory': r,
                'hue': hue,
                'is_persistent': is_p
            })

        active_edges = []
        for i_loc in range(N_act):
            for j_loc in range(i_loc + 1, N_act):
                w_val = float(W[i_loc, j_loc].item())
                if abs(w_val) > 0.04:
                    active_edges.append({
                        'source': act_idx[i_loc].item(),
                        'target': act_idx[j_loc].item(),
                        'weight': w_val
                    })

        num_core = int(engine.is_persistent[act_idx].sum().item())
        num_spines = N_act - num_core

        history_snapshots.append({
            'step': step,
            'stage': stage_name,
            'input_vals': [float(val_a), float(val_b), float(val_c)],
            'num_slots': N_act,
            'num_core': num_core,
            'num_spines': num_spines,
            'resistance': float(R.item()),
            'hopf_residual': float(hopf_residual),
            'order_param': float(order_param),
            'recent_event': event_logs[-1] if event_logs else "Hopf 真空稳态基底",
            'nodes': active_nodes,
            'edges': active_edges
        })

        if step % 30 == 0 or step == total_steps - 1:
            y_max = max([n['y'] for n in active_nodes])
            y_min = min([n['y'] for n in active_nodes])
            print(f"Step {step:3d} | Slots: {N_act:2d} | R: {R.item():5.2f} | HopfResid: {hopf_residual:.4f} | Y-span: [{y_min:+.2f}, {y_max:+.2f}] | {stage_name.split(':')[1]}")

    result = {
        'total_steps': total_steps,
        'snapshots': history_snapshots,
        'summary': {
            'initial_slots': initial_slots,
            'final_slots': engine.num_active,
            'final_hopf_residual': history_snapshots[-1]['hopf_residual'],
            'final_order_param': history_snapshots[-1]['order_param']
        }
    }

    return result

if __name__ == '__main__':
    data = run_cfe_stream_evolution()
    out_path = 'C:/Users/85750/.gemini/antigravity/brain/4d6d1761-7e96-4279-930e-c51f5f2b2b1b/scratch/cfe_stream_evolution_data.json'
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Hopf CFE simulation completed! Saved to {out_path}")
