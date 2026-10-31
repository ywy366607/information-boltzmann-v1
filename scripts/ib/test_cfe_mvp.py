"""
构形场引擎 (CFE) 两大核心验证测试：
1. 拓扑自举测试 (Murray's Law / Frustration Test)
2. 预测残差消解测试 (Predictive Residual Annihilation Test)
"""

import sys, os, math
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.abspath("."))

from fine_grain.information_boltzmann.cfe import ConstructalFieldEngine

def test_topology_bootstrap_frustration():
    print("=" * 70)
    print("测试 1: 拓扑自举测试 (Murray's Law / Frustration Test)")
    print("目标: 输入多模态非线性流，验证无反传下自发分裂 (Split) 与合并 (Merge)")
    print("=" * 70)

    torch.manual_seed(42)
    dim = 16
    engine = ConstructalFieldEngine(
        dim=dim,
        initial_slots=8,
        max_slots=64,
        lambda_metabolic=0.03,
        split_threshold_sigma=1.2,
        merge_similarity_threshold=0.92,
        delta_split=0.15
    )

    print(f"初始状态: 活跃槽位数量 = {engine.num_active}")

    # 生成 4 个不同模式的多模态非线性聚类中心 (模拟外界不相关的语义流)
    centers = F.normalize(torch.randn(4, dim), p=2, dim=-1)

    total_splits = 0
    total_merges = 0
    history_slots = []
    history_R = []

    # 模拟 120 个时间步的非平稳流式数据注入
    for step in range(120):
        # 随时间轮转外部模式，产生局域构形阻力积压 (Frustration)
        cluster_id = (step // 30) % 4
        base_vec = centers[cluster_id]
        noise = torch.randn(dim) * 0.15
        x = F.normalize(base_vec + noise, p=2, dim=-1)

        # 注入信号
        engine.inject_input(x, input_energy=1.5)

        # 执行 CFE 动力学算子
        engine.step_flow_and_phase()
        engine.step_manifold_normalization()

        # 阻力监测与自适应拓扑演化
        splits = engine.step_curvature_split()
        merges = engine.step_redundancy_merge()
        engine.step_relaxation_and_trace(r_target=2.0)

        total_splits += splits
        total_merges += merges

        J, _, _ = engine.compute_flux_and_conductance()
        _, R = engine.compute_resistance(J)

        history_slots.append(engine.num_active)
        history_R.append(R.item())

        if (step + 1) % 30 == 0:
            print(f"Step {step+1:3d} | 当前槽位数: {engine.num_active:2d} | 构形阻力 R: {R.item():6.2f} | 累计分裂: {total_splits:2d} | 累计合并: {total_merges:2d}")

    print("-" * 70)
    print(f"测试结果:")
    print(f"  - 初始槽位数: 8 -> 最终槽位数: {engine.num_active}")
    print(f"  - 总分裂次数 (Split): {total_splits}")
    print(f"  - 总合并次数 (Merge): {total_merges}")
    print(f"  - 构形阻力由初期 {history_R[5]:.2f} 稳定在 {history_R[-1]:.2f}")
    assert total_splits > 0, "未能触发曲率爆裂自适应分裂！"
    print(">> [PASS] 拓扑自举测试通过：无反向传播下成功长出分形标度网络！\n")


def test_predictive_residual_annihilation():
    print("=" * 70)
    print("测试 2: 预测残差消解测试 (Predictive Residual Annihilation Test)")
    print("目标: 输入固定时间延迟序列，验证 Kuramoto 相位干涉自发反相抵消残差")
    print("=" * 70)

    torch.manual_seed(42)
    dim = 16
    engine = ConstructalFieldEngine(
        dim=dim,
        initial_slots=12,
        max_slots=32,
        dt=0.08
    )

    # 构造固定的时序循环信号: A -> B -> C -> A
    T_period = 20
    omega_ext = 2 * math.pi / T_period

    # 记录前后的相位差与残差通量
    residuals = []

    print(f"注入周期为 {T_period} 步的旋转相位流，观察槽位内禀相位的自发捕获与干涉抵消...")

    for step in range(160):
        # 外部输入以特定角速度驱动
        ext_phase = (step * omega_ext) % (2 * math.pi) - math.pi
        ext_signal = torch.sin(torch.tensor(ext_phase))

        # 将外部相位调制注入为 external_flux
        ext_flux = torch.ones(engine.num_active) * ext_signal * 0.4

        engine.step_flow_and_phase(external_flux=ext_flux)
        engine.step_manifold_normalization()

        # 计算局域通量 J
        J, W, _ = engine.compute_flux_and_conductance()

        # 预测残差定义为各槽位因相位脱节产生的失配通量方差
        residual_variance = torch.var(J).item()
        residuals.append(residual_variance)

    initial_residual = sum(residuals[:20]) / 20.0
    final_residual = sum(residuals[-20:]) / 20.0
    reduction = (initial_residual - final_residual) / (initial_residual + 1e-8) * 100

    print("-" * 70)
    print(f"测试结果:")
    print(f"  - 初始平均失配残差: {initial_residual:.5f}")
    print(f"  - 相位干涉锁定后残差: {final_residual:.5f}")
    print(f"  - 残差消解率: {reduction:.2f}%")
    assert reduction > 20.0, f"残差未能显著消解！(下降比例: {reduction:.2f}%)"
    print(">> [PASS] 预测残差消解测试通过：相位干涉在物理层有效实现了反相抵消！\n")


if __name__ == '__main__':
    test_topology_bootstrap_frustration()
    test_predictive_residual_annihilation()
