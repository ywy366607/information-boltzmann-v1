"""
源-汇任务流驱动 3D 构形演化仿真实验
- 外部输入连续算术任务对 (A, B)
- 外部汇端只接受乘积目标 (A * B)
- 记录管道网络从高阻抗 (R > 100) 演化为贯穿低阻抗 (R ~ 1.5) 的全生命周期
"""

import sys, os, math
import json
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from fine_grain.information_boltzmann.source_sink_cfe import SourceSinkConstructalEngine

def run_source_sink_task_simulation(total_epochs=120, seed=42):
    np.random.seed(seed)

    engine = SourceSinkConstructalEngine(
        dim=16,
        initial_internal_slots=6,
        max_slots=36,
        lr=0.08
    )

    history_snapshots = []
    task_errors = []
    resistances = []

    print("=" * 70)
    print("启动源-汇贯穿流仿真：任务驱动的贝让构形进化")
    print("源端 A, B (x = -0.85) ===> [内部构形管道] ===> 汇端 Task Drain (x = +0.85)")
    print("任务目标: 外部强制注入连续乘法流 (A * B)")
    print("=" * 70)

    for epoch in range(total_epochs):
        t = epoch * 0.08
        a = 1.6 + 0.6 * math.sin(t)
        b = 1.4 + 0.6 * math.cos(t)
        target = a * b

        step_res = engine.step(a, b, target)
        task_errors.append(step_res['error'])
        resistances.append(step_res['resistance'])

        # 记录 3D 节点状态
        act_nodes = []
        for i in range(engine.max_slots):
            if not engine.active_mask[i]:
                continue
            p = engine.pos_3d[i]
            is_src = (i == 0 or i == 1)
            is_sink = (i == 2)
            node_type = "source" if is_src else ("sink" if is_sink else engine.node_type[i])

            # 计算节点当前流经活性
            act_nodes.append({
                'id': int(i),
                'type': node_type,
                'x': float(p[0]),
                'y': float(p[1]),
                'z': float(p[2]),
                'is_source': bool(is_src),
                'is_sink': bool(is_sink)
            })

        # 记录 3D 导通管道
        act_tubes = []
        for i in range(engine.max_slots):
            if not engine.active_mask[i]:
                continue
            for j in range(2, engine.max_slots):
                if not engine.active_mask[j]:
                    continue
                w_val = float(engine.W[i, j])
                if w_val > 0.05:
                    act_tubes.append({
                        'source': int(i),
                        'target': int(j),
                        'weight': w_val
                    })

        recent_evt = "构形管道平稳导通"
        if step_res['splits'] > 0:
            recent_evt = "⚡ 汇端高反压阻塞! 触发 Hopf 余乘 Δ，炸开新并联旁路"

        history_snapshots.append({
            'epoch': epoch,
            'a': float(a),
            'b': float(b),
            'target': float(target),
            'output': float(step_res['output']),
            'error': float(step_res['error']),
            'resistance': float(step_res['resistance']),
            'num_slots': step_res['num_slots'],
            'recent_event': recent_evt,
            'nodes': act_nodes,
            'tubes': act_tubes
        })

        if epoch % 20 == 0 or epoch == total_epochs - 1:
            avg_err = np.mean(task_errors[-15:]) if epoch > 0 else task_errors[0]
            avg_r = np.mean(resistances[-15:]) if epoch > 0 else resistances[0]
            print(f"Epoch {epoch:3d} | 活跃槽位: {step_res['num_slots']:2d} | 平均误差: {avg_err:5.3f} | 贯穿阻抗 R: {avg_r:6.2f} | 目标: {target:.2f} vs 实际: {step_res['output']:.2f}")

    print("-" * 70)
    print(f"构形流道演化结果:")
    print(f"  - 初始任务误差: {np.mean(task_errors[:15]):.3f} -> 最终稳定误差: {np.mean(task_errors[-15:]):.3f} (下降 {(1-np.mean(task_errors[-15:])/np.mean(task_errors[:15]))*100:.1f}%)")
    print(f"  - 初始贯穿阻抗: {np.mean(resistances[:15]):.2f} -> 最终构形阻抗: {np.mean(resistances[-15:]):.2f} (阻力剧降 {(1-np.mean(resistances[-15:])/np.mean(resistances[:15]))*100:.1f}%)")

    out_file = 'C:/Users/85750/.gemini/antigravity/brain/4d6d1761-7e96-4279-930e-c51f5f2b2b1b/scratch/source_sink_evolution_data.json'
    with open(out_file, 'w') as f:
        json.dump({'snapshots': history_snapshots}, f, indent=2)
    print(f"演化全景数据已导出至 {out_file}")

if __name__ == '__main__':
    run_source_sink_task_simulation()
