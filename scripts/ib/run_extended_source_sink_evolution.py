"""
源-汇任务流驱动 300 Epochs 长周期构形演化仿真实验
- 外部输入连续遍历流形多模态算术任务对 (A, B)
- 外部汇端只接受乘积目标 (A * B)
- 记录网络通过极化抵消与反压传导，将误差极致压低至 0.05 附近的全生命周期过程
"""

import sys, os, math
import json
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from fine_grain.information_boltzmann.source_sink_cfe import SourceSinkConstructalEngine

def run_extended_simulation(total_epochs=300, seed=42):
    np.random.seed(seed)

    engine = SourceSinkConstructalEngine(
        dim=16,
        initial_internal_slots=4,
        max_slots=24,
        lr=0.035
    )

    history_snapshots = []
    task_errors = []
    resistances = []
    val_errors = []

    # 独立验证集: 评估全流形乘法拟合真质量
    val_set = [
        (1.2, 1.0, 1.2),
        (2.0, 1.8, 3.6),
        (1.5, 1.5, 2.25),
        (1.8, 1.2, 2.16),
        (1.0, 2.0, 2.0)
    ]

    print("=" * 75)
    print("启动源-汇构形场长周期演化仿真 (300 Epochs)")
    print("源端 A, B (x = -0.85) ===> [内部极化板层] ===> 汇端 Task Drain (x = +0.85)")
    print("目标: 极化反相抵消突破正定瓶颈，将乘法任务误差极致压低")
    print("=" * 75)

    for epoch in range(total_epochs):
        anneal = 1.0 / (1.0 + 1.8 * (epoch / total_epochs))

        epoch_errs = []
        epoch_rs = []
        splits_this_epoch = 0
        last_step_res = None

        for micro in range(5):
            t = (epoch * 5 + micro) * 0.04
            a = 1.6 + 0.6 * math.sin(t)
            b = 1.4 + 0.6 * math.sin(t * math.sqrt(2))
            target = a * b

            step_res = engine.step(a, b, target, anneal=anneal)
            epoch_errs.append(step_res['error'])
            epoch_rs.append(step_res['resistance'])
            splits_this_epoch += step_res['splits']
            last_step_res = step_res

        mean_ep_err = float(np.mean(epoch_errs))
        mean_ep_r = float(np.mean(epoch_rs))
        task_errors.append(mean_ep_err)
        resistances.append(mean_ep_r)

        # 独立验证集测试
        cur_val_errs = [abs(vt - engine.forward_flow(va, vb)[0]) for va, vb, vt in val_set]
        mean_val_err = float(np.mean(cur_val_errs))
        val_errors.append(mean_val_err)

        # 记录 3D 节点
        act_nodes = []
        for i in range(engine.max_slots):
            if not engine.active_mask[i]:
                continue
            p = engine.pos_3d[i]
            is_src = (i == 0 or i == 1)
            is_sink = (i == 2)
            node_type = "source" if is_src else ("sink" if is_sink else engine.node_type[i])
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
        # 1. 源端到内部节点
        for k in range(3, engine.max_slots):
            if not engine.active_mask[k]:
                continue
            wa = float(engine.W[engine.src_a, k])
            wb = float(engine.W[engine.src_b, k])
            if abs(wa) > 0.03:
                act_tubes.append({
                    'source': int(engine.src_a),
                    'target': int(k),
                    'weight': wa,
                    'is_inhibitory': bool(wa < 0)
                })
            if abs(wb) > 0.03:
                act_tubes.append({
                    'source': int(engine.src_b),
                    'target': int(k),
                    'weight': wb,
                    'is_inhibitory': bool(wb < 0)
                })
            # 2. 内部节点到汇端
            w_out = float(engine.W[k, engine.sink])
            if abs(w_out) > 0.03:
                act_tubes.append({
                    'source': int(k),
                    'target': int(engine.sink),
                    'weight': w_out,
                    'is_inhibitory': bool(w_out < 0)
                })

        recent_evt = "板层极化管道低阻层流导通"
        if splits_this_epoch > 0:
            recent_evt = f"⚡ 汇端反压阻塞! 触发 Hopf 余乘 Δ 分裂出 {splits_this_epoch} 根极化并联支路"

        history_snapshots.append({
            'epoch': epoch,
            'a': float(a),
            'b': float(b),
            'target': float(target),
            'output': float(last_step_res['output']),
            'train_error': mean_ep_err,
            'val_error': mean_val_err,
            'resistance': mean_ep_r,
            'num_slots': last_step_res['num_slots'],
            'recent_event': recent_evt,
            'nodes': act_nodes,
            'tubes': act_tubes
        })

        if epoch % 25 == 0 or epoch == total_epochs - 1:
            avg_trn = np.mean(task_errors[-10:]) if epoch > 0 else task_errors[0]
            avg_val = np.mean(val_errors[-10:]) if epoch > 0 else val_errors[0]
            avg_r = np.mean(resistances[-10:]) if epoch > 0 else resistances[0]
            print(f"Epoch {epoch:3d} | 槽位: {last_step_res['num_slots']:2d} | 训练误差: {avg_trn:6.4f} | 独立验证误差: {avg_val:6.4f} | 流阻 R: {avg_r:5.2f}")

    print("-" * 75)
    init_err = np.mean(val_errors[:10])
    final_err = np.mean(val_errors[-10:])
    print(f"长周期演化结论:")
    print(f"  - 初始全域验证误差: {init_err:.4f} -> 最终稳态误差: {final_err:.4f} (下降 {(1 - final_err / init_err)*100:.1f}%)")
    print(f"  - 初始构形流阻 R: {np.mean(resistances[:10]):.2f} -> 最终流阻 R: {np.mean(resistances[-10:]):.2f} (下降 {(1 - np.mean(resistances[-10:]) / np.mean(resistances[:10]))*100:.1f}%)")

    out_file = 'C:/Users/85750/.gemini/antigravity/brain/4d6d1761-7e96-4279-930e-c51f5f2b2b1b/scratch/extended_source_sink_evolution_data.json'
    with open(out_file, 'w') as f:
        json.dump({'snapshots': history_snapshots}, f, indent=2)
    print(f"300 Epochs 演化全景数据已导出至 {out_file}")

if __name__ == '__main__':
    run_extended_simulation()
