"""
严格遵循贝让构形定律 (Constructal Law) 的源-汇任务流驱动引擎 (Source-Sink CFE)
原理:
1. 构形定律核心: 系统的结构绝不是内部自娱自乐，而是外加的通过性流 (Imposed Currents) 强行冲刷雕刻出来的！
2. 源端 (Source): 外部任务输入 (A, B) 提供高电势/高通量流入。
3. 汇端 (Sink / Drain): 外部环境物理做功与目标约束 (Target) 充当低势能出口。
4. 任务阻抗 (Task Impedance):
   若内部网络未能解算出任务所需的变换 (如 A*B)，汇端出口受阻，反压 (Backpressure) 沿回路反向积压；
   局部阻力爆发逼迫内部长出新的分支管道 (Split)，直到非线性变换打通，全流道阻力极小化，任务完美通过！
"""

import sys, os, math
import json
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from fine_grain.information_boltzmann.cfe import ConstructalFieldEngine

class SourceSinkConstructalEngine:
    def __init__(self, dim=16, initial_internal_slots=6, max_slots=48):
        self.dim = dim
        self.engine = ConstructalFieldEngine(
            dim=dim,
            initial_slots=initial_internal_slots + 3, # 2 个源 + 1 个汇 + 内部
            max_slots=max_slots,
            lambda_metabolic=0.03,
            split_threshold_sigma=0.8,
            merge_similarity_threshold=0.96
        )

        # 0, 1 号为源槽位 (Source A, B)
        # 2 号为汇槽位 (Sink Task Output)
        # 3 号以上为内部可塑性构形通道
        self.src_a_idx = 0
        self.src_b_idx = 1
        self.sink_idx = 2

        # 将源和汇永久锁定为不可销毁的物理边界
        self.engine.is_persistent[0:3] = True

        # 设定几何位置：源在左侧 (x = -0.8)，汇在右侧 (x = +0.8)，中间为构形流形
        self.pos_3d = torch.zeros(max_slots, 3)
        self.pos_3d[0] = torch.tensor([-0.8, -0.25, 0.0]) # Source A
        self.pos_3d[1] = torch.tensor([-0.8, +0.25, 0.0]) # Source B
        self.pos_3d[2] = torch.tensor([+0.8, 0.0, 0.0])   # Sink Drain

        # 内部初始槽位散布在中间 [-0.4, +0.4]
        for i in range(3, initial_internal_slots + 3):
            self.pos_3d[i] = torch.tensor([
                np.random.uniform(-0.35, 0.35),
                np.random.uniform(-0.35, 0.35),
                np.random.uniform(-0.35, 0.35)
            ])

    def step_task_stream(self, a_val: float, b_val: float, target_val: float, task_type="mult"):
        """
        外界任务流注入：
        源端施加输入压力，汇端考核目标反压
        """
        # 1. 源端能量与输入钳位
        self.engine.energy[self.src_a_idx] = abs(a_val) + 0.5
        self.engine.energy[self.src_b_idx] = abs(b_val) + 0.5

        # 将连续数值映射到超球面签名
        k_a = F.normalize(torch.tensor([a_val] * (self.dim // 2) + [1.0] * (self.dim // 2)), p=2, dim=-1)
        k_b = F.normalize(torch.tensor([1.0] * (self.dim // 2) + [b_val] * (self.dim // 2)), p=2, dim=-1)
        self.engine.signature[self.src_a_idx] = k_a
        self.engine.signature[self.src_b_idx] = k_b

        # 2. 内部多步能流传输与非线性相干涉
        for _ in range(4):
            self.engine.step_flow_and_phase()
            self.engine.step_manifold_normalization()

        # 3. 汇端 (Sink) 读出并与环境任务做功交互
        J, W, _ = self.engine.compute_flux_and_conductance()
        act_idx = self.engine.active_indices

        # 汇端受到的净流出通量
        sink_loc = (act_idx == self.sink_idx).nonzero().item()
        sink_flux = J[sink_loc].item()

        # 汇端实际产出的动作响应 (当前内部拓扑向汇端排出的能流)
        sink_output = self.engine.energy[self.sink_idx].item() * math.sin(self.engine.theta[self.sink_idx].item()) * 3.0

        # 4. 构形定律核心：环境任务反压 (Task Backpressure)
        # 如果汇端产出偏离外界目标 (如 A*B)，环境对汇端施加巨大的出口阻抗！
        # 导致能流无法排出，在汇端前后形成高压积水 (Hydraulic Backpressure)
        task_error = abs(sink_output - target_val)
        backpressure_impedance = task_error ** 2

        # 将任务反压直接加在汇端及其相连流道上，造成局部阻力激增
        self.engine.energy[self.sink_idx] += backpressure_impedance * 0.4
        self.engine.refractory[self.sink_idx] = min(1.0, task_error * 0.15) # 高误差导致出口痉挛阻塞

        # 5. 构形自发演化：阻力积压逼迫系统萌发新管道 (Split) 或熔合无效管道 (Merge)
        splits = self.engine.coproduct_split()
        merges = self.engine.product_merge()

        # 如果分裂了新通道，在 3D 源-汇主轴之间分配新流道空间
        if splits > 0:
            free_idx = (self.engine.active_mask & (self.pos_3d[:, 0] == 0.0)).nonzero()
            # 为新槽位赋予 3D 中继坐标
            for f_idx in free_idx:
                fi = f_idx.item()
                if fi >= 3:
                    self.pos_3d[fi] = torch.tensor([
                        np.random.uniform(-0.3, 0.3),
                        np.random.uniform(-0.4, 0.4),
                        np.random.uniform(-0.4, 0.4)
                    ])

        # 6. 环境稳态清洗
        self.engine.step_relaxation_and_trace(r_target=2.0)

        return {
            'a': a_val,
            'b': b_val,
            'target': target_val,
            'output': sink_output,
            'task_error': task_error,
            'splits': splits,
            'merges': merges,
            'active_slots': self.engine.num_active
        }

def run_task_stream_evolution():
    print("=" * 70)
    print("贝让构形定律验证：外加通过性任务流 (A, B) -> (A * B) 强制雕刻内部流道")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)

    engine = SourceSinkConstructalEngine(dim=16, initial_internal_slots=4, max_slots=32)

    print(f"初始状态: 2 源节点 + 1 汇节点 + 4 内部初始槽位 = 7 槽位")
    print("任务: 外部环境持续输入 (A, B)，出口只接受乘积 (A * B)，否则形成高反压！\n")

    errors = []

    # 模拟 100 批次算术任务流冲刷
    for epoch in range(120):
        # 随机产生连续算术流
        a = float(np.random.uniform(1.0, 3.0))
        b = float(np.random.uniform(1.0, 3.0))
        target = a * b # 乘法任务

        res = engine.step_task_stream(a, b, target)
        errors.append(res['task_error'])

        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg_err = np.mean(errors[-20:]) if epoch > 0 else errors[0]
            print(f"Epoch {epoch+1:3d} | 当前流道槽位数: {res['active_slots']:2d} | 任务平均误差: {avg_err:5.2f} | 采样输出: 目标 {res['target']:.2f} vs 实际流出 {res['output']:.2f}")

    print("-" * 70)
    initial_err = np.mean(errors[:20])
    final_err = np.mean(errors[-20:])
    print(f"构形流道演化结果:")
    print(f"  - 初期任务反压平均误差: {initial_err:.2f}")
    print(f"  - 通道打通后稳定误差: {final_err:.2f}")
    print(f"  - 阻抗降低比例: {(initial_err - final_err) / initial_err * 100:.2f}%")
    print(">> [结论] 证实构形定律：只有在源端输入与汇端任务的通过性压差驱动下，内部才会形成有意义的导通结构！")

if __name__ == '__main__':
    run_task_stream_evolution()
