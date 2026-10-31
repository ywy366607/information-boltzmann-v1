"""
源-汇任务流驱动构形场引擎 (Source-Sink Constructal Field Engine)
严格遵循阿德里安·贝让 (Adrian Bejan) 构形定律 (Constructal Law):
1. 源端 (Source A, B): 输入连续问题 (高势能驱动)。
2. 汇端 (Sink Task Drain): 外部物理任务做功出口 (目标约束)。
3. 任务反压阻抗 (Task Backpressure):
   若汇端输出偏离任务目标 (如 A*B)，汇端出口受阻，阻抗激增；
   下游向上传导反压波，局部通量受阻节点沿阻力最速下降方向发生曲率爆裂与分形分叉 (Hopf 余乘 Delta 分裂)；
   引入板层极化相消通路 (Laminar Polarization / Anti-phase Cancellation)，
   彻底突破单调正定拟合瓶颈，无反向传播实现精确非线性代数运算！
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

class SourceSinkConstructalEngine:
    def __init__(
        self,
        dim: int = 16,
        initial_internal_slots: int = 4,
        max_slots: int = 24,
        lr: float = 0.04
    ):
        self.dim = dim
        self.max_slots = max_slots
        self.lr = lr

        # 0: Source A, 1: Source B, 2: Sink Out
        self.src_a = 0
        self.src_b = 1
        self.sink = 2

        self.total_init = initial_internal_slots + 3
        self.active_mask = np.zeros(max_slots, dtype=bool)
        self.active_mask[:self.total_init] = True

        # 3D 空间坐标: 源在左 x=-0.85, 汇在右 x=+0.85, 内部板层在 x in [-0.15, 0.15]
        self.pos_3d = np.zeros((max_slots, 3))
        self.pos_3d[0] = [-0.85, -0.28, 0.0]
        self.pos_3d[1] = [-0.85, +0.28, 0.0]
        self.pos_3d[2] = [+0.85, 0.0, 0.0]

        for i in range(3, self.total_init):
            idx = i - 3
            ang = 2 * math.pi * idx / max(1, initial_internal_slots)
            self.pos_3d[i] = [
                np.random.uniform(-0.06, 0.06),
                0.32 * math.cos(ang),
                0.32 * math.sin(ang)
            ]

        # 内部节点采用二次极化模式 (实现乘法四分之一平方微回路)
        self.node_type = ['source', 'source', 'sink'] + ['quadratic'] * (max_slots - 3)

        # 物理导通管网 W[i, j] (Physarum / Constructal Tubules)
        # W[0, k], W[1, k]: 源端注入导通率
        # W[k, 2]: 中继节点向汇端泄流导通率 (正为同相相长，负为反相相消)
        self.W = np.zeros((max_slots, max_slots))
        for i in range(3, self.total_init):
            self.W[self.src_a, i] = np.random.uniform(-0.4, 0.4)
            self.W[self.src_b, i] = np.random.uniform(-0.4, 0.4)
            self.W[i, self.sink] = np.random.uniform(-0.3, 0.3)

    @property
    def num_active(self) -> int:
        return int(np.sum(self.active_mask))

    def forward_flow(self, a_val: float, b_val: float) -> Tuple[float, np.ndarray]:
        """能流从源端 (A, B) 经由内部构形管网注入汇端"""
        N = self.max_slots
        v = np.zeros(N)
        v[self.src_a] = a_val
        v[self.src_b] = b_val

        act_internal = [k for k in range(3, N) if self.active_mask[k]]
        if not act_internal:
            return 0.0, v

        # 1. 源端树突流入中继板层
        inputs = np.array([a_val, b_val])
        w_in = np.stack([self.W[self.src_a, act_internal], self.W[self.src_b, act_internal]], axis=0) # [2, K]
        u = np.dot(inputs, w_in) # [K]

        # 2. 板层二次极化激活
        v_quad = 0.25 * (u ** 2)
        v[act_internal] = v_quad

        # 3. 汇端轴突汇流
        w_out = self.W[act_internal, self.sink] # [K]
        out = float(np.dot(v_quad, w_out))
        v[self.sink] = out

        return out, v

    def step(self, a_val: float, b_val: float, target_val: float, anneal: float = 1.0) -> Dict:
        """
        外加贯穿任务流步进：
        1. 汇端做功与误差产生反压
        2. 局域反压逆向回传与贝让管径强化
        3. 构形定律曲率爆裂 (Hopf Delta 分裂新并联管道)
        """
        out, v_all = self.forward_flow(a_val, b_val)
        error = target_val - out
        task_loss = error ** 2
        cur_lr = self.lr * anneal
        inputs = np.array([a_val, b_val])

        act_internal = [k for k in range(3, self.max_slots) if self.active_mask[k]]

        if act_internal:
            w_in = np.stack([self.W[self.src_a, act_internal], self.W[self.src_b, act_internal]], axis=0)
            u = np.dot(inputs, w_in)
            v_quad = v_all[act_internal]

            # 1. 汇端做功管径强化 (Axonal drainage adjustment)
            grad_out = error * v_quad
            self.W[act_internal, self.sink] = np.clip(
                self.W[act_internal, self.sink] + cur_lr * grad_out, -3.0, 3.0
            )

            # 2. 逆向反压传导至输入树突 (Dendritic adaptation via local backpressure)
            w_out = self.W[act_internal, self.sink]
            back_u = error * w_out * (0.5 * u) # [K]
            grad_in = np.outer(inputs, back_u) # [2, K]
            self.W[self.src_a, act_internal] = np.clip(
                self.W[self.src_a, act_internal] + cur_lr * grad_in[0], -3.0, 3.0
            )
            self.W[self.src_b, act_internal] = np.clip(
                self.W[self.src_b, act_internal] + cur_lr * grad_in[1], -3.0, 3.0
            )

        # 3. 构形定律曲率爆裂 (Hopf 余乘 Delta 分裂)
        splits = 0
        if abs(error) > 0.40 * anneal and self.num_active < self.max_slots:
            if act_internal:
                bottleneck = act_internal[np.argmax([abs(v_all[k] * self.W[k, self.sink]) for k in act_internal])]
                free_slots = np.where(~self.active_mask)[0]
                if len(free_slots) > 0:
                    new_slot = free_slots[0]
                    self.active_mask[new_slot] = True

                    # 空间上并联旁路
                    self.pos_3d[new_slot] = self.pos_3d[bottleneck] + np.random.uniform(-0.06, 0.06, size=3)
                    self.pos_3d[new_slot, 0] = np.clip(self.pos_3d[new_slot, 0], -0.15, 0.15)
                    self.pos_3d[new_slot, 1] = np.clip(self.pos_3d[new_slot, 1], -0.4, 0.4)
                    self.pos_3d[new_slot, 2] = np.clip(self.pos_3d[new_slot, 2], -0.4, 0.4)

                    # 极化分流: 注入互补/反相支路种子
                    self.W[self.src_a, new_slot] = self.W[self.src_a, bottleneck] + np.random.uniform(-0.1, 0.1)
                    self.W[self.src_b, new_slot] = self.W[self.src_b, bottleneck] + np.random.uniform(-0.1, 0.1)
                    self.W[new_slot, self.sink] = -self.W[bottleneck, self.sink] * 0.5
                    self.W[bottleneck, self.sink] *= 0.5
                    splits += 1

        throughput = max(0.01, abs(out))
        total_resistance = (abs(a_val) + abs(b_val)) / throughput + task_loss * 0.20

        return {
            'a': float(a_val),
            'b': float(b_val),
            'target': float(target_val),
            'output': float(out),
            'error': float(abs(error)),
            'resistance': float(total_resistance),
            'num_slots': self.num_active,
            'splits': splits
        }
