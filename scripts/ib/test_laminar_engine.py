import math
import numpy as np

class LaminarConstructalEngine:
    def __init__(
        self,
        initial_internal_slots: int = 4,
        max_slots: int = 24,
        lr: float = 0.04
    ):
        self.max_slots = max_slots
        self.lr = lr

        self.src_a = 0
        self.src_b = 1
        self.sink = 2

        self.total_init = initial_internal_slots + 3
        self.active_mask = np.zeros(max_slots, dtype=bool)
        self.active_mask[:self.total_init] = True

        self.pos_3d = np.zeros((max_slots, 3))
        self.pos_3d[0] = [-0.85, -0.28, 0.0]
        self.pos_3d[1] = [-0.85, +0.28, 0.0]
        self.pos_3d[2] = [+0.85, 0.0, 0.0]

        for i in range(3, self.total_init):
            idx = i - 3
            ang = 2 * math.pi * idx / initial_internal_slots
            self.pos_3d[i] = [
                np.random.uniform(-0.05, 0.05),
                0.35 * math.cos(ang),
                0.35 * math.sin(ang)
            ]

        self.node_type = ['source', 'source', 'sink'] + ['quadratic'] * (max_slots - 3)

        # W_in: [2, max_slots]
        # W_out: [max_slots]
        self.W_in = np.zeros((2, max_slots))
        self.W_out = np.zeros(max_slots)

        for i in range(3, self.total_init):
            self.W_in[0, i] = np.random.uniform(-0.4, 0.4)
            self.W_in[1, i] = np.random.uniform(-0.4, 0.4)
            self.W_out[i] = np.random.uniform(-0.3, 0.3)

    @property
    def num_active(self) -> int:
        return int(np.sum(self.active_mask))

    def forward_flow(self, a_val: float, b_val: float):
        inputs = np.array([a_val, b_val])
        act_internal = [k for k in range(3, self.max_slots) if self.active_mask[k]]

        v_all = np.zeros(self.max_slots)
        v_all[self.src_a] = a_val
        v_all[self.src_b] = b_val

        # 中间层二次极化计算
        u = np.dot(inputs, self.W_in[:, act_internal]) # [len(act_internal)]
        v = 0.25 * (u ** 2)
        v_all[act_internal] = v

        out = np.dot(v, self.W_out[act_internal])
        v_all[self.sink] = out
        return float(out), v_all, u

    def step(self, a_val: float, b_val: float, target_val: float, anneal: float = 1.0):
        y_pred, v_all, u = self.forward_flow(a_val, b_val)
        error = target_val - y_pred
        cur_lr = self.lr * anneal
        inputs = np.array([a_val, b_val])

        act_internal = [k for k in range(3, self.max_slots) if self.active_mask[k]]

        # 1. 输出汇端自适应
        grad_out = error * v_all[act_internal]
        self.W_out[act_internal] = np.clip(self.W_out[act_internal] + cur_lr * grad_out, -3.0, 3.0)

        # 2. 输入源端自适应
        back_u = error * self.W_out[act_internal] * (0.5 * u)
        grad_in = np.outer(inputs, back_u)
        self.W_in[:, act_internal] = np.clip(self.W_in[:, act_internal] + cur_lr * grad_in, -3.0, 3.0)

        # 3. 构形定律曲率爆裂 (Hopf 余乘 Delta 分裂)
        splits = 0
        if abs(error) > 0.45 * anneal and self.num_active < self.max_slots:
            bottleneck = act_internal[np.argmax([abs(v_all[k] * self.W_out[k]) for k in act_internal])]
            free_slots = np.where(~self.active_mask)[0]
            if len(free_slots) > 0:
                new_slot = free_slots[0]
                self.active_mask[new_slot] = True

                # 空间并联
                self.pos_3d[new_slot] = self.pos_3d[bottleneck] + np.random.uniform(-0.06, 0.06, size=3)
                self.pos_3d[new_slot, 0] = np.clip(self.pos_3d[new_slot, 0], -0.15, 0.15)

                # 分裂权重: 互补极化 (反相支路种子)
                self.W_in[:, new_slot] = self.W_in[:, bottleneck] + np.random.uniform(-0.1, 0.1, size=2)
                self.W_out[new_slot] = -self.W_out[bottleneck] * 0.5
                self.W_out[bottleneck] *= 0.5
                splits += 1

        throughput = max(0.01, abs(y_pred))
        resistance = (abs(a_val) + abs(b_val)) / throughput + (error ** 2) * 0.2
        return {
            'output': float(y_pred),
            'error': float(abs(error)),
            'resistance': float(resistance),
            'num_slots': self.num_active,
            'splits': splits
        }

if __name__ == '__main__':
    np.random.seed(42)
    engine = LaminarConstructalEngine(initial_internal_slots=4, max_slots=20, lr=0.035)
    total_epochs = 300

    val_set = [
        (1.2, 1.0, 1.2),
        (2.0, 1.8, 3.6),
        (1.5, 1.5, 2.25),
        (1.8, 1.2, 2.16),
        (1.0, 2.0, 2.0)
    ]

    for ep in range(total_epochs):
        anneal = 1.0 / (1.0 + 2.0 * (ep / total_epochs))

        for step_i in range(5):
            t = (ep * 5 + step_i) * 0.05
            a = 1.6 + 0.6 * math.sin(t)
            b = 1.4 + 0.6 * math.sin(t * math.sqrt(2))
            target = a * b
            res = engine.step(a, b, target, anneal=anneal)

        val_errs = [abs(vt - engine.forward_flow(va, vb)[0]) for va, vb, vt in val_set]
        mean_val_err = np.mean(val_errs)

        if ep % 30 == 0 or ep == total_epochs - 1:
            print(f"Epoch {ep:3d} | Slots: {res['num_slots']:2d} | Train Error: {res['error']:.5f} | Val Error: {mean_val_err:.5f} | R: {res['resistance']:.3f}")
