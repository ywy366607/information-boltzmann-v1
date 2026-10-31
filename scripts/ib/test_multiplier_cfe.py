import math
import numpy as np

class MultiplierCFE:
    def __init__(
        self,
        initial_internal_slots: int = 8,
        max_slots: int = 36,
        lr: float = 0.05
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
            self.pos_3d[i] = [
                np.random.uniform(-0.45, 0.45),
                np.random.uniform(-0.35, 0.35),
                np.random.uniform(-0.35, 0.35)
            ]

        self.node_type = ['source', 'source', 'sink'] + [
            ('quadratic' if i % 2 == 0 else 'linear') for i in range(initial_internal_slots)
        ] + ['linear'] * (max_slots - self.total_init)

        self.W = np.zeros((max_slots, max_slots))
        for i in range(self.total_init):
            for j in range(self.total_init):
                if self.pos_3d[j, 0] > self.pos_3d[i, 0] + 0.05:
                    d = np.linalg.norm(self.pos_3d[i] - self.pos_3d[j])
                    self.W[i, j] = 0.25 * math.exp(-d / 0.55)

    @property
    def num_active(self) -> int:
        return int(np.sum(self.active_mask))

    def forward_flow(self, a_val: float, b_val: float):
        N = self.max_slots
        v = np.zeros(N)
        v[self.src_a] = a_val
        v[self.src_b] = b_val

        order = np.argsort(self.pos_3d[:, 0])
        for _ in range(4):
            new_v = np.zeros(N)
            new_v[self.src_a] = a_val
            new_v[self.src_b] = b_val

            for j in order:
                if not self.active_mask[j] or j == self.src_a or j == self.src_b:
                    continue
                inflow = np.dot(self.W[:j, j], v[:j])
                if j == self.sink:
                    new_v[j] = inflow
                elif self.node_type[j] == 'quadratic':
                    new_v[j] = 0.25 * (inflow ** 2)
                else:
                    new_v[j] = inflow
            v = new_v

        return float(v[self.sink]), v

    def train_step(self, a_val: float, b_val: float, target_val: float, anneal: float = 1.0):
        y_pred, v_all = self.forward_flow(a_val, b_val)
        error = target_val - y_pred
        cur_lr = self.lr * anneal

        # 局域反压反传
        delta = np.zeros(self.max_slots)
        delta[self.sink] = error

        order_rev = np.argsort(self.pos_3d[:, 0])[::-1]
        for j in order_rev:
            if not self.active_mask[j] or j == self.sink or j == self.src_a or j == self.src_b:
                continue
            downstream_press = np.dot(self.W[j, :], delta)
            if self.node_type[j] == 'quadratic':
                inflow = np.dot(self.W[:, j], v_all)
                delta[j] = downstream_press * (0.5 * inflow)
            else:
                delta[j] = downstream_press

        # 贝让管网自适应
        for i in range(self.max_slots):
            if not self.active_mask[i]:
                continue
            for j in range(2, self.max_slots):
                if not self.active_mask[j]:
                    continue
                if self.pos_3d[j, 0] > self.pos_3d[i, 0] + 0.02:
                    grad = v_all[i] * delta[j] * 0.08
                    self.W[i, j] = np.clip(self.W[i, j] + cur_lr * grad - 0.0001 * self.W[i, j], -3.0, 3.0)

        # 构形分裂
        splits = 0
        if abs(error) > 0.4 * anneal and self.num_active < self.max_slots:
            act_internal = [k for k in range(3, self.max_slots) if self.active_mask[k]]
            if act_internal:
                bottleneck = act_internal[np.argmax([abs(v_all[k] * delta[k]) for k in act_internal])]
                free_slots = np.where(~self.active_mask)[0]
                if len(free_slots) > 0:
                    new_slot = free_slots[0]
                    self.active_mask[new_slot] = True
                    self.node_type[new_slot] = 'quadratic' if self.node_type[bottleneck] == 'linear' else 'linear'
                    self.pos_3d[new_slot] = self.pos_3d[bottleneck] + np.random.uniform(-0.12, 0.12, size=3)
                    self.pos_3d[new_slot, 0] = np.clip(self.pos_3d[new_slot, 0], -0.6, 0.6)
                    self.pos_3d[new_slot, 1] = np.clip(self.pos_3d[new_slot, 1], -0.4, 0.4)
                    self.pos_3d[new_slot, 2] = np.clip(self.pos_3d[new_slot, 2], -0.4, 0.4)
                    self.W[:, new_slot] = self.W[:, bottleneck] * 0.5
                    self.W[new_slot, :] = -self.W[bottleneck, :] * 0.5 # 反相种子
                    splits += 1

        throughput = max(0.01, abs(y_pred))
        total_resistance = (abs(a_val) + abs(b_val)) / throughput + (error ** 2) * 0.2

        return {
            'output': float(y_pred),
            'error': float(abs(error)),
            'resistance': float(total_resistance),
            'num_slots': self.num_active,
            'splits': splits
        }

if __name__ == '__main__':
    np.random.seed(42)
    engine = MultiplierCFE(initial_internal_slots=8, max_slots=36, lr=0.08)

    total_epochs = 300
    all_eval_errors = []

    # 固定的独立验证集 (Independent Validation Set) 检验整体乘法拟合泛化质量
    val_set = [
        (1.2, 1.0, 1.2),
        (2.0, 1.8, 3.6),
        (1.5, 1.5, 2.25),
        (1.8, 1.2, 2.16),
        (1.0, 2.0, 2.0)
    ]

    for ep in range(total_epochs):
        anneal = 1.0 / (1.0 + 1.5 * (ep / total_epochs))

        # 每个 epoch 经历 4 个微步经验，遍历不同流形点
        for step_i in range(4):
            t = (ep * 4 + step_i) * 0.05
            a = 1.6 + 0.6 * math.sin(t)
            b = 1.4 + 0.6 * math.sin(t * math.sqrt(2)) # 遍历全平面
            target = a * b
            res = engine.train_step(a, b, target, anneal=anneal)

        # 独立评估集误差 (Independent validation across whole 2D space)
        val_errs = []
        for va, vb, vt in val_set:
            vo, _ = engine.forward_flow(va, vb)
            val_errs.append(abs(vt - vo))
        mean_val_err = np.mean(val_errs)
        all_eval_errors.append(mean_val_err)

        if ep % 30 == 0 or ep == total_epochs - 1:
            print(f"Epoch {ep:3d} | Slots: {res['num_slots']:2d} | Train Error: {res['error']:.4f} | Independent Val Error: {mean_val_err:.4f} | R: {res['resistance']:.3f}")
