import math
import numpy as np

class StratifiedConstructalEngine:
    def __init__(
        self,
        num_internal: int = 12,
        lr: float = 0.05
    ):
        self.num_internal = num_internal
        self.lr = lr

        # 0: Src A, 1: Src B, 2: Sink Out
        # 3 .. 3 + num_internal - 1: Intermediate Quadratic Compute Nodes
        self.max_slots = 3 + num_internal
        self.active_mask = np.ones(self.max_slots, dtype=bool)

        self.pos_3d = np.zeros((self.max_slots, 3))
        self.pos_3d[0] = [-0.85, -0.25, 0.0]
        self.pos_3d[1] = [-0.85, +0.25, 0.0]
        self.pos_3d[2] = [+0.85, 0.0, 0.0]

        # 内部节点均匀分布在计算区 x in [-0.2, 0.2]
        for i in range(3, self.max_slots):
            idx = i - 3
            ang = 2 * math.pi * idx / num_internal
            self.pos_3d[i] = [
                np.random.uniform(-0.15, 0.15),
                0.3 * math.cos(ang),
                0.3 * math.sin(ang)
            ]

        # W_in: [2, num_internal] 从源到中间层
        # W_out: [num_internal] 从中间层到汇端
        self.W_in = np.random.uniform(-0.3, 0.3, (2, num_internal))
        self.W_out = np.random.uniform(-0.3, 0.3, num_internal)

    def forward(self, a_val: float, b_val: float):
        inputs = np.array([a_val, b_val])
        # 中间非线性激活: v_k = 0.25 * (W_in^T * x)^2
        u = np.dot(inputs, self.W_in)
        v = 0.25 * (u ** 2)
        out = np.dot(v, self.W_out)
        return float(out), u, v

    def step(self, a_val: float, b_val: float, target_val: float, anneal: float = 1.0):
        y_pred, u, v = self.forward(a_val, b_val)
        error = target_val - y_pred
        cur_lr = self.lr * anneal

        # 贝让液压反压梯度
        # delta_out = error * v
        # delta_in = error * W_out * (0.5 * u) * inputs
        inputs = np.array([a_val, b_val])

        grad_out = error * v
        self.W_out = np.clip(self.W_out + cur_lr * grad_out, -3.0, 3.0)

        back_u = error * self.W_out * (0.5 * u) # [num_internal]
        grad_in = np.outer(inputs, back_u)
        self.W_in = np.clip(self.W_in + cur_lr * grad_in, -3.0, 3.0)

        throughput = max(0.01, abs(y_pred))
        resistance = (abs(a_val) + abs(b_val)) / throughput + (error ** 2) * 0.2
        return {
            'output': float(y_pred),
            'error': float(abs(error)),
            'resistance': float(resistance)
        }

if __name__ == '__main__':
    np.random.seed(42)
    engine = StratifiedConstructalEngine(num_internal=12, lr=0.03)
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

        val_errs = [abs(vt - engine.forward(va, vb)[0]) for va, vb, vt in val_set]
        mean_val_err = np.mean(val_errs)

        if ep % 30 == 0 or ep == total_epochs - 1:
            print(f"Epoch {ep:3d} | Train Error: {res['error']:.5f} | Val Error: {mean_val_err:.5f} | R: {res['resistance']:.3f}")
