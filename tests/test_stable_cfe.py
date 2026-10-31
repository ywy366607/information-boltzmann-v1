import sys, os, time, math
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing Stable Branching CFE on device: {device}")

class StableConstructalEngine(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        dim: int = 64,
        init_slots: int = 384,
        max_slots: int = 768,
        pipe_lr: float = 4.0,
        pipe_decay: float = 0.0005,
        device: str = "cuda"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.init_slots = init_slots
        self.max_slots = max_slots
        self.pipe_lr = pipe_lr
        self.pipe_decay = pipe_decay
        self.device = torch.device(device)

        self.embedding = nn.Embedding(vocab_size, dim, device=self.device)
        nn.init.normal_(self.embedding.weight, std=1.0 / math.sqrt(dim))

        self.register_buffer("e_ground", torch.ones(vocab_size, device=self.device) / vocab_size)
        self.register_buffer("active_mask", torch.zeros(max_slots, dtype=torch.bool, device=self.device))
        self.active_mask[:init_slots] = True

        # Hierarchical gammas: 1/3 fast (0.50), 1/3 mid (0.82), 1/3 slow (0.96)
        gammas = torch.empty(max_slots, device=self.device)
        n_third = max_slots // 3
        gammas[:n_third] = torch.empty(n_third, device=self.device).uniform_(0.45, 0.65)
        gammas[n_third: 2 * n_third] = torch.empty(n_third, device=self.device).uniform_(0.75, 0.88)
        gammas[2 * n_third:] = torch.empty(max_slots - 2 * n_third, device=self.device).uniform_(0.92, 0.98)
        self.register_buffer("gamma", gammas)

        self.register_buffer("energy", torch.ones(max_slots, device=self.device))
        self.register_buffer("theta", torch.rand(max_slots, device=self.device) * 2 * math.pi)
        self.register_buffer("omega", torch.randn(max_slots, device=self.device) * 0.1)
        self.register_buffer("refractory", torch.zeros(max_slots, device=self.device))

        raw_k = torch.randn(max_slots, dim, device=self.device)
        self.register_buffer("k_sig", F.normalize(raw_k, p=2, dim=-1))
        raw_c = torch.randn(max_slots, dim, device=self.device)
        self.register_buffer("c_sig", F.normalize(raw_c, p=2, dim=-1))

        self.register_buffer("trace", torch.zeros(max_slots, dim, device=self.device))
        self.register_buffer("age", torch.zeros(max_slots, dtype=torch.long, device=self.device))

        self.register_buffer("W_out", torch.zeros(max_slots, vocab_size, device=self.device))
        self.register_buffer("residual_var", torch.tensor(1.0, device=self.device))

        self.tokens_processed = 0
        self.last_split_step = -100
        self.total_splits = 0
        self.total_merges = 0

    @property
    def num_active(self) -> int:
        return int(self.active_mask.sum().item())

    def token_sig(self, token_id: int) -> torch.Tensor:
        return F.normalize(self.embedding.weight[token_id], p=2, dim=-1)

    @torch.no_grad()
    def prequential_predict(self, current_token: int):
        act_idx = torch.where(self.active_mask)[0]
        k_in = self.token_sig(current_token)

        k_act = self.k_sig[act_idx]
        c_act = self.c_sig[act_idx]
        tr_act = F.normalize(self.trace[act_idx] + 1e-8, p=2, dim=-1)
        r_act = self.refractory[act_idx]

        cos_k = F.relu(torch.matmul(k_act, k_in))
        cos_c = torch.clamp((torch.sum(c_act * tr_act, dim=-1) + 1.0) * 0.5, 0.2, 1.0)

        act = (1.0 - r_act) * cos_k * cos_c

        phi_0 = torch.log(self.e_ground / self.e_ground.sum() + 1e-12)

        full_act = torch.zeros(self.max_slots, device=self.device)
        if len(act_idx) > 0 and act.sum() > 1e-6:
            act_norm = act / act.sum()
            full_act[act_idx] = act_norm
            z_ac = torch.matmul(full_act, self.W_out)
        else:
            z_ac = torch.zeros(self.vocab_size, device=self.device)

        beta_star = float(1.0 / math.sqrt(self.residual_var.item() + 0.04))
        logits = phi_0 + beta_star * z_ac

        return logits, act, beta_star, k_in, full_act

    @torch.no_grad()
    def step_stream(self, current_token: int, next_token: int):
        self.tokens_processed += 1
        act_idx = torch.where(self.active_mask)[0]
        M = len(act_idx)

        # 1. 预先评分
        logits, act, beta_star, k_in, full_act = self.prequential_predict(current_token)
        target_tensor = torch.tensor([next_token], device=self.device)
        surprisal_nats = F.cross_entropy(logits.unsqueeze(0), target_tensor).item()
        surprisal_bits = surprisal_nats / math.log(2)

        # 2. 直流基态稳态能流
        self.e_ground[next_token] += 0.001
        if self.tokens_processed % 200 == 0:
            self.e_ground /= self.e_ground.sum()

        # 3. 动态阻抗匹配
        normalized_res = (surprisal_nats / 10.825) ** 2
        self.residual_var = 0.98 * self.residual_var + 0.02 * normalized_res

        # 4. 算子 1 & 2: 锁相与流动
        k_act = self.k_sig[act_idx]
        c_act = self.c_sig[act_idx]
        theta_act = self.theta[act_idx]
        energy_act = self.energy[act_idx]
        r_act = self.refractory[act_idx]

        K_mat = F.relu(torch.matmul(k_act, k_act.T))
        d_theta = theta_act.unsqueeze(0) - theta_act.unsqueeze(1)
        W_mat = (1.0 - r_act).unsqueeze(1) * K_mat * torch.cos(d_theta)

        # 强化强度量缩放: 除以 M 保持强度量性质
        flux_term = (K_mat * torch.sin(d_theta)).sum(dim=1) / max(1, M)
        J_act = energy_act * (1.0 - r_act) * flux_term

        kuramoto_force = (W_mat * torch.sin(d_theta) * K_mat).sum(dim=1) / max(1, M)
        self.theta[act_idx] = (theta_act + self.omega[act_idx] + 0.1 * kuramoto_force) % (2 * math.pi)
        self.refractory[act_idx] = torch.clamp(0.85 * r_act + 0.15 * torch.sigmoid(energy_act - 1.0), 0.0, 1.0)

        flow_k = torch.matmul(W_mat, k_act) * (0.03 / max(1, M)) + act.unsqueeze(1) * k_in * 0.08
        self.k_sig[act_idx] = F.normalize(k_act + flow_k, p=2, dim=-1)

        tr_norm = F.normalize(self.trace[act_idx] + 1e-8, p=2, dim=-1)
        flow_c = act.unsqueeze(1) * tr_norm * 0.08
        self.c_sig[act_idx] = F.normalize(c_act + flow_c, p=2, dim=-1)

        # 5. 默里定律管径强化
        self.W_out[:, next_token] += self.pipe_lr * full_act
        if self.tokens_processed % 5 == 0:
            self.W_out.mul_(1.0 - self.pipe_decay)

        # 6. 资格迹
        g_act = self.gamma[act_idx].unsqueeze(1)
        self.trace[act_idx] = g_act * self.trace[act_idx] + (1.0 - g_act) * k_in

        # 7. 算子 3: 严格生物学有丝分裂冷却与成熟度约束 (Mitosis Cooldown)
        rho = (J_act ** 2) / (energy_act + 1e-4) + 0.015 * energy_act + 0.04 * surprisal_bits
        global_resistance = float(rho.sum().item())

        splits = 0
        can_split = (M < self.max_slots) and (self.tokens_processed - self.last_split_step > 30)
        if can_split:
            mature_parents = [k for k in range(M) if self.age[act_idx[k]].item() > 60]
            if len(mature_parents) > 0:
                rho_mature = rho[mature_parents]
                if rho_mature.max() > rho.mean() + 2.0 * rho.std():
                    cand_in_mature = torch.argmax(rho_mature).item()
                    parent_pos = mature_parents[cand_in_mature]
                    parent_idx = act_idx[parent_pos].item()

                    free_slots = torch.where(~self.active_mask)[0]
                    if len(free_slots) > 0:
                        child_idx = free_slots[0].item()
                        self.active_mask[child_idx] = True
                        self.age[child_idx] = 0
                        self.age[parent_idx] = 0

                        # 母体保留 80% 能量与流道，子体获得 20% 初生通量
                        self.energy[child_idx] = self.energy[parent_idx] * 0.2
                        self.energy[parent_idx] *= 0.8
                        self.theta[child_idx] = (self.theta[parent_idx] + math.pi) % (2 * math.pi)
                        self.omega[child_idx] = -self.omega[parent_idx]
                        self.refractory[child_idx] = 0.0

                        rand_tangent = torch.randn(self.dim, device=self.device)
                        rand_tangent -= torch.dot(rand_tangent, self.k_sig[parent_idx]) * self.k_sig[parent_idx]
                        rand_tangent = F.normalize(rand_tangent, p=2, dim=-1)
                        delta_angle = 0.55
                        self.k_sig[child_idx] = F.normalize(
                            self.k_sig[parent_idx] * math.cos(delta_angle) + rand_tangent * math.sin(delta_angle), p=2, dim=-1
                        )
                        self.c_sig[child_idx] = F.normalize(0.6 * self.c_sig[parent_idx] + 0.4 * tr_norm[parent_pos], p=2, dim=-1)
                        self.trace[child_idx] = self.trace[parent_idx].clone()
                        self.gamma[child_idx] = self.gamma[parent_idx]

                        self.W_out[child_idx] = self.W_out[parent_idx] * 0.2
                        self.W_out[parent_idx] *= 0.8

                        splits += 1
                        self.total_splits += 1
                        self.last_split_step = self.tokens_processed

        self.age[act_idx] += 1

        # 8. 算子 4: 拓扑合并 (成熟且高相似度)
        merges = 0
        mature_mask = (self.age > 80) & self.active_mask
        if mature_mask.sum() > self.init_slots:
            mature_idx = torch.where(mature_mask)[0]
            k_m = self.k_sig[mature_idx]
            sim_m = torch.matmul(k_m, k_m.T)
            sim_m.fill_diagonal_(0.0)
            max_sim, max_p = sim_m.max(dim=1)
            pair_i = torch.argmax(max_sim)
            if max_sim[pair_i] > 0.96:
                idx_i = mature_idx[pair_i]
                idx_j = mature_idx[max_p[pair_i]]
                self.energy[idx_i] += self.energy[idx_j]
                self.k_sig[idx_i] = F.normalize(self.k_sig[idx_i] + self.k_sig[idx_j], p=2, dim=-1)
                self.c_sig[idx_i] = F.normalize(self.c_sig[idx_i] + self.c_sig[idx_j], p=2, dim=-1)
                self.W_out[idx_i] = torch.maximum(self.W_out[idx_i], self.W_out[idx_j])
                self.active_mask[idx_j] = False
                merges += 1
                self.total_merges += 1

        return {
            "surprisal_nats": surprisal_nats,
            "surprisal_bits": surprisal_bits,
            "beta_star": beta_star,
            "resistance": global_resistance,
            "num_slots": float(self.num_active),
            "splits": float(splits),
            "merges": float(merges)
        }

if __name__ == "__main__":
    train_tokens = np.load("data/ib_owt_gpt2/train.npy", mmap_mode="r")
    engine = StableConstructalEngine(vocab_size=50257, dim=64, init_slots=384, max_slots=768, pipe_lr=4.0, device=device)
    print(f"Stable CFE Initialized: {engine.num_active} slots | VRAM: {torch.cuda.memory_allocated()/1e6:.1f} MB")

    t0 = time.time()
    surps = []
    total_steps = 5000
    for step in range(total_steps):
        c_tok = int(train_tokens[step])
        n_tok = int(train_tokens[step + 1])
        res = engine.step_stream(c_tok, n_tok)
        surps.append(res["surprisal_nats"])
        if step % 500 == 0 or step == total_steps - 1:
            recent_nats = np.mean(surps[-100:])
            print(f"Step {step:4d} | Slots: {int(res['num_slots']):3d} | NLL (last 100): {recent_nats:.3f} n ({recent_nats/math.log(2):.3f} b) | beta*: {res['beta_star']:.2f} | R: {res['resistance']:.1f} | S:{engine.total_splits} M:{engine.total_merges}")

    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"Completed {total_steps} tokens in {dt:.2f}s ({dt/total_steps*1000:.2f} ms/step)")
    print(f"Net NLL drop: {surps[0]:.3f} -> {np.mean(surps[-100:]):.3f} nats (Bits: {np.mean(surps[-100:])/math.log(2):.3f})")
    print(f"Total splits: {engine.total_splits}, Total merges: {engine.total_merges}")
