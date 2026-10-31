import sys, os, time, math
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on device: {device}")

vocab_size = 50257
dim = 64
init_slots = 256
max_slots = 512

# Check initial NLL and stability with context resonance
class ContextCFE(nn.Module):
    def __init__(self, vocab_size=50257, dim=64, init_slots=256, max_slots=512, device="cuda"):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.init_slots = init_slots
        self.max_slots = max_slots
        self.device = torch.device(device)

        self.embedding = nn.Embedding(vocab_size, dim, device=self.device)
        nn.init.normal_(self.embedding.weight, std=1.0 / math.sqrt(dim))

        self.register_buffer("e_ground", torch.ones(vocab_size, device=self.device) / vocab_size)
        self.register_buffer("active_mask", torch.zeros(max_slots, dtype=torch.bool, device=self.device))
        self.active_mask[:init_slots] = True

        self.register_buffer("energy", torch.ones(max_slots, device=self.device))
        self.register_buffer("theta", torch.rand(max_slots, device=self.device) * 2 * math.pi)
        self.register_buffer("omega", torch.randn(max_slots, device=self.device) * 0.1)
        self.register_buffer("refractory", torch.zeros(max_slots, device=self.device))

        # Token signature k_i
        raw_k = torch.randn(max_slots, dim, device=self.device)
        self.register_buffer("signature", F.normalize(raw_k, p=2, dim=-1))

        # Context signature c_i
        raw_c = torch.randn(max_slots, dim, device=self.device)
        self.register_buffer("context_sig", F.normalize(raw_c, p=2, dim=-1))

        self.register_buffer("age", torch.zeros(max_slots, dtype=torch.long, device=self.device))
        self.register_buffer("W_out", torch.zeros(max_slots, vocab_size, device=self.device))
        self.register_buffer("residual_var", torch.tensor(1.0, device=self.device))

        # Multi-scale continuous context state
        self.register_buffer("h_fast", torch.zeros(dim, device=self.device))
        self.register_buffer("h_slow", torch.zeros(dim, device=self.device))

        self.tokens_processed = 0
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

        # Context signature
        h_ctx = F.normalize(0.7 * self.h_fast + 0.3 * self.h_slow + 1e-8, p=2, dim=-1)

        sig_act = self.signature[act_idx]
        ctx_act = self.context_sig[act_idx]
        r_act = self.refractory[act_idx]

        cos_token = F.relu(torch.matmul(sig_act, k_in))
        cos_ctx = torch.clamp((torch.matmul(ctx_act, h_ctx) + 1.0) * 0.5, 0.1, 1.0)

        # Resonant activation
        act = (1.0 - r_act) * cos_token * cos_ctx

        phi_0 = torch.log(self.e_ground / self.e_ground.sum() + 1e-12)

        if len(act_idx) > 0 and act.sum() > 1e-6:
            act_norm = act / act.sum()
            z_ac = torch.matmul(act_norm, self.W_out[act_idx])
        else:
            z_ac = torch.zeros(self.vocab_size, device=self.device)

        beta_star = float(1.0 / math.sqrt(self.residual_var.item() + 0.04))
        logits = phi_0 + beta_star * z_ac
        return logits, act, beta_star, k_in, h_ctx

    @torch.no_grad()
    def step_stream(self, current_token: int, next_token: int):
        self.tokens_processed += 1
        act_idx = torch.where(self.active_mask)[0]
        M = len(act_idx)

        logits, act, beta_star, k_in, h_ctx = self.prequential_predict(current_token)
        target_tensor = torch.tensor([next_token], device=self.device)
        surprisal_nats = F.cross_entropy(logits.unsqueeze(0), target_tensor).item()
        surprisal_bits = surprisal_nats / math.log(2)

        # Update DC ground state
        self.e_ground[next_token] += 0.001
        if self.tokens_processed % 200 == 0:
            self.e_ground /= self.e_ground.sum()

        # Update impedance
        normalized_res = (surprisal_nats / 10.825) ** 2
        self.residual_var = 0.98 * self.residual_var + 0.02 * normalized_res

        # Operators 1 & 2: Kuramoto and flow
        sig_act = self.signature[act_idx]
        ctx_act = self.context_sig[act_idx]
        theta_act = self.theta[act_idx]
        energy_act = self.energy[act_idx]
        r_act = self.refractory[act_idx]

        K_mat = F.relu(torch.matmul(sig_act, sig_act.T))
        d_theta = theta_act.unsqueeze(0) - theta_act.unsqueeze(1)
        W_mat = (1.0 - r_act).unsqueeze(1) * K_mat * torch.cos(d_theta)

        flux_term = (K_mat * torch.sin(d_theta)).sum(dim=1)
        J_act = energy_act * (1.0 - r_act) * flux_term

        kuramoto_force = (W_mat * torch.sin(d_theta) * K_mat).sum(dim=1) / max(1, M)
        self.theta[act_idx] = (theta_act + self.omega[act_idx] + 0.1 * kuramoto_force) % (2 * math.pi)
        self.refractory[act_idx] = torch.clamp(0.85 * r_act + 0.15 * torch.sigmoid(energy_act - 1.0), 0.0, 1.0)

        # Flow in dual space: signature towards k_in, context_sig towards h_ctx
        flow_k = torch.matmul(W_mat, sig_act) * 0.03 + act.unsqueeze(1) * k_in * 0.08
        self.signature[act_idx] = F.normalize(sig_act + flow_k, p=2, dim=-1)

        flow_c = act.unsqueeze(1) * h_ctx * 0.08
        self.context_sig[act_idx] = F.normalize(ctx_act + flow_c, p=2, dim=-1)

        # Tubule reinforcement & Murray decay
        self.W_out[act_idx, next_token] += 0.40 * act
        if self.tokens_processed % 5 == 0:
            self.W_out.mul_(1.0 - 0.001)

        # Update context momentum
        self.h_fast = 0.55 * self.h_fast + 0.45 * k_in
        self.h_slow = 0.92 * self.h_slow + 0.08 * k_in

        # Resistance and branching
        rho = (J_act ** 2) / (energy_act + 1e-4) + 0.015 * energy_act + 0.04 * surprisal_bits
        splits = 0
        if M < self.max_slots:
            mu_rho = rho.mean()
            std_rho = rho.std()
            split_cand = torch.where(rho > mu_rho + 1.8 * std_rho)[0]
            if len(split_cand) > 0:
                parent_idx = act_idx[split_cand[0]].item()
                free_slots = torch.where(~self.active_mask)[0]
                if len(free_slots) > 0:
                    child_idx = free_slots[0].item()
                    self.active_mask[child_idx] = True
                    self.age[child_idx] = 0
                    self.age[parent_idx] = 0
                    self.energy[parent_idx] *= 0.5
                    self.energy[child_idx] = self.energy[parent_idx]
                    self.theta[child_idx] = (self.theta[parent_idx] + math.pi) % (2 * math.pi)
                    self.omega[child_idx] = -self.omega[parent_idx]
                    self.refractory[child_idx] = 0.0

                    # Split along tangent
                    rand_tangent = torch.randn(self.dim, device=self.device)
                    rand_tangent -= torch.dot(rand_tangent, self.signature[parent_idx]) * self.signature[parent_idx]
                    rand_tangent = F.normalize(rand_tangent, p=2, dim=-1)
                    self.signature[child_idx] = F.normalize(
                        self.signature[parent_idx] * math.cos(0.55) + rand_tangent * math.sin(0.55), p=2, dim=-1
                    )
                    # Child specializes context towards current h_ctx
                    self.context_sig[child_idx] = F.normalize(0.6 * self.context_sig[parent_idx] + 0.4 * h_ctx, p=2, dim=-1)

                    self.W_out[child_idx] = self.W_out[parent_idx] * 0.5
                    self.W_out[parent_idx] *= 0.5
                    splits += 1
                    self.total_splits += 1

        self.age[act_idx] += 1

        # Merging with zero python loop
        merges = 0
        mature_mask = (self.age > 40) & self.active_mask
        if mature_mask.sum() > self.init_slots:
            mature_idx = torch.where(mature_mask)[0]
            sig_m = self.signature[mature_idx]
            sim_m = torch.matmul(sig_m, sig_m.T)
            sim_m.fill_diagonal_(0.0)
            max_sim, max_p = sim_m.max(dim=1)
            pair_i = torch.argmax(max_sim)
            if max_sim[pair_i] > 0.96:
                idx_i = mature_idx[pair_i]
                idx_j = mature_idx[max_p[pair_i]]
                self.energy[idx_i] += self.energy[idx_j]
                self.signature[idx_i] = F.normalize(self.signature[idx_i] + self.signature[idx_j], p=2, dim=-1)
                self.context_sig[idx_i] = F.normalize(self.context_sig[idx_i] + self.context_sig[idx_j], p=2, dim=-1)
                self.W_out[idx_i] = torch.maximum(self.W_out[idx_i], self.W_out[idx_j])
                self.active_mask[idx_j] = False
                merges += 1
                self.total_merges += 1

        return {
            "surprisal_nats": surprisal_nats,
            "surprisal_bits": surprisal_bits,
            "beta_star": beta_star,
            "num_slots": float(self.num_active),
            "splits": float(splits),
            "merges": float(merges)
        }

# Quick test on 1000 real OWT tokens
train_tokens = np.load("data/ib_owt_gpt2/train.npy", mmap_mode="r")
engine = ContextCFE(vocab_size=50257, dim=64, init_slots=256, max_slots=512, device=device)

print(f"Initial slots: {engine.num_active}, VRAM: {torch.cuda.memory_allocated()/1e6:.1f} MB")
t0 = time.time()
surps = []
for step in range(1000):
    c_tok = int(train_tokens[step])
    n_tok = int(train_tokens[step + 1])
    res = engine.step_stream(c_tok, n_tok)
    surps.append(res["surprisal_nats"])
    if step % 200 == 0:
        print(f"Step {step:4d} | Slots: {int(res['num_slots']):3d} | NLL (last 50): {np.mean(surps[-50:]):.3f} | beta: {res['beta_star']:.2f} | Splits: {res['splits']}")

torch.cuda.synchronize()
dt = time.time() - t0
print(f"1000 steps completed in {dt:.2f}s ({dt/1000*1000:.2f} ms/step)!")
print(f"Final 100 tokens NLL: {np.mean(surps[-100:]):.3f} nats ({np.mean(surps[-100:])/math.log(2):.3f} bits)")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e6:.1f} MB")
