import time
import torch

device = torch.device("cuda")
max_slots = 512
vocab_size = 50257
M = 300

W_out = torch.randn(max_slots, vocab_size, device=device)
act_idx = torch.arange(M, device=device)
act_norm = torch.rand(M, device=device)

# Method 1: Indexed slice
torch.cuda.synchronize()
t0 = time.time()
for _ in range(500):
    z1 = torch.matmul(act_norm, W_out[act_idx])
torch.cuda.synchronize()
t_slice = (time.time() - t0) * 1000 / 500
print(f"Method 1 (Indexed slice): {t_slice:.3f} ms/step")

# Method 2: Contiguous GEMV with full vector
full_act = torch.zeros(max_slots, device=device)
full_act[act_idx] = act_norm

torch.cuda.synchronize()
t0 = time.time()
for _ in range(500):
    full_act.zero_()
    full_act[act_idx] = act_norm
    z2 = torch.matmul(full_act, W_out)
torch.cuda.synchronize()
t_gemv = (time.time() - t0) * 1000 / 500
print(f"Method 2 (Contiguous GEMV): {t_gemv:.3f} ms/step")

# Method 3: Update column
torch.cuda.synchronize()
t0 = time.time()
for _ in range(500):
    W_out[:, 1234] += 0.4 * full_act
torch.cuda.synchronize()
t_col = (time.time() - t0) * 1000 / 500
print(f"Method 3 (Vectorized column update): {t_col:.3f} ms/step")
