import sys, os, time
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn.functional as F

device = torch.device("cuda")

# Test 1: Baseline list comprehension with .item()
age = torch.randint(0, 100, (256,), device=device)
act_idx = torch.arange(200, device=device)

t0 = time.time()
for _ in range(500):
    mature_candidates = [k for k in act_idx if age[k].item() > 40]
torch.cuda.synchronize()
print(f"List comp with .item(): {(time.time() - t0)*1000/500:.3f} ms/step")

# Test 2: Pure PyTorch vector mask
active_mask = torch.zeros(256, dtype=torch.bool, device=device)
active_mask[:200] = True

t0 = time.time()
for _ in range(500):
    mature_mask = (age > 40) & active_mask
    if mature_mask.sum() > 64:
        cand = torch.where(mature_mask)[0]
torch.cuda.synchronize()
print(f"Pure PyTorch vector mask: {(time.time() - t0)*1000/500:.3f} ms/step")
