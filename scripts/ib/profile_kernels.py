"""Kernel-level profile of one captured-size chunk on the step-500 checkpoint."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core.torus3d import CBIMTorus3D

saved = torch.load(ROOT / "results/q8_predictive_ports_k64_3000_v2/last.pt",
                   map_location="cuda", weights_only=False)
model = CBIMTorus3D(
    shape=(8, 8, 4), velocities=8, content_dim=16, collision_layers=2,
    relative_address=True, readout_type="belief_agent",
    write_type="w4_predictive_agent", micro_steps=64,
    dissipation_type="quadratic",
).cuda()
model.load_state_dict(saved["model"])
ids = torch.randint(0, 50257, (1, 8), device="cuda")
targets = torch.randint(0, 50257, (1, 8), device="cuda")

for _ in range(2):
    loss, _, _ = model.forward_belief(ids, targets)
    loss.backward()
    model.zero_grad(set_to_none=True)
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    loss, _, _ = model.forward_belief(ids, targets)
    loss.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18))
