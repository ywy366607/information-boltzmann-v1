"""Zero-training PRECISION ladder: fixed total evolution time T = 64*tau0,
varying only the quadrature resolution (dt = T/K) by rescaling tau0."""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core.torus3d import CBIMTorus3D, KineticBeliefState
from information_boltzmann.evaluation import WarmSiteSpec
from scripts.ib.evaluate_continuous_owt import evaluate_warm_sites

CKPT = ROOT / "results/q8_predictive_ports_k64_3000_v2/last.pt"  # step 500

saved = torch.load(CKPT, map_location="cuda", weights_only=False)
model = CBIMTorus3D(
    shape=(8, 8, 4), velocities=8, content_dim=16, collision_layers=2,
    relative_address=True, readout_type="belief_agent",
    write_type="w4_predictive_agent", micro_steps=64,
    dissipation_type="quadratic",
).cuda()
model.load_state_dict(saved["model"])
terminal = KineticBeliefState(
    saved["state"].detach().clone().cuda(), saved["precision"].detach().clone().cuda())
model.set_ness_prior(terminal.field)

validation = np.load(ROOT / "data/ib_owt_gpt2/validation.npy", mmap_mode="r")
spec = WarmSiteSpec(
    site_starts=(8192, 12288, 16384, 20480), warm_in_tokens=256, score_tokens=128)

base_tau = float(model.tau_0_tensor.item())
print(f"checkpoint step {saved['step']} | PRECISION ladder: T = 64*tau0 fixed, dt = T/K")
for k in (64, 16, 4, 1):
    model.tau_0_tensor.fill_(base_tau * 64.0 / k)
    torch.cuda.synchronize()
    started = time.perf_counter()
    report = evaluate_warm_sites(model, terminal, validation, spec, k, ness_phase_seed=11)
    elapsed = time.perf_counter() - started
    print(f"K={k:>3} dt={64.0/k:5.1f}*tau0 | NLL {float(report['nll']):8.4f} "
          f"| eval {elapsed:6.1f} s")
model.tau_0_tensor.fill_(base_tau)
