"""Re-measure the duration and precision ladders on the mature K=16
unified-bath checkpoint (best, step 2500)."""
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

CKPT = ROOT / "results/q8_predictive_ports_k16_unified_bath_3000/BBest.pt"  # step 2500

saved = torch.load(CKPT, map_location="cuda", weights_only=False)
cfg = saved["config"]
model = CBIMTorus3D(
    shape=tuple(cfg["shape"]), velocities=cfg["velocities"],
    content_dim=cfg["content_dim"], collision_layers=cfg["collision_layers"],
    relative_address=True, readout_type="belief_agent",
    write_type="w4_predictive_agent", micro_steps=cfg["K"],
    dissipation_type="unified", dissipation_rank=cfg["dissipation_rank"],
    tau_0=cfg["tau_0"],
).cuda()
model.load_state_dict(saved["model"])
terminal = KineticBeliefState(
    saved["state"].detach().clone().cuda(), saved["precision"].detach().clone().cuda())
model.set_ness_prior(terminal.field)

validation = np.load(ROOT / "data/ib_owt_gpt2/validation.npy", mmap_mode="r")
spec = WarmSiteSpec(
    site_starts=(8192, 12288, 16384, 20480), warm_in_tokens=256, score_tokens=128)

base_tau = float(model.tau_0_tensor.item())  # 4.0
trained_steps, trained_tau = cfg["K"], cfg["tau_0"]

print("DURATION ladder (dt=4*tau0 fixed, T = K*4*tau0 shrinks):")
for k in (16, 8, 4, 1):
    model.tau_0_tensor.fill_(base_tau)
    torch.cuda.synchronize()
    started = time.perf_counter()
    report = evaluate_warm_sites(model, terminal, validation, spec, k, ness_phase_seed=11)
    elapsed = time.perf_counter() - started
    print(f"  K={k:>2} T={k*base_tau:5.1f} | NLL {float(report['nll']):8.4f} | {elapsed:5.1f} s")

print("PRECISION ladder (T = 64*tau0 fixed, dt = 64*tau0/K):")
for k in (16, 8, 4, 1):
    model.tau_0_tensor.fill_(base_tau * trained_steps / k)
    torch.cuda.synchronize()
    started = time.perf_counter()
    report = evaluate_warm_sites(model, terminal, validation, spec, k, ness_phase_seed=11)
    elapsed = time.perf_counter() - started
    print(f"  K={k:>2} dt={trained_tau*trained_steps/k:5.1f} | NLL {float(report['nll']):8.4f} | {elapsed:5.1f} s")
model.tau_0_tensor.fill_(base_tau)
