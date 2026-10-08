"""Probe-position binding test: did probe coordinates drift from the
initialization lattice, and does permuting positions (keeping per-probe
weights) change NLL?"""
import sys
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.evaluation import WarmSiteSpec
from scripts.ib.evaluate_continuous_owt import _model_from_checkpoint, evaluate_warm_sites

saved = torch.load(ROOT / "results/q8_ness_reference_3000/BBest.pt",
                   map_location="cuda", weights_only=False)
model, micro_steps, _ = _model_from_checkpoint(saved)
model = model.cuda()
model.load_state_dict(saved["model"])

trained = model.readout.probe_coords.detach().clone()
init_pts = []
for z in [0.125, 0.375, 0.625, 0.875]:
    for x in [0.25, 0.75]:
        for y in [0.25, 0.75]:
            init_pts.append(torch.tensor([x, y, z], dtype=torch.float32,
                                         device=trained.device))
init = torch.stack(init_pts).reshape(trained.shape)

drift = (trained - init).norm(dim=-1)
torus = 1.0
print(f"probe drift from init lattice: mean {drift.mean():.4f} max {drift.max():.4f} "
      f"(torus period {torus})")
cos = (torch.nn.functional.normalize(trained.reshape(-1, 3), dim=-1)
       * torch.nn.functional.normalize(init.reshape(-1, 3), dim=-1)).sum(-1)
print(f"per-probe direction cosine to init: mean {cos.mean():.4f} min {cos.min():.4f}")
betas = model.readout.log_beta.exp().detach() if hasattr(model.readout, "log_beta") else None
for name in ("beta", "log_beta", "temperature"):
    if hasattr(model.readout, name):
        value = getattr(model.readout, name)
        print(f"readout.{name}: shape {tuple(value.shape)} mean {value.float().mean():.3f}")

validation = np.load(ROOT / "data/ib_owt_gpt2/validation.npy", mmap_mode="r")
spec = WarmSiteSpec(site_starts=(8192,), warm_in_tokens=256, score_tokens=128)


def nll_with(coords):
    model.readout.probe_coords.data.copy_(coords)
    report = evaluate_warm_sites(model, saved["state"].cuda().float(), validation,
                                 spec, micro_steps, ness_phase_seed=11)
    return float(report["nll"])


base = nll_with(trained)
print(f"trained probe positions        : NLL {base:.4f}")
reverse = trained.flip(0)
print(f"positions reversed             : NLL {nll_with(reverse):.4f}")
generator = torch.Generator().manual_seed(5)
perm = torch.randperm(trained.shape[0] * trained.shape[1], generator=generator)
permuted = trained.reshape(-1, 3)[perm].reshape(trained.shape)
print(f"positions randomly permuted    : NLL {nll_with(permuted):.4f}")
shifted = (trained + 0.5) % 1.0
print(f"positions shifted by half torus: NLL {nll_with(shifted):.4f}")
model.readout.probe_coords.data.copy_(trained)
