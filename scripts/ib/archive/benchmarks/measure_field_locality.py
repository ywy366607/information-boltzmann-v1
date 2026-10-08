"""Spatial-structure audit of saved terminal fields across arms."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]

SPECS = [
    ("results/q8_ness_reference_3000/BBest.pt",
     "old champion: W2 + quadratic + K=3/dt=1 (T=3)"),
    ("results/q8_w2_modern_3000/last.pt",
     "W2-modern:    W2 + unified  + K=16/dt=4 (T=64)"),
    ("results/q8_predictive_ports_k64_3000_v2/BBest.pt",
     "v2 step-500:  W4 + quadratic + K=64/dt=1 (T=64)"),
    ("results/q8_predictive_ports_k16_unified_bath_3000/BBest.pt",
     "v4 step-2500: W4 + unified  + K=16/dt=4 (T=64)"),
]

for rel, label in SPECS:
    path = ROOT / rel
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = saved.get("state")
    if state is None:
        print(f"{label:48s} -> no 'state' key; keys: {list(saved.keys())[:8]}")
        continue
    state = state.float()
    spectrum = torch.fft.rfftn(state, dim=(1, 2, 3), norm="ortho")
    mag2 = spectrum.abs().square()
    total = mag2.sum()
    dc = mag2[:, 0, 0, 0].sum()
    flat = state.flatten(1, 3)[0]
    cos = (torch.nn.functional.normalize(flat, dim=-1)
           @ torch.nn.functional.normalize(flat.mean(0), dim=0))
    print(f"{label:48s} DC {100*dc/total:6.2f}% | spatial {100*(1-dc/total):6.2f}% "
          f"| site-cos {cos.mean():.3f}+-{cos.std():.3f}")
