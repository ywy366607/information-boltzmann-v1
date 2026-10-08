import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ib.diagnose_fly_surrogate_vs_hard import (
    load_checkpoint,
    snapshot_parameters,
    observe_gradient_instrument,
    clone_state,
)

print("Loading checkpoint...", flush=True)
saved, learner = load_checkpoint(Path("artifacts/active_training/q8_fly_bptt32_gamma_2333_wide115_100k/last.pt"), device="cuda")
cfg, model = saved["config"], learner.model
cursor, width = saved["train_cursor"], cfg["window"]

train_data = np.load(ROOT / cfg["data"] / "train.npy", mmap_mode="r")
named = {n: p for n, p in model.named_parameters() if p.requires_grad}

for it in range(2):
    print(f"\n=== ITERATION {it+1} ===", flush=True)
    offset = cursor + it * width
    labels = torch.as_tensor(np.array(train_data[offset+1:offset+width+1], dtype=np.int64), device="cuda")

    s0 = clone_state(learner.state)
    old_params = snapshot_parameters(named)

    print("Calling observe...", flush=True)
    with observe_gradient_instrument() as captured:
        scores, metrics = learner.observe(labels)
    print(f"Observe done, loss={np.mean(scores):.4f}, grad_norm={metrics['grad_norm_before_clip']:.4f}", flush=True)

    print("Testing clone_state(learner.state)...", flush=True)
    live_s1 = clone_state(learner.state)
    print("clone_state succeeded!", flush=True)

    print("Testing snapshot_parameters(named)...", flush=True)
    new_params = snapshot_parameters(named)
    print("snapshot_parameters succeeded!", flush=True)

print("TWO ITERATIONS SUCCEEDED!")
