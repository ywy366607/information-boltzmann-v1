import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ib.diagnose_fly_surrogate_vs_hard import (
    load_checkpoint,
    snapshot_parameters,
    restore_parameters,
    observe_gradient_instrument,
    spike_instrument,
    branch_distance,
    clone_state,
)

print("Loading checkpoint...", flush=True)
saved, learner = load_checkpoint(Path("artifacts/active_training/q8_fly_bptt32_gamma_2333_wide115_100k/last.pt"), device="cuda")
cfg, model = saved["config"], learner.model
cursor, width = saved["train_cursor"], cfg["window"]
train_data = np.load(ROOT / cfg["data"] / "train.npy", mmap_mode="r")
named = {n: p for n, p in model.named_parameters() if p.requires_grad}

head_keys = {"output_read.weight", "read_norm.weight", "decoder.weight", "decoder.bias", "logit_read_gamma"} & named.keys()
body_keys = named.keys() - head_keys

def forward_eval(state, ids, targets, fixed_spikes=None):
    learner.state = state
    with torch.no_grad(), spike_instrument(fixed_spikes) as rec:
        scores, terminal, _ = learner.forward_window(ids, targets[None])
    return scores.detach().cpu().tolist(), clone_state(terminal), rec

for it in range(2):
    print(f"\n=== ITERATION {it+1} ===", flush=True)
    offset = cursor + it * width
    labels = torch.as_tensor(np.array(train_data[offset+1:offset+width+1], dtype=np.int64), device="cuda")
    fresh = torch.as_tensor(np.array(train_data[offset+width+1:offset+2*width+1], dtype=np.int64), device="cuda")
    ids = torch.cat((labels.new_tensor([learner.previous_token]), labels[:-1]))[None]
    fresh_ids = torch.cat((labels[-1:], fresh[:-1]))[None]

    s0 = clone_state(learner.state)
    old_params = snapshot_parameters(named)

    base_fit, s1, ref_fit_spk = forward_eval(s0, ids, labels)
    base_fol, _, ref_fol_spk = forward_eval(s1, fresh_ids, fresh)
    print(f"Base Fit: {np.mean(base_fit):.4f}, Base Follow: {np.mean(base_fol):.4f}", flush=True)

    print("Calling observe...", flush=True)
    learner.state = s0
    learner.runner = None
    with observe_gradient_instrument() as captured:
        scores, metrics = learner.observe(labels)
    print(f"Observe done, loss={np.mean(scores):.4f}, grad_norm={metrics['grad_norm_before_clip']:.4f}", flush=True)

    live_s1 = clone_state(learner.state)
    new_params = snapshot_parameters(named)
    print("Snapshots taken.", flush=True)

    # Evaluate Full Adam control
    print("Evaluating Full Adam...", flush=True)
    restore_parameters(named, new_params)
    f_fit, _, _ = forward_eval(s0, ids, labels)
    print(f"Full Adam Fit NLL: {np.mean(f_fit):.4f}", flush=True)

    # Advance live state
    print("Advancing live state...", flush=True)
    restore_parameters(named, new_params)
    learner.state = live_s1

    torch.cuda.empty_cache()
    print(f"End of iteration {it+1}", flush=True)

print("SUCCESSFULLY RAN BOTH ITERATIONS!")
