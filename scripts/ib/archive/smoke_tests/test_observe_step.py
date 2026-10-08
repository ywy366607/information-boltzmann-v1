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
labels = torch.as_tensor(np.array(train_data[cursor+1:cursor+width+1], dtype=np.int64), device="cuda")

named = {n: p for n, p in model.named_parameters() if p.requires_grad}
head_keys = {"output_read.weight", "read_norm.weight", "decoder.weight", "decoder.bias", "logit_read_gamma"} & named.keys()
body_keys = named.keys() - head_keys

print("Snapshotting old params...", flush=True)
s0 = clone_state(learner.state)
old_params = snapshot_parameters(named)

print("Calling observe...", flush=True)
with observe_gradient_instrument() as captured:
    scores, metrics = learner.observe(labels)
print(f"Observe done, loss={np.mean(scores):.4f}, grad_norm={metrics['grad_norm_before_clip']:.4f}", flush=True)

print("Snapshotting new params...", flush=True)
new_params = snapshot_parameters(named)

print("Reconstructing raw grads...", flush=True)
clip_factor = min(1.0, learner.max_grad_norm / (metrics["grad_norm_before_clip"] + 1e-6))
raw_grads = {}
for p_id, p_name in {id(p): n for n, p in named.items()}.items():
    if p_id in captured:
        raw_grads[p_name] = (captured[p_id] / clip_factor).detach().cpu()
print("Raw grads reconstructed.", flush=True)

print("Measuring displacement...", flush=True)
body_disp_norm_sq = sum((new_params[n] - old_params[n]).norm().item()**2 for n in body_keys)
body_disp_norm = float(body_disp_norm_sq**0.5)
print(f"Body disp norm: {body_disp_norm:.6f}", flush=True)

print("Measuring grad norm...", flush=True)
body_grad_norm_sq = sum(raw_grads[n].norm().item()**2 for n in body_keys if n in raw_grads)
body_grad_norm = float(body_grad_norm_sq**0.5)
print(f"Body grad norm: {body_grad_norm:.6f}", flush=True)

print("Constructing pure_sg_params...", flush=True)
pure_sg_params = {n: t.clone() for n, t in old_params.items()}
step_scale = body_disp_norm / body_grad_norm
for n in body_keys:
    print(f"Processing param: {n}", flush=True)
    if n in raw_grads:
        print(f"  old shape: {old_params[n].shape}, raw_grad shape: {raw_grads[n].shape}", flush=True)
        pure_sg_params[n] = old_params[n] - step_scale * raw_grads[n]
        print(f"  done {n}", flush=True)
print("pure_sg_params constructed.", flush=True)

print("Copying to model...", flush=True)
torch.cuda.empty_cache()
with torch.no_grad():
    for n, p in named.items():
        print(f"  Copying {n} directly from CPU to CUDA...", flush=True)
        p.copy_(pure_sg_params[n])
    print("  Clamping edges...", flush=True)
    learner.clamp_edges()
    print("  Snapshotting clamped params to CPU...", flush=True)
    pure_sg_clamped = snapshot_parameters(named)
print("Clamping done, pure_sg_clamped ready.", flush=True)
print("TEST COMPLETED SUCCESSFULLY!")
