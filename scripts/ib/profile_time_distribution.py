"""Component-level time distribution of the current main-line configuration."""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core.torus3d import CBIMTorus3D

saved = torch.load(ROOT / "results/q8_predictive_ports_k16_unified_bath_3000/BBest.pt",
                   map_location="cuda", weights_only=False)
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
for module in (model.collision, model.bath, model.write_agent, model.readout):
    module.forward = torch.compile(module.forward)
model.transport.apply_multiplier = torch.compile(model.transport.apply_multiplier)

K, tau = cfg["K"], cfg["tau_0"]
field = saved["state"].cuda().clone()
precision = saved["precision"].cuda().clone()
token = torch.randint(0, 50257, (1,), device="cuda")


def timed(fn, repeat=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


# Interior: one event's worth of microsteps (transport + collision + bath)
def interior_event():
    f = field
    mult, _ = model.transport.multiplier(torch.tensor(tau, device="cuda"))
    for _ in range(K):
        f = model.transport.apply_multiplier(f, mult)
        f, _ = model.collision(f, tau)
        f, _ = model.bath(f, tau)
    return f

t_interior = timed(interior_event)

# Write port: one event
def write_event():
    _, _, _, diag = model.write_agent(model.source, field, token, precision)
    return diag

t_write = timed(write_event)

# Read + decode: one event
def read_event():
    feature, _ = model.readout(field, precision, return_diag=True)
    return model.decoder(feature)

t_read = timed(read_event)

# Full event for cross-check
def full_event():
    return model.step(field, token, precision=precision)

t_full = timed(full_event)

n_tokens = 128
print(f"per-event interior (K={K} microsteps) : {t_interior:7.3f} ms -> x128 = {t_interior*128/1000:.2f} s/update")
print(f"per-event write port                  : {t_write:7.3f} ms -> x128 = {t_write*128/1000:.2f} s/update")
print(f"per-event readout+decoder             : {t_read:7.3f} ms -> x128 = {t_read*128/1000:.2f} s/update")
print(f"per-event full step (fwd only)        : {t_full:7.3f} ms -> x128 = {t_full*128/1000:.2f} s/update")
print(f"production measured update (fwd+bwd)  : 2010 ms; forward share ~= half")
fwd_est = t_interior*128 + t_write*128 + t_read*128
print(f"component forward sum                 : {fwd_est/1000:.2f} s/update -> bwd ~= {2.01 - 2*fwd_est/1000:.2f} s/update")
