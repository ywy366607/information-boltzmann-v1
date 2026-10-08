"""Trace where spatial differentiation dies inside one event:
write gates -> packets -> field before/after write -> per-microstep decay."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core.torus3d import CBIMTorus3D, KineticBeliefState


def dc_share(x):
    F = torch.fft.rfftn(x, dim=(1, 2, 3), norm="ortho")
    m = F.abs().square()
    return float(m[:, 0, 0, 0].sum() / m.sum())


def report(label, x):
    print(f"  {label:44s} DC {100*dc_share(x):6.2f}%  | spatial {100*(1-dc_share(x)):6.2f}%")


print("=========== W4 line (v4 checkpoint: W4 + unified + K=16/dt=4) ===========")
saved = torch.load(ROOT / "results/q8_predictive_ports_k16_unified_bath_3000/BBest.pt",
                   map_location="cuda", weights_only=False)
cfg = saved["config"]
model = CBIMTorus3D(
    shape=tuple(cfg["shape"]), velocities=cfg["velocities"],
    content_dim=cfg["content_dim"], collision_layers=cfg["collision_layers"],
    relative_address=True, readout_type="belief_agent",
    write_type="w4_predictive_agent", micro_steps=cfg["K"],
    dissipation_type="unified", dissipation_rank=cfg["dissipation_rank"],
    tau_0=cfg["tau_0"]).cuda()
model.load_state_dict(saved["model"])
field = saved["state"].cuda().clone()
precision = saved["precision"].cuda().clone()

tokens = np.load(ROOT / "data/ib_owt_gpt2/validation.npy", mmap_mode="r")
token = torch.tensor([int(tokens[8192])], device="cuda")

agent = model.write_agent
import torch.nn.functional as Fn
import math
with torch.no_grad():
    eps = torch.finfo(field.dtype).eps
    field_summary = field.mean((1, 2, 3))
    log_precision = precision.clamp_min(eps).log()
    prior_features = torch.cat((Fn.rms_norm(field_summary, (agent.d,)), log_precision), -1)
    gates = torch.softmax(agent.chart_gate(prior_features), -1)[0]
    print("chart gates over 8 modes (mode0 = DC):")
    print("  " + "  ".join(f"{g:.3f}" for g in gates.tolist()))
    token_features = Fn.normalize(model.source.embedding.weight, dim=-1)
    natural = agent.port_prior(prior_features)
    logits = natural @ token_features.T + agent.port_logit_bias
    probability = torch.softmax(logits, -1)
    expected_feature = probability @ token_features
    observed_packet, _, _, _ = agent._packet_chart(model.source, field, prior_features, token_features[token])
    predicted_packet, _, _, _ = agent._packet_chart(model.source, field, prior_features, expected_feature)
    innovation = observed_packet - predicted_packet

    print("-- packets (write-time differentiation):")
    report("observed packet", observed_packet)
    report("predicted packet", predicted_packet)
    report("innovation (what actually gets written)", innovation)

    field_next, _, _, diag = agent(model.source, field, token, precision)
    print("-- field:")
    report("field BEFORE write", field)
    report("field AFTER write", field_next)

    f = field_next
    mult, _ = model.transport.multiplier(torch.tensor(cfg["tau_0"], device="cuda"))
    for k in range(cfg["K"]):
        f = model.transport.apply_multiplier(f, mult)
        f, _ = model.collision(f, cfg["tau_0"])
        f, _ = model.bath(f, cfg["tau_0"])
        if k in (0, 1, 3, 7, 15):
            report(f"after microstep {k+1:2d} (transport+collision+bath)", f)
    report("field at read time", f)

print()
print("=========== W2 champion (old checkpoint: W2 + quadratic + K=3/dt=1) ===========")
saved2 = torch.load(ROOT / "results/q8_ness_reference_3000/BBest.pt",
                    map_location="cuda", weights_only=False)
cfg2 = saved2["config"]
model2 = CBIMTorus3D(
    shape=tuple(cfg2["shape"]), velocities=cfg2["velocities"],
    content_dim=cfg2["content_dim"], collision_layers=cfg2["collision_layers"],
    relative_address=bool(cfg2.get("relative_address", False)),
    v2_coordinate_components=bool(cfg2.get("v2_coordinate_components", False)),
    readout_type="kernel_r1",
    readout_probes=int(cfg2.get("readout_probes", 8)),
    readout_rounds=int(cfg2.get("readout_rounds", 1)),
    write_type="w2_impedance", micro_steps=cfg2["micro_steps"],
    adaptive_clock=bool(cfg2.get("adaptive_clock", False)),
    alpha_causal=float(cfg2.get("alpha_causal", 0.90)),
    alpha_max=float(cfg2.get("alpha_max", 2.50)),
    continuous_velocities=bool(cfg2.get("continuous_velocities", False)),
    dissipation_type="quadratic").cuda()
model2.load_state_dict(saved2["model"])
field2 = saved2["state"].cuda().clone()
token2 = torch.tensor([int(tokens[8192])], device="cuda")

with torch.no_grad():
    print("-- W2 write (addressed packet, one call):")
    report("field BEFORE write", field2)
    field2_next, _, diag2 = model2.source(field2, token2)
    report("field AFTER write", field2_next)
    print(f"  write_spatial_support (sites with support>0.1): "
          f"{float(diag2['write_spatial_support']):.3f}")
    f2 = field2_next
    for k in range(cfg2["micro_steps"]):
        if cfg2.get("continuous_velocities"):
            dir_k = model2.direction_controller(f2, model2.source.embedding(token2))
            mult2, _ = model2.transport.multiplier(cfg2["micro_steps"], direction=dir_k)
        else:
            mult2, _ = model2.transport.multiplier(cfg2["micro_steps"])
        f2 = model2.transport.apply_multiplier(f2, mult2)
        f2, _ = model2.collision(f2, 1.0)
        f2, _ = model2.bath(f2, 1.0)
        report(f"after microstep {k+1}", f2)
    report("field at read time", f2)
