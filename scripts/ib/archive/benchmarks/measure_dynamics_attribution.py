"""Causal attribution of the interior dynamics: how much NLL do transport,
collision and the bath actually carry, on the W2 champion versus the W4
belief line? Probes the read-side shortcut question directly."""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core.torus3d import CBIMTorus3D, KineticBeliefState
from information_boltzmann.evaluation import WarmSiteSpec
from scripts.ib.evaluate_continuous_owt import _model_from_checkpoint, evaluate_warm_sites

spec = WarmSiteSpec(
    site_starts=(8192, 12288, 16384, 20480), warm_in_tokens=256, score_tokens=128)
validation = np.load(ROOT / "data/ib_owt_gpt2/validation.npy", mmap_mode="r")


def patched_eval(model, belief_mode, micro_steps, label):
    flags_variants = {
        "full": {},
        "-transport": {"disable_transport": True},
        "-collision": {"disable_collision": True},
        "-bath": {"disable_bath": True},
        "no-dynamics": {"disable_transport": True, "disable_collision": True,
                        "disable_bath": True},
    }
    print(f"== {label} (K={micro_steps})")
    for name, flags in flags_variants.items():
        if belief_mode:
            original = model.belief_step

            def wrapper(state, token, _orig=original, **kwargs):
                return _orig(state, token, **{**kwargs, **flags})
            model.belief_step = wrapper
        else:
            original = model.step

            def wrapper(field, token, _orig=original, **kwargs):
                return _orig(field, token, **{**kwargs, **flags})
            model.step = wrapper
        try:
            started = time.perf_counter()
            report = evaluate_warm_sites(model, terminal, validation, spec,
                                         micro_steps, ness_phase_seed=11)
            elapsed = time.perf_counter() - started
            print(f"  {name:>12} | NLL {float(report['nll']):8.4f} | {elapsed:5.1f} s")
        finally:
            if belief_mode:
                model.belief_step = original
            else:
                model.step = original


# --- W2 champion (Continuous-Q8, kernel_r1 readout consuming tok_embed) ---
saved = torch.load(ROOT / "results/q8_ness_reference_3000/BBest.pt",
                   map_location="cuda", weights_only=False)
model, micro_steps, kind = _model_from_checkpoint(saved)
model = model.cuda()
model.load_state_dict(saved["model"])
state = saved.get("state", saved.get("terminal"))
state = state.detach().clone().cuda().float()
model.set_ness_prior(state)
terminal = state
patched_eval(model, False, micro_steps, "W2 champion (kernel readout, token at read)")

# --- W4 belief line (field-only readout, no token at read) ---
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
terminal = KineticBeliefState(
    saved["state"].detach().clone().cuda(), saved["precision"].detach().clone().cuda())
model.set_ness_prior(terminal.field)
patched_eval(model, True, cfg["K"], "W4 belief line (field-only readout)")
