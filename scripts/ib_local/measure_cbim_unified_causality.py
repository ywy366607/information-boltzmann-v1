"""Frozen single-factor causal interventions for Unified V6 on held-out OWT."""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local.cbim_malecns_unified import CBIMMaleCNSUnifiedV6
from scripts.ib_local.cbim_malecns_internal_time import CBIMMaleCNSInternalTime
from scripts.ib_local.cbim_torus3d import CBIMTorus3D


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--sites", type=int, default=4)
    parser.add_argument("--start", type=int, default=8192)
    args = parser.parse_args()

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    if "Torus3D" in cfg["architecture"] or "ThreeClock" in cfg["architecture"]:
        model = CBIMTorus3D(
            shape=tuple(cfg["shape"]), velocities=cfg["velocities"],
            content_dim=cfg["content_dim"],
            collision_layers=cfg.get("collision_layers", 2),
            relative_address=cfg.get("relative_address", False),
            v2_coordinate_components=cfg.get(
                "v2_coordinate_components",
                "v2" in cfg.get("architecture", "")),
            readout_type=cfg.get("readout_type", "baseline"),
            readout_probes=cfg.get("readout_probes", 8),
            readout_rounds=cfg.get("readout_rounds", 1),
            write_type=cfg.get("write_type", "w0_baseline"),
            micro_steps=cfg.get("micro_steps", 1),
            adaptive_clock=cfg.get("adaptive_clock", False),
            continuous_velocities=cfg.get("continuous_velocities", False),
            alpha_causal=cfg.get("alpha_causal", 0.90),
            alpha_max=cfg.get("alpha_max", 2.50),
            dissipation_type=cfg.get("dissipation_type", "quadratic"),
            dissipation_rank=cfg.get("dissipation_rank", 4),
            three_clock=cfg.get("three_clock", False),
            tau_mem=cfg.get("tau_mem", 3.0),
            nu_s_init=cfg.get("nu_s_init", 0.020))
    elif "unified" in cfg["architecture"]:
        model = CBIMMaleCNSUnifiedV6(
            cfg["graph"], velocities=cfg["velocities"],
            content_dim=cfg["content_dim"], modes=cfg["modes"],
            collision_rank=cfg["collision_rank"])
    else:
        model = CBIMMaleCNSInternalTime(
            cfg["graph"], velocities=cfg["velocities"],
            content_dim=cfg["content_dim"], micro_steps=cfg["micro_steps"],
            checkpoint_tokens=cfg.get("checkpoint_tokens", 1),
            spectral_transport=cfg.get("spectral_transport", False))
    model = model.cuda().eval()
    model.load_state_dict(saved["model"])
    if "state" in saved and not bool(model.has_ness_prior):
        model.set_ness_prior(saved["state"].cuda())
    data = np.load(args.data / "validation.npy", mmap_mode="r")
    window = int(cfg["tokens"])

    def advance(state, start, length, no_collision=False, no_transport=False):
        total = 0.0
        last_diag = {}
        for offset in range(0, length, window):
            x = torch.as_tensor(np.array(data[start + offset:start + offset + window]),
                                dtype=torch.long, device="cuda")[None]
            y = torch.as_tensor(np.array(data[start + offset + 1:start + offset + window + 1]),
                                dtype=torch.long, device="cuda")[None]
            loss, state, diag = model(x, y, state, disable_collision=no_collision,
                                      disable_transport=no_transport)
            total += float(loss) * window
            last_diag = diag
        return state, total / length, last_diag

    records = []
    site_diags = []
    for site in range(args.sites):
        start = args.start + site * 4096
        shared, _, _ = advance(model.initial_state(1, "cuda"), start, args.history)
        values = {}
        full_diag = {}
        for name, nc, nt in (("full", False, False),
                             ("no_collision", True, False),
                             ("no_transport", False, True),
                             ("no_both", True, True)):
            _, values[name], diag = advance(shared.clone(), start + args.history,
                                            args.horizon, nc, nt)
            if name == "full":
                full_diag = diag
        site_diags.append(full_diag)
        records.append({
            "site": site, **values,
            "collision_delta_nll": values["no_collision"] - values["full"],
            "transport_delta_nll": values["no_transport"] - values["full"],
            "joint_delta_nll": values["no_both"] - values["full"],
        })

    diag_summary = {}
    if site_diags and site_diags[0]:
        for k in ["t_packet", "delta_e_field", "reflected_energy", "incident_energy",
                  "cross_interference", "write_to_f_ratio", "write_angle_abs_mean",
                  "collision_angle_abs_mean", "collision_input_snr", "collision_output_snr",
                  "bath_out_energy", "energy", "alpha_1", "alpha_2", "alpha_3",
                  "delta_tau_total", "collision_exposure"]:
            vals = [float(d[k]) for d in site_diags if k in d]
            if vals:
                diag_summary[k] = float(np.mean(vals))

    report = {
        "checkpoint": str(args.checkpoint), "step": saved["step"],
        "protocol": {"dataset": "OWT validation", "sites": args.sites,
                     "shared_history": args.history, "horizon": args.horizon},
        "sites": records,
        "mean_full_nll": float(np.mean([r["full"] for r in records])),
        "mean_collision_delta_nll": float(np.mean([r["collision_delta_nll"] for r in records])),
        "mean_transport_delta_nll": float(np.mean([r["transport_delta_nll"] for r in records])),
        "mean_joint_delta_nll": float(np.mean([r["joint_delta_nll"] for r in records])),
        "physical_diagnostics": diag_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
