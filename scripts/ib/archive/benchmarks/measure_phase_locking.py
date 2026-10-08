"""Measure Phase-Locking Value (PLV) and Perturbation Phase Recovery in the biological fly connectome.

Distinguishes between:
1. True Dynamical Phase-Locking (Attractor dynamics with intrinsic phase-restoring force)
2. Common Input Drive (Feedforward token correlation)
3. Passive Resonance (Uncoupled oscillators)

Formulation:
  PLV_ij = | < exp(i * (phi_i(t) - phi_j(t))) >_t |
Phase extraction via Hilbert transform on continuous regional LFP (mean membrane potential h(t)).
Perturbation recovery test:
  Transient phase kick Delta_h at t_perturb, tracking relaxation of Delta_phi(t) back to equilibrium.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.signal
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from information_boltzmann.core.fly_reservoir import FlyReservoirLM


def extract_instantaneous_phase(signal: np.ndarray, fs: float = 1000.0,
                                lowcut: float = 15.0, highcut: float = 30.0) -> np.ndarray:
    """Extract instantaneous phase phi(t) using bandpass filter + Hilbert transform."""
    sig = signal - np.mean(signal)
    nyq = 0.5 * fs
    b, a = scipy.signal.butter(3, [lowcut / nyq, highcut / nyq], btype="band")
    filtered = scipy.signal.filtfilt(b, a, sig)
    analytic = scipy.signal.hilbert(filtered)
    phase = np.angle(analytic)
    return phase


def compute_plv(phase_i: np.ndarray, phase_j: np.ndarray) -> tuple[float, float]:
    """Compute Phase-Locking Value (PLV) and mean phase angle offset."""
    delta_phi = phase_i - phase_j
    complex_mean = np.mean(np.exp(1j * delta_phi))
    plv = float(np.abs(complex_mean))
    mean_angle = float(np.angle(complex_mean))
    return plv, mean_angle


def main():
    parser = argparse.ArgumentParser(description="Measure cross-circuit phase locking and perturbation recovery")
    parser.add_argument("--graph", type=Path,
                        default=Path("data/malecns_v1/fly_reservoir_biological.npz"))
    parser.add_argument("--data", type=Path,
                        default=Path("data/ib_owt_gpt2/validation.npy"))
    parser.add_argument("--duration", type=int, default=2048,
                        help="Simulation duration in milliseconds")
    parser.add_argument("--start-offset", type=int, default=8192)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--perturb-step", type=int, default=1024,
                        help="Timestep (ms) to inject phase kick")
    parser.add_argument("--perturb-mag", type=float, default=0.2,
                        help="Membrane voltage kick magnitude")
    parser.add_argument("--ckpt", type=Path,
                        default=Path("results/q8_fly_reservoir_continuous_synapse_3000/BBest.pt"),
                        help="Path to trained checkpoint for input projections")
    parser.add_argument("--output", type=Path,
                        default=Path("results/phase_locking_analysis.json"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading biological connectome from {args.graph} on {args.device}...")
    model = FlyReservoirLM(args.graph, threshold=args.threshold,
                           injection="sensory", read_surface="output").to(args.device)
    if args.ckpt and args.ckpt.exists():
        print(f"Loading trained weights from {args.ckpt}...")
        saved = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        io_state = {k: v for k, v in saved["model"].items()
                    if k not in ("edge_index", "edge_pre", "edge_post", "edge_weight",
                                 "dan_edge_pre", "dan_edge_post", "dan_edge_weight", "lambda_0")}
        model.load_state_dict(io_state, strict=False)
    model.eval()

    val_data = np.load(args.data, mmap_mode="r")
    tokens = [int(val_data[args.start_offset + t]) for t in range(args.duration)]

    # Circuit identification from body annotations
    packed = np.load(args.graph)
    body_ids = packed["neuron_body_ids"]
    import pyarrow.feather as feather
    ann = feather.read_table("data/malecns_v1/body-annotations.feather").to_pandas()
    ann_dict = ann.set_index("bodyId").to_dict("index")

    kc_indices = []
    apl_indices = []
    vis_indices = []
    for idx, bid in enumerate(body_ids):
        row = ann_dict.get(bid, {})
        cls = row.get("class", "")
        ctype = row.get("type", "")
        if cls == "Kenyon_Cell":
            kc_indices.append(idx)
        if ctype == "APL":
            apl_indices.append(idx)
        if ctype in ("C2", "C3"):
            vis_indices.append(idx)

    kc_mask = torch.zeros(model.n_neurons, dtype=torch.bool, device=args.device)
    kc_mask[kc_indices] = True
    apl_mask = torch.zeros(model.n_neurons, dtype=torch.bool, device=args.device)
    apl_mask[apl_indices] = True
    vis_mask = torch.zeros(model.n_neurons, dtype=torch.bool, device=args.device)
    vis_mask[vis_indices] = True

    print(f"Circuits: Kenyon Cells={len(kc_indices)}, APL={len(apl_indices)}, Visual C2/C3={len(vis_indices)}")

    def run_trajectory(perturb: bool = False, zero_synapse: bool = False):
        h = torch.zeros(1, model.n_neurons, device=args.device)
        ring = tuple(torch.zeros(1, model.n_neurons, device=args.device) for _ in range(4))
        i_syn = torch.zeros(1, model.n_neurons, device=args.device)

        orig_w = model.edge_weight.clone()
        if zero_synapse:
            model.edge_weight.zero_()

        kc_lfp = []
        apl_lfp = []
        vis_lfp = []
        whole_lfp = []

        with torch.no_grad():
            for t, tok in enumerate(tokens):
                tok_t = torch.tensor([tok], device=args.device)
                h, spk, ring, i_syn = model.step(h, tok_t, ring, i_syn)

                # Transient perturbation kick at t_perturb
                if perturb and t == args.perturb_step:
                    # Depolarizing pulse to Kenyon cells
                    h[0, kc_mask] += args.perturb_mag

                kc_lfp.append(float(h[0, kc_mask].mean()))
                apl_lfp.append(float(h[0, apl_mask].mean()))
                vis_lfp.append(float(h[0, vis_mask].mean()))
                whole_lfp.append(float(h.mean()))

        if zero_synapse:
            model.edge_weight.copy_(orig_w)

        return {
            "kc": np.array(kc_lfp, dtype=np.float32),
            "apl": np.array(apl_lfp, dtype=np.float32),
            "vis": np.array(vis_lfp, dtype=np.float32),
            "whole": np.array(whole_lfp, dtype=np.float32),
        }

    print("\n--- 1. Baseline Simulation (Intact Connectome) ---")
    base_lfp = run_trajectory(perturb=False, zero_synapse=False)
    phase_kc = extract_instantaneous_phase(base_lfp["kc"])
    phase_apl = extract_instantaneous_phase(base_lfp["apl"])
    phase_vis = extract_instantaneous_phase(base_lfp["vis"])

    plv_kc_apl, angle_kc_apl = compute_plv(phase_kc, phase_apl)
    plv_kc_vis, angle_kc_vis = compute_plv(phase_kc, phase_vis)
    plv_apl_vis, angle_apl_vis = compute_plv(phase_apl, phase_vis)

    print(f"  PLV(Kenyon, APL)      = {plv_kc_apl:.4f} (angle offset = {angle_kc_apl:+.3f} rad / {math.degrees(angle_kc_apl):+.1f} deg)")
    print(f"  PLV(Kenyon, Visual)   = {plv_kc_vis:.4f} (angle offset = {angle_kc_vis:+.3f} rad / {math.degrees(angle_kc_vis):+.1f} deg)")
    print(f"  PLV(APL, Visual)      = {plv_apl_vis:.4f} (angle offset = {angle_apl_vis:+.3f} rad / {math.degrees(angle_apl_vis):+.1f} deg)")

    print("\n--- 2. Zero-Transmission Control (Common Input Only, W=0) ---")
    zero_lfp = run_trajectory(perturb=False, zero_synapse=True)
    phase_kc_0 = extract_instantaneous_phase(zero_lfp["kc"])
    phase_apl_0 = extract_instantaneous_phase(zero_lfp["apl"])
    phase_vis_0 = extract_instantaneous_phase(zero_lfp["vis"])

    plv_kc_apl_0, _ = compute_plv(phase_kc_0, phase_apl_0)
    plv_kc_vis_0, _ = compute_plv(phase_kc_0, phase_vis_0)
    print(f"  Zero-W PLV(Kenyon, APL)    = {plv_kc_apl_0:.4f}")
    print(f"  Zero-W PLV(Kenyon, Visual) = {plv_kc_vis_0:.4f}")

    print(f"\n--- 3. Perturbation Phase-Recovery Test (Phase kick at t={args.perturb_step} ms) ---")
    pert_lfp = run_trajectory(perturb=True, zero_synapse=False)
    phase_kc_p = extract_instantaneous_phase(pert_lfp["kc"])
    phase_apl_p = extract_instantaneous_phase(pert_lfp["apl"])

    # Phase difference post-perturbation
    dphi_base = (phase_kc - phase_apl)
    dphi_pert = (phase_kc_p - phase_apl_p)

    # Wrap angles to [-pi, pi]
    def wrap_angle(a):
        return np.arctan2(np.sin(a), np.cos(a))

    phase_error = np.abs(wrap_angle(dphi_pert - dphi_base))

    # Evaluate recovery window: t_perturb to t_perturb + 100 ms
    t0 = args.perturb_step
    window = 100
    err_window = phase_error[t0:t0 + window]
    initial_kick_err = float(err_window[0])
    recovered_err_50ms = float(err_window[min(50, len(err_window)-1)])
    recovered_err_100ms = float(err_window[-1])

    # Fit exponential relaxation: error(t) ~ error(0) * exp(-t / tau_relax)
    valid_t = np.arange(len(err_window))
    valid_mask = err_window > 1e-4
    if np.sum(valid_mask) > 5:
        log_err = np.log(err_window[valid_mask] + 1e-6)
        slope, _ = np.polyfit(valid_t[valid_mask], log_err, 1)
        tau_relax = float(-1.0 / slope) if slope < 0 else float("inf")
    else:
        tau_relax = 0.0

    print(f"  Initial phase error at kick:  {initial_kick_err:.4f} rad ({math.degrees(initial_kick_err):.1f} deg)")
    print(f"  Phase error after 50 ms:      {recovered_err_50ms:.4f} rad ({math.degrees(recovered_err_50ms):.1f} deg)")
    print(f"  Phase error after 100 ms:     {recovered_err_100ms:.4f} rad ({math.degrees(recovered_err_100ms):.1f} deg)")
    print(f"  Relaxation time constant tau: {tau_relax:.2f} ms")

    # Summary conclusions
    conclusions = {
        "is_common_input_driven": bool(plv_kc_vis_0 > 0.5),
        "is_true_phase_locked": bool(plv_kc_vis > 0.6 and plv_kc_vis_0 < 0.3 and math.isfinite(tau_relax) and tau_relax < 80.0),
        "circuit_coupling_delta_plv": float(plv_kc_vis - plv_kc_vis_0),
    }

    report = {
        "duration_ms": args.duration,
        "perturb_step_ms": args.perturb_step,
        "perturb_magnitude_volts": args.perturb_mag,
        "phase_locking_values": {
            "intact_connectome": {
                "plv_kenyon_apl": plv_kc_apl,
                "angle_offset_kenyon_apl_deg": math.degrees(angle_kc_apl),
                "plv_kenyon_visual": plv_kc_vis,
                "angle_offset_kenyon_visual_deg": math.degrees(angle_kc_vis),
                "plv_apl_visual": plv_apl_vis,
                "angle_offset_apl_visual_deg": math.degrees(angle_apl_vis),
            },
            "zero_transmission_control": {
                "plv_kenyon_apl": plv_kc_apl_0,
                "plv_kenyon_visual": plv_kc_vis_0,
            },
        },
        "perturbation_recovery": {
            "initial_phase_error_deg": math.degrees(initial_kick_err),
            "phase_error_50ms_deg": math.degrees(recovered_err_50ms),
            "phase_error_100ms_deg": math.degrees(recovered_err_100ms),
            "relaxation_time_constant_ms": tau_relax,
            "phase_recovered": bool(recovered_err_100ms < 0.5 * initial_kick_err),
        },
        "conclusions": conclusions,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved analysis report to {args.output}")


if __name__ == "__main__":
    main()
