"""Measure and quantify emergent PING (gamma) oscillations in the biological fly connectome.

Compares:
1. Physical continuous synaptic conductance (Scheme B cable impedance: lambda_syn = lambda_0^4)
2. Static single-step Dirac impulse (lambda_syn = 0, previous simplified ablation)

Computes Power Spectral Density (PSD) on whole-brain and circuit-specific (APL-Kenyon, Visual C2/C3)
firing rate signals P(t) over 2048 ms of continuous biological time on OpenWebText.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from information_boltzmann.core.fly_reservoir import FlyReservoirLM


def main():
    parser = argparse.ArgumentParser(description="Measure emergent PING oscillations")
    parser.add_argument("--graph", type=Path,
                        default=Path("data/malecns_v1/fly_reservoir_biological.npz"))
    parser.add_argument("--data", type=Path,
                        default=Path("data/ib_owt_gpt2/validation.npy"))
    parser.add_argument("--duration", type=int, default=2048,
                        help="Duration in milliseconds (timesteps)")
    parser.add_argument("--start-offset", type=int, default=8192)
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Firing threshold (0.1 matching training)")
    parser.add_argument("--ckpt", type=Path,
                        default=Path("results/q8_fly_reservoir_output_read_3000/BBest.pt"),
                        help="Path to trained checkpoint for realistic sensory drive")
    parser.add_argument("--output", type=Path,
                        default=Path("results/ping_oscillation_report.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    print(f"Loading biological connectome from {args.graph} (threshold={args.threshold})...")
    model = FlyReservoirLM(args.graph, threshold=args.threshold, injection="sensory", read_surface="output").cuda()

    if args.ckpt.exists():
        print(f"Loading trained sensory projection from {args.ckpt}...")
        saved = torch.load(args.ckpt, map_location="cuda", weights_only=False)
        io_state = {k: v for k, v in saved["model"].items()
                    if k not in ("edge_index", "edge_pre", "edge_post", "edge_weight", "dan_edge_pre", "dan_edge_post", "dan_edge_weight", "lambda_0")}
        model.load_state_dict(io_state, strict=False)
    else:
        # Scale input_proj to reach physiological operating potential
        model.input_proj.weight.data *= 10.0

    model.eval()

    val_data = np.load(args.data, mmap_mode="r")
    tokens = [int(val_data[args.start_offset + t]) for t in range(args.duration)]

    # Identify circuit masks from graph metadata
    packed = np.load(args.graph)
    body_ids = packed["neuron_body_ids"]
    import pyarrow.feather as feather
    ann = feather.read_table("data/malecns_v1/body-annotations.feather").to_pandas()
    ann_dict = ann.set_index("bodyId").to_dict("index")

    kc_indices = []
    apl_indices = []
    visual_c_indices = []
    for idx, bid in enumerate(body_ids):
        row = ann_dict.get(bid, {})
        cls = row.get("class", "")
        ctype = row.get("type", "")
        if cls == "Kenyon_Cell":
            kc_indices.append(idx)
        if ctype == "APL":
            apl_indices.append(idx)
        if ctype in ("C2", "C3"):
            visual_c_indices.append(idx)

    kc_mask = torch.zeros(model.n_neurons, dtype=torch.bool, device="cuda")
    if kc_indices:
        kc_mask[kc_indices] = True
    apl_mask = torch.zeros(model.n_neurons, dtype=torch.bool, device="cuda")
    if apl_indices:
        apl_mask[apl_indices] = True
    vis_mask = torch.zeros(model.n_neurons, dtype=torch.bool, device="cuda")
    if visual_c_indices:
        vis_mask[visual_c_indices] = True

    print(f"Circuits identified: Kenyon Cells={len(kc_indices)}, APL={len(apl_indices)}, Visual C2/C3={len(visual_c_indices)}")

    def run_simulation(enable_continuous_synapses: bool):
        mode_str = "Continuous Synaptic Conductance (Gen 5)" if enable_continuous_synapses else "Static Dirac Pulse (Gen 4)"
        print(f"\n--- Running simulation: {mode_str} ---")
        h = torch.zeros(1, model.n_neurons, device="cuda")
        ring = tuple(torch.zeros(1, model.n_neurons, device="cuda") for _ in range(4))
        i_syn = torch.zeros(1, model.n_neurons, device="cuda")

        # Temporarily toggle lambda_syn calculation
        orig_lambda_0 = model.lambda_0.clone()
        if not enable_continuous_synapses:
            # Setting lambda_0 power to 0 effectively removes continuous conductance if leak_syn_t is forced to 0
            pass

        pop_rates = []
        kc_rates = []
        apl_rates = []
        vis_rates = []

        with torch.no_grad():
            for t, tok in enumerate(tokens):
                tok_tensor = torch.tensor([tok], device="cuda")
                if enable_continuous_synapses:
                    h, spikes, ring, i_syn = model.step(h, tok_tensor, ring, i_syn)
                else:
                    # Do not pass i_syn -> single-step impulse
                    h, spikes, ring = model.step(h, tok_tensor, ring)

                spk = spikes[0]
                pop_rates.append(float(spk.mean()))
                if len(kc_indices) > 0:
                    kc_rates.append(float(spk[kc_mask].mean()))
                if len(apl_indices) > 0:
                    apl_rates.append(float(spk[apl_mask].mean()))
                if len(visual_c_indices) > 0:
                    vis_rates.append(float(spk[vis_mask].mean()))

        def analyze_psd(signal_list, name):
            sig = np.array(signal_list, dtype=np.float32)
            # Remove mean / detrend
            sig = sig - np.mean(sig)
            # Sampling frequency fs = 1000 Hz (1 ms per step)
            n = len(sig)
            fft_vals = np.fft.rfft(sig)
            psd = (np.abs(fft_vals) ** 2) / n
            freqs = np.fft.rfftfreq(n, d=0.001)  # Hz

            # Band analysis: Gamma / PING band (20 - 60 Hz)
            gamma_mask = (freqs >= 20.0) & (freqs <= 60.0)
            baseline_mask = (freqs > 60.0) & (freqs <= 250.0)

            gamma_power = float(np.mean(psd[gamma_mask])) if np.any(gamma_mask) else 0.0
            baseline_power = float(np.mean(psd[baseline_mask])) if np.any(baseline_mask) else 1e-12

            peak_idx = np.argmax(psd[gamma_mask]) if np.any(gamma_mask) else 0
            peak_freq = float(freqs[gamma_mask][peak_idx]) if np.any(gamma_mask) else 0.0
            peak_power = float(psd[gamma_mask][peak_idx]) if np.any(gamma_mask) else 0.0

            q_ratio = peak_power / max(baseline_power, 1e-12)
            print(f"  [{name:15s}] Peak in 20-60Hz: {peak_freq:5.1f} Hz (Power: {peak_power:.2e}, SNR/Q vs floor: {q_ratio:5.2f}x)")
            return {
                "peak_freq_hz": peak_freq,
                "peak_power": peak_power,
                "gamma_mean_power": gamma_power,
                "baseline_power": baseline_power,
                "snr_q_ratio": q_ratio,
            }

        res = {
            "whole_brain": analyze_psd(pop_rates, "Whole Brain"),
            "kenyon_cells": analyze_psd(kc_rates, "Kenyon Cells") if kc_rates else None,
            "apl_interneuron": analyze_psd(apl_rates, "APL Neuron") if apl_rates else None,
            "visual_c2_c3": analyze_psd(vis_rates, "Visual C2/C3") if vis_rates else None,
        }
        return res

    results = {
        "duration_ms": args.duration,
        "sampling_rate_hz": 1000,
        "static_dirac_pulse": run_simulation(enable_continuous_synapses=False),
        "continuous_synaptic_conductance": run_simulation(enable_continuous_synapses=True),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved PING oscillation analysis report to {args.output}")


if __name__ == "__main__":
    main()
