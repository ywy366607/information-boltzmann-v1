"""Deep Resonance Sweep up to T=64 & Full Echo Cycle Saturation until Inflection.

1. Resonance Sweep up to T=64 ticks:
   Periods T = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64].
   Measures steady-state peak-to-peak amplitude, mean, and std across full wavelength.

2. Full Echo Saturation Probe:
   Repeats 16-token natural phrase up to 40 cycles (or until NLL clearly bottoms out and rebounds).
   Tracks NLL, motor norm, ALIF adaptation b, STP vesicle depletion x, and Gamma traces.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:64"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core import fly_reservoir as reservoir
from information_boltzmann.core.fly_bptt_learning import advance_fly_input_event
from scripts.ib.run_full_diagnosis import load_checkpoint, clone_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/active_training/q8_fly_bptt32_gamma_2333_wide115_100k/last.pt"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path,
                        default=Path("results/published/fly_deep_resonance_and_echo.json"))
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint} on {args.device}...", flush=True)
    saved, learner = load_checkpoint(args.checkpoint, device=args.device)
    model = learner.model
    s_init = clone_state(learner.state)

    graph_data = np.load(ROOT / saved["config"]["graph"])
    s_names = list(graph_data["superclass_names"])
    superclass_id = graph_data["superclass_id"]

    motor_indices = np.flatnonzero(np.isin(superclass_id, [s_names.index(c) for c in model.OUTPUT_CLASSES if c in s_names]))
    sensory_indices = model.topographic_writer.injection_index.cpu().numpy()
    intrinsic_indices = np.setdiff1d(np.arange(model.n_neurons), np.union1d(motor_indices, sensory_indices))

    motor_idx_t = torch.as_tensor(motor_indices, device=args.device)
    sensory_idx_t = torch.as_tensor(sensory_indices, device=args.device)

    rates, thresholds = model.get_decay_rates(), model.get_thresholds()
    gains, alif, stp = model.get_conductance_gains(), model.get_alif_params(), model.get_stp_params()

    results = {
        "deep_resonance_sweep": [],
        "full_echo_cycles": [],
    }

    # =========================================================================
    # PART 1: Deep Resonance Sweep up to Period T = 64
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: Deep Resonance Sweep (Periods T = 1..64)")
    print("=" * 70, flush=True)

    periods = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64]
    token_a = torch.tensor([1301], device=args.device)   # baseline
    token_b = torch.tensor([24166], device=args.device)  # impulse

    with torch.no_grad():
        for T in periods:
            state = clone_state(s_init)
            # Run enough ticks: 1 warm-up cycle + 2 full measurement cycles
            warmup_ticks = max(32, T)
            eval_ticks = 2 * T
            total_ticks = warmup_ticks + eval_ticks

            motor_series = []

            for tick in range(total_ticks):
                tok = token_b if (tick % T == 0) else token_a
                state = advance_fly_input_event(
                    model, state, tok, settle_ticks=0, writer_baseline_clock="input",
                    base_rates=rates, thresholds=thresholds, conductance_gains=gains,
                    alif_params=alif, stp_params=stp
                )
                if tick >= warmup_ticks:
                    m_norm = state.h[:, motor_idx_t].norm().item()
                    motor_series.append(m_norm)

            mean_m = float(np.mean(motor_series))
            std_m = float(np.std(motor_series))
            p2p_m = float(np.ptp(motor_series))

            entry = {
                "period_T": T,
                "freq_1_over_T": 1.0 / T,
                "mean_motor_norm": mean_m,
                "std_motor_fluctuation": std_m,
                "peak_to_peak_amplitude": p2p_m,
            }
            results["deep_resonance_sweep"].append(entry)
            print(f"  T = {T:2d} (Freq = {1.0/T:.4f}) | Mean = {mean_m:6.2f} | Fluct Std = {std_m:6.3f} | Peak-to-Peak Amplitude = {p2p_m:6.3f}", flush=True)

    peak_entry = max(results["deep_resonance_sweep"], key=lambda x: x["peak_to_peak_amplitude"])
    min_entry = min(results["deep_resonance_sweep"], key=lambda x: x["peak_to_peak_amplitude"])
    print(f"\nResonance Summary:")
    print(f"  Maximum Amplitude: T = {peak_entry['period_T']} (Amp = {peak_entry['peak_to_peak_amplitude']:.3f})")
    print(f"  Minimum Amplitude: T = {min_entry['period_T']} (Amp = {min_entry['peak_to_peak_amplitude']:.3f})")
    t64_amp = [x for x in results["deep_resonance_sweep"] if x["period_T"] == 64][0]["peak_to_peak_amplitude"]
    print(f"  T=64 Amplitude   : {t64_amp:.3f}")

    # =========================================================================
    # PART 2: Full Echo Saturation & Inflection Point Probe (up to 30 Cycles)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: Full Echo Saturation Probe (Iterating until NLL Inflection)")
    print("=" * 70, flush=True)

    phrase_tokens = [1212, 318, 257, 1332, 284, 1365, 290, 286, 617, 307, 262, 995, 286, 262, 1110, 13]
    phrase = torch.tensor(phrase_tokens, device=args.device)
    phrase_len = len(phrase_tokens)
    max_cycles = 35

    state = clone_state(s_init)
    echo_history = []
    consecutive_rises = 0
    min_nll = float("inf")
    min_cycle = -1

    with torch.no_grad():
        for cycle in range(1, max_cycles + 1):
            losses = []
            motor_norms = []
            b_norms = []
            x_norms = []
            z2_norms = []

            for i in range(phrase_len):
                tok = phrase[i:i+1]
                target = phrase[(i + 1) % phrase_len]
                state = advance_fly_input_event(
                    model, state, tok, settle_ticks=0, writer_baseline_clock="input",
                    base_rates=rates, thresholds=thresholds, conductance_gains=gains,
                    alif_params=alif, stp_params=stp
                )
                feat = model.output_read(state.gamma_z2)
                logits = model.decoder(model.read_norm(feat))
                loss = torch.nn.functional.cross_entropy(logits, target[None]).item()

                losses.append(loss)
                motor_norms.append(state.h[:, motor_idx_t].norm().item())
                if state.b is not None:
                    b_norms.append(state.b.norm().item())
                if state.x is not None:
                    x_norms.append(state.x.norm().item())
                if state.gamma_z2 is not None:
                    z2_norms.append(state.gamma_z2.norm().item())

            cycle_loss = float(np.mean(losses))
            cycle_motor = float(np.mean(motor_norms))
            cycle_b = float(np.mean(b_norms)) if b_norms else 0.0
            cycle_x = float(np.mean(x_norms)) if x_norms else 0.0
            cycle_z2 = float(np.mean(z2_norms)) if z2_norms else 0.0

            if cycle_loss < min_nll:
                min_nll = cycle_loss
                min_cycle = cycle
                consecutive_rises = 0
            else:
                consecutive_rises += 1

            c_info = {
                "cycle": cycle,
                "nll": cycle_loss,
                "delta_from_first": cycle_loss - (echo_history[0]["nll"] if echo_history else cycle_loss),
                "motor_norm": cycle_motor,
                "alif_adaptation_b": cycle_b,
                "stp_vesicle_x": cycle_x,
                "gamma_z2_norm": cycle_z2,
            }
            echo_history.append(c_info)

            indicator = " * MIN" if cycle == min_cycle else f" (+{consecutive_rises})" if consecutive_rises > 0 else ""
            print(f"  Cycle {cycle:2d} | NLL = {cycle_loss:.4f} {indicator:<7} | Motor = {cycle_motor:6.2f} | ALIF b = {cycle_b:6.2f} | STP x = {cycle_x:6.2f} | Z2 = {cycle_z2:6.2f}", flush=True)

            # Stop if NLL clearly rebounded for 5 consecutive cycles after at least 15 cycles
            if cycle >= 20 and consecutive_rises >= 5:
                print(f"\nInflection confirmed: NLL reached minimum at Cycle {min_cycle} ({min_nll:.4f}) and has rebounded for 5 consecutive cycles.", flush=True)
                break

    results["full_echo_cycles"] = echo_history
    results["echo_minimum"] = {
        "cycle": min_cycle,
        "min_nll": min_nll,
        "first_nll": echo_history[0]["nll"],
        "total_drop": echo_history[0]["nll"] - min_nll,
        "rebound": echo_history[-1]["nll"] - min_nll,
    }

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Echo Bottom: Cycle {min_cycle} reached minimum NLL = {min_nll:.4f} (dropped by {echo_history[0]['nll'] - min_nll:.4f} nats from Cycle 1).")
    print(f"Final Cycle {len(echo_history)} NLL = {echo_history[-1]['nll']:.4f} (rebounded by +{echo_history[-1]['nll'] - min_nll:.4f} nats).")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll deep measurements saved to {args.output}!")


if __name__ == "__main__":
    main()
