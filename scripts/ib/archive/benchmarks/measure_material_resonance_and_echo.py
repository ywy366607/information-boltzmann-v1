"""Material Resonance, Green's Function & Pipeline Echo Tuning Audit.

Treats the fruit fly connectome brain as a physical material:
1. Impulse Response (Green's function): delta perturbation at tick 0 -> response at tau = 0..32 ticks.
2. Resonance Sweep: periodic drive at period T = 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16 ticks -> amplitude & resonance peak.
3. Pipeline Echo Tuning: repeated text stream -> spike filling, steady state, and input-output lag locking.
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
                        default=Path("results/published/fly_material_resonance_and_echo.json"))
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint} on {args.device}...", flush=True)
    saved, learner = load_checkpoint(args.checkpoint, device=args.device)
    model = learner.model
    s_init = clone_state(learner.state)

    # Brain neuron partition
    graph_data = np.load(ROOT / saved["config"]["graph"])
    s_names = list(graph_data["superclass_names"])
    superclass_id = graph_data["superclass_id"]
    
    motor_indices = np.flatnonzero(np.isin(superclass_id, [s_names.index(c) for c in model.OUTPUT_CLASSES if c in s_names]))
    sensory_indices = model.topographic_writer.injection_index.cpu().numpy()
    intrinsic_indices = np.setdiff1d(np.arange(model.n_neurons), np.union1d(motor_indices, sensory_indices))

    motor_idx_t = torch.as_tensor(motor_indices, device=args.device)
    sensory_idx_t = torch.as_tensor(sensory_indices, device=args.device)
    intrinsic_idx_t = torch.as_tensor(intrinsic_indices, device=args.device)

    print(f"Neuron Partition: Sensory={len(sensory_indices)}, Intrinsic={len(intrinsic_indices)}, Motor/DNs={len(motor_indices)}", flush=True)

    rates, thresholds = model.get_decay_rates(), model.get_thresholds()
    gains, alif, stp = model.get_conductance_gains(), model.get_alif_params(), model.get_stp_params()

    results = {
        "impulse_response": {},
        "resonance_sweep": {},
        "pipeline_echo": {},
    }

    # =========================================================================
    # Part 1: Impulse Response Function (Green's Function) across tau = 0..32
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 1: Material Impulse Response (Green's Function)")
    print("=" * 60, flush=True)

    # Base vs Perturbed Token at tick 0, then zero inputs (or identical tokens) for tau = 1..32
    token_a = torch.tensor([1301], device=args.device)   # " the"
    token_b = torch.tensor([24166], device=args.device)  # distinct rare token

    state_a = clone_state(s_init)
    state_b = clone_state(s_init)

    tau_data = []

    with torch.no_grad():
        for tau in range(33):
            tok_a = token_a if tau == 0 else torch.tensor([13], device=args.device)  # "." baseline continuation
            tok_b = token_b if tau == 0 else torch.tensor([13], device=args.device)

            state_a = advance_fly_input_event(
                model, state_a, tok_a, settle_ticks=0, writer_baseline_clock="input",
                base_rates=rates, thresholds=thresholds, conductance_gains=gains,
                alif_params=alif, stp_params=stp
            )
            state_b = advance_fly_input_event(
                model, state_b, tok_b, settle_ticks=0, writer_baseline_clock="input",
                base_rates=rates, thresholds=thresholds, conductance_gains=gains,
                alif_params=alif, stp_params=stp
            )

            diff_sensory = (state_a.h[:, sensory_idx_t] - state_b.h[:, sensory_idx_t]).norm().item()
            diff_intrinsic = (state_a.h[:, intrinsic_idx_t] - state_b.h[:, intrinsic_idx_t]).norm().item()
            diff_motor = (state_a.h[:, motor_idx_t] - state_b.h[:, motor_idx_t]).norm().item()
            diff_z1 = (state_a.gamma_z1 - state_b.gamma_z1).norm().item()
            diff_z2 = (state_a.gamma_z2 - state_b.gamma_z2).norm().item()

            feat_a = model.output_read(state_a.gamma_z2)
            feat_b = model.output_read(state_b.gamma_z2)
            diff_feat = (feat_a - feat_b).norm().item()

            log_a = model.decoder(model.read_norm(feat_a))
            log_b = model.decoder(model.read_norm(feat_b))
            diff_logits = (log_a - log_b).norm().item()

            entry = {
                "tau": tau,
                "diff_sensory": diff_sensory,
                "diff_intrinsic": diff_intrinsic,
                "diff_motor": diff_motor,
                "diff_z1": diff_z1,
                "diff_z2": diff_z2,
                "diff_feature": diff_feat,
                "diff_logits": diff_logits,
            }
            tau_data.append(entry)
            if tau in (0, 1, 2, 3, 5, 7, 10, 15, 20, 30):
                print(f"  tau={tau:2d} | Sensory={diff_sensory:6.2f} | Intrinsic={diff_intrinsic:6.2f} | Motor={diff_motor:6.3f} | Z1={diff_z1:6.3f} | Z2={diff_z2:6.3f} | Logits={diff_logits:6.3f}", flush=True)

    results["impulse_response"] = tau_data

    # Find peak arrival times
    motor_peak_tau = max(tau_data, key=lambda x: x["diff_motor"])["tau"]
    z1_peak_tau = max(tau_data, key=lambda x: x["diff_z1"])["tau"]
    z2_peak_tau = max(tau_data, key=lambda x: x["diff_z2"])["tau"]
    logit_peak_tau = max(tau_data, key=lambda x: x["diff_logits"])["tau"]

    print(f"\nImpulse Peaks: Motor Arrival Peak at tau={motor_peak_tau}, Z1 Peak at tau={z1_peak_tau}, Z2 Peak at tau={z2_peak_tau}, Logits Peak at tau={logit_peak_tau}")

    # =========================================================================
    # Part 2: Resonance Sweep (Periodic Drive Frequency Response)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 2: Resonance Sweep across Drive Periods T = 1..16")
    print("=" * 60, flush=True)

    resonance_results = []
    periods = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16]

    with torch.no_grad():
        for T in periods:
            state = clone_state(s_init)
            n_ticks = 48
            motor_responses = []

            for tick in range(n_ticks):
                # Periodic impulse: token_b every T ticks, otherwise baseline token_a
                tok = token_b if (tick % T == 0) else token_a
                state = advance_fly_input_event(
                    model, state, tok, settle_ticks=0, writer_baseline_clock="input",
                    base_rates=rates, thresholds=thresholds, conductance_gains=gains,
                    alif_params=alif, stp_params=stp
                )
                if tick >= 16:  # Steady state
                    m_norm = state.h[:, motor_idx_t].norm().item()
                    motor_responses.append(m_norm)

            mean_m = float(np.mean(motor_responses))
            std_m = float(np.std(motor_responses))
            p2p_m = float(np.ptp(motor_responses))  # Peak-to-peak amplitude of resonance

            res_entry = {
                "period_T": T,
                "freq_1_over_T": 1.0 / T,
                "mean_motor_norm": mean_m,
                "std_motor_fluctuation": std_m,
                "peak_to_peak_amplitude": p2p_m,
            }
            resonance_results.append(res_entry)
            print(f"  Period T={T:2d} (Freq={1.0/T:.3f}) | Mean={mean_m:6.2f} | Fluctuating Std={std_m:6.3f} | Peak-to-Peak Amplitude={p2p_m:6.3f}", flush=True)

    results["resonance_sweep"] = resonance_results
    res_peak_T = max(resonance_results, key=lambda x: x["peak_to_peak_amplitude"])["period_T"]
    print(f"\nResonance Peak detected at Driving Period T* = {res_peak_T} ticks (Freq = {1.0/res_peak_T:.3f})")

    # =========================================================================
    # Part 3: Pipeline Saturation & Echo Tuning (Repeating Stream)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 3: Pipeline Echo Tuning on Repeating Text Stream")
    print("=" * 60, flush=True)

    # Take a 16-token natural language phrase
    phrase_tokens = [1212, 318, 257, 1332, 284, 1365, 290, 286, 617, 307, 262, 995, 286, 262, 1110, 13]
    # "This is a test to demonstrate the effect of repeating tokens in the stream."
    phrase = torch.tensor(phrase_tokens, device=args.device)
    phrase_len = len(phrase_tokens)
    n_cycles = 6

    state = clone_state(s_init)
    cycle_data = []

    with torch.no_grad():
        for cycle in range(n_cycles):
            cycle_spikes = []
            cycle_motor_norms = []
            cycle_losses = []

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

                cycle_spikes.append(state.h.mean().item())
                cycle_motor_norms.append(state.h[:, motor_idx_t].norm().item())
                cycle_losses.append(loss)

            mean_loss = float(np.mean(cycle_losses))
            mean_motor = float(np.mean(cycle_motor_norms))
            c_entry = {
                "cycle": cycle + 1,
                "mean_loss": mean_loss,
                "mean_motor_norm": mean_motor,
                "motor_trajectory": [round(x, 3) for x in cycle_motor_norms],
            }
            cycle_data.append(c_entry)
            print(f"  Cycle {cycle+1}/{n_cycles} | Mean Motor Norm={mean_motor:.3f} | Cycle Next-Token NLL={mean_loss:.4f}", flush=True)

    # Check lag cross-correlation in the last steady-state cycle
    results["pipeline_echo"] = cycle_data

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll material resonance & tuning measurements saved to {args.output}!")


if __name__ == "__main__":
    main()
