"""Experiment 2: Verify moment-by-moment and long-term energy budget time series in NESS."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fine_grain.information_boltzmann import InformationBoltzmann, PhaseState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/information_boltzmann/ib_owt_2500_champion.pt"))
    parser.add_argument("--tokens", type=Path, default=Path("data/ib_owt_smoke/train.npy"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("results/published/power_budget_timeseries.json"))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = saved["metadata"]["config"]

    model = InformationBoltzmann.from_config(config).to(device)
    model.load_state_dict(saved["model"])
    model.adaptive_gamma = False
    model.force.gamma = float(saved.get("force_gamma", model.force.gamma))
    model.force.kappa = float(saved.get("force_kappa", model.force.kappa))
    model.eval()

    tokens = np.load(args.tokens, mmap_mode="r")[:args.steps]
    generator = torch.Generator(device=device).manual_seed(42)
    state = model.initialize(torch.tensor([1], device=device), generator)

    gamma = model.force.gamma
    kappa = model.force.kappa
    temp = model.force.temperature
    dim = model.force.net[-1].out_features
    dt_event = model.event_interval
    p_bath = gamma * dim * temp

    p_in_series = []
    p_diss_series = []
    p_bath_series = [p_bath] * len(tokens)
    de_dt_series = []
    e_total_series = []
    residual_series = []

    def calc_energy(st: PhaseState) -> float:
        ke = float(st.moments()["kinetic_energy"].item())
        pe = 0.5 * kappa * float(st.moments()["position_second_moment"].item())
        return ke + pe

    curr_e = calc_energy(state)
    e_total_series.append(curr_e)

    with torch.no_grad():
        for t, tok_val in enumerate(tokens):
            tok = int(tok_val)

            # Measure P_in before step
            b_theta = model.force.drive(state.x, tok, state.time)
            p_in = float((state.v * b_theta).sum(-1).mean().item())
            p_in_series.append(p_in)

            # Advance state by one token event
            next_state, _, _ = model.advance(state, tok, generator)

            # Measure dissipation P_diss at step mid-point
            ke_next = float(next_state.moments()["kinetic_energy"].item())
            p_diss = 2.0 * gamma * ke_next
            p_diss_series.append(p_diss)

            # Measure dE/dt
            next_e = calc_energy(next_state)
            e_total_series.append(next_e)
            de_dt = (next_e - curr_e) / dt_event
            de_dt_series.append(de_dt)

            # Instantaneous power balance residual: dE/dt - (P_in + P_bath - P_diss)
            predicted_rate = p_in + p_bath - p_diss
            residual_series.append(de_dt - predicted_rate)

            curr_e = next_e
            state = next_state

    # Statistical summaries
    mean_de_dt = float(np.mean(de_dt_series))
    mean_p_in = float(np.mean(p_in_series))
    mean_p_diss = float(np.mean(p_diss_series))
    mean_p_bath = float(np.mean(p_bath_series))
    net_flux_mean = mean_p_in + mean_p_bath - mean_p_diss

    summary = {
        "num_events": args.steps,
        "parameters": {
            "gamma": gamma,
            "kappa": kappa,
            "temperature": temp,
            "phase_dim": dim,
            "dt_event": dt_event,
        },
        "long_term_averages": {
            "mean_dE_dt": mean_de_dt,
            "mean_P_in": mean_p_in,
            "mean_P_bath": mean_p_bath,
            "mean_P_diss": mean_p_diss,
            "net_predicted_flux": net_flux_mean,
            "power_balance_residual": abs(mean_de_dt-net_flux_mean),
            "energy_drift_magnitude": abs(mean_de_dt),
            "is_true_NESS": None,
            "status": "endpoint_power_estimates_only_requires_substep_integration_and_stationarity_tests",
        },
        "fluctuations": {
            "std_dE_dt": float(np.std(de_dt_series)),
            "std_P_in": float(np.std(p_in_series)),
            "std_P_diss": float(np.std(p_diss_series)),
            "mean_instantaneous_residual": float(np.mean(residual_series)),
            "std_instantaneous_residual": float(np.std(residual_series)),
        },
        "time_series_snapshot_100": {
            "p_in": p_in_series[:100],
            "p_diss": p_diss_series[:100],
            "dE_dt": de_dt_series[:100],
        }
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print("EXPERIMENT 2: ENERGY & POWER BUDGET AUDIT IN NESS")
    print("=" * 80)
    print(f"Total Steps Audited:       {args.steps}")
    print(f"Mean Token Input Power:    P_in   = {mean_p_in:+.6f}")
    print(f"Thermal Bath Power:        P_bath = {mean_p_bath:+.6f}")
    print(f"Damping Dissipation Power: P_diss = {mean_p_diss:+.6f}")
    print(f"Net Energy Influx Rate:    Flux   = {net_flux_mean:+.6f}")
    print(f"Long-term Energy Drift:    <dE/dt>= {mean_de_dt:+.6f}")
    # print(f"Steady-State Criterion:    |dE/dt| < 0.05 => {'PASSED (TRUE NESS)' if summary['long_term_averages']['is_true_NESS'] else 'FAILED'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
