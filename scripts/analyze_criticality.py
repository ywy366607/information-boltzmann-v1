"""Professional criticality analysis: power balance, avalanches, Lyapunov, and 1/f noise."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import scipy.stats as stats
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fine_grain.information_boltzmann import InformationBoltzmann, PhaseState


def compute_criticality(checkpoint_path: Path, tokens_path: Path, device: str = "cuda",
                        num_events: int = 2000):
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = saved["metadata"]["config"]

    model = InformationBoltzmann.from_config(config).to(device=device, dtype=torch.float64)
    model.load_state_dict(saved["model"])
    model.adaptive_gamma = False
    model.force.gamma = float(saved.get("force_gamma", model.force.gamma))
    model.force.kappa = float(saved.get("force_kappa", model.force.kappa))
    model.eval()

    tokens = np.load(tokens_path, mmap_mode="r")[:num_events]
    generator = torch.Generator(device=device).manual_seed(42)
    state = model.initialize(torch.tensor([1], device=device), generator)

    # Trackers for diagnostics
    kinetic_energies = []
    position_variances = []
    p_in_list = []
    p_diss_list = []
    collision_counts = []
    energy_jumps = []
    trajectory_x = []

    # Benettin algorithm for Lyapunov exponent
    eps = 1e-5
    pert_state = PhaseState(state.x.clone(), state.v.clone())
    pert_state.x[0, 0] += eps  # small perturbation to particle 0
    lyapunov_rates = []

    prev_ke = float(state.moments()["kinetic_energy"].detach())

    gamma = model.force.gamma
    kappa = model.force.kappa
    temp = model.force.temperature
    dim = model.force.net[-1].out_features
    p_th = gamma * dim * temp

    print(f"Running criticality diagnostics over {num_events} events...")
    with torch.no_grad():
        for i, token in enumerate(tokens):
            tok = int(token)

            # Record power input and dissipation before advance
            # P_in = 1/N sum v_i . b_theta(x_i)
            b_theta = model.force.drive(state.x, tok, state.time)
            p_in = float((state.v * b_theta).sum(-1).mean().detach())
            p_in_list.append(p_in)

            # Advance base state, capturing generator state so perturbed trajectory
            # experiences the identical realization of thermal noise and collision candidates
            gen_state_before = generator.get_state()
            next_state, _, stats_dict = model.advance(state, tok, generator)

            ke = float(next_state.moments()["kinetic_energy"].detach())
            p_diss = 2.0 * gamma * ke
            p_diss_list.append(p_diss)

            delta_e = abs(ke - prev_ke)
            energy_jumps.append(delta_e)
            prev_ke = ke

            kinetic_energies.append(ke)
            position_variances.append(float(next_state.belief()[1].trace().detach()))
            collision_counts.append(stats_dict["accepted"])
            trajectory_x.append(float(next_state.belief()[0][0].detach()))

            # Advance perturbed state under the IDENTICAL noise realization
            generator.set_state(gen_state_before)
            next_pert, _, _ = model.advance(pert_state, tok, generator)

            # Compute divergence
            dist = float(torch.cat((next_pert.x-next_state.x, next_pert.v-next_state.v), -1).norm().detach())
            if dist > 0:
                rate = np.log(dist / eps) / model.event_interval  # per-event growth
                lyapunov_rates.append(rate)
                # Renormalize perturbation back to eps
                pert_state = PhaseState(next_state.x+(next_pert.x-next_state.x)*(eps/dist), next_state.v+(next_pert.v-next_state.v)*(eps/dist), next_state.time)
            else:
                raise FloatingPointError("Unresolved perturbation; use FP64/larger epsilon")

            state = next_state

    # 1. Power Balance
    mean_p_in = float(np.mean(p_in_list))
    mean_p_diss = float(np.mean(p_diss_list))
    balance_error = abs(mean_p_in + p_th - mean_p_diss)

    # 2. Lyapunov Exponent (Average rate)
    mean_lyapunov = float(np.mean(lyapunov_rates))
    std_lyapunov = float(np.std(lyapunov_rates))

    # 3. Avalanche Distribution Analysis (Energy jumps Delta E)
    jumps = np.array(energy_jumps)[1:]
    jumps = jumps[jumps > 1e-6]

    # Log-log linear fit for power-law exponent: log P(S) = -alpha log S + c
    hist, bin_edges = np.histogram(jumps, bins=30, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    valid = (hist > 0) & (bin_centers > 0)
    log_s = np.log10(bin_centers[valid])
    log_p = np.log10(hist[valid])

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_s, log_p)
    power_law_alpha = -slope
    r_squared = r_value ** 2

    # 4. 1/f Spectral Analysis (Fourier Transform of x trajectory)
    x_signal = np.array(trajectory_x) - np.mean(trajectory_x)
    fft_vals = np.fft.rfft(x_signal)
    psd = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(len(x_signal))

    # Fit PSD slope in intermediate frequency band: S(f) ~ 1/f^beta
    fit_band = (freqs > 0.01) & (freqs < 0.3)
    if fit_band.sum() > 5:
        log_f = np.log10(freqs[fit_band])
        log_psd = np.log10(psd[fit_band])
        psd_slope, _, psd_r, _, _ = stats.linregress(log_f, log_psd)
        spectral_beta = -psd_slope
        spectral_r2 = psd_r ** 2
    else:
        spectral_beta, spectral_r2 = 0.0, 0.0

    return {
        "num_events": num_events,
        "power_budget": {
            "mean_token_input_power_P_in": mean_p_in,
            "thermal_bath_power_P_th": p_th,
            "mean_dissipation_power_P_diss": mean_p_diss,
            "power_balance_residual": balance_error,
            "information_power_ratio": mean_p_in / (mean_p_diss - p_th + 1e-12),
        },
        "lyapunov_analysis": {
            "finite_full_phase_response_rate": mean_lyapunov,
            "lyapunov_fluctuation_std": std_lyapunov,
            "regime": "unclassified_requires_epsilon_step_and_seed_convergence",
        },
        "avalanche_criticality": {
            "power_law_exponent_alpha": power_law_alpha,
            "power_law_fit_R2": r_squared,
            "standard_error": std_err,
            "is_power_law": None,
        },
        "spectral_1_over_f_noise": {
            "spectral_exponent_beta": spectral_beta,
            "spectral_fit_R2": spectral_r2,
            "is_1_over_f_scale_free": None,
        },
        "phase_moments": {
            "mean_position_variance": float(np.mean(position_variances)),
            "mean_kinetic_energy": float(np.mean(kinetic_energies)),
            "total_accepted_collisions": sum(collision_counts),
        }
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--events", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("results/criticality_report.json"))
    args = parser.parse_args()

    results = compute_criticality(args.checkpoint, args.tokens, args.device, args.events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
