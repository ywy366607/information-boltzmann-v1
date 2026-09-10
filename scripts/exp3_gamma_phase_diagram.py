"""Experiment 3: Gamma Phase Diagram and Critical Slowing Down (lambda, tau_c, <H>, <A_eff>)."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fine_grain.information_boltzmann import InformationBoltzmann, PhaseState


def compute_autocorrelation_time(signal: np.ndarray, max_lag: int = 50) -> tuple[float, np.ndarray]:
    """Compute normalized autocorrelation rho(tau) and integrated correlation time tau_c."""
    sig = signal - np.mean(signal)
    var = np.var(sig)
    if var < 1e-12:
        return 0.0, np.zeros(max_lag)

    n = len(sig)
    acf = np.correlate(sig, sig, mode="full")[n - 1:n - 1 + max_lag] / (n * var)

    # Integrated correlation time: integrate up to first zero crossing
    tau_c = 0.5  # tau=0 term contributes 0.5 in trapezoidal integration
    for tau in range(1, len(acf)):
        if acf[tau] <= 0:
            break
        tau_c += acf[tau]

    return float(tau_c), acf


def run_gamma_phase_diagram(config_path: Path, tokens_path: Path, gammas: list[float],
                            steps: int = 300, seeds: list[int] = [11, 42, 99],
                            device: str = "cuda") -> dict:
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokens = np.load(tokens_path, mmap_mode="r")[:steps]

    phase_diagram = {}

    for g in gammas:
        gamma_key = f"gamma_{g:.3f}"
        print(f"Sweeping gamma = {g:.3f} across {len(seeds)} seeds...")

        lyapunov_seeds = []
        tau_c_seeds = []
        entropy_seeds = []
        area_seeds = []

        for seed in seeds:
            cfg = json.loads(json.dumps(config))
            cfg["dynamics"]["gamma"] = g
            cfg["dynamics"]["kappa"] = 1.0  # keep spatial trap fixed to prevent particle escape
            cfg["dynamics"]["temperature"] = 0.1
            cfg["dynamics"]["adaptive_gamma"] = False

            torch.manual_seed(seed)
            model = InformationBoltzmann.from_config(cfg).to(device=device, dtype=torch.float64)
            model.eval()

            gen = torch.Generator(device=device).manual_seed(seed)
            state = model.initialize(torch.tensor([1], device=device), gen)

            # Shadow perturbation for Lyapunov exponent
            eps = 1e-5
            pert_x = state.x.clone()
            pert_x[0, 0] += eps
            pert_state = PhaseState(pert_x, state.v.clone(), state.time)

            lyap_rates = []
            trajectory_x = []
            entropies = []
            areas = []

            with torch.no_grad():
                for t in range(steps):
                    tok = int(tokens[t])

                    gen_snapshot = gen.get_state()
                    next_state, _, _ = model.advance(state, tok, gen)

                    # Advance perturbed with identical noise
                    gen.set_state(gen_snapshot)
                    next_pert, _, _ = model.advance(pert_state, tok, gen)

                    # Compute Lyapunov divergence
                    dist = float(torch.cat((next_pert.x-next_state.x, next_pert.v-next_state.v), -1).norm())
                    if dist > 0:
                        lyap_rates.append(math.log(dist / eps) / model.event_interval)
                        pert_state = PhaseState(next_state.x + (next_pert.x-next_state.x)*(eps/dist), next_state.v + (next_pert.v-next_state.v)*(eps/dist), next_state.time)
                    else:
                        raise FloatingPointError("Unresolved perturbation; use FP64/larger epsilon")

                    state = next_state
                    trajectory_x.append(float(state.belief()[0][0].item()))

                    # Covariance and entropy
                    n, d_dim = state.x.shape
                    xv = torch.cat((state.x, state.v), dim=-1)
                    centered = xv - xv.mean(0, keepdim=True)
                    cov = (centered.T @ centered) / n
                    eigvals = torch.linalg.eigvalsh(cov).clamp_min(1e-30)
                    log_det = float(eigvals.log().sum().item())
                    h = 0.5 * (2 * d_dim * (1.0 + math.log(2.0 * math.pi)) + log_det)
                    entropies.append(h)
                    areas.append(math.exp(min(h, 50.0)) if h > -50.0 else 0.0)

            # Compute tau_c from x trajectory (skip initial transient 50 steps)
            steady_sig = np.array(trajectory_x[50:])
            tau_c, _ = compute_autocorrelation_time(steady_sig)

            lyapunov_seeds.append(float(np.mean(lyap_rates[50:])))
            tau_c_seeds.append(tau_c)
            entropy_seeds.append(float(np.mean(entropies[50:])))
            area_seeds.append(float(np.mean(areas[50:])))

        phase_diagram[gamma_key] = {
            "gamma": g,
            "mean_finite_phase_response": float(np.mean(lyapunov_seeds)),
            "std_finite_phase_response": float(np.std(lyapunov_seeds)),
            "mean_tau_c": float(np.mean(tau_c_seeds)),
            "std_tau_c": float(np.std(tau_c_seeds)),
            "mean_gaussian_entropy_proxy": float(np.mean(entropy_seeds)),
            "mean_effective_area": float(np.mean(area_seeds)),
        }

    return phase_diagram


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/information_boltzmann/smoke_owt.json"))
    parser.add_argument("--tokens", type=Path, default=Path("data/ib_owt_smoke/train.npy"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("results/published/gamma_phase_diagram.json"))
    args = parser.parse_args()

    gammas = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.0]
    diagram = run_gamma_phase_diagram(args.config, args.tokens, gammas, steps=args.steps, device=args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(diagram, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 90)
    print("EXPERIMENT 3: GAMMA PHASE DIAGRAM & CRITICAL SLOWING DOWN")
    print("=" * 90)
    print(f"{'Gamma':<10} | {'Lyapunov lambda':<20} | {'Corr Time tau_c':<18} | {'<H>':<12} | {'<A_eff>':<12}")
    print("-" * 90)
    for k, v in diagram.items():
        print(f"{v['gamma']:<10.3f} | {v['mean_finite_phase_response']:<8.4f} +/- {v['std_finite_phase_response']:<8.4f} | {v['mean_tau_c']:<8.2f} +/- {v['std_tau_c']:<6.2f} | {v['mean_gaussian_entropy_proxy']:<12.3f} | {v['mean_effective_area']:<12.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
