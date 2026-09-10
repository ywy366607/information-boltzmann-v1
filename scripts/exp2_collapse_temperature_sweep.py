"""Experiment 2: Temperature Collapse Sweep proving A_eff -> 0 for T=0 and A_eff -> A* > 0 for T>0."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fine_grain.information_boltzmann import InformationBoltzmann, PhaseState


def compute_effective_area_and_entropy(state: PhaseState) -> tuple[float, float]:
    """Compute phase covariance determinant, differential entropy H, and effective area A_eff."""
    n, d = state.x.shape
    xv = torch.cat((state.x, state.v), dim=-1)  # [N, 2*d]
    centered = xv - xv.mean(0, keepdim=True)
    cov = (centered.T @ centered) / n  # [2d, 2d]

    eigvals = torch.linalg.eigvalsh(cov).clamp_min(1e-30)
    log_det = float(eigvals.log().sum().item())

    dim_total = 2 * d
    # Differential entropy: H = 0.5 * (dim * (1 + ln(2*pi)) + ln(det(cov)))
    h = 0.5 * (dim_total * (1.0 + math.log(2.0 * math.pi)) + log_det)

    if h < -50.0:
        a_eff = 0.0
    else:
        a_eff = math.exp(min(h, 50.0))

    return h, a_eff


def run_temperature_sweep(config_path: Path, tokens_path: Path, temps: list[float],
                          steps: int = 128, seeds: list[int] = [11, 42, 99],
                          device: str = "cuda") -> dict:
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokens = np.load(tokens_path, mmap_mode="r")[:steps]

    sweep_results = {}

    for temp in temps:
        temp_key = f"T_{temp:.2f}"
        seed_runs = []

        for seed in seeds:
            cfg = json.loads(json.dumps(config))
            cfg["dynamics"]["temperature"] = temp
            cfg["dynamics"]["gamma"] = 1.0
            cfg["dynamics"]["kappa"] = 1.0
            cfg["dynamics"]["adaptive_gamma"] = False

            model = InformationBoltzmann.from_config(cfg).to(device)
            model.eval()

            gen = torch.Generator(device=device).manual_seed(seed)
            state = model.initialize(torch.tensor([1], device=device), gen)

            history = []
            with torch.no_grad():
                for t in range(steps):
                    tok = int(tokens[t])
                    state, _, _ = model.advance(state, tok, gen)

                    h, a_eff = compute_effective_area_and_entropy(state)
                    var_x = float(state.moments()["position_variance"].item())
                    var_v = float(state.moments()["velocity_variance"].item())

                    if (t + 1) % 4 == 0 or t == 0:
                        history.append({
                            "step": t + 1,
                            "entropy_H": h,
                            "a_eff": a_eff,
                            "var_x": var_x,
                            "var_v": var_v,
                        })

            seed_runs.append(history)

        # Average across seeds
        avg_history = []
        for i in range(len(seed_runs[0])):
            step = seed_runs[0][i]["step"]
            avg_h = float(np.mean([sr[i]["entropy_H"] for sr in seed_runs]))
            avg_a = float(np.mean([sr[i]["a_eff"] for sr in seed_runs]))
            avg_vx = float(np.mean([sr[i]["var_x"] for sr in seed_runs]))
            avg_vv = float(np.mean([sr[i]["var_v"] for sr in seed_runs]))
            avg_history.append({
                "step": step,
                "mean_entropy_H": avg_h,
                "mean_a_eff": avg_a,
                "mean_var_x": avg_vx,
                "mean_var_v": avg_vv,
            })

        sweep_results[temp_key] = {
            "temperature": temp,
            "initial_a_eff": avg_history[0]["mean_a_eff"],
            "final_a_eff": avg_history[-1]["mean_a_eff"],
            "a_eff_ratio": avg_history[-1]["mean_a_eff"] / max(1e-30, avg_history[0]["mean_a_eff"]),
            "is_collapsed": bool(avg_history[-1]["mean_a_eff"] < 1e-6),
            "history": avg_history,
        }

    return sweep_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/information_boltzmann/smoke_owt.json"))
    parser.add_argument("--tokens", type=Path, default=Path("data/ib_owt_smoke/train.npy"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("results/published/collapse_temperature_sweep.json"))
    args = parser.parse_args()

    temps = [0.0, 0.01, 0.05, 0.1, 0.2]
    results = run_temperature_sweep(args.config, args.tokens, temps, steps=args.steps, device=args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print("EXPERIMENT 2: TEMPERATURE COLLAPSE SWEEP")
    print("=" * 80)
    print(f"{'Temperature T':<15} | {'Final A_eff':<15} | {'A_eff Ratio':<15} | {'State':<15}")
    print("-" * 80)
    for k, v in results.items():
        state_str = "COLLAPSED (A->0)" if v["is_collapsed"] else f"NESS (A* = {v['final_a_eff']:.3f})"
        print(f"{v['temperature']:<15.2f} | {v['final_a_eff']:<15.4e} | {v['a_eff_ratio']:<15.4e} | {state_str:<15}")
    print("=" * 80)


if __name__ == "__main__":
    main()
