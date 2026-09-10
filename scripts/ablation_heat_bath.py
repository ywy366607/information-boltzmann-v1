"""Experiment 1: 4-arm Heat-bath ablation proving No-bath -> T down -> V_eff down -> collapse."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fine_grain.information_boltzmann import InformationBoltzmann, PhaseState


def compute_effective_volume_and_entropy(state: PhaseState) -> tuple[float, float]:
    """Compute phase covariance determinant, differential entropy proxy H, and effective volume V_eff."""
    n, d = state.x.shape
    xv = torch.cat((state.x, state.v), dim=-1)  # [N, 2*d]
    centered = xv - xv.mean(0, keepdim=True)
    cov = (centered.T @ centered) / n  # [2d, 2d]

    # Compute log determinant with eigenvalue floor for numerical safety
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(1e-30)
    log_det = float(eigvals.log().sum().item())

    dim_total = 2 * d
    # Differential entropy: H = 0.5 * (dim * (1 + ln(2*pi)) + ln(det(cov)))
    h = 0.5 * (dim_total * (1.0 + math.log(2.0 * math.pi)) + log_det)

    # V_eff = exp(H)
    # Clamp H to avoid overflow / underflow
    if h < -100.0:
        v_eff = 0.0
    else:
        v_eff = math.exp(min(h, 100.0))

    return h, v_eff


def run_arm(name: str, config: dict, tokens: np.ndarray, drive: bool, dissipation: bool,
            bath: bool, device: torch.device, steps: int = 128) -> dict:
    cfg = json.loads(json.dumps(config))
    d = cfg["dynamics"]

    d["acceleration_norm_bound"] = 1.0 if drive else 0.0
    d["gamma"] = 1.0 if dissipation else 0.0
    d["temperature"] = 0.1 if bath else 0.0
    d["adaptive_gamma"] = False  # isolate pure physical parameters

    model = InformationBoltzmann.from_config(cfg).to(device)
    model.eval()

    gen = torch.Generator(device=device).manual_seed(42)
    state = model.initialize(torch.tensor([1], device=device), gen)

    trajectory = []
    kappa = d["kappa"]
    dim = cfg["model"]["phase_dim"]

    with torch.no_grad():
        for t in range(steps):
            tok = int(tokens[t])
            state, _, _ = model.advance(state, tok, gen)

            # 1. Total energy E = 0.5 * v^2 + 0.5 * kappa * x^2
            ke = float(state.moments()["kinetic_energy"].item())
            pe = 0.5 * kappa * float(state.moments()["position_second_moment"].item())
            total_e = ke + pe

            # 2. Variances
            var_x = float(state.moments()["position_variance"].item())
            var_v = float(state.moments()["velocity_variance"].item())

            # 3. Effective kinetic temperature: T_eff = Var(v) / d
            t_eff = var_v / dim

            # 4. Phase entropy and effective volume
            h_entropy, v_eff = compute_effective_volume_and_entropy(state)

            if (t + 1) % 8 == 0 or t == 0:
                trajectory.append({
                    "step": t + 1,
                    "total_energy": total_e,
                    "kinetic_energy": ke,
                    "t_eff": t_eff,
                    "var_x": var_x,
                    "var_v": var_v,
                    "entropy_H": h_entropy,
                    "v_eff": v_eff,
                })

    return {
        "arm": name,
        "drive": drive,
        "dissipation": dissipation,
        "heat_bath": bath,
        "initial_v_eff": trajectory[0]["v_eff"],
        "final_v_eff": trajectory[-1]["v_eff"],
        "v_eff_ratio": trajectory[-1]["v_eff"] / max(1e-30, trajectory[0]["v_eff"]),
        "final_var_x": trajectory[-1]["var_x"],
        "final_var_v": trajectory[-1]["var_v"],
        "final_t_eff": trajectory[-1]["t_eff"],
        "final_entropy_H": trajectory[-1]["entropy_H"],
        "trajectory": trajectory,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/information_boltzmann/smoke_owt.json"))
    parser.add_argument("--tokens", type=Path, default=Path("data/ib_owt_smoke/train.npy"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("results/ablation_heat_bath.json"))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    tokens = np.load(args.tokens, mmap_mode="r")

    arms = [
        ("A: Drive + Dissipation + Bath", True, True, True),
        ("B: Drive + Dissipation - Bath (No Bath)", True, True, False),
        ("C: No Drive + Dissipation + Bath (Equilibrium)", False, True, True),
        ("D: Drive - Dissipation + Bath (No Dissipation)", True, False, True),
    ]

    results = {}
    for name, drive, diss, bath in arms:
        print(f"Running Arm: {name}...")
        res = run_arm(name, config, tokens, drive, diss, bath, device, steps=args.steps)
        results[name] = res

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    # Print summary table
    print("\n" + "=" * 95)
    print(f"{'Condition':<40} | {'T_eff':<10} | {'Var(x)':<10} | {'Var(v)':<10} | {'Entropy H':<10} | {'V_eff':<12}")
    print("-" * 95)
    for name, r in results.items():
        print(f"{name:<40} | {r['final_t_eff']:<10.4f} | {r['final_var_x']:<10.4e} | {r['final_var_v']:<10.4e} | {r['final_entropy_H']:<10.2f} | {r['final_v_eff']:<12.4e}")
    print("=" * 95)


if __name__ == "__main__":
    main()
