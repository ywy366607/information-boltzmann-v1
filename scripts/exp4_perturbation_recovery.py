"""Experiment 4: Perturbation-Recovery & Dynamic Self-Maintenance."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fine_grain.information_boltzmann import InformationBoltzmann, PhaseState


def gaussian_kl_divergence(state1: PhaseState, state2: PhaseState) -> float:
    """Compute analytical KL divergence between two empirical phase distributions."""
    n, d = state1.x.shape
    k = 2 * d

    xv1 = torch.cat((state1.x, state1.v), dim=-1)
    xv2 = torch.cat((state2.x, state2.v), dim=-1)

    mu1 = xv1.mean(0)
    mu2 = xv2.mean(0)

    cov1 = ((xv1 - mu1).T @ (xv1 - mu1)) / n + 1e-6 * torch.eye(k, device=xv1.device)
    cov2 = ((xv2 - mu2).T @ (xv2 - mu2)) / n + 1e-6 * torch.eye(k, device=xv2.device)

    cov2_inv = torch.linalg.inv(cov2)
    diff = (mu2 - mu1)[:, None]

    term_tr = torch.trace(cov2_inv @ cov1)
    term_quad = (diff.T @ cov2_inv @ diff).squeeze()
    term_logdet = torch.linalg.slogdet(cov2)[1] - torch.linalg.slogdet(cov1)[1]

    kl = 0.5 * (term_tr + term_quad - k + term_logdet)
    return max(0.0, float(kl.item()))


def run_perturbation_experiment(checkpoint_path: Path, tokens_path: Path,
                                gammas: list[float] = [1.0, 0.5, 0.2, 0.05],
                                recovery_steps: int = 40,
                                device: str = "cuda") -> dict:
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = saved["metadata"]["config"]
    tokens = np.load(tokens_path, mmap_mode="r")

    results = {}

    perturbation_types = {
        "shift_x": lambda st: PhaseState(st.x + torch.tensor([1.0, -0.8], device=device), st.v.clone(), st.time),
        "kick_v": lambda st: PhaseState(st.x.clone(), st.v + torch.tensor([2.0, 1.5], device=device), st.time),
        "spread_x": lambda st: PhaseState(st.x.mean(0) + 2.5 * (st.x - st.x.mean(0)), st.v.clone(), st.time),
    }

    for p_name, p_fn in perturbation_types.items():
        print(f"\n--- Testing Perturbation: {p_name} ---")
        gamma_curves = {}

        for g in gammas:
            cfg = json.loads(json.dumps(config))
            cfg["dynamics"]["gamma"] = g
            cfg["dynamics"]["kappa"] = 1.0
            cfg["dynamics"]["temperature"] = 0.1
            cfg["dynamics"]["adaptive_gamma"] = False

            model = InformationBoltzmann.from_config(cfg).to(device)
            model.load_state_dict(saved["model"])
            model.eval()

            # Burn-in for 50 tokens to reach steady state
            gen = torch.Generator(device=device).manual_seed(42)
            ctrl_state = model.initialize(torch.tensor([1], device=device), gen)
            for t in range(50):
                ctrl_state, _, _ = model.advance(ctrl_state, int(tokens[t]), gen)

            # Apply perturbation at t = 50
            pert_state = p_fn(ctrl_state)

            d_kl_phase_list = []
            d_kl_logits_list = []

            # Save generator state for synchronization
            with torch.no_grad():
                for tau in range(recovery_steps):
                    tok = int(tokens[50 + tau])

                    # Get predictions
                    ctrl_logits = model.decode(ctrl_state)
                    pert_logits = model.decode(pert_state)

                    kl_logits = float(F.kl_div(
                        F.log_softmax(pert_logits, dim=-1),
                        F.softmax(ctrl_logits, dim=-1),
                        reduction="sum"
                    ).item())
                    kl_phase = gaussian_kl_divergence(pert_state, ctrl_state)

                    d_kl_phase_list.append(kl_phase)
                    d_kl_logits_list.append(kl_logits)

                    # Advance both trajectories under identical token and noise
                    gen_snap = gen.get_state()
                    ctrl_state, _, _ = model.advance(ctrl_state, tok, gen)
                    gen.set_state(gen_snap)
                    pert_state, _, _ = model.advance(pert_state, tok, gen)

            # Integrated recovery time tau_recovery = sum D(t) / D(0)
            d0 = max(1e-8, d_kl_phase_list[0])
            norm_curve = [d / d0 for d in d_kl_phase_list]
            tau_rec = sum(norm_curve)

            gamma_curves[f"gamma_{g:.2f}"] = {
                "gamma": g,
                "d0_phase": d_kl_phase_list[0],
                "d_final_phase": d_kl_phase_list[-1],
                "recovery_ratio": d_kl_phase_list[-1] / d0,
                "tau_recovery": tau_rec,
                "d_kl_phase_curve": d_kl_phase_list,
                "d_kl_logits_curve": d_kl_logits_list,
                "regime": "dynamic_self_maintenance" if d_kl_phase_list[-1] < 0.1 * d0 and tau_rec > 3.0 else ("strong_attractor" if tau_rec <= 3.0 else "unrecovered"),
            }

        results[p_name] = gamma_curves

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/information_boltzmann/ib_owt_2500_champion.pt"))
    parser.add_argument("--tokens", type=Path, default=Path("data/ib_owt_smoke/train.npy"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("results/published/perturbation_recovery.json"))
    args = parser.parse_args()

    res = run_perturbation_experiment(args.checkpoint, args.tokens, recovery_steps=args.steps, device=args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 90)
    print("EXPERIMENT 4: PERTURBATION-RECOVERY & DYNAMIC SELF-MAINTENANCE")
    print("=" * 90)
    print(f"{'Perturbation':<12} | {'Gamma':<8} | {'D(0)':<10} | {'D(final)':<10} | {'tau_recovery':<14} | {'Regime':<24}")
    print("-" * 90)
    for p_name, g_dict in res.items():
        for g_k, v in g_dict.items():
            print(f"{p_name:<12} | {v['gamma']:<8.2f} | {v['d0_phase']:<10.2f} | {v['d_final_phase']:<10.4f} | {v['tau_recovery']:<14.2f} | {v['regime']:<24}")
    print("=" * 90)


if __name__ == "__main__":
    main()
