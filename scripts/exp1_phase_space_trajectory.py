"""Experiment 1: Phase Space Density, Probability Current J=(J_x, J_v), and Live Animation Trajectory Data."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fine_grain.information_boltzmann import InformationBoltzmann, PhaseState


def generate_live_trajectory(checkpoint_path: Path, tokens_path: Path, steps: int = 150,
                             device: str = "cuda") -> dict:
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = saved["metadata"]["config"]
    tokens = np.load(tokens_path, mmap_mode="r")[:steps]

    model = InformationBoltzmann.from_config(config).to(device)
    model.load_state_dict(saved["model"])
    model.eval()

    gen = torch.Generator(device=device).manual_seed(42)
    state = model.initialize(torch.tensor([1], device=device), gen)

    frames = []
    tokens_text = []

    # Map byte token to printable char
    def token_to_char(t_id: int) -> str:
        if t_id < 4:
            return ["<pad>", "<bos>", "<eos>", "<unk>"][t_id]
        byte_val = t_id - 4
        try:
            return bytes([byte_val]).decode("utf-8")
        except:
            return f"\\x{byte_val:02x}"

    gamma = model.force.gamma
    kappa = model.force.kappa
    temp = model.force.temperature
    dim = model.force.net[-1].out_features

    with torch.no_grad():
        for t in range(steps):
            tok = int(tokens[t])
            char_repr = token_to_char(tok)
            tokens_text.append(char_repr)

            # Record particle state before advancing
            x_np = state.x.cpu().numpy().tolist()
            v_np = state.v.cpu().numpy().tolist()

            # Measure drive at particle positions: a_I = -kappa*x - gamma*v + b_theta
            b_theta = model.force.drive(state.x, tok, state.time)
            accel = -kappa * state.x - gamma * state.v + b_theta
            a_np = accel.cpu().numpy().tolist()

            # Probability current at particles: J_x = v, J_v = a_I
            # Covariance and effective area
            cov_x = float(state.moments()["position_variance"].item())
            cov_v = float(state.moments()["velocity_variance"].item())
            ke = float(state.moments()["kinetic_energy"].item())
            pe = 0.5 * kappa * float(state.moments()["position_second_moment"].item())

            xv = torch.cat((state.x, state.v), dim=-1)
            centered = xv - xv.mean(0)
            cov = (centered.T @ centered) / len(state.x)
            eigvals = torch.linalg.eigvalsh(cov).clamp_min(1e-30)
            h = 0.5 * (4 * (1.0 + math.log(2.0 * math.pi)) + float(eigvals.log().sum().item()))
            a_eff = math.exp(min(h, 50.0)) if h > -50.0 else 0.0

            # Decode prediction
            logits = model.decode(state)
            probs = torch.softmax(logits, dim=-1)
            top5_indices = torch.topk(probs, 5).indices.cpu().numpy().tolist()
            top5_tokens = [{"token": token_to_char(idx), "prob": float(probs[idx].item())} for idx in top5_indices]
            target_prob = float(probs[tok].item())
            ce_loss = -math.log(max(1e-12, target_prob))

            # Advance state
            next_state, _, collision_stats = model.advance(state, tok, gen)

            frames.append({
                "step": t,
                "token_id": tok,
                "token_char": char_repr,
                "target_prob": target_prob,
                "ce_loss": ce_loss,
                "top5": top5_tokens,
                "particles": {
                    "x": x_np,
                    "v": v_np,
                    "a": a_np,
                },
                "collisions": collision_stats["accepted"],
                "metrics": {
                    "var_x": cov_x,
                    "var_v": cov_v,
                    "a_eff": a_eff,
                    "entropy_h": h,
                    "total_energy": ke + pe,
                    "kinetic_energy": ke,
                    "t_eff": cov_v / dim,
                }
            })

            state = next_state

    # 2D Grid Vector Field of Force/Flow at steady-state center
    grid_res = 11
    grid_coords = np.linspace(-1.5, 1.5, grid_res).tolist()
    gx, gy = np.meshgrid(grid_coords, grid_coords)
    grid_points = torch.tensor(np.stack((gx.flatten(), gy.flatten()), axis=-1), device=device, dtype=torch.float32)
    with torch.no_grad():
        b_grid = model.force.drive(grid_points, tokens[0], 0.0)
        force_grid = -kappa * grid_points + b_grid
    flow_grid = {
        "x": grid_coords,
        "fx": force_grid[:, 0].reshape(grid_res, grid_res).cpu().numpy().tolist(),
        "fy": force_grid[:, 1].reshape(grid_res, grid_res).cpu().numpy().tolist(),
    }

    return {
        "metadata": {
            "particles": config["model"]["particles"],
            "phase_dim": config["model"]["phase_dim"],
            "gamma": gamma,
            "kappa": kappa,
            "temperature": temp,
            "total_frames": len(frames),
        },
        "flow_grid": flow_grid,
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/information_boltzmann/ib_owt_2500_champion.pt"))
    parser.add_argument("--tokens", type=Path, default=Path("data/ib_owt_smoke/train.npy"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--output", type=Path, default=Path("results/published/phase_space_trajectory.json"))
    args = parser.parse_args()

    data = generate_live_trajectory(args.checkpoint, args.tokens, steps=args.steps, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data), encoding="utf-8")
    print(f"Generated {len(data['frames'])} trajectory frames saved to {args.output}")


if __name__ == "__main__":
    main()
