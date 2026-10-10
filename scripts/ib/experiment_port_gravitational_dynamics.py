"""Physical verification experiment: Port boundary dynamics in self-gravitating continuous medium.

Tests three arms proposed by GPT & user:
Arm A: Injection only, no outward transport (u_port = 0, c_s = 0)
Arm B: Injection + outward convective flux (u_port > 0, c_s = 0)
Arm C: Injection + outward convective flux + degeneracy pressure (u_port > 0, c_s > 0)

Measures:
1. Port clumping ratio: fraction of total mass within port radius vs volume fraction.
2. Distance from peak density to nearest port: does structure decouple from port location?
3. Topological structure: filaments, sheets, knots, voids fractions.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from information_boltzmann.core.self_gravity import FourierPoissonSolver3D


class PortGravityExperiment3D:
    def __init__(
        self,
        shape: tuple[int, int, int] = (32, 32, 32),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        ports: list[tuple[float, float, float]] | None = None,
        dt: float = 0.02,
        G: float = 1.0,
    ):
        self.shape = shape
        self.device = torch.device(device)
        self.dt = dt
        self.G = G
        self.solver = FourierPoissonSolver3D(shape, G=G, device=device)

        nx, ny, nz = shape
        self.dx = 1.0 / nx
        self.dy = 1.0 / ny
        self.dz = 1.0 / nz

        # Two interacting ports by default to see if structures form between them
        if ports is None:
            self.ports = torch.tensor([
                [0.35, 0.50, 0.50],
                [0.65, 0.50, 0.50]
            ], device=self.device)
        else:
            self.ports = torch.tensor(ports, device=self.device)

        # Coordinate grid
        qx = torch.linspace(0, 1.0 - self.dx, nx, device=self.device) + 0.5 * self.dx
        qy = torch.linspace(0, 1.0 - self.dy, ny, device=self.device) + 0.5 * self.dy
        qz = torch.linspace(0, 1.0 - self.dz, nz, device=self.device) + 0.5 * self.dz
        self.grid = torch.stack(torch.meshgrid(qx, qy, qz, indexing="ij"), dim=-1) # [Nx, Ny, Nz, 3]

        # Port masks and outward unit vectors
        self.port_masks = []
        self.port_outward_vel = torch.zeros((*shape, 3), device=self.device)
        r_port = 0.10

        for p in self.ports:
            # Periodic displacement
            disp = (self.grid - p.view(1, 1, 1, 3) + 0.5) % 1.0 - 0.5
            dist = disp.norm(dim=-1)
            mask = torch.exp(-0.5 * (dist / r_port).square())
            self.port_masks.append(mask)

            # Radial outward vector from port center
            unit_radial = disp / dist.clamp_min(1e-4).unsqueeze(-1)
            self.port_outward_vel += mask.unsqueeze(-1) * unit_radial

        self.combined_mask = torch.stack(self.port_masks).sum(dim=0).clamp_max(1.0)

    def run_arm(
        self,
        arm_name: str,
        steps: int = 250,
        u_out: float = 0.0,
        c_s: float = 0.0,
        M: float = 0.5,
        source_rate: float = 5.0,
        damping: float = 0.15,
    ) -> dict:
        """Run a single experimental arm with specified transport & pressure physics."""
        nx, ny, nz = self.shape
        rho = torch.ones(self.shape, device=self.device) # background mean = 1.0
        vel = torch.zeros((*self.shape, 3), device=self.device)

        history = []
        r_eval = 0.12 # Radius around ports for clumping evaluation

        # Precompute port volume fraction
        mask_eval = torch.zeros(self.shape, device=self.device, dtype=torch.bool)
        for p in self.ports:
            disp = (self.grid - p.view(1, 1, 1, 3) + 0.5) % 1.0 - 0.5
            mask_eval |= (disp.norm(dim=-1) <= r_eval)
        vol_frac_port = mask_eval.float().mean().item()

        for step in range(steps):
            # 1. Source injection S(x, t) and global baseline balancing
            # We inject localized packets/matter at ports
            injection = source_rate * self.combined_mask
            # Uniform dissipation to maintain finite mean mass
            rho = rho + self.dt * (injection - damping * (rho - 1.0))
            rho = rho.clamp_min(1e-3)

            # 2. Self-gravitational potential and acceleration
            phi, phi_k = self.solver.solve_potential(rho)
            g_grav = self.solver.gravitational_acceleration(phi_k) # -grad(Phi)

            # 3. Flux and pressure gradient
            # J = rho * u_port - M * rho * grad(Phi + h'(rho))
            # grad(h'(rho)) = c_s^2 * grad(rho) / rho
            grad_rho_x = (torch.roll(rho, -1, 0) - torch.roll(rho, 1, 0)) / (2.0 * self.dx)
            grad_rho_y = (torch.roll(rho, -1, 1) - torch.roll(rho, 1, 1)) / (2.0 * self.dy)
            grad_rho_z = (torch.roll(rho, -1, 2) - torch.roll(rho, 1, 2)) / (2.0 * self.dz)
            grad_rho = torch.stack([grad_rho_x, grad_rho_y, grad_rho_z], dim=-1)

            # Pressure acceleration = -c_s^2 * grad(rho) / rho
            a_press = - (c_s**2) * grad_rho / rho.unsqueeze(-1)

            # Outward port convective transport
            v_conv = u_out * self.port_outward_vel

            # Total acceleration for momentum update
            a_total = g_grav + a_press

            vel = (vel + self.dt * (a_total - damping * vel)).clamp(-4.0, 4.0)
            # Add port outward convective velocity component
            v_effective = (vel + v_conv).clamp(-4.0, 4.0)

            # 4. Continuity advection step: d(rho)/dt = - div(rho * v_effective)
            # Compute conservative flux j = rho * v_effective
            j_flux = rho.unsqueeze(-1) * v_effective
            div_jx = (torch.roll(j_flux[..., 0], -1, 0) - torch.roll(j_flux[..., 0], 1, 0)) / (2.0 * self.dx)
            div_jy = (torch.roll(j_flux[..., 1], -1, 1) - torch.roll(j_flux[..., 1], 1, 1)) / (2.0 * self.dy)
            div_jz = (torch.roll(j_flux[..., 2], -1, 2) - torch.roll(j_flux[..., 2], 1, 2)) / (2.0 * self.dz)
            div_j = div_jx + div_jy + div_jz

            # Artificial numerical diffusion for shock smoothing
            lap_rho = (
                (torch.roll(rho, -1, 0) - 2 * rho + torch.roll(rho, 1, 0)) / (self.dx**2) +
                (torch.roll(rho, -1, 1) - 2 * rho + torch.roll(rho, 1, 1)) / (self.dy**2) +
                (torch.roll(rho, -1, 2) - 2 * rho + torch.roll(rho, 1, 2)) / (self.dz**2)
            )

            rho = rho - self.dt * div_j + self.dt * 0.01 * lap_rho
            rho = rho.clamp(1e-3, 100.0)

            # 5. Measure metrics every 10 steps
            if step % 10 == 0 or step == steps - 1:
                # Port mass concentration ratio C_port
                mass_port = rho[mask_eval].sum().item()
                total_mass = rho.sum().item()
                c_port = (mass_port / total_mass) / max(1e-5, vol_frac_port)

                # Distance from peak density to nearest port
                peak_idx = torch.argmax(rho).item()
                peak_coord = self.grid.reshape(-1, 3)[peak_idx]
                dists_to_ports = [
                    ((peak_coord - p + 0.5) % 1.0 - 0.5).norm().item()
                    for p in self.ports
                ]
                min_dist_to_port = min(dists_to_ports)

                # Tidal tensor and cosmic web classification
                try:
                    H = self.solver.tidal_tensor(phi_k)
                    web = self.solver.classify_web(H, threshold=0.15)
                    filaments = web["filaments"].float().mean().item()
                    knots = web["knots"].float().mean().item()
                except Exception:
                    filaments = 0.0
                    knots = 0.0

                history.append({
                    "step": step,
                    "c_port": c_port,
                    "min_dist_to_port": min_dist_to_port,
                    "max_density": rho.max().item(),
                    "density_contrast": (rho.std() / rho.mean()).item(),
                    "filaments": filaments,
                    "knots": knots,
                })

        return {
            "arm": arm_name,
            "final_rho": rho.detach().cpu().numpy(),
            "final_vel": vel.detach().cpu().numpy(),
            "history": history,
            "vol_frac_port": vol_frac_port,
        }


def main():
    parser = argparse.ArgumentParser(description="Port Dynamics Verification in Continuous Self-Gravitating Medium")
    parser.add_argument("--steps", type=int, default=250, help="Steps per arm")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="results/port_gravitational_dynamics")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print(" CBIM · Boundary Port Dynamics vs Gravitational Self-Organization")
    print(f" Arms: A (Inject only), B (Inject + Outward flux), C (Inject + Flux + Pressure)")
    print(f" Grid: 32x32x32 | Steps: {args.steps} | Device: {args.device}")
    print("======================================================================")

    exp = PortGravityExperiment3D(shape=(32, 32, 32), device=args.device)

    # Arm A: Injection only, no outward velocity, no pressure
    print("\n[Running Arm A] Inject only (u_out=0, c_s=0)...")
    t0 = time.time()
    res_A = exp.run_arm("Arm A (Inject only)", steps=args.steps, u_out=0.0, c_s=0.0)
    print(f"  Arm A completed in {time.time() - t0:.2f}s | Final C_port: {res_A['history'][-1]['c_port']:.2f}x | MinDist: {res_A['history'][-1]['min_dist_to_port']:.4f}")

    # Arm B: Injection + outward convective flux, but no pressure
    print("\n[Running Arm B] Inject + Outward Flux (u_out=1.8, c_s=0)...")
    t0 = time.time()
    res_B = exp.run_arm("Arm B (Inject + Outward Flux)", steps=args.steps, u_out=1.8, c_s=0.0)
    print(f"  Arm B completed in {time.time() - t0:.2f}s | Final C_port: {res_B['history'][-1]['c_port']:.2f}x | MinDist: {res_B['history'][-1]['min_dist_to_port']:.4f}")

    # Arm C: Injection + outward convective flux + degeneracy pressure
    print("\n[Running Arm C] Inject + Outward Flux + Pressure (u_out=1.8, c_s=0.8)...")
    t0 = time.time()
    res_C = exp.run_arm("Arm C (Inject + Flux + Pressure)", steps=args.steps, u_out=1.8, c_s=0.8)
    print(f"  Arm C completed in {time.time() - t0:.2f}s | Final C_port: {res_C['history'][-1]['c_port']:.2f}x | MinDist: {res_C['history'][-1]['min_dist_to_port']:.4f}")

    # Save summary report
    report = {
        "arms": {
            "A": {
                "description": "Inject only (no transport, no pressure)",
                "final_c_port": res_A["history"][-1]["c_port"],
                "final_min_dist_to_port": res_A["history"][-1]["min_dist_to_port"],
                "final_max_density": res_A["history"][-1]["max_density"],
                "final_filaments": res_A["history"][-1]["filaments"],
                "clumping_at_port": bool(res_A["history"][-1]["min_dist_to_port"] < 0.04),
            },
            "B": {
                "description": "Inject + Outward convective flux",
                "final_c_port": res_B["history"][-1]["c_port"],
                "final_min_dist_to_port": res_B["history"][-1]["min_dist_to_port"],
                "final_max_density": res_B["history"][-1]["max_density"],
                "final_filaments": res_B["history"][-1]["filaments"],
                "decoupled_from_port": bool(res_B["history"][-1]["min_dist_to_port"] > 0.08),
            },
            "C": {
                "description": "Inject + Outward flux + Degeneracy pressure",
                "final_c_port": res_C["history"][-1]["c_port"],
                "final_min_dist_to_port": res_C["history"][-1]["min_dist_to_port"],
                "final_max_density": res_C["history"][-1]["max_density"],
                "final_filaments": res_C["history"][-1]["filaments"],
                "stable_structures": bool(res_C["history"][-1]["filaments"] > 0.25),
            },
        }
    }
    (out_dir / "port_dynamics_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Generate Publication-Quality Figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=150)
    fig.patch.set_facecolor("#0b0813")

    z_mid = exp.shape[2] // 2

    # Row 1: 2D Mid-plane density slices
    for idx, (res, title, letter) in enumerate([
        (res_A, "Arm A: Inject Only (u=0, P=0)", "A"),
        (res_B, "Arm B: Outward Flux Only (u>0, P=0)", "B"),
        (res_C, "Arm C: Flux + Pressure (u>0, P>0)", "C"),
    ]):
        ax = axes[0, idx]
        ax.set_facecolor("#0b0813")
        im = ax.imshow(
            res["final_rho"][:, :, z_mid].T,
            origin="lower",
            cmap="inferno",
            extent=[0, 1, 0, 1],
        )
        # Plot ports
        p_np = exp.ports.detach().cpu().numpy()
        ax.scatter(p_np[:, 0], p_np[:, 1], color="#67e5d3", s=70, marker="^", label="Input Ports", zorder=5)
        # Plot peak density
        peak_idx = np.unravel_index(np.argmax(res["final_rho"]), res["final_rho"].shape)
        ax.scatter([peak_idx[0] / 32], [peak_idx[1] / 32], color="#ff70a6", s=60, marker="x", label="Peak Density", zorder=6)

        ax.set_title(title, color="#e8dcf5", fontsize=12, pad=10)
        ax.tick_params(colors="#8f80a6", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#392b4d")
        if idx == 0:
            ax.set_ylabel("Y (Spatial Coordinate)", color="#c4b5dc", fontsize=10)
        ax.set_xlabel("X (Spatial Coordinate)", color="#c4b5dc", fontsize=10)
        if idx == 2:
            ax.legend(loc="upper right", facecolor="#161024", edgecolor="#4d3866", labelcolor="#eee0fa", fontsize=8)

    # Row 2, Col 1: Port Clumping Ratio C_port over time
    ax = axes[1, 0]
    ax.set_facecolor("#0e0a1a")
    for res, col, label in [(res_A, "#ff6b6b", "Arm A (Inject only)"), (res_B, "#ffd166", "Arm B (Outward Flux)"), (res_C, "#06d6a0", "Arm C (Flux + Press)")]:
        steps = [h["step"] for h in res["history"]]
        c_vals = [h["c_port"] for h in res["history"]]
        ax.plot(steps, c_vals, label=label, color=col, lw=2.2)
    ax.axhline(1.0, color="#8879a3", ls="--", lw=1.2, label="Uniform Baseline (C=1)")
    ax.set_title("Port Clumping Ratio C_port (Mass in Port / Expected)", color="#e8dcf5", fontsize=11)
    ax.set_xlabel("Simulation Step", color="#c4b5dc", fontsize=10)
    ax.set_ylabel("Concentration Ratio (×)", color="#c4b5dc", fontsize=10)
    ax.tick_params(colors="#8f80a6", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#392b4d")
    ax.legend(facecolor="#161024", edgecolor="#4d3866", labelcolor="#eee0fa", fontsize=8)
    ax.grid(True, color="#251b36", ls=":", alpha=0.6)

    # Row 2, Col 2: Distance from Peak Density to Nearest Port
    ax = axes[1, 1]
    ax.set_facecolor("#0e0a1a")
    for res, col, label in [(res_A, "#ff6b6b", "Arm A"), (res_B, "#ffd166", "Arm B"), (res_C, "#06d6a0", "Arm C")]:
        steps = [h["step"] for h in res["history"]]
        d_vals = [h["min_dist_to_port"] for h in res["history"]]
        ax.plot(steps, d_vals, label=label, color=col, lw=2.2)
    ax.set_title("Distance: Peak Density to Nearest Port", color="#e8dcf5", fontsize=11)
    ax.set_xlabel("Simulation Step", color="#c4b5dc", fontsize=10)
    ax.set_ylabel("Distance in Space", color="#c4b5dc", fontsize=10)
    ax.tick_params(colors="#8f80a6", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#392b4d")
    ax.legend(facecolor="#161024", edgecolor="#4d3866", labelcolor="#eee0fa", fontsize=8)
    ax.grid(True, color="#251b36", ls=":", alpha=0.6)

    # Row 2, Col 3: Filament Fraction over time
    ax = axes[1, 2]
    ax.set_facecolor("#0e0a1a")
    for res, col, label in [(res_A, "#ff6b6b", "Arm A"), (res_B, "#ffd166", "Arm B"), (res_C, "#06d6a0", "Arm C")]:
        steps = [h["step"] for h in res["history"]]
        f_vals = [h["filaments"] * 100 for h in res["history"]]
        ax.plot(steps, f_vals, label=label, color=col, lw=2.2)
    ax.set_title("Cosmic Web Filaments Fraction (%)", color="#e8dcf5", fontsize=11)
    ax.set_xlabel("Simulation Step", color="#c4b5dc", fontsize=10)
    ax.set_ylabel("Filaments Volume %", color="#c4b5dc", fontsize=10)
    ax.tick_params(colors="#8f80a6", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#392b4d")
    ax.legend(facecolor="#161024", edgecolor="#4d3866", labelcolor="#eee0fa", fontsize=8)
    ax.grid(True, color="#251b36", ls=":", alpha=0.6)

    plt.tight_layout()
    fig_path = out_dir / "port_dynamics_comparison.png"
    plt.savefig(fig_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"\nFigure saved to: {fig_path}")
    print(f"Report saved to: {out_dir / 'port_dynamics_report.json'}")
    print("\n[Verification Complete!]")


if __name__ == "__main__":
    main()
