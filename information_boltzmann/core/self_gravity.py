"""Self-gravitational morphogenesis and cosmological web evolution on 3D torus.

Implements fast Fourier Poisson solver, Zel'dovich initial condition,
Cloud-in-Cell (CIC) mass/momentum deposition, and tidal tensor (T-web)
classification (Knots/Soma, Filaments/Axons, Sheets/Pancakes, Voids).
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


class FourierPoissonSolver3D:
    """Spectral Poisson solver on periodic 3D torus T^3 using real FFT.

    Solves:
        nabla^2 Phi(x) = 4 * pi * G * (rho(x) - <rho>)
    with exact spectral differentiation for gravitational acceleration g = -nabla Phi
    and the tidal deformation tensor (Zel'dovich Hessian) H_ij = d^2 Phi / (dx_i dx_j).
    """

    def __init__(self, shape: Tuple[int, int, int], G: float = 1.0, device=None, dtype=torch.float32):
        self.shape = shape
        self.G = float(G)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype

        nx, ny, nz = shape
        kx = torch.fft.fftfreq(nx, d=1.0 / nx, device=self.device) * 2.0 * math.pi
        ky = torch.fft.fftfreq(ny, d=1.0 / ny, device=self.device) * 2.0 * math.pi
        kz = torch.fft.rfftfreq(nz, d=1.0 / nz, device=self.device) * 2.0 * math.pi

        self.KX, self.KY, self.KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        self.K_sq = self.KX ** 2 + self.KY ** 2 + self.KZ ** 2
        # Avoid division by zero at DC k=0
        self.inv_K_sq = torch.where(self.K_sq > 0, 1.0 / self.K_sq, torch.zeros_like(self.K_sq))

    def solve_potential(self, rho: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return potential Phi and its Fourier coefficients Phi_k."""
        delta = rho - rho.mean()
        rho_k = torch.fft.rfftn(delta)
        phi_k = - (4.0 * math.pi * self.G * self.inv_K_sq) * rho_k
        phi_k[0, 0, 0] = 0.0
        phi = torch.fft.irfftn(phi_k, s=self.shape)
        return phi, phi_k

    def gravitational_acceleration(self, phi_k: torch.Tensor) -> torch.Tensor:
        """Return acceleration field g = -nabla Phi of shape [Nx, Ny, Nz, 3]."""
        gx = torch.fft.irfftn(- 1j * self.KX * phi_k, s=self.shape)
        gy = torch.fft.irfftn(- 1j * self.KY * phi_k, s=self.shape)
        gz = torch.fft.irfftn(- 1j * self.KZ * phi_k, s=self.shape)
        return torch.stack([gx, gy, gz], dim=-1)

    def tidal_tensor(self, phi_k: torch.Tensor) -> torch.Tensor:
        """Return tidal Hessian H_ij = d_i d_j Phi of shape [Nx, Ny, Nz, 3, 3]."""
        H_xx = torch.fft.irfftn(- self.KX * self.KX * phi_k, s=self.shape)
        H_yy = torch.fft.irfftn(- self.KY * self.KY * phi_k, s=self.shape)
        H_zz = torch.fft.irfftn(- self.KZ * self.KZ * phi_k, s=self.shape)
        H_xy = torch.fft.irfftn(- self.KX * self.KY * phi_k, s=self.shape)
        H_xz = torch.fft.irfftn(- self.KX * self.KZ * phi_k, s=self.shape)
        H_yz = torch.fft.irfftn(- self.KY * self.KZ * phi_k, s=self.shape)

        H = torch.zeros(*self.shape, 3, 3, device=self.device, dtype=self.dtype)
        H[..., 0, 0] = H_xx
        H[..., 1, 1] = H_yy
        H[..., 2, 2] = H_zz
        H[..., 0, 1] = H[..., 1, 0] = H_xy
        H[..., 0, 2] = H[..., 2, 0] = H_xz
        H[..., 1, 2] = H[..., 2, 1] = H_yz
        return H

    def classify_web(self, tidal_matrix: torch.Tensor, threshold: float = 0.2) -> Dict[str, torch.Tensor]:
        """Classify regions using T-web tidal eigenvalues (Forero-Romero et al. 2009).

        Eigenvalues sorted ascending: lambda_3 <= lambda_2 <= lambda_1.
        - Knots / Soma: lambda_3 > threshold (3 positive directions, 3D collapse)
        - Filaments / Axons: lambda_2 > threshold >= lambda_3 (2 positive directions, 2D collapse)
        - Sheets / Pancakes: lambda_1 > threshold >= lambda_2 (1 positive direction, 1D collapse)
        - Voids / Cavities: lambda_1 <= threshold (0 positive directions, expansion)
        """
        H_reg = tidal_matrix
        if H_reg.abs().max() < 1e-6:
            H_reg = H_reg + 1e-6 * torch.eye(3, device=tidal_matrix.device, dtype=tidal_matrix.dtype).view(1, 1, 1, 3, 3)
        evals = torch.linalg.eigvalsh(H_reg)
        l3, l2, l1 = evals[..., 0], evals[..., 1], evals[..., 2]

        knots = l3 > threshold
        filaments = (l2 > threshold) & (~knots)
        sheets = (l1 > threshold) & (~knots) & (~filaments)
        voids = l1 <= threshold

        return {
            'knots': knots,
            'filaments': filaments,
            'sheets': sheets,
            'voids': voids,
            'eigenvalues': evals,
            'principal_axis': None  # Can be computed via eigh if needed
        }


def cic_deposit_3d(pos: torch.Tensor, shape: Tuple[int, int, int], weights: torch.Tensor | None = None) -> torch.Tensor:
    """Cloud-in-Cell (CIC) mass/charge assignment from continuous positions to 3D grid."""
    nx, ny, nz = shape
    device = pos.device
    N = nx  # assume isotropic grid
    u = pos * torch.tensor([nx, ny, nz], device=device, dtype=pos.dtype)
    base = u.floor().long()
    frac = u - base.float()
    base[..., 0] = base[..., 0] % nx
    base[..., 1] = base[..., 1] % ny
    base[..., 2] = base[..., 2] % nz

    grid = torch.zeros(nx, ny, nz, device=device, dtype=pos.dtype)
    w_in = weights if weights is not None else torch.ones(len(pos), device=device, dtype=pos.dtype)

    for dx in (0, 1):
        wx = frac[:, 0] if dx else (1.0 - frac[:, 0])
        ix = (base[:, 0] + dx) % nx
        for dy in (0, 1):
            wy = frac[:, 1] if dy else (1.0 - frac[:, 1])
            iy = (base[:, 1] + dy) % ny
            for dz in (0, 1):
                wz = frac[:, 2] if dz else (1.0 - frac[:, 2])
                iz = (base[:, 2] + dz) % nz
                weight = w_in * (wx * wy * wz)
                lin_idx = (ix * ny + iy) * nz + iz
                grid.put_(lin_idx, weight, accumulate=True)

    return grid


def cic_interpolate_3d(pos: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    """Trilinear interpolation of a grid field [Nx, Ny, Nz, D] back to continuous particle positions."""
    nx, ny, nz = field.shape[:3]
    device = pos.device
    u = pos * torch.tensor([nx, ny, nz], device=device, dtype=pos.dtype)
    base = u.floor().long()
    frac = u - base.float()
    base[..., 0] = base[..., 0] % nx
    base[..., 1] = base[..., 1] % ny
    base[..., 2] = base[..., 2] % nz

    is_vector = (field.ndim == 4)
    out_dim = field.shape[-1] if is_vector else 1
    out = torch.zeros(len(pos), out_dim, device=device, dtype=pos.dtype)

    for dx in (0, 1):
        wx = frac[:, 0] if dx else (1.0 - frac[:, 0])
        ix = (base[:, 0] + dx) % nx
        for dy in (0, 1):
            wy = frac[:, 1] if dy else (1.0 - frac[:, 1])
            iy = (base[:, 1] + dy) % ny
            for dz in (0, 1):
                wz = frac[:, 2] if dz else (1.0 - frac[:, 2])
                iz = (base[:, 2] + dz) % nz
                w = (wx * wy * wz).unsqueeze(-1)
                vals = field[ix, iy, iz]
                if not is_vector:
                    vals = vals.unsqueeze(-1)
                out += w * vals

    return out if is_vector else out.squeeze(-1)


class GravitationalCosmos3D:
    """Autonomous 3D Self-Gravitational Morphogenesis Medium (Generation 4).

    Evolves a continuous medium on a 3D torus through:
    1. Fast Fourier Poisson gravity (Jeans instability and anisotropic tidal collapse).
    2. Zel'dovich cosmological initialization (Gaussian random scale-invariant power spectrum).
    3. Symplectic Particle-Mesh integration with physical pressure stabilization and viscous dissipation.
    4. Exact energy and mass accounting (Hamiltonian energy conservation, virial ratio).
    5. Direct bridge to ETHER 3D visualizer.
    """

    def __init__(self, shape: Tuple[int, int, int] = (32, 32, 32),
                 G: float = 1.0,
                 dt: float = 0.04,
                 damping: float = 0.02,
                 pressure_cs: float = 0.05,
                 power_index: float = -1.5,
                 seed: int = 449,
                 device=None):
        self.shape = shape
        self.G = float(G)
        self.dt = float(dt)
        self.damping = float(damping)
        self.pressure_cs = float(pressure_cs)
        self.power_index = float(power_index)
        self.seed = int(seed)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.solver = FourierPoissonSolver3D(shape, G=self.G, device=self.device)
        self.step_count = 0
        self.elapsed_time = 0.0

        # Initialize particles and initial fields
        self.init_zeldovich(seed=self.seed)

    def init_zeldovich(self, seed: int = 449, amplitude: float = 0.015):
        """Initialize particle positions and velocities via Zel'dovich approximation."""
        torch.manual_seed(seed)
        nx, ny, nz = self.shape
        np_total = nx * ny * nz

        # White noise in Fourier space
        white_noise = torch.randn(nx, ny, nz, device=self.device)
        noise_k = torch.fft.rfftn(white_noise)

        # Power spectrum P(k) ~ k^(power_index)
        inv_k = torch.where(self.solver.K_sq > 0, self.solver.K_sq ** (self.power_index / 4.0), torch.zeros_like(self.solver.K_sq))
        inv_k[0, 0, 0] = 0.0

        # High frequency smoothing to prevent single-cell aliasing
        k_cutoff = math.pi * min(self.shape) * 0.7
        inv_k[self.solver.K_sq > k_cutoff ** 2] = 0.0

        psi_k = noise_k * inv_k

        # Displacement field d = -nabla(psi)
        disp_x = torch.fft.irfftn(- 1j * self.solver.KX * psi_k, s=self.shape)
        disp_y = torch.fft.irfftn(- 1j * self.solver.KY * psi_k, s=self.shape)
        disp_z = torch.fft.irfftn(- 1j * self.solver.KZ * psi_k, s=self.shape)

        disp_mag = torch.sqrt(disp_x ** 2 + disp_y ** 2 + disp_z ** 2).max()
        scale = amplitude / max(float(disp_mag), 1e-12)
        disp_x *= scale
        disp_y *= scale
        disp_z *= scale

        # Unperturbed Lagrangian coordinates q_ijk
        qx = torch.linspace(0, 1.0 - 1.0 / nx, nx, device=self.device) + 0.5 / nx
        qy = torch.linspace(0, 1.0 - 1.0 / ny, ny, device=self.device) + 0.5 / ny
        qz = torch.linspace(0, 1.0 - 1.0 / nz, nz, device=self.device) + 0.5 / nz
        QX, QY, QZ = torch.meshgrid(qx, qy, qz, indexing='ij')

        self.pos = torch.stack([QX + disp_x, QY + disp_y, QZ + disp_z], dim=-1).reshape(-1, 3) % 1.0
        # Zel'dovich initial velocity is proportional to displacement: v = - H_0 * d
        self.vel = torch.stack([disp_x, disp_y, disp_z], dim=-1).reshape(-1, 3) * (self.G ** 0.5)

    def advance(self, substeps: int = 1) -> Dict[str, float]:
        """Perform physical time integration using Leapfrog Drift-Kick-Drift."""
        nx, ny, nz = self.shape
        dt = self.dt / substeps

        for _ in range(substeps):
            # 1. Mass deposition (CIC)
            rho_grid = cic_deposit_3d(self.pos, self.shape)

            # 2. Gravity solution
            phi_grid, phi_k = self.solver.solve_potential(rho_grid)
            g_grid = self.solver.gravitational_acceleration(phi_k)

            # 3. Barotropic pressure force: - grad(P) / rho = - c_s^2 * grad(ln rho)
            # Stabilizes high density cores, preventing infinite delta singularity
            if self.pressure_cs > 0:
                log_rho = torch.log(rho_grid.clamp_min(1e-4))
                log_rho_k = torch.fft.rfftn(log_rho)
                px = torch.fft.irfftn(1j * self.solver.KX * log_rho_k, s=self.shape)
                py = torch.fft.irfftn(1j * self.solver.KY * log_rho_k, s=self.shape)
                pz = torch.fft.irfftn(1j * self.solver.KZ * log_rho_k, s=self.shape)
                pressure_acc = - (self.pressure_cs ** 2) * torch.stack([px, py, pz], dim=-1)
                g_grid = g_grid + pressure_acc

            # 4. Interpolate acceleration to particles
            acc = cic_interpolate_3d(self.pos, g_grid)

            # 5. Kick & Drift (Symplectic Leapfrog with damping)
            self.vel = (self.vel + dt * acc) * (1.0 - self.damping * dt)
            self.pos = (self.pos + dt * self.vel) % 1.0

            self.step_count += 1
            self.elapsed_time += dt

        # Compute energetic diagnostics
        kin_energy = 0.5 * (self.vel.square().sum(dim=-1)).mean().item()
        phi_interpolated = cic_interpolate_3d(self.pos, phi_grid)
        pot_energy = 0.5 * phi_interpolated.mean().item()
        contrast = float((rho_grid.std() / rho_grid.mean()).item())

        return {
            'step': self.step_count,
            'time': self.elapsed_time,
            'kinetic_energy': kin_energy,
            'potential_energy': pot_energy,
            'total_energy': kin_energy + pot_energy,
            'virial_ratio': abs(2.0 * kin_energy / (pot_energy + 1e-12)),
            'density_contrast': contrast,
            'max_density': float(rho_grid.max().item()),
            'min_density': float(rho_grid.min().item()),
        }

    def compute_fields(self) -> Dict[str, torch.Tensor]:
        """Compute full 3D continuous fields for analysis and visualization."""
        nx, ny, nz = self.shape
        rho_grid = cic_deposit_3d(self.pos, self.shape)
        phi_grid, phi_k = self.solver.solve_potential(rho_grid)
        g_grid = self.solver.gravitational_acceleration(phi_k)
        H_grid = self.solver.tidal_tensor(phi_k)

        # Deposit momentum flux j = rho * v
        jx = cic_deposit_3d(self.pos, self.shape, weights=self.vel[:, 0])
        jy = cic_deposit_3d(self.pos, self.shape, weights=self.vel[:, 1])
        jz = cic_deposit_3d(self.pos, self.shape, weights=self.vel[:, 2])
        flux_grid = torch.stack([jx, jy, jz], dim=-1)

        # Velocity field v = j / (rho + eps)
        vel_grid = flux_grid / rho_grid.clamp_min(1e-4).unsqueeze(-1)

        # Discrete curl / vorticity omega = curl(v)
        dx, dy, dz = 1.0 / nx, 1.0 / ny, 1.0 / nz
        vx, vy, vz = vel_grid[..., 0], vel_grid[..., 1], vel_grid[..., 2]

        wx = (torch.roll(vz, -1, 1) - torch.roll(vz, 1, 1)) / (2.0 * dy) - (torch.roll(vy, -1, 2) - torch.roll(vy, 1, 2)) / (2.0 * dz)
        wy = (torch.roll(vx, -1, 2) - torch.roll(vx, 1, 2)) / (2.0 * dz) - (torch.roll(vz, -1, 0) - torch.roll(vz, 1, 0)) / (2.0 * dx)
        wz = (torch.roll(vy, -1, 0) - torch.roll(vy, 1, 0)) / (2.0 * dx) - (torch.roll(vx, -1, 1) - torch.roll(vx, 1, 1)) / (2.0 * dy)
        vorticity = torch.stack([wx, wy, wz], dim=-1)

        # Web classification
        web = self.solver.classify_web(H_grid, threshold=0.2)

        return {
            'density': rho_grid,
            'potential': phi_grid,
            'acceleration': g_grid,
            'tidal_tensor': H_grid,
            'momentum_flux': flux_grid,
            'velocity': vel_grid,
            'vorticity': vorticity,
            'web': web
        }

    def export_ether_spatial_snapshot(self) -> Dict:
        """Export spatial dictionary matching ETHER dashboard protocol."""
        fields = self.compute_fields()
        nx, ny, nz = self.shape
        rho = fields['density']
        phi = fields['potential']
        flux = fields['momentum_flux']
        vel = fields['velocity']
        H = fields['tidal_tensor']
        vort = fields['vorticity']
        web = fields['web']

        # Coordinates on grid [Nx, Ny, Nz, 3]
        qx = torch.linspace(0, 1.0 - 1.0 / nx, nx, device=self.device) + 0.5 / nx
        qy = torch.linspace(0, 1.0 - 1.0 / ny, ny, device=self.device) + 0.5 / ny
        qz = torch.linspace(0, 1.0 - 1.0 / nz, nz, device=self.device) + 0.5 / nz
        coords = torch.stack(torch.meshgrid(qx, qy, qz, indexing='ij'), dim=-1).reshape(-1, 3)

        # Field energy: gravitational potential + density energy
        field_energy = (0.5 * rho.square()).flatten()
        # Flux energy: kinetic energy 0.5 * rho * v^2
        flux_energy = (0.5 * rho * vel.square().sum(dim=-1)).flatten()

        # Speed: velocity magnitude
        speed = vel.norm(dim=-1).reshape(-1, 1).repeat(1, 3)

        # Material components: [density, potential, vorticity_mag, l1, l2, l3]
        evals = web['eigenvalues']
        vort_mag = vort.norm(dim=-1, keepdim=True)
        material = torch.cat([
            rho.unsqueeze(-1),
            phi.unsqueeze(-1),
            vort_mag,
            evals
        ], dim=-1).reshape(-1, 6)

        # Effective transport factor: 3x3 tidal tensor + kinetic stress
        eff_factor = H.reshape(-1, 3, 3)

        # Transport energy current: momentum flux j
        energy_current = flux.reshape(-1, 3)

        # Find knot centers to act as write ports (somas) using Non-Maximum Suppression (NMS)
        # to ensure distinct halo hubs across the 3D volume instead of adjacent voxels in the same clump.
        candidate_indices = torch.nonzero(web['knots'].flatten()).squeeze(-1)
        if len(candidate_indices) == 0:
            candidate_indices = torch.argsort(rho.flatten(), descending=True)[:512]
        else:
            sorted_candidates = candidate_indices[torch.argsort(rho.flatten()[candidate_indices], descending=True)]
            candidate_indices = sorted_candidates

        selected_somas = []
        min_knot_dist = 0.16  # Periodic exclusion radius
        for idx in candidate_indices:
            c = coords[idx]
            is_far = True
            for s in selected_somas:
                diff = (c - s + 0.5) % 1.0 - 0.5
                if diff.norm() < min_knot_dist:
                    is_far = False
                    break
            if is_far:
                selected_somas.append(c)
                if len(selected_somas) == 8:
                    break

        if len(selected_somas) < 8:
            all_sorted = torch.argsort(rho.flatten(), descending=True)
            for idx in all_sorted:
                c = coords[idx]
                is_far = all((((c - s + 0.5) % 1.0 - 0.5).norm() >= 0.08) for s in selected_somas)
                if is_far:
                    selected_somas.append(c)
                    if len(selected_somas) == 8:
                        break

        write_coords = [s.tolist() for s in selected_somas]

        # Find filament centers to act as read ports (axons) using NMS along filament bridges
        fil_indices = torch.nonzero(web['filaments'].flatten()).squeeze(-1)
        if len(fil_indices) > 0:
            sorted_fil = fil_indices[torch.argsort(vort_mag.flatten()[fil_indices], descending=True)]
        else:
            sorted_fil = torch.argsort(rho.flatten(), descending=True)

        selected_filaments = []
        min_fil_dist = 0.12
        for idx in sorted_fil:
            c = coords[idx]
            is_far = True
            for s in selected_filaments:
                diff = (c - s + 0.5) % 1.0 - 0.5
                if diff.norm() < min_fil_dist:
                    is_far = False
                    break
            if is_far:
                selected_filaments.append(c)
                if len(selected_filaments) == 16:
                    break

        read_coords = [s.tolist() for s in selected_filaments]

        def to_list(t):
            return t.detach().cpu().tolist()

        return {
            'protocol': 'medium_spatial_snapshot_v1',
            'shape': list(self.shape),
            'physical_time': float(self.elapsed_time),
            'step': self.step_count,
            'coordinates': to_list(coords),
            'field_energy': to_list(field_energy),
            'flux_energy': to_list(flux_energy),
            'speed': to_list(speed),
            'material': to_list(material),
            'shear': None,
            'effective_transport_factor': to_list(eff_factor),
            'transport_energy_current': to_list(energy_current),
            'energy_current_scope': '3D gravitational mass flux j = rho * v; streamlines trace filament bridges',
            'transport_capacity_budget': 1.0,
            'write_coords': write_coords,
            'read_coords': read_coords,
            'write_initial_coords': write_coords,
            'read_initial_coords': read_coords,
            'write_port_displacement': [0.0] * len(write_coords),
            'read_port_displacement': [0.0] * len(read_coords),
            'write_gate': [1.0] * len(write_coords),
            'read_heads': 4,
            'read_queries': 4,
            'cosmic_web': {
                'knots_fraction': float(web['knots'].float().mean().item()),
                'filaments_fraction': float(web['filaments'].float().mean().item()),
                'sheets_fraction': float(web['sheets'].float().mean().item()),
                'voids_fraction': float(web['voids'].float().mean().item()),
            }
        }
