"""Unit tests for self-gravitational Poisson solver, CIC deposition, and cosmic web classification."""
import math
import pytest
import torch
from information_boltzmann.core.self_gravity import (
    FourierPoissonSolver3D,
    cic_deposit_3d,
    cic_interpolate_3d,
    GravitationalCosmos3D
)


def test_fourier_poisson_analytic_solution():
    """Verify Fourier Poisson solver matches analytic solution on 1D sinusoidal mode in 3D box."""
    N = 32
    G = 1.0
    solver = FourierPoissonSolver3D((N, N, N), G=G, device='cpu')

    # Construct rho(x, y, z) = sin(2 * pi * x)
    x = torch.linspace(0, 1.0 - 1.0 / N, N)
    X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
    rho = torch.sin(2.0 * math.pi * X)

    phi, phi_k = solver.solve_potential(rho)
    g = solver.gravitational_acceleration(phi_k)

    # Analytic potential: Phi = - 4*pi*G / (2*pi)^2 * sin(2*pi*x) = - G / pi * sin(2*pi*x)
    phi_analytic = - (G / math.pi) * torch.sin(2.0 * math.pi * X)
    # Analytic acceleration: g_x = - dPhi/dx = 2 * G * cos(2*pi*x), g_y = 0, g_z = 0
    gx_analytic = 2.0 * G * torch.cos(2.0 * math.pi * X)

    assert torch.allclose(phi, phi_analytic, atol=1e-5, rtol=1e-5), "Numerical potential must match analytic solution"
    assert torch.allclose(g[..., 0], gx_analytic, atol=1e-5, rtol=1e-5), "X-acceleration must match analytic derivative"
    assert torch.allclose(g[..., 1], torch.zeros_like(g[..., 1]), atol=1e-5), "Y-acceleration must be zero"
    assert torch.allclose(g[..., 2], torch.zeros_like(g[..., 2]), atol=1e-5), "Z-acceleration must be zero"


def test_cloud_in_cell_exact_mass_conservation():
    """Verify Cloud-in-Cell deposition preserves exact total mass."""
    N = 16
    Np = 2000
    torch.manual_seed(123)
    pos = torch.rand(Np, 3)

    grid = cic_deposit_3d(pos, (N, N, N))
    assert math.isclose(float(grid.sum().item()), float(Np), rel_tol=1e-5), "Deposited mass must equal particle count"


def test_gravitational_cosmos_evolution_and_ether_export():
    """Verify GravitationalCosmos3D evolution, structure formation, and ETHER snapshot format."""
    cosmos = GravitationalCosmos3D(shape=(16, 16, 16), G=1.0, dt=0.05, damping=0.02, seed=42, device='cpu')

    diag = cosmos.advance(substeps=20)
    assert diag['density_contrast'] > 0.05, "Density contrast must grow under self-gravity"
    assert diag['max_density'] > 1.5, "Peak density must increase"

    # Export ETHER snapshot
    snapshot = cosmos.export_ether_spatial_snapshot()
    assert snapshot['protocol'] == 'medium_spatial_snapshot_v1'
    assert len(snapshot['coordinates']) == 16**3
    assert len(snapshot['field_energy']) == 16**3
    assert len(snapshot['flux_energy']) == 16**3
    assert len(snapshot['transport_energy_current']) == 16**3

    # Cosmic web fractions
    web = snapshot['cosmic_web']
    total_frac = web['knots_fraction'] + web['filaments_fraction'] + web['sheets_fraction'] + web['voids_fraction']
    assert math.isclose(total_frac, 1.0, rel_tol=1e-4), "Web fractions must partition the volume to 100%"
    assert web['filaments_fraction'] > 0.0, "Filamentary structures must spontaneously emerge"
