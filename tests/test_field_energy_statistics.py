"""Numerical monitoring decomposition, independent of model capability."""
import torch

from information_boltzmann.evaluation import field_energy_statistics


def test_dc_and_spatial_energy_decompose_each_sample():
    torch.manual_seed(19)
    field = torch.randn(2, 8, 8, 4, 7, dtype=torch.float64)
    field[0] += 2
    field[1] -= 3
    stats = field_energy_statistics(field)
    assert abs(stats["field_energy"] - stats["dc_energy"] - stats["spatial_energy"]) < 1e-12
    assert abs(stats["dc_share"] + stats["spatial_share"] - 1) < 1e-12
    spectrum = torch.fft.fftn(field, dim=(1, 2, 3), norm="ortho")
    share = spectrum[:, 0, 0, 0].abs().square().sum() / spectrum.abs().square().sum()
    assert abs(stats["dc_share"] - float(share)) < 1e-12


def test_uniform_and_empty_field_statistics():
    field = torch.ones(1, 8, 8, 4, 7, dtype=torch.float64)
    stats = field_energy_statistics(field)
    assert stats["dc_share"] == 1
    assert stats["spatial_energy"] == 0
    assert all(value == 0 for value in field_energy_statistics(field * 0).values())
