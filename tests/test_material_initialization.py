import torch

from information_boltzmann.core.plastic_medium import ContinuousMaterial, complete_periodic_modes


def grid(shape):
    return torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float64) / n
                                       for n in shape], indexing='ij'), -1)


def test_spectral_xavier_is_centered_calibrated_and_refinement_consistent():
    modes, sine = complete_periodic_modes((8, 8, 4))
    torch.manual_seed(449)
    material = ContinuousMaterial(8, modes, sine).double()
    material.initialize_spectral_xavier()
    values = material(grid((8, 8, 4)))
    torch.testing.assert_close(values.mean((0, 1, 2)), torch.zeros(8, dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(values.reshape(-1, 8).std(0, unbiased=False), torch.ones(8, dtype=torch.float64))
    torch.testing.assert_close(material(grid((16, 16, 8)))[::2, ::2, ::2], values)
    assert torch.count_nonzero(material.coefficients[0]) == 0
    values.square().mean().backward()
    assert torch.isfinite(material.coefficients.grad).all()


def test_spectral_xavier_reproducible():
    torch.manual_seed(42)
    first = ContinuousMaterial()
    first.initialize_spectral_xavier()
    torch.manual_seed(42)
    second = ContinuousMaterial()
    second.initialize_spectral_xavier()
    torch.testing.assert_close(first.coefficients, second.coefficients)
