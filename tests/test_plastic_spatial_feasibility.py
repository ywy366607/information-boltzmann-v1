"""Actual inverse parameter synthesis and localized operator modes, no training."""
import pytest

from information_boltzmann.core.plastic_feasibility import spatial_heterogeneity_certificate


@pytest.fixture(scope='module')
def certificate():
    return spatial_heterogeneity_certificate()


def test_weak_compartment_is_in_the_actual_material_parameter_family(certificate):
    result = certificate['compartment']
    assert result['speed_synthesis_max_error'] < 1e-12
    assert result['contrast_rayleigh_quotient'] == pytest.approx(result['analytic_rayleigh_quotient'], abs=2e-12)
    # Min-max principle: the first eigenvalue cannot exceed a zero-mean trial quotient.
    assert 0 < result['first_nonzero_eigenvalue'] <= result['contrast_rayleigh_quotient'] + 1e-11
    assert result['contrast_rayleigh_quotient'] < result['homogeneous_first_nonzero_eigenvalue'] / 1000


def test_local_resonance_is_an_eigenmode_of_the_heterogeneous_wave_operator(certificate):
    result = certificate['localized_resonance']
    assert result['speed_synthesis_max_error'] < 1e-12
    assert result['two_cell_mode_energy_fraction'] > 0.999
    assert result['mode_ipr'] > 0.49
    assert result['mode_ipr'] > 100 * result['plane_wave_ipr']
