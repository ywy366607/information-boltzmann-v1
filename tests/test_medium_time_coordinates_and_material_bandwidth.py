"""CPU numerical migrations; no learning/capability experiments."""
import copy

import pytest
import torch

from information_boltzmann.core.plastic_medium import (
    ContinuousMaterial, PlasticMedium3D, complete_periodic_modes,
)
from information_boltzmann.core.temporal_probes import (
    CausalProbeFilterBank, TemporalProbeReadout, TemporalProbeState,
)


def filter_bank(time_reference=1.0):
    return CausalProbeFilterBank(1, 2, [2., 8.], [0., 37.],
                                 time_reference=time_reference).double()


def grid(shape):
    return torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float64) / n
                                       for n in shape], indexing='ij'), -1)


def test_legacy_frequency_load_preserves_filter_and_scales_coordinate_gradient():
    old = filter_bank()
    legacy = {key: value.clone() for key, value in old.state_dict().items()
              if key != 'frequency_time_reference'}
    new = filter_bank(.005)
    new.load_state_dict(legacy, strict=True)
    torch.testing.assert_close(new.physical_frequency, old.frequency, atol=1e-14, rtol=1e-14)
    signal = torch.tensor([[[.7, -.2]]], dtype=torch.float64, requires_grad=True)
    initial = old.initial_state()
    initial = TemporalProbeState(torch.full_like(initial.value, .3 + .2j), initial.elapsed)
    expected, actual = old(signal, initial, .005), new(signal, initial, .005)
    torch.testing.assert_close(actual.value, expected.value, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(actual.elapsed, expected.elapsed, atol=0, rtol=0)
    old_grads = torch.autograd.grad(expected.value.real.sum(), (signal, old.log_rate, old.frequency))
    new_grads = torch.autograd.grad(actual.value.real.sum(), (signal, new.log_rate, new.frequency))
    for before, after in zip(old_grads[:2], new_grads[:2]):
        torch.testing.assert_close(after, before, atol=1e-14, rtol=1e-14)
    assert old_grads[2].abs().sum() > 0
    torch.testing.assert_close(new_grads[2], old_grads[2] / new.frequency_time_reference,
                               atol=1e-13, rtol=1e-13)
    # The caller's old payload remains in physical units.
    torch.testing.assert_close(legacy['frequency'], old.frequency, atol=0, rtol=0)


def test_default_reference_keeps_legacy_values_and_new_checkpoint_keeps_units():
    original = filter_bank()
    legacy = {key: value.clone() for key, value in original.state_dict().items()
              if key != 'frequency_time_reference'}
    restored = filter_bank()
    restored.load_state_dict(legacy, strict=True)
    for before, after in zip(original.parameters(), restored.parameters()):
        torch.testing.assert_close(after, before, atol=0, rtol=0)
    scaled = filter_bank(.005)
    continued = filter_bank(1.0)
    continued.load_state_dict(copy.deepcopy(scaled.state_dict()), strict=True)
    torch.testing.assert_close(continued.frequency_time_reference,
                               scaled.frequency_time_reference, atol=0, rtol=0)
    signal = torch.ones(1, 1, 2, dtype=torch.float64)
    actual = continued(signal, continued.initial_state(), .017)
    expected = scaled(signal, scaled.initial_state(), .017)
    torch.testing.assert_close(actual.value, expected.value, atol=0, rtol=0)


def test_nested_legacy_readout_loads_strictly_in_physical_frequency_units():
    old = TemporalProbeReadout(2, 3, [2., 8.], [0., 37.]).double()
    new = TemporalProbeReadout(2, 3, [2., 8.], [0., 37.], time_reference=.005).double()
    legacy = {key: value.clone() for key, value in old.state_dict().items()
              if key != 'bank.frequency_time_reference'}
    new.load_state_dict(legacy, strict=True)
    signal = torch.randn(1, 2, 6, dtype=torch.float64)
    expected = old.bank(signal, old.bank.initial_state(), .014)
    actual = new.bank(signal, new.bank.initial_state(), .014)
    torch.testing.assert_close(new(actual), old(expected), atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize('optimizer_type', [torch.optim.Adam, torch.optim.AdamW])
def test_explicit_frequency_branch_migrates_adam_moments_and_pending_gradients(optimizer_type):
    bank = filter_bank()
    optimizer = optimizer_type(bank.parameters(), lr=2e-4, amsgrad=True)
    bank.frequency.grad = torch.tensor([.3, -.7], dtype=torch.float64)
    bank.log_rate.grad = torch.tensor([.2, .4], dtype=torch.float64)
    optimizer.state[bank.frequency] = {
        'step': torch.tensor(17.),
        'exp_avg': torch.tensor([.1, -.2], dtype=torch.float64),
        'exp_avg_sq': torch.tensor([.2, .5], dtype=torch.float64),
        'max_exp_avg_sq': torch.tensor([.3, .6], dtype=torch.float64),
    }
    before = copy.deepcopy(optimizer.state[bank.frequency])
    omega = bank.physical_frequency.detach().clone()
    gradient = bank.frequency.grad.clone()
    signal = torch.tensor([[[.7, -.2]]], dtype=torch.float64)
    history = bank(signal, bank.initial_state(), .02).detach()
    expected = bank(signal, history, .013)
    record = bank.reparameterize_frequency(.005, optimizer=optimizer)
    torch.testing.assert_close(bank.physical_frequency, omega, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(bank.frequency.grad, gradient / .005, atol=0, rtol=0)
    state = optimizer.state[bank.frequency]
    torch.testing.assert_close(state['exp_avg'], before['exp_avg'] / .005, atol=0, rtol=0)
    for key in ('exp_avg_sq', 'max_exp_avg_sq'):
        torch.testing.assert_close(state[key], before[key] / .005 / .005, atol=0, rtol=0)
    torch.testing.assert_close(state['step'], before['step'], atol=0, rtol=0)
    torch.testing.assert_close(bank.log_rate.grad, torch.tensor([.2, .4], dtype=torch.float64), atol=0, rtol=0)
    assert optimizer.param_groups[0]['lr'] == 2e-4
    assert record['migration'] == 'explicit_frequency_coordinate_branch'
    assert record['coordinate_scale'] == .005
    actual = bank(signal, history, .013)
    torch.testing.assert_close(actual.value, expected.value, atol=1e-14, rtol=1e-14)
    # A round trip retains complete moment and pending-gradient history.
    bank.reparameterize_frequency(1., optimizer=optimizer)
    for key, value in before.items():
        torch.testing.assert_close(optimizer.state[bank.frequency][key], value, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(bank.frequency.grad, gradient, atol=1e-14, rtol=1e-14)


def test_frequency_branch_validation_is_atomic_and_rejects_wrong_optimizer():
    bank = filter_bank()
    before = copy.deepcopy(bank.state_dict())
    with pytest.raises(TypeError, match='Adam'):
        bank.reparameterize_frequency(.005, optimizer=torch.optim.SGD(bank.parameters(), lr=.1))
    other = filter_bank()
    with pytest.raises(ValueError, match='does not own'):
        bank.reparameterize_frequency(.005, optimizer=torch.optim.Adam(other.parameters()))
    optimizer = torch.optim.Adam(bank.parameters())
    optimizer.state[bank.frequency] = {'exp_avg': torch.ones(3, dtype=torch.float64)}
    with pytest.raises(ValueError, match='shape mismatch'):
        bank.reparameterize_frequency(.005, optimizer=optimizer)
    for key, value in before.items():
        torch.testing.assert_close(bank.state_dict()[key], value, atol=0, rtol=0)


@pytest.mark.parametrize('reference', [0., -1., float('nan'), float('inf'), 1e-100])
def test_invalid_frequency_reference_is_rejected(reference):
    with pytest.raises(ValueError, match='reference'):
        filter_bank(reference)


def test_runtime_complete_material_is_explicit_and_initialization_uses_its_grid():
    fixed = PlasticMedium3D((8, 8, 8), channels=2, hidden=4)
    complete = PlasticMedium3D((8, 8, 8), channels=2, hidden=4,
                               material_reference_shape=None).double()
    assert fixed.material.coefficients.shape == (256, 8)
    assert complete.material.coefficients.shape == (512, 8)
    assert complete.material.reference_shape == (8, 8, 8)
    assert complete.material_reference_shape == (8, 8, 8)
    complete.material.initialize_spectral_xavier(field_std=.7)
    values = complete.material(grid((8, 8, 8))).reshape(-1, 8)
    torch.testing.assert_close(values.mean(0), torch.zeros(8, dtype=torch.float64),
                               atol=1e-13, rtol=0)
    torch.testing.assert_close(values.std(0, unbiased=False),
                               torch.full((8,), .7, dtype=torch.float64), atol=1e-13, rtol=1e-13)


def test_bandwidth_expansion_preserves_continuous_field_with_zero_new_coefficients():
    torch.manual_seed(572)
    old_modes, old_sine = complete_periodic_modes((8, 8, 4))
    new_modes, new_sine = complete_periodic_modes((8, 8, 8))
    old = ContinuousMaterial(3, old_modes, old_sine).double()
    new = ContinuousMaterial(3, new_modes, new_sine, reference_shape=(8, 8, 8)).double()
    with torch.no_grad():
        old.coefficients.normal_(std=.1)
        new.coefficients.fill_(9.)
    new.load_expanded_state_dict(copy.deepcopy(old.state_dict()))
    points = torch.rand(43, 3, dtype=torch.float64)
    torch.testing.assert_close(new(points), old(points), atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(new(grid((8, 8, 8))), old(grid((8, 8, 8))),
                               atol=2e-14, rtol=2e-14)
    assert torch.count_nonzero(new.coefficients).item() == old.coefficients.numel()
    # The added bandwidth is learnable even though its initial residual is zero.
    new(points).square().sum().backward()
    added = new.coefficients.detach() == 0
    assert new.coefficients.grad[added].abs().sum() > 0


def test_bandwidth_migration_handles_true_fourier_signs_and_rejects_missing_modes():
    old = ContinuousMaterial(2, [(1, 0, 0)], [(1, 0, 0)]).double()
    new = ContinuousMaterial(2, [(-1, 0, 0)], [(-1, 0, 0)]).double()
    with torch.no_grad():
        old.coefficients.copy_(torch.tensor([[1., 2.], [.3, .4], [.5, .6]], dtype=torch.float64))
    new.load_expanded_state_dict(old.state_dict())
    torch.testing.assert_close(new.coefficients[1], old.coefficients[1], atol=0, rtol=0)
    torch.testing.assert_close(new.coefficients[2], -old.coefficients[2], atol=0, rtol=0)
    points = torch.rand(11, 3, dtype=torch.float64)
    torch.testing.assert_close(new(points), old(points), atol=1e-14, rtol=1e-14)
    incompatible = ContinuousMaterial(2, [(0, 1, 0)], [(0, 1, 0)]).double()
    with pytest.raises(ValueError, match='missing source mode'):
        incompatible.load_expanded_state_dict(old.state_dict())
