"""Analytic/numerical contracts, not synthetic capability benchmarks."""
import io
import math

import pytest
import torch

from information_boltzmann.core.temporal_probes import (
    CausalProbeFilterBank, TemporalProbeState, sample_compact_probes,
)
from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D


def bank(probes=1, channels=2):
    return CausalProbeFilterBank(probes, channels, [0.7, 2.3], [0.0, 1.9]).double()


def test_held_input_matches_augmented_matrix_exponential():
    f = bank()
    r = torch.tensor([[[0.8, -0.4]]], dtype=torch.float64)
    state = f.initial_state()
    state = TemporalProbeState(torch.full_like(state.value, 0.3 + 0.2j), state.elapsed)
    actual = f(r, state, 0.37)
    for j, (alpha, omega) in enumerate(zip(f.log_rate.exp(), f.frequency)):
        matrix = torch.zeros(2, 2, dtype=torch.complex128)
        matrix[0, 0], matrix[0, 1] = -torch.complex(alpha, omega), alpha
        propagator = torch.matrix_exp(matrix * 0.37)
        expected = propagator[0, 0] * state.value[:, :, j] + propagator[0, 1] * r
        torch.testing.assert_close(actual.value[:, :, j], expected, atol=1e-14, rtol=1e-14)


def test_constant_input_refinement_and_zero_time_identity():
    f = bank()
    r = torch.randn(2, 1, 2, dtype=torch.float64)
    s = f.initial_state(2)
    whole = f(r, s, 0.41)
    split = f(r, f(r, s, 0.13), 0.28)
    torch.testing.assert_close(whole.value, split.value, atol=1e-14, rtol=1e-14)
    zero = f(r, whole, 0.)
    torch.testing.assert_close(zero.value, whole.value, atol=0, rtol=0)
    torch.testing.assert_close(zero.elapsed, whole.elapsed, atol=0, rtol=0)


def test_sampled_oscillation_has_exact_zoh_transfer_and_signed_phase():
    f = CausalProbeFilterBank(1, 2, [2.], [1.7]).double()
    s = f.initial_state()
    dt, nu, steps = 0.04, 1.3, 500
    for n in range(steps):
        r = torch.tensor([[[math.cos(nu*n*dt), math.sin(nu*n*dt)]]], dtype=torch.float64)
        s = f(r, s, dt)
    lam = torch.complex(f.log_rate.exp()[0], f.frequency[0])
    decay = torch.exp(-lam*dt)
    gain = -f.log_rate.exp()[0] * torch.expm1(-lam*dt) / lam
    reference = gain / (1 - decay * torch.exp(torch.tensor(1j*nu*dt, dtype=torch.complex128)))
    measured = (s.value[0, 0, 0, 0] - 1j*s.value[0, 0, 0, 1]) / torch.exp(
        torch.tensor(-1j*nu*(steps-1)*dt, dtype=torch.complex128))
    torch.testing.assert_close(measured, reference, atol=2e-14, rtol=2e-14)
    assert abs(float(torch.angle(measured.detach()))) > 0.01


def test_equal_current_observation_can_have_distinct_histories():
    f = bank()
    one = torch.ones(1, 1, 2, dtype=torch.float64)
    a = f(one, f.initial_state(), 0.3)
    b = f(-one, f.initial_state(), 0.3)
    zero = torch.zeros_like(one)
    a, b = f(zero, a, 0.1), f(zero, b, 0.1)
    assert (a.value - b.value).abs().max() > 0.1


def test_equal_future_drive_contracts_history_difference():
    f = bank()
    a = TemporalProbeState(torch.ones_like(f.initial_state().value), f.initial_state().elapsed)
    b = TemporalProbeState(-a.value, a.elapsed)
    r = torch.randn(1, 1, 2, dtype=torch.float64)
    after_a, after_b = f(r, a, 0.7), f(r, b, 0.7)
    expected = (a.value-b.value).abs() * torch.exp(-f.log_rate.exp()*0.7)[None, None, :, None]
    torch.testing.assert_close((after_a.value-after_b.value).abs(), expected, atol=1e-14, rtol=1e-14)


def test_serialized_continuation_and_chunk_gradients_match():
    f = bank()
    torch.manual_seed(988)
    sequence = torch.randn(7, 1, 1, 2, dtype=torch.float64, requires_grad=True)
    def run(start, signals):
        for r in signals:
            start = f(r, start, 0.03)
        return start
    whole = run(f.initial_state(), sequence)
    prefix = run(f.initial_state(), sequence[:3])
    # In-memory continuation does not detach the live tape.
    split = run(TemporalProbeState.from_state_dict(prefix.state_dict()), sequence[3:])
    torch.testing.assert_close(whole.value, split.value, atol=0, rtol=0)
    a = torch.autograd.grad(whole.value.abs().square().sum(), [sequence, *f.parameters()], retain_graph=True)
    b = torch.autograd.grad(split.value.abs().square().sum(), [sequence, *f.parameters()], retain_graph=True)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    # Disk-style continuation preserves values, not a Python autograd history.
    buffer = io.BytesIO()
    torch.save({'model': f.state_dict(), 'state': prefix.detach().state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored = bank()
    restored.load_state_dict(saved['model'])
    s = TemporalProbeState.from_state_dict(saved['state'])
    for r in sequence[3:]:
        s = restored(r, s, 0.03)
    torch.testing.assert_close(s.value, whole.value, atol=0, rtol=0)
    torch.testing.assert_close(s.elapsed, whole.elapsed, atol=0, rtol=0)


def test_gradcheck_input_state_time_rate_and_frequency():
    from torch.func import functional_call
    f = bank()
    r = torch.randn(1, 1, 2, dtype=torch.float64, requires_grad=True)
    s = torch.randn(1, 1, 2, 2, dtype=torch.complex128, requires_grad=True)
    t = torch.tensor(0.08, dtype=torch.float64, requires_grad=True)
    def call(r, s, t, alpha, omega):
        out = functional_call(f, {'log_rate': alpha, 'frequency': omega},
                              (r, TemporalProbeState(s, t.new_zeros(1)), t))
        return torch.view_as_real(out.value)
    assert torch.autograd.gradcheck(call, (r, s, t, f.log_rate, f.frequency), atol=1e-6, rtol=1e-5)


def test_full_dtype_batch_shapes_and_small_intervals():
    f = bank().float()
    r = torch.ones(2, 1, 2)
    s = f(r, f.initial_state(2), torch.tensor([1e-10, 0.1]))
    assert s.value.isfinite().all() and s.value.dtype == torch.complex64
    assert s.value[0].abs().min() > 0
    assert CausalProbeFilterBank.features(s).shape == (2, 1, 2, 2, 2)


def test_spatial_probe_dependency_is_exactly_local():
    torch.manual_seed(989)
    model = PlasticMediumPorts3D(vocab_size=17, shape=(4,4,4), channels=8,
                                heads=2, queries=2, hidden=6, material_width=3).double()
    field = torch.randn_like(model.initial_belief().medium.field, requires_grad=True)
    pooled = sample_compact_probes(model.readout, field)
    f = bank(probes=4, channels=8)
    history = f(pooled, f.initial_state(), 0.08)
    loss = history.value[:,0].real.sum() + history.value[:,0].imag.sum()
    grad, coords = torch.autograd.grad(loss, [field, model.readout.probe_coords])
    mask = (model.readout.footprint()[0] > 0).reshape(1,4,4,4,1).expand_as(field)
    assert grad.masked_select(~mask).abs().max() == 0
    assert grad.masked_select(mask).abs().max() > 0
    assert coords[0].norm() > 0 and coords[1:].abs().max() == 0


@pytest.mark.parametrize('duration', [-0.1, float('nan'), float('inf')])
def test_invalid_duration_rejected(duration):
    f = bank()
    with pytest.raises(ValueError, match='duration'):
        f(torch.ones(1,1,2,dtype=torch.float64), f.initial_state(), duration)


def test_future_samples_do_not_affect_already_emitted_features():
    f = bank()
    signal = torch.ones(1,1,2,dtype=torch.float64)
    s = f(signal, f.initial_state(), .1)
    before = f.features(s).clone()
    later = f(signal * -100, s, .2)
    torch.testing.assert_close(f.features(s), before, atol=0, rtol=0)
    assert (later.value-s.value).abs().max() > 0


@pytest.mark.parametrize('duration', [-0.1, float('nan'), float('inf')])
def test_invalid_tensor_duration_rejected(duration):
    f = bank()
    with pytest.raises(RuntimeError, match='duration'):
        f(torch.ones(1,1,2,dtype=torch.float64), f.initial_state(), torch.tensor(duration))


def test_underflowed_rate_is_rejected_instead_of_silent_zero_division():
    f = bank()
    with torch.no_grad():
        f.log_rate.fill_(-1000.)
    with pytest.raises(RuntimeError, match='rate'):
        f(torch.ones(1,1,2,dtype=torch.float64), f.initial_state(), 0.1)


def test_float32_history_has_double_precision_physical_clock():
    f = bank().float()
    s = f.initial_state()
    reference = torch.zeros(1, dtype=torch.float64)
    r = torch.ones(1,1,2)
    for _ in range(1000):
        s = f(r, s, 0.005)
        reference = reference + 0.005
    assert s.elapsed.dtype == torch.float64
    torch.testing.assert_close(s.elapsed, reference, atol=0, rtol=0)


@pytest.mark.parametrize('component', ['signal', 'history', 'elapsed'])
def test_nonfinite_input_state_fails_fast(component):
    f = bank()
    signal = torch.ones(1,1,2,dtype=torch.float64)
    state = f.initial_state()
    if component == 'signal':
        signal.fill_(float('nan'))
    elif component == 'history':
        state.value.fill_(complex(float('inf'),0))
    else:
        state.elapsed.fill_(float('nan'))
    with pytest.raises(RuntimeError, match='Finite'):
        f(signal, state, 0.01)


def test_nonrepresentable_rate_duration_product_fails_fast():
    f = bank().float()
    with torch.no_grad():
        f.log_rate.fill_(80.)
    with pytest.raises(RuntimeError, match='product'):
        f(torch.ones(1,1,2), f.initial_state(), 1e10)
