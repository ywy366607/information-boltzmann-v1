"""Numerical checks of augmented surrogate algebra, not capability studies."""

from contextlib import contextmanager

import torch
import numpy as np
import pytest

import information_boltzmann.core.fly_reservoir as reservoir
from test_fly_bptt_learning import make_model, physical


@contextmanager
def fixed_spike(value):
    original = reservoir.SpikeFn
    class FixedSpike:
        @staticmethod
        def apply(unused_margin, *unused):
            return value
    reservoir.SpikeFn = FixedSpike
    try:
        yield
    finally:
        reservoir.SpikeFn = original


def test_full_augmented_jacobian_separates_passive_and_event_feedback(tmp_path):
    torch.manual_seed(91)
    model = make_model(tmp_path, detach_reset=True).double()
    state = physical(model)
    # All channels and every delay slot are independent input coordinates.
    channels = [state.h, state.ge, state.gi, state.b, state.x, state.u, *state.ring]
    z = torch.cat([value.double().flatten() for value in channels])
    z[:6] = torch.linspace(-.04, .08, 6, dtype=torch.float64)
    drive = torch.zeros(1, 6, dtype=torch.float64)

    def context(value):
        h, ge, gi, b, x, u, *ring = value.reshape(10, 1, 6).unbind(0)
        return model.prepare_coba_tick(h, tuple(ring), ge, gi, b, x, u)

    def margin(value):
        c = context(value)
        return (c['alpha'] * c['h'] + c['beta_int'] *
                (c['base_current'] + drive) - c['eff_threshold']).flatten()

    def tick(value, spike=None):
        def run():
            h, s, ring, ge, gi, b, x, u = model.finish_coba_tick(context(value), drive)
            return torch.cat([v.flatten() for v in (h, ge, gi, b, x, u, *ring)])
        if spike is None:
            return run()
        with fixed_spike(spike.reshape(1, 6)):
            return run()

    y = margin(z).detach()
    s = (y >= 0).double()
    j = torch.autograd.functional.jacobian(tick, z)
    a = torch.autograd.functional.jacobian(lambda value: tick(value, s), z)
    b = torch.autograd.functional.jacobian(lambda spike: tick(z, spike), s)
    c = torch.autograd.functional.jacobian(margin, z)
    psi = 1 / (1 + (torch.pi * y)**2)
    torch.testing.assert_close(j, a + (b * psi[None, :]) @ c, rtol=1e-12, atol=1e-12)
    # Detached reset removes event feedback only from the h block. Slow
    # adaptation/STP and new pulse slot still have event dependence.
    assert torch.count_nonzero(b[:6]) == 0
    assert torch.count_nonzero(b[18:36]) > 0
    assert torch.count_nonzero(b[36:42]) > 0
    assert torch.count_nonzero((b * psi[None, :]) @ c) > 0
    assert torch.count_nonzero(s) == 0  # silent forward can carry proxy feedback


def test_nondimensional_similarity_preserves_feedback_eigenvalues():
    j = torch.tensor([[.95, .2], [.4, .8]], dtype=torch.float64)
    scale = torch.diag(torch.tensor([10., .1], dtype=torch.float64))
    transformed = scale @ j @ torch.linalg.inv(scale)
    old = torch.linalg.eigvals(j).real.sort().values
    new = torch.linalg.eigvals(transformed).real.sort().values
    torch.testing.assert_close(old, new, rtol=1e-12, atol=1e-12)
    assert old.max() > 1  # a units change alone does not remove this mode


def test_weighted_row_bound_allows_positive_event_credit_without_gain_growth():
    # Analytic sufficient-bound contract, not a calibrated real-brain model.
    a = torch.diag(torch.tensor([.9, .8], dtype=torch.float64))
    b = torch.tensor([[.2], [.4]], dtype=torch.float64)
    c = torch.tensor([[.3, .7]], dtype=torch.float64)
    psi = torch.ones(1, dtype=torch.float64)
    passive = a.abs().sum(1)
    feedback = ((b.abs() * psi) @ c.abs()).sum(1)
    chi = torch.minimum(torch.ones(1, dtype=torch.float64),
                        ((1 - passive) / feedback).min().reshape(1))
    j = a + (b * (chi * psi)) @ c
    assert 0 < chi.item() < 1
    assert j.abs().sum(1).max() <= 1
    assert torch.linalg.eigvals(j).abs().max() <= 1
    # Original event feedback need not satisfy the bound.
    assert (a + b @ c).abs().sum(1).max() > 1


@pytest.mark.parametrize('regime', ['silent', 'firing', 'clipped'])
def test_sparse_envelopes_cover_actual_full_state_jacobian_blocks(tmp_path, regime):
    from scripts.ib.fly_feedback_bound import tick_envelope, transmission_rows, common_certificate
    torch.manual_seed(43)
    model = make_model(tmp_path, detach_reset=True).double()
    initial = physical(model)
    z = torch.cat([v.double().flatten() for v in
                   (initial.h, initial.ge, initial.gi, initial.b, initial.x, initial.u, *initial.ring)])
    if regime != 'silent':
        z[:6] = torch.tensor([1., -.3, 1., -.3, 1., -.3], dtype=torch.float64)
    if regime == 'clipped':
        z[24:30] = 10.  # numerical saturation fixture, not a biological state claim
    drive = torch.zeros(1, 6, dtype=torch.float64)
    def context(value):
        h, ge, gi, b, x, u, *ring = value.reshape(10, 1, 6).unbind(0)
        return model.prepare_coba_tick(h, tuple(ring), ge, gi, b, x, u)
    def margin(value):
        c = context(value)
        return (c['alpha']*c['h']+c['beta_int']*(c['base_current']+drive)-c['eff_threshold']).flatten()
    s = (margin(z) >= 0).double()
    def tick(value, spike):
        with fixed_spike(spike.reshape(1, 6)):
            h, _, ring, ge, gi, b, x, u = model.finish_coba_tick(context(value), drive)
        return torch.cat([v.flatten() for v in (h, ge, gi, b, x, u, *ring)])
    a = torch.autograd.functional.jacobian(lambda value: tick(value, s), z)
    b = torch.autograd.functional.jacobian(lambda spike: tick(z, spike), s)
    c = torch.autograd.functional.jacobian(margin, z)
    context_value = context(z)
    outputs = model.finish_coba_tick(context_value, drive)
    bounds = tick_envelope(model, context_value, drive, outputs, transmission_rows(model, context_value['h']))
    def infinity_blocks(matrix, row_blocks, col_blocks):
        return np.array([[float(matrix[i*6:(i+1)*6, j*6:(j+1)*6].abs().sum(1).max())
                          for j in range(col_blocks)] for i in range(row_blocks)])
    for actual, bound in zip((infinity_blocks(a, 10, 10), infinity_blocks(b, 10, 1).flatten(),
                              infinity_blocks(c, 1, 10).flatten()), bounds[:3]):
        assert np.all(actual <= bound + 1e-12)
    certificate = common_certificate(*bounds)
    if certificate['feasible']:
        p = np.repeat(certificate['inverse_metric_weights'], 6)
        psi = 1/(1+(torch.pi*margin(z).detach())**2)
        j = a + (b * (psi * certificate['surrogate_coefficient'])[None]) @ c
        weighted = j.detach().numpy()*p[None, :]/p[:, None]
        assert np.max(np.abs(weighted).sum(1)) <= 1+1e-12


def test_clamp_endpoint_derivative_matches_envelope_convention():
    value = torch.tensor([3.-1e-5, 3., 3.+1e-5], dtype=torch.float64, requires_grad=True)
    value.clamp(max=3).sum().backward()
    assert torch.equal(value.grad, torch.tensor([1., 1., 0.], dtype=torch.float64))
