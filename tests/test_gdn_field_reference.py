"""CPU numerical/interface checks only; no model loads or capability claims."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from torch.nn import functional as F

from scripts.ib_local.gdn_field_reference import (
    conv_step_jvp, delta_read_jvp, gates_jvp, normalize_jvp, softmax_read_jvp,
)
from scripts.ib_local.pretrained_gdn import streaming_short_convolution


def tensor(x):
    return torch.tensor(x, dtype=torch.float64)


def centered(fn, xs, ds, h=1e-6):
    return (fn(*(x + h * d for x, d in zip(xs, ds))) -
            fn(*(x - h * d for x, d in zip(xs, ds)))) / (2 * h)


@pytest.mark.parametrize('branch', ['recurrent', 'chunk', 'recall'])
@pytest.mark.parametrize('x', [[0.4, -0.8, 0.1], [0.01, 0.02, -0.01]])
def test_normalize_derivative_and_gain(branch, x):
    x, dx, eps = np.array(x), np.array([0.2, 0.3, -0.7]), 0.1
    value, tangent, gain = normalize_jvp(x, dx, branch=branch, eps=eps)

    def reference(t):
        n = torch.linalg.vector_norm(t)
        denominator = {'recurrent': n + eps,
                       'chunk': torch.sqrt(t @ t + eps),
                       'recall': torch.clamp(n, min=eps)}[branch]
        return t / denominator

    expected, derivative = torch.autograd.functional.jvp(reference, tensor(x), tensor(dx))
    assert_allclose(value, expected.numpy(), rtol=1e-13)
    assert_allclose(tangent, derivative.numpy(), rtol=1e-12, atol=1e-14)
    finite = centered(lambda z: normalize_jvp(z, dx, branch=branch, eps=eps)[0],
                      [x], [dx])
    assert_allclose(tangent, finite, rtol=1e-8, atol=1e-9)
    assert np.linalg.norm(tangent) <= gain * np.linalg.norm(dx) + 1e-12


@pytest.mark.parametrize('branch,denominator', [('recurrent', 1e-6),
                                               ('chunk', 1e-3), ('recall', 1e-6)])
def test_zero_normalize(branch, denominator):
    dx = np.array([1., -2.])
    value, tangent, gain = normalize_jvp(np.zeros(2), dx, branch=branch)
    assert_allclose(value, 0.)
    assert_allclose(tangent, dx / denominator)
    assert_allclose(gain, 1 / denominator)
    # The recurrent vector derivative exists although the scalar norm has a cusp.
    h = 1e-13
    assert_allclose(tangent, centered(lambda x: normalize_jvp(x, dx, branch=branch)[0],
                                     [np.zeros(2)], [dx], h=h), rtol=3e-7)


def test_strict_branch_boundaries():
    with pytest.raises(ValueError, match='norm == eps'):
        normalize_jvp([0.1, 0.], [1., 0.], branch='recall', eps=0.1)
    with pytest.raises(ValueError, match='input == 20'):
        gates_jvp(np.array([0., 19.]), 0., -3., np.array([0., 1.]))
    for z in (np.nextafter(20., -np.inf), np.nextafter(20., np.inf)):
        assert np.isfinite(gates_jvp(z, 0., -3., 0., da=1.)['dalpha'])


def test_gates_mixed_directions_match_torch_and_finite_difference():
    xs = [np.array([-30., -0.2, 19., 21., 1000.]),
          np.array([-1000., -0.6, 0.7, 1.4, 1000.]),
          np.array([-2., 0.1, -4., -5., -8.]), np.zeros(5)]
    ds = [np.array([0.2, -0.3, 0.1, 0.7, -0.4]), np.full(5, 0.4),
          np.full(5, -0.2), np.full(5, 0.1)]
    result = gates_jvp(*xs, *ds)

    def reference(a, b, A, dt):
        return torch.stack((torch.exp(-A.exp() * F.softplus(a + dt)), b.sigmoid()))

    value, derivative = torch.autograd.functional.jvp(
        reference, tuple(map(tensor, xs)), tuple(map(tensor, ds)))
    assert_allclose([result['alpha'], result['beta']], value.numpy(), atol=1e-14)
    assert_allclose([result['dalpha'], result['dbeta']], derivative.numpy(), atol=1e-14)
    finite = centered(lambda *x: np.array([gates_jvp(*x)['alpha'], gates_jvp(*x)['beta']]),
                      xs, ds)
    assert_allclose([result['dalpha'], result['dbeta']], finite, rtol=1e-6, atol=1e-10)


def delta_fixture(keydim=3):
    rng = np.random.default_rng(421)
    xs = [rng.normal(size=(keydim, 2)), rng.normal(size=keydim),
          rng.normal(size=keydim), rng.normal(size=2), 0.7, 0.6]
    ds = [rng.normal(size=np.shape(x)) for x in xs]
    return xs, ds


def run_delta(xs, ds, scale=-0.8):
    return delta_read_jvp(*xs, *ds[:4], dalpha=ds[4], dbeta=ds[5], scale=scale)


@pytest.mark.parametrize('keydim', [1, 3])
def test_delta_joint_derivative_and_gain_rows(keydim):
    xs, ds = delta_fixture(keydim)
    result = run_delta(xs, ds)

    def reference(S, q, k, v, alpha, beta):
        A = alpha * S
        state = A + beta * torch.outer(k, v - k @ A)
        return state, -0.8 * (q @ state)

    values, derivatives = torch.autograd.functional.jvp(
        reference, tuple(map(tensor, xs)), tuple(map(tensor, ds)))
    for i, name in enumerate(('state', 'output')):
        assert_allclose(result[name], values[i].numpy(), atol=1e-14)
        assert_allclose(result['d' + name], derivatives[i].numpy(), atol=2e-14)
        finite = centered(lambda *x: run_delta(x, ds)[name], xs, ds)
        assert_allclose(result['d' + name], finite, rtol=1e-7, atol=1e-9)
        row = result['gain_rows'][name]
        assert set(row) == {'S', 'q', 'k', 'v', 'alpha', 'beta'}
        bound = sum(row[field] * np.linalg.norm(d)
                    for field, d in zip(('S', 'q', 'k', 'v', 'alpha', 'beta'), ds))
        assert np.linalg.norm(result['d' + name]) <= bound + 1e-12
    expected = abs(xs[4]) * abs(1 - xs[5] * (xs[2] @ xs[2]))
    if keydim >= 2:
        expected = max(abs(xs[4]), expected)
    assert_allclose(result['gain_rows']['state']['S'], expected)
    # Individual gain columns must bound each isolated derivative as well.
    for j, field in enumerate(('S', 'q', 'k', 'v', 'alpha', 'beta')):
        isolated = [np.zeros_like(d) for d in ds]
        isolated[j] = ds[j]
        single = run_delta(xs, isolated)
        for name in ('state', 'output'):
            assert np.linalg.norm(single['d' + name]) <= (
                result['gain_rows'][name][field] * np.linalg.norm(ds[j]) + 1e-12)


def test_source_zero_transport_is_separate_from_additive_new_source():
    xs, ds = delta_fixture()
    transport = ds[:4] + [0., 0.]
    newsource = [np.zeros_like(d) for d in ds[:4]] + ds[4:]
    full, old, new = (run_delta(xs, d) for d in (ds, transport, newsource))
    for name in ('dstate', 'doutput'):
        assert np.linalg.norm(old[name]) > 0
        assert np.linalg.norm(new[name]) > 0
        assert_allclose(full[name], old[name] + new[name], atol=1e-14)
    # Frozen upper gate parameters still transport old state/input directions.
    upper_xs = [full['state'], xs[1], xs[2], xs[3], 0.9, 0.3]
    upper_ds = [old['dstate'], ds[1], ds[2], ds[3], 0., 0.]
    upper = run_delta(upper_xs, upper_ds)
    finite = centered(lambda *x: run_delta(x, upper_ds)['output'], upper_xs, upper_ds)
    assert_allclose(upper['doutput'], finite, atol=1e-9)
    assert np.linalg.norm(upper['doutput']) > 0


@pytest.mark.parametrize('width', [1, 4])
@pytest.mark.parametrize('activation', [None, 'silu', 'swish'])
def test_convolution_matches_streaming_helper_and_torch_jvp(width, activation):
    rng = np.random.default_rng(53)
    raw, cache, weight = rng.normal(size=2), rng.normal(size=(2, width)), rng.normal(size=(2, width))
    draw, dcache, bias, dbias = rng.normal(size=2), rng.normal(size=(2, width)), rng.normal(size=2), rng.normal(size=2)
    original, original_direction = cache.copy(), dcache.copy()
    result = conv_step_jvp(raw, cache, weight, draw, dcache, bias=bias,
                           dbias=dbias, activation=activation)

    def reference(raw_t, cache_t, bias_t):
        module = SimpleNamespace(kernel_size=(width,), weight=tensor(weight)[:, None, :],
                                 bias=bias_t, activation=activation)
        out, state = streaming_short_convolution(module, raw_t[None, None, :],
                                                cache=cache_t[None, :, :],
                                                output_final_state=True)
        return out[0, 0], state[0]

    values, derivatives = torch.autograd.functional.jvp(
        reference, (tensor(raw), tensor(cache), tensor(bias)),
        (tensor(draw), tensor(dcache), tensor(dbias)))
    for i, name in enumerate(('value', 'cache')):
        assert_allclose(result[name], values[i].numpy(), atol=1e-14)
        assert_allclose(result['d' + name], derivatives[i].numpy(), atol=1e-14)
        finite = centered(lambda r, c, b: conv_step_jvp(r, c, weight, draw, dcache,
                                                      bias=b, activation=activation)[name],
                          [raw, cache, bias], [draw, dcache, dbias])
        assert_allclose(result['d' + name], finite, atol=1e-9)
    assert_allclose(cache, original)
    assert_allclose(dcache, original_direction)
    assert not np.shares_memory(result['cache'], cache)
    assert not np.shares_memory(result['dcache'], dcache)
    no_bias = conv_step_jvp(raw, cache, weight, draw, dcache, activation=None)
    assert_allclose(no_bias['value'], np.sum(no_bias['cache'] * weight, axis=1))


def test_softmax_two_legs_gain_and_active_null_score():
    scores = np.array([1., -0.4, 0.7])
    values = np.array([[2., -1.], [-0.3, 0.4], [0., 0.]])
    dscores = np.array([0.2, -0.7, 1.4])
    dvalues = np.array([[0.3, 0.6], [-0.2, 0.1], [0., 0.]])
    result = softmax_read_jvp(scores, values, dscores, dvalues)
    expected, derivative = torch.autograd.functional.jvp(
        lambda s, v: s.softmax(dim=0) @ v,
        (tensor(scores), tensor(values)), (tensor(dscores), tensor(dvalues)))
    assert_allclose(result['value'], expected.numpy(), atol=1e-14)
    assert_allclose(result['dvalue'], derivative.numpy(), atol=1e-14)
    finite = centered(lambda s, v: softmax_read_jvp(s, v, dscores, dvalues)['value'],
                      [scores, values], [dscores, dvalues])
    assert_allclose(result['dvalue'], finite, atol=1e-9)
    w = result['weights']
    assert_allclose(result['value_coefficients'], w)
    assert_allclose(result['score_coefficients'], w * np.linalg.norm(values - result['value'], axis=1))
    bound = (result['value_coefficients'] @ np.linalg.norm(dvalues, axis=1) +
             result['score_coefficients'] @ abs(dscores))
    assert np.linalg.norm(result['dvalue']) <= bound
    score_leg = softmax_read_jvp(scores, values, dscores, np.zeros_like(values))
    value_leg = softmax_read_jvp(scores, values, np.zeros_like(scores), dvalues)
    assert_allclose(result['dvalue'], score_leg['dvalue'] + value_leg['dvalue'])
    null_only = softmax_read_jvp(scores, values, [0., 0., 1.], np.zeros_like(values))
    assert_allclose(null_only['dvalue'], -w[-1] * result['value'])
    assert np.linalg.norm(null_only['dvalue']) > 0
    assert result['score_coefficients'][-1] > 0
    history_only = softmax_read_jvp(scores[:-1], values[:-1], dscores[:-1], dvalues[:-1])
    assert not np.allclose(result['value'], history_only['value'])
    common_shift = softmax_read_jvp(scores + 1000., values, np.ones(3), np.zeros_like(values))
    assert_allclose(common_shift['value'], result['value'], atol=1e-13)
    assert_allclose(common_shift['dvalue'], 0., atol=1e-14)


def test_mixed_field_composition_gates_normalization_delta_and_read():
    xs, ds = delta_fixture()
    params, dparams = np.array([0.2, -0.5, -1.2, 0.1]), np.array([0.3, -0.2, 0.1, 0.4])

    def composed(fields, directions, p, dp):
        S, q, k, v, _, _ = fields
        dS, dq, dk, dv, _, _ = directions
        q, dq, _ = normalize_jvp(q, dq, branch='recurrent')
        k, dk, _ = normalize_jvp(k, dk, branch='chunk')
        gates = gates_jvp(*p, *dp)
        delta = delta_read_jvp(S, q, k, v, gates['alpha'], gates['beta'],
                               dS, dq, dk, dv, gates['dalpha'], gates['dbeta'])
        return softmax_read_jvp(delta['output'], np.stack((delta['state'][0], np.zeros(2))),
                                delta['doutput'], np.stack((delta['dstate'][0], np.zeros(2))))

    result = composed(xs, ds, params, dparams)
    h = 1e-6
    plus = composed([x + h * d for x, d in zip(xs, ds)], ds, params + h * dparams, dparams)
    minus = composed([x - h * d for x, d in zip(xs, ds)], ds, params - h * dparams, dparams)
    assert_allclose(result['dvalue'], (plus['value'] - minus['value']) / (2 * h), atol=1e-9)
