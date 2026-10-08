"""Numerical instruments only; these tests make no memory/capability claim."""
import importlib.util
import json
from pathlib import Path

import pytest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'fly_harness_diagnostic', ROOT/'scripts/ib/diagnose_fly_harness_update.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_fixed_spike_branch_preserves_forward_and_has_ordinary_derivative():
    voltage = torch.tensor([0.1, 0.3], dtype=torch.float64, requires_grad=True)
    threshold = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    original = audit.reservoir.SpikeFn
    with audit.spike_instrument() as reference:
        actual = voltage * (1-original.apply(voltage-threshold))
        audit.reservoir.SpikeFn.apply(voltage-threshold)
    with audit.spike_instrument(reference):
        fixed = voltage * (1-audit.reservoir.SpikeFn.apply(voltage-threshold))
        dv, dt = torch.autograd.grad(fixed.sum(), (voltage, threshold), allow_unused=True)
    torch.testing.assert_close(fixed, actual)
    torch.testing.assert_close(dv, torch.tensor([1., 0.], dtype=torch.float64))
    assert dt is None
    assert audit.reservoir.SpikeFn is original


def test_spike_instrument_restores_original_even_on_failure():
    original = audit.reservoir.SpikeFn
    with pytest.raises(RuntimeError), audit.spike_instrument():
        raise RuntimeError('instrument failure')
    assert audit.reservoir.SpikeFn is original
    with pytest.raises(ValueError), audit.spike_instrument([]):
        audit.reservoir.SpikeFn.apply(torch.zeros(1))
    assert audit.reservoir.SpikeFn is original


def test_actual_displacement_dot_product_includes_missing_gradient_as_zero():
    old = {'a': torch.tensor([2., 3.]), 'b': torch.tensor([5.])}
    new = {'a': torch.tensor([3., 1.]), 'b': torch.tensor([6.])}
    gradient = {'a': torch.tensor([4., -2.])}
    result = audit.directional_summary(gradient, old, new, {'a', 'b'})
    assert result['gradient_dot_actual_displacement'] == 8.
    assert result['displacement_norm'] == pytest.approx(6**.5)
    assert result['gradient_norm'] == pytest.approx(20**.5)
    assert result['float64_accumulation_floor'] > 0


def test_execution_gate_requires_complete_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit.gate_registry(tmp_path)


def test_chunked_restore_matches_fp32_interpolation_and_selected_controls():
    old = {'a': torch.arange(15, dtype=torch.float32).reshape(3, 5),
           'b': torch.tensor([7., 9.])}
    new = {name: value + 3. for name, value in old.items()}
    named = {name: torch.nn.Parameter(torch.zeros_like(value))
             for name, value in old.items()}
    audit.restore_parameters(named, new, {'a'}, .125, old, chunk_size=4)
    torch.testing.assert_close(named['a'], old['a'] + .125 * (new['a'] - old['a']),
                               rtol=0, atol=0)
    torch.testing.assert_close(named['b'], old['b'], rtol=0, atol=0)
    audit.restore_parameters(named, new, chunk_size=4)
    for name in named:
        torch.testing.assert_close(named[name], new[name], rtol=0, atol=0)


def test_json_scalar_keeps_numpy_values_and_rejects_nonfinite_values():
    payload = {'floor': np.float32(.125), 'count': np.int64(32)}
    assert json.loads(json.dumps(payload, default=audit.json_scalar,
                                allow_nan=False)) == {'floor': .125, 'count': 32}
    with pytest.raises(ValueError):
        json.dumps({'floor': np.float32(float('nan'))},
                   default=audit.json_scalar, allow_nan=False)


def test_observe_gradient_instrument_preserves_clipping_and_restores_on_failure():
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([3., 4.])
    original = torch.nn.utils.clip_grad_norm_
    with audit.observe_gradient_instrument() as captured:
        norm = torch.nn.utils.clip_grad_norm_([parameter], 1., error_if_nonfinite=True)
    assert norm == 5.
    torch.testing.assert_close(captured[id(parameter)], parameter.grad, rtol=0, atol=0)
    assert parameter.grad.norm() == pytest.approx(1., abs=1e-6)
    assert torch.nn.utils.clip_grad_norm_ is original
    with pytest.raises(RuntimeError), audit.observe_gradient_instrument():
        raise RuntimeError('observe failed')
    assert torch.nn.utils.clip_grad_norm_ is original
