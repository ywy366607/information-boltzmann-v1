"""Unified intervention controls must restore the live model on every exit."""
import pytest
import torch
from torch import nn

from scripts.ib.fly_interface.native_system_matrix import intervene, SYSTEM_FLAGS, deterministic_replay


def model():
    value = nn.Module()
    value.register_buffer('graded_mask', torch.tensor([[1., 1., 0.]]), persistent=False)
    value.register_buffer('visual_graded_index', torch.tensor([1]), persistent=False)
    for flag in SYSTEM_FLAGS.values():
        setattr(value, flag, True)
    return value


@pytest.mark.parametrize('system', list(SYSTEM_FLAGS))
def test_unified_intervention_restores_original_state_on_error(system):
    value = model()
    original = value.graded_mask
    with pytest.raises(RuntimeError):
        with intervene(value, system):
            if system == 'visual':
                assert value.graded_mask[0, 1] == 0
                assert value.graded_mask[0, 0] == 1
            else:
                assert getattr(value, SYSTEM_FLAGS[system]) is False
            raise RuntimeError('controlled replay failure')
    assert value.graded_mask is original
    assert all(getattr(value, name) for name in SYSTEM_FLAGS.values())


@pytest.mark.parametrize('system', ['intact', 'sham', 'restored'])
def test_unified_sham_and_restore_are_identity(system):
    value = model()
    original = value.graded_mask
    with intervene(value, system):
        assert value.graded_mask is original
        assert all(getattr(value, name) for name in SYSTEM_FLAGS.values())


@pytest.mark.parametrize('enabled', [False, True])
def test_deterministic_replay_restores_execution_flags_on_error(enabled, monkeypatch):
    import os
    from scripts.ib.fly_interface.native_system_matrix import fl
    transmission = fl.execute_delayed_synaptic_transmission
    original = torch.are_deterministic_algorithms_enabled()
    warning = torch.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=True)
        with pytest.raises(RuntimeError, match='controlled error'):
            with deterministic_replay():
                assert torch.are_deterministic_algorithms_enabled()
                assert not torch.is_deterministic_algorithms_warn_only_enabled()
                assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
                assert fl.execute_delayed_synaptic_transmission is not transmission
                packets = (torch.tensor([[1., 2., 3.]]),) * 4
                actual = fl.execute_delayed_synaptic_transmission(packets,
                    torch.tensor([0, 1]), torch.tensor([2, 2]), torch.tensor([.5, .25]), [0, 2])
                torch.testing.assert_close(actual, torch.tensor([[0., 0., 1.]]))
                raise RuntimeError('controlled error')
        assert torch.are_deterministic_algorithms_enabled() is enabled
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert 'CUBLAS_WORKSPACE_CONFIG' not in os.environ
        assert fl.execute_delayed_synaptic_transmission is transmission
    finally:
        torch.use_deterministic_algorithms(original, warn_only=warning)
