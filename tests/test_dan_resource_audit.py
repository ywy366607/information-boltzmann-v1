"""Instrumentation must observe the real tick without changing its state."""
import copy

import pytest
import torch

from test_fly_dan_release import setup
from information_boltzmann.core import fly_bptt_learning as fl
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from scripts.ib.fly_interface.dan_resource_audit import observe_release


def test_observer_preserves_tick_and_restores_methods_on_failure(tmp_path):
    model, state, _ = setup(tmp_path)
    finish, advance = FlyReservoirLM._finish_coba_tick, fl.advance_dan_release
    with torch.no_grad():
        expected = fl.advance_fly_input_event(model, copy.deepcopy(state), torch.tensor([1]), settle_ticks=3)
        with pytest.raises(RuntimeError, match='test interruption'):
            with observe_release() as (material, rows):
                actual = fl.advance_fly_input_event(model, copy.deepcopy(state), torch.tensor([1]), settle_ticks=3)
                # The input event itself advances one tick before settling.
                assert len(rows) == 1 + 3
                assert material['graded_DAN'] == 0
                assert max(row['accounting_absmax'] for row in rows) < 2e-7
                for name in expected.state_dict():
                    left, right = getattr(expected, name), getattr(actual, name)
                    if name == 'ring':
                        for a, b in zip(left, right):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                    else:
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                raise RuntimeError('test interruption')
    assert FlyReservoirLM._finish_coba_tick is finish
    assert fl.advance_dan_release is advance
