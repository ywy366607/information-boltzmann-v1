"""Bounded arithmetic storage preserves long-lived cumulative accounting."""
import copy
import math

import numpy as np
import pytest

from information_boltzmann.cgb.exchange_checkpoint import save_exchange, load_exchange
from information_boltzmann.cgb.material_fork import fork_material
from scripts.ib.generative.compare_support_transport import same
from test_cgb_growing_material import growing


def test_compensated_sum_survives_cancellation_without_rebasing():
    m = growing()
    m.enable_compensated_ledgers()
    initial = m.heat_out
    increments = [1e16, 1., -1e16, 1e-12]*100
    for delta in increments:
        m._add_ledger('heat_out', delta)
    assert math.fsum([m.heat_out, m._ledger_roundoff['heat_out']]) == math.fsum([initial, *increments])
    assert len(m._ledger_roundoff) == 10


def test_arithmetic_changes_no_physical_rng_or_event_trajectory(tmp_path):
    old = growing()
    new = fork_material(old)
    before = new.state_dict()
    new.enable_compensated_ledgers()
    assert new.initial_energy == before['initial_energy']
    assert new.initial_free_energy == before['initial_free_energy']
    for _ in range(8):
        old.step(2., np.array([2.]))
        new.step(2., np.array([2.]))
    for name in old._arrays:
        np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
    for name in ('rng_reaction', 'rng_motion', 'rng_field'):
        assert getattr(old,name).bit_generator.state == getattr(new,name).bit_generator.state
    assert old.reaction_counts == new.reaction_counts
    assert 'ledger_roundoff' not in old.state_dict()
    path = tmp_path/'complete.npz'
    save_exchange(path, new)
    restored, _ = load_exchange(path)
    same(new.state_dict(), restored.state_dict())
    fork = fork_material(new)
    assert fork._ledger_roundoff is not new._ledger_roundoff
    for _ in range(8):
        for m in (new, restored, fork):
            m.step(2., np.array([-2.]))
    same(new.state_dict(), restored.state_dict())
    same(new.state_dict(), fork.state_dict())
    assert max(abs(float(v)) for v in new.balances().values()) < 1e-10


@pytest.mark.parametrize('damage', ['missing', 'nan', 'mode'])
def test_incomplete_arithmetic_checkpoint_is_rejected(damage):
    m = growing()
    m.enable_compensated_ledgers()
    state = copy.deepcopy(m.state_dict())
    if damage == 'missing':
        state['ledger_roundoff'].pop('heat_out')
    elif damage == 'nan':
        state['ledger_roundoff']['heat_out'] = float('nan')
    else:
        state['ledger_accumulation'] = 'unknown'
    with pytest.raises(ValueError):
        type(m).from_state_dict(state)
