"""External preparation accounting, never a toy timing capability task."""
import copy

import numpy as np

from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from scripts.ib.generative.compare_support_transport import same
from scripts.ib.generative.native_timing_audit import neutralize_charge
from test_cgb_growing_material import growing


def test_complete_fast_charge_preparation_pays_work_and_preserves_slow_history():
    m = growing()
    original = m.state_dict()
    energy = m.energy()
    receipt = neutralize_charge(m)
    assert not np.any(m.charge) and not np.any(m.world_charge)
    assert receipt['external_work'] == m.energy()-energy
    assert m.t == original['t']
    for name in m._arrays:
        if name not in ('charge', 'world_charge'):
            np.testing.assert_array_equal(getattr(m, name), original['arrays'][name])
    current = m.state_dict()
    for key in original:
        if 'rng' in key:
            same(current[key], original[key])
    assert max(abs(float(v)) for v in m.balances().values()) < 1e-10


def test_prepared_all_field_state_and_ledgers_restore_into_actual_future(tmp_path):
    m = growing()
    neutralize_charge(m)
    path = tmp_path/'prepared.npz'
    save_exchange(path, m, {'diagnostic_only': True})
    restored, meta = load_exchange(path)
    assert meta['diagnostic_only']
    for _ in range(3):
        m.step(2., np.array([3.]))
        restored.step(2., np.array([3.]))
    same(m.state_dict(), restored.state_dict())
    assert max(abs(float(v)) for v in restored.balances().values()) < 1e-10
