"""Actual complete production loop and cold event provenance contracts."""
import pytest
import torch

from scripts.ib_local.identified_event_credit import configure
from scripts.ib_local.causal_use_value import configure as configure_use
from scripts.ib_local.shared_unknown_bindings import _read_evidence, _supports
from test_joint_relation_update import body


@pytest.mark.parametrize('mode', ['compressed','identified'])
def test_merge_preserves_actual_event_credit_and_delayed_boundary(tmp_path, mode):
    e,a = body(tmp_path, revision=4)
    try:
        e.autonomous_event_clock = 0
        configure_use(e,'C2')
        # Numerical/interface condition: force the production merge decision,
        # retaining all real generation, world, optimization and allocation.
        e.model.memory.threshold = -1.
        configure(e,mode)
        rows = [e.step(torch.randn(e.model.encoder.out_features,4)) for _ in range(3)]
        assert rows[1]['episode_merged'] and rows[2]['episode_merged']
        assert rows[1]['causal_use']['reward_version'] == 'finite_event_targeted_use_v3'
        if mode == 'identified':
            for r in rows[1:]:
                assert r['causal_use']['finite_U']['verified']
                assert r['shared_unknown']['revision']['committed']
                eid = r['shared_unknown']['revision']['observed_episode']
                assert a.db.execute('SELECT origin_slot FROM episodes WHERE id=?',(eid,)).fetchone()[0] == -1
                payload = a.db.execute('SELECT payload FROM episodes WHERE id=?',(eid,)).fetchone()[0]
                saved = a._unpack(payload)
                assert int(saved['wood_target_count']) == 1
                torch.testing.assert_close(torch.from_numpy(saved['wood_target']),torch.tensor(r['wood_target']))
                assert float(saved['progress']) == pytest.approx(r['metal_g'])
                assert r['causal_program']['committed_episode_identity'] == f'episode:{eid}'
                assert f'episode:{eid}' in {f.fact_id for f in _supports(a,r['shared_unknown']['identity'])}
            first_cold = rows[1]['shared_unknown']['revision']['observed_episode']
            facts = _read_evidence(a, first_cold)
            assert {f.kind.value for f in facts} == {'mother','decision'}
            assert rows[2]['causal_program_delayed_fact'] == f'episode:{first_cold}'
            assert rows[2]['internal_transition_samples'] > 0
        else:
            assert not rows[1]['causal_use']['finite_U']['committed']
            assert not rows[1]['shared_unknown']['revision']['committed']
            assert rows[1]['metal_g'] == 0.
        assert rows[-1]['delayed_actor_applied'] > 0
        assert a.db.execute('PRAGMA foreign_key_check').fetchall() == []
        before = e.causal_use_calibration.copy()
        configure(e,mode)
        assert e.causal_use_calibration == before
        with pytest.raises(ValueError):
            configure(e,'identified' if mode == 'compressed' else 'compressed')
    finally:
        a.close()
