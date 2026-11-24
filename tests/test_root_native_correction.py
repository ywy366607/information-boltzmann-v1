"""Evaluated successor replacement at the real body's root actor boundary."""
import copy

import pytest
import torch

from test_persistent_rhythm import body
from scripts.ib_local.autonomous_cycle import dispatch


@pytest.mark.parametrize('mode', ['observe', 'apply'])
def test_real_root_patch_uses_same_mass_before_choice_and_real_settlement(tmp_path, monkeypatch, mode):
    from scripts.ib_local import root_native_correction as correction
    torch.set_num_threads(1)
    torch.manual_seed(201120)
    e, archive = body(tmp_path, root_mode=mode)
    observed = []
    original = correction.correct
    def verify(engine, values, checks):
        bank = copy.deepcopy(engine.native_value_bank)
        version = tuple(p._version for p in engine.model.parameters())
        before = values.clone()
        result = original(engine, values, checks)
        receipt = copy.deepcopy(engine._root_native_correction_last)
        assert len(checks) <= 2
        assert tuple(p._version for p in engine.model.parameters()) == version
        assert engine.native_value_bank['revision'] == bank['revision']
        assert len(engine.native_value_bank['pending']) == len(bank['pending'])
        expected = torch.zeros_like(values)
        for row in receipt['contributions']:
            assert row['root'] in engine._competing_action_roots
            replaced = row['mass'] * row['predicted'] + row['future_correction']
            assert replaced == pytest.approx(row['mass'] * row['native'], abs=1e-7)
            for option in row['options']:
                expected[option] += 10 * engine.gamma * row['future_correction']
        for check in checks:
            if check['root_key'] is not None:
                mass = engine._root_native_mass[check['root_key']]
                assert check['mass'] == pytest.approx(mass['weights'][check['key']]/mass['calls'])
                assert check['response_calls'] == mass['calls']
        torch.testing.assert_close(result, before + (expected if mode == 'apply' else 0))
        torch.testing.assert_close(values, before)
        observed.append(receipt)
        return result
    monkeypatch.setattr(correction, 'correct', verify)
    try:
        rows = []
        for _ in range(96):
            r = dispatch(e)
            if r['autonomous_event']['kind'] == 'external':
                rows.append(r)
            if len(rows) == 2:
                break
        assert len(rows) == len(observed) == 2
        for r, receipt in zip(rows, observed):
            assert receipt['contributions'] and receipt['teacher_count'] == 2
            assert r['causal_use']['epistemic_value']['root_native_correction'] == receipt
            assert r['causal_use']['finite_U']['verified']
            assert r['causal_use']['native_amortization']['new_world_label']
            assert r['policy_replay_max_error'] < 1e-3
        assert e._competing_action_roots is None
        assert not hasattr(e, '_root_native_mass')
    finally:
        archive.close()


def test_response_mean_mass_counts_repeated_and_aliased_nodes():
    from types import SimpleNamespace
    from scripts.ib_local.competing_action_teachers import root_node_mass
    # Two equally weighted response branches, with one node shared by both.
    e = SimpleNamespace(_root_native_mass={'root': dict(calls=2,
        weights={'only_first': .8, 'shared': .2+.7, 'only_second': .3})})
    assert root_node_mass(e, 'root', 'only_first', .8) == (.4, 2)
    assert root_node_mass(e, 'root', 'shared', .7)[0] == pytest.approx(.45)
    assert sum(root_node_mass(e, 'root', k, 0)[0] for k in e._root_native_mass['root']['weights']) == pytest.approx(1.)
    # Recomputing the same physical root under two action aliases duplicates
    # numerator AND response count, leaving every actor-option coefficient.
    e._root_native_mass['root'] = dict(calls=4,
        weights={k:2*v for k,v in e._root_native_mass['root']['weights'].items()})
    assert root_node_mass(e, 'root', 'only_first', .8) == (.4, 4)
