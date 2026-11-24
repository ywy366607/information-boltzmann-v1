"""Production U, transaction timing, quadrature and complete restore checks."""
import copy
from dataclasses import replace
import gzip

import pytest
import torch

from scripts.ib_local.causal_use_value import configure, queries
from scripts.ib_local.finite_use_value import expected
from scripts.ib_local.causal_relation_program import mechanism
from scripts.ib_local.causal_use_transport import finish


def test_paid_use_matches_sql_U_and_prediction_is_permutation_invariant(tmp_path):
    from test_joint_relation_update import body
    torch.manual_seed(21)
    e, archive = body(tmp_path, revision=4)
    configure(e, 'C2')
    e.causal_use_transport_audit = True
    verified = 0
    try:
        for i in range(3):
            e.autonomous_event_clock = i
            row = e.step(torch.randn(8, 4))
            finish(e, row)  # Runner's repeated observer must be idempotent.
            assert not hasattr(e, '_deferred_causal_use_optimizer_step')
            assert not hasattr(e, '_finite_use_commit_pending')
            assert row['wood_target'][3] == float(torch.tensor(10*row['metal_g']))
            if row['use_transport']['committed']:
                gain = row['use_transport']['mean']['actual_U_gain']
                if gain is not None:
                    verified += 1
                    assert abs(row['metal_g']-gain) < 3e-6
                else:
                    assert row['metal_g'] == 0.
                assert row['use_transport']['measurement_event'] == 'finite_U_commit_before_value_fit'
                assert row['causal_use']['finite_U']['verified']
        assert verified > 0
        state = e._program_question_state
        facts = queries(e, state)
        source = e._internal_source.reshape(1, -1).expand(3, -1)
        actions = e._internal_action
        with torch.no_grad():
            d = mechanism(e.model, state, source, actions, ridge=e.relation_ridge)
            before = copy.deepcopy(e.model.state_dict())
            actual = expected(e, state, source, actions, d, facts)
            reordered = replace(state, constraints=tuple(reversed(state.constraints)))
            d2 = mechanism(e.model, reordered, source.flip(0), actions.flip(0), ridge=e.relation_ridge)
            other = expected(e, reordered, source.flip(0), actions.flip(0), d2, facts[::-1])
        torch.testing.assert_close(actual, other.flip(0), rtol=2e-5, atol=2e-6)
        from test_joint_world_cycle import assert_same_saved_values
        assert_same_saved_values(before, e.model.state_dict())
    finally:
        archive.close()


@pytest.mark.parametrize('reference_use', [False, True])
def test_finite_U_full_restore_keeps_next_action_update_and_receipt(tmp_path, reference_use):
    from test_joint_relation_update import body
    from test_joint_world_cycle import assert_same_saved_values
    from scripts.ib_local.mother_checkpoint import save
    from scripts.ib_local.navigation_checkpoint import restore
    from scripts.ib_local.causal_archive import attach_archive
    from scripts.ib_local.shared_unknown_bindings import configure_shared_unknowns
    from scripts.ib_local.joint_relation_update import configure_joint_relation_update
    e, archive = body(tmp_path/'live', revision=4)
    restored_archive = None
    try:
        from scripts.ib_local.reference_use import configure as configure_reference_use
        configure(e, 'C2'); e.autonomous_event_clock = 0
        config = dict(joint_relation_update={'schema': 1, 'implementation_revision': 4},
            shared_unknowns={'schema': 1}, causal_use_value={'schema': 1, 'arm': 'C2'},
            identity_preview_revision=1)
        if reference_use:
            configure_reference_use(e, config)
        e.step(torch.randn(8, 4))
        save(e, tmp_path/'saved', 1, config)
        with gzip.open(tmp_path/'saved/last.pt.gz', 'rb') as stream:
            state = torch.load(stream, map_location='cpu', weights_only=False)
        row = e.step(torch.randn(8, 4))
        model, optimizer = copy.deepcopy(e.model.state_dict()), copy.deepcopy(e.optimizer.state_dict())
        other, unused = body(tmp_path/'fresh', revision=4)
        unused.close(); del other.model._causal_archive
        # Production restoration loads engine fields before configuring C2.
        # Pre-configuring the fresh body here would mask lost runtime flags.
        restore(other, state)
        restored_archive = attach_archive(other, tmp_path/'restored',
            manifest=state['config']['causal_archive'], relation_capacity=archive.budgets['relations'])
        configure_shared_unknowns(other); configure_joint_relation_update(other, revision=4)
        configure(other, 'C2')
        if state['config'].get('reference_use') == 1:
            configure_reference_use(other, state['config'])
        assert getattr(other.model, 'identity_preview_revision', 0) == 1
        actual = other.step(torch.randn(8, 4))
        if reference_use:
            assert actual['causal_use']['finite_U']['reference_use'] is not None
            query = 'SELECT * FROM reference_use ORDER BY relation_id,unknown_id'
            assert restored_archive.db.execute(query).fetchall() == archive.db.execute(query).fetchall()
        for key in ('choice', 'utterance_hex', 'causal_use', 'shared_unknown', 'metal_g'):
            assert row[key] == actual[key], key
        assert_same_saved_values(model, other.model.state_dict())
        assert_same_saved_values(optimizer, other.optimizer.state_dict())
    finally:
        archive.close()
        if restored_archive is not None:
            restored_archive.close()


def test_finite_use_begin_reuses_exact_expected_across_cloned_inputs(tmp_path, monkeypatch):
    from test_joint_relation_update import body
    from scripts.ib_local import finite_use_value as fuv
    torch.manual_seed(29)
    e, archive = body(tmp_path, revision=4)
    configure(e, 'C2')
    try:
        for i in range(2):
            e.autonomous_event_clock = i
            e.step(torch.randn(8, 4))
        state = e._program_question_state
        facts = queries(e, state)
        assert facts
        source = e._internal_source.reshape(1, -1).expand(len(e._internal_action), -1).clone()
        actions = e._internal_action.clone()
        with torch.no_grad():
            d = mechanism(e.model, state, source, actions, ridge=e.relation_ridge)
            vals = fuv.expected(e, state, source, actions, d, facts)
        calls = [0]
        orig_expected = fuv.expected
        def counting_expected(*a, **kw):
            calls[0] += 1
            return orig_expected(*a, **kw)
        monkeypatch.setattr(fuv, 'expected', counting_expected)
        e.causal_use_pending = {}
        fuv.begin(e, dict(choice=1), d, facts)
        assert calls[0] == 0
        assert e.causal_use_pending['finite']['expected'] == float(vals[1])
        # Mutating source invalidates the cache and falls back cleanly.
        e._finite_use_last_expected = None
        with torch.no_grad():
            orig_expected(e, state, source, actions, d, facts)
        e._internal_source = e._internal_source + 0.25
        e.causal_use_pending = {}
        fuv.begin(e, dict(choice=1), d, facts)
        assert calls[0] == 1
    finally:
        archive.close()
