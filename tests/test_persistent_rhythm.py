"""Actual learner interleaving, conditional-plan coordinates and persistence."""
import copy

import pytest
import torch

from test_relation_retention_use import configured
from scripts.ib_local import autonomous_cycle, epistemic_value_reference as ev
from scripts.ib_local import native_amortized_decision as amort
from scripts.ib_local import persistent_rhythm as rhythm


def body(tmp_path, *, interleave=True, root_mode=None, scope_prior=None):
    e, archive = configured(tmp_path)
    e.gain_units_schema = 1
    ev.configure(e, schema=5, word_execution='native_anchor')
    e.native_word_runtime = 'batch'
    e.model._preview_batch_enabled = True
    specification = dict(schema=1, seed=201120, planner='action_batch', correction='decision_budget')
    if root_mode is not None:
        specification['root_native_correction'] = root_mode
    amort.configure(e, specification)
    autonomous_cycle.configure(e)
    rhythm.configure(e, interleave=interleave)
    if scope_prior is not None:
        from scripts.ib_local.retention_scope_prior import configure
        configure(e, scope_prior)
    return e, archive


@pytest.mark.parametrize('interleave,internal_review', [(False, False), (True, False), (True, True)])
def test_real_learning_handoffs_keep_prediction_and_delayed_credit(tmp_path, interleave, internal_review, monkeypatch):
    from scripts.ib_local.native_delayed_credit import attach

    torch.set_num_threads(1)
    torch.manual_seed(201120)
    e, archive = body(tmp_path, interleave=interleave)
    if internal_review:
        e.native_amortization['review_cadence'] = 'internal_step'
    attach(e)
    sources = []
    poll = autonomous_cycle.poll

    def observed_poll(engine):
        before = engine.autonomous_eligibility[0].double()
        row = poll(engine)
        sources.append((engine.autonomous_event_clock+1, engine.autonomous_eligibility[0].double()-before))
        return row

    monkeypatch.setattr(autonomous_cycle, 'poll', observed_poll)
    count = [0]
    original = e.optimizer.step
    def step(*args, **kwargs):
        count[0] += 1
        return original(*args, **kwargs)
    e.optimizer.step = step
    rows, events = [], []
    try:
        for _ in range(96):
            before = count[0]
            r = autonomous_cycle.dispatch(e)
            events.append((r, count[0]-before))
            if r['autonomous_event']['kind'] == 'external':
                rows.append(r)
                assert r['causal_use']['finite_U']['verified']
                assert r['causal_use']['reward_version'] == 'finite_relation_use_v5'
                plan = r['causal_use']['native_amortization']['action_planner']
                steps = r['generative_relation']['autonomous']['internal_steps']
                assert plan['teacher_count'] == (2 if steps or not internal_review else 0)
                assert plan['new_world_label']
                assert r['policy_replay_max_error'] < 1e-3
                assert e._rhythm_runtime is None
            if len(rows) == 3: break
        assert len(rows) == 3 and count[0] >= 12
        previous_boundary = 0
        for previous_row, row in zip(rows, rows[1:]):
            report = row['selector_age_delivery']
            assert report['complete']
            boundary, applied = report['action_boundary'], report['applied_event']
            used = [(tick, g) for tick, g in sources if previous_boundary < tick <= boundary]
            previous_boundary = boundary
            assert report['source_count'] == len(used) == previous_row['generative_relation']['autonomous']['choices']
            assert report['advantage'] == row['td_previous']['advantage']
            got = torch.tensor(report['age_moment'], dtype=torch.float64)
            torch.testing.assert_close(got, sum((applied-tick)*g.flatten() for tick, g in used))
            event, updates = next((r['autonomous_event'], n) for r, n in events
                                  if r['autonomous_event']['event'] == applied)
            assert updates > 0
            assert event['phase'] == 'continuation_updated' if interleave else event['kind'] == 'external'
        assert all(r['external_bytes'] == 0 for r, _ in events
                   if r['autonomous_event']['kind'] != 'external')
        if interleave:
            handoffs = [r for r, _ in events if r['autonomous_event']['kind'] == 'handoff']
            phases = [r['autonomous_event']['phase'] for r in handoffs]
            assert phases.count('world_arrived') == phases.count('response_updated') == 3
            assert phases.count('continuation_updated') >= 3
            assert any(r['autonomous_event'].get('within_external') for r, _ in events)
            assert e.rhythm_state['within_learning_choices'] > 0
        else:
            assert not any(r['autonomous_event']['kind'] == 'handoff' for r, _ in events)
        assert e.rhythm_state['reads'] > 0
        assert 0 < len(e.rhythm_state['plans']) <= e.model.causal_relations.evidence_budget
        assert all(p['packet']['raw'].device.type == 'cpu' for p in e.rhythm_state['plans'].values())
        assert archive.db.execute('PRAGMA foreign_key_check').fetchall() == []
    finally:
        archive.close()


def test_plan_read_uses_current_bank_and_encoder_not_cached_prediction(tmp_path):
    torch.set_num_threads(1)
    e, archive = body(tmp_path, interleave=False)
    try:
        source = e._read_source()
        state = e.model.causal_relations.bind(e.model, source)
        p, _, _ = rhythm.read_plans(e, [state], [source], carried=state)
        saved = copy.deepcopy(e.rhythm_state)
        record = next(iter(saved['plans'].values()))
        with torch.no_grad():
            e.model.encoder.weight.add_(.01)
        got, _, _ = rhythm.read_plans(e, [state], [source], carried=state)
        fresh = amort.recipe(e, state, source, selected=state)
        expected, _, _ = amort.predict(e, [fresh])
        torch.testing.assert_close(got, expected)
        assert record['packet']['raw'].device.type == 'cpu'
        assert e.rhythm_state['reads'] == saved['reads']+1
    finally:
        archive.close()


@pytest.mark.parametrize('root_mode,scope_prior', [(None, None), ('apply', None), ('observe', 'relation')])
def test_complete_disk_restore_reproduces_next_interleaved_life(tmp_path, root_mode, scope_prior):
    import gzip
    from scripts.ib_local.mother_checkpoint import save
    from scripts.ib_local.navigation_checkpoint import restore
    from scripts.ib_local.causal_archive import attach_archive
    from scripts.ib_local.shared_unknown_bindings import configure_shared_unknowns
    from scripts.ib_local.joint_relation_update import configure_joint_relation_update
    from scripts.ib_local.causal_use_value import configure as configure_value
    from scripts.ib_local.identified_event_credit import configure as configure_events
    from scripts.ib_local.relation_retention_use import configure as configure_use
    from test_joint_world_cycle import assert_same_saved_values
    torch.set_num_threads(1)
    e, archive = body(tmp_path/'live', root_mode=root_mode, scope_prior=scope_prior)
    if root_mode == 'observe':
        e.native_amortization['review_cadence'] = 'internal_step'
    restored = None
    def next_external(engine):
        events = []
        for _ in range(96):
            row = autonomous_cycle.dispatch(engine)
            events.append(copy.deepcopy(row['autonomous_event']))
            if row['autonomous_event']['kind'] == 'external': return row, events
        raise AssertionError('no external transition within numerical contract budget')
    try:
        # Snapshot admission also covers a suspended world transaction.
        while e._rhythm_runtime is None:
            autonomous_cycle.dispatch(e)
        with pytest.raises(RuntimeError, match='active external transaction'):
            save(e, tmp_path/'invalid', 0, {})
        next_external(e)
        config = dict(relation_retention_use=dict(schema=2,mode='apply'),
            epistemic_value=dict(schema=5,word_execution='native_anchor'),
            native_word_runtime=dict(schema=2,mode='batch',batch_size=4),
            native_amortization=e.native_amortization,
            persistent_rhythm=dict(schema=1,interleave=True))
        if scope_prior is not None:
            config['retention_scope_prior'] = dict(schema=1, mode=scope_prior)
        save(e, tmp_path/'saved', 1, config)
        with gzip.open(tmp_path/'saved/last.pt.gz','rb') as stream:
            stored = torch.load(stream,map_location='cpu',weights_only=False)
        assert '_rhythm_runtime' not in stored['engine']
        from scripts.ib_local.native_delayed_credit import attach
        attach(e)
        expected, expected_events = next_external(e)
        assert expected['selector_age_delivery']['reason'] == 'action predates observation'
        weights, optimizer = copy.deepcopy(e.model.state_dict()), copy.deepcopy(e.optimizer.state_dict())
        other, unused = body(tmp_path/'fresh')
        unused.close(); del other.model._causal_archive
        restore(other, stored)
        restored = attach_archive(other,tmp_path/'restored',manifest=stored['config']['causal_archive'],
                                  relation_capacity=archive.budgets['relations'])
        configure_shared_unknowns(other); configure_joint_relation_update(other,revision=4)
        configure_value(other,'C2'); configure_events(other,'identified')
        ev.configure(other,schema=5,word_execution='native_anchor')
        amort.configure(other,stored['config']['native_amortization'])
        configure_use(other,'apply',schema=2)
        if scope_prior is not None:
            from scripts.ib_local.retention_scope_prior import configure
            configure(other, scope_prior)
        rhythm.configure(other,interleave=True)
        actual, actual_events = next_external(other)
        assert expected_events == actual_events
        for key in ('choice','utterance_hex','metal_g','causal_use','shared_unknown'):
            assert expected[key] == actual[key], key
        assert_same_saved_values(weights,other.model.state_dict())
        assert_same_saved_values(optimizer,other.optimizer.state_dict())
        # PyTorch mutation counters are process-local cache stamps. A restored
        # body cold-starts those caches; physical state and the next real event
        # above are exact. Compare the persisted plans' semantic payload.
        left, right = copy.deepcopy(e.rhythm_state), copy.deepcopy(other.rhythm_state)
        for state in (left,right):
            state.pop('cache_hits'); state.pop('reencoded')
            for plan in state['plans'].values(): plan.pop('stamp')
        assert_same_saved_values(left,right)
    finally:
        archive.close()
        if restored is not None: restored.close()
