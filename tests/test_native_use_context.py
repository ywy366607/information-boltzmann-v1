"""CPU fixtures for the read-only C2 diagnostic, never a full body or training."""
import copy
from dataclasses import replace
import json

import pytest
import torch

from scripts.audit_native_use_context import (
    compare_supports, current_binding, historical_errors, raw_options, support_states,
)
from scripts.ib_local.causal_relation_program import Evidence, Port, RelationState, Variable
from scripts.ib_local.causal_use_value import configure
from scripts.ib_local.relation_evidence import mechanism_evidence


@pytest.fixture
def fixture(tmp_path):
    from test_joint_relation_update import body
    # This constructs the existing tiny CPU interface (width=16), no e.step().
    torch.manual_seed(31)
    engine, archive = body(tmp_path, revision=4)
    configure(engine, 'C2')
    width = engine.model.memory.source.shape[-1]
    source = torch.randn(1, width)
    facts = tuple(Evidence(source+float(i)/5, source+torch.randn_like(source)/4,
        torch.randn_like(source), Port.MOTHER, True, slot=-1, stamp=i,
        fact_id=f'episode:{i+1}') for i in range(3))
    decision = tuple(replace(f, kind=Port.DECISION, outcome=f.outcome+.1,
                             mother=f.outcome) for f in facts)
    budget = engine.model.causal_relations.evidence_budget
    ids = torch.zeros(budget, dtype=torch.long)
    state = RelationState(1, 1, 7, 0, ids, ids.clone(), torch.zeros(budget, dtype=torch.bool),
        binding=Variable('fixture-current-unknown', Port.MOTHER),
        constraints=facts+decision, evidence_bound=True, program_weights=torch.tensor([.5, .5]))
    organization = replace(state, binding=None, organization=None)
    state = replace(state, organization=organization,
                    variable_uses=((state.binding, 4, 3),))
    archive.db.execute('PRAGMA query_only=ON')
    try:
        yield engine, archive, state, facts, source
    finally:
        archive.close()


def test_same_facts_reordered_keep_pair_scores_and_inputs(fixture):
    engine, archive, state, facts, source = fixture
    actions = torch.cat((source, source+facts[0].action-facts[0].source))
    sources = source.expand_as(actions)
    sql_before = archive.db.total_changes
    parameters = copy.deepcopy(engine.model.state_dict())
    optimizer = copy.deepcopy(engine.optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    result = compare_supports(engine, state, sources, actions, facts)
    full, reordered = result['rows'][:2]
    assert full['query_ids'] == reordered['query_ids']
    assert full['raw']['input_hash'] == reordered['raw']['input_hash']
    assert full['raw'] == reordered['raw']
    for key in ('expected', 'score_before', 'total'):
        torch.testing.assert_close(torch.tensor(full[key]), torch.tensor(reordered[key]),
                                   rtol=2e-5, atol=2e-6)
    assert archive.db.total_changes == sql_before
    assert torch.equal(rng, torch.get_rng_state())
    from test_joint_world_cycle import assert_same_saved_values
    assert_same_saved_values(parameters, engine.model.state_dict())
    assert_same_saved_values(optimizer, engine.optimizer.state_dict())
    # The delivered data is JSON-compatible without tensor serialization hooks.
    json.dumps(result, allow_nan=False)


def test_support_projection_removes_both_ports_and_stale_organization(fixture):
    engine, _, state, facts, _ = fixture
    variants = support_states(engine.model, state, exclude=[facts[-1].fact_id])
    for _, variant in variants:
        assert variant.organization is None and variant.variable_uses == ()
        assert variant.evidence_bound and variant.extra == ()
        assert torch.equal(variant.program_weights, state.program_weights)
        for kind in (Port.MOTHER, Port.DECISION):
            supports = mechanism_evidence(engine.model, variant, kind)
            assert facts[-1].fact_id not in {f.fact_id for f in supports}
    removed_id = facts[0].fact_id
    without = dict(variants)[f'without:{removed_id}']
    assert all(f.fact_id != removed_id for f in without.constraints)
    # The actual bound state is untouched.
    assert len(state.constraints) == 6 and state.organization is not None


def test_actual_incoming_held_out_and_query_fits_exclude_self(fixture):
    engine, archive, state, facts, _ = fixture
    incoming, old_queries = facts[-1], facts[:-1]
    sql_before = archive.db.total_changes
    result = compare_supports(engine, state, incoming.source, incoming.action,
                              old_queries, incoming=incoming)
    for row in result['rows']:
        assert incoming.fact_id not in row['supports']
        assert row['query_ids'] == [f.fact_id for f in old_queries]
        for qid, fit in zip(row['query_ids'], row['query_fit_ids']):
            assert qid not in fit and incoming.fact_id not in fit
        for qid, fit in zip(row['query_ids'], row['observed']['query_fit_ids_after']):
            assert qid not in fit
        assert row['raw'] == row['observed']['raw_after']
        before = torch.tensor(row['score_before'])
        after = torch.tensor(row['observed']['score_after'])
        torch.testing.assert_close(after-before, torch.tensor(row['observed']['gains']))
        assert row['observed']['net_gain'] == pytest.approx(float((after-before).mean()))
    assert archive.db.total_changes == sql_before
    with pytest.raises(ValueError, match='own old query'):
        compare_supports(engine, state, incoming.source, incoming.action, facts, incoming=incoming)


def test_known_incoming_outcome_never_enters_prospective_forecasts(fixture):
    engine, _, state, facts, _ = fixture
    incoming = facts[-1]
    changed_outcome = replace(incoming, outcome=incoming.outcome+7)
    first = compare_supports(engine, state, incoming.source, incoming.action,
                             facts[:-1], incoming=incoming)
    second = compare_supports(engine, state, incoming.source, incoming.action,
                              facts[:-1], incoming=changed_outcome)
    for a, b in zip(first['rows'], second['rows']):
        assert a['supports'] == b['supports']
        assert a['expected'] == b['expected']
        assert a['score_before'] == b['score_before']
        assert a['raw'] == b['raw'] and a['path'] == b['path']
    for row, contrast in zip(first['rows'][1:], first['contrasts_to_full']):
        assert contrast['fixed_observed_outcome_gain_delta'] == pytest.approx(
            row['observed']['net_gain']-first['rows'][0]['observed']['net_gain'])
        torch.testing.assert_close(torch.tensor(contrast['prospective_expected_delta']),
            torch.tensor(row['expected'])-torch.tensor(first['rows'][0]['expected']))


def test_same_source_action_raw_is_identical_across_supports_and_options(fixture):
    engine, _, state, facts, source = fixture
    raw = raw_options(engine.model, source, source)
    result = compare_supports(engine, state, source, source, facts)
    assert raw.shape == (1, 3, 4)
    for row in result['rows']:
        torch.testing.assert_close(torch.tensor(row['raw']['prediction']), raw, rtol=0, atol=0)
        assert row['raw']['input_hash'] == result['rows'][0]['raw']['input_hash']


def test_different_supports_are_allowed_to_have_equal_gain(fixture):
    engine, _, state, facts, source = fixture
    incoming = facts[-1]
    # A budget of one and a sole old query give the same score gain with or
    # without that query's support: it is excluded from its own fit either way.
    one = replace(state, valid=state.valid[:1], slots=state.slots[:1], stamps=state.stamps[:1],
                  constraints=(facts[0],), organization=None)
    # With no old support vs a query's own support, score_scope excludes that
    # query in both cases. Both before and after scores are therefore equal.
    result = compare_supports(engine, one, incoming.source, incoming.action,
                              (facts[0],), incoming=incoming)
    full, _, without = result['rows']
    assert full['supports'] != without['supports']
    assert full['observed']['net_gain'] == pytest.approx(without['observed']['net_gain'], abs=2e-6)
    # No expected/observed inequality assertion belongs in this diagnostic.
    assert set(full['observed']) == set(without['observed'])


def test_historical_errors_describe_actual_same_version_unmerged_labels(fixture):
    engine, _, _, _, source = fixture
    m = engine.model.memory
    with torch.no_grad():
        m.occupied[:3] = True
        m.count[:3] = torch.tensor([1, 2, 1])
        m.wood_target_count[:3] = 1
        m.wood_option[:3] = torch.tensor([0, 1, 2])
        m.source[:3].copy_(source.expand(3, -1))
        m.action[:3].copy_(source.expand(3, -1))
        m.wood_target[:3, 3] = torch.tensor([.2, .3, -.4])
    # Exercise objective compatibility independently of incidental fixture tags.
    engine.reward_label_valid = lambda **kw: torch.tensor(
        [True, True, False]+[False]*(len(m.occupied)-3))
    report = historical_errors(engine)
    assert report['aggregate']['samples'] == 1
    assert report['by_option'][0]['samples'] == 1
    row = report['sample_rows'][0]
    assert row['base_error'] == pytest.approx(row['base']-row['label'])
    assert report['aggregate']['raw_squared_error']['mean'] == pytest.approx(row['raw_error']**2)


def test_current_binding_refuses_allocation_without_mutating_bank(fixture):
    engine, _, _, _, source = fixture
    before = copy.deepcopy(engine.model.causal_relations.state_dict())
    with pytest.raises(ValueError, match='allocate a new relation'):
        current_binding(engine, source)
    from test_joint_world_cycle import assert_same_saved_values
    assert_same_saved_values(before, engine.model.causal_relations.state_dict())
