"""Exact reuse of an immutable relation snapshot inside one imagination."""
from dataclasses import replace
import copy
from types import SimpleNamespace

import pytest
import torch

from scripts.ib_local.causal_relation_program import Evidence, Port, RelationState, mechanism
from scripts.ib_local.preview_runtime import preview_scope
from scripts.ib_local.prospective_decision import preview_bind
from scripts.ib_local.shared_relation_basis import SharedRelationBasis


def test_snapshot_reuse_keeps_branch_overlay_and_expires_at_scope_end(monkeypatch):
    slots = torch.tensor([0, 1])
    stamps = torch.tensor([1, 1])
    initial = RelationState(7, 2, 3, 0, slots, stamps, torch.tensor([True, False]))
    calls = []

    def snapshot(index):
        calls.append(index)
        return replace(initial, version=len(calls))

    bank = SimpleNamespace(binding_address=lambda *a: (True, 0, torch.tensor(.9)),
                           preview_state=snapshot)
    memory = SimpleNamespace(occupied=torch.ones(2, dtype=torch.bool), count=stamps,
                             last_write=stamps, mother_valid=torch.ones(2, dtype=torch.bool),
                             mother_stamp=stamps)
    model = SimpleNamespace(causal_relations=bank, memory=memory)
    source = torch.randn(1, 5)
    overlay = replace(initial, constraints=('private branch',), binding=None)
    with torch.no_grad(), preview_scope(model):
        first, _ = preview_bind(model, source)
        private, _ = preview_bind(model, source + 1, overlay=overlay)
        again, _ = preview_bind(model, source - 1)
        assert first.version == again.version == 1
        assert private.constraints == ('private branch',)
        assert first.constraints == again.constraints == ()
        assert calls == [0]
    with torch.no_grad(), preview_scope(model):
        fresh, _ = preview_bind(model, source)
        assert fresh.version == 2
    assert calls == [0, 0]


@pytest.mark.parametrize('kind,transport', [(Port.MOTHER, False),
    (Port.DECISION, False), (Port.DECISION, True)])
@pytest.mark.parametrize('count', [0, 4])
def test_canonical_empty_prefix_matches_original_hot_prefix_values_and_gradients(kind, transport, count):
    torch.manual_seed(1123)
    linear = lambda a, b: torch.nn.Linear(a, b, dtype=torch.float64)
    model = torch.nn.Module()
    model.encoder = linear(12, 8)
    model.long_predictor = torch.nn.Sequential(linear(16, 8), torch.nn.SiLU(), linear(8, 8))
    model.mother_from_boundary = linear(8, 8)
    model.relation_basis = SharedRelationBasis(8, dtype=torch.float64)
    raw = torch.randn(3, 12, dtype=torch.float64)
    memory = SimpleNamespace(source=raw, action=raw + .1, mother_next=raw - .2,
        internal_next=raw + .3, occupied=torch.ones(3, dtype=torch.bool),
        count=torch.ones(3, dtype=torch.long), last_write=torch.ones(3, dtype=torch.long),
        mother_valid=torch.ones(3, dtype=torch.bool), mother_stamp=torch.ones(3, dtype=torch.long),
        internal_valid=torch.ones(3, dtype=torch.bool), internal_stamp=torch.ones(3, dtype=torch.long))
    model.memory = memory
    facts = tuple(Evidence(torch.randn(1, 12, dtype=torch.float64),
        torch.randn(1, 12, dtype=torch.float64), torch.randn(1, 12, dtype=torch.float64),
        kind, True, mother=torch.randn(1, 12, dtype=torch.float64), fact_id=f'episode:{i}')
        for i in range(count))
    empty = torch.empty(0, dtype=torch.long)
    state = RelationState(1, 1, 1, 0, empty, empty, torch.empty(0, dtype=torch.bool),
                          constraints=facts, evidence_bound=True)
    source, action = raw[:2], raw[1:]
    mother = model.encoder(raw[:2]) if transport else None
    values, gradients = [], []
    for revision in (2, 3):
        model.joint_relation_update_revision = revision
        result = mechanism(model, state, source, action, kind=kind, mother=mother, ridge=.2)
        selected = [result[k] for k in ('mean', 'scale2', 'noise', 'leverage')]
        values.append(selected)
        gradients.append(torch.autograd.grad(sum(t.square().sum() for t in selected),
            tuple(model.parameters()), allow_unused=True, retain_graph=True))
    for before, after in zip(*values):
        torch.testing.assert_close(before, after, rtol=1e-11, atol=1e-11)
    for before, after in zip(*gradients):
        if before is None or after is None:
            assert before is after
        else:
            torch.testing.assert_close(before, after, rtol=1e-10, atol=1e-10)


def test_actual_typed_context_matches_cancelled_legacy_context():
    from test_generative_cycle import configured
    reference = configured()
    reference._binding_choices = []
    current = copy.deepcopy(reference)
    current.native_direct_question = True
    raw = reference.model.memory.source[0]
    context = raw.new_zeros(1, reference.model.encoder.out_features)
    rng = torch.get_rng_state()
    before = reference._condition_mother_question(raw, context)
    torch.set_rng_state(rng)
    after = current._condition_mother_question(raw, context)
    torch.testing.assert_close(after, before, atol=2e-6, rtol=2e-6)
    assert reference._generative_query['role'] == current._generative_query['role']
    torch.testing.assert_close(reference._program_question_action, current._program_question_action)
    assert current._program_pre_question['runtime'].startswith('actual typed question')


def test_shared_external_planning_scope_keeps_actual_delivery_and_rng():
    from test_generative_cycle import configured
    from test_joint_world_cycle import assert_same_saved_values
    from scripts.ib_local.preview_runtime import _scope
    torch.set_num_threads(1)
    torch.manual_seed(231123)
    reference = configured()
    reference.step(torch.randn(8, 4))
    current = copy.deepcopy(reference)
    current.native_shared_utter_preview = True
    rng = torch.get_rng_state()
    words, record = reference._utter()
    following_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    other_words, other_record = current._utter()
    assert words == other_words
    assert_same_saved_values(record, other_record)
    assert_same_saved_values(reference.model.state_dict(), current.model.state_dict())
    assert_same_saved_values(reference.table.state_dict(), current.table.state_dict())
    assert_same_saved_values(reference.decision.state, current.decision.state)
    assert torch.equal(following_rng, torch.get_rng_state())
    assert _scope.get() is None


def test_shared_decision_scope_starts_after_real_question_binding(tmp_path, monkeypatch):
    from test_persistent_rhythm import body
    from scripts.ib_local import autonomous_cycle, shared_unknown_bindings
    from scripts.ib_local.preview_runtime import _scope
    torch.set_num_threads(1)
    torch.manual_seed(201120)
    engine, archive = body(tmp_path, root_mode='observe')
    engine.native_shared_utter_preview = True
    select = shared_unknown_bindings.select
    calls = []

    def checked_select(*args, **kwargs):
        # This call opens/touches the actual SQLite unknown and its support.
        # A snapshot captured before it is stale for later native teachers.
        assert _scope.get() is None
        calls.append(1)
        return select(*args, **kwargs)

    monkeypatch.setattr(shared_unknown_bindings, 'select', checked_select)
    try:
        completed = 0
        for _ in range(96):
            row = autonomous_cycle.dispatch(engine)
            completed += row['autonomous_event']['kind'] == 'external'
            if completed == 2:
                break
        assert completed == 2 and len(calls) == 2
        assert _scope.get() is None
    finally:
        archive.close()


def test_preview_encoder_cache_invalidates_inputs_weights_and_bypasses_gradients():
    from scripts.ib_local.latent_navigation_v3 import ExperienceMemory
    from scripts.ib_local.epistemic_value_reference import encode_fact
    model = SimpleNamespace(encoder=torch.nn.Linear(3, 2, bias=False))
    memory = ExperienceMemory(3, capacity=4, short=1, long=2, topk=1)
    memory.occupied[0] = True
    memory.source[0] = torch.tensor([1., 2., 3.])
    memory.action[0] = torch.tensor([2., 3., 4.])
    fact = Evidence(memory.source[:1], memory.action[:1], memory.long_successor[:1], Port.MOTHER,
                    True, fact_id='episode:1')
    calls = []
    hook = model.encoder.register_forward_hook(lambda *args: calls.append(1))
    query = torch.ones(1, 2)
    with torch.no_grad(), preview_scope(model):
        before = memory.read_encoded(query, model.encoder)
        first = len(calls)
        again = memory.read_encoded(query * 2, model.encoder)
        assert len(calls) == first
        torch.testing.assert_close(before.value, again.value)
        a, b = encode_fact(model, fact), encode_fact(model, fact)
        assert a is b
        memory.action.add_(.2)
        changed = memory.read_encoded(query, model.encoder)
        assert len(calls) > first
        assert not torch.equal(before.value, changed.value)
        model.encoder.weight.add_(.1)
        assert encode_fact(model, fact) is not a
        with torch.enable_grad():
            trained = memory.read_encoded(query, model.encoder)
            trained.value.sum().backward()
            assert model.encoder.weight.grad is not None
    hook.remove()
