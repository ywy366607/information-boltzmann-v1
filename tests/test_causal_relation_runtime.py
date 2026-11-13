import pytest

from scripts.ib_local.causal_relation_runtime import RelationMemory, admit, revise_constant_relation


def test_admit_predict_then_settle_and_capacity():
    code = [{'op': 'input', 'arg': 0}, {'op': 'input', 'arg': 1},
            {'op': 'add', 'arg': 0}]
    relation = admit(code, 2, [[3, 4], [-1, 2]])
    bank = RelationMemory(capacity=1)
    bank.insert(relation)
    assert bank.predict(relation.digest, [5, 2]) == 7
    with pytest.raises(ValueError):
        bank.predict(relation.digest, [1, 2])
    assert bank.settle(9) == 2
    assert bank.entries[relation.digest]['count'] == 1
    with pytest.raises(ValueError):
        bank.settle(9)


def test_reject_unbounded_and_nonadmitted_code():
    with pytest.raises(ValueError):
        admit([{'op': 'exec', 'arg': 0}], 2, [[0, 0]])
    with pytest.raises(ValueError):
        RelationMemory().insert({'code': 'untrusted'})


def test_revision_requires_feedback_and_lean_checks_the_next_program():
    original = admit([{'op': 'const', 'arg': 0}], 2, [[0, 0]])
    state = {'code': [{'op': 'const', 'arg': 0}], 'digest': original.digest,
             'count': 0, 'feedback_sum': 0.}
    with pytest.raises(ValueError):
        revise_constant_relation(state)
    state.update(count=3, feedback_sum=-0.4)
    revised, receipt = revise_constant_relation(state)
    assert revised.code[0].arg == -20000
    assert receipt['old_digest'] == original.digest
    assert receipt['new_digest'] == revised.digest
    assert state['code'][0]['arg'] == 0
