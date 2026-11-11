import copy

import torch

from scripts.ib_local.navigation_mother_boundary import configure_mother_boundary
from scripts.ib_local.navigation_mother_boundary import MotherBoundaryEngine
from scripts.ib_local.navigation_internal_rollout import InternalRolloutEngine
from scripts.ib_local.navigation_persistent_question import (
    actual_boundary_evidence, configure_persistent_question, mother_response_evidence,
    retained_receipt, relation_from_receipt)
from scripts.ib_local.navigation_question_context import configure_question_context
from scripts.ib_local.navigation_relation_wood import configure_relation_wood, local_relation
from scripts.ib_local.navigation_shadow_preview import configure_shadow_preview
from test_navigation_internal_rollout import populate
from test_navigation_momentum_wood import configured


def candidate():
    e = configured(); populate(e); configure_relation_wood(e)
    configure_question_context(e); configure_shadow_preview(e); configure_mother_boundary(e)
    configure_persistent_question(e)
    return e


def test_migration_zero_and_versioned_receipt():
    e = candidate(); m = e.model.memory
    assert not m.question_valid.any()
    source = torch.randn(1, m.width)
    assert retained_receipt(e.model, source) is None
    _, packet = local_relation(e.model, source, source)
    m.question_valid[0] = True
    m.question_stamp[0] = m.last_write[0]
    m.question_slots[0] = packet['slots'][0]
    m.question_stamps[0] = packet['stamps'][0]
    m.question_support[0] = packet['valid'][0]
    stored = retained_receipt(e.model, source)
    value, used = relation_from_receipt(e, source, source, stored)
    assert used and torch.isfinite(value).all()
    assert torch.count_nonzero(e.model.question_reactivation(value)) == 0
    m.last_write[0] += 1
    assert retained_receipt(e.model, source) is None
    assert not relation_from_receipt(e, source, source, stored)[1]


def test_nested_receipts_reject_replaced_evidence():
    e = candidate(); m = e.model.memory
    source = torch.randn(1, m.width)
    _, packet = local_relation(e.model, source, source)
    m.question_valid[0] = True; m.question_stamp[0] = m.last_write[0]
    for name, key in [('question_slots','slots'), ('question_stamps','stamps'),
                      ('question_support','valid')]:
        getattr(m, name)[0] = packet[key][0]
    stored = retained_receipt(e.model, source)
    m.last_write[stored['slots'][0]] += 1
    assert not relation_from_receipt(e, source, source, stored)[1]


def test_buffers_survive_model_state_roundtrip():
    e = candidate(); m = e.model.memory
    m.question_valid[0] = True; m.question_stamp[0] = m.last_write[0]
    state = copy.deepcopy(e.model.state_dict())
    other = candidate(); other.model.load_state_dict(state)
    assert torch.equal(other.model.memory.question_valid, m.question_valid)
    assert torch.equal(other.model.memory.question_stamp, m.question_stamp)


def test_real_mother_outcome_trains_reactivated_read(monkeypatch):
    e = candidate(); m = e.model.memory
    source, action, target = torch.randn(m.width), torch.randn(m.width), torch.randn(m.width)
    _, packet = local_relation(e.model, source[None], source[None])
    m.question_valid[0] = True; m.question_stamp[0] = m.last_write[0]
    m.question_slots[0] = packet['slots'][0]
    m.question_stamps[0] = packet['stamps'][0]
    m.question_support[0] = packet['valid'][0]
    e._retained_question_receipt = retained_receipt(e.model, source[None])
    e._mother_boundary_target = target
    e._momentum_decision_read = torch.randn(e.model.momentum_to_boundary.in_features)
    monkeypatch.setattr(MotherBoundaryEngine, '_outcome_loss',
        lambda self, *args: (self.model.encoder.weight.sum()*0, {}))
    loss, diag = e._outcome_loss(source, action, target)
    loss.backward()
    assert diag['retained_question_samples'] == 1
    assert e.model.question_reactivation.weight.grad.norm() > 0
    assert e.model.long_predictor[0].weight.grad.norm() > 0


def test_actual_boundary_version_and_joint_relation_update():
    e = candidate(); m = e.model.memory
    source = torch.randn(1, m.width)
    # The original receipt precedes this new episode and excludes its slot.
    _, p = local_relation(e.model, source, source, exclude=torch.tensor([0]))
    m.question_valid[0] = True; m.question_stamp[0] = m.last_write[0]
    m.question_slots[0] = p['slots'][0]; m.question_stamps[0] = p['stamps'][0]
    m.question_support[0] = p['valid'][0]
    receipt = retained_receipt(e.model, source)
    e.joint_question_update = True
    actual = actual_boundary_evidence(e, receipt)
    assert actual is not None
    reconstructed, used = relation_from_receipt(e, source, source, receipt)
    expected, _ = local_relation(e.model, source, source,
        receipt=tuple(receipt[k] for k in ('slots','stamps','valid')), prior_extra=actual)
    assert used
    torch.testing.assert_close(reconstructed, expected)
    m.internal_stamp[0] -= 1
    assert actual_boundary_evidence(e, receipt) is None


def test_mother_and_later_boundary_keep_distinct_versioned_evidence():
    e = candidate(); m = e.model.memory
    source = torch.randn(1, m.width)
    _, p = local_relation(e.model, source, source, exclude=torch.tensor([0]))
    m.question_valid[0] = True; m.question_stamp[0] = m.last_write[0]
    m.question_slots[0] = p['slots'][0]; m.question_stamps[0] = p['stamps'][0]
    m.question_support[0] = p['valid'][0]
    m.mother_valid[0] = True; m.mother_stamp[0] = m.last_write[0]
    m.mother_next[0].fill_(3); m.internal_next[0].fill_(7)
    receipt = retained_receipt(e.model, source)
    mom = mother_response_evidence(e, receipt, batch=2)
    later = actual_boundary_evidence(e, receipt, batch=2)
    assert mom is not None and later is not None
    assert torch.all(mom[2] == 3) and torch.all(later[2] == 7)
    m.mother_stamp[0] -= 1
    assert mother_response_evidence(e, receipt) is None
    assert actual_boundary_evidence(e, receipt) is not None


def test_mother_relation_real_and_imagined_use_same_solve():
    e = candidate(); m = e.model.memory
    source, action = torch.randn(1, m.width), torch.randn(1, m.width)
    m.mother_valid.copy_(m.internal_valid); m.mother_stamp.copy_(m.last_write)
    m.mother_next.copy_(m.internal_next)
    _, p = local_relation(e.model, source, action, outcome_kind='mother')
    receipt = tuple(p[k] for k in ('slots', 'stamps', 'valid'))
    a = (torch.randn(1, m.width), torch.randn(1, m.width), torch.randn(1, m.width), torch.tensor([True]))
    b = (torch.randn(1, m.width), torch.randn(1, m.width), torch.randn(1, m.width), torch.tensor([True]))
    ab, _ = local_relation(e.model, source, action, receipt=receipt, prior_extra=a, extra=b, outcome_kind='mother')
    ba, _ = local_relation(e.model, source, action, receipt=receipt, prior_extra=b, extra=a, outcome_kind='mother')
    torch.testing.assert_close(ab, ba, atol=1e-6, rtol=1e-5)


def test_mother_shadow_migration_preserves_pending_and_calibration():
    from scripts.ib_local.navigation_shadow_preview import enable_mother_shadow
    e = candidate(); e.joint_question_update = True
    e.shadow_gain_state['count'][1] = 4
    e.internal_pending = {'shadow_forecast': {'old': 1}, 'gain': 2}
    enable_mother_shadow(e)
    assert e.mother_shadow_enabled
    assert e.shadow_gain_archive[-1]['state']['count'][1] == 4
    assert e.shadow_gain_state['count'][1] == 0
    assert e.internal_pending['pre_mother_shadow_forecast'] == {'old': 1}
    assert e.internal_pending['gain'] == 2


def test_real_and_imagined_evidence_share_bounded_solver():
    e = candidate(); m = e.model.memory
    source, action = torch.randn(1, m.width), torch.randn(1, m.width)
    _, p = local_relation(e.model, source, action)
    receipt = tuple(p[k] for k in ('slots','stamps','valid'))
    actual = (torch.randn(1,m.width), torch.randn(1,m.width), torch.randn(1,m.width), torch.tensor([True]))
    imagined = (torch.randn(1,m.width), torch.randn(1,m.width), torch.randn(1,m.width), torch.tensor([True]))
    ab, _ = local_relation(e.model,source,action,receipt=receipt,prior_extra=actual,extra=imagined)
    ba, _ = local_relation(e.model,source,action,receipt=receipt,prior_extra=imagined,extra=actual)
    torch.testing.assert_close(ab, ba, atol=1e-6, rtol=1e-5)
    off = (*imagined[:3], torch.tensor([False]))
    with_off, _ = local_relation(e.model,source,action,receipt=receipt,prior_extra=actual,extra=off)
    only_actual, _ = local_relation(e.model,source,action,receipt=receipt,prior_extra=actual)
    torch.testing.assert_close(with_off, only_actual, atol=1e-6, rtol=1e-5)


def test_boundary_is_captured_once_before_question_and_shadow(monkeypatch):
    from scripts.ib_local.navigation_question_context import QuestionContextEngine
    e = candidate(); e.align_question_boundary = True
    calls = []
    monkeypatch.setattr(InternalRolloutEngine, '_capture_boundary', lambda self: calls.append('capture'))
    def parent_utter(self):
        calls.append('question')
        self._capture_boundary()
        calls.append('shadow')
        return 'ok'
    monkeypatch.setattr(QuestionContextEngine, '_utter', parent_utter)
    assert e._utter() == 'ok'
    assert calls == ['capture', 'question', 'shadow']
    assert not e._question_boundary_captured
