from types import SimpleNamespace

import pytest
import torch

from scripts.ib_local.reactivated_credit_audit import audit_reactivated_credit, paired_reactivated_vjp


def _fixture():
    parameter = torch.nn.Parameter(torch.tensor(2.))
    old = parameter * torch.tensor([1., 2.])
    recent = parameter * torch.tensor([3., 4.])
    capsule = SimpleNamespace(version=7, inputs=torch.ones(1, 65, 3),
                              complete_inputs=torch.ones(1, 65, 3))
    memory = SimpleNamespace(occupied=[True, True], read_credit_version=[7, 7],
                             birth=[0, 200], clock=256)
    wrapper = SimpleNamespace(memory=memory, capsules=[capsule, capsule], replay_tokens=65,
                              reactivation_mode='complete_attention',
                              replayed_by_slot={0: old, 1: recent})
    loss = (old.sum() + recent.sum()) ** 2
    return parameter, wrapper, loss


def test_only_actual_strict_old_record_gets_current_teacher_without_mutation():
    parameter, wrapper, loss = _fixture()
    before = parameter.detach().clone()
    records = audit_reactivated_credit(loss, wrapper, [('weight', parameter)])
    assert len(records) == 1
    assert records[0]['slot'] == 0 and records[0]['end_age'] == 256
    assert records[0]['teaching_norm'] > 0 and records[0]['gradients']['weight'] > 0
    assert parameter.grad is None and torch.equal(parameter, before)
    loss.backward()
    assert parameter.grad is not None


def test_replaced_episode_rejected_and_detached_episode_not_fabricated():
    parameter, wrapper, loss = _fixture()
    wrapper.memory.read_credit_version[0] = 8
    with pytest.raises(RuntimeError, match='identity changed'):
        audit_reactivated_credit(loss, wrapper, [('weight', parameter)])
    wrapper.memory.read_credit_version[0] = 7
    wrapper.replayed_by_slot[0] = wrapper.replayed_by_slot[0].detach()
    assert audit_reactivated_credit(loss, wrapper, [('weight', parameter)])[0]['teaching_norm'] is None


def test_paired_vjp_swaps_only_actual_old_teachers():
    parameter, wrapper, _ = _fixture()
    wrapper.memory.birth[1] = 32
    old, other = wrapper.replayed_by_slot.values()
    loss = 2 * old.sum() + 7 * other.sum()
    result = paired_reactivated_vjp(loss, wrapper, [('weight', parameter)])
    assert result['identities'] == [0, 1]
    assert result['matched']['weight'].item() == 55
    assert result['mismatched']['weight'].item() == 35
    assert parameter.grad is None


def test_mismatch_handles_singleton_response_shape_without_fabricating_padding():
    parameter, wrapper, _ = _fixture()
    wrapper.memory.birth[1] = 32
    wrapper.memory.occupied.append(True)
    wrapper.memory.read_credit_version.append(7)
    wrapper.memory.birth.append(40)
    wrapper.capsules.append(wrapper.capsules[0])
    wrapper.replayed_by_slot[2] = parameter * torch.ones(3)
    old, other, singleton = wrapper.replayed_by_slot.values()
    loss = 2*old.sum()+7*other.sum()+11*singleton.sum()
    result = paired_reactivated_vjp(loss, wrapper, [('weight', parameter)])
    assert result['rotated'] == 2
    assert result['matched']['weight'].item() == 88
    assert result['mismatched']['weight'].item() == 68
