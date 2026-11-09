"""Numerical wiring checks, not synthetic memory capability experiments."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from scripts.ib_local.experience_consolidation import (
    bounded_preference, replacement_slot, select_read_slots,
)
from scripts.ib_local.navigation_read_credit import configure_read_credit
from scripts.ib_local.pretrained_gdn_experience import LayerExperienceRecall, LocalExperience


def populated():
    wrapper = LayerExperienceRecall(3, capacity=2, latent_dim=3, consolidation=True)
    with torch.no_grad():
        wrapper.encoder.weight.copy_(torch.eye(3))
        wrapper.encoder.bias.zero_()
    memory = wrapper.memory
    memory.clock.fill_(256)
    for i, value in enumerate((2., -2.)):
        x = torch.tensor([1., 0., 0.])
        slot, _ = memory.commit_episode(x, x, x, x, wrapper.encoder, source_time=i)
        inputs = torch.zeros(1, 65, 3)
        inputs[..., 0] = value
        wrapper.capsules[slot] = LocalExperience(inputs, None, int(memory.read_credit_version[slot]))
    memory.last_write.copy_(torch.tensor([0, 10]))
    memory.clock.fill_(300)
    return wrapper


def settle(wrapper, target):
    # Large test LR makes the direction visible; production retains its real
    # optimizer settings. No chronological graph or fabricated reward is used.
    wrapper.begin(dict(lr=.5, betas=(0., 0.), eps=1e-8))
    read = wrapper.retrieve(torch.tensor([[1., 0., 0.]]), lambda capsule: capsule.inputs)
    logits = torch.stack((read[0, 0], read[0, 0]*0))[None]
    F.cross_entropy(logits, torch.tensor([target])).backward()
    return wrapper.memory.settle_read_credit()


def test_actual_prediction_credit_changes_retention_and_opposing_evidence_releases():
    wrapper = populated()
    m = wrapper.memory
    assert replacement_slot(m) == 0  # oldest before any outcome evidence
    result = settle(wrapper, 0)
    assert result['records'] == 2
    assert m.read_credit_mean[0] > 0 > m.read_credit_mean[1]
    assert replacement_slot(m) == 1  # useful older capsule survives
    for _ in range(2):
        settle(wrapper, 1)
    assert m.read_credit_mean[0] < 0 < m.read_credit_mean[1]
    assert replacement_slot(m) == 0  # real contrary outcome reverses protection


def test_preference_participates_before_topk_without_double_application():
    wrapper = populated()
    m = wrapper.memory
    m.topk = 1
    m.read_credit_enabled = True
    m.read_credit_mean.copy_(torch.tensor([1., -1.]))
    scores = torch.tensor([[.8, .9]], requires_grad=True)
    selected, slots = select_read_slots(m, scores)
    assert slots.tolist() == [[0]]  # weaker content match can win on evidence
    assert selected.item() == scores[0, 0].item()  # unmodified similarity
    selected.sum().backward()
    assert scores.grad.tolist() == [[1., 0.]]


def test_zero_preference_is_exact_legacy_selection_and_oldest_replacement():
    wrapper = populated()
    m = wrapper.memory
    scores = torch.tensor([[.3, .6]])
    selected, slots = select_read_slots(m, scores)
    expected = scores.topk(m.topk, dim=-1)
    assert torch.equal(selected, expected.values)
    assert torch.equal(slots, expected.indices)
    assert replacement_slot(m) == 0
    before = wrapper.retrieve(torch.ones(1, 3), lambda c: c.inputs)
    m.consolidation_enabled = False
    after = wrapper.retrieve(torch.ones(1, 3), lambda c: c.inputs)
    torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_reads_alone_cannot_refresh_protection_and_saturation_is_bounded():
    wrapper = populated()
    m = wrapper.memory
    m.read_credit_enabled = True
    m.read_credit_mean.copy_(torch.tensor([1e6, -1e6]))
    multiplier = bounded_preference(m).exp()
    assert float(multiplier.max()) <= math.e + 1e-6
    assert float(multiplier.min()) >= 1/math.e - 1e-6
    before = replacement_slot(m)
    m.last_touch.fill_(1_000_000)
    assert replacement_slot(m) == before
    m.last_write.copy_(torch.tensor([0, 999]))
    m.clock.fill_(1000)
    assert replacement_slot(m) == 0  # old saturation cannot defeat arbitrary age


def test_empty_slots_and_replacement_reset_identity_without_increasing_storage():
    wrapper = populated()
    m = wrapper.memory
    original_bytes = wrapper.storage_nbytes
    m.occupied[1] = False
    assert replacement_slot(m) == 1
    m.occupied[1] = True
    m.read_credit_enabled = True
    m.read_credit_mean.copy_(torch.tensor([.5, -.5]))
    version = int(m.read_credit_version[1])
    m.read_credit_first[1] = 2
    x = torch.ones(3)
    slot, _ = m.commit_episode(x, x, x, x, wrapper.encoder, source_time=128)
    wrapper.capsules[slot] = LocalExperience(torch.zeros(1, 65, 3), None,
                                            int(m.read_credit_version[slot]))
    assert slot == 1
    assert int(m.read_credit_version[slot]) == version + 1
    assert m.read_credit_mean[slot] == m.read_credit_first[slot] == 0
    assert wrapper.storage_nbytes == original_bytes


def test_restore_preserves_policy_and_legacy_snapshot_defaults_off():
    wrapper = populated()
    settle(wrapper, 0)
    state = copy.deepcopy(wrapper.state_dict())
    other = LayerExperienceRecall(3, capacity=2, latent_dim=3)
    other.load_state_dict(state)
    assert other.memory.consolidation_enabled
    assert replacement_slot(other.memory) == replacement_slot(wrapper.memory)
    query = torch.tensor([[1., 0., 0.]])
    with torch.no_grad():
        torch.testing.assert_close(wrapper.retrieve(query, lambda c: c.inputs),
                                   other.retrieve(query, lambda c: c.inputs), atol=0, rtol=0)
    del state['memory._extra_state']['consolidation_enabled']
    other.load_state_dict(state)
    assert not other.memory.consolidation_enabled


def test_disabled_credit_does_not_reuse_dormant_preferences():
    wrapper = populated()
    m = wrapper.memory
    m.read_credit_mean.copy_(torch.tensor([100., -100.]))
    configure_read_credit(m, 'off')
    assert torch.equal(bounded_preference(m), torch.zeros(2))
    assert replacement_slot(m) == 0


def test_single_selected_record_cannot_claim_relative_credit_from_softmax():
    wrapper = populated()
    wrapper.memory.topk = 1
    settle(wrapper, 0)
    # Current credit compares selected alternatives, not absolute usefulness.
    # A single softmax weight is fixed at one; do not invent an advantage.
    assert wrapper.memory.read_credit_mean.abs().max() == 0


def test_bounded_protection_does_not_exempt_a_slot_from_continuing_admissions():
    wrapper = populated()
    m = wrapper.memory
    m.read_credit_enabled = True
    m.read_credit_mean.copy_(torch.tensor([1e6, -1e6]))
    m.clock.zero_()
    m.last_write.zero_()
    original_version = int(m.read_credit_version[0])
    x = torch.ones(3)
    for tick in range(1, 40):
        m.clock.fill_(tick + m.long)
        slot, _ = m.commit_episode(x, x, x, x, wrapper.encoder, source_time=tick)
        if slot == 0:
            break
        # Even under maximally unfavorable competition the old positive bias
        # cannot stop its age growing or create an extra protected slot.
        m.read_credit_mean[slot] = -1e6
    assert int(m.read_credit_version[0]) > original_version
    assert int(m.occupied.sum()) == 2


def test_existing_moment_evidence_has_no_optimizer_step_size_bottleneck():
    wrappers = [populated(),populated()]
    for wrapper,lr in zip(wrappers,(3e-4,1e-2)):
        wrapper.memory.consolidation_rule = 'evidence'
        wrapper.begin(dict(lr=lr,betas=(.9,.999),eps=1e-8))
        value = wrapper.retrieve(torch.tensor([[1.,0.,0.]]),lambda c:c.inputs)
        F.cross_entropy(torch.stack((value[0,0],value[0,0]*0))[None],torch.tensor([0])).backward()
        wrapper.memory.settle_read_credit()
    first,second = [bounded_preference(w.memory) for w in wrappers]
    torch.testing.assert_close(first,second,atol=0,rtol=0)
    assert first[0] > .5 > first[1]
    assert replacement_slot(wrappers[0].memory) == 1
    assert wrappers[0].memory.read_credit_mean[0] < .001  # original displacement still small


def test_moment_evidence_is_neutral_without_observations_and_scale_consistent():
    wrapper = populated()
    m = wrapper.memory
    m.consolidation_rule = 'evidence'
    m.read_credit_rule = 'adam'
    m.read_credit_enabled = True
    m.read_credit_mean.fill_(100.)
    assert bounded_preference(m).abs().max() == 0
    m.read_credit_steps.fill_(1)
    m.read_credit_first.copy_(torch.tensor([.1,-.1]))
    m.read_credit_second.fill_(.001)
    first = bounded_preference(m).clone()
    m.read_credit_first.mul_(10)
    m.read_credit_second.mul_(100)
    torch.testing.assert_close(first,bounded_preference(m),atol=1e-6,rtol=1e-6)
    assert first[0] > 0 > first[1]


def test_moment_policy_receipt_matches_actual_selected_attention_logits_and_restores():
    import copy
    wrapper = populated()
    wrapper.memory.consolidation_rule = 'evidence'
    settle(wrapper,0)
    state = copy.deepcopy(wrapper.state_dict())
    restored = LayerExperienceRecall(3,capacity=2,latent_dim=3)
    restored.load_state_dict(state)
    assert restored.memory.consolidation_rule == 'evidence'
    query = torch.tensor([[1.,0.,0.]])
    torch.testing.assert_close(wrapper.retrieve(query,lambda c:c.inputs),
                               restored.retrieve(query,lambda c:c.inputs),atol=0,rtol=0)
    assert replacement_slot(wrapper.memory) == replacement_slot(restored.memory)


def test_retention_only_preserves_reads_and_read_learning_but_changes_eviction():
    legacy = populated()
    legacy.memory.consolidation_enabled = False
    upgraded = copy.deepcopy(legacy)
    upgraded.memory.consolidation_enabled = True
    upgraded.memory.consolidation_rule = 'retention'
    assert settle(legacy, 0) == settle(upgraded, 0)
    for name in ('read_credit_mean', 'read_credit_first', 'read_credit_second',
                 'read_credit_steps', 'read_credit_count'):
        torch.testing.assert_close(getattr(legacy.memory, name),
                                   getattr(upgraded.memory, name), atol=0, rtol=0)
    query = torch.tensor([[1., 0., 0.]])
    left = legacy.retrieve(query, lambda c: c.inputs)
    right = upgraded.retrieve(query, lambda c: c.inputs)
    torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert replacement_slot(upgraded.memory) == 1
    for _ in range(2):
        settle(upgraded, 1)
    assert replacement_slot(upgraded.memory) == 0


def test_retention_only_does_not_route_by_lifetime_evidence():
    from scripts.ib_local.navigation_read_credit import read_logits
    from scripts.ib_local.experience_consolidation import consolidates_reads

    wrapper = populated()
    m = wrapper.memory
    m.consolidation_rule = 'retention'
    m.read_credit_enabled = True
    m.read_credit_rule = 'adam'
    m.read_credit_steps.fill_(1)
    m.read_credit_first.copy_(torch.tensor([.1, -.1]))
    m.read_credit_second.fill_(.001)
    m.read_credit_mean.copy_(torch.tensor([.02, -.03]))
    assert not consolidates_reads(m)
    scores = torch.tensor([[.8, .9]])
    slots = scores.topk(2, dim=-1).indices
    valid = torch.ones_like(slots, dtype=torch.bool)
    actual = read_logits(m, scores.gather(-1, slots), slots, valid)
    expected = scores.gather(-1, slots)/m.temperature + m.read_credit_mean[slots].tanh()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert bounded_preference(m)[0] > .5
    assert replacement_slot(m) == 1


def test_retention_upgrade_and_restore_preserve_all_existing_learning_state():
    from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
    from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same

    wrapper = populated()
    wrapper.memory.consolidation_enabled = False
    settle(wrapper, 0)
    before = copy.deepcopy(wrapper.state_dict())
    learner = PretrainedExperienceLearner.__new__(PretrainedExperienceLearner)
    learner.memories = {1: wrapper}
    learner.enable_credit_retention()
    after = copy.deepcopy(wrapper.state_dict())
    expected = copy.deepcopy(before)
    expected['memory._extra_state']['consolidation_enabled'] = True
    expected['memory._extra_state']['consolidation_rule'] = 'retention'
    assert_same(after, expected)
    restored = LayerExperienceRecall(3, capacity=2, latent_dim=3)
    restored.load_state_dict(after)
    assert restored.memory.consolidation_rule == 'retention'
    assert replacement_slot(restored.memory) == replacement_slot(wrapper.memory)
    wrapper.current = {'unsettled': True}
    import pytest
    with pytest.raises(RuntimeError, match='finish the current observation'):
        learner.enable_credit_retention()
