"""Measured-gain migration and TD write isolation; no training."""
import copy
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.ib_local.language_navigation_kanli_v4 import UtteranceMemory
from scripts.ib_local.language_navigation_kanli_v4b import ImaginingEngine
from scripts.ib_local.language_navigation_kanli_v4c import WitnessEngine
from scripts.ib_local.language_navigation_kanli_v4p import ResidualWoodEngine
from scripts.ib_local.navigation_settlement import configure_reward_memory
from scripts.ib_local import navigation_checkpoint


def engine():
    m = SimpleNamespace(occupied=torch.ones(6, dtype=torch.bool),
        count=torch.tensor([2, 3, 1, 2, 1, 1]), progress=torch.arange(6.)+100,
        progress_valid=torch.ones(6, dtype=torch.bool),
        wood_target=torch.tensor([[0., 0., 0., 2.], [0., 0., 0., 5.],
                                  [0., 0., 0., 6.], [0., 0., 0., 9.],
                                  [0., 0., 0., 8.], [0., 0., 0., float('nan')]]),
        wood_target_count=torch.tensor([2, 2, 1, 2, 1, 1]),
        wood_option=torch.tensor([0, 1, 2, 1, 2, 0]), option=torch.tensor([0, 1, 2, 1, 1, 0]))
    e = SimpleNamespace(model=SimpleNamespace(memory=m),settlement_kind='continuation_joint',
        settlement_versions=torch.tensor([3, 3, 0, 2, 3, 3], dtype=torch.uint8),
        wood_settlement_versions=torch.tensor([3, 3, 0, 2, 3, 3], dtype=torch.uint8),
        prev=dict(slot=0, g=.1, value=9., choice=0, settlement_kind='continuation_joint'),
        mode='silence', baseline=.02, baseline_decay=.95)
    e._write_return_label = MethodType(WitnessEngine._write_return_label, e)
    return e


def test_migration_requires_complete_same_action_finite_and_known_objective():
    e = engine()
    before = e.model.memory.progress.clone()
    configure_reward_memory(e, 'observed')
    m = e.model.memory
    assert m.progress_valid.tolist() == [True, False, True, False, False, False]
    torch.testing.assert_close(m.progress[[0, 2]], torch.tensor([.2, .6]))
    assert torch.equal(m.progress[[1, 3, 4, 5]], before[[1, 3, 4, 5]])
    assert ImaginingEngine.reward_label_valid(e).tolist() == [True, False, False, False, False, False]
    assert ImaginingEngine.reward_label_valid(e, wood=True).tolist() == [True, True, False, False, True, True]
    assert e.reward_memory_migration['extra_persistent_array_bytes'] == 0


def test_td_variation_never_overwrites_measured_mean_or_provenance():
    e = engine()
    configure_reward_memory(e, 'observed')
    gains, versions = e.model.memory.progress.clone(), e.settlement_versions.clone()
    for y in (-1e6, 0., 1e6):
        e._write_return_label(y)
        assert torch.equal(e.model.memory.progress, gains)
        assert torch.equal(e.settlement_versions, versions)
    # Full settlement takes the same y and advances its existing baseline.
    result = WitnessEngine._settle_previous(e, 12.)
    assert result['y'] == 12. and result['g'] == .1
    assert result['memory_label_kind'] == 'observed_gain'
    assert result['memory_gain'] == pytest.approx(.2) and result['memory_gain_valid']
    assert e.baseline == pytest.approx(.95*.02+.05*12.)
    assert torch.equal(e.model.memory.progress, gains)


def test_legacy_write_remains_identical_and_migration_is_idempotent():
    e = engine()
    e._write_return_label(9.)
    assert e.model.memory.progress[0] == 9.
    configure_reward_memory(e, 'observed')
    e.model.memory.progress[0] = 7.
    configure_reward_memory(e, 'observed')
    assert e.model.memory.progress[0] == 7.
    with pytest.raises(ValueError):
        configure_reward_memory(e, 'td')


def test_unknown_history_without_wood_is_preserved_but_not_claimed_observed():
    e = engine()
    del e.model.memory.wood_target
    before = e.model.memory.progress.clone()
    configure_reward_memory(e, 'observed')
    assert not e.model.memory.progress_valid.any()
    assert torch.equal(e.model.memory.progress, before)


def test_failed_popart_checkpoint_is_not_silently_converted_into_reference():
    e = engine()
    e.value_normalization = 'popart'
    before = e.model.memory.progress.clone()
    with pytest.raises(ValueError):
        configure_reward_memory(e, 'observed')
    assert torch.equal(e.model.memory.progress, before)


def test_real_episode_merge_keeps_average_gain_not_latest_gain_or_td():
    memory = UtteranceMemory(2, capacity=2, short=1, long=2, topk=1,
                             history_size=2, threshold=.7)
    memory.clock.fill_(10)
    encoder = nn.Identity()
    feature = torch.tensor([1., 0.])
    slot, merged = memory.commit_experience(feature,feature,feature,feature,encoder,b'x',.2,0)
    assert not merged
    e = SimpleNamespace(reward_memory_kind='observed',model=SimpleNamespace(memory=memory),
                        prev={'slot':slot})
    WitnessEngine._write_return_label(e, 100.)
    other, merged = memory.commit_experience(feature,feature,feature,feature,encoder,b'x',.6,0)
    assert other == slot and merged
    assert float(memory.progress[slot]) == pytest.approx(.4)
    assert memory.progress_valid[slot]
    WitnessEngine._write_return_label(e, -100.)
    assert float(memory.progress[slot]) == pytest.approx(.4)


def test_restoring_legacy_snapshot_clears_observed_mode_metadata(monkeypatch):
    e = engine()
    configure_reward_memory(e, 'observed')
    def restore_fields(target, state):
        for name, value in state['engine'].items():
            setattr(target, name, value)
    monkeypatch.setattr(navigation_checkpoint.legacy, 'restore', restore_fields)
    navigation_checkpoint.restore(e, dict(engine={},optimizer={}))
    assert not hasattr(e, 'reward_memory_kind')
    assert not hasattr(e, 'reward_memory_migration')


def test_observed_snapshot_keeps_mode_and_migration_on_restore(monkeypatch):
    e = engine()
    configure_reward_memory(e, 'observed')
    saved=dict(engine=dict(reward_memory_kind=e.reward_memory_kind,
                           reward_memory_migration=e.reward_memory_migration.copy()),optimizer={})
    def restore_fields(target, state):
        for name, value in state['engine'].items():
            setattr(target, name, value)
    monkeypatch.setattr(navigation_checkpoint.legacy, 'restore', restore_fields)
    navigation_checkpoint.restore(e,saved)
    assert e.reward_memory_kind == 'observed'
    assert e.reward_memory_migration == saved['engine']['reward_memory_migration']


def action_memory_engine():
    """Actual finite memory and read ports; no world, optimizer or training."""
    memory = UtteranceMemory(2, capacity=2, short=1, long=2, topk=1,
                             history_size=2, threshold=.7)
    for name, value in dict(
        option=torch.full((2,), -1, dtype=torch.long),
        wood_target=torch.zeros(2, 4),
        wood_target_count=torch.zeros(2, dtype=torch.long),
        wood_option=torch.full((2,), -1, dtype=torch.long),
    ).items():
        memory.register_buffer(name, value)
    memory.clock.fill_(10)
    model = SimpleNamespace(memory=memory, encoder=nn.Identity())
    model.plan_from_state = lambda h: dict(
        slots=torch.tensor([[0]]), weights=torch.ones(1, 1),
        decision_feature=h, action=h)
    model.kanli = SimpleNamespace(choice=lambda _: torch.zeros(1, 3))
    e = SimpleNamespace(model=model, felt_recall=False, reward_memory_kind='observed',
                        settlement_kind='continuation_joint',
                        settlement_versions=torch.full((2,), 3, dtype=torch.uint8))
    e.reward_label_valid = MethodType(ImaginingEngine.reward_label_valid, e)
    e.intero = lambda: torch.zeros(1, 3)
    return e


def commit_action(e, option, gain):
    feature = torch.tensor([1., 0.])
    m = e.model.memory
    slot, merged = m.commit_experience(
        feature, feature, feature, feature, e.model.encoder, b'x', gain, 0)
    # Match WitnessEngine.step followed by ResidualWoodEngine.step.
    m.option[slot] = option
    ResidualWoodEngine._store_label(e, dict(episode_slot=slot, episode_merged=merged),
                                    option, torch.tensor([0., 0., 1., 10*gain]))
    return slot


@pytest.mark.parametrize('first,second', [(1, 2), (2, 1), (0, 1)])
def test_mixed_action_measurements_stay_stored_but_cannot_be_recalled_as_last_action(first, second):
    e = action_memory_engine()
    slot = commit_action(e, first, .2)
    assert e.reward_label_valid()[slot]
    assert commit_action(e, second, .6) == slot
    m = e.model.memory
    assert m.occupied[slot] and m.progress_valid[slot]
    assert float(m.progress[slot]) == pytest.approx(.4)
    assert m.wood_option[slot] == -1 and m.wood_target_count[slot] == 0
    assert not e.reward_label_valid()[slot]
    assert not e.reward_label_valid(torch.tensor([slot])).any()
    recall = ImaginingEngine.recall_outcomes(e, torch.tensor([1., 0.]))
    assert torch.equal(recall, torch.zeros(3, 2))
    logits, _, reusable = ImaginingEngine.plan(e, torch.tensor([[1., 0.]]))
    assert not reusable.any() and logits[0, 2] == -1e4
    # Representation/content reads remain available despite ambiguous reward credit.
    read = m.read(torch.tensor([[1., 0.]]), e.model.encoder)
    assert float(read.weights.sum()) == pytest.approx(1.)


def test_same_action_mean_remains_recallable_and_survives_existing_memory_state_restore():
    e = action_memory_engine()
    slot = commit_action(e, 2, .2)
    assert commit_action(e, 2, .6) == slot
    assert e.reward_label_valid()[slot]
    recall = ImaginingEngine.recall_outcomes(e, torch.tensor([1., 0.]))
    assert float(recall[2, 0]) == pytest.approx(40.)
    logits, _, reusable = ImaginingEngine.plan(e, torch.tensor([[1., 0.]]))
    assert reusable.any() and logits[0, 2] == 0
    restored = action_memory_engine()
    restored.model.memory.load_state_dict(e.model.memory.state_dict())
    torch.testing.assert_close(restored.reward_label_valid(), e.reward_label_valid())
    torch.testing.assert_close(
        ImaginingEngine.recall_outcomes(restored, torch.tensor([1., 0.])), recall)


def test_old_observed_mode_restores_with_mixed_labels_filtered_without_migration_or_new_state():
    e = action_memory_engine()
    commit_action(e, 1, .2)
    slot = commit_action(e, 2, .6)
    restored = action_memory_engine()
    restored.model.memory.load_state_dict(e.model.memory.state_dict())
    before = copy.deepcopy(restored.model.memory.state_dict())
    # This is already observed mode; configure_reward_memory deliberately does nothing.
    configure_reward_memory(restored, 'observed')
    assert not restored.reward_label_valid()[slot]
    for key, value in restored.model.memory.state_dict().items():
        if torch.is_tensor(value):
            assert torch.equal(value, before[key])
        else:
            assert value == before[key]
    # Historical TD mode retains its original read contract.
    restored.reward_memory_kind = 'td'
    assert restored.reward_label_valid()[slot]
