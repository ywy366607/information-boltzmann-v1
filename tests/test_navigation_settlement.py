"""Signal provenance and wiring checks; not capability experiments."""
from types import SimpleNamespace

import pytest
import torch

from scripts.ib_local.language_navigation_kanli_v4b import ImaginingEngine
from scripts.ib_local.language_navigation_kanli_v4c import WitnessEngine, choice_replay_error
from scripts.ib_local.navigation_settlement import configure_settlement


def stub():
    memory = SimpleNamespace(capacity=5, occupied=torch.ones(5, dtype=torch.bool),
                             progress=torch.arange(5.), source=torch.ones(5, 3))
    return SimpleNamespace(model=SimpleNamespace(memory=memory))


def test_select_actual_continuation_not_a_sum_or_witness_alias():
    engine = stub()
    assert WitnessEngine.settlement_value(engine, -.4, .2) == -.4
    configure_settlement(engine, 'continuation')
    assert WitnessEngine.settlement_value(engine, -.4, .2) == .2
    with pytest.raises(FloatingPointError):
        WitnessEngine.settlement_value(engine, 0., float('nan'))


def test_migration_preserves_content_and_marks_old_labels_incompatible():
    engine = stub()
    before = {k: v.clone() for k, v in vars(engine.model.memory).items() if torch.is_tensor(v)}
    configure_settlement(engine, 'continuation')
    for key, value in before.items():
        assert torch.equal(getattr(engine.model.memory, key), value)
    assert not ImaginingEngine.reward_label_valid(engine).any()
    assert not ImaginingEngine.reward_label_valid(engine, wood=True).any()
    engine.settlement_versions[2] = 1
    engine.settlement_versions[3] = 2
    assert ImaginingEngine.reward_label_valid(engine).tolist() == [False, False, True, False, False]
    engine.wood_settlement_versions[1] = 1
    assert ImaginingEngine.reward_label_valid(engine, torch.tensor([0, 1, 2]), wood=True).tolist() == [False, True, False]
    assert engine.settlement_migration['metadata_bytes'] == 10


def test_configuration_is_idempotent_and_legacy_needs_no_metadata():
    engine = stub()
    configure_settlement(engine, 'witness')
    assert not hasattr(engine, 'settlement_versions')
    assert ImaginingEngine.reward_label_valid(engine).all()
    configure_settlement(engine, 'continuation')
    engine.settlement_versions[0] = 1
    configure_settlement(engine, 'continuation')
    assert int(engine.settlement_versions[0]) == 1
    with pytest.raises(ValueError):
        configure_settlement(engine, 'witness')


def test_joint_labels_cannot_reuse_cached_or_mixed_labels():
    engine = stub()
    configure_settlement(engine, 'continuation_joint')
    engine.settlement_versions[:] = torch.tensor([0, 1, 2, 3, 0], dtype=torch.uint8)
    assert ImaginingEngine.reward_label_valid(engine).tolist() == [False, False, False, True, False]
    assert WitnessEngine.settlement_value(engine, -.4, -.001, .36) == .36
    with pytest.raises(ValueError):
        WitnessEngine.settlement_value(engine, -.4, -.001)


def test_replay_checks_large_available_logits_instead_of_dropping_them():
    assert choice_replay_error([2000., -3000., -10000.],
                               [2000., -3000., -9999.], [0., 1., -10000.]) == 0.
    assert choice_replay_error([2001., -3000.], [2000., -3000.], [0., 1.]) == 1.


def test_replay_reports_nonfinite_or_missing_available_choices_explicitly():
    with pytest.raises(FloatingPointError):
        choice_replay_error([float('nan'), 0.], [0., 0.], [0., 0.])
    with pytest.raises(RuntimeError):
        choice_replay_error([-10000.], [-10000.], [-10000.])
