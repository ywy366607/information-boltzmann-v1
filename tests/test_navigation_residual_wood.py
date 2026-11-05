"""Numerical and causal bookkeeping checks, not capability experiments."""
from types import SimpleNamespace

import torch

from scripts.ib_local.language_navigation_kanli_v4p import ActionEvidence, ResidualWoodEngine
from scripts.ib_local.language_navigation_kanli_v4q import SharedActionEvidence, SharedContextWoodEngine


def test_forecast_is_prior_and_unseen_options_use_blind_prior():
    e = ActionEvidence()
    prior = e.bases().clone()
    target = torch.tensor([.3, .5, 1., -.2])
    e.observe(1, target, prior[1], prior[1])
    assert torch.equal(prior, torch.zeros(3, 4))
    assert torch.equal(e.bases(), target.expand(3, -1))
    assert e.seen.tolist() == [0, 1, 0]
    assert not e.confidence().any()


def test_bad_residual_keeps_mean_and_calibration_is_per_channel():
    e = ActionEvidence()
    t = torch.ones(4)
    e.observe(0, t, torch.zeros(4), torch.zeros(4))
    base = torch.zeros(4)
    raw = torch.tensor([1., 2., 3., 0.])
    e.observe(0, t, base, raw)
    assert torch.equal(e.confidence()[0], torch.tensor([1., 0., 0., 0.]))
    assert torch.equal((base+e.confidence()[0]*(raw-base))[1:], base[1:])


def test_evidence_state_roundtrip():
    e, restored = ActionEvidence(), ActionEvidence()
    for i in range(4):
        target = torch.full((4,), float(i))
        e.observe(i % 3, target, e.bases()[i % 3].clone(), torch.zeros(4))
    restored.load_state_dict(e.state_dict())
    assert torch.equal(e.bases(), restored.bases())
    assert torch.equal(e.confidence(), restored.confidence())


def test_mixed_action_merge_is_not_relabelled_as_last_action():
    m = SimpleNamespace(wood_target=torch.zeros(1, 4), wood_target_count=torch.zeros(1, dtype=torch.long),
                        wood_option=torch.full((1,), -1, dtype=torch.long), count=torch.ones(1, dtype=torch.long))
    engine = SimpleNamespace(model=SimpleNamespace(memory=m))
    row = dict(episode_slot=0, episode_merged=False)
    ResidualWoodEngine._store_label(engine, row, 1, torch.ones(4))
    row['episode_merged'] = True; m.count[0] = 2
    ResidualWoodEngine._store_label(engine, row, 2, torch.full((4,), 2.))
    assert m.wood_target_count[0] == 0 and m.wood_option[0] == -1


def test_same_action_merge_keeps_aligned_average():
    m = SimpleNamespace(wood_target=torch.ones(1, 4), wood_target_count=torch.ones(1, dtype=torch.long),
                        wood_option=torch.ones(1, dtype=torch.long), count=torch.full((1,), 2))
    engine = SimpleNamespace(model=SimpleNamespace(memory=m))
    ResidualWoodEngine._store_label(engine, dict(episode_slot=0, episode_merged=True), 1, torch.full((4,), 3.))
    assert torch.equal(m.wood_target, torch.full((1, 4), 2.))
    assert m.wood_target_count[0] == 2


def test_shared_innovation_moves_unselected_forecasts_and_preserves_contrasts():
    e = SharedActionEvidence(decay=.9)
    for choice in range(3):
        target = torch.full((4,), float(choice))
        e.observe(choice, target, e.bases()[choice].clone(), target)
    before = e.bases().clone()
    target = before[1] + 2
    e.observe(1, target, before[1], before[1])
    torch.testing.assert_close(e.bases()[1], before[1]+.2)
    torch.testing.assert_close(e.bases()[[0, 2]], before[[0, 2]]+.08)
    torch.testing.assert_close(e.mean.sum(0), torch.zeros(4), atol=1e-6, rtol=0)
    restored = SharedActionEvidence(decay=.9)
    restored.load_state_dict(e.state_dict())
    torch.testing.assert_close(restored.bases(), e.bases(), rtol=0, atol=0)


def test_first_action_initialization_does_not_move_other_forecasts():
    e = SharedActionEvidence()
    e.observe(0, torch.ones(4), torch.zeros(4), torch.zeros(4))
    before = e.bases().clone()
    e.observe(2, torch.full((4,), 4.), before[2], before[2])
    torch.testing.assert_close(e.bases()[:2], before[:2])
    torch.testing.assert_close(e.bases()[2], torch.full((4,), 4.))


def test_replay_residual_uses_historical_baseline_not_current_level():
    e = SimpleNamespace(_pending_record={'choice': 1}, device='cpu',
        _wood_source=torch.zeros(5), _wood_action=torch.ones(5),
        _forecast={'base': torch.full((3, 4), 9.)},
        _wood_replay=(torch.ones(2, 5), torch.ones(2, 5), torch.full((2, 4), 3.),
                      torch.tensor([0, 2]), torch.full((2, 4), 2.)))
    _, _, choices, labels = SharedContextWoodEngine._training_batch(e, torch.ones(4))
    base = SharedContextWoodEngine._training_base(e, choices)
    torch.testing.assert_close(base[0], torch.full((4,), 9.))
    torch.testing.assert_close((labels-base)[1:], torch.ones(2, 4))
