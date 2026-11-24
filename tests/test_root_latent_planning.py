"""Unit tests for RootLatentPlanning, Rank-Flip Sparse Teacher & Proportional 1280B Replay."""
from __future__ import annotations

from types import SimpleNamespace
import torch

from scripts.ib_local import root_latent_planning as rlp
from scripts.ib_local import alif_executive as alif


def _dummy_engine():
    eng = SimpleNamespace(
        gamma=0.9,
        joint_proposal_budget=4,
        _competing_action_roots={'r0': [0], 'r1': [1]},
    )
    return eng


def test_disabled_by_default():
    eng = _dummy_engine()
    assert not rlp.enabled(eng)
    assert rlp.proposal_budget(eng, 4) == 4
    assert rlp.teacher_budget(eng, [dict(priority=0.1)], torch.tensor([10.0, 0.0]), 2) == 2
    assert rlp.diagnostics(eng) is None


def test_rank_flip_teacher_gate_and_periodic_liveness():
    eng = _dummy_engine()
    cfg = {}
    rlp.configure(eng, cfg, teacher_cal_every=4, teacher_error_init=0.05)
    assert cfg['root_latent_planning']['schema'] == 1

    # Cold start (< 2 checks seen): always allocates full 2 teachers.
    menu = [dict(priority=0.1)]
    vals_wide = torch.tensor([5.0, 1.0])  # wood margin = 0.40 >> 2*0.9*0.1*0.05 = 0.009
    assert rlp.teacher_budget(eng, menu, vals_wide, 2) == 2
    rlp.observe_teacher_checks(eng, [dict(error=0.04), dict(error=0.06)])

    # Wide margin outside periodic tick -> 0 teachers (Theorem 1: rank-flip impossible).
    assert rlp.teacher_budget(eng, menu, vals_wide, 2) == 0

    # Narrow margin (wood margin = 0.002 < 0.009) -> full 2 teachers.
    vals_close = torch.tensor([1.02, 1.00])
    assert rlp.teacher_budget(eng, menu, vals_close, 2) == 2

    # Moderate margin (0.009 <= bound < 2*margin => margin=0.006 -> 2*margin=0.012) -> 1 teacher.
    vals_mid = torch.tensor([1.06, 1.00])
    assert rlp.teacher_budget(eng, menu, vals_mid, 2) == 1

    # Tick #4 (0-indexed: 0,1,2,3 done -> next is 4, 4 % 4 == 0): periodic calibration forces 2 teachers (Theorem 2).
    assert rlp.teacher_budget(eng, menu, vals_wide, 2) == 2

    d = rlp.diagnostics(eng)
    assert d['teacher_budget_0'] == 1
    assert d['teacher_budget_1'] == 1
    assert d['teacher_budget_2'] == 3
    assert d['teacher_checks_seen'] == 2


def test_proposal_budget_and_single_receipt():
    eng = _dummy_engine()
    rlp.configure(eng)
    eng._root_latent_plan = dict(
        escalated=False, reason='latent_single', effective_budget=1,
        candidate_count=4, selected_intent=2, selected_logp=-0.4,
        distribution=[0.1, 0.1, 0.7, 0.1],
        predicted_after_value=[0.0, 0.1, 0.5, 0.2],
        selection_value=[0.0, 0.1, 0.5, 0.2],
        novelty=[0.2, 0.2, 0.1, 0.3],
        selected_novelty=0.1, max_novelty=0.3, top_margin=0.3,
        steered=True, relation_identity=7, relation_version=3,
    )
    assert rlp.proposal_budget(eng, 4) == 1
    assert rlp.is_latent_single_active(eng)
    rec = rlp.build_single_selection_receipt(eng)
    assert rec['budget'] == 1 and rec['selected'] == 0 and rec['logp'] == 0.0
    assert rec['latent_intent_selected'] == 2 and rec['steered'] is True

    # Escalated case restores full budget 4.
    eng._root_latent_plan['escalated'] = True
    assert rlp.proposal_budget(eng, 4) == 4
    assert not rlp.is_latent_single_active(eng)


def test_proportional_replay_conservation_1280b():
    mem = SimpleNamespace(long=512, max_utterance=128)
    model = SimpleNamespace(max_chunk=640, memory=mem)
    eng = SimpleNamespace(model=model, replay_batch=32)
    alif.configure(eng, piece_bytes=1280)
    assert eng.model.max_chunk == 1280
    assert eng.model.memory.max_commit_chunk == 1280

    seen_batches = []

    def fake_wood_batch():
        seen_batches.append(eng.replay_batch)
        return None

    eng._wood_replay_batch = fake_wood_batch
    eng.model.kanli = SimpleNamespace()
    alif.hippocampal_anchor_loss(eng, torch.zeros((4, 8)))
    # At piece_bytes=1280 (2x of 640), replay_batch doubles to 64 inside call and restores to 32 after!
    assert seen_batches == [64]
    assert eng.replay_batch == 32
