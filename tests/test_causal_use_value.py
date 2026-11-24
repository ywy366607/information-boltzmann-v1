"""Algebra and full production-interface contracts, not capability training."""
import copy

import pytest
import torch

from scripts.ib_local.causal_use_value import configure, information
from scripts.ib_local.causal_relation_program import posterior


def test_predictive_information_matches_variance_reduction_and_permutation():
    torch.manual_seed(19)
    x, q, a = (torch.randn(n, 4, dtype=torch.float64) for n in (7, 3, 2))
    coefficient, targeted = information(x, q, a, .1)
    before = posterior(x, torch.zeros(7, 2, dtype=x.dtype), q, .1)
    for i in range(2):
        after = posterior(torch.cat((x, a[i:i+1])), torch.zeros(8, 2, dtype=x.dtype), q, .1)
        expected = .5*torch.log((1+before['leverage'])/(1+after['leverage']))
        torch.testing.assert_close(targeted[:, i], expected)
    permuted = information(x.flip(0), q.flip(0), a.flip(0), .1)
    torch.testing.assert_close(permuted[0], coefficient.flip(0))
    torch.testing.assert_close(permuted[1], targeted.flip((0, 1)))
    assert bool((targeted <= coefficient[None]+1e-12).all())


def test_orthogonal_unknown_has_zero_information_despite_novel_parameter():
    coefficient, targeted = information(torch.zeros(0, 2), torch.tensor([[1., 0.]]),
                                        torch.tensor([[0., 1.]]), .1)
    assert coefficient.item() > 0
    assert targeted.item() == 0


@pytest.mark.parametrize('arm', ['shadow', 'B', 'C', 'C2'])
def test_complete_world_cycle_uses_distinct_reward_contract_and_delayed_actor(tmp_path, arm):
    from test_joint_relation_update import body
    engine, archive = body(tmp_path, revision=4)
    engine.autonomous_event_clock = 0  # fixture uses explicit external dispatch
    configure(engine, arm)
    try:
        rows = []
        for i in range(3):
            engine.autonomous_event_clock = i
            rows.append(engine.step(torch.randn(engine.model.encoder.out_features, 4)))
        for row in rows:
            r = row['causal_use']
            assert r['prediction_event'] == r['settlement_event']
            assert r['prediction_update'] < r['settlement_update']
            assert r['unknown'] == row['causal_program']['roles']['unknown_identity']
            assert r['choice'] == ('silence', 'generate', 'reuse').index(row['choice'])
            assert row['metal_g'] == r['settlement']
            assert row['world_learning']['settlement'] == r['text_gain']
            if arm in ('C', 'C2'):
                assert r['settlement'] == r['observed_use']
            if arm == 'C2':
                assert r['reward_version'] == 'finite_targeted_use_v2'
                if r['finite_U']['committed']:
                    assert r['finite_U']['verified']
                    assert r['finite_U']['committed_score_max_error'] < 3e-5
                else:
                    assert r['settlement'] == 0.
            if arm == 'shadow':
                assert r['settlement'] == r['text_gain']
        assert rows[-1]['td_previous']['g'] == rows[-2]['metal_g']
        assert rows[-1]['delayed_actor_applied'] > 0
        assert rows[-1]['causal_program']['evidence_committed']
        assert engine.causal_use_pending is None
        # Fields are ordinary snapshot state; restoring configuration retains
        # learned calibration and clocks rather than silently reinitializing.
        saved = copy.deepcopy(engine.causal_use_calibration)
        configure(engine, arm)
        assert engine.causal_use_calibration == saved
    finally:
        archive.close()
