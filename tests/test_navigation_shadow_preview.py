"""Shadow predictions are finite, bounded, and cannot write real evidence."""
import torch

from scripts.ib_local.navigation_relation_wood import configure_relation_wood, local_relation
from scripts.ib_local.navigation_question_context import configure_question_context
from scripts.ib_local.navigation_shadow_preview import configure_shadow_preview, preview, score_candidate_receipt, migrate_shadow_candidates
from test_navigation_momentum_wood import configured
from test_navigation_internal_rollout import populate


def test_preview_keeps_real_water_unchanged_and_supports_empty_bank():
    e = configured(); valid = populate(e); configure_relation_wood(e)
    configure_question_context(e); configure_shadow_preview(e)
    m, model = e.model.memory, e.model
    before = {k:v.clone() for k,v in m.state_dict().items() if torch.is_tensor(v)}
    source, action, latent = torch.randn(3,16),torch.randn(3,16),torch.randn(3,8)
    result = preview(model,source,action,latent,valid,ridge=.1,scale=.1)
    assert result['valid'].shape == (3,2) and result['valid'].all()
    assert torch.isfinite(result['future_delta']).all()
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    m.internal_valid.zero_()
    empty = preview(model,source,action,latent,valid,ridge=.1,scale=.1)
    assert not empty['usable'].any()
    assert torch.equal(empty['future_delta'],torch.zeros(3))


def test_preview_can_reuse_the_exact_versioned_relation_receipt():
    e = configured(); valid = populate(e); configure_relation_wood(e)
    configure_question_context(e); configure_shadow_preview(e)
    source, action, latent = torch.randn(3,16),torch.randn(3,16),torch.randn(3,8)
    _, packet = local_relation(e.model, source, action)
    receipt = tuple(packet[k] for k in ('slots', 'stamps', 'valid'))
    before = {k:v.clone() for k,v in e.model.memory.state_dict().items() if torch.is_tensor(v)}
    default = preview(e.model,source,action,latent,valid,ridge=.1,scale=.1)
    bound = preview(e.model,source,action,latent,valid,ridge=.1,scale=.1,
                    relation_receipt=receipt)
    torch.testing.assert_close(default['future_delta'], bound['future_delta'])
    torch.testing.assert_close(default['shadow_state_shift'], bound['shadow_state_shift'])
    assert all(torch.equal(v,e.model.memory.state_dict()[k]) for k,v in before.items())


def test_centered_candidates_keep_current_mean_and_keep_old_calibration():
    e = configured(); valid = populate(e); configure_relation_wood(e)
    configure_question_context(e); configure_shadow_preview(e)
    e.shadow_gain_state['count'][1] = 7
    migrate_shadow_candidates(e, 'centered')
    assert e.shadow_gain_archive[-1]['state']['count'][1] == 7
    assert e.shadow_gain_state['count'][1] == 0
    source, action, latent = torch.randn(3,16), torch.randn(3,16), torch.randn(3,8)
    result = preview(e.model,source,action,latent,valid,ridge=.1,scale=.1,candidate_kind='centered')
    means = (result['weights'][...,None]*result['candidate_latents']).sum(1)
    assert torch.allclose(means, latent, atol=1e-6)


def test_candidate_scoring_uses_decision_time_coordinates_and_valid_candidates():
    receipt = dict(candidates=torch.tensor([[2., 0.], [0., 2.]]),
                   weights=torch.tensor([.75, .25]), valid=torch.tensor([True, True]),
                   baseline=torch.tensor([0., 0.]))
    scores = score_candidate_receipt(receipt, torch.tensor([1., 0.]), 2*torch.eye(2))
    assert scores['baseline_mse'] == 2.
    assert scores['candidate_best_mse'] == 0.
    assert scores['candidate_mean_mse'] == .25
    receipt['valid'].zero_()
    assert score_candidate_receipt(receipt, torch.zeros(2), torch.eye(2)) is None
