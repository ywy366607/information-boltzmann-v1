import torch

from scripts.ib_local.navigation_mother_boundary import configure_mother_boundary, mother_boundary_feature
from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
from scripts.ib_local.navigation_relation_wood import configure_relation_wood
from scripts.ib_local.navigation_question_context import configure_question_context
from scripts.ib_local.navigation_shadow_preview import ShadowPreviewEngine, configure_shadow_preview
from test_navigation_momentum_wood import configured
from test_navigation_internal_rollout import populate


def test_actual_boundary_uses_last_mother_byte_before_ambient():
    features = torch.arange(30).reshape(10, 3)
    info = dict(source='mother', mother_bytes=3)
    assert torch.equal(mother_boundary_feature(features, 2, 6, info), features[5])
    assert mother_boundary_feature(features, 2, 3, info) is None
    assert mother_boundary_feature(features, 2, 6, dict(source='other', mother_bytes=3)) is None


def test_real_mother_boundary_updates_shared_relation_and_momentum(monkeypatch):
    e = configured(); populate(e); configure_relation_wood(e)
    configure_question_context(e); configure_shadow_preview(e); configure_mother_boundary(e)
    source, action, target = torch.randn(16), torch.randn(16), torch.randn(16)
    for p in e.bfss_momenta:
        p.normal_()
    e._momentum_decision_read = decision_momentum_read(e.model, e.bfss_momenta, source[None])[0].detach()
    e._mother_boundary_target = target
    monkeypatch.setattr(ShadowPreviewEngine, '_outcome_loss',
        lambda self, *args: (self.model.encoder.weight.sum()*0, {}))
    loss, diag = e._outcome_loss(source, action, target)
    loss.backward()
    assert diag['mother_boundary_samples'] == 1
    assert e.model.mother_from_boundary.weight.grad.norm() > 0
    assert e.model.momentum_to_boundary.weight.grad.norm() > 0
    assert e.model.encoder.weight.grad.norm() > 0
