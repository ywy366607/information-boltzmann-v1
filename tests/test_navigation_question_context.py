"""Pre-draft relation conditioning, migration and gradients."""
import copy

import torch
from torch.nn import functional as F

from scripts.ib_local.navigation_relation_wood import configure_relation_wood, local_relation, RelationWoodEngine
from scripts.ib_local.navigation_question_context import configure_question_context
from scripts.ib_local.navigation_internal_rollout import next_boundary_prediction
from test_navigation_momentum_wood import configured
from test_navigation_internal_rollout import populate


def test_question_context_zero_migration_and_action_logits():
    e = configured(); populate(e); configure_relation_wood(e)
    old = copy.deepcopy(e.model.state_dict())
    rng = torch.get_rng_state().clone()
    configure_question_context(e)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, e.model.state_dict()[k]) for k,v in old.items() if torch.is_tensor(v))
    ids = torch.randint(0, 256, (1, 2))
    logp = torch.full((1, 2, 256), -5.54517744448)
    e.model._question_context = None
    plain = e.model.predict_observed(ids, e.recurrent, logp)[0]
    e.model._question_context = torch.randn(8)
    zero = e.model.predict_observed(ids, e.recurrent, logp)[0]
    assert torch.equal(plain, zero)
    with torch.no_grad(): e.model.question_to_logits.weight.normal_()
    conditioned = e.model.predict_observed(ids, e.recurrent, logp)[0]
    assert not torch.equal(plain, conditioned)
    conditioned.square().mean().backward()
    assert e.model.question_to_logits.weight.grad.norm() > 0


def test_real_outcome_credits_only_valid_pre_draft_receipt_without_double_count(monkeypatch):
    e = configured(); populate(e); configure_relation_wood(e); configure_question_context(e)
    s, a, target = torch.randn(16), torch.randn(16), torch.randn(16)
    _, receipt = local_relation(e.model, s[None], a[None])
    p = torch.randn(e.model.momentum_to_boundary.in_features)
    e._internal_label = (s,a,target)
    e.internal_pending = dict(slot=7, momentum_read=p)
    for prefix in ('relation_', 'question_'):
        for name in ('slots', 'stamps', 'valid'):
            e.internal_pending[prefix+name] = receipt[name][0].clone()
    correction, _ = local_relation(e.model, s[None], a[None], exclude=torch.tensor([7]),
        receipt=tuple(receipt[k] for k in ('slots','stamps','valid')))
    base = next_boundary_prediction(e.model, s[None], a[None]) + e.model.momentum_to_boundary(p[None])
    old = F.mse_loss(base+correction, e.model.encoder(target[None]))
    monkeypatch.setattr(RelationWoodEngine, '_outcome_loss', lambda self,*args: (old, {'relation_samples':1}))
    loss, diag = e._outcome_loss(None,None,None)
    torch.testing.assert_close(loss, old)
    assert diag['question_evidence_samples'] == 1
    e.model.memory.last_write[receipt['slots'][0]] += 1
    loss, diag = e._outcome_loss(None,None,None)
    assert diag['question_evidence_samples'] == 0
