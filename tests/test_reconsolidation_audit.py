"""Numerical credit and diagnostic-isolation checks, not capability studies."""
from types import SimpleNamespace

import pytest
import torch

from scripts.ib_local.audit_pretrained_gdn_reconsolidation import lineages, receipt_batch, shadow_settle
from scripts.ib_local.latent_navigation_v3 import ExperienceMemory
from scripts.ib_local.navigation_read_credit import configure_read_credit, ReadParticipation
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same


def test_positive_value_can_have_negative_competitive_credit_even_with_null():
    logits = torch.zeros(3, requires_grad=True)
    weights = logits.softmax(0)
    values = torch.tensor([1., 3., 0.])
    prediction = (weights*values).sum()
    target = prediction.detach()+1
    loss = (prediction-target).square()/2
    logit_grad, weight_grad = torch.autograd.grad(loss, (logits, weights))
    assert -weights[0]*weight_grad[0] > 0 > -logit_grad[0]
    drop = prediction.detach()-weights[0].detach()*values[0]
    assert (drop-target).square()/2 > loss.detach()


def test_shadow_settlement_uses_actual_routine_without_changing_live_memory():
    m = ExperienceMemory(3,capacity=2,topk=2)
    m.occupied.fill_(True)
    m.consolidation_enabled, m.consolidation_rule = True, 'retention'
    configure_read_credit(m, 'adam', dict(lr=3e-4,betas=(.9,.999),eps=1e-8))
    slots=torch.tensor([[0,1]])
    receipt=ReadParticipation(torch.zeros(1,2,requires_grad=True),slots,
        m.read_credit_version[slots].clone(),torch.ones(1,2,dtype=torch.bool))
    before={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in m.state_dict().items()}
    prior,after=shadow_settle(m,receipt,torch.tensor([[-.1,.2]]))
    assert after[0] > prior[0] and after[1] < prior[1]
    assert_same(before,m.state_dict())
    assert m._read_participation==[] and not m._read_credit_active


def test_receipt_batch_restores_ephemeral_state_on_failure():
    m=ExperienceMemory(3,capacity=2,topk=2)
    original=m._read_participation
    learner=SimpleNamespace(memories={1:SimpleNamespace(memory=m)})
    with pytest.raises(RuntimeError,match='probe'):
        with receipt_batch(learner):
            assert m._read_credit_active
            raise RuntimeError('probe')
    assert m._read_participation is original
    assert not m._read_credit_active and m._read_credit_entries==0


def test_lineages_track_all_identities_and_do_not_merge_replaced_slots():
    def row(step,identity):
        return dict(step=step,admissions=[dict(reads=[dict(identity=identity,
            age_tokens=512, signed_read_parameter_credit=.1)])])
    result=lineages(dict(arms=dict(null=dict(steps=[row(1,'1:0:2'),
        row(4,'1:0:2'),row(8,'1:0:3')]))))['null']
    assert result['positive_old_reads']==3 and result['distinct_identities']==2
    assert result['repeated_positive_reads']==result['repeated_beyond_window']==1
    assert result['maximum_positive_reuse_gap_tokens']==384
