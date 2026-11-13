import copy

import numpy as np
import pytest
import torch

from test_native_s_learner import tiny_adapter
from scripts.ib_local.delayed_s_learner import DelayedFeedbackLearner, DelayedFeedbackStream, live_read_receipts, _FeedbackReadout


def test_unit_feedback_fraction_reuses_arrays_after_releasing_unused_workspace():
    from types import SimpleNamespace
    events=[]
    teaching={1:np.ones((1,1,2,2),np.float32),12:np.ones((1,1,2,2),np.float32)}
    def capture(actual):
        assert events==['released']
        assert actual is teaching
        return {k:v.sum() for k,v in actual.items()}
    proxy=_FeedbackReadout(SimpleNamespace(raw_credit=capture),1.,lambda:events.append('released'))
    assert proxy.raw_credit(teaching)=={1:4.,12:4.}
from scripts.ib_local.native_s_learner import NativeSEligibilityLearner


def learner():
    result = tiny_adapter()
    result.__class__ = DelayedFeedbackLearner
    result.eligibility.code.U[0,0] = 1
    result.eligibility.code.V[0] = 1
    result.feedback_mask = [False, True]
    return result


def test_delayed_mask_is_exact_ce_contribution_without_label_amplification():
    selected = learner()
    original = copy.deepcopy(selected)
    target = torch.tensor([[0,1]])
    x = torch.randn(1,2,2)
    with NativeSEligibilityLearner._teaching_context(original,target,torch.tensor([[False,True]])):
        original.model(x,past_key_values=original.cache)
    with selected._teaching_context(target,torch.tensor([[True,True]])):
        selected.model(x,past_key_values=selected.cache)
    for name,value in original._native_extra.items():
        np.testing.assert_allclose(selected._native_extra[name],value*.5,rtol=1e-6,atol=1e-7)
    assert selected._last_native_s['feedback_fraction'] == .5
    assert all(p.grad is None for p in selected.model.parameters())


def test_no_feedback_does_not_disable_ordinary_learning_or_invent_extra_credit():
    l = learner()
    l.feedback_mask = [False,False]
    with l._teaching_context(torch.tensor([[0,1]]),None):
        output=l.model(torch.randn(1,2,2),past_key_values=l.cache)
    assert l._native_extra is None
    torch.nn.functional.cross_entropy(output.logits.reshape(-1,2),torch.tensor([0,1])).backward()
    assert any(p.grad is not None for p in l.model.parameters())


def test_invalid_feedback_mask_is_rejected_before_teaching():
    l=learner()
    with pytest.raises(ValueError,match='actual external teaching'):
        with l._teaching_context(torch.tensor([[0,1]]),torch.tensor([[True,False]])):
            pass


class Backend:
    def __init__(self):
        from types import SimpleNamespace
        self.learner=SimpleNamespace(feedback_mask=None)
        self.device=torch.device('cpu')
        self.masks=[]
    def boundary(self):return {}
    def advance(self,tokens,**kw):return torch.zeros(len(tokens),4)
    def commit(self,inputs,targets,mask,boundary):
        self.masks.append((list(mask),list(self.learner.feedback_mask)))
        return dict(ce=1.3862943611198906)


def test_feedback_tags_follow_arriving_targets_across_boundary_and_restore():
    backend=Backend()
    stream=DelayedFeedbackStream(backend)
    stream.receive([0]*128,phase='intervening',teach=True)
    stream.receive([1,2],phase='feedback_answer',teach=True)
    assert backend.masks[0][1] == [False]*127+[True]
    saved=stream.state_dict()
    restored=DelayedFeedbackStream(Backend())
    restored.restore_latest(saved)
    assert restored.feedback_positions == [True]
    assert restored.events == stream.events


def test_read_receipt_uses_last_actual_query_and_current_slot_version():
    from types import SimpleNamespace
    wrapper=SimpleNamespace(last_read=SimpleNamespace(slots=torch.tensor([[0,1],[1,2]]),
        weights=torch.tensor([[.9,.1],[.3,.7]])),query_clock_offsets=torch.tensor([0,7]),
        memory=SimpleNamespace(clock=200,birth=torch.tensor([190,40,100]),read_credit_version=torch.tensor([3,4,5])))
    receipt=live_read_receipts({1:wrapper})
    assert [r['slot'] for r in receipt['2']]==[1,2]
    assert [r['version'] for r in receipt['2']]==[4,5]
    assert [r['age_tokens'] for r in receipt['2']]==[167,107]
