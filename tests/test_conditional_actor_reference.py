"""Numeric binding to the unchanged live actor; these are interface tests."""
import torch

from scripts.ib_local.conditional_actor_reference import (
    encoded_view,evaluate,candidate_policy_decomposition)
from test_relation_retention_use import configured


def test_encoded_actor_matches_original_habit_offset_and_gradients(tmp_path,monkeypatch):
    e,a=configured(tmp_path)
    try:
        e.step(torch.randn(8,4))
        h=e._internal_source.reshape(1,-1)
        monkeypatch.setattr(e,'precision',lambda:1.7)
        with torch.no_grad():
            habit,packet,reusable=e.plan(h)
            view=encoded_view(e.model,e.model.encoder(h),e.intero(),
                              reward_valid=e.reward_label_valid())
            torch.testing.assert_close(view.habit,habit,rtol=0,atol=0)
            torch.testing.assert_close(view.reuse_weight,
                (packet['weights']*reusable).sum(-1,keepdim=True),rtol=0,atol=0)
            q=h.new_tensor([[.04,-.02,.12]])
            original=e._choice_offset(10*q[0],habit[0]>-1e3)
            result=evaluate(view,q,precision=e.precision())
            torch.testing.assert_close(result['offset'][0],original,rtol=0,atol=0)
            torch.testing.assert_close(result['probability'],(habit+original).softmax(-1),rtol=0,atol=0)
        # Reusing the same view expression preserves the real choice derivative.
        view=encoded_view(e.model,e.model.encoder(h),e.intero(),reward_valid=e.reward_label_valid())
        loss=evaluate(view,q,precision=1.7)['probability'][0,0].log()
        grad=torch.autograd.grad(loss,e.model.kanli.choice.weight)[0]
        assert bool(torch.isfinite(grad).all()) and float(grad.norm())>0
    finally:a.close()


def test_real_option_values_decompose_candidate_and_policy_terms():
    q=torch.tensor([.02,.07,-.01],dtype=torch.float64)
    p=torch.tensor([.3,.6,.1],dtype=torch.float64)
    w=torch.tensor([.1,.2,.7],dtype=torch.float64)
    options=torch.tensor([0,1,1])
    gains=torch.tensor([.03,.09,.04],dtype=torch.float64)
    d=candidate_policy_decomposition(w,gains,options,p,q)
    assert d['reconstruction_error']<1e-15
    # Exact finite action expectation is the least-square predictor of q(A).
    v=d['actual_policy_value']
    for prediction in (v,v+.2,v-.1):
        risk=(p*(prediction-q).square()).sum()
        torch.testing.assert_close(risk,(prediction-v).square()+d['action_sampling_variance'])


def test_encoded_use_keeps_actual_momentum_conditioned_response(tmp_path):
    from scripts.ib_local import epistemic_value_reference as ev
    from scripts.ib_local.causal_relation_program import mechanism
    from scripts.ib_local.causal_use_value import queries
    from scripts.ib_local.finite_use_value import expected
    e,a=configured(tmp_path)
    try:
        for _ in range(2):e.step(torch.randn(8,4))
        state=e._program_question_state
        source=e._internal_source.reshape(1,-1).expand_as(e._internal_action)
        with torch.no_grad():
            d=mechanism(e.model,state,source,e._internal_action,ridge=e.relation_ridge,
                momentum=e._momentum_decision_read[None].expand(len(source),-1))
            ctx=ev.from_body(e,state,queries(e,state))
            raw=expected(e,state,source,e._internal_action,d,queries(e,state))
            encoded=ev.expected(e.model,ctx,e.model.encoder(source),e.model.encoder(e._internal_action),
                                ridge=e.relation_ridge,distribution=d)
            torch.testing.assert_close(raw,encoded,rtol=2e-5,atol=3e-6)
            carried=ev.from_body(e,state,queries(e,state),momentum=e._momentum_decision_read[None])
            reference=ev.response(e.model,carried,e.model.encoder(source),e.model.encoder(e._internal_action),
                ridge=e.relation_ridge,query_shift=carried.prediction_shift)
            torch.testing.assert_close(reference['mean'],d['mean'],rtol=2e-5,atol=3e-6)
            torch.testing.assert_close(ev.scores(e.model,carried,ridge=e.relation_ridge),
                                       ev.scores(e.model,ctx,ridge=e.relation_ridge),rtol=0,atol=0)
            torch.testing.assert_close(ev.expected(e.model,carried,e.model.encoder(source),
                e.model.encoder(e._internal_action),ridge=e.relation_ridge),raw,rtol=2e-5,atol=3e-6)
    finally:a.close()


def test_complete_momentum_context_preserves_same_real_settlement(tmp_path):
    from scripts.ib_local import epistemic_value_reference as ev
    e,a=configured(tmp_path)
    try:
        e.step(torch.randn(8,4))
        params=sum(p.numel() for p in e.model.parameters())
        ev.configure(e,schema=2)
        row=e.step(torch.randn(8,4))
        assert row['causal_use']['epistemic_value']['schema']==2
        assert row['causal_use']['reward_version']=='finite_relation_use_v5'
        assert row['causal_use']['finite_U']['verified']
        assert torch.isfinite(e._program_forecasts['epistemic_after_value']).all()
        assert 'prediction_momentum' in e._program_forecasts['path']
        assert sum(p.numel() for p in e.model.parameters())==params
    finally:a.close()
