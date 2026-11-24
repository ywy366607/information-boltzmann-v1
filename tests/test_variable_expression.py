from types import SimpleNamespace

import pytest
import torch

from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.train_gdn_fast_table import TableGDN
from scripts.ib_local.latent_navigation import LatentNavigationBody
from scripts.ib_local.variable_expression import configure_variable_expression,film,proposal_score
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.navigation_retrieval import ExactUtteranceTable


def test_recorded_read_route_keeps_identity_but_recomputes_weights_and_gradients():
    from scripts.ib_local.latent_navigation_v3 import ExperienceMemory
    from scripts.ib_local.variable_expression import read_route_scope
    memory=ExperienceMemory(4,capacity=4,short=1,long=2,topk=2)
    encoder=torch.nn.Linear(4,2,bias=False)
    with torch.no_grad():
        encoder.weight.copy_(torch.tensor([[1.,0.,.2,.1],[0.,1.,.3,-.2]]))
        memory.occupied[:3]=True
        memory.source[:3]=torch.tensor([[1.,0.,0.,0.],[.5,.5,0.,0.],[0.,1.,0.,0.]])
        memory.action[:3]=torch.tensor([[1.,0.,0.,0.],[0.,0.,0.,1.],[0.,0.,1.,0.]])
    query=torch.tensor([[1.,.1,0.,0.]],requires_grad=True)
    engine=SimpleNamespace(model=SimpleNamespace(memory=memory))
    receipt=dict(read_slots=torch.tensor([[2,1]]),read_versions=torch.zeros(1,2,dtype=torch.long))
    assert memory.read(query,encoder).slots.tolist()==[[0,1]]
    with read_route_scope(engine,receipt):
        actual=memory.read(query,encoder)
        assert actual.slots.tolist()==[[2,1]]
        later=memory.read(query+.2,encoder)
        assert not torch.equal(actual.weights,later.weights)
        actual.value.square().sum().backward()
    assert query.grad.norm()>0 and encoder.weight.grad.norm()>0
    assert not hasattr(memory,'_replay_read_slots')
    assert memory.read(query,encoder).slots.tolist()==[[0,1]]
    memory.read_credit_version[2]+=1
    with pytest.raises(RuntimeError,match='replaced experience'):
        with read_route_scope(engine,receipt):pass


def test_variable_draft_records_every_actual_read_including_final_self_hearing():
    from scripts.ib_local.latent_navigation_v3 import LatentNavigationV3
    torch.manual_seed(19)
    base=GatedDeltaNetLM(256,32,1,1,'decay_then_delta',64,4)
    model=LatentNavigationV3(TableGDN(base,True,inject_layer=0),latent_dim=8,
        capacity=8,short=2,long=16,topk=2,max_chunk=8)
    model.retrieval=ExactUtteranceTable(8)
    optimizer=torch.optim.AdamW(model.parameters())
    engine=SimpleNamespace(model=model,optimizer=optimizer,params=list(model.parameters()),
        groups=(list(model.parameters()),),prev=None,device=torch.device('cpu'),
        pending=97,recurrent=model.initial_state(1),_draft_actions=[])
    configure_variable_expression(engine)
    with torch.no_grad():model.handover.bias.fill_(20.)
    table=RetrievedCreditTable(orders=(0,),heads=1,slots_log2=1)
    words,logps,probs,_=film(engine,None,table)
    receipt=engine._last_expression_receipt
    assert receipt['read_slots'].shape==(len(words)+1,2)
    assert receipt['read_versions'].shape==receipt['read_slots'].shape
    terms,error=proposal_score(engine,dict(words=words,logp=logps,probs=probs,handover=receipt))
    assert error<1e-5
    terms.sum().backward()
    assert not hasattr(model.memory,'_replay_read_slots')


def test_rejected_batch_graph_is_freed_before_exact_replay_and_score_gradient_matches():
    import weakref
    torch.manual_seed(19)
    base=GatedDeltaNetLM(256,32,1,1,'decay_then_delta',64,4)
    model=LatentNavigationBody(TableGDN(base,True,inject_layer=0),latent_dim=8,
                              capacity=8,horizon=2,max_chunk=128)
    model.retrieval=ExactUtteranceTable(8)
    optimizer=torch.optim.AdamW(model.parameters())
    e=SimpleNamespace(model=model,optimizer=optimizer,params=list(model.parameters()),
        groups=(list(model.parameters()),),prev=None,device=torch.device('cpu'),
        pending=97,recurrent=model.initial_state(1),_draft_actions=[])
    configure_variable_expression(e)
    with torch.no_grad():model.handover.bias.fill_(20.)
    table=RetrievedCreditTable(orders=(0,),heads=1,slots_log2=1)
    words,logps,probs,_=film(e,None,table)
    proposal=dict(words=words,logp=logps,probs=probs,handover=e._last_expression_receipt)
    expected,_=proposal_score(e,proposal)
    expected=expected.detach()
    original=model.predict_observed
    anchors=[]; fallback=[]; rejected=[]; mode={'retain':True}
    class Retain(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x,anchor):
            ctx.save_for_backward(anchor)
            return x
        @staticmethod
        def backward(ctx,g):return g,None
    def mismatch(ids,state,lp,**kw):
        if ids.shape[1]==1:
            assert anchors and ((anchors[0]() is not None) if mode['retain'] else (anchors[0]() is None))
            fallback.append(True)
            return original(ids,state,lp,**kw)
        out=original(ids,state,lp,**kw)
        anchor=torch.zeros(10000)
        anchors.append(weakref.ref(anchor))
        # Fault injection only into the batch diagnostic. The exact actual
        # one-byte path, weights, and targets remain unchanged.
        changed=out[0].clone()
        changed[...,0]+=10.
        result=(Retain.apply(changed,anchor),*out[1:])
        if mode['retain']:rejected.append(result)  # reproduce the old graph lifetime
        return result
    model.predict_observed=mismatch
    try:
        baseline,_=proposal_score(e,proposal)
        reference=torch.autograd.grad(baseline.sum(),e.params,allow_unused=True)
        rejected.clear();anchors.clear();mode['retain']=False
        actual,error=proposal_score(e,proposal)
        gradient=torch.autograd.grad(actual.sum(),e.params,allow_unused=True)
    finally:del model.predict_observed
    assert fallback and error<1e-5
    torch.testing.assert_close(expected,actual)
    for a,b in zip(reference,gradient):
        assert (a is None)==(b is None)
        if a is not None:assert torch.equal(a,b)


def test_allocator_release_only_follows_new_cuda_fallback(monkeypatch):
    from scripts.ib_local.variable_expression import release_fallback_cache
    calls=[]
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda:100)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:calls.append(True))
    e=SimpleNamespace(device=torch.device('cuda'),_batched_replay_fallbacks=2)
    assert release_fallback_cache(e,2) is None and calls==[]
    assert release_fallback_cache(e,1)['reserved_before_mib']==100/2**20
    assert calls==[True]
    e.device=torch.device('cpu')
    assert release_fallback_cache(e,1) is None and calls==[True]


def test_handover_is_replayed_and_credited_with_the_actual_byte_prefix():
    torch.manual_seed(19)
    base=GatedDeltaNetLM(256,32,1,1,'decay_then_delta',64,4)
    model=LatentNavigationBody(TableGDN(base,True,inject_layer=0),latent_dim=8,
                              capacity=8,horizon=2,max_chunk=128)
    model.retrieval=ExactUtteranceTable(8)
    optimizer=torch.optim.AdamW(model.parameters())
    e=SimpleNamespace(model=model,optimizer=optimizer,params=list(model.parameters()),
        groups=(list(model.parameters()),),prev=None,device=torch.device('cpu'),
        pending=97,recurrent=model.initial_state(1),_draft_actions=[])
    configure_variable_expression(e)
    with torch.no_grad():model.handover.bias.fill_(20.)
    table=RetrievedCreditTable(orders=(0,),heads=1,slots_log2=1)
    words,logps,probs,_=film(e,None,table)
    assert len(words)==1
    receipt=e._last_expression_receipt
    assert receipt['reason']=='handover' and receipt['stops'][0]['ended']
    terms,error=proposal_score(e,dict(words=words,logp=logps,probs=probs,handover=receipt))
    assert error<1e-5 and len(terms)==2
    (-terms.sum()).backward()
    assert model.handover.bias.grad is not None
    assert model.expression_budget_to_hidden.weight.grad is not None
    assert model.retrieval.words.shape==(8,128)


@pytest.mark.parametrize('own', [32, 33, 128])
def test_full_self_heard_expression_and_480_byte_observation_fit_the_same_update(own):
    """Reproduce the production 512/513 boundary without a capability experiment."""
    from scripts.ib_local.navigation_internal_rollout import InternalRolloutBody
    base=GatedDeltaNetLM(256,32,1,1,'decay_then_delta',64,4)
    model=InternalRolloutBody(TableGDN(base,True,inject_layer=0),latent_dim=8,
                             capacity=640,short=2,long=512,topk=2,max_chunk=512)
    model.retrieval=ExactUtteranceTable(8)
    optimizer=torch.optim.AdamW(model.parameters())
    engine=SimpleNamespace(model=model,optimizer=optimizer,params=list(model.parameters()),
        groups=(list(model.parameters()),),prev=None,device=torch.device('cpu'))
    configure_variable_expression(engine)
    length=own+480
    ids=torch.arange(length)[None]%256
    logp=torch.full((1,length,256),-torch.log(torch.tensor(256.)).item())
    mask=torch.ones_like(ids,dtype=torch.float32);mask[:,:own]=0
    loss,state,features,diag=model.objective(ids,(ids+1)%256,model.initial_state(1),
        logp,torch.randn(8,4),score_mask=mask)
    loss.backward()
    model.commit_observed(features)
    assert torch.isfinite(loss) and torch.isfinite(state).all()
    assert model.encoder.weight.grad is not None
    assert int(model.memory.clock)==length
    assert model.memory.pending.shape==(512,32)
    assert model.memory.pending_count==min(length,512)
    assert torch.equal(model.memory.pending[-min(length,512):],features.detach()[-512:])
    assert diag['latent_long'].item()==0


def test_proposal_scores_batched_matches_serial_gradients():
    from scripts.ib_local.variable_expression import proposal_scores_batched
    torch.manual_seed(29)
    base=GatedDeltaNetLM(256,32,1,1,'decay_then_delta',64,4)
    model=LatentNavigationBody(TableGDN(base,True,inject_layer=0),latent_dim=8,
                               capacity=8,horizon=2,max_chunk=128)
    model.retrieval=ExactUtteranceTable(8)
    optimizer=torch.optim.AdamW(model.parameters())
    e=SimpleNamespace(model=model,optimizer=optimizer,params=list(model.parameters()),
        groups=(list(model.parameters()),),prev=None,device=torch.device('cpu'),
        pending=97,recurrent=model.initial_state(1),_draft_actions=[])
    configure_variable_expression(e)
    with torch.no_grad():
        model.handover.bias.fill_(-10.)
    e.max_utterance=8
    proposals=[]
    torch.manual_seed(7)
    with torch.no_grad():
        model.handover.bias.fill_(-0.3)
    for _ in range(4):
        table=RetrievedCreditTable(orders=(0,),heads=1,slots_log2=1)
        words,logps,probs,_=film(e,None,table)
        proposals.append(dict(words=words,logp=logps,probs=probs,handover=e._last_expression_receipt))
    assert len({len(p['words']) for p in proposals}) > 1
    ref_grads=[None]*len(e.params)
    ref_terms=[]
    for p in proposals:
        t,err=proposal_score(e,p)
        assert err<1e-5
        ref_terms.append(t.detach())
        for i,g in enumerate(torch.autograd.grad(t.sum(),e.params,allow_unused=True)):
            if g is not None:
                ref_grads[i]=g.detach() if ref_grads[i] is None else ref_grads[i]+g.detach()
    valid_terms,fallbacks,max_err=proposal_scores_batched(e,proposals)
    assert not fallbacks and max_err<1e-5 and len(valid_terms)==4
    for k in range(4):
        torch.testing.assert_close(valid_terms[k].detach(),ref_terms[k],atol=1e-5,rtol=1e-5)
    batched_sum=torch.stack([t.sum() for t in valid_terms.values()]).sum()
    batched_grads=torch.autograd.grad(batched_sum,e.params,allow_unused=True)
    for a,b in zip(ref_grads,batched_grads):
        assert (a is None)==(b is None)
        if a is not None:
            torch.testing.assert_close(b,a,atol=1e-5,rtol=1e-5)
