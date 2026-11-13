"""Numerical intervention integrity, not a synthetic learning benchmark."""
import torch

from scripts.ib_local.audit_eligibility_routing import (
    RouteTeaching, old_permutation, eligibility_routing, receipt_summary,
)
from scripts.ib_local.pretrained_gdn_experience import LayerExperienceRecall, LocalExperience


def test_teaching_exchange_preserves_values_and_total_norm():
    values=torch.tensor([[1.,2.],[3.,4.],[5.,6.]],requires_grad=True)
    permutation=old_permutation([300,64,700],128,exchange=True)
    output=RouteTeaching.apply(values,permutation)
    assert torch.equal(values,output)
    teaching=torch.tensor([[.1,.2],[.3,.4],[.5,.6]])
    (output*teaching).sum().backward()
    assert torch.equal(values.grad,teaching[permutation.argsort()])
    assert torch.equal(values.grad[1],teaching[1])
    assert torch.allclose(values.grad.norm(),teaching.norm(),rtol=1e-7,atol=1e-9)


def test_one_old_history_cannot_be_exchanged_with_recent_history():
    assert torch.equal(old_permutation([300,64],128,exchange=True),torch.tensor([0,1]))


def memory_and_native(capacity=2):
    torch.manual_seed(22)
    wrapper=LayerExperienceRecall(6,capacity=capacity,latent_dim=3,topk=2)
    wrapper.memory.clock.fill_(512)
    for i in range(capacity):
        x=torch.randn(6)
        wrapper.memory.commit_episode(x,x+1,x+2,x+3,wrapper.encoder,source_time=i)
        wrapper.capsules[i]=LocalExperience(torch.randn(1,65,6),None,
            int(wrapper.memory.read_credit_version[i]))
    wrapper.memory.clock.fill_(512)
    class Learner:
        memories={1:wrapper}
    return Learner(),wrapper,torch.nn.Linear(6,6)


def test_exchange_changes_native_credit_but_preserves_encoder_learning():
    learner,wrapper,native=memory_and_native()
    query=torch.randn(1,6)
    initial_retrieve=wrapper.retrieve
    rows=[]
    for exchanged in (False,True):
        native.zero_grad();wrapper.zero_grad()
        wrapper.begin(dict(lr=3e-4,betas=(.9,.999),eps=1e-8))
        with eligibility_routing(learner,exchange=exchanged) as receipts:
            value=wrapper.retrieve(query,lambda c:native(c.inputs))
            value.square().sum().backward()
        rows.append((value.detach().clone(),wrapper.encoder.weight.grad.clone(),native.weight.grad.clone()))
        summary=receipt_summary(receipts)[0]
        assert summary['correspondence_exact']
        assert abs(summary['sent_norm']-summary['received_norm'])<1e-12
        assert wrapper.retrieve == initial_retrieve
    assert torch.equal(rows[0][0],rows[1][0])
    assert torch.equal(rows[0][1],rows[1][1])
    assert not torch.equal(rows[0][2],rows[1][2])


def test_diagnostic_patch_restores_original_method_after_exception():
    learner,wrapper,_=memory_and_native()
    original=wrapper.retrieve
    try:
        with eligibility_routing(learner,exchange=True):
            raise RuntimeError('intentional interface interruption')
    except RuntimeError:
        pass
    assert wrapper.retrieve == original


def test_other_participation_changes_credit_without_changing_read_or_encoder():
    learner,wrapper,native=memory_and_native(capacity=4)
    query=torch.randn(1,6)
    results=[]
    for foreign in (False,True):
        native.zero_grad();wrapper.zero_grad()
        wrapper.begin(dict(lr=3e-4,betas=(.9,.999),eps=1e-8))
        with eligibility_routing(learner,other_participation=foreign) as receipts:
            value=wrapper.retrieve(query,lambda c:native(c.inputs))
            value.square().sum().backward()
        r=receipt_summary(receipts)[0]
        assert r['correspondence_exact']
        assert abs(r['sent_norm']-r['received_norm'])<1e-12
        if foreign:
            assert set(r['receiver_slots']).isdisjoint(r['slots'])
            assert r['wrong_receiver_count']==2
        results.append((value.detach().clone(),wrapper.encoder.weight.grad.clone(),native.weight.grad.clone()))
    assert torch.equal(results[0][0],results[1][0])
    assert torch.equal(results[0][1],results[1][1])
    assert not torch.equal(results[0][2],results[1][2])
