"""Operator and causal-port tests, not a memory capability benchmark."""
import torch
import pytest
from types import SimpleNamespace
from scripts.ib_local.native_contextual_recall import conditional_head_read,native_successor_residual
from scripts.ib_local.pretrained_gdn_experience import LayerExperienceRecall,LocalExperience


def test_native_read_is_conditioned_on_current_query_and_preserves_value_width():
    query=torch.tensor([[[1.,0.]],[[0.,1.]]],requires_grad=True)
    keys=torch.tensor([[[1.,0.]],[[0.,1.]]],requires_grad=True)
    values=torch.tensor([[[1.,2.,3.]],[[4.,5.,6.]]],requires_grad=True)
    result=conditional_head_read(query,keys,values,torch.ones(2,2),.2)
    assert result.shape==(2,1,3)
    assert result[0,0,0]<result[1,0,0]
    result.square().sum().backward()
    assert query.grad.norm()>0 and keys.grad.norm()>0 and values.grad.norm()>0


def test_null_prior_has_exact_zero_read_and_no_value_gradient():
    q=torch.randn(3,2,4,requires_grad=True)
    k=torch.randn(7,2,4,requires_grad=True)
    v=torch.randn(7,2,6,requires_grad=True)
    out=conditional_head_read(q,k,v,torch.zeros(3,7),.2)
    assert torch.equal(out,torch.zeros_like(out))
    out.sum().backward()
    assert torch.equal(v.grad,torch.zeros_like(v))


def test_current_decoder_keeps_learning_when_old_native_response_is_held():
    torch.manual_seed(5)
    old=torch.nn.Linear(4,4)
    decoder=torch.nn.Linear(4,3)
    x=torch.randn(6,4)
    response=old(x)
    keys=torch.randn(6,1,4)
    q=torch.randn(2,1,4)
    full=conditional_head_read(q,keys,decoder(response)[:,None],torch.ones(2,6),.2)
    held=conditional_head_read(q,keys,decoder(response.detach())[:,None],torch.ones(2,6),.2)
    torch.testing.assert_close(full,held,atol=0,rtol=0)
    native,readout=torch.autograd.grad(held.square().sum(),(old.weight,decoder.weight),allow_unused=True)
    assert native is None and readout.norm()>0


def test_whole_native_read_port_flattens_heads_and_teaches_the_old_program():
    class Filter(torch.nn.Module):
        def forward(self,x,cache=None,**kwargs):
            raw=x.transpose(1,2)
            cache=raw.new_zeros(1,raw.shape[1],3) if cache is None else cache
            extended=torch.cat((cache[...,-2:],raw),dim=-1)
            weight=raw.new_ones(raw.shape[1],1,3)/3
            out=torch.nn.functional.conv1d(extended,weight,groups=raw.shape[1]).transpose(1,2)
            return torch.nn.functional.silu(out),torch.cat((cache,raw),dim=-1)[...,-3:].clone()
    class GateNorm(torch.nn.Module):
        def forward(self,x,g):
            return torch.nn.functional.normalize(x,dim=-1)*g.sigmoid()
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads=self.num_v_heads=2
            self.head_k_dim=self.head_v_dim=2
            for name in ('q_proj','k_proj','v_proj','g_proj','o_proj'):
                setattr(self,name,torch.nn.Linear(4,4,bias=False))
            self.q_conv1d=self.k_conv1d=Filter()
            self.o_norm=GateNorm()
    torch.manual_seed(19)
    wrapper=LayerExperienceRecall(4,capacity=2,latent_dim=2)
    wrapper.memory.occupied[0]=True
    wrapper.capsules[0]=LocalExperience(torch.randn(1,65,4),None,0,torch.randn(1,63,4))
    block=SimpleNamespace(attn=Attention(),attn_norm=torch.nn.Identity())
    old=torch.nn.Linear(4,4)
    query=torch.randn(3,4,requires_grad=True)
    out=native_successor_residual(wrapper,block,query,lambda c:old(c.inputs).tanh(),forced_slot=0)
    assert out.shape==query.shape
    out.square().sum().backward()
    assert old.weight.grad.norm()>0 and query.grad.norm()>0
    wrapper.detach_replay=True
    held=native_successor_residual(wrapper,block,query,lambda c:old(c.inputs).tanh(),forced_slot=0)
    native,current=torch.autograd.grad(held.square().sum(),(old.weight,block.attn.v_proj.weight),allow_unused=True)
    assert native is None and current.norm()>0
    with torch.no_grad():
        incoming={}
        batch=native_successor_residual(wrapper,block,query,lambda c:old(c.inputs).tanh(),forced_slot=0,query_boundary=incoming)
        sequential_state={}
        sequential=torch.cat([native_successor_residual(wrapper,block,x[None],
            lambda c:old(c.inputs).tanh(),forced_slot=0,query_boundary=sequential_state) for x in query])
    torch.testing.assert_close(batch,sequential,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(incoming['recall_query_conv_state'],sequential_state['recall_query_conv_state'],atol=1e-6,rtol=1e-5)
    wrapper.detach_replay_older_than=128
    with pytest.raises(RuntimeError,match='not admitted'):
        native_successor_residual(wrapper,block,query,lambda c:old(c.inputs),forced_slot=0)


def test_sparse_token_fan_in_keeps_query_learning_and_bounds_participation():
    q=torch.tensor([[[1.,.4]]],requires_grad=True)
    k=torch.tensor([[[1.,0.]],[[.7,.7]],[[0.,1.]]],requires_grad=True)
    v=torch.tensor([[[1.]],[[3.]],[[8.]]],requires_grad=True)
    result,weights=conditional_head_read(q,k,v,torch.ones(1,3),.2,return_weights=True,topk=2)
    assert int((weights>0).sum())==2
    result.sum().backward()
    assert q.grad.norm()>0 and k.grad.norm()>0
