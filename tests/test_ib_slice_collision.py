import torch
from fine_grain.information_boltzmann.state import PhaseState
from scripts.ib_slice_collision import SliceCayleyCollision

def test_slice_cayley_invariants_and_permutation():
    torch.manual_seed(9);torch.set_num_threads(1)
    m=SliceCayleyCollision(4,8,16,4.).double()
    s=PhaseState(torch.randn(32,4,dtype=torch.double),torch.randn(32,4,dtype=torch.double))
    out,lp,_=m(s,.25)
    assert out.x is s.x and out.time==s.time and lp==0
    torch.testing.assert_close(out.v.sum(0),s.v.sum(0),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(out.v.square().sum(),s.v.square().sum(),atol=1e-12,rtol=1e-12)
    assert (out.v-s.v).norm()>1e-8
    assert (out.v.T@out.v-s.v.T@s.v).norm()>1e-8
    perm=torch.randperm(32)
    other,_,_=m(PhaseState(s.x[perm],s.v[perm]),.25)
    torch.testing.assert_close(other.v,out.v[perm],atol=1e-12,rtol=1e-12)
    constant=PhaseState(s.x,torch.ones_like(s.v))
    same,_,_=m(constant,.25)
    torch.testing.assert_close(same.v,constant.v,atol=1e-12,rtol=1e-12)


def test_slice_cayley_gradients_and_zero_duration():
    torch.manual_seed(19)
    m=SliceCayleyCollision(2,4,8,2.).double()
    x=torch.randn(6,2,dtype=torch.double,requires_grad=True)
    v=torch.randn(6,2,dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda a,b:m(PhaseState(a,b),.5)[0].v,(x,v),fast_mode=True)
    out,_,_=m(PhaseState(x,v),.5)
    (out.v*torch.arange(6,dtype=torch.double)[:,None]).sum().backward()
    for p in m.parameters():assert p.grad is not None and torch.isfinite(p.grad).all()
    assert m.assignment.weight.grad.norm()>0
    same,_,_=m(PhaseState(x,v),0)
    assert same.v is v


def test_slice_cayley_fp32_multistep_energy_and_small_channels():
    import pytest
    for channels in [1,2]:
        with pytest.raises(ValueError):SliceCayleyCollision(4,channels)
    torch.manual_seed(12);torch.set_num_threads(1)
    m=SliceCayleyCollision(4,16,32,rate=20.)
    state=PhaseState(torch.randn(128,4),torch.randn(128,4))
    start=state.v.clone()
    with torch.no_grad():
        for _ in range(100):state,_,_=m(state,1.)
    torch.testing.assert_close(state.v.sum(0),start.sum(0),rtol=1e-5,atol=2e-5)
    torch.testing.assert_close(state.v.square().sum(),start.square().sum(),rtol=2e-5,atol=1e-4)
    assert (state.v-start).norm()>1e-3
