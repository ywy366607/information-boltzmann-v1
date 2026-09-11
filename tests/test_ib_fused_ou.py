import pytest
import torch

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA kernel contract')
@pytest.mark.parametrize('temperature',[0.,.1])
@pytest.mark.parametrize('gamma',[.01,.3,2.])
def test_fused_ou_nonzero_gradient_contract(gamma,temperature):
    pytest.importorskip('triton')
    from scripts.ib_fused_ou import FusedOU
    torch.manual_seed(17)
    v=torch.randn(32,4,device='cuda',requires_grad=True)
    c=torch.randn_like(v,requires_grad=True)
    g=torch.full((32,1),gamma,device='cuda',requires_grad=True)
    z=torch.randn_like(v);h=.0625
    ref=torch.exp(-g*h)*v-torch.expm1(-g*h)/g*c+torch.sqrt(-temperature*torch.expm1(-2*g*h))*z
    if temperature==0:ref=torch.exp(-g*h)*v-torch.expm1(-g*h)/g*c
    actual=FusedOU.apply(v,c,g,z,h,temperature)
    torch.testing.assert_close(actual,ref,atol=2e-6,rtol=2e-6)
    left=torch.autograd.grad(ref.square().sum(),(v,c,g),retain_graph=True)
    right=torch.autograd.grad(actual.square().sum(),(v,c,g))
    for a,b in zip(left,right):torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-4)
