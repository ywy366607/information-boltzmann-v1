import torch

from scripts.ib_local.bfss_gdn_reference import coupling_potential_and_gradient,quiet_step as reference
from scripts.ib_local.learned_head_coupling import HeadPotential,quiet_step


def test_weighted_attention_contains_original_force_and_step():
    torch.manual_seed(17)
    s,p=torch.randn(1,4,4,4),torch.randn(1,4,4,4)
    u=HeadPotential(4,4,4,'weighted',.01)
    energy,gradient=coupling_potential_and_gradient(s,.01)
    torch.testing.assert_close(u(s),energy)
    torch.testing.assert_close(u.force(s),gradient,atol=1e-7,rtol=2e-5)
    actual=quiet_step(u,s,p,dt=.01,frequency=.8)
    expected=reference(s,p,dt=.01,frequency=.8,coupling=.01)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b)


def test_attention_weight_derivative_is_part_of_force():
    torch.manual_seed(18)
    u=HeadPotential(4,4,4,'weighted',.01).double()
    torch.nn.init.normal_(u.key.weight,std=.2)
    s=torch.randn(1,4,4,4,dtype=torch.float64,requires_grad=True)
    force=u.force(s,create_graph=True)
    direction=torch.randn_like(s);epsilon=1e-5
    fd=(u(s+epsilon*direction)-u(s-epsilon*direction))/(2*epsilon)
    torch.testing.assert_close((force*direction).sum(),fd.sum(),atol=1e-7,rtol=2e-5)


def test_both_potentials_receive_language_side_state_gradient_and_reverse():
    for kind in ('weighted','free'):
        torch.manual_seed(19)
        u=HeadPotential(4,4,4,kind,.01)
        s,p=torch.randn(1,4,4,4),torch.randn(1,4,4,4)
        u.calibrate(s)
        state,momentum=quiet_step(u,s,p,dt=.01,frequency=.8,create_graph=True)
        loss=(state-torch.randn_like(state)).square().mean()
        loss.backward()
        assert sum(float(a.grad.norm()) for a in u.parameters() if a.grad is not None)>0
        reverse=quiet_step(u,state.detach(),momentum.detach(),dt=-.01,frequency=.8)
        torch.testing.assert_close(reverse[0],s,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(reverse[1],p,atol=2e-6,rtol=2e-5)
        assert bool((u(s)>=0).all())
