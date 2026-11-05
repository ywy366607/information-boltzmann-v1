"""Local score, gradient and probability contracts, not capability studies."""
import torch
from scripts.ib_local.language_navigation_kanli_v4t import (
    ProbabilityActionEvidence, TypedWoodEngine, match_probability,
)


def test_binary_probabilities_are_valid_and_differentiable():
    base=torch.tensor([0.,.2,.8,1.])
    correction=torch.tensor([-100.,-.4,.4,100.],requires_grad=True)
    p=match_probability(base,correction)
    distribution=torch.stack((1-p,p),-1)
    assert torch.isfinite(distribution).all() and (distribution>=0).all()
    torch.testing.assert_close(distribution.sum(-1),torch.ones(4))
    p[1:3].sum().backward()
    assert (correction.grad[1:3]>0).all()


def test_brier_expected_gap_matches_lean_statement_and_gradient():
    p=torch.tensor([.15,.5,.85],dtype=torch.float64,requires_grad=True)
    q=torch.tensor([.25,.7,.6],dtype=torch.float64)
    loss=q*(p-1).square()+(1-q)*p.square()
    optimum=q*(q-1).square()+(1-q)*q.square()
    torch.testing.assert_close(loss-optimum,(p-q).square())
    loss.sum().backward()
    torch.testing.assert_close(p.grad,2*(p.detach()-q))


def test_merged_binary_replay_preserves_shared_prediction_gradient():
    p=torch.tensor(.7,requires_grad=True)
    labels=torch.tensor([1.,0.,1.,1.])
    actual=(p-labels).square().mean()
    merged=(p-labels.mean()).square()
    ga=torch.autograd.grad(actual,p)[0]
    gm=torch.autograd.grad(merged,p)[0]
    torch.testing.assert_close(ga,gm)


def test_typed_composition_preserves_other_channels_and_uses_same_path():
    # Call the unbound methods with an instance that needs no body execution.
    engine=object.__new__(TypedWoodEngine)
    base=torch.tensor([[.3,.5,.9,-.2],[.7,.4,.1,.3]])
    residual=torch.tensor([[.1,.2,.6,.4],[-.2,.3,-.7,-.1]],requires_grad=True)
    weight=torch.tensor([[.2,.4,.6,.8],[.1,.3,.5,.7]])
    raw=engine._compose_prediction(base,residual)
    used=engine._compose_used_prediction(base,residual,weight)
    torch.testing.assert_close(raw[:,[0,1,3]],(base+residual)[:,[0,1,3]])
    torch.testing.assert_close(used[:,[0,1,3]],(base+weight*residual)[:,[0,1,3]])
    assert ((used[:,2]>=0)&(used[:,2]<=1)).all()
    torch.testing.assert_close(used[:,2],base[:,2]+weight[:,2]*(raw[:,2]-base[:,2]))
    (raw[:,2]-torch.tensor([1.,0.])).square().mean().backward()
    assert torch.isfinite(residual.grad).all() and (residual.grad[:,2].abs()>0).all()


def test_shared_probability_baseline_is_bounded():
    e=ProbabilityActionEvidence()
    e.blind[2]=.8;e.mean[:,2]=torch.tensor([-.9,.1,.8])
    p=e.bases()[:,2]
    torch.testing.assert_close(p,torch.tensor([0.,.9,1.]))
