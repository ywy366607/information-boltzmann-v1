"""Numerical growth invariants; no learning or capability claims."""
from types import SimpleNamespace

import torch

from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.latent_navigation import LatentNavigationBody
from scripts.ib_local.navigation_body_growth import configure_body_growth, parameter_lift
from scripts.ib_local.train_gdn_fast_table import TableGDN
from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
from scripts.ib_local.bfss_gdn_capture import quiet_packed


def test_growth_preserves_logits_features_momentum_and_delayed_participation():
    torch.manual_seed(44)
    base=GatedDeltaNetLM(256,128,4,4,'decay_then_delta',256,4,32.,decay_init='log_timescale')
    model=LatentNavigationBody(TableGDN(base,True),capacity=16,horizon=2,max_chunk=24)
    model.momentum_to_boundary=torch.nn.Linear(512,64,bias=False)
    params=list(model.parameters())
    optimizer=torch.optim.AdamW(params,lr=3e-4)
    ids=torch.randint(0,256,(1,8));logp=torch.randn(1,8,256).log_softmax(-1)
    state=model.initial_state(1)
    with torch.no_grad():
        state=model.predict(ids,state,logp)[1]
    p=[torch.randn(1,4,32,32)*.01 for _ in range(4)]
    e=SimpleNamespace(model=model,optimizer=optimizer,params=params,groups=(params,),
        recurrent=state,bfss_momenta=p,bfss_coupling=.01,
        prev={'actor':tuple(torch.ones_like(t) for t in params)})
    with torch.no_grad():
        old=model.predict_observed(ids,state,logp)
        old_p=model.momentum_to_boundary(decision_momentum_read(model,p,old[3][:1]))
        quiet_old,po=quiet_packed(base,state,p,dt=.01,frequency=.8,coupling=.01,steps=1)
    spec=dict(schema=1,backbone=dict(vocab_size=256,d=256,layers=32,heads=8,
        recurrence='decay_then_delta',mlp_hidden=768,conv_kernel=4,content_budget=32.,decay_init='log_timescale'))
    receipt=configure_body_growth(e,spec)
    with torch.no_grad():
        new=model.predict_observed(ids,e.recurrent,logp)
        new_p=model.momentum_to_boundary(decision_momentum_read(model,e.bfss_momenta,new[3][:1]))
        quiet_new,pn=quiet_packed(model.base,e.recurrent,e.bfss_momenta,
                                  dt=.01,frequency=.8,coupling=e.bfss_coupling,steps=1)
    torch.testing.assert_close(new[0],old[0],atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(new[3],old[3],atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(new_p,old_p,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(pn[0][:,:4],po[0],atol=2e-6,rtol=2e-5)
    assert model.memory.width==128 and len(e.prev['actor'])==len(e.params)
    assert all(g is None or g.shape==p.shape for p,g in zip(e.params,e.prev['actor']))
    assert receipt['all_parameters_trainable']
    assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in model.parameters()}
    # Added residual projections receive gradients immediately at identity birth.
    model.zero_grad(set_to_none=True)
    model.predict(ids,e.recurrent,logp)[0].square().mean().backward()
    assert model.base.layers[-1].o_proj.weight.grad.norm()>0
    assert model.feature_read.weight.grad is not None


def test_gradient_transport_preserves_restricted_directional_derivative():
    for name,shape in [('body.base.layers.0.q_proj.weight',(256,256)),
                       ('body.base.source.embedding.weight',(256,256)),
                       ('body.base.final_norm.weight',(256,)),
                       ('body.base.layers.0.mlp.down.weight',(256,768))]:
        old_shape={'body.base.layers.0.q_proj.weight':(128,128),
                   'body.base.source.embedding.weight':(256,128),
                   'body.base.final_norm.weight':(128,),
                   'body.base.layers.0.mlp.down.weight':(128,256)}[name]
        tangent=torch.randn(old_shape);gradient=torch.randn(old_shape)
        a=parameter_lift(name,tangent,shape)
        b=parameter_lift(name,gradient,shape,covector=True)
        torch.testing.assert_close((a*b).sum(),(tangent*gradient).sum(),atol=1e-3,rtol=2e-5)
