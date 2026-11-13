"""Probability/derivative and bounded native-execution contracts, not capabilities."""
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from scripts.ib_local.predictive_responsibility import predictive_mixture
from scripts.ib_local.small_gdn_experience import configure_small_replay
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.train_gdn_fast_table import TableGDN
from scripts.ib_local.latent_navigation_v3 import LatentNavigationV3


def test_outcome_responsibility_and_complete_mixture_derivatives():
    scores=torch.tensor([[.2,-.4],[.1,.8]],dtype=torch.float64,requires_grad=True)
    logits=torch.tensor([[[3.,-1.],[-2.,2.]],[[1.,-1.],[-1.,1.]]],requires_grad=True)
    targets=torch.tensor([0,1])
    result=predictive_mixture(scores,logits)
    prior=result.log_prior.exp()
    posterior=result.posterior(targets)
    likelihood=logits.softmax(-1).gather(-1,targets[:,None,None].expand(-1,2,1)).squeeze(-1)
    expected=prior*likelihood
    torch.testing.assert_close(posterior,expected/expected.sum(-1,keepdim=True))
    torch.testing.assert_close(result.logp.exp().sum(-1),torch.ones(2))
    assert posterior[0,0]>prior[0,0] and posterior[1,1]>prior[1,1]
    gs,gl=torch.autograd.grad(F.nll_loss(result.logp,targets),(result.prior_scores,logits))
    torch.testing.assert_close(gs,(prior-posterior)/2)
    expected_expert=logits.softmax(-1)-F.one_hot(targets,2)[:,None]
    torch.testing.assert_close(gl,expected_expert*posterior[:,:,None]/2)


def test_identical_likelihood_preserves_prior_and_extreme_scores_are_finite():
    scores=torch.tensor([[1000.,-1000.],[0.,0.]])
    logits=torch.tensor([[[1000.,-1000.],[1000.,-1000.]],[[1.,2.],[1.,2.]]])
    result=predictive_mixture(scores,logits)
    torch.testing.assert_close(result.posterior(torch.tensor([1,0])),scores.softmax(-1))
    assert torch.isfinite(result.logp).all()


def test_shared_parameter_score_path_is_not_silently_detached():
    p=torch.tensor(.3,requires_grad=True)
    def objective(value):
        scores=torch.stack((value,2*value))[None]
        logits=torch.stack((torch.stack((value,-value)),torch.stack((-value,3*value))))[None]
        return -predictive_mixture(scores,logits).logp[0,1]
    derivative,=torch.autograd.grad(objective(p),p)
    step=1e-3
    finite=(objective(p.detach()+step)-objective(p.detach()-step))/(2*step)
    torch.testing.assert_close(derivative,finite,atol=1e-4,rtol=1e-3)


def setup(schema=3, output_projection=False):
    torch.manual_seed(27)
    base=GatedDeltaNetLM(256,8,4,2,'decay_then_delta',16,4,32.,decay_init='log_timescale')
    model=LatentNavigationV3(TableGDN(base,True),latent_dim=4,capacity=8,
        short=2,long=4,topk=2,max_chunk=4,history_size=8)
    params=list(model.parameters())
    engine=SimpleNamespace(model=model,device=torch.device('cpu'),params=params,
        optimizer=torch.optim.Adam(params,lr=3e-4),groups=(params,))
    bank=configure_small_replay(engine,dict(schema=schema,capacity=2,latent_dim=4,
        **({'native_output_projection':True} if output_projection else {})))
    ids=torch.tensor([[2,3,4,5]])
    logp=torch.randn(1,4,256).log_softmax(-1)
    state=model.initial_state(1)
    return model,bank,engine,ids,logp,state


def fill(model,bank,engine,ids,logp,state):
    with bank.observation(engine.optimizer.param_groups[-1],len(ids[0])):
        with torch.no_grad():
            model.predict(ids,state,logp)
            bank.settle_and_commit()


def test_empty_and_zero_inlet_preserve_full_body_and_global_ports():
    model,bank,engine,ids,logp,state=setup()
    bank.enabled=False
    native=model.predict(ids,state,logp)
    bank.enabled=True
    empty=model.predict(ids,state,logp)
    torch.testing.assert_close(empty[0],native[0],atol=0,rtol=0)
    fill(model,bank,engine,ids,logp,state)
    # Populate the ORIGINAL global memory too, so the conditional scorer must
    # implement its nonzero read port as well as the lexical port.
    model.memory.occupied[0]=True
    model.memory.source[0].normal_();model.memory.action[0].normal_()
    bank.enabled=False;native=model.predict(ids,state,logp)
    bank.enabled=True;mixed=model.predict(ids,state,logp)
    forecast=bank.forecast
    torch.testing.assert_close(mixed[0],native[0].log_softmax(-1),atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(mixed[1],native[1],atol=0,rtol=0)
    assert forecast.identities==[None]+[(i,0,1) for i in range(4)]
    assert forecast.report(ids)['predictive_probability_sum_error']<1e-6
    bank.enabled=False
    model.predict(ids,state,logp)
    assert bank.forecast is None


def test_held_cuts_old_native_inputs_only_and_output_is_prefix_causal():
    model,bank,engine,ids,logp,state=setup()
    fill(model,bank,engine,ids,logp,state)
    for m in bank.memories:
        torch.nn.init.normal_(m.inlet.projection.weight,std=.2)
        m.capsules[0].inputs.requires_grad_()
    bank.set_held(False)
    full=model.predict(ids,state,logp)[0]
    loss=F.cross_entropy(full.flatten(0,1),torch.tensor([2,6,7,8]))
    old=[m.capsules[0].inputs for m in bank.memories]
    gradients=torch.autograd.grad(loss,old)
    assert all(g.norm()>0 for g in gradients)
    bank.set_held(True)
    held=model.predict(ids,state,logp)[0]
    torch.testing.assert_close(full,held,atol=0,rtol=0)
    g=torch.autograd.grad(held.square().sum(),old,allow_unused=True)
    assert all(v is None for v in g)
    short=model.predict(ids[:,:2],state,logp[:,:2])[0]
    torch.testing.assert_close(held[:,:2],short,atol=2e-6,rtol=1e-5)
    assert all(m.current is None and m.position_read is None for m in bank.memories)


def test_outcomes_only_update_after_prediction_and_masked_prior_identity():
    model,bank,engine,ids,logp,state=setup()
    fill(model,bank,engine,ids,logp,state)
    for m in bank.memories:torch.nn.init.normal_(m.inlet.projection.weight,std=.2)
    logits=model.predict(ids,state,logp)[0]
    forecast=bank.forecast
    before=logits.detach().clone()
    a=forecast.posterior(torch.tensor([0,1,2,3]))
    b=forecast.posterior(torch.tensor([4,5,6,7]))
    assert not torch.equal(a,b)
    torch.testing.assert_close(logits,before,atol=0,rtol=0)
    derivative,=torch.autograd.grad(-logits[0,2:,7].mean(),forecast.prior_scores)
    expected=(forecast.log_prior.exp()-forecast.posterior(torch.full((4,),7)))
    expected[:2]=0
    torch.testing.assert_close(derivative,expected/2,atol=2e-7,rtol=1e-5)


def test_dual_views_share_records_and_split_prior_without_changing_capacity():
    model,bank,engine,ids,logp,state=setup(4)
    fill(model,bank,engine,ids,logp,state)
    with bank.observation(engine.optimizer.param_groups[-1],4):
        model.predict(ids,state,logp)
        for layer,m in zip(model.base.layers,bank.memories):
            pr=m.position_read
            assert pr['prediction_views']==['state','transition']
            assert pr['prediction_slots']==[0,0]
            assert pr['value_positions'].tolist()==[0,1,2,3,1,2,3]
            assert len(m.replayed)==1 and len(m.capsules)==2
            # Both views share the SAME encoder, with actually observed values.
            e=m.encoder(m.replayed_by_slot[0][0])
            w=pr['weights'];mass=pr['prediction_weights']
            reconstructed=(mass[:,0:1]*pr['prediction_values'][0]
                           +mass[:,1:2]*pr['prediction_values'][1])
            torch.testing.assert_close(reconstructed,w@torch.cat((e,e[1:])))
            torch.testing.assert_close(m.last_read.weights.sum(-1)+pr['null_weight'],torch.ones(4))
            torch.testing.assert_close(m.last_read.weights[:,0],mass.sum(-1))
        assert len(bank.forecast.identities)==9
        assert bank.forecast.identities[1:3]==[(0,0,1,'state'),(0,0,1,'transition')]


def test_dual_single_position_never_fabricates_a_successor():
    model,bank,engine,ids,logp,state=setup(4)
    fill(model,bank,engine,ids[:,:1],logp[:,:1],state)
    model.predict(ids,state,logp)
    assert bank.forecast.identities==[None]+[(i,0,1,'state') for i in range(4)]
    # A view prior cannot affect a record with no observed transition.
    grad=torch.autograd.grad(bank.forecast.logp.sum(),
        [m.view_logit for m in bank.memories],allow_unused=True)
    assert all(g is None for g in grad)


def test_dual_view_credit_reaches_native_computation_and_actual_later_layers():
    model,bank,engine,ids,logp,state=setup(4)
    fill(model,bank,engine,ids,logp,state)
    for m in bank.memories:
        torch.nn.init.normal_(m.inlet.projection.weight,std=.2)
        m.capsules[0].inputs.requires_grad_()
    full=model.predict(ids,state,logp)[0]
    forecast=bank.forecast
    objective=-full[0,:,7].mean()
    grads=torch.autograd.grad(objective,[m.view_logit for m in bank.memories],retain_graph=True)
    assert all(g.abs()>0 for g in grads)
    old=[m.capsules[0].inputs for m in bank.memories]
    assert all(g.norm()>0 for g in torch.autograd.grad(objective,old,retain_graph=True))
    # Layer0's state AND transition expert must execute layer1's native MLP.
    for component in (1,2):
        g,=torch.autograd.grad(-forecast.component_logp[:,component,7].mean(),
            model.base.layers[1].mlp['up'].weight,retain_graph=True)
        assert g.norm()>0
    bank.set_held(True)
    held=model.predict(ids,state,logp)[0]
    torch.testing.assert_close(held,full,atol=0,rtol=0)
    cut=torch.autograd.grad(-held[0,:,7].mean(),old,allow_unused=True)
    assert all(g is None for g in cut)


def test_dual_prefix_restore_and_zero_inlet_ports():
    import copy
    model,bank,engine,ids,logp,state=setup(4)
    bank.enabled=False;expected=model.predict(ids,state,logp)
    bank.enabled=True;empty=model.predict(ids,state,logp)
    torch.testing.assert_close(empty[0],expected[0],atol=0,rtol=0)
    fill(model,bank,engine,ids,logp,state)
    neutral=model.predict(ids,state,logp)
    torch.testing.assert_close(neutral[0],expected[0].log_softmax(-1),atol=2e-6,rtol=1e-5)
    for m in bank.memories:torch.nn.init.normal_(m.inlet.projection.weight,std=.2)
    full=model.predict(ids,state,logp)
    short=model.predict(ids[:,:2],state,logp[:,:2])
    torch.testing.assert_close(full[0][:,:2],short[0],atol=2e-6,rtol=1e-5)
    restored,_,_,_,_,_=setup(4)
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    actual=restored.predict(ids,state,logp)
    torch.testing.assert_close(full[0],actual[0],atol=0,rtol=0)
    torch.testing.assert_close(full[1],actual[1],atol=0,rtol=0)


def test_dual_view_prior_derivative_matches_finite_difference():
    model,bank,engine,ids,logp,state=setup(4)
    fill(model,bank,engine,ids,logp,state)
    for m in bank.memories:torch.nn.init.normal_(m.inlet.projection.weight,std=.2)
    p=bank.memories[0].view_logit
    def objective():return -model.predict(ids,state,logp)[0][0,:,7].mean()
    gradient,=torch.autograd.grad(objective(),p)
    step=.01
    with torch.no_grad():
        p.add_(step);plus=objective().item()
        p.sub_(2*step);minus=objective().item()
        p.add_(step)
    assert gradient.item()==pytest.approx((plus-minus)/(2*step),rel=.04,abs=4e-5)


def test_dual_views_settle_one_versioned_receipt_per_record_query_and_stay_bounded():
    model,bank,engine,ids,logp,state=setup(4)
    for _ in range(4):fill(model,bank,engine,ids,logp,state)
    for m in bank.memories:
        assert sum(c is not None for c in m.capsules)==2
        assert int(m.memory.clock)==16
        torch.nn.init.normal_(m.inlet.projection.weight,std=.2)
    with bank.observation(engine.optimizer.param_groups[-1],4):
        output=model.predict(ids,state,logp)[0]
        for m in bank.memories:
            assert m.memory._read_credit_entries==4*2
            assert len(m.memory._read_participation)==1
        (-output[0,:,7].mean()).backward()
        for m in bank.memories:
            before=m.memory.read_credit_count.clone()
            assert torch.isfinite(m.memory._read_participation[0].bias.grad).all()
            m.memory.settle_read_credit()
            torch.testing.assert_close(m.memory.read_credit_count-before,torch.full_like(before,4))
            assert m.memory._read_credit_entries==0


def test_dual_record_prior_does_not_reward_record_length_or_extra_view():
    model,bank,engine,ids,logp,state=setup(4)
    fill(model,bank,engine,ids,logp,state)
    fill(model,bank,engine,ids[:,:1],logp[:,:1],state)
    for m in bank.memories:
        with torch.no_grad():
            m.encoder.weight.zero_();m.encoder.bias.fill_(1)
    with bank.observation(engine.optimizer.param_groups[-1],4):
        model.predict(ids,state,logp)
        for m in bank.memories:
            # Identical evidence, one length4 record with two views, one
            # length1 record with state only: equal total record prior.
            torch.testing.assert_close(m.last_read.weights[:,0],m.last_read.weights[:,1])


def test_direct_attention_zero_migration_and_one_decoder_no_predictive_suffixes(monkeypatch):
    import scripts.ib_local.predictive_responsibility as api
    model,bank,engine,ids,logp,state=setup(5)
    bank.enabled=False;expected=model.predict(ids,state,logp)
    bank.enabled=True;empty=model.predict(ids,state,logp)
    torch.testing.assert_close(empty[0],expected[0],atol=0,rtol=0)
    fill(model,bank,engine,ids,logp,state)
    def forbidden(*args,**kwargs):raise AssertionError('conditional suffix must not run')
    monkeypatch.setattr(api,'native_predictive_mixture',forbidden)
    calls=[];layers=[]
    handles=[model.base.decoder.register_forward_hook(lambda *args:calls.append(True))]
    handles.extend(layer.register_forward_pre_hook(lambda *args:layers.append(True))
                   for layer in model.base.layers)
    try:actual=model.predict(ids,state,logp)
    finally:
        for h in handles:h.remove()
    torch.testing.assert_close(actual[0],expected[0],atol=0,rtol=0)
    torch.testing.assert_close(actual[1],expected[1],atol=0,rtol=0)
    assert len(calls)==1 and len(layers)==4 and bank.forecast is None


def test_direct_attention_returns_observed_native_values_without_latent_value_projection():
    model,bank,engine,ids,logp,state=setup(5)
    fill(model,bank,engine,ids,logp,state)
    m=bank.memories[0];layer=model.base.layers[0]
    query=torch.randn(3,8)
    result=m.retrieve_context(query,layer)
    actual=m.replayed_by_slot[0][0]
    torch.testing.assert_close(result,m.position_read['weights']@torch.cat((actual,actual[1:])))
    assert result.shape==(3,8) and m.encoder.out_features==4
    assert 'prediction_values' not in m.position_read
    assert m.query_encoder.weight is not m.encoder.weight
    assert m.inlet.projection.gain.numel()==1
    m.discard()


def test_direct_attention_held_cuts_old_native_only_and_retains_query_key_learning():
    model,bank,engine,ids,logp,state=setup(5)
    fill(model,bank,engine,ids,logp,state)
    for m in bank.memories:
        m.inlet.projection.gain.data.fill_(.2)
        m.capsules[0].inputs.requires_grad_()
    old=[m.capsules[0].inputs for m in bank.memories]
    full=model.predict(ids,state,logp)[0]
    assert all(g.norm()>0 for g in torch.autograd.grad(-full[0,:,7].mean(),old))
    bank.set_held(True)
    held=model.predict(ids,state,logp)[0]
    torch.testing.assert_close(held,full,atol=0,rtol=0)
    addresses=[p for m in bank.memories for p in (m.encoder.weight,m.query_encoder.weight)]
    grads=torch.autograd.grad(-held[0,:,7].mean(),old+addresses,allow_unused=True)
    assert all(g is None for g in grads[:4])
    assert all(g is not None and g.norm()>0 for g in grads[4:])


def test_direct_attention_prefix_and_full_native_state_restore():
    import copy
    model,bank,engine,ids,logp,state=setup(5)
    fill(model,bank,engine,ids,logp,state)
    for m in bank.memories:m.inlet.projection.gain.data.fill_(.2)
    full=model.predict(ids,state,logp)
    short=model.predict(ids[:,:2],state,logp[:,:2])
    torch.testing.assert_close(full[0][:,:2],short[0],atol=2e-6,rtol=1e-5)
    restored,_,_,_,_,_=setup(5)
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    actual=restored.predict(ids,state,logp)
    torch.testing.assert_close(actual[0],full[0],atol=0,rtol=0)
    torch.testing.assert_close(actual[1],full[1],atol=0,rtol=0)


def test_direct_attention_single_position_has_no_unobserved_successor():
    model,bank,engine,ids,logp,state=setup(5)
    fill(model,bank,engine,ids[:,:1],logp[:,:1],state)
    with bank.observation(engine.optimizer.param_groups[-1],4):
        model.predict(ids,state,logp)
        for m in bank.memories:
            assert m.position_read['prediction_views']==['state']
            assert m.position_read['value_positions'].tolist()==[0]
            assert m.memory._read_credit_entries==4


def test_direct_attention_gate_derivative_matches_boundary_loss_finite_difference():
    model,bank,engine,ids,logp,state=setup(5)
    fill(model,bank,engine,ids,logp,state)
    p=bank.memories[0].inlet.projection.gain
    # Use the actual vocabulary likelihood, not a newly installed local head.
    def objective():return -model.predict(ids,state,logp)[0].log_softmax(-1)[0,:,7].mean()
    derivative,=torch.autograd.grad(objective(),p)
    step=.01
    with torch.no_grad():
        p.add_(step);plus=objective().item()
        p.sub_(2*step);minus=objective().item()
        p.add_(step)
    assert derivative.item()==pytest.approx((plus-minus)/(2*step),rel=.03,abs=5e-5)


@pytest.mark.parametrize('gain', [0., -.2])
def test_native_output_projection_neutral_migration_preserves_old_state_and_rng(gain):
    from scripts.ib_local.small_gdn_experience import enable_native_output_projection
    model,bank,engine,ids,logp,state=setup(5)
    fill(model,bank,engine,ids,logp,state)
    for m in bank.memories:m.inlet.projection.gain.data.fill_(gain)
    before=model.predict(ids,state,logp)
    old={n:p.detach().clone() for n,p in model.named_parameters()}
    rng=torch.get_rng_state().clone()
    added=enable_native_output_projection(engine)
    assert len(added)==4 and sum(p.numel() for p in added)==4*8*8
    assert bank.config['native_output_projection']
    assert torch.equal(torch.get_rng_state(),rng)
    after=model.predict(ids,state,logp)
    torch.testing.assert_close(after[0],before[0],atol=0,rtol=0)
    torch.testing.assert_close(after[1],before[1],atol=0,rtol=0)
    for name,p in model.named_parameters():
        if name in old:torch.testing.assert_close(p,old[name],atol=0,rtol=0)
    assert enable_native_output_projection(engine)==[]


def test_native_output_matrix_receives_boundary_gradient_at_zero_scalar_gain():
    from scripts.ib_local.small_gdn_experience import enable_native_output_projection
    model,bank,engine,ids,logp,state=setup(5)
    fill(model,bank,engine,ids,logp,state)
    matrices=enable_native_output_projection(engine)
    logits=model.predict(ids,state,logp)[0]
    loss=-logits.log_softmax(-1)[0,:,7].mean()
    assert all(m.inlet.projection.gain.item()==0 for m in bank.memories)
    gradients=torch.autograd.grad(loss,matrices)
    assert all(torch.isfinite(g).all() and g.norm()>0 for g in gradients)
    p=matrices[0]
    index=tuple(torch.nonzero(gradients[0].abs()==gradients[0].abs().max())[0].tolist())
    def objective():return -model.predict(ids,state,logp)[0].log_softmax(-1)[0,:,7].mean()
    with torch.no_grad():
        p[index]+=.01;plus=objective().item()
        p[index]-=.02;minus=objective().item()
        p[index]+=.01
    assert gradients[0][index].item()==pytest.approx((plus-minus)/.02,rel=.03,abs=5e-5)


def test_projected_native_held_cuts_old_participation_but_output_map_still_learns():
    from scripts.ib_local.small_gdn_experience import enable_native_output_projection
    model,bank,engine,ids,logp,state=setup(5)
    fill(model,bank,engine,ids,logp,state)
    matrices=enable_native_output_projection(engine)
    for m,p in zip(bank.memories,matrices):
        torch.nn.init.normal_(p,std=.03)
        m.capsules[0].inputs.requires_grad_()
    old=[m.capsules[0].inputs for m in bank.memories]
    full=model.predict(ids,state,logp)[0]
    assert all(g.norm()>0 for g in torch.autograd.grad(-full[0,:,7].mean(),old))
    bank.set_held(True)
    held=model.predict(ids,state,logp)[0]
    torch.testing.assert_close(held,full,atol=0,rtol=0)
    short=model.predict(ids[:,:2],state,logp[:,:2])[0]
    torch.testing.assert_close(short,held[:,:2],atol=2e-6,rtol=1e-5)
    gradients=torch.autograd.grad(-held[0,:,7].mean(),old+matrices,allow_unused=True)
    assert all(g is None for g in gradients[:4])
    assert all(g is not None and g.norm()>0 for g in gradients[4:])


def test_output_projection_adam_parameter_order_restores_next_actual_update():
    import copy
    from scripts.ib_local.small_gdn_experience import enable_native_output_projection
    model,bank,engine,ids,logp,state=setup(5)
    # This fixture's engine.params aliases its original groups list. Use actual
    # optimizer group lists to model the production disjoint clipping groups.
    engine.groups=tuple(g['params'] for g in engine.optimizer.param_groups)
    fill(model,bank,engine,ids,logp,state)
    for p in model.parameters():p.grad=torch.full_like(p,.01)
    engine.optimizer.step();engine.optimizer.zero_grad(set_to_none=True)
    old={p:copy.deepcopy(v) for p,v in engine.optimizer.state.items()}
    enable_native_output_projection(engine)
    for p,expected in old.items():torch.testing.assert_close(engine.optimizer.state[p],expected)
    assert [id(p) for p in engine.optimizer.param_groups[-1]['params']]==[id(p) for p in bank.parameters()]
    for p in model.parameters():p.grad=torch.full_like(p,.02)
    engine.optimizer.step();engine.optimizer.zero_grad(set_to_none=True)
    weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(engine.optimizer.state_dict())
    restored,rbank,re,_,_,_=setup(5,output_projection=True)
    restored.load_state_dict(weights);re.optimizer.load_state_dict(adam)
    for m,e in ((model,engine),(restored,re)):
        e.optimizer.zero_grad(set_to_none=True)
        loss=-m.predict(ids,state,logp)[0].log_softmax(-1)[0,:,7].mean()
        loss.backward();e.optimizer.step()
    for a,b in zip(model.parameters(),restored.parameters()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    torch.testing.assert_close(engine.optimizer.state_dict(),re.optimizer.state_dict(),atol=0,rtol=0)
