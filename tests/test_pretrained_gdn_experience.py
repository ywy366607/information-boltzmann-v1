"""Derivative and storage boundaries; no synthetic capability benchmark."""
import copy

import pytest
import torch
from torch import nn

from scripts.ib_local.pretrained_gdn_experience import (
    LayerExperienceRecall, LocalExperience, copy_tree,
)


def populated():
    torch.manual_seed(22)
    memory = LayerExperienceRecall(6, capacity=2, latent_dim=3, topk=2,read_interval=None)
    m = memory.memory
    m.clock.fill_(256)
    for i in range(2):
        x = torch.randn(6)
        m.commit_episode(x, x+1, x+2, x+3, memory.encoder, source_time=64+i)
        memory.capsules[i] = LocalExperience(torch.randn(1, 65, 6), None,
                                             int(m.read_credit_version[i]))
    return memory


def test_checked_relation_drives_actual_read_then_receives_actual_loss_and_restores():
    from scripts.ib_local.causal_relation_runtime import admit
    memory = populated()
    # A fixed, checked +0.25 change to the actual read multiplier.
    relation = admit([{'op': 'const', 'arg': 250000}], 2, [[0, 0]])
    query = torch.randn(1, 6)
    original = memory.retrieve(query, lambda c: c.inputs)
    memory.install_causal_relation(relation)
    memory.begin(dict(lr=3e-4, betas=(.9, .999), eps=1e-8))
    altered = memory.retrieve(query, lambda c: c.inputs)
    torch.testing.assert_close(altered, original * 1.25)
    altered.square().sum().backward()
    feedback = memory.settle_causal_relation()
    assert feedback['records'] == 2
    assert memory.causal_relation['count'] == 2
    assert all(torch.isfinite(torch.tensor(x)).all() for x in feedback['feedback'])
    memory.memory.settle_read_credit()
    restored = LayerExperienceRecall(6, capacity=2, latent_dim=3, topk=2)
    restored.load_state_dict(copy.deepcopy(memory.state_dict()))
    assert restored.causal_relation == memory.causal_relation
    torch.testing.assert_close(restored.retrieve(query, lambda c: c.inputs), altered.detach())


def test_current_error_updates_reexecuted_native_parameters_after_graph_detach():
    memory = populated()
    original = nn.Linear(6, 6)
    query = torch.randn(1, 6)
    memory.begin(dict(lr=3e-4, betas=(.9, .999), eps=1e-8))
    old = []

    def replay(capsule):
        assert capsule.inputs.grad_fn is None
        value = original(capsule.inputs)
        old.append(value)
        return value

    result = memory.retrieve(query, replay)
    loss = result.square().sum()
    loss.backward()
    assert original.weight.grad.norm() > 0
    assert memory.encoder.weight.grad.norm() > 0
    credit = memory.memory.settle_read_credit()
    assert credit['records'] == 2
    assert memory.memory.read_credit_mean.abs().max() > 0
    assert all(x.inputs.grad is None for x in memory.capsules)


def test_old_computation_gradient_matches_finite_difference_with_held_boundary():
    memory = populated()
    query = torch.randn(1, 6)
    gain = nn.Parameter(torch.tensor(.3))

    def value():
        return memory.retrieve(query, lambda c: (gain*c.inputs).tanh()).square().sum()

    actual, = torch.autograd.grad(value(), gain)
    with torch.no_grad():
        gain.add_(1e-3)
        plus = value().item()
        gain.sub_(2e-3)
        minus = value().item()
        gain.add_(1e-3)
    assert actual.item() == pytest.approx((plus-minus)/2e-3, rel=2e-3, abs=1e-4)


def test_replaced_episode_invalidates_old_executable_capsule():
    memory = populated()
    memory.memory.read_credit_version.add_(1)
    with pytest.raises(RuntimeError, match='identity diverged'):
        memory.retrieve(torch.randn(1, 6), lambda c: c.inputs)


def test_executable_capsules_and_read_preferences_restore_together():
    memory = populated()
    memory.memory.read_credit_mean.copy_(torch.tensor([.2, -.4]))
    restored = LayerExperienceRecall(6, capacity=2, latent_dim=3, topk=2)
    restored.load_state_dict(copy.deepcopy(memory.state_dict()))
    query = torch.randn(1, 6)
    replay = lambda c: c.inputs.sin()
    torch.testing.assert_close(memory.retrieve(query, replay), restored.retrieve(query, replay), atol=0, rtol=0)
    assert memory.storage_nbytes == restored.storage_nbytes
    assert all(not capsule.inputs.requires_grad for capsule in restored.capsules)


def test_boundary_snapshot_is_detached_and_independent():
    x = torch.randn(1, 4, 3, 3, requires_grad=True)
    source = dict(recurrent_state=x*2, conv_state=(x, None))
    observed = copy_tree(source)
    assert observed['recurrent_state'].grad_fn is None
    assert observed['conv_state'][0].grad_fn is None
    with torch.no_grad():
        observed['recurrent_state'].zero_()
    assert source['recurrent_state'].abs().sum() > 0


def test_held_replay_cuts_native_credit_but_preserves_forward_and_current_encoder():
    memory = populated()
    native = nn.Linear(6, 6)
    query = torch.randn(1, 6)
    replay = lambda capsule: native(capsule.inputs)
    full = memory.retrieve(query, replay)
    memory.detach_replay = True
    held = memory.retrieve(query, replay)
    torch.testing.assert_close(full, held, atol=0, rtol=0)
    native_grad, encoder_grad = torch.autograd.grad(held.square().sum(),
        (native.weight, memory.encoder.weight), allow_unused=True)
    assert native_grad is None
    assert encoder_grad.norm() > 0


def test_strict_age_cut_preserves_within_window_and_current_encoder_gradients():
    memory = populated()
    memory.memory.birth.copy_(torch.tensor([128, 127]))
    memory.query_clock_offsets = torch.tensor([0, 1])
    native = nn.Parameter(torch.tensor([.4, .7]))
    identities = {id(c): i for i, c in enumerate(memory.capsules)}
    replay = lambda c: native[identities[id(c)]] * c.inputs
    query = torch.randn(2, 6)
    full = memory.retrieve(query, replay)
    first_native, = torch.autograd.grad(full[0].sum(), native, retain_graph=True)
    full_encoder, = torch.autograd.grad(full.sum(), memory.encoder.weight)
    memory.detach_replay_older_than = 128
    cut = memory.retrieve(query, replay)
    torch.testing.assert_close(full.detach(), cut.detach(), atol=0, rtol=0)
    cut_native, cut_encoder = torch.autograd.grad(cut.sum(), (native, memory.encoder.weight))
    # Record0 at offset0 is exactly128 old, not cut. At offset1 it is129.
    assert cut_native[0] == pytest.approx(float(first_native[0]), abs=1e-6)
    assert abs(float(cut_native[0])) > 0
    assert float(cut_native[1]) == 0
    torch.testing.assert_close(full_encoder, cut_encoder, atol=1e-7, rtol=1e-6)


def test_strict_age_policy_round_trips_without_changing_legacy_default():
    memory = populated()
    assert memory.detach_replay_older_than is None
    memory.detach_replay_older_than = 128
    restored = LayerExperienceRecall(6, capacity=2, latent_dim=3, topk=2)
    restored.load_state_dict(copy.deepcopy(memory.state_dict()))
    assert restored.detach_replay_older_than == 128


def test_old_read_cut_keeps_unmasked_contribution_fixed():
    memory = populated()
    query = torch.randn(1, 6)
    replay = lambda capsule: capsule.inputs
    full = memory.retrieve(query, replay)
    memory.suppress_older_than = 128
    cut = memory.retrieve(query, replay)
    assert full.norm() > 0
    assert cut.norm() == 0  # both actual source ages exceed128
    memory.suppress_older_than = None
    torch.testing.assert_close(full, memory.retrieve(query, replay), atol=0, rtol=0)


def test_observation_cannot_append_to_live_training_responses():
    from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
    memory = populated()
    live = torch.randn(1, 6, requires_grad=True)
    memory.replayed = [live]
    memory.current = {'source': live}
    original_current = memory.current
    learner = PretrainedExperienceLearner.__new__(PretrainedExperienceLearner)
    learner.memories = {1: memory}
    with learner.observation(suppress_older_than=128):
        memory.replayed.append(live.detach())
        memory.current = {'probe': True}
    assert len(memory.replayed) == 1
    assert memory.replayed[0] is live
    assert memory.current is original_current
    assert memory.suppress_older_than is None


def test_answer_aligned_query_cannot_inject_later_information_into_prefix():
    from types import SimpleNamespace

    class CausalBlock(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return (hidden_states.cumsum(dim=1),)

    memory = populated()
    block = CausalBlock()
    model = SimpleNamespace(model=SimpleNamespace(layers=[block]))
    with torch.no_grad():
        memory.inlet.projection.weight.fill_(.05)
    x = torch.randn(1, 128, 6)
    baseline = block(x)[0]
    with memory.attach(model, layer=0, read_position=90):
        original = block(x)[0]
    torch.testing.assert_close(original[:, :90], baseline[:, :90], atol=0, rtol=0)
    assert (original[:, 90:] - baseline[:, 90:]).abs().max() > 0
    changed = x.clone()
    changed[:, 91:] += 5
    with memory.attach(model, layer=0, read_position=90):
        future = block(changed)[0]
    torch.testing.assert_close(future[:, :91], original[:, :91], atol=0, rtol=0)
    with pytest.raises(ValueError, match='observed input'):
        with memory.attach(model, layer=0, read_position=128):
            block(x)


def test_single_token_read_is_allowed_but_cannot_become_an_episode():
    from types import SimpleNamespace

    class Block(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return (hidden_states.sin(),)

    memory = populated()
    memory.read_interval = 1
    block = Block()
    model = SimpleNamespace(model=SimpleNamespace(layers=[block]))
    x = torch.randn(1, 1, 6)
    clock = memory.memory.clock.clone()
    with memory.attach(model, layer=0, capture=False):
        output = block(x)[0]
    assert output.shape == x.shape
    assert memory.current['query_positions'] == [0]
    with pytest.raises(RuntimeError, match='read-only'):
        memory.commit()
    torch.testing.assert_close(memory.memory.clock, clock, atol=0, rtol=0)
    with pytest.raises(ValueError, match='>=128'):
        with memory.attach(model, layer=0):
            block(x)


def test_full_window_capture_preserves_all_inputs_without_changing_legacy_response():
    legacy=populated()
    complete=populated()
    complete.capture_full_window=True
    inputs=torch.randn(1,128,6,requires_grad=True)
    for wrapper in (legacy,complete):
        wrapper.current=dict(source=inputs,inputs=inputs*2,response=inputs*3,
                             boundary=None,capture=True)
    old_slot,new_slot=legacy.commit(),complete.commit()
    assert old_slot==new_slot
    a,b=legacy.capsules[old_slot],complete.capsules[new_slot]
    assert a.continuation is None and b.continuation.shape==(1,63,6)
    torch.testing.assert_close(a.inputs,b.inputs,atol=0,rtol=0)
    torch.testing.assert_close(b.complete_inputs,inputs.detach()*2,atol=0,rtol=0)
    assert b.complete_inputs.grad_fn is None
    assert complete.storage_nbytes-legacy.storage_nbytes==63*6*4
    query=torch.randn(3,6)
    torch.testing.assert_close(legacy.retrieve(query,lambda c:c.inputs),
                               complete.retrieve(query,lambda c:c.inputs),atol=0,rtol=0)


def test_full_window_capture_round_trips_and_legacy_defaults_stay_bounded():
    wrapper=populated()
    wrapper.capture_full_window=True
    wrapper.capsules[0].continuation=torch.randn(1,63,6)
    saved=copy.deepcopy(wrapper.state_dict())
    restored=populated()
    restored.load_state_dict(saved)
    assert restored.capture_full_window
    torch.testing.assert_close(restored.capsules[0].complete_inputs,
                               wrapper.capsules[0].complete_inputs,atol=0,rtol=0)
    del saved['_extra_state']['capture_full_window']
    saved['_extra_state']['capsules'][0].continuation=None
    restored.load_state_dict(saved)
    assert not restored.capture_full_window
    assert restored.capsules[0].complete_inputs.shape==(1,65,6)


def test_native_reactivation_reaches_tail_participation_only_when_enabled():
    from types import SimpleNamespace

    class LocalProgram(nn.Module):
        def __init__(self):
            super().__init__()
            self.position_gain=nn.Parameter(torch.ones(128))

        def forward(self,hidden_states,**kwargs):
            return ((hidden_states*self.position_gain[None,:hidden_states.shape[1],None]).cumsum(1),)

    wrapper=populated()
    for capsule in wrapper.capsules:
        if capsule is not None:
            capsule.inputs=torch.ones(1,65,6)
            capsule.continuation=torch.ones(1,63,6)
    block=LocalProgram()
    model=SimpleNamespace(model=SimpleNamespace(layers=[block]))
    original_capsules=list(wrapper.capsules)
    for enabled in (False,True):
        wrapper.capsules=([LocalExperience(c.complete_inputs,c.boundary,c.version)
                           if c is not None else None for c in original_capsules]
                          if enabled else original_capsules)
        block.zero_grad(set_to_none=True)
        with wrapper.attach(model,layer=0,capture=False):
            block(torch.ones(1,1,6))
        # Inspect the old native program itself, excluding the live block's path.
        wrapper.replayed[0].sum().backward()
        tail=block.position_gain.grad[65:]
        assert bool((tail!=0).all())==enabled
        wrapper.replayed=[]


def test_full_window_age_cut_keeps_values_and_current_encoder_but_cuts_old_native():
    wrapper=populated()
    for capsule in wrapper.capsules:
        capsule.continuation=torch.randn(1,63,6)
    native=nn.Linear(6,6)
    query=torch.randn(1,6)
    replay=lambda capsule:native(capsule.complete_inputs).cumsum(1)
    full=wrapper.retrieve(query,replay)
    full_native,=torch.autograd.grad(full.square().sum(),native.weight)
    wrapper.detach_replay_older_than=128
    held=wrapper.retrieve(query,replay)
    torch.testing.assert_close(full,held,atol=0,rtol=0)
    old_grad,current_grad=torch.autograd.grad(held.square().sum(),
        (native.weight,wrapper.encoder.weight),allow_unused=True)
    assert full_native.norm()>0
    assert old_grad is None or not bool(old_grad.any())
    assert current_grad.norm()>0


def test_current_query_uses_old_program_state_and_held_cuts_only_old_stage(monkeypatch):
    from types import SimpleNamespace
    from scripts.ib_local import probe_pending_delivery as module
    monkeypatch.setattr(module,'new_recurrent_cache',lambda:SimpleNamespace(states=[],_seen_tokens=0))
    # clone_cache uses the actual FLA cache constructor, so isolate only its
    # container here; the native GPU operator is checked by the runtime probe.
    from scripts.ib_local import pretrained_gdn
    monkeypatch.setattr(pretrained_gdn,'new_recurrent_cache',lambda seen_tokens=0:
                        SimpleNamespace(states=[],_seen_tokens=seen_tokens))

    class Program(nn.Module):
        def __init__(self):
            super().__init__()
            self.gain=nn.Parameter(torch.tensor(2.))

        def forward(self,x,past_key_values,use_cache):
            state=past_key_values.states[0].get('recurrent_state',x.new_zeros(1,1,1))
            out=state+self.gain*x.cumsum(1)
            past_key_values.states[0]['recurrent_state']=out[:,-1:]
            return (out,)

    program=Program()
    held_boundary={'recurrent_state':torch.tensor([[[3.]]])}
    capsule=LocalExperience(torch.ones(1,65,1),held_boundary,0,torch.ones(1,63,1))
    query=torch.tensor([[[4.]]],requires_grad=True)
    full=module.execute_local_query(program,capsule,query,layer=0)
    held=module.execute_local_query(program,capsule,query,layer=0,detach_history=True)
    torch.testing.assert_close(full,held,atol=0,rtol=0)
    a,b=torch.autograd.grad(full,(program.gain,query))
    c,d=torch.autograd.grad(held,(program.gain,query))
    assert a==132 and c==4
    torch.testing.assert_close(b,d,atol=0,rtol=0)
    assert held_boundary['recurrent_state'].item()==3


def test_joint_archive_is_shared_across_layers_and_contains_only_detached_vectors(tmp_path):
    from types import SimpleNamespace
    from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
    from scripts.ib_local.pretrained_gdn_experience import tree_nbytes
    first,second=populated(),populated()
    embedding=nn.Embedding(11,6)
    learner=PretrainedExperienceLearner.__new__(PretrainedExperienceLearner)
    learner.model=SimpleNamespace(get_input_embeddings=lambda:embedding)
    learner.memories={1:first,12:second}
    boundary=SimpleNamespace(states=[{'recurrent_state':torch.randn(1,2,3,3)}
                                     for _ in range(24)],_seen_tokens=1280)
    tokens=torch.ones(1,128,dtype=torch.long)
    before=learner.experience_storage_nbytes
    joint=learner.archive_joint_participation(tokens,boundary,{1:0,12:1},9)
    assert first.capsules[0].joint is second.capsules[1].joint
    assert joint.incoming_features.grad_fn is None and not joint.incoming_features.requires_grad
    assert len(joint.boundary['states'])==24 and joint.parameter_version==9
    assert learner.experience_storage_nbytes-before==tree_nbytes(joint)
    boundary.states[0]['recurrent_state'].zero_()
    assert joint.boundary['states'][0]['recurrent_state'].norm()>0
    path=tmp_path/'capsules.pt'
    torch.save([first.capsules[0],second.capsules[1]],path)
    a,b=torch.load(path,weights_only=False)
    assert a.joint is b.joint
    assert set(vars(joint))=={'incoming_features','boundary','parameter_version'}


def test_joint_program_held_control_preserves_query_gradient_and_cuts_only_old_stage(monkeypatch):
    from types import SimpleNamespace
    from scripts.ib_local import pretrained_gdn_experience as module,pretrained_gdn
    constructor=lambda seen_tokens=0:SimpleNamespace(states=[],_seen_tokens=seen_tokens)
    monkeypatch.setattr(module,'new_recurrent_cache',constructor)
    monkeypatch.setattr(pretrained_gdn,'new_recurrent_cache',constructor)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.gain=nn.Parameter(torch.tensor(2.))

        def forward(self,inputs_embeds,past_key_values,use_cache):
            old=past_key_values.states[0]['recurrent_state']
            value=old+self.gain*inputs_embeds.cumsum(1)
            past_key_values.states[0]['recurrent_state']=value[:,-1:]
            return SimpleNamespace(last_hidden_state=value,past_key_values=past_key_values)

    class World(nn.Module):
        def __init__(self):
            super().__init__()
            self.model=Inner()

        def forward(self,**kwargs):
            output=self.model(**kwargs)
            return SimpleNamespace(logits=output.last_hidden_state,past_key_values=output.past_key_values)

    archive=module.JointExperience(torch.ones(1,128,1),
        dict(states=[{'recurrent_state':torch.tensor([[[3.]]])}],seen_tokens=1280),0)
    model=World()
    query=torch.tensor([[[4.]]],requires_grad=True)
    full=module.execute_joint_query(model,archive,query).logits
    held=module.execute_joint_query(model,archive,query,detach_history=True).logits
    torch.testing.assert_close(full,held,atol=0,rtol=0)
    a,b=torch.autograd.grad(full,(model.model.gain,query))
    c,d=torch.autograd.grad(held,(model.model.gain,query))
    assert a==132 and c==4
    torch.testing.assert_close(b,d,atol=0,rtol=0)
    assert archive.boundary['states'][0]['recurrent_state'].item()==3
