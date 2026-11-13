"""Native execution/gradient contracts, not a synthetic capability study."""
import copy

import torch

from scripts.ib_local.gated_deltanet import GatedDeltaNetLayer
from scripts.ib_local.small_gdn_experience import SmallReplayBank


def setup():
    torch.manual_seed(19)
    layers = torch.nn.ModuleList([GatedDeltaNetLayer(8, 2, 'decay_then_delta',
        16, 4, 32., decay_init='log_timescale') for _ in range(4)])
    # Historical schema must remain reproducible after the repair.
    bank = SmallReplayBank(4, 8, 'cpu', capacity=2, latent_dim=4,
                           schema=1, replay_tokens=65)
    x = torch.randn(1, 128, 8)
    states = [torch.zeros(1, l.state_size) for l in layers]
    return layers, bank, x, states


def forward(layers, bank, x, states):
    with bank.forward_context(layers):
        for layer, state in zip(layers, states):
            x = layer(x, state)[0]
    return x


def fill(layers, bank, x, states):
    with bank.observation(dict(lr=3e-4, betas=(.9, .999), eps=1e-8), 128):
        with torch.no_grad():
            forward(layers, bank, x, states)
            bank.settle_and_commit()


def test_zero_migration_and_actual_boundary_capture():
    layers, bank, x, states = setup()
    bank.enabled = False
    expected = forward(layers, bank, x, states)
    bank.enabled = True
    actual = forward(layers, bank, x, states)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    fill(layers, bank, x, states)
    for i, m in enumerate(bank.memories):
        c = m.capsules[0]
        assert c.inputs.shape == (1, 65, 8)
        torch.testing.assert_close(c.boundary, states[i])
        assert c.inputs.grad_fn is None and c.boundary.grad_fn is None


def test_replay_bypasses_live_hooks_and_keeps_actual_posthook_input():
    layers, bank, x, states = setup()
    m = bank.memories[0]
    with bank.observation(dict(lr=3e-4, betas=(.9, .999), eps=1e-8), 128):
        with bank.forward_context(layers):
            hook = layers[0].register_forward_pre_hook(lambda module, a: (a[0]+3, a[1]))
            try:
                layers[0](x, states[0])
                torch.testing.assert_close(m.current['inputs'], x+3)
                m.memory.settle_read_credit()
                m.commit()
            finally:
                hook.remove()
    calls = []
    hook = layers[0].register_forward_pre_hook(lambda module, a: calls.append(True))
    try:
        with m.attach_native(layers[0], capture=False):
            layers[0](x, states[0])
        assert len(calls) == 1
    finally:
        hook.remove()
        m.discard()


def test_held_same_forward_but_old_gdn_and_mlp_derivatives_cut_all_layers():
    layers, bank, x, states = setup()
    fill(layers, bank, x, states)
    for layer, m in zip(layers, bank.memories):
        query = x[0, :2]
        replay = lambda c: layer.forward(c.inputs, c.boundary)[0]
        m.detach_replay = False
        full = m.retrieve(query, replay)
        grads = torch.autograd.grad(full.square().sum(),
            (layer.alpha_proj.weight, layer.beta_proj.weight, layer.mlp['up'].weight),
            allow_unused=True)
        assert all(g is not None and g.norm() > 0 for g in grads)
        m.detach_replay = True
        held = m.retrieve(query, replay)
        torch.testing.assert_close(full, held, atol=0, rtol=0)
        g, e = torch.autograd.grad(held.square().sum(),
            (layer.mlp['up'].weight, m.encoder.weight), allow_unused=True)
        assert g is None and e.norm() > 0
        m.discard()


def test_prefix_causality_and_restore_of_finite_records():
    layers, bank, x, states = setup()
    for _ in range(4):
        fill(layers, bank, x, states)
    for m in bank.memories:
        torch.nn.init.normal_(m.inlet.projection.weight, std=.01)
        assert sum(c is not None for c in m.capsules) == 2
    full = forward(layers, bank, x, states)
    short = forward(layers, bank, x[:, :9], states)
    torch.testing.assert_close(full[:, :9], short, atol=1e-6, rtol=1e-5)
    restored = SmallReplayBank(4, 8, 'cpu', capacity=2, latent_dim=4,
                               schema=1, replay_tokens=65)
    restored.load_state_dict(copy.deepcopy(bank.state_dict()))
    torch.testing.assert_close(full, forward(layers, restored, x, states), atol=0, rtol=0)


def test_partial_observation_advances_clock_without_fabricating_future():
    layers, bank, x, states = setup()
    with bank.observation(dict(lr=3e-4, betas=(.9, .999), eps=1e-8), 17):
        forward(layers, bank, x[:, :17], states)
        bank.settle_and_commit()
    assert all(int(m.memory.clock) == 17 and not m.memory.occupied.any()
               for m in bank.memories)


def context_setup():
    layers,_,x,states = setup()
    bank = SmallReplayBank(4,8,'cpu',capacity=2,latent_dim=4,schema=2)
    return layers,bank,x,states


def test_whole_observation_and_partial_tail_are_both_retained():
    layers,bank,x,states = context_setup()
    fill(layers,bank,x,states)
    for i,m in enumerate(bank.memories):
        assert m.capsules[0].inputs.shape == (1,128,8)
        if i==0:torch.testing.assert_close(m.capsules[0].inputs,x,atol=0,rtol=0)
    with bank.observation(dict(lr=3e-4,betas=(.9,.999),eps=1e-8),17):
        forward(layers,bank,x[:,:17],states)
        bank.settle_and_commit()
    for m in bank.memories:
        assert int(m.memory.clock)==145
        assert sorted(c.inputs.shape[1] for c in m.capsules)==[17,128]


def test_contextual_query_distinguishes_equal_current_input_with_different_past():
    layers,bank,x,states = context_setup()
    fill(layers,bank,x,states)
    captured=[]
    m=bank.memories[0];old=m.retrieve_context
    def read(q,block):
        captured.append(q.detach().clone())
        return old(q,block)
    m.retrieve_context=read
    a=x[:,-1:];other=[s.clone() for s in states]
    other[0][:,:layers[0].matrix_size].fill_(.5)
    forward(layers,bank,a,states)
    forward(layers,bank,a,other)
    assert not torch.equal(captured[0],captured[1])


def test_all_retained_positions_receive_credit_including_previous_second_half():
    layers,bank,x,states = context_setup()
    fill(layers,bank,x,states)
    for layer,m in zip(layers,bank.memories):
        c=m.capsules[0]
        c.inputs.requires_grad_()
        query=torch.randn(2,8)
        value=m.retrieve_context(query,layer)
        grad,=torch.autograd.grad(value.square().sum(),c.inputs)
        assert grad[:,90:].norm()>0
        assert (m.position_read['weights']>0).all()
        assert torch.all(m.last_read.weights.sum(-1)<1)
        m.discard()


def test_context_held_forward_same_and_native_derivative_only_is_cut():
    layers,bank,x,states = context_setup()
    fill(layers,bank,x,states)
    for layer,m in zip(layers,bank.memories):
        query=torch.randn(2,8)
        m.detach_replay=False
        a=m.retrieve_context(query,layer)
        g=torch.autograd.grad(a.square().sum(),(layer.alpha_proj.weight,
            layer.beta_proj.weight,layer.mlp['up'].weight),retain_graph=True)
        assert all(v.norm()>0 for v in g)
        m.discard();m.detach_replay=True
        b=m.retrieve_context(query,layer)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        native,encoder=torch.autograd.grad(b.square().sum(),
            (layer.mlp['up'].weight,m.encoder.weight),allow_unused=True)
        assert native is None and encoder.norm()>0
        m.discard()


def test_context_prefix_causality_and_restored_variable_records():
    layers,bank,x,states=context_setup()
    fill(layers,bank,x,states)
    for m in bank.memories:torch.nn.init.normal_(m.inlet.projection.weight,std=.01)
    a=forward(layers,bank,x,states)
    b=forward(layers,bank,x[:,:9],states)
    torch.testing.assert_close(a[:,:9],b,atol=1e-6,rtol=1e-5)
    restored=SmallReplayBank(4,8,'cpu',capacity=2,latent_dim=4,schema=2)
    restored.load_state_dict(copy.deepcopy(bank.state_dict()))
    torch.testing.assert_close(a,forward(layers,restored,x,states),atol=0,rtol=0)


def test_context_replay_bypasses_current_hooks_and_zero_migration_is_exact():
    layers,bank,x,states=context_setup()
    bank.enabled=False;a=forward(layers,bank,x,states)
    bank.enabled=True;b=forward(layers,bank,x,states)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    fill(layers,bank,x,states)
    calls=[]
    hook=layers[0].register_forward_pre_hook(lambda module,args:calls.append(True))
    try:forward(layers,bank,x,states)
    finally:hook.remove()
    assert len(calls)==1


def test_context_native_credit_matches_finite_difference_including_reencoded_keys():
    layers,bank,x,states=context_setup()
    fill(layers,bank,x,states)
    layer=layers[0];m=bank.memories[0];query=torch.randn(3,8)
    def value():
        m.discard()
        return m.retrieve_context(query,layer).square().sum()
    gradient,=torch.autograd.grad(value(),layer.alpha_proj.weight)
    index=tuple(torch.nonzero(gradient.abs()==gradient.abs().max())[0].tolist())
    step=.01
    with torch.no_grad():
        original=layer.alpha_proj.weight[index].item()
        layer.alpha_proj.weight[index]=original+step;plus=value().item()
        layer.alpha_proj.weight[index]=original-step;minus=value().item()
        layer.alpha_proj.weight[index]=original
    import pytest
    assert gradient[index].item()==pytest.approx((plus-minus)/(2*step),rel=.03,abs=2e-4)


def test_context_memory_remains_bounded_across_many_commits():
    layers,bank,x,states=context_setup()
    for _ in range(7):fill(layers,bank,x,states)
    for m in bank.memories:
        assert len(m.capsules)==2
        assert m.memory.occupied.sum()==2
        assert all(c.inputs.shape[1]<=128 and c.inputs.grad_fn is None for c in m.capsules)
        assert int(m.memory.clock)==7*128
