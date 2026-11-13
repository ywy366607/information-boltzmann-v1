"""Causal, gradient and bounded-storage checks for position-wise retrieval."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.ib_local.pretrained_gdn_experience import LayerExperienceRecall
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from tests.test_pretrained_gdn_experience import populated


class CausalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(6,6,bias=False)

    def forward(self, hidden_states, **kwargs):
        return (self.projection(hidden_states).cumsum(1).tanh(),)


def setup(interval=1, placement='input'):
    w = populated()
    w.read_interval, w.recall_placement = interval, placement
    with torch.no_grad():
        w.inlet.projection.weight.fill_(.05)
    block = CausalBlock()
    model = SimpleNamespace(model=SimpleNamespace(layers=[block]))
    return w,block,model,torch.randn(1,128,6)


@pytest.mark.parametrize('placement',['input','output'])
def test_each_position_queries_causally_and_changes_only_its_future(placement):
    w,block,model,x = setup(placement=placement)
    baseline = block(x)[0]
    with w.attach(model,layer=0,read_position=7):
        first = block(x)[0]
    assert w.last_read.slots.shape == (121,2)
    assert w.current['query_positions'] == list(range(7,128))
    assert w.current['injection'].shape == (1,121,6)
    assert torch.equal(first[:,:7],baseline[:,:7])
    assert not torch.equal(w.last_read.weights[0],w.last_read.weights[-1])
    changed = x.clone(); changed[:,66:] += 2
    with w.attach(model,layer=0,read_position=7):
        second = block(changed)[0]
    torch.testing.assert_close(first[:,:66],second[:,:66],atol=0,rtol=0)


def test_queries_share_old_computation_but_preserve_values_gradients_and_credit():
    w = populated(); w.enable_null_read()
    native = nn.Linear(6,6)
    source = torch.randn(11,6)
    calls = []

    def replay(capsule):
        calls.append(capsule.version)
        return native(capsule.inputs)

    w.begin(dict(lr=3e-4,betas=(.9,.999),eps=1e-8))
    actual = w.retrieve(source,replay)
    actual.square().mean().backward()
    actual_gradient = native.weight.grad.clone()
    actual_encoder = w.encoder.weight.grad.clone()
    actual_bias = w.memory._read_participation[0].bias.grad.clone()
    assert len(calls) == len(w.replayed_by_slot) == 2  # not 11*2 replays
    credit = w.memory.settle_read_credit()
    assert credit['participations'] == 22
    native.zero_grad(); w.zero_grad()
    # Reset preferences to compare independent per-position execution at the
    # same parameter/version, retaining a separate logit receipt for each read.
    w.memory.read_credit_mean.zero_()
    w.begin(dict(lr=3e-4,betas=(.9,.999),eps=1e-8))
    reference = torch.cat([w.retrieve(row[None],replay) for row in source])
    reference.square().mean().backward()
    torch.testing.assert_close(actual,reference,atol=2e-7,rtol=2e-6)
    torch.testing.assert_close(actual_gradient,native.weight.grad,atol=2e-7,rtol=2e-5)
    torch.testing.assert_close(actual_encoder,w.encoder.weight.grad,atol=2e-7,rtol=2e-5)
    torch.testing.assert_close(actual_bias,torch.cat([r.bias.grad for r in w.memory._read_participation]),
                               atol=2e-7,rtol=2e-5)


def test_held_multiquery_forward_is_equal_and_only_old_native_credit_is_cut():
    w = populated(); native = nn.Linear(6,6)
    x = torch.randn(17,6)
    full = w.retrieve(x,lambda c:native(c.inputs))
    w.detach_replay = True
    held = w.retrieve(x,lambda c:native(c.inputs))
    assert torch.equal(full,held)
    old,current = torch.autograd.grad(held.square().sum(),(native.weight,w.encoder.weight),allow_unused=True)
    assert old is None and current.norm() > 0


def test_interval_controls_piecewise_refresh_without_changing_observation_or_storage():
    w,block,model,x = setup(interval=7)
    before = w.storage_nbytes
    with w.attach(model,layer=0,read_position=3):
        block(x)
    assert w.current['query_positions'] == list(range(3,128,7))
    injection = w.current['injection']
    assert torch.equal(injection[:,0],injection[:,6])
    assert not torch.equal(injection[:,0],injection[:,7])
    assert w.storage_nbytes == before
    assert len(w.replayed_by_slot) <= w.memory.capacity
    w.commit()
    assert int(w.memory.clock) == 384  # 128 events, not 18 query events
    assert len(w.capsules) == 2
    assert all(c.inputs.shape == (1,65,6) and c.inputs.grad_fn is None for c in w.capsules)


def test_receipt_budget_follows_finite_observation_not_capacity_or_128():
    w,block,model,_ = setup()
    x = torch.randn(1,512,6)
    w.begin(dict(lr=3e-4,betas=(.9,.999),eps=1e-8),query_limit=512)
    with w.attach(model,layer=0):
        value = block(x)[0]
    value.square().mean().backward()
    assert w.memory.settle_read_credit()['participations'] == 1024


def test_cadence_migration_roundtrip_legacy_restore_and_live_guard():
    w,_,_,_ = setup()
    state = copy.deepcopy(w.state_dict())
    restored = populated(); restored.load_state_dict(state)
    assert restored.read_interval == 1
    del state['_extra_state']['read_interval']
    restored.load_state_dict(state)
    assert restored.read_interval is None
    learner = PretrainedExperienceLearner.__new__(PretrainedExperienceLearner)
    learner.memories = {1:restored}
    learner.set_read_interval(4)
    assert restored.read_interval == 4
    restored.current = {}
    with pytest.raises(RuntimeError,match='finish'):
        learner.set_read_interval(1)


@pytest.mark.parametrize('interval',[0,-1,True,1.5])
def test_invalid_cadence_is_rejected(interval):
    with pytest.raises(ValueError,match='interval'):
        LayerExperienceRecall(6,read_interval=interval)


def test_new_instances_query_every_position_and_zero_inlet_is_neutral():
    assert LayerExperienceRecall(6).read_interval == 1
    w,block,model,x = setup()
    with torch.no_grad():
        w.inlet.projection.weight.zero_()
    expected = block(x)[0]
    for placement in ('input','output'):
        w.recall_placement = placement
        with w.attach(model,layer=0):
            actual = block(x)[0]
        assert torch.equal(expected,actual)


def test_identity_removal_cuts_every_query_without_renormalizing_other_values():
    from scripts.ib_local.run_pretrained_gdn_retention import suppress_records,identity
    w = populated()
    learner = SimpleNamespace(memories={1:w})
    x = torch.randn(9,6)
    full = w.retrieve(x,lambda c:c.inputs)
    read = w.last_read
    slot = int(read.slots[0,0])
    value = w.encoder(w.replayed_by_slot[slot].float())
    weight = (read.weights*(read.slots == slot)).sum(-1,keepdim=True)
    expected = full-weight*value
    key = identity(1,slot,int(w.memory.read_credit_version[slot]))
    with suppress_records(learner,{key}):
        actual = w.retrieve(x,lambda c:c.inputs)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)


def test_finite_identity_effect_is_counted_once_per_window_not_once_per_query():
    from scripts.ib_local.run_pretrained_gdn_retention import summarize_reuse
    read = dict(identity='1:0:1',age_tokens=256,removal_benefit=.1,
                preference_before=0.,preference_after=.1)
    rows = [dict(step=0,phase='X',admissions=[dict(reads=[read]*128)],later_rescued_reads=[])]
    result = summarize_reuse(rows)
    assert result['beneficial_old_reads'] == 1
    assert result['repeated_beneficial_old_reads'] == 0
