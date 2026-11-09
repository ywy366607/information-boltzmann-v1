"""Zero-value choice, gradients and old-optimizer migration, not a capability task."""
import copy
import math

import pytest
import torch

from scripts.ib_local.pretrained_gdn_experience import LayerExperienceRecall, LocalExperience
from scripts.ib_local.pretrained_gdn_learning import with_null_optimizer_group
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same


def populated():
    w = LayerExperienceRecall(3,capacity=2,latent_dim=3,consolidation=True,
                             consolidation_rule='retention')
    with torch.no_grad():
        w.encoder.weight.copy_(torch.eye(3))
        w.encoder.bias.zero_()
    w.memory.clock.fill_(256)
    for i,value in enumerate((1.,2.)):
        key = torch.tensor([1.,0.,0.])
        result = key*value
        slot,_ = w.memory.commit_episode(key,result,result,result,w.encoder,source_time=i)
        w.capsules[slot] = LocalExperience(result[None,None].expand(1,65,3).clone(),None,
                                          int(w.memory.read_credit_version[slot]))
    return w


def test_null_candidate_takes_mass_without_changing_real_slot_identity():
    w = populated()
    q = torch.tensor([[1.,0.,0.]])
    legacy = w.retrieve(q,lambda c:c.inputs)
    original_slots = w.last_read.slots.clone()
    w.enable_null_read()
    actual = w.retrieve(q,lambda c:c.inputs)
    assert torch.equal(w.last_read.slots,original_slots)
    torch.testing.assert_close(w.last_read.weights,torch.full((1,2),1/3))
    torch.testing.assert_close(actual,legacy*(2/3))
    assert len(w.capsules) == 2
    w.null_read_enabled = False
    torch.testing.assert_close(w.retrieve(q,lambda c:c.inputs),legacy,atol=0,rtol=0)


def test_all_real_candidates_can_receive_negative_credit_against_zero():
    w = populated()
    w.enable_null_read()
    with torch.no_grad():
        w.null_bias.fill_(math.log(4))
    w.begin(dict(lr=3e-4,betas=(.9,.999),eps=1e-8))
    value = w.retrieve(torch.tensor([[1.,0.,0.]]),lambda c:c.inputs)
    value.square().sum().backward()
    assert w.null_bias.grad < 0  # descent raises null probability
    settled = w.memory.settle_read_credit()
    assert bool((w.memory.read_credit_mean < 0).all())
    assert settled['credit_sum'] == pytest.approx(float(w.null_bias.grad),abs=1e-6)
    assert settled['credit_sum'] < -.1


def test_zero_value_is_zero_even_with_a_biased_encoder():
    w = populated()
    w.enable_null_read()
    with torch.no_grad():
        w.encoder.bias.fill_(2)
        w.null_bias.fill_(1000)
    value = w.retrieve(torch.ones(1,3),lambda c:c.inputs)
    assert value.abs().max() == 0
    assert w.last_read.weights.sum() == 0


def test_one_real_candidate_still_gets_credit_with_a_null_option():
    w = populated()
    w.memory.topk = 1
    w.enable_null_read()
    w.begin(dict(lr=3e-4,betas=(.9,.999),eps=1e-8))
    value = w.retrieve(torch.tensor([[1.,0.,0.]]),lambda c:c.inputs)
    value.square().sum().backward()
    result = w.memory.settle_read_credit()
    assert result['credit_sum'] < 0


def test_empty_memory_is_finite_and_uses_no_real_record():
    w = LayerExperienceRecall(3,capacity=2,latent_dim=3)
    w.enable_null_read()
    value = w.retrieve(torch.ones(1,3),lambda c:c.inputs)
    assert torch.isfinite(value).all() and value.abs().max() == 0
    assert w.last_read.weights.sum() == 0


def test_null_parameters_gradients_match_held_input_finite_difference():
    w = populated()
    w.enable_null_read()
    q = torch.tensor([[.8,.1,0.]])
    value = lambda:w.retrieve(q,lambda c:c.inputs).square().sum()
    gradient, = torch.autograd.grad(value(),w.null_bias)
    with torch.no_grad():
        w.null_bias.add_(1e-3)
        plus = value().item()
        w.null_bias.sub_(2e-3)
        minus = value().item()
        w.null_bias.add_(1e-3)
    assert float(gradient) == pytest.approx((plus-minus)/2e-3,rel=2e-3,abs=1e-4)


def test_null_restore_and_legacy_load_reset_only_new_fields():
    w = populated()
    legacy = copy.deepcopy(w.state_dict())
    del legacy['null_key'],legacy['null_bias']
    del legacy['_extra_state']['null_read_enabled']
    w.enable_null_read()
    with torch.no_grad():
        w.null_bias.fill_(2.)
    current = copy.deepcopy(w.state_dict())
    other = populated()
    other.load_state_dict(current)
    assert_same(current,other.state_dict())
    other.load_state_dict(legacy)
    assert not other.null_read_enabled and other.null_bias == 0
    torch.testing.assert_close(other.null_key,torch.full((3,),3**-.5),atol=0,rtol=0)
    q = torch.ones(1,3)
    w.null_read_enabled = False
    torch.testing.assert_close(other.retrieve(q,lambda c:c.inputs),w.retrieve(q,lambda c:c.inputs),atol=0,rtol=0)


def test_old_adam_moments_keep_their_parameter_mapping_after_null_group_append():
    a,b = torch.nn.Parameter(torch.tensor(1.)),torch.nn.Parameter(torch.tensor(2.))
    old = torch.optim.AdamW([dict(params=[a],lr=1e-4),dict(params=[b],lr=3e-4)],weight_decay=0)
    (a.square()+3*b.square()).backward(); old.step()
    stored = copy.deepcopy(old.state_dict())
    aa,bb = torch.nn.Parameter(a.detach().clone()),torch.nn.Parameter(b.detach().clone())
    n = torch.nn.Parameter(torch.zeros(3))
    new = torch.optim.AdamW([dict(params=[aa],lr=1e-4),dict(params=[bb],lr=3e-4),
                            dict(params=[n],lr=3e-4)],weight_decay=0)
    migrated = with_null_optimizer_group(stored,new.state_dict(),1)
    new.load_state_dict(migrated)
    assert_same(old.state[a],new.state[aa])
    assert_same(old.state[b],new.state[bb])
    assert not new.state[n]
    assert_same(stored,old.state_dict())
