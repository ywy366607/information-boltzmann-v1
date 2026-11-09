"""Exact diagnostic interventions, not synthetic memory capability tests."""
import copy

import pytest
import torch

from scripts.ib_local.pretrained_gdn_experience import LayerExperienceRecall, LocalExperience
from scripts.ib_local.probe_pretrained_gdn_retained_values import retained_value_probe, summarize
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same


def stored():
    w = LayerExperienceRecall(3,capacity=3,latent_dim=3,topk=2,
                             consolidation=True,consolidation_rule='retention')
    with torch.no_grad():
        w.encoder.weight.copy_(torch.eye(3))
        w.encoder.bias.zero_()
        w.value_map.weight.copy_(torch.eye(3))
        w.value_map.bias.zero_()
    w.memory.clock.fill_(256)
    for i,(key,value) in enumerate(zip(([1.,0.,0.],[.9,.1,0.],[0.,0.,1.]),
                                     ([2.,1.,0.],[-1.,0.,2.],[1.,3.,4.]))):
        key,value = torch.tensor(key),torch.tensor(value)
        slot,_ = w.memory.commit_episode(key,value,value,value,w.encoder,source_time=i)
        w.capsules[slot] = LocalExperience(value[None,None].expand(1,65,3).clone(),None,
                                          int(w.memory.read_credit_version[slot]))
    return w


def test_swap_and_vacancy_preserve_actual_mixture_weight_and_other_contributions():
    w = stored()
    query = torch.tensor([[1.,0.,0.]])
    version = int(w.memory.read_credit_version[2])
    before = copy.deepcopy(w.state_dict())
    with torch.no_grad():
        full = w.retrieve(query,lambda c:c.inputs)
        weights = w.last_read.weights.clone()
        original = w.value_map(w.replayed[1])
        candidate = w.value_map(w.capsules[2].inputs[:,-1])
        with retained_value_probe(w,2,version,'drop') as dropped:
            drop = w.retrieve(query,lambda c:c.inputs)
        with retained_value_probe(w,2,version,'swap') as swapped:
            swap = w.retrieve(query,lambda c:c.inputs)
    torch.testing.assert_close(drop,full-weights[:,1,None]*original,atol=0,rtol=0)
    torch.testing.assert_close(swap,full+weights[:,1,None]*(candidate-original),atol=0,rtol=0)
    assert dropped == swapped
    assert swapped[0]['candidate_rank'] == 3
    assert not swapped[0]['already_selected']
    assert 'retrieve' not in w.__dict__
    assert_same(before,w.state_dict())


def test_swapping_in_already_selected_record_is_exact_identity():
    w = stored()
    query = torch.tensor([[1.,0.,0.]])
    with torch.no_grad():
        full = w.retrieve(query,lambda c:c.inputs)
        slot = int(w.last_read.slots[0,0])
        with retained_value_probe(w,slot,int(w.memory.read_credit_version[slot]),'swap') as trace:
            other = w.retrieve(query,lambda c:c.inputs)
    torch.testing.assert_close(full,other,atol=0,rtol=0)
    assert trace[0]['already_selected']


def test_invalid_identity_rejected_and_probe_restores_after_abort():
    w = stored()
    with pytest.raises(ValueError,match='replaced'):
        with retained_value_probe(w,2,999,'swap'):
            pass
    assert 'retrieve' not in w.__dict__
    with pytest.raises(RuntimeError):
        with retained_value_probe(w,2,int(w.memory.read_credit_version[2]),'drop'):
            raise RuntimeError('abort')
    assert 'retrieve' not in w.__dict__


def test_removing_bad_value_is_not_mistaken_for_useful_replacement():
    row = dict(already_selected=False,candidate_rank=3,swap_minus_intact=-.1,
               swap_minus_drop=.1,drop_minus_intact=-.2)
    assert summarize([row])['improves_both'] == 0
    row['swap_minus_drop'] = -.1
    assert summarize([row])['improves_both'] == 1
