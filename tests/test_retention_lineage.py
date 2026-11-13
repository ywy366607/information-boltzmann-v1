"""Numerical tracing/rollback checks; these are not memory capability tasks."""
from types import SimpleNamespace

import pytest
import torch

from scripts.ib_local.pretrained_gdn_experience import LayerExperienceRecall, LocalExperience
from scripts.ib_local.run_pretrained_gdn_retention import (
    admission_trace, identity, real_schedule, snapshots, suppress_records,
)


def wrapper():
    w = LayerExperienceRecall(3,capacity=2,latent_dim=3,
                             consolidation=True,consolidation_rule='retention')
    m = w.memory
    m.clock.fill_(256)
    for index in range(2):
        value = torch.tensor([1., float(index), 0.])
        slot,_ = m.commit_episode(value,value,value,value,w.encoder,source_time=index)
        w.capsules[slot] = LocalExperience(value[None,None].expand(1,65,3).clone(),
                                           None,int(m.read_credit_version[slot]))
    m.clock.fill_(300)
    m.last_write.copy_(torch.tensor([0,10]))
    return w


def test_real_schedule_preserves_exact_revisit_and_real_bridges():
    curriculum = dict(window=128,warm_tokens=128,phase_steps=4,stream=torch.arange(2000))
    sequence,phases,endpoints = real_schedule(curriculum,384,2)
    assert endpoints == [2,4,6]
    assert phases == ['A_continue']*2+['B_fresh']*2+['A_revisit']*2
    assert torch.equal(sequence[:256],curriculum['stream'][384:640])
    assert torch.equal(sequence[256:512],curriculum['stream'][640:896])
    assert torch.equal(sequence[512:768],sequence[:256])
    assert sequence[-1] == 640
    with pytest.raises(ValueError):
        real_schedule(curriculum,385,2)


def test_selected_record_cut_preserves_other_contribution_and_removes_hook():
    w = wrapper()
    learner = SimpleNamespace(memories={1:w})
    query = torch.ones(1,3)
    with torch.no_grad():
        full = w.retrieve(query,lambda c:c.inputs)
        read = w.last_read
        slot = int(read.slots[0,0])
        key = identity(1,slot,int(w.memory.read_credit_version[slot]))
        expected = read.weights[:,1,None]*w.encoder(w.replayed[1].float())
        with suppress_records(learner,{key}):
            cut = w.retrieve(query,lambda c:c.inputs)
        torch.testing.assert_close(cut,expected,atol=1e-6,rtol=1e-6)
        assert 'retrieve' not in w.__dict__
        torch.testing.assert_close(full,w.retrieve(query,lambda c:c.inputs),atol=0,rtol=0)


def test_admission_receipt_uses_settled_credit_and_exact_replaced_version():
    w = wrapper()
    learner = SimpleNamespace(memories={1:w})
    previous = snapshots(learner)
    w.begin(dict(lr=3e-4,betas=(.9,.999),eps=1e-8))
    read = w.retrieve(torch.ones(1,3),lambda c:c.inputs)
    read.square().sum().backward()
    w.memory.settle_read_credit()
    version = w.memory.read_credit_version.clone()
    with admission_trace(learner,previous) as events:
        value = torch.ones(3)
        slot,hit = w.memory.commit_episode(value,value,value,value,w.encoder,source_time=128)
    assert not hit
    assert len(events) == 1
    event = events[0]
    assert event['victim_slot'] == slot
    assert event['victim'] == identity(1,slot,int(version[slot]))
    assert event['new_identity'] == identity(1,slot,int(version[slot])+1)
    assert len(event['reads']) == 2
    assert 'commit_episode' not in w.memory.__dict__


def test_probe_hooks_are_removed_after_exceptions():
    w = wrapper()
    learner = SimpleNamespace(memories={1:w})
    with pytest.raises(RuntimeError):
        with suppress_records(learner,set()):
            raise RuntimeError('abort')
    assert 'retrieve' not in w.__dict__
    with pytest.raises(RuntimeError):
        with admission_trace(learner,snapshots(learner)):
            raise RuntimeError('abort')
    assert 'commit_episode' not in w.memory.__dict__
