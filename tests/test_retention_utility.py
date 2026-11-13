"""Finite-use evidence, bounded capacity and legacy restore invariants."""
import copy
import math

import pytest
import torch

from scripts.ib_local.experience_consolidation import (
    bounded_preference, consolidates_reads, replacement_slot, settle_retention_utility,
)
from scripts.ib_local.latent_navigation_v3 import ExperienceMemory
from scripts.ib_local.navigation_read_credit import configure_read_credit
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same


def memory():
    m=ExperienceMemory(3,capacity=2,topk=2,mode='fifo')
    m.consolidation_enabled,m.consolidation_rule=True,'utility'
    configure_read_credit(m,'adam',dict(lr=3e-4,betas=(.9,.999),eps=1e-8))
    m.occupied.fill_(True)
    m.clock.fill_(256)
    m.last_write.copy_(torch.tensor([0,20]))
    return m


def item(m,slot,value):
    return dict(slot=slot,version=int(m.read_credit_version[slot]),benefit=value)


def test_usefulness_can_strengthen_retention_while_read_dose_is_reduced():
    m=memory()
    m.read_credit_first.fill_(-1)
    m.read_credit_second.fill_(1)
    m.read_credit_steps.fill_(1)
    old=[m.read_credit_first.clone(),m.read_credit_second.clone(),m.read_credit_mean.clone()]
    assert replacement_slot(m)==0 and not consolidates_reads(m)
    settle_retention_utility(m,[item(m,0,.01),item(m,1,-.01)])
    assert bounded_preference(m)[0]>0>bounded_preference(m)[1]
    assert replacement_slot(m)==1
    for before,after in zip(old,(m.read_credit_first,m.read_credit_second,m.read_credit_mean)):
        assert torch.equal(before,after)


def test_opposing_finite_evidence_releases_and_cannot_refresh_write_age():
    m=memory()
    settle_retention_utility(m,[item(m,0,.1)])
    for _ in range(4):
        settle_retention_utility(m,[item(m,0,-.1)])
    assert bounded_preference(m)[0]<0
    m.last_touch.fill_(100000)
    assert replacement_slot(m)==0
    multipliers=bounded_preference(m).exp()
    assert float(multipliers.max())<=math.e+1e-6
    assert float(multipliers.min())>=1/math.e-1e-6


def test_stale_identity_is_rejected_and_legacy_restore_resets_new_fields():
    m=memory()
    stale=item(m,0,1.)
    m.read_credit_version[0].add_(1)
    assert settle_retention_utility(m,[stale])==dict(records=0,stale=1)
    before=copy.deepcopy(m.state_dict())
    for key in list(before):
        if key.startswith('retention_utility_'):
            del before[key]
    m.retention_utility_first.fill_(99)
    m.load_state_dict(before)
    assert not bool(m.retention_utility_first.any())
    assert not bool(m.retention_utility_second.any())
    assert not bool(m.retention_utility_steps.any())


def test_utility_state_restores_exactly_and_adds_only_16_bytes_per_slot():
    m=memory()
    settle_retention_utility(m,[item(m,0,.1),item(m,1,-.2)])
    state=copy.deepcopy(m.state_dict())
    other=memory(); other.load_state_dict(state)
    assert_same(state,other.state_dict())
    assert sum(v.numel()*v.element_size() for k,v in state.items()
               if k.startswith('retention_utility_'))==16*m.capacity


@pytest.mark.parametrize('value',[float('nan'),1e30])
def test_invalid_or_overflowing_utility_cannot_corrupt_persistent_moments(value):
    m=memory()
    before=copy.deepcopy(m.state_dict())
    with pytest.raises(FloatingPointError):
        settle_retention_utility(m,[item(m,0,value)])
    assert_same(before,m.state_dict())


def test_replacing_or_merging_record_clears_utility_of_old_identity():
    m=memory()
    m.occupied.zero_()
    m.clock.fill_(1024)
    encoder=torch.nn.Identity()
    x=torch.tensor([1.,0.,0.])
    slot,_=m.commit_episode(x,x,x,x,encoder,source_time=0)
    settle_retention_utility(m,[item(m,slot,.1)])
    m.occupied[1]=True
    m.last_write[1]=256
    m.retention_utility_first[slot]=-1
    m.clock.fill_(1536)
    replaced,_=m.commit_episode(-x,-x,-x,-x,encoder,source_time=128)
    assert replaced==slot
    assert m.retention_utility_steps[slot]==m.retention_utility_first[slot]==0
    m.mode='merge'
    settle_retention_utility(m,[item(m,slot,.1)])
    merged,hit=m.commit_episode(-x,-x,-x,-x,encoder,source_time=256)
    assert hit and merged==slot
    assert m.retention_utility_steps[slot]==m.retention_utility_first[slot]==0
