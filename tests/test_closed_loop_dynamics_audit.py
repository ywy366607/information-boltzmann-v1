"""Numerical/interface tests only; no synthetic capability experiment."""
import copy
from types import SimpleNamespace
import numpy as np
import torch

from scripts.audit_closed_loop_dynamics import pulse, replay_take


def test_pulse_changes_only_declared_coordinates_and_preserves_capacity():
    first=SimpleNamespace(matrix_size=8*32*32,heads=8,dk=32,dv=32)
    g=torch.Generator().manual_seed(7)
    state=torch.randn(32,1,first.matrix_size+12,generator=g)
    old=state.clone()
    receipts=pulse(state,first,3)
    assert len(receipts)==2
    torch.testing.assert_close(state[...,first.matrix_size:],old[...,first.matrix_size:])
    for layer,head in ((7,0),(23,1)):
        a=state[layer,...,:first.matrix_size].reshape(8,32,32)[head]
        b=old[layer,...,:first.matrix_size].reshape(8,32,32)[head]
        torch.testing.assert_close(a.norm(),b.norm())
        torch.testing.assert_close(a.abs().sum(1),b.abs().sum(1))
        torch.testing.assert_close(a.abs().sum(0).sort().values,b.abs().sum(0).sort().values)
        assert not torch.equal(a,b)
    untouched=[i for i in range(32) if i not in (7,23)]
    assert torch.equal(state[untouched],old[untouched])
    pulse(state,first,3)
    assert torch.equal(state,old)


def test_yoke_replays_observations_and_only_its_world_boundary():
    rng=np.random.default_rng(2)
    base=SimpleNamespace(cursor=2,anchor=0,offset=0,moves=0,rng=rng,seen=np.zeros(20,bool))
    world=SimpleNamespace(base=base)
    boundary=dict(cursor=9,anchor=5,offset=4,moves=1,rng=copy.deepcopy(rng.bit_generator.state))
    record=dict(response='6162',holdout='6364',world_after=boundary,
                info=dict(start=5,child_text='old',provider_calls=0))
    a,b,info=replay_take(world,record,b'new',2,2)
    assert bytes(a)==b'ab' and bytes(b)==b'cd'
    assert info['child_text']=='new' and record['info']['child_text']=='old'
    assert base.seen.tolist()==[False]*5+[True]*4+[False]*11
    assert (base.cursor,base.anchor,base.offset,base.moves)==(9,5,4,1)
