"""Layer-boundary placement, local credit and checkpoint compatibility."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same
from tests.test_pretrained_gdn_experience import populated


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(6,6,bias=False)
        self.last_inputs=None

    def forward(self, hidden_states, **kwargs):
        self.last_inputs=hidden_states
        return self.projection(hidden_states).tanh(), None, None


def setup():
    w=populated()
    w.recall_placement='output'
    block=Block()
    model=SimpleNamespace(model=SimpleNamespace(layers=[block]))
    with torch.no_grad():
        w.inlet.projection.weight.fill_(.05)
    return w,block,model,torch.randn(1,128,6)


def test_output_recall_preserves_native_input_and_delivers_only_to_causal_suffix():
    w,block,model,x=setup()
    original=block(x)[0]
    with w.attach(model,layer=0,read_position=90):
        result=block(x)[0]
    assert torch.equal(block.last_inputs,x)
    assert torch.equal(w.current['inputs'],x)
    assert torch.equal(result[:,:90],original[:,:90])
    torch.testing.assert_close(result[:,90:],original[:,90:]+w.current['injection'],atol=0,rtol=0)
    assert not torch.equal(result[:,90:],original[:,90:])
    assert w.current['response'] is result
    changed=x.clone(); changed[:,91:]+=5
    with w.attach(model,layer=0,read_position=90):
        after=block(changed)[0]
    assert torch.equal(result[:,:91],after[:,:91])


def test_output_replay_preserves_held_forward_and_adds_old_native_credit():
    w,block,model,x=setup()
    outputs=[]
    for held in (False,True):
        w.detach_replay=held
        with w.attach(model,layer=0):
            y=block(x)[0]
        outputs.append(y.detach().clone())
        # Direct local contribution held fixed by subtracting the native path.
        old_response=y-block.projection(x).tanh()
        gradient=torch.autograd.grad(old_response.square().sum(),block.projection.weight)[0]
        if not held:
            assert gradient.norm()>0
        else:
            assert gradient.norm()==0
    assert torch.equal(*outputs)


def test_zero_inlet_restores_original_block_in_either_placement():
    w,block,model,x=setup()
    with torch.no_grad():
        w.inlet.projection.weight.zero_()
    original=block(x)[0]
    for mode in ('input','output'):
        w.recall_placement=mode
        with w.attach(model,layer=0):
            actual=block(x)[0]
        assert torch.equal(actual,original)


def test_placement_roundtrip_and_legacy_restore_are_explicit():
    w,block,model,x=setup()
    state=copy.deepcopy(w.state_dict())
    other=populated(); other.load_state_dict(state)
    assert other.recall_placement=='output'
    assert_same(state,other.state_dict())
    del state['_extra_state']['recall_placement']
    other.load_state_dict(state)
    assert other.recall_placement=='input'
    learner=PretrainedExperienceLearner.__new__(PretrainedExperienceLearner)
    learner.memories={0:other}
    learner.set_recall_placement('output')
    assert other.recall_placement=='output'
    other.current={}
    with pytest.raises(RuntimeError,match='finish'):
        learner.set_recall_placement('input')
