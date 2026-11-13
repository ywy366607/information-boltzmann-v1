"""Units and actual whole-transition wiring, not a synthetic ability task."""
from types import SimpleNamespace
import copy

import pytest
import torch

from scripts.ib_local.navigation_gain_units import (
    CRITIC_GAIN_SCALE, WOOD_GAIN_SCALE, configure_gain_units, td_gain_target, wood_choice_logits)


def test_repaired_bellman_derivative_is_gamma_in_critic_units():
    e=SimpleNamespace(gamma=.8,gain_units_schema=1)
    value=torch.tensor(25.,dtype=torch.float64,requires_grad=True)
    target=CRITIC_GAIN_SCALE*td_gain_target(e,.03,value)
    assert float(target.detach())==pytest.approx(23.)
    assert float(torch.autograd.grad(target,value)[0])==pytest.approx(.8)
    # Historical execution is preserved explicitly, including its bad units.
    old=SimpleNamespace(gamma=.8)
    historical=CRITIC_GAIN_SCALE*td_gain_target(old,.03,value)
    assert float(torch.autograd.grad(historical,value)[0])==pytest.approx(8.)


def test_gain_unit_conversion_does_not_change_physical_softmax_or_gradient():
    e=SimpleNamespace(gain_units_schema=1,precision=lambda:.02)
    gain=torch.tensor([.01,.03,-.02],dtype=torch.float64,requires_grad=True)
    logits=wood_choice_logits(e,WOOD_GAIN_SCALE*gain)
    reference=.02*CRITIC_GAIN_SCALE*gain
    torch.testing.assert_close(logits,reference)
    torch.testing.assert_close(logits.softmax(-1),reference.softmax(-1))
    assert td_gain_target(SimpleNamespace(gamma=.8,gain_units_schema=1),.05,1000.,False)==.05


def test_actual_joint_event_uses_repaired_target_and_preserves_next_event_restore():
    from test_generative_cycle import configured
    from scripts.ib_local.navigation_checkpoint import snapshot,restore
    torch.set_num_threads(1)
    e=configured();original_model=copy.deepcopy(e.model.state_dict());original_rng=torch.get_rng_state().clone()
    configure_gain_units(e)
    assert torch.equal(original_rng,torch.get_rng_state())
    for k,v in original_model.items():
        if torch.is_tensor(v):assert torch.equal(v,e.model.state_dict()[k])
    e.step(torch.randn(8,4));saved=snapshot(e)
    previous_gain=e.prev['g']
    row=e.step(torch.randn(8,4))
    assert row['td_previous']['y']==pytest.approx(previous_gain+e.gamma*row['value_now']/CRITIC_GAIN_SCALE)
    after=copy.deepcopy(e.model.state_dict())
    fresh=configured();restore(fresh,saved)
    assert fresh.gain_units_schema==1
    again=fresh.step(torch.randn(8,4))
    assert again['choice']==row['choice'] and again['utterance_hex']==row['utterance_hex']
    assert again['td_previous']['y']==row['td_previous']['y']
    assert again['generative_relation']==row['generative_relation']
    for k,v in after.items():
        if torch.is_tensor(v):assert torch.equal(v,fresh.model.state_dict()[k]),k
