"""Production transition and role scopes; numerical contracts, not capabilities."""
import copy

import pytest
import torch

from scripts.ib_local.question_language_port import (
    configure_content_question, observation_scope)
from test_joint_world_cycle import complete_interface_engine


def configured():
    torch.set_num_threads(1)
    e = complete_interface_engine()
    configure_content_question(e)
    return e


def test_migration_preserves_old_parameters_rng_optimizer_and_zero_speech():
    e = complete_interface_engine()
    before = copy.deepcopy(e.model.state_dict())
    opt = copy.deepcopy(e.optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    ids = torch.tensor([[32, 65, 66]])
    lp = torch.full((1, 3, 256), -5.54517744)
    e.model._question_context = torch.ones(8)
    old = e.model.predict_observed(ids, e.recurrent, lp)
    configure_content_question(e)
    assert torch.equal(rng, torch.get_rng_state())
    new = e.model.predict_observed(ids, e.recurrent, lp)
    assert torch.equal(old[0], new[0]) and torch.equal(old[1], new[1])
    for name, value in before.items():
        if torch.is_tensor(value):
            assert torch.equal(value, e.model.state_dict()[name])
    assert e.optimizer.state_dict()['param_groups'][:-1] == opt['param_groups']
    assert len(e.optimizer.state_dict()['state']) == len(opt['state'])
    assert id(e.model.question_to_hidden.weight) in {id(p) for p in e.params}


def test_question_changes_content_and_recurrence_not_just_constant_odds():
    e = configured(); m = e.model
    with torch.no_grad():
        m.question_to_hidden.weight.normal_(std=.1)
    m._question_context = torch.randn(8)
    lp = torch.full((1, 2, 256), -5.54517744)
    shifts = []
    for ids in (torch.tensor([[32,65]]),torch.tensor([[65,32]])):
        yes = m.predict_observed(ids,e.recurrent,lp)
        q = m._question_context; m._question_context = None
        no = m.predict_observed(ids,e.recurrent,lp)
        m._question_context = q
        assert (yes[1]-no[1]).norm() > 0
        shifts.append(yes[0]-no[0])
    assert not torch.allclose(shifts[0],shifts[1])


@pytest.mark.parametrize('own', [0, 2])
def test_scoped_self_hearing_matches_exact_draft_and_world_gets_no_direct_bias(own):
    e = configured(); m=e.model
    with torch.no_grad():
        m.question_to_hidden.weight.normal_(std=.03)
        m.question_to_logits.weight.normal_(std=.1)
    m._question_context = torch.randn(8)
    ids = torch.tensor([[32,65,66,67,68]])
    lp = torch.full((1,5,256),-5.54517744)
    state=e.recurrent
    for i in range(own+1):
        draft=m.predict_observed(ids[:,i:i+1],state,lp[:,i:i+1])
        state=draft[1]
    with observation_scope(m,own,5):
        full=m.predict_observed(ids,e.recurrent,lp)
        # Same prefix alone with the correctly sliced scope.
        with observation_scope(m,own,own+1):
            prefix=m.predict_observed(ids[:,:own+1],e.recurrent,lp[:,:own+1])
        torch.testing.assert_close(prefix[1],state,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(prefix[3][-1],draft[3][-1],atol=2e-6,rtol=2e-5)
        direct=m.question_to_logits.weight.detach().clone()
        with torch.no_grad():m.question_to_logits.weight.add_(5)
        changed=m.predict_observed(ids,e.recurrent,lp)
        torch.testing.assert_close(full[0][:,own:],changed[0][:,own:],atol=0,rtol=0)
        with torch.no_grad():m.question_to_logits.weight.copy_(direct)
    assert m._question_input_mask is None and m._question_output_mask is None


def test_real_response_loss_reaches_question_transition_but_not_direct_byte_bias():
    e=configured();m=e.model
    m._question_context=torch.randn(8)
    ids=torch.tensor([[32,65,66,67,68]])
    lp=torch.full((1,5,256),-5.54517744)
    with observation_scope(m,2,5):
        logits=m.predict_observed(ids,e.recurrent,lp)[0]
        torch.nn.functional.cross_entropy(logits[:,2:].flatten(0,1),ids[:,2:].flatten()).backward()
    assert m.question_to_hidden.weight.grad.norm()>0
    assert m.question_to_logits.weight.grad.abs().max()==0


def test_complete_cycle_keeps_policy_replay_self_hearing_and_new_learning_together():
    e=configured()
    committed=[]
    for _ in range(2):
        row=e.step(torch.randn(8,4))
        assert row['question_context']['conditional_transition']['scope_cleared']
        if row['question_context']['context_norm'] > 0:
            assert row['question_context']['conditional_transition']['parameter_step']>0
        else:
            # The empty interface fixture has no question evidence at birth.
            assert row['question_context']['conditional_transition']['parameter_step']==0
        assert row['policy_replay_max_error']<1e-3
        assert row['joint_language_search']['replay_max_error']<1e-3
        assert row['draft_actual_action_l2']<1e-3
        committed.append(row['causal_program']['evidence_committed'])
    assert any(committed)  # Repeated fixture facts may merge on the next turn.


def test_scope_is_exception_safe_and_rejects_wrong_lengths():
    e=configured()
    with pytest.raises(RuntimeError):
        with observation_scope(e.model,2,4):raise RuntimeError('interrupted')
    assert e.model._question_input_mask is None
    with pytest.raises(ValueError):
        with observation_scope(e.model,4,4):pass
