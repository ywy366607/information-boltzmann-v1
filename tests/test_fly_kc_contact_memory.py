"""Contact-coordinate memory and credit interfaces, not capability studies."""
import copy
from dataclasses import replace

import pytest
import torch

from test_fly_kc_terminals import setup
from information_boltzmann.core.fly_kc_contact_memory import (
    attach_kc_contact_memory, contact_state, contact_arrival, advance_contact_learning,
    KC_CONTACT_STATE_FIELDS,
)
from information_boltzmann.core.fly_kc_terminals import terminal_state, kc_arrival
from information_boltzmann.core.fly_bptt_learning import advance_compartment_learning, FlyBPTTLearner
from information_boltzmann.core.fly_carried_credit import open_window, apply_carried_credit
from information_boltzmann.core.fly_native_motor_port import checkpoint_model_weights


def local_setup(tmp_path):
    model, state, _ = setup(tmp_path)
    attach_kc_contact_memory(model)
    model.carried_credit = ('comp', 'dopamine')
    state = replace(state, dopamine=torch.full((2, model.n_neurons), .5),
                    credit_dopamine=torch.zeros(3, model.n_neurons))
    return model, state


def event(model, state, releases):
    values = terminal_state(model, state)
    values[3] = torch.cat((state.h.new_tensor([releases]), values[3][:3]), 0)
    result = advance_contact_learning(model, state, state.h, torch.zeros_like(state.h),
        state.ring, torch.cat((torch.full_like(state.h, 2.), torch.full_like(state.h, .5))),
        state.credit_dopamine, state.dpm_trace, values, advance_compartment_learning)
    return replace(state, **dict(zip(KC_CONTACT_STATE_FIELDS, result)))


def test_initial_expansion_preserves_existing_efficacy_and_partner_history(tmp_path):
    model, state = local_setup(tmp_path)
    state = replace(state, comp_memory=torch.tensor([[-.3, .6]]),
                    kc_terminal_packets=torch.tensor([[.2, .8, .4]] * 4))
    memory = contact_state(model, state)[0]
    torch.testing.assert_close(memory, torch.tensor([[-.3, -.3, .6]]))
    old = kc_arrival(model, state, 'comp', weights=model.comp_weight * (1 + state.comp_memory[0]))
    torch.testing.assert_close(contact_arrival(model, state), old)
    assert model.comp_pre.numel() == 2 and model.kcm_site.numel() == 3
    assert 'kc_contact_ids' in checkpoint_model_weights(model)


def test_actual_site_activity_writes_different_phi_without_optimizer(tmp_path):
    model, state = local_setup(tmp_path)
    updated = event(model, state, [.2, .8, .4])
    assert updated.kc_contact_trace[0, 0] < updated.kc_contact_trace[0, 1]
    assert 0 > updated.kc_contact_memory[0, 0] > updated.kc_contact_memory[0, 1]
    torch.testing.assert_close(updated.comp_memory, state.comp_memory, rtol=0, atol=0)
    exact = torch.autograd.grad(updated.kc_contact_memory.sum(), model.comp_logit_trace)[0]
    torch.testing.assert_close(exact, updated.kc_contact_credit[2].sum(), rtol=2e-5, atol=1e-8)
    model.kc_contact_write_enabled = False
    blocked = event(model, state, [.2, .8, .4])
    assert torch.count_nonzero(blocked.kc_contact_memory) == 0
    assert torch.count_nonzero(blocked.kc_contact_trace) > 0


def test_contact_memory_changes_actual_weighted_readout_and_reference_restores(tmp_path):
    model, state = local_setup(tmp_path)
    values = contact_state(model, state)
    values[0] = torch.tensor([[.5, -.2, .3]])
    state = replace(state, **dict(zip(KC_CONTACT_STATE_FIELDS, values)),
                    kc_terminal_packets=torch.tensor([[.2, .8, .4]] * 4))
    before = contact_arrival(model, state)
    model.kc_contact_read_enabled = False
    reference = contact_arrival(model, state)
    assert not torch.allclose(before, reference)
    old = kc_arrival(model, state, 'comp', weights=model.comp_weight * (1 + state.comp_memory[0]))
    torch.testing.assert_close(reference, old)
    model.kc_contact_read_enabled = True
    torch.testing.assert_close(contact_arrival(model, state), before, rtol=0, atol=0)


def test_contact_carried_credit_matches_two_event_conditional_gradient(tmp_path):
    model, state = local_setup(tmp_path)
    other = copy.deepcopy(model)
    full = event(model, event(model, state, [.2, .8, .4]), [.7, .1, .5])
    full.kc_contact_memory.sum().backward()
    first = event(other, copy.deepcopy(state), [.2, .8, .4]).detached()
    opened, leaves, credit = open_window(first, other.carried_credit)
    second = event(other, opened, [.7, .1, .5])
    second.kc_contact_memory.sum().backward()
    apply_carried_credit(other, leaves, credit)
    for name in ('comp_eta', 'comp_logit_trace', 'comp_logit_forget'):
        torch.testing.assert_close(getattr(model, name).grad, getattr(other, name).grad, rtol=3e-5, atol=1e-7)


def test_one_token_window_recomputation_and_local_state_persist(tmp_path):
    model, state = local_setup(tmp_path)
    state = replace(state, h=state.h.index_fill(1, model.kct_index, 3.))
    a = FlyBPTTLearner(model, copy.deepcopy(state), use_checkpointing=False, settle_ticks=3)
    b = FlyBPTTLearner(copy.deepcopy(model), copy.deepcopy(state), use_checkpointing=True, settle_ticks=3)
    a.previous_token = b.previous_token = 0
    p, _ = a.observe([1])
    q, _ = b.observe([1])
    torch.testing.assert_close(torch.tensor(p), torch.tensor(q))
    for name in KC_CONTACT_STATE_FIELDS:
        torch.testing.assert_close(getattr(a.state, name), getattr(b.state, name))
    assert a.state.kc_contact_memory.shape == (1, 3)


def test_contact_id_and_nonempty_history_fail_closed(tmp_path):
    model, state = local_setup(tmp_path)
    saved = checkpoint_model_weights(model)
    saved['kc_contact_ids'] = saved['kc_contact_ids'] + 1
    with pytest.raises(ValueError, match='identity'):
        attach_kc_contact_memory(copy.deepcopy(model), saved=saved)
    with pytest.raises(ValueError, match='Incomplete'):
        contact_state(model, replace(state, kc_contact_trace=torch.ones(1, 3)))
    values = contact_state(model, state)
    values[0] = torch.zeros(1, 2)
    with pytest.raises(ValueError, match='Malformed'):
        contact_state(model, replace(state, **dict(zip(KC_CONTACT_STATE_FIELDS, values))))


def test_polyadic_site_keeps_partner_memory_and_shared_resource_coordinate(tmp_path):
    model, state, _ = setup(tmp_path)
    model.comp_pre = torch.tensor([3, 4, 3])
    model.comp_post = torch.tensor([5, 5, 2])
    model.comp_weight = torch.nn.Parameter(torch.ones(3))
    model.kct_comp_1_site = torch.tensor([0, 1, 2, 0])
    model.kct_comp_1_edge = torch.tensor([0, 0, 1, 2])
    model.kct_comp_1_fraction = torch.tensor([.5, .5, 1., 1.])
    attach_kc_contact_memory(model)
    state = replace(state, comp_memory=torch.tensor([[-.3, .6, .2]]),
                    kc_terminal_eligibility=torch.zeros(1, 3),
                    kc_terminal_eligibility_credit=torch.zeros(1, 3))
    memory = contact_state(model, state)[0]
    # Same physical presynaptic coordinate; distinct existing partner efficacies.
    assert model.kcm_site[0] == model.kcm_site[3] == 0
    assert memory[0, 0] == -.3 and memory[0, 3] == .2
    assert model.kct_site_owner.numel() == 3  # no duplicate vesicle/resource site


def test_write_block_preserves_dopamine_trace_sensitivity(tmp_path):
    from information_boltzmann.core.fly_carried_credit import COMP_PARAMS
    model, state = local_setup(tmp_path)
    model.comp_eta_pot = torch.nn.Parameter(torch.tensor(-3.))
    state = replace(state, comp_da_trace=torch.full_like(state.comp_memory, .2))
    model.kc_contact_write_enabled = False
    updated = event(model, state, [.2, .8, .4])
    assert torch.count_nonzero(updated.kc_contact_memory) == 0
    exact = torch.autograd.grad(updated.kc_contact_da_trace.sum(), model.comp_logit_trace)[0]
    assert exact.abs() > 0
    torch.testing.assert_close(exact, updated.kc_contact_credit[len(COMP_PARAMS)+2].sum(),
                               rtol=2e-5, atol=1e-8)


def test_active_contact_memory_reaches_context_chart_and_action_features(tmp_path):
    from information_boltzmann.core.fly_native_context import compartment_chart, apply_context
    from information_boltzmann.core.fly_consequence_belief import NativeConsequenceCredit
    model, state = local_setup(tmp_path)
    state = replace(state, comp_memory=torch.tensor([[.2, -.1]]))
    values = contact_state(model, state)
    state = replace(state, **dict(zip(KC_CONTACT_STATE_FIELDS, values)))
    chart, _, _ = compartment_chart(model, state)
    model.kc_contact_memory_enabled = False
    legacy_chart, _, _ = compartment_chart(model, state)
    model.kc_contact_memory_enabled = True
    torch.testing.assert_close(chart, legacy_chart)
    credit = object.__new__(NativeConsequenceCredit)
    credit.model = model
    credit.posts, credit.groups = torch.unique(model.comp_post, return_inverse=True)
    credit.counts = torch.bincount(credit.groups).to(state.h)
    credit.thresholds = torch.ones_like(state.h)
    before, before_kc = credit.features(state)
    changed = apply_context(model, state, chart+.3)
    torch.testing.assert_close(compartment_chart(model, changed)[0], chart+.3)
    torch.testing.assert_close(changed.comp_memory, state.comp_memory, rtol=0, atol=0)
    after, after_kc = credit.features(changed)
    assert not torch.allclose(before, after)
    torch.testing.assert_close(before_kc, after_kc)  # no invented sensory outcome
    model.kc_contact_read_enabled = False
    torch.testing.assert_close(compartment_chart(model, changed)[0], legacy_chart)
    blocked, _ = credit.features(changed)
    torch.testing.assert_close(blocked, before)


def test_first_context_materializes_complete_contact_history(tmp_path):
    from information_boltzmann.core.fly_native_context import compartment_chart, apply_context
    model, state = local_setup(tmp_path)
    center = compartment_chart(model, state)[0]
    changed = apply_context(model, state, center+.1)
    values = contact_state(model, changed)
    assert values[0].shape == (1, 3)
    assert values[1].shape == (1, 3)
    torch.testing.assert_close(compartment_chart(model, changed)[0], center+.1)
    assert not state.kc_contact_memory.numel()
