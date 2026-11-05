"""Numerical contracts for local S plasticity, not language-capability tests."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from information_boltzmann.core.cgb_column_body import CGBColumnLanguageBody


def _body(*, double=True):
    torch.manual_seed(19)
    body = CGBColumnLanguageBody(4, (2, 3), active_columns=2, contact_rank=2)
    with torch.no_grad():
        body.reciprocal_contact.mul_(0.2)
    return body.double() if double else body


def _state(body, batch=2):
    state = body.initial_state(batch)
    values = torch.linspace(0.2, 0.8, state.participation.numel(), dtype=state.activity.dtype)
    occupied = values.reshape_as(state.participation)
    occupied = occupied / occupied.sum(dim=(-1, -2), keepdim=True) * 0.4
    return replace(state, participation=occupied)


@torch.no_grad()
def _pair(body):
    before = _state(body)
    source = torch.tensor([0, 1], dtype=torch.long)
    first = body.step(before, torch.tensor([31, 47]), elapsed=0.0, source=(source == 0),
                      record_contact_return=True)
    target = torch.tensor([73, 89])
    current = body.step(first.state, target, elapsed=0.0, source=(source == 0),
                        record_contact_return=True)
    index, _, gate, unique = body._pack_local_receipt(
        first.contact_pre, first.contact_response_gate, first.route,
    )
    teaching = body._s_contact_return_from_tensors(
        current.logits, current.read_route, index, gate, unique,
        first.edge_flux, first.contact_return_weight, target,
    )
    return before, first, current, target, source, teaching


@pytest.mark.parametrize("flow_only", [False, True])
def test_one_spatial_s_return_matches_score_finite_difference(flow_only):
    body = _body()
    if flow_only:
        with torch.no_grad():
            body.reciprocal_contact[body._contact_h_output, body._contact_s_input].zero_()
    before, first, current, target, source, teaching = _pair(body)
    index, _, gate, unique = body._pack_local_receipt(
        first.contact_pre, first.contact_response_gate, first.route,
    )
    assert (~unique).any()  # Periodic neighbours must not duplicate the return.
    assert teaching.abs().sum() > 0
    direction = torch.linspace(-0.3, 0.7, before.participation.numel(), dtype=torch.float64)
    direction = direction.reshape_as(before.participation)

    @torch.no_grad()
    def score(offset):
        altered = replace(before, participation=before.participation + offset * direction)
        emitted = body.step(altered, torch.tensor([31, 47]), elapsed=0.0, source=(source == 0))
        # Only forward evaluation is used; no graph is unrolled through time.
        following = body.step(emitted.state, target, elapsed=0.0, source=(source == 0))
        return F.log_softmax(following.logits, dim=-1).gather(1, target[:, None]).sum()

    h = 1e-5
    finite_difference = (score(h) - score(-h)) / (2 * h)
    torch.testing.assert_close((teaching * direction).sum(), finite_difference,
                               rtol=2e-5, atol=1e-8)
    zero = body._s_contact_return_from_tensors(
        current.logits, current.read_route, index, gate, unique,
        first.edge_flux, first.contact_return_weight, target,
        torch.zeros(2, dtype=torch.bool),
    )
    assert torch.count_nonzero(zero) == 0


def test_optional_return_receipt_is_an_independent_snapshot_not_a_forward_switch():
    body = _body()
    state = _state(body)
    token = torch.tensor([31, 47])
    plain = body.step(state, token, elapsed=0.0)
    recorded = body.step(state, token, elapsed=0.0, record_contact_return=True)
    assert plain.contact_return_weight is None
    torch.testing.assert_close(plain.logits, recorded.logits, rtol=0.0, atol=0.0)
    for name in state.__dataclass_fields__:
        torch.testing.assert_close(getattr(plain.state, name), getattr(recorded.state, name),
                                   rtol=0.0, atol=0.0)
    snapshot = recorded.contact_return_weight.clone()
    with torch.no_grad():
        body.reciprocal_contact[:body.d_model + 2, body._contact_s_input].add_(0.5)
    torch.testing.assert_close(recorded.contact_return_weight, snapshot, rtol=0.0, atol=0.0)
    assert recorded.contact_return_weight.grad_fn is None


def test_s_return_uses_the_issued_weights_after_current_weights_change():
    body = _body()
    _, first, current, target, _, teaching = _pair(body)
    index, _, gate, unique = body._pack_local_receipt(
        first.contact_pre, first.contact_response_gate, first.route,
    )
    with torch.no_grad():
        body.reciprocal_contact[:body.d_model + 2, body._contact_s_input].add_(3.0)
    actual = body._s_contact_return_from_tensors(
        current.logits, current.read_route, index, gate, unique,
        first.edge_flux, first.contact_return_weight, target,
    )
    torch.testing.assert_close(actual, teaching, rtol=0.0, atol=0.0)


def test_same_pool_instantaneous_vjp_matches_only_the_frozen_local_potential():
    body = _body()
    _, previous, current, _, source, teaching = _pair(body)
    state = previous.state
    actual = body._s_contact_direction_from_tensors(
        state.participation, state.precision, current.contact_pre,
        current.contact_response_gate, current.route, source, teaching,
    )
    weight = body.reciprocal_contact.detach().clone().requires_grad_()
    cotangent = (teaching * body.trace(state)).detach()

    def potential(w):
        changed = body._reciprocal_contact_update(
            state.participation.detach(), F.linear(current.contact_pre.detach(), w),
            current.contact_response_gate.detach(), current.route.detach(), source,
        )
        return (cotangent * changed).sum()

    expected, = torch.autograd.grad(potential(weight), weight)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
    start = body.d_model + 2
    assert torch.count_nonzero(actual[:start]) == 0
    allocation_stop = start + 4 * body.contact_rank
    assert actual[start:allocation_stop].abs().sum() > 0
    assert actual[allocation_stop:].abs().sum() > 0
    perturbation = torch.linspace(-0.2, 0.3, weight.numel(), dtype=weight.dtype).reshape_as(weight)
    h = 1e-5
    with torch.no_grad():
        finite_difference = (potential(weight + h * perturbation)
                             - potential(weight - h * perturbation)) / (2 * h)
    torch.testing.assert_close((actual * perturbation).sum(), finite_difference,
                               rtol=2e-5, atol=1e-9)
    assert actual.grad_fn is None


def test_s_update_changes_only_s_rows_and_cannot_rewrite_the_issued_prediction():
    body = _body()
    _, previous, current, target, source, teaching = _pair(body)
    direction = body._s_contact_direction_from_tensors(
        previous.state.participation, previous.state.precision, current.contact_pre,
        current.contact_response_gate, current.route, source, teaching,
    )
    parameters = {name: parameter.detach().clone() for name, parameter in body.named_parameters()}
    issued = current.logits.clone()
    resulting_state = body.clone_state(current.state)
    body.update_s_contact_from_following(previous, current, previous.state, target,
                                        step_size=0.05, source=(source == 0))
    expected = parameters["reciprocal_contact"].clone()
    start = body.d_model + 2
    expected[start:] += 0.05 * direction[start:]
    for name, parameter in body.named_parameters():
        torch.testing.assert_close(parameter, expected if name == "reciprocal_contact" else parameters[name])
        assert parameter.grad is None
    torch.testing.assert_close(current.logits, issued, rtol=0.0, atol=0.0)
    for name in resulting_state.__dataclass_fields__:
        torch.testing.assert_close(getattr(current.state, name), getattr(resulting_state, name),
                                   rtol=0.0, atol=0.0)
        assert getattr(current.state, name).grad_fn is None


def test_language_receipt_redistributes_only_existing_s_with_a_conserved_free_partner():
    body = _body()
    _, previous, current, target, _, teaching = _pair(body)
    parameters = {name: parameter.detach().clone() for name, parameter in body.named_parameters()}
    issued = current.logits.clone()
    updated = body.update_s_participation_from_following(
        previous, current, target, step_size=0.05,
    )

    # The delayed third factor changes only S.  H/C and every common learned
    # contact parameter retain their issued values.
    for name, parameter in body.named_parameters():
        torch.testing.assert_close(parameter, parameters[name], rtol=0.0, atol=0.0)
    torch.testing.assert_close(updated.activity, current.state.activity, rtol=0.0, atol=0.0)
    torch.testing.assert_close(updated.natural, current.state.natural, rtol=0.0, atol=0.0)
    torch.testing.assert_close(updated.precision, current.state.precision, rtol=0.0, atol=0.0)
    torch.testing.assert_close(updated.causal_token, current.state.causal_token, rtol=0.0, atol=0.0)
    torch.testing.assert_close(current.logits, issued, rtol=0.0, atol=0.0)

    occupied = current.state.participation.flatten(start_dim=-2)
    after = updated.participation.flatten(start_dim=-2)
    returned = teaching.flatten(start_dim=-2)
    positive = returned.clamp_min(0.0)
    admit = -torch.expm1(-0.05 * positive.sum(dim=-1))
    release = -torch.expm1(-0.05 * (-returned).clamp_min(0.0))
    expected_free = ((1.0 - admit) * (1.0 - occupied.sum(dim=-1))
                     + (release * occupied).sum(dim=-1))
    assert torch.all(after >= 0.0)
    assert torch.all(after.sum(dim=-1) <= 1.0)
    torch.testing.assert_close(1.0 - after.sum(dim=-1), expected_free, rtol=1e-10, atol=1e-10)
    assert updated.participation.grad_fn is None


def test_zero_language_receipt_holds_the_complete_s_bundle():
    body = _body()
    state = _state(body)
    unchanged = body.evidence_gated_s_exchange(
        state.participation, torch.zeros_like(state.participation), 0.2,
    )
    torch.testing.assert_close(unchanged, state.participation, rtol=0.0, atol=0.0)


def test_muting_both_actual_s_read_blocks_closes_language_return():
    body = _body()
    body.mute_contact_readout_()
    _, previous, current, _, source, teaching = _pair(body)
    assert previous.state.participation.abs().sum() > 0
    assert torch.count_nonzero(teaching) == 0
    direction = body._s_contact_direction_from_tensors(
        previous.state.participation, previous.state.precision, current.contact_pre,
        current.contact_response_gate, current.route, source, teaching,
    )
    assert torch.count_nonzero(direction) == 0


def test_zero_tags_or_zero_return_close_the_local_learning_signal():
    body = _body()
    _, previous, current, _, source, teaching = _pair(body)
    for occupied, returned in ((previous.state.participation, torch.zeros_like(teaching)),
                               (torch.zeros_like(previous.state.participation), teaching)):
        direction = body._s_contact_direction_from_tensors(
            occupied, previous.state.precision, current.contact_pre,
            current.contact_response_gate, current.route, source, returned,
        )
        assert torch.count_nonzero(direction) == 0


def test_retained_participation_can_modulate_current_learning_without_a_history_graph():
    body = _body()
    _, previous, current, _, source, teaching = _pair(body)
    occupied = previous.state.participation.detach()
    # A forward retention identity, not a cue/reward capability experiment.
    for _ in range(24):
        flat = occupied.flatten(start_dim=-2)
        flat = body._shared_contact_pool_exchange(
            flat, torch.zeros_like(flat), torch.zeros_like(flat[..., 0]),
            torch.full_like(flat, 0.02),
        )
        occupied = flat.unflatten(-1, (2, body.contact_rank))
    assert occupied.abs().sum() > 0
    retained = body._s_contact_direction_from_tensors(
        occupied, previous.state.precision, current.contact_pre,
        current.contact_response_gate, current.route, source, teaching,
    )
    assert retained[body.d_model + 2:].abs().sum() > 0
    assert retained.grad_fn is None
    assert occupied.grad_fn is None


def test_disabled_or_first_event_learning_does_not_construct_a_vjp(monkeypatch):
    body = _body()
    _, previous, current, target, source, _ = _pair(body)

    def unexpected(*args, **kwargs):
        raise AssertionError("a disabled/first-event learner constructed a VJP")

    monkeypatch.setattr(torch.func, "vjp", unexpected)
    before = body.reciprocal_contact.detach().clone()
    body.update_s_contact_from_following(previous, current, previous.state, target,
                                        step_size=0.0, source=(source == 0))
    body.update_s_contact_from_following(None, current, previous.state, target,
                                        step_size=0.1, source=(source == 0))
    torch.testing.assert_close(body.reciprocal_contact, before, rtol=0.0, atol=0.0)


def test_cpu_phase_continuation_keeps_the_single_learning_receipt():
    from scripts.run_cgb_column_language_admission import _eager_value_credit_phase

    single = _body(double=False)
    split = copy.deepcopy(single)
    events = np.asarray([35, 77, 0, 18, 111, 9], dtype=np.int64)
    all_at_once = _eager_value_credit_phase(
        single, single.initial_state(1), events, 0.01, 0.001, 0.002, 0.002, 100,
        s_contact_step=0.01,
    )
    first = _eager_value_credit_phase(
        split, split.initial_state(1), events[:2], 0.01, 0.001, 0.002, 0.002, 100,
        s_contact_step=0.01,
    )
    second = _eager_value_credit_phase(
        split, first["_state"], events[2:], 0.01, 0.001, 0.002, 0.002, 100,
        s_contact_step=0.01, previous=first["_previous"],
    )
    for name in second["_state"].__dataclass_fields__:
        torch.testing.assert_close(getattr(second["_state"], name), getattr(all_at_once["_state"], name),
                                   rtol=0.0, atol=0.0)
    for left, right in zip(single.parameters(), split.parameters()):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_s_vjp_stream_matches_cpu_cuda_capture_and_phase_continuation():
    from scripts.run_cgb_column_language_admission import (
        CudaGraphValueCredit, _eager_value_credit_phase,
    )

    cpu = _body(double=False)
    gpu = copy.deepcopy(cpu).cuda()
    initial = gpu.initial_state(1)
    parameters = {name: value.detach().clone() for name, value in gpu.named_parameters()}
    runner = CudaGraphValueCredit(gpu, initial, 0.01, 0.001, 0.002, 0.002, 2,
                                  s_contact_step=0.01)
    for name in initial.__dataclass_fields__:
        torch.testing.assert_close(getattr(runner.state(), name), getattr(initial, name), rtol=0.0, atol=0.0)
    for name, value in gpu.named_parameters():
        torch.testing.assert_close(value, parameters[name], rtol=0.0, atol=0.0)
    receipt_shapes = tuple(value.shape for value in (
        runner.previous_contact_pre, runner.previous_s_flux, runner.previous_s_return_weight,
    ))
    events = np.asarray([35, 77, 0, 18, 111, 9], dtype=np.int64)
    expected = _eager_value_credit_phase(
        cpu, cpu.initial_state(1), events, 0.01, 0.001, 0.002, 0.002, 100,
        s_contact_step=0.01,
    )
    runner.phase(events[:2], 100)
    runner.phase(events[2:], 100)
    final = runner.state()
    for name in final.__dataclass_fields__:
        torch.testing.assert_close(getattr(final, name).cpu(), getattr(expected["_state"], name),
                                   rtol=3e-5, atol=3e-6)
        assert getattr(final, name).grad_fn is None
    for left, right in zip(cpu.parameters(), gpu.parameters()):
        torch.testing.assert_close(left, right.cpu(), rtol=3e-5, atol=3e-6)
        assert left.grad is None and right.grad is None
    assert receipt_shapes == tuple(value.shape for value in (
        runner.previous_contact_pre, runner.previous_s_flux, runner.previous_s_return_weight,
    ))
    assert gpu.persistent_state_bytes(initial) == gpu.persistent_state_bytes(final)
