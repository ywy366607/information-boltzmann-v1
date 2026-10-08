"""Numerical local derivatives/continuation, not task capability experiments."""
import copy
import os

import pytest
import torch

from information_boltzmann.core.conductance_response import LocalConductanceResponse
from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.local_credit import (
    CapturedLocalEvent, LocalPlasticTrainer, LocalReceptorEligibility, receptor_partials)
from information_boltzmann.runtime.training import belief_tensors, quiet_training_chunk


@pytest.mark.parametrize('adaptation', [False, True])
def test_kinetic_partials_match_actual_gate_and_closing_derivatives(adaptation):
    torch.manual_seed(32)
    response = LocalConductanceResponse(3, 2, 5, activity_adaptation=adaptation).double()
    material = torch.randn(2, 2, 2, 2, dtype=torch.float64) * .1
    coefficients = response.coefficients(material)
    field = torch.randn(1, 2, 2, 2, 3, dtype=torch.float64)
    flux = tuple(torch.randn_like(field) for _ in range(3))
    gates = torch.full((1, 2, 2, 2, 2, 3), .8, dtype=torch.float64, requires_grad=True)
    dt = torch.full_like(field[..., :1], .037)
    opening, feedback, baseline = response.opening_rates(
        field, flux, coefficients, gates, local_partials=True)
    jac, injection = receptor_partials(gates, opening, coefficients.closing, dt, feedback, baseline)
    output = response.gates_step(gates, opening, coefficients.closing, dt)
    for branch in range(2):
        dgate, dclosing = torch.autograd.grad(output[..., branch, :].sum(),
                                             (gates, coefficients.closing), retain_graph=True)
        torch.testing.assert_close(dgate.transpose(-2, -1), jac[..., branch, :])
        torch.testing.assert_close(dclosing.transpose(-2, -1), injection[..., branch, :].sum(0))


def test_trace_matches_long_conditional_receptor_recurrence_without_a_window():
    response = LocalConductanceResponse(2, 2, 4, activity_adaptation=True).double()
    material = torch.zeros(2, 2, 2, 2, dtype=torch.float64)
    coefficients = response.coefficients(material)
    field = torch.full((1, 2, 2, 2, 2), .2, dtype=torch.float64)
    flux = tuple(torch.ones_like(field) * .1 for _ in range(3))
    gates = torch.ones(1, 2, 2, 2, 2, 2, dtype=torch.float64) * .8
    credit = LocalReceptorEligibility(gates)
    dt = torch.full_like(field[..., :1], .019)
    for _ in range(137):
        opening, feedback, baseline = response.opening_rates(
            field, flux, coefficients, gates, local_partials=True)
        credit.observe(gates, opening, coefficients.closing, dt, feedback, baseline)
        gates = response.gates_step(gates, opening, coefficients.closing, dt)
    signal = torch.randn_like(gates)
    exact = torch.autograd.grad((gates * signal).sum(), coefficients.closing)[0]
    torch.testing.assert_close(credit.closing_feedback(signal), exact, rtol=1e-11, atol=1e-11)
    assert credit.trace.grad_fn is None
    assert credit.kinetic_steps == 137


def make_model():
    torch.manual_seed(491)
    return PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=8,
                                hidden=8, bath_type='conductance',
                                activity_adaptation=True, short_term_plasticity=True).double()


def test_birth_event_matches_all_original_physics_and_parameter_gradients():
    model = make_model()
    belief = model.initial_belief()
    ids, target = torch.tensor([[1]]), torch.tensor([[2]])
    loss, expected, nll = quiet_training_chunk(model, ids, target, belief, event_duration=.005)
    loss.backward()
    expected_grad = {name: None if p.grad is None else p.grad.clone()
                     for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    learner = LocalPlasticTrainer(model, belief, event_duration=.005)
    result = learner.backward_event(ids, target)
    torch.testing.assert_close(result['loss'], loss)
    torch.testing.assert_close(result['token_nll'], nll)
    assert result['history_gradient_norm'] == 0
    for a, b in zip(belief_tensors(learner.belief), belief_tensors(expected)):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    for name, parameter in model.named_parameters():
        reference = expected_grad[name]
        if reference is None:
            assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        else:
            torch.testing.assert_close(parameter.grad, reference, rtol=1e-10, atol=1e-11)


def test_history_is_deterministic_local_and_survives_optimizer_and_resume():
    model = make_model()
    learner = LocalPlasticTrainer(model, event_duration=.005, substeps=2)
    learner.backward_event(torch.tensor([1]), torch.tensor([2]))
    snapshot = learner.state_dict()
    saved_trace = snapshot['trace'].clone()
    assert learner.eligibility.kinetic_steps == 4
    assert learner.eligibility_bytes() == 8 * 8 * 4 * 8  # sites * channels * 2x2 * fp64
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    optimizer.step()
    torch.testing.assert_close(snapshot['trace'], learner.eligibility.trace)
    other = copy.deepcopy(model)
    resumed = LocalPlasticTrainer(other, event_duration=.005, substeps=2)
    resumed.load_state_dict(snapshot)
    model.zero_grad(set_to_none=True)
    torch.manual_seed(1)
    rng = torch.get_rng_state().clone()
    result = learner.backward_event(torch.tensor([2]), torch.tensor([3]))
    assert torch.equal(rng, torch.get_rng_state())
    torch.manual_seed(123)
    other_result = resumed.backward_event(torch.tensor([2]), torch.tensor([3]))
    for key in result:
        torch.testing.assert_close(result[key], other_result[key])
    assert result['history_gradient_norm'] > 0
    for p, q in zip(model.parameters(), other.parameters()):
        if p.grad is None or q.grad is None:
            assert p.grad is None and q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad)
    assert learner.events == 2
    assert learner.eligibility.trace.grad_fn is None
    assert all(t.grad_fn is None for t in belief_tensors(learner.belief))
    assert not hasattr(model.medium.conductance_response, '_local_credit_observer')
    torch.testing.assert_close(snapshot['trace'], saved_trace)


def test_failed_event_rolls_back_credit_and_releases_observer():
    model = make_model()
    learner = LocalPlasticTrainer(model, event_duration=.005)
    before = learner.state_dict()
    with pytest.raises((IndexError, RuntimeError)):
        learner.backward_event(torch.tensor([1]), torch.tensor([1000]))
    torch.testing.assert_close(before['trace'], learner.eligibility.trace)
    assert learner.events == 0 and learner.eligibility.kinetic_steps == 0
    assert not hasattr(model.medium.conductance_response, '_local_credit_observer')


@pytest.mark.skipif(os.getenv('IB_ENABLE_RUNTIME_CUDA_TESTS') != '1' or not torch.cuda.is_available(),
                    reason='Optional CUDA capture test')
def test_cuda_local_replay_matches_eager_and_refreshes_parameters():
    torch.set_num_threads(1)
    model = make_model().float().cuda()
    other = copy.deepcopy(model)
    eager = LocalPlasticTrainer(model, event_duration=.005, substeps=2)
    online = LocalPlasticTrainer(other, event_duration=.005, substeps=2)
    snapshot = online.state_dict()
    captured = CapturedLocalEvent(online, loss_scale=.5)
    torch.testing.assert_close(snapshot['trace'], online.eligibility.trace)
    assert online.events == 0 and online.eligibility.kinetic_steps == 0
    for index in range(3):
        ids, target = torch.tensor([index + 1], device='cuda'), torch.tensor([index + 2], device='cuda')
        model.zero_grad(set_to_none=False)
        other.zero_grad(set_to_none=False)
        actual = eager.backward_event(ids, target, loss_scale=.5)
        replay = captured.backward(ids, target)
        for key in actual:
            torch.testing.assert_close(actual[key], replay[key], rtol=3e-4, atol=2e-6)
        for p, q in zip(model.parameters(), other.parameters()):
            if p.grad is None or q.grad is None:
                assert p.grad is None and q.grad is None
            else:
                torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=2e-6)
        for p, q in zip(belief_tensors(eager.belief), belief_tensors(online.belief)):
            torch.testing.assert_close(p, q, rtol=3e-4, atol=2e-6)
        with torch.no_grad():
            for p, q in zip(model.parameters(), other.parameters()):
                if p.grad is not None:
                    p.add_(-1e-5 * p.grad)
                    q.add_(-1e-5 * q.grad)
    assert online.events == 3 and online.eligibility.kinetic_steps == 12
