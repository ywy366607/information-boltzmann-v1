"""Numerical derivatives, stochastic factor identity and persistent continuation.

Small tensors verify the algorithm/interface, not memory or task capability.
"""
import copy
import itertools
import os

import pytest
import torch

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.online_credit import (
    CapturedOnlineEvent, OnlinePlasticTrainer, credit_tensors, merge_rank_one)
from information_boltzmann.runtime.training import belief_tensors, quiet_training_chunk


def model():
    torch.manual_seed(621)
    return PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=8,
                                hidden=8, bath_type='conductance',
                                activity_adaptation=True,
                                short_term_plasticity=True).double()


def test_compression_expectation_equals_rtrl_including_cross_terms():
    propagated = torch.tensor([0.2, -0.7], dtype=torch.float64)
    previous = torch.tensor([0.5, -0.9, 0.1], dtype=torch.float64)
    injection = torch.tensor([[0.4, -0.1, 0.8], [0.3, 0.7, -0.2]], dtype=torch.float64)
    samples = []
    for signs in itertools.product((-1., 1.), repeat=2):
        signs = torch.tensor(signs, dtype=torch.float64)
        u, v = merge_rank_one((propagated,), (previous,), (signs,), (injection.T @ signs,))
        samples.append(u[0][:, None] * v[0][None, :])
    torch.testing.assert_close(torch.stack(samples).mean(0),
                               propagated[:, None] * previous[None, :] + injection,
                               rtol=1e-13, atol=1e-13)


def test_first_event_matches_original_physics_objective_and_all_gradients():
    m = model()
    initial = m.initial_belief()
    ids, targets = torch.tensor([[1]]), torch.tensor([[2]])
    loss, expected, _ = quiet_training_chunk(m, ids, targets, initial, event_duration=.005)
    loss.backward()
    expected_grad = {n: None if p.grad is None else p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad(set_to_none=True)
    learner = OnlinePlasticTrainer(m, initial, event_duration=.005)
    metrics = learner.backward_event(ids, targets)
    torch.testing.assert_close(metrics['loss'], loss)
    assert metrics['history_gradient_norm'] == 0
    for a, b in zip(belief_tensors(learner.belief), belief_tensors(expected)):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    for name, p in m.named_parameters():
        if expected_grad[name] is None:
            assert torch.count_nonzero(p.grad) == 0
        else:
            torch.testing.assert_close(p.grad, expected_grad[name], atol=1e-11, rtol=1e-10)


def test_history_reaches_internal_operators_without_a_read_mask():
    m = model()
    learner = OnlinePlasticTrainer(m, event_duration=.005)
    learner.backward_event(torch.tensor([1]), torch.tensor([2]))
    # This is a parameter-sensitivity trace, including the whole physical chain.
    for prefix in ('medium.material.', 'medium.collision', 'write_agent.'):
        entries = [v for n, v in zip(learner.names, learner.parameter_factors[0]) if n.startswith(prefix)]
        assert entries, prefix
        assert sum(float(v.square().sum()) for v in entries) > 0, prefix
    m.zero_grad(set_to_none=True)
    result = learner.backward_event(torch.tensor([2]), torch.tensor([3]))
    assert result['history_gradient_norm'] > 0
    # Pure readout parameters have direct gradients, not ds/dtheta traces.
    assert sum(float(p.grad.square().sum()) for n, p in m.named_parameters()
               if n.startswith('readout.')) > 0


def test_fixed_trace_storage_no_historical_graph_and_optimizer_boundaries():
    m = model()
    learner = OnlinePlasticTrainer(m, event_duration=.005, rank=2)
    optimizer = torch.optim.SGD(m.parameters(), lr=1e-5)
    size = learner.eligibility_bytes()
    for i in range(12):
        optimizer.zero_grad(set_to_none=True)
        learner.backward_event(torch.tensor([i % 15 + 1]), torch.tensor([(i+1) % 15 + 1]))
        optimizer.step()
        assert learner.eligibility_bytes() == size
        tensors = list(belief_tensors(learner.belief))
        tensors += [t for group in (learner.state_factors, learner.parameter_factors)
                    for factor in group for t in factor]
        assert all(t.grad_fn is None and not t.requires_grad for t in tensors)
    assert learner.events == 12
    assert float(learner.belief.medium.elapsed[0]) == pytest.approx(.06)


def test_checkpoint_continuation_preserves_eligibility_rng_and_physics(tmp_path):
    m = model()
    learner = OnlinePlasticTrainer(m, event_duration=.005)
    learner.backward_event(torch.tensor([1]), torch.tensor([2]))
    path = tmp_path / 'continuation.pt'
    torch.save({'model': m.state_dict(), 'online': learner.state_dict()}, path)
    saved = torch.load(path, weights_only=False)
    m2 = model()
    m2.load_state_dict(saved['model'])
    resumed = OnlinePlasticTrainer(m2, event_duration=.005)
    resumed.load_state_dict(saved['online'])
    m.zero_grad(set_to_none=True)
    learner.backward_event(torch.tensor([2]), torch.tensor([3]))
    resumed.backward_event(torch.tensor([2]), torch.tensor([3]))
    for a, b in zip(m.parameters(), m2.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    for group1, group2 in ((learner.state_factors, resumed.state_factors),
                          (learner.parameter_factors, resumed.parameter_factors)):
        for x, y in zip(group1, group2):
            for a, b in zip(x, y):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(belief_tensors(learner.belief), belief_tensors(resumed.belief)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    malformed = copy.deepcopy(saved['online'])
    malformed['state_factors'][0] = malformed['state_factors'][0][:-1]
    with pytest.raises(ValueError, match='shapes'):
        resumed.load_state_dict(malformed)


def test_loss_scaling_changes_gradients_not_eligibility_or_state():
    a, b = model(), model()
    ta = OnlinePlasticTrainer(a, event_duration=.005)
    tb = OnlinePlasticTrainer(b, event_duration=.005)
    for token in (1, 2):
        a.zero_grad(set_to_none=True)
        b.zero_grad(set_to_none=True)
        ta.backward_event(torch.tensor([token]), torch.tensor([token+1]), loss_scale=1.)
        tb.backward_event(torch.tensor([token]), torch.tensor([token+1]), loss_scale=.25)
        for x, y in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(y.grad, x.grad*.25)
        for u, v in zip(ta.state_factors[0], tb.state_factors[0]):
            torch.testing.assert_close(u, v, rtol=0, atol=0)


def test_initial_zero_factors_and_zero_injection_are_finite():
    zero = (torch.zeros(2),)
    signs = (torch.ones(2),)
    u, v = merge_rank_one(zero, zero, signs, zero)
    assert all(torch.isfinite(t).all() for t in (*u, *v))
    assert torch.count_nonzero(v[0]) == 0


@pytest.mark.skipif(os.environ.get('IB_ENABLE_RUNTIME_CUDA_TESTS') != '1',
                    reason='Explicit GPU allocation')
def test_captured_online_jvp_vjp_rng_continuation_and_weight_updates():
    a,b = model().float().cuda(),model().float().cuda()
    ta = OnlinePlasticTrainer(a,event_duration=.005)
    tb = OnlinePlasticTrainer(b,event_duration=.005)
    captured = CapturedOnlineEvent(ta,loss_scale=.25)
    for token in (1,2,3):
        if token == 3:
            with torch.no_grad():
                for m in (a,b):
                    m.medium.material.coefficients.add_(.001)
                    m.write_agent.chart_gate[-1].weight.add_(.001)
                    m.readout.k_proj.weight.mul_(1.01)
        a.zero_grad(set_to_none=False)
        b.zero_grad(set_to_none=True)
        ids = torch.tensor([token],device='cuda')
        targets = torch.tensor([token+1],device='cuda')
        actual = captured.backward(ids,targets)
        expected = tb.backward_event(ids,targets,loss_scale=.25)
        for key in actual:
            torch.testing.assert_close(actual[key],expected[key],atol=1e-6,rtol=1e-5)
        for x,y in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(x.grad,y.grad,atol=2e-6,rtol=2e-4)
        for x,y in zip(belief_tensors(ta.belief),belief_tensors(tb.belief)):
            torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)
        for x,y in zip(ta.state_factors[0],tb.state_factors[0]):
            torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)
    assert ta.events == tb.events == 3
    a.zero_grad(set_to_none=True)
    with pytest.raises(RuntimeError,match='gradient storage'):
        captured.backward(ids,targets)
