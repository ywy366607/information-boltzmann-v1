"""Exact first-order/continuation contracts for serialized auxiliary execution."""
from dataclasses import replace
import copy

import pytest
import torch
from torch.nn import functional as F

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime import training


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def candidate():
    torch.manual_seed(813)
    net = PlasticMediumPorts3D(vocab_size=13, shape=(2, 2, 2), channels=4,
        material_width=2, hidden=4, heads=1, queries=1, anisotropic_transport=True,
        bath_type='conductance', short_term_plasticity=True, activity_adaptation=True,
        material_reference_shape=None, read_mode='temporal', temporal_rates=[1., 4.],
        temporal_frequencies=[0., 3.], intrinsic_time_reference=.023,
        solver_max_step=.01, observer_max_step=.02,
        structure_options=dict(resource_density=4., speed_reference=4., structure_time=1.,
            prior_std=.2, initial_std=.2, maintenance_supply=2., initial_dual=.1)).double()
    with torch.no_grad():
        net.medium.material.coefficients.normal_(std=.05)
        net.intrinsic_time.head.weight.fill_(.03)
    net.medium.structural_posterior.begin_window(
        torch.randn_like(net.medium.structural_posterior.mean))
    net._structural_window_events = 64
    return net


def initial(net, live_prefix=False):
    belief = net.initial_belief()
    field = torch.linspace(-.15, .2, belief.medium.field.numel(), dtype=torch.float64)
    field = field.reshape_as(belief.medium.field)
    if live_prefix:
        field = field + .01 * net.source.channel_scale.square().sum()
    return replace(belief, medium=belief.medium.with_field(field))


def same_graph_reference(net, ids, targets, belief):
    """Previous block-gradient formula, with one uninterrupted recurrent graph."""
    table = F.normalize(net.source.embedding.weight, dim=-1)
    prepared = net.medium.prepare_evolution()
    features, objectives = [], []
    for token in ids.unbind(1):
        _, belief, info, _, feature = training.training_event(
            net, belief, token, table, prepared, .005, 1, False)
        features.append(feature)
        objectives.append(info['_write_free_energy'])
    logits = net.decode(torch.stack(features, 1))
    scores = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                             reduction='none').reshape_as(targets)
    nll, port = scores.mean(), torch.stack(objectives).mean()
    q = net.medium.structural_posterior
    cost = q.maintenance(prepared.structural_allocation, 1 / 8)
    main = q.objective(nll, cost, net._structural_window_events)
    parameters = tuple(p for name, p in net.learning_named_parameters()
                       if name.startswith(('source.', 'write_agent.')))
    gradients = torch.autograd.grad(port, parameters, retain_graph=True, allow_unused=True)
    auxiliary = port.detach()
    for parameter, gradient in zip(parameters, gradients):
        if gradient is not None:
            auxiliary = auxiliary + ((parameter - parameter.detach()) * gradient.detach()).sum()
    return main + auxiliary, belief, nll, scores


def assert_parameter_gradients_equal(left, right):
    for (name, a), (other, b) in zip(left.named_parameters(), right.named_parameters()):
        assert name == other
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=3e-11, rtol=3e-9, msg=name)


@pytest.mark.parametrize('checkpointed', [False, True])
def test_32_step_replay_matches_full_graph_values_state_and_scaled_accumulation(checkpointed, monkeypatch):
    reference = candidate()
    actual = copy.deepcopy(reference)
    ids = (torch.arange(32) % 12 + 1)[None]
    targets = ((torch.arange(32) + 2) % 12 + 1)[None]
    # Existing pending credit is preserved by the auxiliary autograd.grad pass.
    for (_, a), (_, b) in zip(reference.named_parameters(), actual.named_parameters()):
        if a.requires_grad:
            pending = torch.full_like(a, .0003)
            a.grad, b.grad = pending.clone(), pending.clone()
    routed = []
    original_apply = training._WriterAuxiliaryGradient.apply

    def observe_cpu_storage(value, gradients, *parameters):
        assert all(g is None or g.device.type == 'cpu' for g in gradients)
        assert all(g is None or not g.requires_grad for g in gradients)
        routed.append(len(gradients))
        return original_apply(value, gradients, *parameters)

    monkeypatch.setattr(training._WriterAuxiliaryGradient, 'apply', observe_cpu_storage)
    q_before = {name: value.clone() for name, value in actual.medium.structural_posterior.state_dict().items()}
    expected = same_graph_reference(reference, ids, targets, initial(reference))
    result = training.quiet_training_chunk(actual, ids, targets, initial(actual),
        event_duration=.005, activation_checkpointing=checkpointed, return_token_nll=True)
    assert len(routed) == 1
    for a, b in zip(expected[:1] + expected[2:], result[:1] + result[2:]):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)
    for a, b in zip(training.belief_tensors(expected[1]), training.belief_tensors(result[1])):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)
    for name, value in actual.medium.structural_posterior.state_dict().items():
        torch.testing.assert_close(q_before[name], value, atol=0, rtol=0)
    # quiet forward's local autograd.grad never accumulates into pending .grad.
    assert all(p.grad is None or torch.equal(p.grad, torch.full_like(p, .0003))
               for p in actual.parameters())
    (expected[0] * .37).backward()
    (result[0] * .37).backward()
    assert_parameter_gradients_equal(reference, actual)
    assert actual.intrinsic_time.head.bias.grad.abs().max() > .0003


def test_replay_preserves_explicit_live_prefix_and_rng_advancement(monkeypatch):
    reference = candidate()
    actual = copy.deepcopy(reference)
    original_event = training.training_event

    def stochastic_value_for_rng_contract(*args, **kwargs):
        written, out, info, evolution, feature = original_event(*args, **kwargs)
        # Exercise RNG replay without adding a stochastic architecture to the model.
        info['_write_free_energy'] = info['_write_free_energy'] * (.9 + .2 * torch.rand(()))
        return written, out, info, evolution, feature

    monkeypatch.setattr(training, 'training_event', stochastic_value_for_rng_contract)
    ids, targets = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])
    rng = torch.get_rng_state()
    expected = same_graph_reference(reference, ids, targets, initial(reference, live_prefix=True))
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    result = training.quiet_training_chunk(actual, ids, targets, initial(actual, live_prefix=True),
                                            event_duration=.005)
    assert torch.equal(expected_rng, torch.get_rng_state())
    torch.testing.assert_close(expected[0], result[0], atol=0, rtol=0)
    expected[0].backward()
    result[0].backward()
    assert_parameter_gradients_equal(reference, actual)


def test_first_order_vjp_repeated_scaling_does_not_mutate_cpu_gradients():
    p = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
    cpu_gradient = torch.tensor([.3, -.4], dtype=torch.float64)
    saved = cpu_gradient.clone()
    value = training._WriterAuxiliaryGradient.apply(torch.tensor(5.), (cpu_gradient,), p)
    first = torch.autograd.grad(2 * value, p, retain_graph=True)[0]
    second = torch.autograd.grad(3 * value, p)[0]
    torch.testing.assert_close(first, 2 * saved, atol=0, rtol=0)
    torch.testing.assert_close(second, 3 * saved, atol=0, rtol=0)
    torch.testing.assert_close(cpu_gradient, saved, atol=0, rtol=0)


def test_health_and_no_grad_execution_commit_only_one_main_forward(monkeypatch):
    net = candidate()
    ids, targets = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])

    class Capture:
        def __init__(self):
            self.selected, self.events, self.decodes = [], 0, 0

        def select(self, index):
            self.selected.append(index)

        def record(self, *args):
            self.events += 1

        def record_decode(self, *args):
            self.decodes += 1

    capture = Capture()
    training.quiet_training_chunk(net, ids, targets, initial(net),
        event_duration=.005, health_capture=capture)
    assert capture.selected == [0, 1, 2] and capture.events == 3 and capture.decodes == 1
    calls = []
    original = training.training_event

    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(training, 'training_event', count)
    with torch.no_grad():
        training.quiet_training_chunk(net, ids, targets, initial(net), event_duration=.005)
    assert len(calls) == 3
