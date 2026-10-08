"""First-order VJP equivalence on the real recurrent model, not capabilities."""
from dataclasses import replace
import copy

import pytest
import torch
from torch.nn import functional as F

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.training import belief_tensors, training_event


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def numerical_model(intrinsic_clock=False):
    torch.manual_seed(624)
    model = PlasticMediumPorts3D(
        vocab_size=11, shape=(2, 2, 2), channels=4, material_width=2,
        hidden=4, heads=1, queries=1, anisotropic_transport=True,
        bath_type='conductance', short_term_plasticity=True,
        activity_adaptation=True, material_reference_shape=None,
        read_mode='temporal', temporal_rates=[1., 4.],
        temporal_frequencies=[0., 3.], temporal_time_reference=.02,
        intrinsic_time_reference=.006 if intrinsic_clock else None,
        solver_max_step=.01, observer_max_step=.02,
        structure_options=dict(
            resource_density=4., speed_reference=3., structure_time=1.,
            prior_std=.2, initial_std=.2, maintenance_supply=2.,
            initial_dual=.1)).double()
    with torch.no_grad():
        model.medium.material.coefficients.normal_(std=.03)
        if intrinsic_clock:
            model.intrinsic_time.head.weight.fill_(.03)
    posterior = model.medium.structural_posterior
    noise = torch.linspace(-.3, .4, posterior.mean.numel(), dtype=torch.float64)
    posterior.begin_window(noise.reshape_as(posterior.mean))
    return model


def leaf_belief(model):
    belief = model.initial_belief()

    def leaf(value):
        return None if value is None else value.detach().clone().requires_grad_(True)

    medium = belief.medium
    field = torch.linspace(-.12, .16, medium.field.numel(), dtype=torch.float64)
    field = field.reshape_as(medium.field)
    medium = replace(
        medium, field=leaf(field),
        flux=tuple(leaf(field * ((axis + 1) / 9)) for axis in range(3)),
        elapsed=leaf(torch.full_like(medium.elapsed, 17.)),
        conduction=leaf(medium.conduction), receptors=leaf(medium.receptors),
        transmission=leaf(medium.transmission))
    temporal = replace(
        belief.temporal, value=leaf(belief.temporal.value),
        elapsed=leaf(torch.full_like(belief.temporal.elapsed, 17.)))
    return replace(belief, medium=medium, precision=leaf(belief.precision),
                   temporal=temporal)


def run_32_events(model, initial, duration_seed, checkpointed):
    # These shared non-leaf graphs must cross the boundary as explicit inputs.
    # Otherwise a per-event VJP could consume their common parameter graph.
    table = F.normalize(model.source.embedding.weight, dim=-1)
    prepared = model.medium.prepare_evolution()
    duration = .004 + .001 * duration_seed.sin()
    parameters = tuple(parameter for _, parameter in model.learning_named_parameters())

    def event(current, token, token_table, coefficients, interval):
        _, outgoing, write, _, feature = training_event(
            model, current, token, token_table, coefficients, interval,
            1, False, activation_checkpointing=checkpointed)
        return outgoing, feature, write['_write_free_energy']

    if checkpointed:
        # Import here so this new test can be collected before helper landing.
        from information_boltzmann.core.state_checkpoint import checkpoint_state_vjp

    belief, features, objectives = initial, [], []
    tokens = (torch.arange(32) % 10 + 1)[None]
    targets = ((torch.arange(32) + 2) % 10 + 1)[None]
    for token in tokens.unbind(1):
        arguments = (token, table, prepared, duration)
        if checkpointed:
            belief, feature, objective = checkpoint_state_vjp(
                event, belief, *arguments, parameters=parameters)
        else:
            belief, feature, objective = event(belief, *arguments)
        features.append(feature)
        objectives.append(objective)
    expression = torch.stack(features, 1)
    scores = F.cross_entropy(model.decode(expression).flatten(0, 1),
                             targets.flatten(), reduction='none').reshape_as(targets)
    port_objective = torch.stack(objectives).mean()
    # Exercise cotangents for every real output state, including complex
    # temporal history and both FP64 clocks, rather than only decoded features.
    state_objective = sum(value.abs().square().mean() for value in belief_tensors(belief))
    objective = scores.mean() + .07 * port_objective + 1e-4 * state_objective
    return objective, belief, expression, scores, port_objective


def assert_gradients_equal(left, right, label):
    assert (left.grad is None) == (right.grad is None), label
    if left.grad is not None:
        assert torch.isfinite(left.grad).all(), label
        assert torch.isfinite(right.grad).all(), label
        torch.testing.assert_close(left.grad, right.grad, atol=3e-10, rtol=3e-9,
                                   msg=label)


@pytest.mark.parametrize('scale,pending,intrinsic_clock',
                         [(.37, False, False), (-1.25, True, False), (.63, False, True)])
def test_vjp_32_events_matches_all_parameter_state_and_duration_gradients(scale, pending, intrinsic_clock):
    reference = numerical_model(intrinsic_clock)
    checked = copy.deepcopy(reference)
    left_state, right_state = leaf_belief(reference), leaf_belief(checked)
    left_duration = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    right_duration = left_duration.detach().clone().requires_grad_(True)
    left_leaves, right_leaves = belief_tensors(left_state), belief_tensors(right_state)
    if pending:
        for index, ((_, left), (_, right)) in enumerate(zip(
                reference.named_parameters(), checked.named_parameters())):
            if left.requires_grad and index % 2 == 0:
                left.grad = torch.full_like(left, .0003)
                right.grad = left.grad.clone()
        for left, right in zip((*left_leaves, left_duration),
                               (*right_leaves, right_duration)):
            left.grad = torch.full_like(left, .0002)
            right.grad = left.grad.clone()
    expected = run_32_events(reference, left_state, left_duration, False)
    actual = run_32_events(checked, right_state, right_duration, True)
    for left, right in zip(expected[:1] + expected[2:], actual[:1] + actual[2:]):
        torch.testing.assert_close(left, right, atol=2e-12, rtol=2e-12)
    for left, right in zip(belief_tensors(expected[1]), belief_tensors(actual[1])):
        torch.testing.assert_close(left, right, atol=2e-12, rtol=2e-12)
    assert actual[1].medium.elapsed.dtype == torch.float64
    assert actual[1].temporal.elapsed.dtype == torch.float64
    assert actual[1].temporal.value.is_complex()
    (scale * expected[0]).backward()
    (scale * actual[0]).backward()
    for (name, left), (other, right) in zip(reference.named_parameters(),
                                          checked.named_parameters()):
        assert name == other
        assert_gradients_equal(left, right, name)
    for index, (left, right) in enumerate(zip((*left_leaves, left_duration),
                                             (*right_leaves, right_duration))):
        assert_gradients_equal(left, right, f'initial state/duration {index}')
    if intrinsic_clock:
        assert left_duration.grad is None and right_duration.grad is None
        assert reference.intrinsic_time.head.bias.grad.abs().max() > 0
    else:
        assert left_duration.grad is not None and left_duration.grad.abs() > 0
        # The physical duration traverses the same shared non-leaf graph 32 times.
        torch.testing.assert_close(actual[1].medium.elapsed,
            left_state.medium.elapsed.detach() + 32 * (.004 + .001 * left_duration.detach().sin()),
            atol=5e-14, rtol=0)
