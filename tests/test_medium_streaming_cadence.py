"""Short physical cadence contracts on the real medium; no capability study."""
from dataclasses import replace
import copy
import math

import pytest
import torch
from torch.nn import functional as F

from information_boltzmann.core.intrinsic_time import EvolutionSchedule
from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.core.temporal_probes import sample_compact_probes
from information_boltzmann.runtime.active_medium_training import ActiveMediumTrainer
from information_boltzmann.runtime.optimization import make_medium_optimizer
from information_boltzmann.runtime import training


# Physical model-time references, deliberately independent of runtime grid size.
ARRIVAL_INTERVAL = .012
SOLVER_MAX_STEP = .0031
OBSERVER_MAX_STEP = .02


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def numerical_model(*, shape=(2, 2, 2), structure=True, clock=True,
                    port_radius=None, solver_step=SOLVER_MAX_STEP):
    torch.manual_seed(1091)
    options = dict(
        resource_density=4., speed_reference=3., structure_time=1.,
        prior_std=.2, initial_std=.2, maintenance_supply=2., initial_dual=.1)
    model = PlasticMediumPorts3D(
        vocab_size=13, shape=shape, channels=4, material_width=2, hidden=4,
        heads=1, queries=1, anisotropic_transport=True,
        bath_type='conductance', short_term_plasticity=True,
        activity_adaptation=True, material_reference_shape=(2, 2, 2),
        read_mode='temporal', temporal_rates=[1., 4.],
        temporal_frequencies=[0., 3.], temporal_time_reference=ARRIVAL_INTERVAL,
        intrinsic_time_reference=ARRIVAL_INTERVAL if clock else None,
        solver_max_step=solver_step, observer_max_step=OBSERVER_MAX_STEP,
        write_port_radius=port_radius, read_port_radius=port_radius,
        write_aperture_budget=.125 if port_radius is not None else None,
        read_aperture_budget=.25 if port_radius is not None else None,
        structure_options=options if structure else None).double()
    with torch.no_grad():
        model.medium.material.coefficients.normal_(std=.03)
        if clock:
            model.intrinsic_time.head.weight.fill_(.005)
    return model


def begin_structure_window(model):
    posterior = model.medium.structural_posterior
    noise = torch.linspace(-.3, .4, posterior.mean.numel(), dtype=torch.float64)
    posterior.begin_window(noise.reshape_as(posterior.mean))
    model._structural_window_events = 32


def leaf_initial(model):
    belief = model.initial_belief()

    def leaf(value):
        return None if value is None else value.detach().clone().requires_grad_(True)

    field = torch.linspace(-.08, .11, belief.medium.field.numel(), dtype=torch.float64)
    field = field.reshape_as(belief.medium.field)
    medium = replace(
        belief.medium, field=leaf(field),
        flux=tuple(leaf(field * ((axis + 1) / 13)) for axis in range(3)),
        elapsed=leaf(torch.full_like(belief.medium.elapsed, 7.)),
        conduction=leaf(belief.medium.conduction),
        receptors=leaf(belief.medium.receptors),
        transmission=leaf(belief.medium.transmission))
    temporal = replace(belief.temporal, value=leaf(belief.temporal.value),
                       elapsed=leaf(torch.full_like(belief.temporal.elapsed, 7.)))
    return replace(belief, medium=medium, precision=leaf(belief.precision), temporal=temporal)


def full_graph_reference(model, ids, targets, belief):
    """Explicit physical intervals, with the unchanged block-writer objective."""
    table = F.normalize(model.source.embedding.weight, dim=-1)
    prepared = model.medium.prepare_evolution()
    features, objectives = [], []
    for token in ids.unbind(1):
        duration = model.event_time(belief, ARRIVAL_INTERVAL)
        belief, write = model.assimilate(belief, token, token_features=table,
                                         diagnostics=False)
        schedule = EvolutionSchedule.for_duration(duration,
            solver_max_step=model.solver_max_step,
            observer_max_step=model.observer_max_step,
            max_steps=model.max_evolution_steps)
        interval = schedule.interval_duration(duration)
        for _ in range(schedule.observer_count):
            state, _ = model.medium.native_advance(
                belief.medium, interval, substeps=schedule.solver_substeps,
                prepared=prepared, diagnostics=False)
            motion = model.read_time_reference * model.medium.field_rhs(state, prepared=prepared)
            belief = model.complete_advance(belief, state, interval,
                                            prepared=prepared, motion=motion)
        feature, _ = model.read(belief, decode=False, prepared=prepared, motion=motion)
        features.append(feature)
        objectives.append(write['_write_free_energy'])
    logits = model.decode(torch.stack(features, 1))
    scores = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                             reduction='none').reshape_as(targets)
    nll, port = scores.mean(), torch.stack(objectives).mean()
    posterior = model.medium.structural_posterior
    maintenance = posterior.maintenance(prepared.structural_allocation,
                                        1 / math.prod(model.medium.shape))
    objective = posterior.objective(nll, maintenance, 32)
    writer = tuple(p for name, p in model.learning_named_parameters()
                   if name.startswith(('source.', 'write_agent.')) and p.requires_grad)
    gradients = torch.autograd.grad(port, writer, retain_graph=True, allow_unused=True)
    auxiliary = port.detach()
    for parameter, gradient in zip(writer, gradients):
        if gradient is not None:
            auxiliary = auxiliary + ((parameter - parameter.detach()) * gradient.detach()).sum()
    return objective + auxiliary, belief, nll, scores


def assert_tree_equal(left, right, label='root'):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0, msg=label)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), label
        for key in left:
            assert_tree_equal(left[key], right[key], f'{label}.{key}')
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), label
        for index, (a, b) in enumerate(zip(left, right)):
            assert_tree_equal(a, b, f'{label}[{index}]')
    else:
        assert left == right, label


def test_full_32_short_events_match_explicit_values_and_complete_credit():
    reference = numerical_model()
    begin_structure_window(reference)
    actual = copy.deepcopy(reference)
    left, right = leaf_initial(reference), leaf_initial(actual)
    left_leaves, right_leaves = training.belief_tensors(left), training.belief_tensors(right)
    ids = (torch.arange(32) % 12 + 1)[None]
    targets = ((torch.arange(32) + 2) % 12 + 1)[None]
    for a, b in zip(reference.parameters(), actual.parameters()):
        if a.requires_grad:
            a.grad, b.grad = torch.full_like(a, .0002), torch.full_like(b, .0002)
    expected = full_graph_reference(reference, ids, targets, left)
    result = training.quiet_training_chunk(actual, ids, targets, right,
        event_duration=ARRIVAL_INTERVAL, activation_checkpointing=True, return_token_nll=True)
    for a, b in zip(expected[:1] + expected[2:], result[:1] + result[2:]):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)
    for a, b in zip(training.belief_tensors(expected[1]), training.belief_tensors(result[1])):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)
    # Include all outgoing leaves to exercise state cotangents, not just CE.
    for values in (expected, result):
        state_cost = sum(value.abs().square().mean()
                         for value in training.belief_tensors(values[1]))
        (.37 * values[0] + 1e-4 * state_cost).backward()
    for label, a, b in (
            *((name, a, b) for (name, a), (_, b) in zip(
                reference.named_parameters(), actual.named_parameters())),
            *((f'initial leaf {i}', a, b) for i, (a, b) in enumerate(zip(left_leaves, right_leaves)))):
        assert (a.grad is None) == (b.grad is None), label
        if a.grad is not None:
            assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all(), label
            torch.testing.assert_close(a.grad, b.grad, atol=3e-10, rtol=3e-9, msg=label)
    assert actual.intrinsic_time.head.bias.grad.abs().max() > .0002
    assert actual.medium.structural_posterior.evidence_events == 0
    torch.testing.assert_close(result[1].medium.elapsed, result[1].temporal.elapsed,
                               atol=5e-14, rtol=0)


def test_future_occurrences_and_targets_cannot_change_sealed_prefix(monkeypatch):
    model = numerical_model()
    begin_structure_window(model)
    traces = []
    original = training.training_event

    def record_feature(*args, **kwargs):
        result = original(*args, **kwargs)
        traces.append(result[-1].detach().clone())
        return result

    monkeypatch.setattr(training, 'training_event', record_feature)

    def run(ids, targets):
        traces.clear()
        with torch.no_grad():
            result = training.quiet_training_chunk(model, ids, targets,
                model.initial_belief(), event_duration=ARRIVAL_INTERVAL,
                activation_checkpointing=True, return_token_nll=True)
        return torch.stack(traces, 1), result

    ids, targets = torch.tensor([[1, 2, 3, 4]]), torch.tensor([[2, 3, 4, 5]])
    features, baseline = run(ids, targets)
    future_features, future = run(torch.tensor([[1, 2, 8, 9]]), torch.tensor([[2, 8, 9, 7]]))
    torch.testing.assert_close(features[:, :2], future_features[:, :2], atol=0, rtol=0)
    assert not torch.equal(features[:, 2:], future_features[:, 2:])
    target_features, changed_targets = run(ids, torch.tensor([[9, 8, 7, 6]]))
    torch.testing.assert_close(features, target_features, atol=0, rtol=0)
    for a, b in zip(training.belief_tensors(baseline[1]),
                    training.belief_tensors(changed_targets[1])):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(baseline[3][:, :1], future[3][:, :1], atol=0, rtol=0)


def nearby_model(*, solver_step=SOLVER_MAX_STEP):
    model = numerical_model(shape=(8, 4, 4), structure=False, clock=False,
                            port_radius=(.07, .13, .13), solver_step=solver_step)
    # Controlled physical units: even the two-cell center route fits in one
    # arrival interval. A slow route's tiny semidiscrete tail is not acceptance.
    model.medium.speed_reference = 28.
    with torch.no_grad():
        # Co-located finite write modes still pay eight aperture volumes. The
        # actual continuous support boxes are separated from the read box.
        model.write_agent.local_ports.centers.copy_(torch.tensor([[.125, .25, .25]]))
        model.readout.probe_coords.copy_(torch.tensor([[[.375, .25, .25]]]))
    return model


def probe_difference(model, left, right):
    return (sample_compact_probes(model.readout, left.medium.field)
            - sample_compact_probes(model.readout, right.medium.field))


def blocked_advance(model, belief, prepared):
    schedule = EvolutionSchedule.for_duration(ARRIVAL_INTERVAL,
        solver_max_step=model.solver_max_step, observer_max_step=model.observer_max_step,
        max_steps=model.max_evolution_steps)
    interval = ARRIVAL_INTERVAL / schedule.observer_count
    for _ in range(schedule.observer_count):
        state, _ = model.medium.advance(belief.medium, interval,
            substeps=schedule.solver_substeps, transport=False,
            prepared=prepared, diagnostics=False)
        motion = model.read_time_reference * model.medium.field_rhs(state, prepared=prepared)
        belief = model.complete_advance(belief, state, interval, prepared=prepared, motion=motion)
    return belief


def test_controlled_disjoint_route_response_and_older_response_survive_new_arrivals():
    """A placed, physically reachable route exercises the finite-port interface.

    The production individual's learned placement and deeper-path use require
    their own real-data measurements; this fixture specifies neither of them.
    """
    model = nearby_model()
    write_support = model.write_agent.local_ports.weights().sum(0) > 0
    read_support = model.readout.weights().sum(0) > 0
    assert not (write_support & read_support).any()
    assert model.write_agent.local_ports.aperture_volume <= .125
    assert model.readout.aperture_volume <= .25
    initial = model.initial_belief()
    with torch.no_grad():
        left, _ = model.assimilate(initial, torch.tensor([1]), diagnostics=False)
        right, _ = model.assimilate(initial, torch.tensor([2]), diagnostics=False)
        assert torch.count_nonzero(left.medium.field - right.medium.field) > 0
        assert torch.count_nonzero(probe_difference(model, left, right)) == 0
        prepared = model.medium.prepare_evolution()
        blocked_left = blocked_advance(model, left, prepared)
        blocked_right = blocked_advance(model, right, prepared)
        assert torch.count_nonzero(probe_difference(model, blocked_left, blocked_right)) == 0
        # This includes the real motion-RHS path and its incident edge stores.
        blocked_feature_left = model.read(blocked_left, decode=False, prepared=prepared)[0]
        blocked_feature_right = model.read(blocked_right, decode=False, prepared=prepared)[0]
        torch.testing.assert_close(blocked_feature_left, blocked_feature_right, atol=0, rtol=0)
        coarse_left, _ = model.advance(left, ARRIVAL_INTERVAL, prepared=prepared, diagnostics=False)
        coarse_right, _ = model.advance(right, ARRIVAL_INTERVAL, prepared=prepared, diagnostics=False)
        coarse = probe_difference(model, coarse_left, coarse_right)
        model.solver_max_step = SOLVER_MAX_STEP / 4
        fine_left, _ = model.advance(left, ARRIVAL_INTERVAL, prepared=prepared, diagnostics=False)
        fine_right, _ = model.advance(right, ARRIVAL_INTERVAL, prepared=prepared, diagnostics=False)
        fine = probe_difference(model, fine_left, fine_right)
        written_norm = (left.medium.field - right.medium.field).norm()
        roundoff = 100 * torch.finfo(fine.dtype).eps * written_norm
        # The actual installed/utilized factor, including conduction and STP,
        # must support this finite route. Check both inputs before and after it.
        minimum_speed = min(torch.linalg.svdvals(model.medium.current_transport_factor(
            state.medium, prepared))[..., -1].min()
            for state in (left, right, fine_left, fine_right))
        center_route_length = .375 - .125
        assert center_route_length / minimum_speed < ARRIVAL_INTERVAL
        assert fine.norm() > roundoff
        assert coarse.norm() > .01 * written_norm
        assert coarse.norm() > (fine - coarse).norm()
        model.solver_max_step = SOLVER_MAX_STEP / 8
        refined_left, _ = model.advance(left, ARRIVAL_INTERVAL, prepared=prepared, diagnostics=False)
        refined_right, _ = model.advance(right, ARRIVAL_INTERVAL, prepared=prepared, diagnostics=False)
        refined = probe_difference(model, refined_left, refined_right)
        assert (refined - fine).norm() < .1 * refined.norm()
        fine_left, fine_right = refined_left, refined_right
        # Subsequent observations are identical. Only the older input differs;
        # its response must continue through the same persistent physical state.
        for token in (3, 4, 5, 6):
            fine_left, _ = model.assimilate(fine_left, torch.tensor([token]), diagnostics=False)
            fine_right, _ = model.assimilate(fine_right, torch.tensor([token]), diagnostics=False)
            fine_left, _ = model.advance(fine_left, ARRIVAL_INTERVAL, prepared=prepared, diagnostics=False)
            fine_right, _ = model.advance(fine_right, ARRIVAL_INTERVAL, prepared=prepared, diagnostics=False)
        assert probe_difference(model, fine_left, fine_right).norm() > roundoff
        torch.testing.assert_close(fine_left.medium.elapsed,
            torch.tensor([5 * ARRIVAL_INTERVAL], dtype=torch.float64), atol=2e-16, rtol=0)
        torch.testing.assert_close(fine_left.temporal.elapsed, fine_left.medium.elapsed,
                                   atol=2e-16, rtol=0)


def test_dynamic_read_uses_actual_incident_flux_halo_and_no_remote_state():
    model = nearby_model()
    with torch.no_grad():
        model.readout.motion_merge.weight.normal_(std=.04)
        model.readout.motion_policy.weight.normal_(std=.03)
    initial = model.initial_belief()
    field = (torch.randn_like(initial.medium.field) * .03).requires_grad_()
    flux = tuple((torch.randn_like(field) * .02).requires_grad_() for _ in range(3))
    belief = replace(initial, medium=replace(initial.medium, field=field, flux=flux))
    feature = model.read(belief, decode=False)[0]
    gradients = torch.autograd.grad(feature.square().sum(), (field, *flux))
    support = (model.readout.weights().sum(0) > 0).reshape(model.medium.shape)
    halo = support.clone()
    for axis in range(3):
        halo |= torch.roll(support, -1, axis)
    assert torch.count_nonzero(gradients[0][:, ~support]) == 0
    for gradient in gradients[1:]:
        assert torch.count_nonzero(gradient[:, ~halo]) == 0
    incident = halo & ~support
    assert sum(gradient[:, incident].abs().sum() for gradient in gradients[1:]) > 1e-10


def test_grid_refinement_preserves_physical_arrival_and_apertures():
    radius = (.07, .13, .13)
    coarse = numerical_model(shape=(8, 4, 4), structure=False, clock=False,
                             port_radius=radius)
    fine = numerical_model(shape=(16, 8, 8), structure=False, clock=False,
                           port_radius=radius, solver_step=SOLVER_MAX_STEP / 2)
    with torch.no_grad():
        for model in (coarse, fine):
            model.write_agent.local_ports.centers.copy_(torch.tensor([[.2, .2, .2]]))
            model.readout.probe_coords.copy_(torch.tensor([[[.6, .2, .2]]]))
        outputs = [model.advance(model.initial_belief(), ARRIVAL_INTERVAL,
                                 diagnostics=True) for model in (coarse, fine)]
    assert coarse.write_agent.local_ports.physical_radius == fine.write_agent.local_ports.physical_radius
    assert coarse.readout.physical_radius == fine.readout.physical_radius
    torch.testing.assert_close(coarse.write_agent.local_ports.centers,
                               fine.write_agent.local_ports.centers, atol=0, rtol=0)
    torch.testing.assert_close(coarse.readout.probe_coords, fine.readout.probe_coords, atol=0, rtol=0)
    for state, info in outputs:
        torch.testing.assert_close(state.medium.elapsed,
            torch.tensor([ARRIVAL_INTERVAL], dtype=torch.float64), atol=0, rtol=0)
        torch.testing.assert_close(state.temporal.elapsed, state.medium.elapsed, atol=0, rtol=0)
        assert info['intrinsic_duration'] == ARRIVAL_INTERVAL
    assert outputs[0][1]['solver_steps'] == 4
    assert outputs[1][1]['solver_steps'] == 8


def test_short_cadence_bptt32_resume_preserves_pending_adam_and_physical_life():
    model = numerical_model()
    optimizer = make_medium_optimizer(model, lr=2e-4)
    live = ActiveMediumTrainer(model, optimizer, model.initial_belief(), carry_token=1,
        event_duration=ARRIVAL_INTERVAL, chunk_tokens=32, tokens_per_update=32,
        activation_checkpointing=True)
    targets = torch.arange(64) % 12 + 1
    prior = torch.full((13,), math.log(13), dtype=torch.float64)
    live.consume(targets[:35], phase='numerical_contract', prior_nll=prior)
    assert live.optimizer_updates == 1 and live.pending == 3
    assert live.optimizer.state and model.medium.structural_posterior.window_active
    saved = dict(optimizer=copy.deepcopy(live.optimizer.state_dict()),
                 learner=copy.deepcopy(live.state_dict()), belief=training.clone_belief(live.belief))
    restored_model = copy.deepcopy(model)
    restored_optimizer = make_medium_optimizer(restored_model, lr=2e-4,
                                               saved_state=saved['optimizer'])
    restored = ActiveMediumTrainer(restored_model, restored_optimizer, saved['belief'],
        carry_token=1, event_duration=ARRIVAL_INTERVAL, chunk_tokens=32,
        tokens_per_update=32, activation_checkpointing=True)
    restored.load_state_dict(saved['learner'])
    assert_tree_equal(live.state_dict(), restored.state_dict())
    rng = torch.get_rng_state()
    expected = live.consume(targets[35:], phase='numerical_contract', prior_nll=prior)
    torch.set_rng_state(rng)
    actual = restored.consume(targets[35:], phase='numerical_contract', prior_nll=prior)
    assert expected == actual
    assert live.optimizer_updates == restored.optimizer_updates == 2
    assert live.events == restored.events == 64 and live.pending == restored.pending == 0
    assert_tree_equal(live.model.state_dict(), restored.model.state_dict(), 'model')
    assert_tree_equal(live.optimizer.state_dict(), restored.optimizer.state_dict(), 'AdamW')
    assert_tree_equal(live.state_dict(), restored.state_dict(), 'learner')
    assert_tree_equal(training.belief_tensors(live.belief),
                      training.belief_tensors(restored.belief), 'physical_state')
    torch.testing.assert_close(live.belief.medium.elapsed, live.belief.temporal.elapsed,
                               atol=5e-14, rtol=0)
    posterior = live.model.medium.structural_posterior
    assert posterior.evidence_events == 64 and posterior.windows_committed == 2
    torch.testing.assert_close(posterior.elapsed, live.belief.medium.elapsed.squeeze(),
                               atol=5e-14, rtol=0)
