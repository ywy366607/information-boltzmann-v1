"""Numerical propagation and credit tests; these make no task-capability claim."""
from dataclasses import replace
import copy

import torch

from information_boltzmann.core.plastic_ports import PlasticBelief, PlasticMediumPorts3D
from information_boltzmann.core.temporal_probes import sample_compact_probes
from information_boltzmann.runtime.training import quiet_training_chunk


def model(temporal=False, clock=True):
    torch.manual_seed(251)
    net = PlasticMediumPorts3D(vocab_size=13, shape=(4, 2, 2), channels=4,
        material_width=2, hidden=4, heads=1, queries=1, port_modes=8,
        adaptive_conduction=False, anisotropic_transport=True,
        write_port_radius=(.13, .4, .4), read_port_radius=(.13, .4, .4),
        read_mode='temporal' if temporal else 'instantaneous',
        temporal_rates=[1., 4.] if temporal else None,
        temporal_frequencies=[0., 3.] if temporal else None,
        intrinsic_time_reference=.21 if clock else None,
        solver_max_step=.01, observer_max_step=.035).double()
    with torch.no_grad():
        net.medium.material.coefficients.normal_(std=.05)
        net.write_agent.local_ports.centers.copy_(torch.tensor([[0., .25, .25]]))
        net.readout.probe_coords.copy_(torch.tensor([[[.5, .25, .25]]]))
        # Isolate linear propagation for a known physical mechanism.
        for p in net.medium.collision_rate.parameters():
            p.zero_()
        for p in net.medium.bath_rate.parameters():
            p.zero_()
        net.medium.bath_rate[-1].bias.fill_(-100.)
    return net


def test_separated_ports_need_evolution_and_transport_with_credit():
    net = model()
    belief = net.initial_belief()
    assert not ((net.write_agent.local_ports.weights() > 0).any(0)
                & (net.readout.weights() > 0).any(0)).any()
    amplitude = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
    footprint = net.write_agent.local_ports.weights()[0].reshape(4, 2, 2)
    vector = torch.tensor([1., -1., .5, -.5], dtype=torch.float64)
    field = amplitude * footprint[None, ..., None] * vector
    written = replace(belief, medium=belief.medium.with_field(field))
    duration = net.event_time(belief, .005)
    from information_boltzmann.core.intrinsic_time import EvolutionSchedule
    schedule = EvolutionSchedule.for_duration(duration.detach() * 1.001,
        solver_max_step=.01, observer_max_step=.035, max_steps=4096)
    outgoing, _ = net.advance_interval(written, duration, schedule=schedule, diagnostics=False)
    remote = sample_compact_probes(net.readout, outgoing.medium.field).square().sum()
    immediate = sample_compact_probes(net.readout, written.medium.field).square().sum()
    tiny, _ = net.advance(written, .005, diagnostics=False)
    tiny_remote = sample_compact_probes(net.readout, tiny.medium.field).square().sum()
    assert immediate == 0
    assert remote > 1000 * tiny_remote
    d_amp, d_clock, d_material = torch.autograd.grad(remote, (
        amplitude, net.intrinsic_time.head.bias, net.medium.log_speed.weight))
    assert d_amp.abs() > 1e-8 and d_clock.abs().max() > 1e-8
    assert torch.isfinite(d_material).all() and d_material.abs().max() > 1e-10
    # A blocked carrier cannot turn elapsed time into an input shortcut.
    frozen, _ = net.medium.advance(written.medium, duration, substeps=21,
                                   transport=False, collision=False, bath=False)
    assert sample_compact_probes(net.readout, frozen.field).square().sum() == 0
    # Clock derivative agrees with direct physical-duration finite differences.
    epsilon = 1e-5
    values = []
    with torch.no_grad():
        for shift in (epsilon, -epsilon):
            out, _ = net.advance_interval(written,
                duration.detach() * torch.exp(torch.tensor(shift, dtype=torch.float64)),
                schedule=schedule, diagnostics=False)
            values.append(sample_compact_probes(net.readout, out.medium.field).square().sum())
    torch.testing.assert_close(d_clock.squeeze(), (values[0] - values[1]) / (2 * epsilon),
                               atol=2e-5, rtol=2e-3)


def test_interval_sampling_matches_explicit_continuation_and_clocks():
    net = model(temporal=True)
    b = net.initial_belief()
    written, _ = net.assimilate(b, torch.tensor([2]), diagnostics=False)
    from information_boltzmann.core.intrinsic_time import EvolutionSchedule
    duration = torch.tensor(.21, dtype=torch.float64, requires_grad=True)
    schedule = EvolutionSchedule.for_duration(duration, solver_max_step=.01,
                                              observer_max_step=.035, max_steps=4096)
    out, _ = net.advance_interval(written, duration, schedule=schedule, diagnostics=False)
    explicit = written
    for _ in range(schedule.observer_count):
        state, _ = net.medium.advance(explicit.medium, duration / schedule.observer_count,
                                      substeps=schedule.solver_substeps, diagnostics=False)
        explicit = net.complete_advance(explicit, state, duration / schedule.observer_count)
    torch.testing.assert_close(out.medium.field, explicit.medium.field)
    torch.testing.assert_close(out.temporal.value, explicit.temporal.value)
    torch.testing.assert_close(out.medium.elapsed, out.temporal.elapsed)
    assert torch.autograd.grad(out.temporal.value.real.square().sum(), duration)[0].abs() > 0


def test_labels_cannot_select_schedule_and_checkpoint_keeps_credit():
    a = model(temporal=True)
    b = copy.deepcopy(a)
    ids = torch.tensor([[2, 3]])
    loss, state, nll = quiet_training_chunk(a, ids, torch.tensor([[3, 4]]),
        a.initial_belief(), event_duration=.005)
    other, other_state, _ = quiet_training_chunk(b, ids, torch.tensor([[5, 6]]),
        b.initial_belief(), event_duration=.005, activation_checkpointing=True)
    torch.testing.assert_close(state.medium.field, other_state.medium.field)
    torch.testing.assert_close(state.medium.elapsed, other_state.medium.elapsed)
    torch.testing.assert_close(state.temporal.value, other_state.temporal.value)
    c = copy.deepcopy(a)
    checked, checked_state, checked_nll = quiet_training_chunk(c, ids, torch.tensor([[3, 4]]),
        c.initial_belief(), event_duration=.005, activation_checkpointing=True)
    nll.backward()
    checked_nll.backward()
    torch.testing.assert_close(loss, checked)
    for (name, p), (_, q) in zip(a.named_parameters(), c.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-10, rtol=2e-8, msg=name)
    assert a.intrinsic_time.head.bias.grad.abs().max() > 0


def test_endpoint_motion_reuse_preserves_forward_and_gradient():
    a = model(temporal=True, clock=False)
    b = a.initial_belief()
    b, _ = a.assimilate(b, torch.tensor([3]), diagnostics=False)
    out, _, motion = a.advance(b, .07, diagnostics=False, return_motion=True)
    reused, _ = a.read(out, decode=False, motion=motion)
    direct, _ = a.read(out, decode=False)
    torch.testing.assert_close(reused, direct, atol=0, rtol=0)
    parameter = a.medium.log_speed.weight
    g1 = torch.autograd.grad(reused.sum(), parameter, retain_graph=True)[0]
    g2 = torch.autograd.grad(direct.sum(), parameter)[0]
    torch.testing.assert_close(g1, g2, atol=2e-12, rtol=2e-12)


def test_actual_observer_refines_independently_of_same_physics():
    net = model(temporal=True, clock=False)
    written, _ = net.assimilate(net.initial_belief(), torch.tensor([2]), diagnostics=False)
    outputs = []
    for samples in (2, 4, 8, 32):
        net.solver_max_step = .2 / 64
        net.observer_max_step = .2 / samples
        out, _ = net.advance(written, .2, diagnostics=False)
        outputs.append(out)
    for out in outputs[:-1]:
        # 64 identical physical microsteps; only the history sampling differs.
        torch.testing.assert_close(out.medium.field, outputs[-1].medium.field,
                                   atol=1e-12, rtol=1e-12)
    errors = [(out.temporal.value - outputs[-1].temporal.value).abs().norm()
              for out in outputs[:-1]]
    assert errors[1] < errors[0] and errors[2] < errors[1], errors


def test_clock_uses_only_declared_preinput_read_aperture():
    net = model()
    with torch.no_grad():
        net.intrinsic_time.head.weight.fill_(.2)
    belief = net.initial_belief()
    outside = (net.readout.weights().sum(0) == 0).reshape(4, 2, 2)
    modified = replace(belief, medium=belief.medium.with_field(
        outside[None, ..., None].expand_as(belief.medium.field).double() * 100.))
    torch.testing.assert_close(net.event_time(belief, .005), net.event_time(modified, .005),
                               atol=0, rtol=0)


def test_intrinsic_conductance_stp_clock_and_runtime_continuation():
    from information_boltzmann.runtime.continuous import ContinuousStream
    net = PlasticMediumPorts3D(vocab_size=13, shape=(2, 2, 2), channels=8,
        material_width=2, hidden=4, heads=1, queries=1,
        bath_type='conductance', short_term_plasticity=True, activity_adaptation=True,
        anisotropic_transport=True, read_mode='temporal', temporal_rates=[1., 4.],
        temporal_frequencies=[0., 3.], intrinsic_time_reference=.023,
        solver_max_step=.01, observer_max_step=.02).double()
    with torch.no_grad():
        net.medium.material.coefficients.normal_(std=.1)
    stream = ContinuousStream(net, max_step=.03)
    response, _ = stream.observe_event(torch.tensor([2]), event_duration=.005)
    response.value.sum().backward()
    assert net.intrinsic_time.head.bias.grad.abs().max() > 0
    assert net.medium.short_term_plasticity.parameters_map.weight.grad.abs().max() > 0
    assert net.medium.conductance_response.log_parameters.weight.grad.abs().max() > 0
    payload = stream.state_dict()
    restored = ContinuousStream.from_state_dict(copy.deepcopy(net), payload)
    with torch.no_grad():
        a, _ = stream.observe_event(torch.tensor([3]), event_duration=.005)
        b, _ = restored.observe_event(torch.tensor([3]), event_duration=.005)
    torch.testing.assert_close(a.value, b.value, atol=0, rtol=0)
    torch.testing.assert_close(stream.belief.temporal.value, restored.belief.temporal.value,
                               atol=0, rtol=0)
