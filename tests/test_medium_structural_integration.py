"""Coupled numerical contracts; no language-performance conclusions."""
from dataclasses import replace
import copy

import torch
import pytest

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.training import quiet_training_chunk, clone_belief
from information_boltzmann.runtime.optimization import make_medium_optimizer
from information_boltzmann.runtime.active_medium_training import ActiveMediumTrainer


def candidate():
    torch.manual_seed(193)
    return PlasticMediumPorts3D(vocab_size=13, shape=(2, 2, 2), channels=8,
        material_width=3, hidden=4, heads=1, queries=1, anisotropic_transport=True,
        bath_type='conductance', short_term_plasticity=True, activity_adaptation=True,
        material_reference_shape=None, read_mode='temporal', temporal_rates=[1., 4.],
        temporal_frequencies=[0., 3.], temporal_time_reference=.02,
        intrinsic_time_reference=.023, solver_max_step=.01, observer_max_step=.02,
        structure_options=dict(resource_density=4., speed_reference=4., structure_time=1.,
            prior_std=.2, initial_std=.2, maintenance_supply=2., initial_dual=.1)).double()


def test_structural_capacity_bounds_fast_gain_energy_and_no_rescaling_escape():
    net = candidate()
    posterior = net.medium.structural_posterior
    posterior.begin_window(torch.randn_like(posterior.mean))
    state = net.initial_belief().medium
    state = replace(state, field=torch.randn_like(state.field),
        flux=tuple(torch.randn_like(x) for x in state.flux),
        conduction=torch.full_like(state.conduction, 100.))
    prepared = net.medium.prepare_evolution()
    factor = net.medium.current_transport_factor(state, prepared)
    installed = prepared.structural_factor
    assert (factor.norm(dim=-1) <= installed.norm(dim=-1)[None] + 1e-12).all()
    torch.testing.assert_close(prepared.structural_allocation.sum(-1),
                               torch.full((2, 2, 2), 4., dtype=torch.float64))
    before = net.medium.energy(state)
    out = net.medium.transport(state, .17, structural_factor=installed)
    torch.testing.assert_close(net.medium.energy(out), before, atol=1e-12, rtol=1e-12)
    assert not net.medium.log_speed.weight.requires_grad
    with torch.no_grad():
        net.medium.log_speed.weight.add_(100.)
    changed = net.medium.current_transport_factor(state)
    torch.testing.assert_close(changed, factor, atol=0, rtol=0)
    loss = out.field.square().sum()
    g = torch.autograd.grad(loss, (posterior.mean, posterior.log_std))
    assert all(torch.isfinite(v).all() and v.norm() > 0 for v in g)


def test_writer_auxiliary_cannot_train_structural_posterior_or_medium():
    net = candidate()
    q = net.medium.structural_posterior
    q.begin_window(torch.randn_like(q.mean))
    loss, _, task, components = quiet_training_chunk(net, torch.tensor([[1, 2]]),
        torch.tensor([[2, 3]]), net.initial_belief(), event_duration=.005,
        return_loss_components=True, activation_checkpointing=True)
    p = tuple(net.medium.parameters())
    p = tuple(v for v in p if v.requires_grad)
    joint_gradient = torch.autograd.grad(loss, p, retain_graph=True, allow_unused=True)
    prepared = net.medium.prepare_evolution()
    maintenance = q.maintenance(prepared.structural_allocation, 1 / 8)
    expected = q.objective(task, maintenance, 2)
    task_gradient = torch.autograd.grad(expected, p, allow_unused=True)
    for a, b in zip(joint_gradient, task_gradient):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)
    assert components['structural_kl'] >= 0
    assert q.evidence_events == 0  # pure computation never commits evidence


def test_writer_auxiliary_block_gradient_is_exact():
    net = candidate()
    q = net.medium.structural_posterior
    q.begin_window(torch.randn_like(q.mean))
    ids, targets = torch.tensor([[1, 2]]), torch.tensor([[2, 3]])
    loss, _, task = quiet_training_chunk(net, ids, targets, net.initial_belief(),
                                         event_duration=.005, activation_checkpointing=True)
    writer = tuple(p for name, p in net.learning_named_parameters()
                   if name.startswith(('source.', 'write_agent.')))
    actual = torch.autograd.grad(loss, writer, retain_graph=True, allow_unused=True)
    # Disable only the structural routing switch, keeping the same already
    # prepared physical law/sample in the event execution for this reference.
    from information_boltzmann.runtime.training import training_event
    from torch.nn import functional as F
    belief, objectives = net.initial_belief(), []
    table = F.normalize(net.source.embedding.weight, dim=-1)
    prepared = net.medium.prepare_evolution()
    for token in ids.unbind(1):
        _, belief, info, _, _ = training_event(net, belief, token, table, prepared,
                                               .005, 1, False, True)
        objectives.append(info['_write_free_energy'])
    reference = task + torch.stack(objectives).mean()
    expected = torch.autograd.grad(reference, writer, allow_unused=True)
    for a, b in zip(actual, expected):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)


def test_structural_individual_and_runtime_law_cannot_change_silently():
    from information_boltzmann.runtime.continuous import ContinuousStream
    net = candidate()
    with pytest.raises(ValueError, match='one persistent individual'):
        net.initial_belief(2)
    stream = ContinuousStream(net, max_step=.02)
    saved = stream.state_dict()
    altered = copy.deepcopy(net)
    altered.medium.structural_posterior = None
    with pytest.raises(ValueError, match='structural propagation law'):
        ContinuousStream.from_state_dict(altered, saved)
    altered = copy.deepcopy(net)
    altered.medium.structural_posterior.maintenance_supply.mul_(2)
    with pytest.raises(ValueError, match='structural propagation law'):
        ContinuousStream.from_state_dict(altered, saved)


def test_structural_sampling_invalidates_inference_coefficient_cache():
    from information_boltzmann.runtime.continuous import ContinuousStream
    net = candidate()
    stream = ContinuousStream(net, max_step=.02)
    with torch.no_grad():
        before = stream._coefficients().structural_factor.clone()
        q = net.medium.structural_posterior
        q.begin_window(torch.ones_like(q.mean))
        after = stream._coefficients().structural_factor
        direct = net.medium.prepare_evolution().structural_factor
        assert not torch.equal(before, after)
        torch.testing.assert_close(after, direct, atol=0, rtol=0)


def test_pending_structural_window_resume_preserves_next_update_and_evidence():
    a = candidate()
    oa = make_medium_optimizer(a, lr=2e-4)
    la = ActiveMediumTrainer(a, oa, a.initial_belief(), carry_token=1,
        event_duration=.005, chunk_tokens=1, tokens_per_update=2, activation_checkpointing=True)
    prior = torch.ones(13)
    la.consume(torch.tensor([2]), phase='train_first_pass', prior_nll=prior)
    assert la.pending == 1 and a.medium.structural_posterior.window_active
    b = copy.deepcopy(a)
    ob = make_medium_optimizer(b, lr=2e-4, saved_state=copy.deepcopy(oa.state_dict()))
    lb = ActiveMediumTrainer(b, ob, clone_belief(la.belief), carry_token=1,
        event_duration=.005, chunk_tokens=1, tokens_per_update=2, activation_checkpointing=True)
    lb.load_state_dict(copy.deepcopy(la.state_dict()))
    ra = la.consume(torch.tensor([3]), phase='fresh_B', prior_nll=prior)
    rb = lb.consume(torch.tensor([3]), phase='fresh_B', prior_nll=prior)
    assert ra == rb and la.optimizer_updates == lb.optimizer_updates == 1
    for (name, p), (_, r) in zip(a.state_dict().items(), b.state_dict().items()):
        torch.testing.assert_close(p, r, atol=0, rtol=0, msg=name)
    q = a.medium.structural_posterior
    assert q.evidence_events == 2 and q.windows_committed == 1
    torch.testing.assert_close(q.elapsed, la.belief.medium.elapsed.squeeze(), atol=0, rtol=0)
