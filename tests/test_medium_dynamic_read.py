"""Numerical and interface contracts; no capability training."""
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from information_boltzmann.core.plastic_medium import PlasticMedium3D
from information_boltzmann.core.plastic_ports import PlasticBelief, PlasticMediumPorts3D
from information_boltzmann.runtime.continuous import ContinuousStream
from information_boltzmann.runtime.optimization import initialize_dynamic_read_branch
from information_boltzmann.runtime.training import quiet_training_chunk


def net(mode='dynamic', **kwargs):
    return PlasticMediumPorts3D(vocab_size=29, shape=(4, 4, 4), channels=8,
        material_width=3, hidden=6, heads=2, queries=2, bath_type='conductance',
        short_term_plasticity=True, activity_adaptation=True, read_mode=mode,
        **kwargs).double()


def random_belief(model, batch=1):
    empty = model.initial_belief(batch)
    state = empty.medium
    state = replace(state, field=torch.randn_like(state.field),
        flux=tuple(torch.randn_like(x) for x in state.flux),
        conduction=torch.randn_like(state.conduction) * .2,
        receptors=torch.rand_like(state.receptors),
        transmission=.1 + .8 * torch.rand_like(state.transmission))
    return PlasticBelief(state, empty.precision)


@pytest.mark.parametrize('bath', ['conductance', 'quadratic'])
def test_analytic_rhs_and_parameter_vjp_match_zero_time_jvp(bath):
    torch.manual_seed(704)
    model = PlasticMedium3D((4, 4, 4), 8, 3, 6, 3,
        bath_type=bath, adaptive_conduction=True, short_term_plasticity=True).double()
    with torch.no_grad():
        model.material.coefficients.normal_(std=.2)
    state = model.initial_state()
    state = replace(state, field=torch.randn_like(state.field),
        flux=tuple(torch.randn_like(x) for x in state.flux),
        conduction=torch.randn_like(state.conduction) * .2,
        receptors=None if state.receptors is None else torch.rand_like(state.receptors),
        transmission=.1 + .8 * torch.rand_like(state.transmission))
    zero = torch.tensor(0., dtype=torch.float64)
    _, oracle = torch.autograd.functional.jvp(
        lambda t: model.native_advance(state, t, diagnostics=False)[0].field,
        zero, torch.ones_like(zero), create_graph=True)
    analytic = model.field_rhs(state)
    torch.testing.assert_close(analytic, oracle, atol=3e-12, rtol=3e-12)
    direction = torch.randn_like(analytic)
    parameters = list(model.parameters())
    a = torch.autograd.grad((analytic * direction).sum(), parameters, allow_unused=True)
    b = torch.autograd.grad((oracle * direction).sum(), parameters, allow_unused=True)
    for left, right in zip(a, b):
        if left is None or right is None:
            other = right if left is None else left
            assert other is None or other.abs().max() < 1e-11
        else:
            torch.testing.assert_close(left, right, atol=2e-10, rtol=2e-10)


def test_legacy_migration_exact_logits_and_live_new_ce_gradients():
    torch.manual_seed(705)
    old, new = net('instantaneous'), net()
    initialize_dynamic_read_branch(new, old.state_dict())
    belief = random_belief(old)
    before, _ = old.read(belief)
    after, _ = new.read(belief)
    torch.testing.assert_close(after, before, atol=0, rtol=0)
    F.cross_entropy(after, torch.tensor([7])).backward()
    for name in ('motion_policy', 'motion_keys', 'motion_merge'):
        gradient = getattr(new.readout, name).weight.grad
        assert gradient is not None and gradient.isfinite().all() and gradient.norm() > 0
    with pytest.raises(ValueError, match='exact instantaneous'):
        initialize_dynamic_read_branch(new, new.state_dict())


def test_fresh_read_sees_hidden_flux_without_advancing_state():
    torch.manual_seed(706)
    model = net()
    belief = random_belief(model)
    changed = PlasticBelief(replace(belief.medium,
        flux=tuple(-x for x in belief.medium.flux)), belief.precision)
    original = [x.clone() for x in (belief.medium.field, *belief.medium.flux, belief.medium.elapsed)]
    first, _ = model.read(belief, decode=False)
    second, _ = model.read(changed, decode=False)
    assert (first - second).norm() > 1e-6
    for actual, expected in zip((belief.medium.field, *belief.medium.flux, belief.medium.elapsed), original):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    zero, _ = model.read(model.initial_belief())
    assert zero.isfinite().all()


def test_selected_probe_has_only_local_and_incident_edge_state_gradients():
    torch.manual_seed(707)
    model = net()
    belief = random_belief(model)
    state = belief.medium
    state = replace(state, field=state.field.requires_grad_(),
        flux=tuple(x.requires_grad_() for x in state.flux),
        conduction=state.conduction.requires_grad_(), receptors=state.receptors.requires_grad_(),
        transmission=state.transmission.requires_grad_())
    # Isolate query 0, head 0 at the existing final mixing operation. Their
    # policy is independent of other probes and shared writer precision.
    with torch.no_grad():
        model.readout.merge.weight.zero_()
        model.readout.motion_merge.weight.zero_()
        model.readout.motion_merge.weight[0, :model.readout.head_dim] = 1
    feature, _ = model.read(PlasticBelief(state, belief.precision), decode=False)
    leaves = [state.field, *state.flux, state.conduction, state.receptors, state.transmission]
    gradients = torch.autograd.grad(feature[0, 0], leaves)
    sites = model.readout.footprint().reshape(
        model.readout.heads, model.readout.queries, *model.medium.shape)[0, 0] > 0
    for index, gradient in enumerate(gradients):
        if index in (0, 5):
            mask = sites[None, ..., None] if index == 0 else sites[None, ..., None, None]
        elif index in (1, 2, 3):
            axis = index - 1
            mask = (sites | torch.roll(sites, -1, axis))[None, ..., None]
        else:
            edge = torch.stack([sites | torch.roll(sites, -1, axis) for axis in range(3)], -1)
            mask = edge[None] if index == 4 else edge[None, ..., None]
        assert gradient.masked_select(~mask.expand_as(gradient)).abs().max() == 0
        assert gradient.norm() > 0


def test_chunk_causality_continuation_and_runtime_read_mode_guard():
    torch.manual_seed(708)
    model = net()
    ids, targets = torch.tensor([[2, 3, 4]]), torch.tensor([[3, 4, 5]])
    belief = model.initial_belief()
    loss, whole, _ = quiet_training_chunk(model, ids, targets, belief, event_duration=.005)
    forward_loss, reference, _ = model(ids, targets, belief, event_duration=.005)
    torch.testing.assert_close(loss, forward_loss)
    torch.testing.assert_close(whole.medium.field, reference.medium.field)
    _, altered, _ = quiet_training_chunk(model, ids, targets + 1, belief, event_duration=.005)
    torch.testing.assert_close(whole.medium.field, altered.medium.field, atol=0, rtol=0)
    loss.backward()
    assert model.medium.collision_rate[-1].weight.grad.norm() > 0
    stream = ContinuousStream(model.eval(), belief=whole.detach(), max_step=.005)
    with torch.no_grad():
        before = stream.read_current().value
        saved = stream.state_dict()
        restored = ContinuousStream.from_state_dict(model, saved)
        torch.testing.assert_close(before, restored.read_current().value, atol=0, rtol=0)
        stream.advance_to(stream.time + .01)
        restored.advance_to(restored.time + .01)
        torch.testing.assert_close(stream.read_current().value, restored.read_current().value)
    with pytest.raises(ValueError, match='read mode'):
        ContinuousStream.from_state_dict(net('instantaneous'), saved)
    changed_unit = net(response_time_reference=2)
    with pytest.raises(ValueError, match='time reference'):
        ContinuousStream.from_state_dict(changed_unit, saved)


def test_dynamic_mode_rejects_implicit_global_read_and_missing_motion():
    with pytest.raises(ValueError, match='compact'):
        net(port_scope='global')
    model = net()
    belief = model.initial_belief()
    with pytest.raises(ValueError, match='motion'):
        model.readout(belief.medium.field, belief.precision)


def test_full_graph_read_preserves_output_and_parameter_gradients():
    torch.manual_seed(710)
    model = net()
    belief = random_belief(model)
    compiled = torch.compile(model.native_read, backend='eager', fullgraph=True)
    original, _ = model.native_read(belief)
    reference = torch.autograd.grad(original.square().sum(), list(model.parameters()),
                                    allow_unused=True)
    actual, _ = compiled(belief)
    gradients = torch.autograd.grad(actual.square().sum(), list(model.parameters()),
                                    allow_unused=True)
    torch.testing.assert_close(actual, original, atol=0, rtol=0)
    for left, right in zip(reference, gradients):
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_runtime_dynamic_read_reuses_coefficients_and_keeps_time():
    model = net().eval()
    stream = ContinuousStream(model, max_step=.005)
    with torch.no_grad():
        stream.read_current()
        stream.read_current()
        assert stream.coefficient_preparations == 1
        assert stream.time == 0.
        model.medium.log_speed.weight.add_(.01)
        stream.read_current()
        assert stream.coefficient_preparations == 2


def test_mean_zero_motion_direction_is_in_production_subspace_and_lowers_local_ce():
    """An algebra/gradient identity, not a synthetic capability experiment."""
    torch.manual_seed(713)
    model = net()
    with torch.no_grad():
        for name in ('motion_policy', 'motion_keys', 'motion_merge'):
            getattr(model.readout, name).weight.zero_()
    belief = random_belief(model, batch=3)
    measurements, bases = [], []
    hooks = [model.readout.motion_merge.register_forward_pre_hook(
        lambda _module, values: measurements.append(values[0])),
        model.readout.correction.register_forward_pre_hook(
        lambda _module, values: bases.append(values[0]))]
    try:
        predicted, _ = model.read(belief)
    finally:
        for hook in hooks:
            hook.remove()
    labels = torch.tensor([3, 7, 11])
    loss = F.cross_entropy(predicted, labels)
    gradient, v = torch.autograd.grad(loss, (model.readout.motion_merge.weight, bases[0]))
    m = measurements[0].detach()
    torch.testing.assert_close(gradient, v.T @ m, atol=2e-13, rtol=2e-13)
    mu = m.mean(0)
    projected = gradient - (gradient @ mu)[:, None] * mu[None] / mu.square().sum()
    direction = -projected
    assert projected.norm() > 0
    torch.testing.assert_close(direction @ mu, torch.zeros(model.medium.channels, dtype=mu.dtype),
                               atol=2e-13, rtol=0)
    analytic = (gradient * direction).sum()
    torch.testing.assert_close(analytic, -projected.square().sum(), atol=2e-13, rtol=2e-13)
    offset = m @ direction.T
    epsilon = 2**-12 * bases[0].detach().square().mean().sqrt() / offset.square().mean().sqrt()
    with torch.no_grad():
        model.readout.motion_merge.weight.copy_(epsilon * direction)
        positive, _ = model.read(belief)
        positive_ce = F.cross_entropy(positive, labels)
        model.readout.motion_merge.weight.copy_(-epsilon * direction)
        negative, _ = model.read(belief)
        negative_ce = F.cross_entropy(negative, labels)
        model.readout.motion_merge.weight.zero_()
    central = (positive_ce - negative_ce) / (2 * epsilon)
    torch.testing.assert_close(central, analytic, atol=1e-9, rtol=1e-5)
    assert positive_ce < loss.detach()


def test_whole_continuous_clock_derivative_matches_replay_with_persistent_state():
    """Numerical trajectory contract, not a task/capability experiment."""
    torch.manual_seed(714)
    model = net('instantaneous')
    initial = random_belief(model)
    observed, targets = torch.tensor([2, 3, 4, 5]), torch.tensor([3, 4, 5, 6])
    table = F.normalize(model.source.embedding.weight, dim=-1)
    prepared = model.medium.prepare_evolution()

    def replay(log_scale, substeps=1):
        belief = initial
        duration = .005 * log_scale.exp()
        predictions = []
        for token in observed:
            belief, _ = model.assimilate(belief, token.reshape(1),
                token_features=table, diagnostics=False, training_terms=False)
            belief, _ = model.advance(belief, duration, substeps=substeps,
                prepared=prepared, diagnostics=False)
            predicted, _ = model.read(belief, prepared=prepared)
            predictions.append(predicted)
        return F.cross_entropy(torch.cat(predictions), targets), belief

    zero = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    baseline, final = replay(zero)
    analytic = torch.autograd.grad(baseline, zero)[0]
    assert analytic.abs() > 1e-10
    epsilon = torch.tensor(2**-14, dtype=torch.float64)
    with torch.no_grad():
        positive, plus = replay(epsilon)
        negative, minus = replay(-epsilon)
        _, refined = replay(zero.detach(), substeps=2)
    torch.testing.assert_close((positive - negative) / (2 * epsilon), analytic,
                               atol=1e-9, rtol=1e-5)
    for state, scale in ((plus, epsilon), (minus, -epsilon)):
        torch.testing.assert_close(state.medium.elapsed - initial.medium.elapsed,
                                   (.005 * scale.exp() * len(observed)).reshape(1))
    torch.testing.assert_close(refined.medium.elapsed, final.medium.elapsed, atol=0, rtol=0)
    assert (plus.medium.field - minus.medium.field).norm() > 0
