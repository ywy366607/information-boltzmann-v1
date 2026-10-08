"""Numerical/interface audits, not capability or convergence experiments."""
import copy

import pytest
import torch

from information_boltzmann.core.plastic_medium import (
    ContinuousMaterial, MediumState, PlasticMedium3D, boundary_scatter,
)


def random_state(model, batch=2):
    empty = model.initial_state(batch)
    return MediumState(torch.randn_like(empty.field),
                       tuple(torch.randn_like(x) for x in empty.flux), empty.elapsed)


def make_model(shape=(4, 4, 2)):
    torch.manual_seed(301)
    model = PlasticMedium3D(shape, channels=4, material_width=3, hidden=12).double()
    with torch.no_grad():
        model.material.coefficients.normal_(std=0.2)
    return model


def test_uniform_initial_material_and_periodic_continuous_coordinates():
    material = ContinuousMaterial(width=3).double()
    points = torch.randn(17, 3, dtype=torch.float64)
    assert torch.count_nonzero(material(points)) == 0
    with torch.no_grad():
        material.coefficients.normal_()
    shift = torch.tensor([2, -3, 1], dtype=torch.float64)
    torch.testing.assert_close(material(points), material(points + shift), atol=2e-14, rtol=2e-14)


def test_local_transport_preserves_energy_and_each_content_integral():
    model = make_model()
    state = random_state(model)
    output = model.transport(state, 0.023)
    torch.testing.assert_close(model.energy(output), model.energy(state), atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(output.field.sum((1, 2, 3)), state.field.sum((1, 2, 3)),
                               atol=2e-13, rtol=2e-13)


def test_transport_is_linear_for_fixed_material_and_moves_stored_flux():
    model = make_model()
    a, b = random_state(model), random_state(model)
    combined = MediumState(0.3 * a.field - 0.7 * b.field,
                           tuple(0.3 * x - 0.7 * y for x, y in zip(a.flux, b.flux)), a.elapsed)
    left, right = model.transport(a, 0.017), model.transport(b, 0.017)
    actual = model.transport(combined, 0.017)
    torch.testing.assert_close(actual.field, 0.3 * left.field - 0.7 * right.field)
    for z, x, y in zip(actual.flux, left.flux, right.flux):
        torch.testing.assert_close(z, 0.3 * x - 0.7 * y)
    empty = model.initial_state()
    stored = list(empty.flux)
    stored[0] = torch.zeros_like(empty.field)
    stored[0][0, 0, 0, 0, 0] = 1
    arrived = model.transport(MediumState(empty.field, tuple(stored), empty.elapsed), 0.017)
    assert arrived.field.norm() > 0


def test_collision_preserves_local_energy_and_four_linear_invariants():
    model = make_model()
    state = random_state(model)
    actual = model.collide(state, 0.17)
    before, after = model._pack(state), model._pack(actual)
    torch.testing.assert_close(after.sum(-1), before.sum(-1), atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(after.square().sum((-2, -1)), before.square().sum((-2, -1)),
                               atol=3e-13, rtol=3e-13)
    assert (actual.field - state.field).norm() > 0
    # The generator sees both state content and a continuous spatial material.
    same_state = state.field[:, :1, :1, :1].expand_as(state.field)
    heterogeneous = MediumState(same_state, tuple(same_state for _ in range(3)), state.elapsed)
    result = model.collide(heterogeneous, 0.1)
    assert result.field.flatten(1, 3).var(1).sum() > 0


def test_bath_is_passive_and_closes_energy_account():
    model = make_model()
    state = random_state(model)
    output, released = model.dissipate(state, 0.23)
    assert torch.all(released > 0)
    assert torch.all(model.energy(output) < model.energy(state))
    torch.testing.assert_close(model.energy(output) + released, model.energy(state))


def test_uniform_bath_matches_exact_radial_flow_and_semigroup():
    model = make_model()
    for parameter in model.bath_rate.parameters():
        with torch.no_grad():
            parameter.zero_()
    state = random_state(model)
    value = model._pack(state)
    rate = torch.nn.functional.softplus(model.bath_bias)
    expected = value / (1 + 2 * 0.3 * rate * value.square().mean((-2, -1), keepdim=True)).sqrt()
    whole, _ = model.dissipate(state, 0.3)
    half, _ = model.dissipate(state, 0.15)
    halves, _ = model.dissipate(half, 0.15)
    torch.testing.assert_close(model._pack(whole), expected, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(model._pack(whole), model._pack(halves), atol=2e-14, rtol=2e-14)


def test_joint_advance_accounts_for_bath_and_per_batch_physical_time():
    model = make_model()
    state = random_state(model)
    times = torch.tensor([0.12, 0.07], dtype=torch.float64)
    result, diag = model.advance(state, times, substeps=3)
    torch.testing.assert_close(result.elapsed, times)
    torch.testing.assert_close(diag['energy_after'] + diag['bath_out_energy'],
                               diag['energy_before'], atol=3e-13, rtol=3e-13)
    for resolution in (1, 4):
        result, _ = model.advance(state, 0.1, substeps=resolution)
        torch.testing.assert_close(result.elapsed, torch.full_like(result.elapsed, 0.1))


def test_port_scatter_preserves_inflight_state_and_total_energy():
    model = make_model()
    state = random_state(model)
    packet = torch.randn_like(state.field)
    angle = torch.randn_like(packet)
    output, reflected = boundary_scatter(state, packet, angle)
    volume = 1.0 / 32
    torch.testing.assert_close(model.energy(output) + 0.5 * volume * reflected.square().flatten(1).sum(1),
                               model.energy(state) + 0.5 * volume * packet.square().flatten(1).sum(1))
    assert output.flux is state.flux
    assert output.elapsed is state.elapsed


def test_full_gradient_to_state_and_all_material_and_operator_parameters():
    model = make_model()
    original = random_state(model)
    state = MediumState(original.field.requires_grad_(),
                        tuple(x.requires_grad_() for x in original.flux), original.elapsed)
    output, _ = model.advance(state, 0.025, substeps=2)
    loss = sum((x * torch.randn_like(x)).sum() for x in (output.field, *output.flux))
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.norm() > 0, name
    for x in (state.field, *state.flux):
        assert x.grad is not None and torch.isfinite(x.grad).all()


def test_checkpoint_continuation_strict_grid_transfer_and_no_implicit_reset():
    model = make_model()
    state = random_state(model)
    first, _ = model.advance(state, 0.02)
    other = make_model()
    other.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    expected, _ = model.advance(first, 0.03)
    actual, _ = other.advance(first.detach(), 0.03)
    torch.testing.assert_close(actual.field, expected.field)
    for x, y in zip(actual.flux, expected.flux):
        torch.testing.assert_close(x, y)
    torch.testing.assert_close(actual.elapsed, expected.elapsed)
    finer = make_model((8, 4, 4))
    finer.load_state_dict(model.state_dict(), strict=True)
    assert finer.material_field().shape == (8, 4, 4, 3)
    result, _ = finer.advance(finer.initial_state(), 0.02)
    assert result.field.shape == (1, 8, 4, 4, 4)


def test_refining_solver_converges_at_fixed_duration():
    model = make_model()
    state = random_state(model, batch=1)
    reference, _ = model.advance(state, 0.08, substeps=128, collision=False, bath=False)
    errors = []
    for resolution in (4, 16):
        actual, _ = model.advance(state, 0.08, substeps=resolution, collision=False, bath=False)
        errors.append((model._pack(actual) - model._pack(reference)).norm())
    assert errors[1] < errors[0] / 2


def test_zero_duration_identity_and_explicit_detach():
    model = make_model()
    state = random_state(model)
    actual, _ = model.advance(state, 0.0)
    torch.testing.assert_close(model._pack(actual), model._pack(state), atol=2e-14, rtol=2e-14)
    linked = MediumState(state.field.requires_grad_(), state.flux, state.elapsed)
    assert linked.detach().field.requires_grad is False
    torch.testing.assert_close(linked.detach().field, linked.field)
    with pytest.raises(ValueError):
        model.advance(state, -1.0)
    with pytest.raises(ValueError):
        model.advance(state, 0.1, substeps=0)


def test_compile_cpu_forward_and_backward_equivalence():
    # CPU tracing only: keep the GPU free for the independent fly training.
    model = make_model()
    compiled = torch.compile(copy.deepcopy(model), backend='eager', fullgraph=True)
    state = random_state(model, batch=1)
    expected, _ = model.advance(state, 0.02)
    actual, _ = compiled(state, 0.02)
    torch.testing.assert_close(model._pack(actual), model._pack(expected))
    model._pack(actual).square().sum().backward()
    assert compiled.material.coefficients.grad is not None


def test_state_and_physical_duration_gradient_matches_finite_difference():
    model = PlasticMedium3D((2, 2, 2), channels=2, material_width=2, hidden=4).double()
    state = random_state(model, batch=1)
    values = tuple(x.detach().requires_grad_() for x in (state.field, *state.flux))
    time = torch.tensor(0.013, dtype=torch.float64, requires_grad=True)

    def evaluate(f, x, y, z, duration):
        candidate = MediumState(f, (x, y, z), state.elapsed)
        output, _ = model(candidate, duration)
        return model._pack(output)

    assert torch.autograd.gradcheck(evaluate, (*values, time), fast_mode=True,
                                   atol=2e-5, rtol=2e-4)
