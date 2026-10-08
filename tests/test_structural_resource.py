"""Numerical structure/content contracts; no synthetic capability training."""
from dataclasses import replace

import pytest
import torch

from information_boltzmann.core.plastic_medium import PlasticMedium3D
from information_boltzmann.core.structural_resource import (
    TransportCapacityBudget, transport_capacity)


def test_budget_is_grid_independent_and_individual_specific():
    a = torch.eye(3, dtype=torch.float64).expand(2, 2, 2, 2, 3, 3).clone()
    a[1] *= 2
    budget = TransportCapacityBudget(3.)
    torch.testing.assert_close(transport_capacity(a), torch.tensor([3., 12.]).double())
    out = budget(a)
    torch.testing.assert_close(transport_capacity(out), torch.tensor([3., 3.]).double())
    torch.testing.assert_close(out[0], a[0])
    torch.testing.assert_close(transport_capacity(budget(a.repeat_interleave(2, 1))),
                               transport_capacity(out))


def test_projection_preserves_spd_anisotropy_and_gradient():
    torch.manual_seed(7)
    a = torch.randn(1, 2, 2, 2, 3, 3, dtype=torch.float64)
    a = (a + 3 * torch.eye(3)).requires_grad_()
    budget = TransportCapacityBudget(3.)
    out = budget(a)
    assert torch.linalg.eigvalsh(out @ out.transpose(-1, -2)).min() > 0
    scale = out.flatten()[0] / a.flatten()[0]
    torch.testing.assert_close(out, a * scale)
    assert torch.autograd.gradcheck(budget, (a,), eps=1e-6)
    small = (.1 * a.detach()).requires_grad_()
    assert torch.autograd.gradcheck(budget, (small,), eps=1e-6)
    # A common radial increase cannot buy more capacity above the budget.
    objective = (out * torch.randn_like(out)).sum()
    gradient, = torch.autograd.grad(objective, a)
    assert abs((gradient * a).sum().item()) < 1e-12


def test_budget_competes_between_sites_instead_of_erasing_activity():
    a = (2 * torch.eye(3, dtype=torch.float64)).expand(1, 2, 2, 2, 3, 3).clone()
    a.requires_grad_()
    budget = TransportCapacityBudget(3.)
    out = budget(a)
    # Numerical sensitivity of one coefficient, not a task or learned reward.
    gradient, = torch.autograd.grad(out[0, 0, 0, 0, 0, 0], a)
    assert gradient[0, 0, 0, 0, 0, 0] > 0
    assert gradient[0, 1, 1, 1, 0, 0] < 0


def test_zero_factor_and_invalid_budget():
    a = torch.zeros(1, 2, 2, 2, 3, 3, dtype=torch.float64, requires_grad=True)
    TransportCapacityBudget(3.)(a).sum().backward()
    assert torch.isfinite(a.grad).all()
    for value in (0., -1., float('inf'), float('nan')):
        with pytest.raises(ValueError):
            TransportCapacityBudget(value)


def test_budget_alone_does_not_select_concentrated_over_uniform_structure():
    uniform = torch.eye(3, dtype=torch.float64).expand(1, 2, 2, 2, 3, 3).clone()
    concentrated = uniform / 4
    # Same total capacity, with one strong site and seven weak, strictly SPD sites.
    concentrated[0, 0, 0, 0] = uniform[0, 0, 0, 0] * (8 - 7 / 16) ** .5
    torch.testing.assert_close(transport_capacity(uniform), transport_capacity(concentrated))
    assert torch.linalg.eigvalsh(concentrated @ concentrated.transpose(-1, -2)).min() > 0
    budget = TransportCapacityBudget(transport_capacity(uniform).item())
    torch.testing.assert_close(budget(uniform), uniform)
    torch.testing.assert_close(budget(concentrated), concentrated)


def test_material_change_preserves_content_energy_but_changes_distinctions():
    torch.manual_seed(12)
    torch.set_num_threads(1)
    net = PlasticMedium3D((2, 2, 2), channels=4, hidden=8,
                         anisotropic_transport=True, adaptive_conduction=True,
                         bath_type='conductance', activity_adaptation=True,
                         short_term_plasticity=True).double()
    state = net.initial_state()
    state = replace(state, field=torch.randn_like(state.field),
                    flux=tuple(torch.randn_like(x) for x in state.flux))
    saved = [x.clone() for x in (state.field, *state.flux)]
    other = {name: getattr(state, name).clone() for name in
             ('elapsed', 'conduction', 'receptors', 'transmission')}
    energy = net.energy(state).clone()
    material = net.material_field()
    speed = torch.ones(1, 2, 2, 2, 3, dtype=torch.float64)
    budget = TransportCapacityBudget(3.)
    before = budget(net.transport_factor(material, speed)).detach()
    with torch.no_grad():
        net.transport_shear.bias.copy_(torch.tensor([.4, -.3, .5]))
    after = budget(net.transport_factor(material, speed)).detach()
    for old, current in zip(saved, (state.field, *state.flux)):
        torch.testing.assert_close(old, current, atol=0, rtol=0)
    torch.testing.assert_close(net.energy(state), energy, atol=0, rtol=0)
    for name, value in other.items():
        torch.testing.assert_close(getattr(state, name), value, atol=0, rtol=0)
    for factor in (before, after):
        evolved = net._tensor_transport(state, net._duration(.02, state.field), factor)
        torch.testing.assert_close(net.energy(evolved), energy, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(evolved.field.sum((1, 2, 3)),
                                   state.field.sum((1, 2, 3)), atol=1e-12, rtol=1e-12)

    # R_m maps the same full activity coordinates to one finite-time local
    # observation. Different kernels demonstrate different observable distinctions.
    width = state.field.numel()
    def observation(vector, factor):
        blocks = vector.split(width)
        sample = replace(state, field=blocks[0].reshape_as(state.field),
                         flux=tuple(x.reshape_as(state.field) for x in blocks[1:]))
        result = net._tensor_transport(sample, net._duration(.02, sample.field), factor)
        return result.field[0, 0, 0, 0, 0]
    coordinates = torch.zeros(4 * width, dtype=torch.float64, requires_grad=True)
    r0 = torch.autograd.functional.jacobian(lambda v: observation(v, before), coordinates)
    r1 = torch.autograd.functional.jacobian(lambda v: observation(v, after), coordinates)
    witness = r1 - (r1 @ r0) / (r0 @ r0) * r0
    assert abs((r0 @ witness).item()) < 1e-12
    assert (r1 @ witness).item() > 1e-6
    # New observations need not preserve old readout values, despite exact
    # continuity of stored state. This is the boundary of the memory claim.


def test_integrated_budget_matches_dynamic_read_and_serializes():
    from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
    from information_boltzmann.runtime.continuous import ContinuousStream
    net = PlasticMedium3D((2, 2, 2), channels=4, hidden=8,
                         anisotropic_transport=True, transport_capacity_budget=.3).double()
    state = net.initial_state()
    state = replace(state, field=torch.randn_like(state.field),
                    flux=tuple(torch.randn_like(x) for x in state.flux))
    assert transport_capacity(net.current_transport_factor(state)).max() <= .3 + 1e-12
    rhs = net.field_rhs(state)
    evolved, _ = net.advance(state, 1e-8)
    torch.testing.assert_close((evolved.field - state.field) / 1e-8, rhs,
                               atol=1e-5, rtol=1e-5)
    model = PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=8,
        hidden=8, anisotropic_transport=True, transport_capacity_budget=.3).double()
    payload = ContinuousStream(model, max_step=.005).state_dict()
    assert payload['transport_capacity_budget'] == .3
    ContinuousStream.from_state_dict(model, payload)
    payload['transport_capacity_budget'] = None
    with pytest.raises(ValueError, match='capacity budget'):
        ContinuousStream.from_state_dict(model, payload)


def test_energy_current_closes_local_transport_energy_derivative():
    net = PlasticMedium3D((2, 2, 2), channels=4, hidden=8,
                         anisotropic_transport=True, transport_capacity_budget=1.).double()
    with torch.no_grad():
        net.transport_shear.bias.copy_(torch.tensor([.4, -.3, .5]))
    state = net.initial_state()
    state = replace(state, field=torch.randn_like(state.field),
                    flux=tuple(torch.randn_like(x) for x in state.flux))
    factor = net.current_transport_factor(state)
    flow = net.transport_energy_current(state, factor)
    df = torch.zeros_like(state.field)
    dq = [torch.zeros_like(x) for x in state.flux]
    divergence = torch.zeros_like(flow[..., 0])
    for axis in range(3):
        current = sum(factor[..., axis, k, None] * state.flux[k] for k in range(3))
        df = df + net.shape[axis] * (torch.roll(current, 1, axis + 1) - current)
        difference = state.field - torch.roll(state.field, -1, axis + 1)
        for k in range(3):
            dq[k] = dq[k] + net.shape[axis] * factor[..., axis, k, None] * difference
        divergence += net.shape[axis] * (torch.roll(flow[..., axis], 1, axis + 1) - flow[..., axis])
    derivative = (state.field * df).sum(-1) + sum((x * y).sum(-1) for x, y in zip(state.flux, dq))
    torch.testing.assert_close(derivative, divergence, atol=1e-12, rtol=1e-12)


def test_disabled_budget_exactly_preserves_raw_factor_and_state_dict():
    net = PlasticMedium3D((2, 2, 2), channels=4, hidden=8,
                         anisotropic_transport=True).double()
    material = net.material_field()
    speed = torch.ones(1, 2, 2, 2, 3, dtype=torch.float64)
    expected = torch.diag_embed(speed)
    torch.testing.assert_close(net.transport_factor(material, speed), expected, atol=0, rtol=0)
    assert all('transport_budget' not in key for key in net.state_dict())


def test_structural_budget_does_not_broadcast_local_activity_instantaneously():
    net = PlasticMedium3D((2, 2, 2), channels=4, hidden=8, adaptive_conduction=True,
                         anisotropic_transport=True, transport_capacity_budget=.3).double()
    state = net.initial_state()
    before = net.current_transport_factor(state)
    local = state.conduction.clone()
    local[0, 0, 0, 0, 0] += .5
    after = net.current_transport_factor(replace(state, conduction=local))
    torch.testing.assert_close(before[0, 1, 1, 1], after[0, 1, 1, 1], atol=0, rtol=0)
    assert not torch.equal(before[0, 0, 0, 0], after[0, 0, 0, 0])
