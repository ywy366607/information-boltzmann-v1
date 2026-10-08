"""Energy, untouched modes and differentiability of the selected boundary."""
import math

import torch
import pytest

from information_boltzmann.core.mode_port import scatter_contact_mode
from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.core.torus3d import FullRankTorusWrite, PredictiveImpedanceWriteAgent


def inner(left, right):
    return (left * right).sum((1, 2, 3, 4)) / math.prod(left.shape[1:4])


def test_energy_and_orthogonal_old_modes_are_preserved():
    torch.manual_seed(419)
    field = torch.randn(2, 4, 4, 4, 8, dtype=torch.float64)
    coupling = torch.randn_like(field)
    incident = torch.randn(2, dtype=torch.float64)
    output, outgoing, r2 = scatter_contact_mode(field, coupling, incident)
    torch.testing.assert_close(inner(output, output) + outgoing.square(),
                               inner(field, field) + incident.square(), atol=2e-12, rtol=2e-12)
    projection = lambda value: coupling * (inner(coupling, value) / r2)[:, None, None, None, None]
    torch.testing.assert_close(output - projection(output), field - projection(field),
                               atol=2e-12, rtol=2e-12)


def test_zero_coupling_is_identity_with_finite_gradients():
    field = torch.randn(1, 2, 2, 2, 4, dtype=torch.float64, requires_grad=True)
    coupling = torch.zeros_like(field, requires_grad=True)
    incident = torch.randn(1, dtype=torch.float64, requires_grad=True)
    output, outgoing, _ = scatter_contact_mode(field, coupling, incident)
    torch.testing.assert_close(output, field, atol=0, rtol=0)
    torch.testing.assert_close(outgoing, incident, atol=0, rtol=0)
    (output.square().sum() + outgoing.square().sum()).backward()
    assert all(torch.isfinite(x.grad).all() for x in (field, coupling, incident))


def test_full_boundary_derivatives_match_finite_differences():
    torch.manual_seed(421)
    args = (torch.randn(1, 2, 2, 2, 3, dtype=torch.float64, requires_grad=True),
            torch.randn(1, 2, 2, 2, 3, dtype=torch.float64, requires_grad=True),
            torch.randn(1, dtype=torch.float64, requires_grad=True))
    assert torch.autograd.gradcheck(lambda *values: scatter_contact_mode(*values)[:2], args)


def test_frozen_port_field_jacobian_keeps_complement_gradient():
    torch.manual_seed(422)
    field = torch.randn(1, 2, 2, 2, 3, dtype=torch.float64, requires_grad=True)
    coupling = torch.randn_like(field)
    tangent = torch.randn_like(field)
    tangent -= coupling * (inner(coupling, tangent) / inner(coupling, coupling))[:, None, None, None, None]
    output, _, _ = scatter_contact_mode(field, coupling, torch.ones(1, dtype=torch.float64))
    gradient, = torch.autograd.grad((output * tangent).sum(), field)
    torch.testing.assert_close(gradient, tangent, atol=2e-12, rtol=2e-12)


def test_real_writer_preserves_energy_and_null_innovation():
    torch.manual_seed(423)
    writer = FullRankTorusWrite(19, (4, 4, 4), 16, write_type='w2_impedance').double()
    agent = PredictiveImpedanceWriteAgent(16, 19, exchange='contact_mode').double()
    field = torch.randn(2, 4, 4, 4, 16, dtype=torch.float64)
    precision = agent.initial_precision(2, device=field.device, dtype=field.dtype)
    tokens = torch.tensor([3, 11])
    output, _, reflected, diag = agent(writer, field, tokens, precision)
    assert diag['write_balance_residual'] < 2e-12
    assert reflected.shape == (2, 1, 1, 1, 1)
    assert 0 < diag['accepted_fraction'] < 1
    observed = torch.nn.functional.normalize(writer.embedding(tokens), dim=-1)
    unchanged, _, outgoing, zero = agent(writer, field, tokens, precision,
                                         predicted_feature=observed)
    torch.testing.assert_close(unchanged, field, atol=0, rtol=0)
    torch.testing.assert_close(outgoing, torch.zeros_like(outgoing), atol=0, rtol=0)
    (output.square().mean() + unchanged.square().mean() + diag['_write_free_energy']).backward()
    assert agent.action_posterior[-1].weight.grad.norm() > 0
    assert agent.chart_gate[-1].weight.grad.norm() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in list(writer.parameters()) + list(agent.parameters()))


def test_new_port_loads_identical_weights_and_preserves_incident_budget():
    torch.manual_seed(424)
    old = PlasticMediumPorts3D(vocab_size=19, shape=(4, 4, 4), channels=16,
                                hidden=8, bath_type='conductance', write_exchange='global', port_scope='global').double()
    new = PlasticMediumPorts3D(vocab_size=19, shape=(4, 4, 4), channels=16,
                                hidden=8, bath_type='conductance', port_scope='global').double()
    new.load_state_dict(old.state_dict(), strict=True)
    belief = old.initial_belief()
    _, old_diag = old.assimilate(belief, torch.tensor([3]))
    _, new_diag = new.assimilate(belief, torch.tensor([3]))
    torch.testing.assert_close(old_diag['incident_energy'], new_diag['incident_energy'],
                               atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(old_diag['innovation_norm'], new_diag['innovation_norm'],
                               atol=0, rtol=0)
    assert new.write_agent.exchange == 'contact_mode'


def test_continuation_preserves_port_law_and_rejects_silent_switch():
    from information_boltzmann.runtime.continuous import ContinuousStream

    model = PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=8, hidden=8)
    stream = ContinuousStream(model, max_step=0.005)
    saved = stream.state_dict()
    continued = ContinuousStream.from_state_dict(model, saved)
    torch.testing.assert_close(continued.belief.medium.field, stream.belief.medium.field)
    saved.pop('write_exchange')  # Historical payloads used global exchange.
    with pytest.raises(ValueError, match='write exchange'):
        ContinuousStream.from_state_dict(model, saved)
    model.write_agent.exchange = 'global'
    ContinuousStream.from_state_dict(model, saved)
