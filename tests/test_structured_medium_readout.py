"""Fixed-basis execution equivalence; no training or capability claims."""
import copy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from information_boltzmann.core.plastic_ports import PlasticBelief, PlasticMediumPorts3D
from information_boltzmann.core.readout_probes import PredictivePhysicalReadAgent
from information_boltzmann.core.temporal_probes import sample_compact_probes


def mean_basis(d=8):
    w = -torch.ones(d, dtype=torch.float64) / d ** .5
    w[0] += 1
    w = w / w.norm()
    return w, (torch.eye(d, dtype=w.dtype) - 2 * w[:, None] * w[None, :])[:, 1:]


def reader(dtype, *, structured=True, nullspace=None):
    w, basis = mean_basis()
    return PredictivePhysicalReadAgent(
        (4, 4, 4), 8, basis if nullspace is None else nullspace,
        heads=2, queries=2, aperture_type='compact_probes',
        port_radius=(.34, .34, .34), dynamic=True,
        coordinate_reflector=w if structured else None).to(dtype=dtype)


def tolerance(dtype):
    return {'atol': 2e-5, 'rtol': 2e-5} if dtype == torch.float32 else {
        'atol': 3e-12, 'rtol': 3e-12}


def assert_gradients_close(actual, expected, dtype):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, **tolerance(dtype))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_householder_coordinates_inverse_and_upstream_vjp_match_dense(dtype):
    torch.manual_seed(1141)
    fast, dense = reader(dtype), reader(dtype, structured=False)
    assert fast.coordinate_reflector is not None
    # A caller's QR complement sign is part of the existing coordinate chart.
    fast.invariant_basis.mul_(-1)
    dense.invariant_basis.copy_(fast.invariant_basis)
    source = torch.randn(2, 11, 8, dtype=dtype, requires_grad=True)
    upstream = torch.randn(8, 8, dtype=dtype, requires_grad=True)
    field = source @ upstream
    actual, expected = fast.physical_coordinates(field), dense.physical_coordinates(field)
    torch.testing.assert_close(actual, expected, **tolerance(dtype))
    torch.testing.assert_close(actual[..., :1], expected[..., :1], atol=0, rtol=0)
    direction = torch.randn_like(actual)
    a = torch.autograd.grad((actual * direction).sum(), (field, source, upstream),
                            retain_graph=True)
    b = torch.autograd.grad((expected * direction).sum(), (field, source, upstream))
    assert_gradients_close(a, b, dtype)

    coordinates = torch.randn_like(actual, requires_grad=True)
    restored = fast.reconstruct_physical_coordinates(coordinates)
    reference = dense.reconstruct_physical_coordinates(coordinates)
    torch.testing.assert_close(restored, reference, **tolerance(dtype))
    direction = torch.randn_like(restored)
    a = torch.autograd.grad((restored * direction).sum(), coordinates)[0]
    b = torch.autograd.grad((reference * direction).sum(), coordinates)[0]
    torch.testing.assert_close(a, b, **tolerance(dtype))
    torch.testing.assert_close(fast.reconstruct_physical_coordinates(actual.detach()),
                               field.detach(), **tolerance(dtype))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('structured', [False, True])
def test_linear_probe_reordering_preserves_field_position_and_upstream_gradients(dtype, structured):
    torch.manual_seed(1142)
    agent = reader(dtype, structured=structured)
    with torch.no_grad():
        agent.probe_coords.add_(agent.probe_coords.new_tensor([.013, .021, .037]))
    source = torch.randn(2, 4, 4, 4, 8, dtype=dtype, requires_grad=True)
    upstream = torch.randn(8, 8, dtype=dtype, requires_grad=True)
    field = source @ upstream
    # The previous production order is the numerical reference.
    physical = agent.physical_coordinates(field.flatten(1, 3))
    expected = torch.einsum('pn,bnd->bpd', agent.weights().to(physical), physical)
    with patch.object(agent, 'physical_coordinates', wraps=agent.physical_coordinates) as call:
        actual = sample_compact_probes(agent, field)
        assert call.call_count == 1
        assert call.call_args.args[0].shape[1] == agent.heads * agent.queries
    torch.testing.assert_close(actual, expected, **tolerance(dtype))
    direction = torch.randn_like(actual)
    leaves = (field, source, upstream, agent.probe_coords)
    a = torch.autograd.grad((actual * direction).sum(), leaves, retain_graph=True)
    b = torch.autograd.grad((expected * direction).sum(), leaves)
    assert a[-1].norm() > 0
    assert_gradients_close(a, b, dtype)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_compact_nonlinear_reader_keeps_pointwise_outputs_and_all_parameter_gradients(dtype):
    torch.manual_seed(1143)
    fast = reader(dtype)
    dense = copy.deepcopy(fast)
    dense.coordinate_reflector = None
    field = torch.randn(2, 4, 4, 4, 8, dtype=dtype, requires_grad=True)
    motion = torch.randn_like(field, requires_grad=True)
    precision = torch.ones(2, 8, dtype=dtype)
    actual, _ = fast(field, precision, motion=motion)
    expected, _ = dense(field, precision, motion=motion)
    torch.testing.assert_close(actual, expected, **tolerance(dtype))
    direction = torch.randn_like(actual)
    a = torch.autograd.grad((actual * direction).sum(),
                            (field, motion, *fast.parameters()), allow_unused=True)
    b = torch.autograd.grad((expected * direction).sum(),
                            (field, motion, *dense.parameters()), allow_unused=True)
    assert_gradients_close(a, b, dtype)


@pytest.mark.parametrize('mismatch', ['arbitrary_basis', 'different_nullity', 'nonunit', 'nonfinite'])
def test_incompatible_explicit_reflector_uses_generic_dense_fallback(mismatch):
    torch.manual_seed(1144)
    w, basis = mean_basis()
    if mismatch == 'arbitrary_basis':
        basis = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))[0][:, 1:]
    elif mismatch == 'different_nullity':
        basis = basis[:, 1:]
    elif mismatch == 'nonunit':
        w = w * 2
    else:
        w[0] = float('nan')
    agent = PredictivePhysicalReadAgent((2, 2, 2), 8, basis,
                                       heads=2, queries=2, coordinate_reflector=w).double()
    assert agent.coordinate_reflector is None
    field = torch.randn(1, 8, 8, dtype=torch.float64)
    expected = torch.cat((torch.einsum('dk,bnd->bnk', agent.invariant_basis, field),
                          torch.einsum('dk,bnd->bnk', agent.nullspace, field)), -1)
    torch.testing.assert_close(agent.physical_coordinates(field), expected, atol=0, rtol=0)


def test_fixed_reflector_rejects_learnable_or_wrong_shaped_arguments():
    w, basis = mean_basis()
    with pytest.raises(ValueError, match='fixed'):
        PredictivePhysicalReadAgent((2, 2, 2), 8, basis,
                                    coordinate_reflector=w.requires_grad_())
    with pytest.raises(ValueError, match='shape'):
        PredictivePhysicalReadAgent((2, 2, 2), 8, basis,
                                    coordinate_reflector=w.detach()[:4])


def test_nonpersistent_basis_keeps_old_state_dict_strictly_compatible():
    torch.manual_seed(1145)
    dense, fast = reader(torch.float64, structured=False), reader(torch.float64)
    saved = dense.state_dict()
    assert set(saved) == set(fast.state_dict())
    assert not any('reflector' in name or 'nullspace' in name or 'invariant_basis' in name
                   for name in saved)
    fast.load_state_dict(saved, strict=True)
    dense.load_state_dict(fast.state_dict(), strict=True)
    for key, value in saved.items():
        torch.testing.assert_close(fast.state_dict()[key], value, atol=0, rtol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_medium_wiring_preserves_material_writer_temporal_and_time_credit(dtype):
    torch.manual_seed(1146)
    fast = PlasticMediumPorts3D(
        vocab_size=17, shape=(4, 4, 4), channels=8, material_width=3,
        hidden=6, heads=2, queries=2, bath_type='conductance',
        activity_adaptation=True, short_term_plasticity=True,
        read_mode='temporal', temporal_rates=[.7, 1.1],
        temporal_frequencies=[0., .4]).to(dtype=dtype)
    assert fast.readout.coordinate_reflector is not None
    dense = copy.deepcopy(fast)
    dense.readout.coordinate_reflector = None
    initial = fast.initial_belief()
    field = torch.randn_like(initial.medium.field, requires_grad=True)
    medium = replace(initial.medium, field=field,
                     flux=tuple(torch.randn_like(x) for x in initial.medium.flux))
    belief = PlasticBelief(medium, initial.precision, initial.temporal)
    duration = torch.tensor(.007, dtype=dtype, requires_grad=True)

    def execute(model, old_sample_order):
        def sample(reader, value):
            physical = reader.physical_coordinates(value.flatten(1, 3))
            return torch.einsum('pn,bnd->bpd', reader.weights().to(physical), physical)
        if old_sample_order:
            with patch('information_boltzmann.core.plastic_ports.sample_compact_probes', sample):
                return execute(model, False)
        written, _ = model.assimilate(belief, torch.tensor([2]), diagnostics=False)
        outgoing, _ = model.advance(written, duration, diagnostics=False)
        result, _ = model.read(outgoing, decode=False)
        return result, outgoing.temporal.value

    actual, ahistory = execute(fast, False)
    expected, ehistory = execute(dense, True)
    torch.testing.assert_close(actual, expected, **tolerance(dtype))
    torch.testing.assert_close(ahistory, ehistory, **tolerance(dtype))
    direction = torch.randn_like(actual)
    direction_history = torch.randn_like(torch.view_as_real(ahistory))

    def loss(feature, history):
        return ((feature * direction).sum()
                + (torch.view_as_real(history) * direction_history).sum())

    a = torch.autograd.grad(loss(actual, ahistory),
                            (field, duration, *fast.parameters()), allow_unused=True)
    b = torch.autograd.grad(loss(expected, ehistory),
                            (field, duration, *dense.parameters()), allow_unused=True)
    assert_gradients_close(a, b, dtype)
    named_grads = dict(zip(dict(fast.named_parameters()), a[2:]))
    for name in ('medium.material.coefficients', 'write_agent.local_content.weight',
                 'readout.probe_coords', 'temporal_readout.bank.log_rate'):
        assert named_grads[name] is not None and named_grads[name].norm() > 0
    assert a[1].norm() > 0
