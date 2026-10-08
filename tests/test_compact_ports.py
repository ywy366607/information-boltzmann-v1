"""Spatial access, gradients and continuation; no capability claims."""
import pytest
import torch

from information_boltzmann.core.local_ports import CompactTorusPorts
from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D


def model():
    torch.manual_seed(127)
    return PlasticMediumPorts3D(vocab_size=19, shape=(8, 8, 4), channels=16,
                                hidden=8, heads=2, queries=2).double()


def test_compact_kernel_periodic_gradients_and_nonempty_moving_ports():
    ports = CompactTorusPorts((8, 8, 4), 8).double()
    with torch.no_grad():
        ports.centers.add_(0.013)
    weights = ports.weights()
    assert (weights == 0).any() and (weights.sum(-1) > 0).all()
    shifted = weights.detach().clone()
    with torch.no_grad():
        ports.centers.add_(1)
    torch.testing.assert_close(ports.weights(), shifted)
    (ports.weights() * torch.randn_like(weights)).sum().backward()
    assert torch.isfinite(ports.centers.grad).all() and ports.centers.grad.norm() > 0


def test_production_initial_banks_leave_remote_space_and_all_axes_learnable():
    write = CompactTorusPorts((8, 8, 4), 8).double()
    read = CompactTorusPorts((8, 8, 4), 16).double()
    for bank in (write, read):
        assert not (bank.footprint() > 0).any(0).all()
        (bank.weights() * torch.randn_like(bank.weights())).sum().backward()
        assert (bank.centers.grad.abs().sum(0) > 0).all()


def test_each_read_controller_and_attention_ignore_outside_its_port():
    net = model()
    reader = net.readout
    with torch.no_grad():
        reader.probe_coords.add_(0.013)
    field = torch.randn(1, 8, 8, 4, 16, dtype=torch.float64, requires_grad=True)
    precision = torch.ones(1, 16, dtype=torch.float64)
    _, first = reader(field, precision, return_diag=True)
    inside = reader.footprint()[0] > 0
    changed = field.detach().clone().reshape(1, -1, 16)
    changed[:, ~inside] += torch.randn_like(changed[:, ~inside]) * 7
    _, second = reader(changed.reshape_as(field), precision * 9, return_diag=True)
    torch.testing.assert_close(first['read_attention_weights'][:, 0, 0],
                               second['read_attention_weights'][:, 0, 0], atol=1e-12, rtol=1e-12)
    assert (first['read_attention_weights'][:, 0, 0, ~inside] == 0).all()
    # Isolate the first measurement at the decoder-facing merge. The policy
    # and full local value derivative must have exactly zero remote support.
    with torch.no_grad():
        reader.merge.weight.zero_()
        reader.merge.weight[0, 0] = 1
    feature, _ = reader(field, precision)
    gradient, = torch.autograd.grad(feature[0, 0], field)
    assert (gradient.reshape(1, -1, 16)[:, ~inside] == 0).all()
    assert gradient.norm() > 0


def test_write_preserves_uncontacted_cells_and_port_prediction_is_local():
    net = model()
    ports = net.write_agent.local_ports
    with torch.no_grad():
        ports.centers[:] = torch.tensor([0.13, 0.18, 0.17])
    belief = net.initial_belief()
    field = torch.randn_like(belief.medium.field)
    belief = type(belief)(belief.medium.with_field(field), belief.precision)
    inside = (ports.footprint() > 0).any(0)
    output, info = net.assimilate(belief, torch.tensor([3]))
    torch.testing.assert_close(output.medium.field.flatten(1, 3)[:, ~inside],
                               field.flatten(1, 3)[:, ~inside], atol=0, rtol=0)
    assert info['write_balance_residual'] < 2e-12
    changed = field.clone().flatten(1, 3)
    changed[:, ~inside] += 5
    alternate, other = net.assimilate(type(belief)(belief.medium.with_field(
        changed.reshape_as(field)), belief.precision), torch.tensor([3]))
    torch.testing.assert_close(output.precision, alternate.precision, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(info['port_nll'], other['port_nll'], atol=1e-12, rtol=1e-12)


def test_overlap_and_separation_are_allowed_and_collapse_is_observable():
    net = model()
    write, read = net.write_agent.local_ports, net.readout
    field = torch.randn_like(net.initial_belief().medium.field)
    with torch.no_grad():
        write.centers[:] = 0.1
        read.probe_coords[:] = 0.1
    overlap = net.port_diagnostics(field)
    assert overlap['port_write_pair_overlap'] == pytest.approx(1)
    assert overlap['port_nearest_read_distance'] == 0
    with torch.no_grad():
        read.probe_coords[:] = 0.6
    separate = net.port_diagnostics(field)
    assert separate['port_write_read_overlap_mean'] < overlap['port_write_read_overlap_mean']
    assert separate['port_nearest_read_distance'] > 0.8


def test_joint_likelihood_trains_centers_and_preserves_stream_state():
    net = model()
    initial = net.initial_belief()
    loss, belief, _ = net(torch.tensor([[2, 3]]), torch.tensor([[3, 4]]), initial,
                           event_duration=0.005)
    loss.backward()
    for parameter in (net.write_agent.local_ports.centers, net.readout.probe_coords,
                      net.write_agent.local_content.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    assert belief.medium.elapsed[0] == 0.01


def test_checkpoint_rejects_scope_and_aperture_changes():
    from information_boltzmann.runtime.continuous import ContinuousStream
    net = model()
    saved = ContinuousStream(net, max_step=0.005).state_dict()
    ContinuousStream.from_state_dict(net, saved)
    saved['read_port_radius'] = (0.49, 0.49, 0.49)
    with pytest.raises(ValueError, match='radius'):
        ContinuousStream.from_state_dict(net, saved)
    saved['port_scope'] = 'global'
    with pytest.raises(ValueError, match='scope'):
        ContinuousStream.from_state_dict(net, saved)


def test_monitor_reports_actual_policy_without_changing_state():
    net = model()
    belief, _ = net.assimilate(net.initial_belief(), torch.tensor([3]))
    old = belief.medium.field.clone()
    snapshot = net.port_snapshot(belief)
    assert snapshot['write_effective_port_count'] > 1
    assert snapshot['read_entropy_per_head'].shape == (2,)
    assert torch.isfinite(snapshot['active_write_read_overlap'])
    torch.testing.assert_close(belief.medium.field, old, atol=0, rtol=0)


def test_local_chart_is_linear_in_features_and_radius_survives_refinement():
    net = model()
    field = torch.randn_like(net.initial_belief().medium.field)
    prior = torch.randn(1, 32, dtype=field.dtype)
    first, second = torch.randn(2, 1, 16, dtype=field.dtype)
    chart = lambda feature: net.write_agent._packet_chart(net.source, field, prior, feature)[0]
    torch.testing.assert_close(chart(first + second), chart(first) + chart(second), atol=1e-12, rtol=1e-12)
    coarse = net.write_agent.local_ports
    fine = CompactTorusPorts((16, 16, 8), 8, coarse.physical_radius).double()
    fine.load_state_dict(coarse.state_dict(), strict=True)
    torch.testing.assert_close(fine.footprint().reshape(8, 16, 16, 8)[:, ::2, ::2, ::2],
                               coarse.footprint().reshape(8, 8, 8, 4), atol=1e-12, rtol=1e-12)
