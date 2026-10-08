"""Geometric/gradient contracts for opt-in finite-interface budgets."""
import math

import pytest
import torch

from information_boltzmann.core.local_ports import CompactTorusPorts, port_overlap_diagnostics
from information_boltzmann.core.readout_probes import PredictivePhysicalReadAgent
from information_boltzmann.core.temporal_probes import sample_compact_probes
from information_boltzmann.core.torus3d import PredictiveImpedanceWriteAgent


SHAPE = (8, 8, 8)
WRITE_RADIUS = (.1875, .1875, .25)
READ_RADIUS = (.125, .1875, .25)


def test_budget_counts_all_support_boxes_and_none_preserves_legacy_exactly():
    legacy = CompactTorusPorts(SHAPE, 8, WRITE_RADIUS).double()
    default = CompactTorusPorts(SHAPE, 8, WRITE_RADIUS, aperture_budget=None).double()
    assert legacy.aperture_volume == .5625
    assert default.aperture_budget is None
    assert set(default.state_dict()) == {'centers'}
    torch.testing.assert_close(default.footprint(), legacy.footprint(), atol=0, rtol=0)
    budget = legacy.aperture_volume / 2
    ports = CompactTorusPorts(SHAPE, 8, WRITE_RADIUS, aperture_budget=budget).double()
    actual_volume = ports.count * float((2 * ports.radius).prod())
    assert 0 < actual_volume <= budget
    assert ports.aperture_volume == pytest.approx(actual_volume)
    ratios = [r / before for r, before in zip(ports.physical_radius, WRITE_RADIUS)]
    assert max(ratios) - min(ratios) < 1e-7
    assert ratios[0] == pytest.approx(2 ** (-1 / 3), rel=2e-7)


def test_overlap_is_allowed_and_repeatedly_charged_positions_remain_learnable():
    ports = CompactTorusPorts(SHAPE, 8, WRITE_RADIUS, aperture_budget=.3).double()
    before = ports.aperture_volume
    with torch.no_grad():
        ports.centers.fill_(.123)
    footprint = ports.footprint()
    torch.testing.assert_close(footprint, footprint[:1].expand_as(footprint), atol=0, rtol=0)
    assert ports.aperture_volume == before
    assert ports.aperture_volume == 8 * math.prod(2 * r for r in ports.physical_radius)
    signal = torch.randn(1, math.prod(SHAPE), 3, dtype=torch.float64)
    ports.observe(signal).square().sum().backward()
    assert torch.isfinite(ports.centers.grad).all()
    assert (ports.centers.grad.abs().sum(0) > 0).all()
    with torch.no_grad():
        ports.centers.copy_(torch.rand_like(ports.centers))
    assert (ports.footprint().sum(-1) > 0).all()
    assert ports.aperture_volume == before


def test_budget_survives_refinement_with_saved_physical_radius_and_positions():
    coarse = CompactTorusPorts(SHAPE, 8, WRITE_RADIUS, aperture_budget=.3).double()
    fine = CompactTorusPorts((16, 16, 16), 8, coarse.physical_radius,
                             aperture_budget=.3).double()
    fine.load_state_dict(coarse.state_dict(), strict=True)
    assert fine.physical_radius == coarse.physical_radius
    assert fine.aperture_volume == coarse.aperture_volume
    torch.testing.assert_close(fine.footprint().reshape(8, 16, 16, 16)[:, ::2, ::2, ::2],
                               coarse.footprint().reshape(8, *SHAPE), atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize('budget', [0., -1., float('nan'), float('inf'), .001])
def test_invalid_or_geometrically_infeasible_budget_is_rejected(budget):
    with pytest.raises(ValueError, match='budget'):
        CompactTorusPorts(SHAPE, 8, WRITE_RADIUS, aperture_budget=budget)


def test_write_and_read_constructors_share_budget_geometry_and_local_measurement():
    write = PredictiveImpedanceWriteAgent(8, 17, exchange='contact_mode',
        local_shape=SHAPE, port_radius=WRITE_RADIUS, aperture_budget=.3).double()
    read = PredictivePhysicalReadAgent(SHAPE, 8, torch.eye(8)[:, 1:], heads=2, queries=2,
        aperture_type='compact_probes', port_radius=READ_RADIUS, aperture_budget=.1).double()
    assert write.local_ports.aperture_volume <= .3
    assert read.aperture_volume <= .1
    field = torch.randn(1, *SHAPE, 8, dtype=torch.float64, requires_grad=True)
    samples = sample_compact_probes(read, field)
    gradient, coordinates = torch.autograd.grad(samples.square().sum(), (field, read.probe_coords))
    support = (read.footprint() > 0).any(0)
    assert gradient.flatten(1, 3)[:, ~support].abs().max() == 0
    assert gradient.flatten(1, 3)[:, support].abs().sum() > 0
    assert torch.isfinite(coordinates).all() and coordinates.norm() > 0
    diagnostics = port_overlap_diagnostics(write.local_ports, read, field)
    assert float(diagnostics['port_total_aperture_volume']) <= .4
    assert float(diagnostics['port_write_aperture_volume']) == pytest.approx(write.local_ports.aperture_volume)
    assert float(diagnostics['port_read_aperture_volume']) == pytest.approx(read.aperture_volume)


def test_global_interfaces_reject_a_compact_support_budget():
    with pytest.raises(ValueError, match='compact write'):
        PredictiveImpedanceWriteAgent(8, 17, aperture_budget=.3)
    with pytest.raises(ValueError, match='compact read'):
        PredictivePhysicalReadAgent(SHAPE, 8, torch.eye(8)[:, 1:], aperture_budget=.3)
