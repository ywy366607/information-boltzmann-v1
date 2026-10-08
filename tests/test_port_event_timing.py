"""Physical duration stays fixed when the integration grid is refined."""
import math

import pytest
import torch

from information_boltzmann.core.torus3d import CBIMTorus3D, VelocityCayleyTransport3D
from scripts.ib.train_q8_port_agents import resolve_event_timing


@pytest.mark.parametrize("resolution", [3, 16, 64])
def test_default_reference_duration_is_independent_of_resolution(resolution):
    duration, dt = resolve_event_timing(resolution, None, None)
    assert duration == 3.0
    assert resolution * dt == duration


def test_legacy_timing_is_explicit_and_reproducible():
    assert resolve_event_timing(16, None, 4.0) == (64.0, 4.0)
    assert resolve_event_timing(64, 3.0, None) == (3.0, 3.0 / 64)
    with pytest.raises(ValueError):
        resolve_event_timing(16, 3.0, 4.0)
    for duration in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            resolve_event_timing(16, duration, None)


def test_belief_step_resolution_override_preserves_duration_and_gradients():
    torch.manual_seed(112)
    model = CBIMTorus3D(
        vocab_size=19, shape=(4, 4, 4), velocities=8, content_dim=2,
        write_type="w4_predictive_agent", readout_type="belief_agent",
        micro_steps=2, event_duration=3.0).double()
    original = model.state_agent.evolve
    observed = []

    def record(field, precision, **options):
        observed.append(options["micro_steps"] * float(options["tau_0"]))
        return original(field, precision, **options)

    model.state_agent.evolve = record
    ids = torch.tensor([[2, 3]])
    belief = model.initial_belief(1, warm_start=False)
    loss, _, _ = model.forward_belief(ids, torch.tensor([[3, 4]]), belief, micro_steps=4)
    loss.backward()
    assert observed == [3.0, 3.0]
    for module in (model.collision.angle, model.write_agent.chart_gate):
        assert module[-1].weight.grad is not None
        assert torch.isfinite(module[-1].weight.grad).all()
        assert module[-1].weight.grad.norm() > 0


def test_cayley_refinement_converges_at_fixed_physical_duration():
    torch.manual_seed(113)
    transport = VelocityCayleyTransport3D((4, 4, 4), 8, 2).double()
    field = torch.randn(1, 4, 4, 4, 16, dtype=torch.float64)
    duration = 0.3
    _, omega = transport.multiplier(duration)
    exact = transport.apply_multiplier(field, torch.exp(-1j * duration * omega))
    errors = []
    for resolution in (8, 32):
        multiplier, _ = transport.multiplier(duration / resolution)
        output = field
        for _ in range(resolution):
            output = transport.apply_multiplier(output, multiplier)
        errors.append((output - exact).norm())
    assert errors[1] < errors[0] / 4


def test_field_only_forward_uses_same_fixed_duration_policy():
    model = CBIMTorus3D(vocab_size=19, shape=(4, 4, 4), velocities=8,
                        content_dim=2, micro_steps=2, event_duration=3.0)
    original = model.collision.forward
    observed = []

    def record(field, dt):
        observed.append(float(dt))
        return original(field, dt)

    model.collision.forward = record
    model(torch.tensor([[2]]), torch.tensor([[3]]), micro_steps=4)
    assert observed == [0.75] * 4
