"""Causal port integration and full fast-state persistence."""
import torch

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D


def make_model():
    torch.manual_seed(311)
    return PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=8,
                                 material_width=3, hidden=8, port_modes=8,
                                 heads=2, queries=2).double()


def test_actual_likelihood_reaches_write_read_and_all_medium_generators():
    model = make_model()
    with torch.no_grad():
        model.medium.material.coefficients.normal_(std=0.1)
    loss, belief, info = model(torch.tensor([[2, 3, 4]]), torch.tensor([[3, 4, 5]]),
                               event_duration=0.01)
    loss.backward()
    for name in ("material", "log_speed", "collision_rate", "bath_rate"):
        params = list(getattr(model.medium, name).parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert sum(p.grad.norm() for p in params) > 0
    assert model.source.embedding.weight.grad.norm() > 0
    assert model.write_agent.chart_gate[-1].weight.grad.norm() > 0
    assert model.readout.probe_coords.grad.norm() > 0
    assert all(x.norm() > 0 for x in belief.medium.flux)
    assert torch.isfinite(info['token_nll'])
    torch.testing.assert_close(belief.medium.elapsed, torch.tensor([0.03], dtype=torch.float64))


def test_assimilation_keeps_inflight_memory_and_read_has_no_side_effect():
    model = make_model()
    belief, _ = model.assimilate(model.initial_belief(), torch.tensor([2]))
    belief, _ = model.advance(belief, 0.02)
    old_flux, old_time = belief.medium.flux, belief.medium.elapsed.clone()
    updated, _ = model.assimilate(belief, torch.tensor([3]))
    assert updated.medium.flux is old_flux
    first, _ = model.read(updated)
    second, _ = model.read(updated)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(updated.medium.elapsed, old_time)
    later, _ = model.advance(updated, 0.03)
    torch.testing.assert_close(later.medium.elapsed, old_time + 0.03)


def test_target_is_loss_only_and_stream_continuation_matches_whole_segment():
    model = make_model()
    ids, targets = torch.tensor([[2, 3, 4]]), torch.tensor([[3, 4, 5]])
    _, whole, _ = model(ids, targets, event_duration=0.01)
    _, altered, _ = model(ids, torch.tensor([[7, 8, 9]]), event_duration=0.01)
    torch.testing.assert_close(whole.medium.field, altered.medium.field)
    torch.testing.assert_close(whole.precision, altered.precision)
    _, first, _ = model(ids[:, :1], targets[:, :1], event_duration=0.01)
    _, continued, _ = model(ids[:, 1:], targets[:, 1:], first.detach(), event_duration=0.01)
    torch.testing.assert_close(whole.medium.field, continued.medium.field)
    for left, right in zip(whole.medium.flux, continued.medium.flux):
        torch.testing.assert_close(left, right)
    torch.testing.assert_close(whole.medium.elapsed, continued.medium.elapsed)
