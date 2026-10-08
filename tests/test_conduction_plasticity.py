"""Train-free structural-law invariants, integration, and state persistence."""
import copy
import math

import pytest
import torch

from information_boltzmann.core.plastic_medium import MediumState, PlasticMedium3D
from information_boltzmann.core.plastic_feasibility import continuous_rhs


def make_model(shape=(2, 2, 2), channels=3):
    torch.manual_seed(431)
    return PlasticMedium3D(shape, channels, hidden=8, adaptive_conduction=True).double()


def random_state(model, batch=2):
    empty = model.initial_state(batch)
    return MediumState(torch.randn_like(empty.field),
                       tuple(torch.randn_like(x) for x in empty.flux), empty.elapsed,
                       torch.randn_like(empty.conduction) * 0.2)


def test_evidence_is_bounded_local_and_content_sensitive():
    model = make_model((4, 4, 4))
    state = random_state(model)
    rule = model.conduction_plasticity
    metric = rule.coefficients(model.material_field())[-1]
    evidence = rule.evidence(state.field, state.flux, metric)
    assert evidence.abs().max() <= 1
    empty = model.initial_state()
    assert torch.count_nonzero(rule.evidence(empty.field, empty.flux, metric)) == 0
    # A changed site affects only its incident edges, not the entire medium.
    field = state.field.clone()
    field[:, 1, 1, 1] *= -2
    changed = rule.evidence(field, state.flux, metric) - evidence
    support = torch.zeros_like(changed, dtype=torch.bool)
    support[:, 1, 1, 1, :] = True
    support[:, 0, 1, 1, 0] = True
    support[:, 1, 0, 1, 1] = True
    support[:, 1, 1, 0, 2] = True
    assert changed[support].abs().sum() > 0
    assert torch.count_nonzero(changed[~support]) == 0
    with torch.no_grad():
        rule.metric_logits[0, 0] = 5
    assert (rule.evidence(state.field, state.flux, rule.coefficients(model.material_field())[-1])
            - evidence).abs().sum() > 0


def test_local_material_can_choose_different_content_on_each_edge():
    from information_boltzmann.core.conduction_plasticity import LocalConductionPlasticity
    rule = LocalConductionPlasticity(2, 1, spatial_metric=True).double()
    with torch.no_grad():
        rule.metric_material.weight.copy_(torch.tensor([[2.], [-2.]] * 3))
    material = torch.zeros(2, 2, 2, 1, dtype=torch.float64)
    material[0] = 1
    material[1] = -1
    _, _, metric = rule.coefficients(material)
    torch.testing.assert_close(metric.sum(-1), torch.ones_like(metric[..., 0]))
    assert metric.min() > 0
    assert torch.all(metric[0, ..., 0] > metric[0, ..., 1])
    assert torch.all(metric[1, ..., 1] > metric[1, ..., 0])
    # Same endpoint content has different edge evidence under local preferences.
    field = torch.ones(1, 2, 2, 2, 2, dtype=torch.float64)
    field[:, :, 1, :, 1] = -1
    flux = tuple(torch.zeros_like(field) for _ in range(3))
    evidence = rule.evidence(field, flux, metric)
    assert evidence.abs().max() <= 1
    assert torch.all(evidence[:, 0, ..., 1] > 0)
    assert torch.all(evidence[:, 1, ..., 1] < 0)


def test_local_metric_receives_likelihood_gradients_and_shared_control_stays_available():
    from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
    torch.manual_seed(533)
    model = PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=8,
                                 material_width=3, hidden=8, heads=2, queries=2,
                                 bath_type='conductance').double()
    rule = model.medium.conduction_plasticity
    assert rule.metric_material is not None
    with torch.no_grad():
        model.medium.material.coefficients.normal_()
    loss, _, _ = model(torch.tensor([[2, 3]]), torch.tensor([[3, 4]]), event_duration=0.02)
    loss.backward()
    assert torch.isfinite(rule.metric_material.weight.grad).all()
    assert rule.metric_material.weight.grad.abs().sum() > 0
    historical = make_model()
    assert historical.conduction_plasticity.metric_material is None


def test_frozen_activity_exact_semigroup_bound_and_quiet_recovery():
    model = make_model()
    state = random_state(model)
    gain, rate, metric = model.conduction_plasticity.coefficients(model.material_field())
    target = gain[None] * model.conduction_plasticity.evidence(state.field, state.flux, metric)
    expected = target + (state.conduction - target) * (-0.7 * rate[None]).exp()
    actual = model.adapt_conduction(state, 0.7)
    torch.testing.assert_close(actual.conduction, expected)
    half = model.adapt_conduction(state, 0.35)
    halves = model.adapt_conduction(half, 0.35)
    torch.testing.assert_close(halves.conduction, actual.conduction)
    huge = model.adapt_conduction(state, 1e6)
    bound = torch.maximum(state.conduction.abs(), gain[None])
    assert torch.all(huge.conduction.abs() <= bound + 1e-14)
    torch.testing.assert_close(model.energy(actual), model.energy(state))
    quiet = model.initial_state(state.field.shape[0]).with_conduction(actual.conduction)
    restored = model.adapt_conduction(quiet, 0.7)
    torch.testing.assert_close(restored.conduction, actual.conduction * (-0.7 * rate[None]).exp())


def test_content_can_form_and_relocate_a_fast_path_without_optimizer():
    model = make_model((4, 4, 4))
    with torch.no_grad():
        model.conduction_plasticity.gain.bias.fill_(math.log(math.expm1(4)))
    state = model.initial_state()
    field = state.field.clone()
    field[:, :, 0, 0, 0] = 1
    first = model.adapt_conduction(state.with_field(field), 10.0)
    speeds = model.edge_log_speeds(first).exp()
    assert speeds[0, :, 0, 0, 0].min() > 54
    torch.testing.assert_close(speeds[0, :, 0, 0, 1:], torch.ones_like(speeds[0, :, 0, 0, 1:]))
    new_field = torch.zeros_like(field)
    new_field[:, :, 2, 0, 0] = 1
    moved = model.adapt_conduction(first.with_field(new_field), 10.0)
    speeds = model.edge_log_speeds(moved).exp()
    assert speeds[0, :, 2, 0, 0].min() > 54
    assert speeds[0, :, 0, 0, 0].max() < 1.001


def test_adaptive_conduction_preserves_passivity_and_is_batch_specific():
    model = make_model()
    state = random_state(model)
    output, info = model.advance(state, torch.tensor([0.02, 0.03]), substeps=3)
    torch.testing.assert_close(model.energy(output) + info['bath_out_energy'],
                               model.energy(state), atol=3e-13, rtol=3e-13)
    assert (output.conduction[0] - output.conduction[1]).norm() > 0
    for index, duration in enumerate((0.02, 0.03)):
        one = MediumState(state.field[index:index+1],
                          tuple(x[index:index+1] for x in state.flux),
                          state.elapsed[index:index+1], state.conduction[index:index+1])
        separate, _ = model(one, duration, substeps=3)
        torch.testing.assert_close(separate.conduction, output.conduction[index:index+1])
        torch.testing.assert_close(separate.field, output.field[index:index+1])


def test_extended_infinitesimal_generator_and_solver_refinement():
    model = make_model()
    state = random_state(model, batch=1)
    def flatten(s):
        return torch.cat((model._pack(s).flatten(), s.conduction.flatten()))
    def at_duration(time):
        return flatten(model(state, time)[0])
    _, derivative = torch.autograd.functional.jvp(
        at_duration, torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(derivative, flatten(continuous_rhs(model, state)),
                               atol=5e-13, rtol=5e-13)
    reference = model(state, 0.06, substeps=128)[0]
    errors = [(flatten(model(state, 0.06, substeps=k)[0]) - flatten(reference)).norm()
              for k in (4, 16)]
    assert errors[1] < errors[0] / 2


def test_conduction_persists_through_ports_detach_and_checkpoint(tmp_path):
    from information_boltzmann.core.plastic_medium import boundary_scatter
    model = make_model()
    state = random_state(model, batch=1)
    first, _ = model(state, 0.01)
    written, _ = boundary_scatter(first, torch.randn_like(first.field), first.field.new_tensor(0.2))
    assert written.conduction is first.conduction
    torch.testing.assert_close(first.detach().conduction, first.conduction)
    assert not first.detach().conduction.requires_grad
    path = tmp_path / 'continuation.pt'
    torch.save({'weights': model.state_dict(), 'field': first.field, 'flux': first.flux,
                'elapsed': first.elapsed, 'conduction': first.conduction}, path)
    saved = torch.load(path, weights_only=True)
    other = make_model()
    other.load_state_dict(saved['weights'], strict=True)
    recovered = MediumState(saved['field'], saved['flux'], saved['elapsed'], saved['conduction'])
    expected, _ = model(first, 0.02)
    actual, _ = other(recovered, 0.02)
    torch.testing.assert_close(actual.conduction, expected.conduction)
    torch.testing.assert_close(actual.field, expected.field)
    with pytest.raises(ValueError, match='conduction'):
        model(MediumState(first.field, first.flux, first.elapsed), 0.01)


def test_compile_and_gradients_for_coupled_runtime_plasticity():
    model = make_model()
    state = random_state(model, batch=1)
    state = state.with_conduction(state.conduction.detach().requires_grad_())
    compiled = torch.compile(copy.deepcopy(model), backend='eager', fullgraph=True)
    expected = model(state, 0.03, substeps=2)[0]
    actual = compiled(state, 0.03, substeps=2)[0]
    torch.testing.assert_close(actual.conduction, expected.conduction)
    torch.testing.assert_close(actual.field, expected.field)
    loss = (actual.field * torch.randn_like(actual.field)).sum() + actual.conduction.square().sum()
    loss.backward()
    for name, parameter in compiled.conduction_plasticity.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        # Uniform initial material makes weight gradients zero; biases/metric are active.
        if not name.endswith('weight'):
            assert parameter.grad.norm() > 0, name
    assert state.conduction.grad is not None and state.conduction.grad.norm() > 0


def test_actual_w4_readout_loss_trains_plasticity_and_preserves_it_across_observations():
    from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
    model = PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=8,
                                 material_width=3, hidden=8, heads=2, queries=2).double()
    ids, targets = torch.tensor([[2, 3, 4]]), torch.tensor([[3, 4, 5]])
    loss, whole, _ = model(ids, targets, event_duration=0.02)
    loss.backward()
    for parameter in (model.medium.conduction_plasticity.metric_logits,
                      model.medium.conduction_plasticity.gain.bias,
                      model.medium.conduction_plasticity.log_rate.bias):
        assert torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0
    _, first, _ = model(ids[:, :1], targets[:, :1], event_duration=0.02)
    _, continued, _ = model(ids[:, 1:], targets[:, 1:], first.detach(), event_duration=0.02)
    torch.testing.assert_close(whole.medium.conduction, continued.medium.conduction)
    assert whole.medium.conduction.abs().sum() > 0


def test_adaptive_material_preserves_the_certified_stable_locking_region():
    from scripts.ib.audit_conduction_plasticity import phase_locking_certificate
    certificate = phase_locking_certificate()
    assert certificate['passed']
    assert certificate['total_state_dimension'] == 120
    assert certificate['neutral_phase_directions'] == 1
    assert certificate['continuous_transverse_gap'] > 0.12
