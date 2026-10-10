"""Physical identities and integration contracts, not capability training."""
import copy
import io
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from information_boltzmann.core.hopf_recomposition import RootedTree, coproduct, graft
from information_boltzmann.core.medium_junction import LocalFluxJunction, compile_junction_forest
from information_boltzmann.core.plastic_medium import MediumState, PlasticMedium3D
from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.optimization import initialize_hopf_branch, make_medium_optimizer

OPTIONS = dict(resource_density=2., speed_reference=1., structure_time=3.,
               prior_std=.4, initial_std=.4, maintenance_supply=1.4, initial_dual=.1)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def state(n=4, channels=4, *, constant=False):
    torch.manual_seed(127)
    shape = (1, n, n, n, channels)
    if constant:
        q = [torch.randn(1, 1, 1, 1, channels, dtype=torch.float64).expand(shape).clone() for _ in range(3)]
    else:
        q = [torch.randn(shape, dtype=torch.float64) for _ in range(3)]
    return MediumState(torch.randn(shape, dtype=torch.float64), tuple(q), torch.tensor([2.], dtype=torch.float64),
                       torch.zeros(1, n, n, n, 3, dtype=torch.float64),
                       torch.rand(1, n, n, n, 2, channels, dtype=torch.float64),
                       torch.rand(1, n, n, n, 3, 2, dtype=torch.float64))


def test_rotation_retains_all_modes_aux_energy_and_same_tape_inverse():
    incoming = state()
    junction = LocalFluxJunction(3, (4, 4, 4)).double()
    with torch.no_grad(): junction.gate.bias.copy_(torch.tensor([.3, -.4, .2]))
    allocation = junction.allocate(incoming, torch.eye(3, dtype=torch.float64).expand(4, 4, 4, 3, 3),
                                   material=torch.ones(4, 4, 4, 3, dtype=torch.float64))
    output = junction.rotate(incoming, allocation.rates, .13)
    before = sum(q.square().sum() for q in incoming.flux)
    after = sum(q.square().sum() for q in output.flux)
    torch.testing.assert_close(before, after, atol=1e-12, rtol=1e-13)
    for name in ('field', 'elapsed', 'conduction', 'receptors', 'transmission'):
        assert getattr(output, name) is getattr(incoming, name)
    recovered = junction.rotate(output, allocation.rates, -.13)
    for a, b in zip(incoming.flux, recovered.flux):
        torch.testing.assert_close(a, b, atol=1e-13, rtol=1e-13)
    assert any(not torch.equal(a, b) for a, b in zip(incoming.flux, output.flux))


def test_paid_row_budget_and_fixed_physical_length_survive_refinement():
    junction = LocalFluxJunction(3, (4, 4, 4)).double()
    with torch.no_grad(): junction.gate.bias.copy_(torch.tensor([.3, -.4, .2]))
    outputs = []
    for n in (4, 8):
        incoming = state(n, constant=True)
        factor = torch.eye(3, dtype=torch.float64).expand(n, n, n, 3, 3) * .7
        allocation = junction.allocate(incoming, factor, material=torch.ones(n, n, n, 3, dtype=torch.float64))
        torch.testing.assert_close(allocation.transport_row_capacity.square() + allocation.junction_row_capacity.square(),
                                   allocation.full_row_capacity.square(), atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(allocation.rates.abs() / junction.inverse_length,
                                   allocation.junction_row_capacity, atol=1e-14, rtol=1e-14)
        assert allocation.rates.abs().max() <= .7 * 4
        outputs.append(junction.rotate(incoming, allocation.rates, .17))
    for a, b in zip(outputs[0].flux, outputs[1].flux):
        torch.testing.assert_close(a[0, 0, 0, 0], b[0, 0, 0, 0], atol=1e-13, rtol=1e-13)


def test_ck_cut_changes_executable_coupling_without_clearing_stored_modes():
    tree = graft((RootedTree('q1'), RootedTree('q2')), 'q0')
    junction = LocalFluxJunction(3, (4, 4, 4)).double()
    with torch.no_grad(): junction.gate.bias.fill_(.4)
    incoming = state()
    factor = torch.eye(3, dtype=torch.float64).expand(4, 4, 4, 3, 3)
    material = torch.ones(4, 4, 4, 3, dtype=torch.float64)
    junction.set_topology((tree,))
    full = junction.allocate(incoming, factor, material=material)
    cut = next(term for term in coproduct(tree) if term.pruned and term.remainder
               and len(term.remainder[0].children) == 1)
    junction.set_topology(cut.remainder)
    pruned = junction.allocate(incoming, factor, material=material)
    assert torch.count_nonzero(pruned.rates) < torch.count_nonzero(full.rates)
    assert not torch.equal(junction.rotate(incoming, full.rates, .2).flux[0],
                           junction.rotate(incoming, pruned.rates, .2).flux[0])
    assert incoming.flux[2].square().sum() > 0
    # Rejoining actual grafted children restores the operator, not old state.
    junction.set_topology((tree,))
    restored = junction.allocate(incoming, factor, material=material)
    torch.testing.assert_close(restored.rates, full.rates, atol=0, rtol=0)
    with pytest.raises(ValueError, match='Repeated'):
        compile_junction_forest((tree, tree))
    with pytest.raises(ValueError, match='distinct'):
        compile_junction_forest((graft((RootedTree('q0'),), 'q0'),))


def medium(hopf):
    return PlasticMedium3D((4, 4, 4), 4, material_width=3, hidden=4,
                           anisotropic_transport=True, structure_options=OPTIONS,
                           hopf_recomposition=hopf).double()


def test_zero_birth_is_exact_transport_identity_and_nonzero_routes_have_real_effect():
    torch.manual_seed(41)
    source = medium(False)
    destination = medium(True)
    destination.load_state_dict(source.state_dict(), strict=False)
    incoming = replace(state(), conduction=None, receptors=None, transmission=None)
    old, _ = source.advance(incoming, .03, substeps=3, collision=False, bath=False)
    new, _ = destination.advance(incoming, .03, substeps=3, collision=False, bath=False)
    for a, b in zip((old.field, *old.flux, old.elapsed), (new.field, *new.flux, new.elapsed)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    with torch.no_grad(): destination.hopf_pathway.gate.bias.fill_(.3)
    changed, _ = destination.advance(incoming, .03, substeps=3, collision=False, bath=False)
    torch.testing.assert_close(destination.energy(changed), destination.energy(incoming), atol=1e-12, rtol=1e-13)
    assert (changed.field - old.field).square().sum() > 1e-8
    assert changed.field.shape == incoming.field.shape


def ports(hopf):
    return PlasticMediumPorts3D(vocab_size=19, channels=4, shape=(4, 4, 4), hidden=4,
                                material_width=3, anisotropic_transport=True,
                                structure_options=OPTIONS, hopf_recomposition=hopf).double()


def test_complete_event_ce_can_change_zero_initialized_physical_routes():
    torch.manual_seed(42)
    source = ports(False)
    destination = ports(True)
    destination.load_state_dict(source.state_dict(), strict=False)
    belief = source.initial_belief()
    belief = replace(belief, medium=replace(belief.medium, flux=state().flux))
    logits = []
    for model in (source, destination):
        written, _ = model.assimilate(belief, torch.tensor([3]), diagnostics=False)
        advanced, _ = model.advance(written, .07, substeps=3, diagnostics=False)
        output, _ = model.read(advanced, diagnostics=False)
        logits.append(output)
    torch.testing.assert_close(logits[0], logits[1], atol=0, rtol=0)
    F.cross_entropy(logits[1], torch.tensor([7])).backward()
    grad = destination.hopf_pathway.gate.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 1e-10


def test_optimizer_migration_preserves_every_old_moment_and_zero_birth():
    torch.manual_seed(43)
    source = ports(False)
    opt = make_medium_optimizer(source, lr=.0002)
    for group in opt.param_groups:
        for p in group['params']: p.grad = torch.ones_like(p) * .01
    opt.step()
    saved = copy.deepcopy(opt.state_dict())
    destination = ports(True)
    adopted = initialize_hopf_branch(destination, source.state_dict(), saved, {'pending': 0})
    opt_new = make_medium_optimizer(destination, lr=.0002, saved_state=adopted)
    for old, new in zip(saved['param_groups'], opt_new.state_dict()['param_groups']):
        assert new['param_names'][:len(old['param_names'])] == old['param_names']
    for key, moments in saved['state'].items():
        for name, value in moments.items():
            torch.testing.assert_close(value, opt_new.state_dict()['state'][key][name], atol=0, rtol=0)
    assert torch.count_nonzero(destination.hopf_pathway.gate.weight) == 0
    assert torch.count_nonzero(destination.hopf_pathway.gate.bias) == 0
    with pytest.raises(ValueError, match='completed'):
        initialize_hopf_branch(destination, source.state_dict(), saved, {'pending': 1})


def test_tree_and_length_checkpoint_restore_and_compile_has_no_python_tensor_checks():
    junction = LocalFluxJunction(3, (4, 4, 4)).double()
    junction.set_topology((graft((RootedTree('q2'),), 'q0'),))
    with torch.no_grad(): junction.gate.bias.fill_(.3)
    buffer = io.BytesIO()
    torch.save(junction.state_dict(), buffer)
    buffer.seek(0)
    restored = LocalFluxJunction(3, (8, 8, 8)).double()
    restored.load_state_dict(torch.load(buffer, weights_only=False))
    assert restored.forest == junction.forest
    torch.testing.assert_close(restored.inverse_length, junction.inverse_length, atol=0, rtol=0)
    incoming = state()
    material = torch.ones(4, 4, 4, 3, dtype=torch.float64)
    factor = torch.eye(3, dtype=torch.float64).expand(4, 4, 4, 3, 3)
    def run(current, value):
        allocation = restored.allocate(current, factor, material=value)
        return restored.rotate(current, allocation.rates, .01)
    compiled = torch.compile(run, backend='eager', fullgraph=True)
    expected, result = run(incoming, material), compiled(incoming, material)
    for a, b in zip(expected.flux, result.flux):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_zero_gate_true_ce_derivative_matches_both_finite_directions():
    model = ports(True)
    incoming = model.initial_belief()
    incoming = replace(incoming, medium=replace(incoming.medium, flux=state().flux))
    def objective():
        written, _ = model.assimilate(incoming, torch.tensor([3]), diagnostics=False)
        advanced, _ = model.advance(written, .07, substeps=3, diagnostics=False)
        return F.cross_entropy(model.read(advanced, diagnostics=False)[0], torch.tensor([7]))
    base = objective()
    derivative = torch.autograd.grad(base, model.hopf_pathway.gate.bias)[0][0]
    epsilon = 1e-6
    with torch.no_grad():
        model.hopf_pathway.gate.bias[0] = epsilon
        plus = objective()
        model.hopf_pathway.gate.bias[0] = -epsilon
        minus = objective()
        model.hopf_pathway.gate.bias[0] = 0
    for numerical in ((plus - base.detach()) / epsilon, (base.detach() - minus) / epsilon):
        torch.testing.assert_close(numerical, derivative, atol=1e-7, rtol=1e-4)


@pytest.mark.parametrize('amplitude', [0., 1e20])
def test_fp32_large_finite_state_and_saturated_gate_have_finite_forward_backward(amplitude):
    incoming = state()
    incoming = replace(incoming, field=incoming.field.float(),
                       flux=tuple((q.float() * amplitude).requires_grad_() for q in incoming.flux))
    junction = LocalFluxJunction(3, (4, 4, 4)).float()
    with torch.no_grad(): junction.gate.bias.copy_(torch.tensor([1000., -1000., 0.]))
    allocation = junction.allocate(incoming, torch.eye(3).expand(4, 4, 4, 3, 3),
                                   material=torch.ones(4, 4, 4, 3))
    assert torch.isfinite(allocation.factor).all() and torch.isfinite(allocation.rates).all()
    (allocation.factor.sum() + allocation.rates.sum()).backward()
    assert torch.isfinite(junction.gate.weight.grad).all()
    assert all(torch.isfinite(q.grad).all() for q in incoming.flux)
    torch.testing.assert_close(allocation.transport_row_capacity.square() + allocation.junction_row_capacity.square(),
                               allocation.full_row_capacity.square(), atol=3e-7, rtol=3e-7)


def test_prepared_topology_is_a_snapshot_and_bad_load_changes_nothing():
    model = ports(True)
    with torch.no_grad(): model.hopf_pathway.gate.bias.fill_(.3)
    prepared = model.medium.prepare_evolution()
    saved = copy.deepcopy(model.state_dict())
    broken = copy.deepcopy(saved)
    broken['source.embedding.weight'].fill_(321.)
    broken['medium.hopf_pathway.route_mask'].zero_()
    with pytest.raises(ValueError, match='disagree'):
        model.load_state_dict(broken)
    for key, value in saved.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
    random = state()
    incoming = replace(model.initial_belief().medium, field=random.field, flux=random.flux)
    old = model.medium.native_advance(incoming, .03, prepared=prepared, diagnostics=False)[0]
    model.hopf_pathway.set_topology((RootedTree('q0'),))
    replay = model.medium.native_advance(incoming, .03, prepared=prepared, diagnostics=False)[0]
    for a, b in zip((old.field, *old.flux), (replay.field, *replay.flux)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    new = model.medium.native_advance(incoming, .03, diagnostics=False)[0]
    assert not torch.equal(new.field, old.field)


def test_rejected_migration_has_no_state_or_rng_mutation():
    source, destination = ports(False), ports(True)
    saved = make_medium_optimizer(source, lr=.0002).state_dict()
    saved['param_groups'][1]['weight_decay'] = .01
    before = copy.deepcopy(destination.state_dict())
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match='zero-decay'):
        initialize_hopf_branch(destination, source.state_dict(), saved, {'pending': 0})
    for key, value in before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(destination.state_dict()[key], value, atol=0, rtol=0)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_junction_stage_ledger_closes_and_actual_field_rhs_matches_tiny_step():
    model = medium(True)
    with torch.no_grad(): model.hopf_pathway.gate.bias.copy_(torch.tensor([.3, -.4, .2]))
    incoming = replace(state(), conduction=None, receptors=None, transmission=None)
    prepared = model.prepare_evolution()
    outgoing, ledger = model.native_advance(incoming, .03, substeps=3, prepared=prepared)
    stages = ('transport', 'junction', 'collision', 'bath')
    total = sum(ledger[f'{stage}_energy_change'] for stage in stages)
    torch.testing.assert_close(total, model.energy(outgoing) - model.energy(incoming), atol=1e-12, rtol=1e-12)
    assert ledger['junction_energy_change'].abs().max() < 1e-12
    dt = 1e-7
    tiny = model.native_advance(incoming, dt, prepared=prepared, diagnostics=False)[0]
    numerical = (tiny.field - incoming.field) / dt
    torch.testing.assert_close(numerical, model.field_rhs(incoming, prepared=prepared), atol=5e-5, rtol=5e-5)


def test_numerical_rate_bound_includes_junction_and_zero_birth_clock_is_exact():
    junction = LocalFluxJunction(3, (4, 4, 4)).double()
    material = torch.ones(4, 4, 4, 3, dtype=torch.float64)
    assert junction.solver_rate_multiplier((4, 4, 4), material) == 1.
    with torch.no_grad(): junction.gate.bias.fill_(1e-7)
    tiny = junction.solver_rate_multiplier((4, 4, 4), material)
    assert 1 < tiny < 1.000001
    with torch.no_grad(): junction.gate.bias.fill_(.3)
    factor = torch.eye(3, dtype=torch.float64).expand(4, 4, 4, 3, 3)
    allocation = junction.allocate(state(), factor, material=material)
    old_bound = 2 * 2 ** .5 * 4 * 3
    new_bound = junction.generator_rate_bound(allocation, (4, 4, 4))
    assert new_bound <= old_bound * junction.solver_rate_multiplier((4, 4, 4), material) + 1e-12
    # Coarsening respects the saved constitutive length rather than shrinking it.
    assert junction.solver_rate_multiplier((2, 2, 2), material) > junction.solver_rate_multiplier((4, 4, 4), material)


def test_split_rate_bound_handles_transport_and_junction_peaks_at_different_sites():
    junction = LocalFluxJunction(3, (4, 4, 4)).double()
    junction.set_topology((graft((RootedTree('q1'),), 'q0'),))
    with torch.no_grad(): junction.gate.weight[0, 0] = 1.
    material = torch.zeros(4, 4, 4, 3, dtype=torch.float64)
    material[2:, ..., 0] = 1000.
    factor = torch.eye(3, dtype=torch.float64).expand(4, 4, 4, 3, 3)
    allocation = junction.allocate(state(), factor, material=material)
    old = 2 * 2 ** .5 * 4 * 3
    measured = junction.generator_rate_bound(allocation, (4, 4, 4))
    torch.testing.assert_close(measured, torch.tensor([old + 4], dtype=torch.float64))
    assert measured <= old * junction.solver_rate_multiplier((4, 4, 4), material)
