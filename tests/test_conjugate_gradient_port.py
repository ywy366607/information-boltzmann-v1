"""Exact event/interface checks; no learner or synthetic capability task."""
from fractions import Fraction as F
from itertools import product
from copy import deepcopy

import pytest
import torch

from information_boltzmann.conjugate_gradient_port import (
    MaterialOptimizer, FiberMaterialOptimizer, prepare_readout_fiber,
    encode_gradient_force, fractional_probability_bounds,
    parameter_value, prepare_gradient_event,
    prepare_source_gradient_event,
)
from information_boltzmann.core.cgb_local_medium import (
    LocalLaw, LocalMedium, OpenMedium, ReservoirLaw,
)
from information_boltzmann.core.cgb_low_resource_life import PrefixReplayWorld


def _pack(law, states):
    word = 0
    for edge, (p, q, w) in enumerate(states):
        code = (p * (law.charge_cap + 1) + q) * (law.slot_cap + 1) + w
        word |= code << (edge * law.bits)
    return word.to_bytes(law.state_bytes, "little")


def _successor(medium, edits, sign):
    body = LocalMedium(medium.body.law, bytes(medium.body.payload))
    for edge, state in edits:
        body._put(edge, state)
    return OpenMedium(body, ReservoirLaw(medium.stock - sign,
                      content_cost=medium.reservoir.content_cost,
                      chemical_base=medium.reservoir.chemical_base, stimulus=0))


def _energy(medium):
    body, law = medium.body, medium.body.law
    row_scale = F(law.pool_reference_size, law.labels) if law.pool_reference_size else 1
    col_scale = F(law.pool_reference_size, law.rows) if law.pool_reference_size else 1
    return (sum(body.unit_energy(body.get(e)) for e in range(law.edges))
            + row_scale * sum(law.potential[body.total(0, r)] for r in range(law.rows))
            + col_scale * sum(law.potential[body.total(1, c)] for c in range(law.labels))
            + (medium.reservoir.content_cost - medium.reservoir.chemical_base)
            * sum(sum(body.get(e)) for e in range(law.edges)))


def test_resource_feedback_matches_whole_energy_and_actual_pair_clock():
    """Lean Q-port margins at varying shared loads; no learning trajectory.

    Inhibition here comes from the existing quadratic pool, with no winner
    detector. The check includes the load where a damaged full code loses its
    downhill repair margin. It does not assert a temporal refractory period.
    """
    law = LocalLaw(10, 1, 2, 2, tuple(n*n for n in range(21)),
                   epsilon=0, coupling=12)
    forces = (F(-64), F(-1, 7), F(0), F(31), F(64))
    checked = 0
    for p, q, w in product(range(3), repeat=3):
        for load in range(19):
            others = [(0, min(2, max(0, load - 2*i)), 0) for i in range(9)]
            medium = OpenMedium(LocalMedium(law, _pack(law, [(p, q, w)] + others)),
                                ReservoirLaw(1, content_cost=16,
                                             chemical_base=2, stimulus=0))
            base = _energy(medium)
            for sign in (-1, 1):
                if not 0 <= q + sign <= 2:
                    continue
                index = law.events + int(sign == -1)
                delta = (14*sign - 12*p*w*sign
                         + (load+q+sign)**2 - (load+q)**2)
                for force in forces:
                    edits, exponent, bound, actual_sign = prepare_gradient_event(
                        medium, index, 0, lambda _: force)
                    assert actual_sign == sign and exponent is not None
                    after = _successor(medium, edits, actual_sign)
                    assert _energy(after) - base == delta
                    assert exponent - bound == min(0, -2*(delta + force*sign))
                    checked += 1
    assert checked == 3420


def test_rational_gradient_encoding_and_readout():
    for g in (F(n, 11) for n in range(-60, 61)):
        code = encode_gradient_force(g, F(3, 7), F(1, 13))
        assert abs(F(code) - g * F(3, 7) / F(1, 13)) <= F(1, 2)
    with pytest.raises(TypeError):
        encode_gradient_force(0.1, F(1))
    with pytest.raises(ValueError):
        encode_gradient_force(F(1), F(0))
    law = LocalLaw(1, 1, 1, 1, (0, 0))
    for state in product(range(2), repeat=3):
        medium = OpenMedium(LocalMedium(law, _pack(law, [state])), ReservoirLaw(4))
        assert parameter_value(medium, 0, F(2, 3), F(-1)) == F(-1) + F(2, 3) * sum(state)


@pytest.mark.parametrize("shape", [(1, 2), (2, 1)])
def test_paired_force_port_exhaustive_reverse_work_and_conservation(shape):
    law = LocalLaw(*shape, 1, 1, (0, 1, 4))
    seen, transfers, asymmetric, changed = 0, 0, 0, 0
    for states in product(list(product(range(2), repeat=3)), repeat=2):
        base = OpenMedium(LocalMedium(law, _pack(law, states)),
                          ReservoirLaw(3, content_cost=1, chemical_base=-1, stimulus=0))
        before = base.state_dict()
        base_energy = _energy(base)
        for forces in [(0, 0), (-3, 4), (2, 2), (F(-3, 7), F(4, 11)), (F(1, 7), F(1, 7))]:
            for index in range(base.events):
                accessed = []
                def force(edge):
                    accessed.append(edge)
                    return forces[edge]
                edits, e, bound, sign = prepare_source_gradient_event(base, index, 0, force)
                source = base.prepare(index, 0)
                assert {e for e in accessed} <= {e for e, _ in edits}
                assert len(accessed) <= 2
                if e is None:
                    assert (edits, e, bound, sign) == source
                    continue
                if all(sum(state) == sum(states[j]) for j, state in edits):
                    assert accessed == []
                after = _successor(base, edits, sign)
                back, er, br, sr = prepare_source_gradient_event(after, base.reverse(index), 0, forces.__getitem__)
                reverted = _successor(after, back, sr)
                assert reverted.body.payload == base.body.payload
                assert reverted.stock == base.stock
                work = sum(forces[j] * (sum(state) - sum(states[j])) for j, state in edits)
                expected_delta = _energy(after) - base_energy + work
                assert e - er == -2 * expected_delta
                assert bound == br and e <= bound and er <= br
                assert source[1] - e == work
                assert sum(map(sum, states)) + base.stock == sum(
                    sum(after.body.get(j)) for j in range(law.edges)) + after.stock
                if work == 0:
                    assert (edits, e, bound, sign) == source
                if len(edits) == 2:
                    transfers += 1
                    assert sign == 0
                reverse_source = after.prepare(base.reverse(index), 0)
                asymmetric += int(source[1] != -reverse_source[1])
                changed += int(work != 0)
                seen += 1
        assert base.state_dict() == before
    assert seen > 1000 and transfers > 0 and asymmetric > 0 and changed > 0


def test_force_port_does_not_read_remote_gradients_or_mutate_on_bad_input():
    law = LocalLaw(1, 1, 1, 1, (0, 0))
    medium = OpenMedium(LocalMedium(law, b"\x00"), ReservoirLaw(3))
    before = medium.state_dict()
    boundary = medium.body.law.events
    with pytest.raises(TypeError):
        prepare_gradient_event(medium, boundary, 0, lambda _: 1.5)
    assert medium.state_dict() == before
    with pytest.raises(ValueError):
        parameter_value(medium, 0, F(0))


@pytest.mark.parametrize("shape", [(1, 2), (2, 1)])
def test_maximal_pair_clock_matches_global_energy_and_improves_legal_ceiling(shape):
    # All two-cell ternary-field states, including nonzero local potentials.
    # The oracle sums actual global energy rather than copying local exponents.
    law = LocalLaw(*shape, 2, 2, (0, 1, 4, 9, 16))
    checked = 0
    for states in product(list(product(range(3), repeat=3)), repeat=2):
        base = OpenMedium(LocalMedium(law, _pack(law, states)),
                          ReservoirLaw(3, content_cost=1, chemical_base=-1, stimulus=0))
        before = base.state_dict()
        energy = _energy(base)
        for forces in [(0, 0), (F(-3, 7), F(4, 11)), (F(1, 7), F(1, 7))]:
            for index in range(base.events):
                accessed = []
                def force(edge):
                    accessed.append(edge)
                    return forces[edge]
                edits, e, bound, sign = prepare_gradient_event(base, index, 0, force)
                assert len(accessed) <= 2
                assert set(accessed) <= {edge for edge, _ in edits}
                if e is None:
                    continue
                after = _successor(base, edits, sign)
                back, er, br, sr = prepare_gradient_event(
                    after, base.reverse(index), 0, forces.__getitem__)
                reverted = _successor(after, back, sr)
                assert reverted.body.payload == base.body.payload and reverted.stock == base.stock
                work = sum(forces[j] * (sum(state) - sum(states[j])) for j, state in edits)
                delta = _energy(after) - energy + work
                assert e - er == -2 * delta
                assert bound == br == max(e, er)
                assert e - bound == min(0, -2 * delta)
                old = prepare_source_gradient_event(base, index, 0, forces.__getitem__)
                assert old[0] == edits and old[1] == e and old[3] == sign
                assert bound <= old[2]
                assert sum(map(sum, states)) + base.stock == sum(
                    sum(after.body.get(j)) for j in range(law.edges)) + after.stock
                checked += 1
        assert base.state_dict() == before
    assert checked > 10000


def _optimizer(seed=13):
    law = LocalLaw(1, 2, 1, 1, (0, 0, 0), epsilon=0, coupling=3)
    medium = OpenMedium(LocalMedium(law, _pack(law, [(1, 0, 0), (1, 1, 1)])),
                        ReservoirLaw(2, content_cost=1, stimulus=0), opportunities_per_tick=12)
    p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    return p, MaterialOptimizer([p], medium, quantum=F(1, 4), offset=F(-3, 8), seed=seed)


def test_material_optimizer_same_content_cache_conservation_and_complete_resume():
    p, optimizer = _optimizer()
    total = sum(sum(optimizer.medium.body.get(i)) for i in range(2)) + optimizer.medium.stock
    for _ in range(3):
        p.grad = torch.tensor([1.0, -2.0], dtype=p.dtype)
        optimizer.step()
        assert torch.equal(p.grad, torch.tensor([1.0, -2.0], dtype=p.dtype))
        for i in range(2):
            assert p[i].item() == float(parameter_value(optimizer.medium, i, F(1, 4), F(-3, 8)))
        assert sum(sum(optimizer.medium.body.get(i)) for i in range(2)) + optimizer.medium.stock == total
    assert optimizer.costs["attempts"] == 36
    snapshot = optimizer.state_dict()
    restored_p, restored = _optimizer(seed=999)
    restored.load_state_dict(snapshot)
    assert torch.equal(p, restored_p)
    for future_gradient in ([3.0, -4.0], [0.0, 0.0], [-2.0, 2.0]):
        p.grad = torch.tensor(future_gradient, dtype=p.dtype)
        restored_p.grad = p.grad.clone()
        optimizer.step()
        restored.step()
        assert torch.equal(p, restored_p)
        assert optimizer.state_dict() == restored.state_dict()


def test_actual_parameter_repair_and_force_tilt_are_existing_physical_events():
    law = LocalLaw(1, 1, 1, 1, (0, 0), epsilon=0, coupling=3)
    medium = OpenMedium(LocalMedium(law, _pack(law, [(1, 0, 1)])),
                        ReservoirLaw(1, content_cost=1, stimulus=0))
    p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    optimizer = MaterialOptimizer([p], medium, quantum=F(1))
    assert p.item() == 2
    # One specified source event and ticket: execution refinement, not a
    # stochastic retention/capability trial or a fitted memory target.
    medium.sample_proposal = lambda _: law.events
    optimizer._rng.getrandbits = lambda _: 0
    p.grad = torch.zeros_like(p)
    optimizer.step()
    assert p.item() == 3 and medium.body.get(0) == (1, 1, 1)
    medium.sample_proposal = lambda _: law.events + 1
    p.grad.fill_(5)
    optimizer.step()
    assert p.item() == 2 and medium.body.get(0) == (1, 0, 1)


def test_material_optimizer_uses_derived_clock_for_actual_ternary_repair():
    law = LocalLaw(1, 1, 2, 2, (0, 0, 0), epsilon=0, coupling=12)
    medium = OpenMedium(LocalMedium(law, _pack(law, [(2, 1, 2)])),
                        ReservoirLaw(1, content_cost=16, stimulus=0))
    p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    optimizer = MaterialOptimizer([p], medium, quantum=F(1))
    assert law.states == 27 and law.bits == 5 and p.item() == 5
    # Prescribed source events, not a stochastic capability trial. Both are
    # downhill in their declared force regime and must have acceptance one.
    medium.sample_proposal = lambda _: law.events
    def unexpected_coin(_):
        raise AssertionError("acceptance-one event must not request a random coin")
    optimizer._rng.getrandbits = unexpected_coin
    p.grad = torch.zeros_like(p)
    optimizer.step()
    assert p.item() == 6 and medium.body.get(0) == (2, 2, 2) and medium.stock == 0
    medium.sample_proposal = lambda _: law.events + 1
    p.grad.fill_(192)
    optimizer.step()
    assert p.item() == 5 and medium.body.get(0) == (2, 1, 2) and medium.stock == 1
    assert optimizer.costs["attempts"] == optimizer.costs["accepted"] == 2
    assert optimizer.costs["random_bits"] == 0 and not optimizer.state


def test_material_optimizer_nonfinite_gradient_is_transactional():
    p, optimizer = _optimizer()
    before, weights = optimizer.state_dict(), p.detach().clone()
    p.grad = torch.tensor([float("nan"), 1.0], dtype=p.dtype)
    with pytest.raises(ValueError):
        optimizer.step()
    assert optimizer.state_dict() == before and torch.equal(p, weights)


@pytest.mark.parametrize("cross_parameter", [False, True])
def test_gradient_cache_alias_is_rejected_before_payment_or_state_changes(cross_parameter):
    law = LocalLaw(1, 2, 1, 1, (0, 0, 0), epsilon=0, coupling=3)
    medium = OpenMedium(LocalMedium(law, _pack(law, [(1, 0, 1), (1, 1, 0)])),
                        ReservoirLaw(2, content_cost=1, stimulus=0))
    p, q = [torch.nn.Parameter(torch.zeros(1, dtype=torch.float64)) for _ in range(2)]
    optimizer = MaterialOptimizer([p, q], medium, quantum=F(1))
    p.grad = (q if cross_parameter else p).detach()
    before, weights = optimizer.state_dict(), [p.detach().clone(), q.detach().clone()]
    payments = []
    with pytest.raises(ValueError, match="alias"):
        optimizer.step(pay_write=lambda fee: payments.append(fee) or True)
    assert payments == [] and optimizer.state_dict() == before
    assert torch.equal(p, weights[0]) and torch.equal(q, weights[1])


def test_exact_ticket_uses_bounded_workspace_for_large_rejection_exponents():
    _, optimizer = _optimizer()
    requested = []
    def reject(bits):
        requested.append(bits)
        return 1
    optimizer._rng.getrandbits = reject
    assert optimizer._zero_ticket(10**100) == (False, 32)
    assert requested == [32]
    optimizer._rng.getrandbits = lambda _: 0
    assert optimizer._zero_ticket(70) == (True, 70)
    assert optimizer._zero_ticket(0) == (True, 0)


def test_bad_checkpoint_does_not_partially_restore_physical_weights_or_rng():
    p, optimizer = _optimizer()
    before, weights = optimizer.state_dict(), p.detach().clone()
    bad = deepcopy(before)
    bad["param_groups"][0]["lr"] = -1
    with pytest.raises(ValueError):
        optimizer.load_state_dict(bad)
    assert optimizer.state_dict() == before and torch.equal(p, weights)
    bad = deepcopy(before)
    bad["material_optimizer"]["medium"]["stock"] += 1
    with pytest.raises(ValueError):
        optimizer.load_state_dict(bad)
    assert optimizer.state_dict() == before and torch.equal(p, weights)
    bad = deepcopy(before)
    bad["material_optimizer"]["costs"]["accepted"] = 1
    with pytest.raises(ValueError):
        optimizer.load_state_dict(bad)
    assert optimizer.state_dict() == before and torch.equal(p, weights)
    bad = deepcopy(before)
    bad["material_optimizer"]["clock"] = "source_clock_v2"
    with pytest.raises(ValueError):
        optimizer.load_state_dict(bad)
    assert optimizer.state_dict() == before and torch.equal(p, weights)
    with pytest.raises(ValueError):
        optimizer.add_param_group({"params": [torch.nn.Parameter(torch.zeros(1))]})


def test_overlapping_float_cache_parameters_are_rejected_before_initialization():
    law = LocalLaw(1, 2, 1, 1, (0, 0, 0))
    medium = OpenMedium(LocalMedium(law, b"\x00"), ReservoirLaw(2, stimulus=0))
    shared = torch.tensor([4.0], dtype=torch.float64)
    p, q = torch.nn.Parameter(shared), torch.nn.Parameter(shared)
    with pytest.raises(ValueError):
        MaterialOptimizer([p, q], medium, quantum=F(1))
    assert shared.item() == 4


def test_finite_precision_alphabet_has_exact_cache_and_declared_storage():
    # cap=5 gives 16 readout values but 216 physical microstates: 8 stored
    # bits, not a claim that all sixteen values are stable memory codes.
    law = LocalLaw(1, 1, 5, 5, (0,) * 6, epsilon=0, coupling=3)
    assert law.states == 216 and law.bits == 8 and law.state_bytes == 1
    medium = OpenMedium(LocalMedium(law, _pack(law, [(2, 3, 4)])),
                        ReservoirLaw(5, content_cost=1, stimulus=0))
    p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = MaterialOptimizer([p], medium, quantum=F(1, 128), offset=F(-15, 256))
    assert F.from_float(p.item()) == F(-15, 256) + F(9, 128)
    before = p.detach().clone()
    with pytest.raises(ValueError):
        MaterialOptimizer([p], medium, quantum=F(1, 3))
    assert torch.equal(p, before)


def test_random_ledger_includes_proposal_and_acceptance_draws():
    p, optimizer = _optimizer()
    requested = []
    original = optimizer._rng.getrandbits
    def counted(bits):
        requested.append(bits)
        return original(bits)
    optimizer._rng.getrandbits = counted
    p.grad = torch.ones_like(p)
    optimizer.step()
    assert optimizer.costs["random_bits"] == sum(requested) > 0


def test_fractional_probability_enclosures_have_exact_algebraic_certificates():
    # For rational a/b the target p obeys p**b=2**(-a); no floating
    # transcendental implementation is used as this test's oracle.
    for denominator in range(2, 18):
        for numerator in range(denominator):
            power = F(numerator, denominator)
            for digits in (1, 4, 12, 24):
                low, high = fractional_probability_bounds(power, digits)
                reduced = power.denominator
                target_power = F(1, 2**power.numerator)
                assert low**reduced <= target_power <= high**reduced
                assert high - low <= F(2, 2**digits)


def test_nonzero_subquantum_gradient_changes_the_actual_event_kernel():
    p, optimizer = _optimizer()
    index = optimizer.medium.body.law.events  # Q addition at cell 0
    quiet = prepare_gradient_event(optimizer.medium, index, 0, lambda _: 0)
    tiny = 2.0**-80
    p.grad = torch.full_like(p, tiny)
    positive = prepare_gradient_event(optimizer.medium, index, 0, optimizer._force)
    assert optimizer._force(0) == F(1, 2**82)
    p.grad.neg_()
    negative = prepare_gradient_event(optimizer.medium, index, 0, optimizer._force)
    assert positive[1] < quiet[1] < negative[1]
    assert encode_gradient_force(F(1, 2**82), F(1)) == 0  # old blind band


def test_fractional_coin_uses_the_proved_rational_geometric_mixture():
    _, optimizer = _optimizer()
    for words, expected, bits in [([1, 0], False, 2), ([1, 1], True, 2),
                                  ([0, 1, 0, 1], True, 4),
                                  ([0, 1, 0, 0, 0], False, 5),
                                  ([0, 0, 1, 0, 0, 0, 0], False, 7)]:
        stream = iter(words)
        optimizer._rng.getrandbits = lambda n: next(stream)
        assert optimizer._probability_ticket(F(1, 2)) == (expected, bits)


def test_rational_coin_all_finite_prefixes_match_exact_acceptance_and_pending_mass():
    _, optimizer = _optimizer()
    probabilities = {F(n, d) for d in range(1, 10) for n in range(d+1)}
    probabilities |= {F(1, 2**200), 1-F(1, 2**200)}
    for probability in probabilities:
        for depth in range(7):
            accepted = pending = 0
            for bits in product((0, 1), repeat=depth):
                stream = iter(bits)
                optimizer._rng.getrandbits = lambda n: next(stream)
                try:
                    result, used = optimizer._rational_ticket(probability)
                    assert used <= depth
                    accepted += int(result)
                except StopIteration:
                    pending += 1
            scaled = probability * 2**depth
            lower = scaled.numerator // scaled.denominator
            upper = -(-scaled.numerator // scaled.denominator)
            assert accepted == lower and pending == upper-lower
            assert pending <= 1
    before = optimizer.state_dict()
    for invalid in (F(-1, 7), F(8, 7)):
        with pytest.raises(ValueError):
            optimizer._rational_ticket(invalid)
    assert optimizer.state_dict() == before


def test_tiny_fractional_force_keeps_rare_outcome_beyond_fixed_float_precision():
    _, optimizer = _optimizer()
    stream = iter([1] + [0]*200)  # N=0, then an exact rare rational outcome
    optimizer._rng.getrandbits = lambda n: next(stream)
    accepted, used = optimizer._probability_ticket(F(1, 2**200))
    assert accepted is False and used > 200
    assert optimizer.costs["max_rate_precision"] > 200
    stream = iter([1, 1])
    optimizer._rng.getrandbits = lambda n: next(stream)
    assert optimizer._probability_ticket(F(1, 2**200)) == (True, 2)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_old_force_or_clock_checkpoint_is_not_silently_reinterpreted(version):
    p, optimizer = _optimizer()
    before = optimizer.state_dict()
    bad = deepcopy(before)
    bad["material_optimizer"]["version"] = version
    with pytest.raises(ValueError):
        optimizer.load_state_dict(bad)
    assert optimizer.state_dict() == before
    bad = deepcopy(before)
    bad["material_optimizer"]["sampler"] = "incompatible_sampling_sequence"
    with pytest.raises(ValueError):
        optimizer.load_state_dict(bad)
    assert optimizer.state_dict() == before


def test_coupled_force_units_cannot_reverse_true_gradient_work():
    law = LocalLaw(1, 2, 1, 1, (0, 0, 0))
    medium = OpenMedium(LocalMedium(law, b"\x00"), ReservoirLaw(2, stimulus=0))
    p, q = torch.nn.Parameter(torch.tensor([4.])), torch.nn.Parameter(torch.tensor([5.]))
    with pytest.raises(ValueError):
        MaterialOptimizer([{"params": [p], "lr": 3.0}, {"params": [q], "lr": 1.0}],
                          medium, quantum=F(1))
    assert p.item() == 4 and q.item() == 5
    optimizer = MaterialOptimizer([{"params": [p]}, {"params": [q]}], medium, quantum=F(1))
    optimizer.param_groups[0]["lr"] = 3.0
    before = optimizer.state_dict()
    with pytest.raises(ValueError):
        optimizer.step()
    assert optimizer.state_dict() == before


def test_finite_world_funds_actual_attempts_and_time_continues_after_refusal():
    # Payment/clock interface audit only; these bytes are not a learning task.
    p, optimizer = _optimizer()
    world = PrefixReplayWorld(b"abc", source_id="unit:optimizer-payment", reserve=3, capacity=3)
    assert world.begin().active
    world.publish()
    p.grad = torch.ones_like(p)
    optimizer.step(pay_write=world.pay_write)
    assert optimizer.costs["attempts"] == 2 and optimizer.costs["fee_units"] == 2
    assert optimizer.costs["refused"] == 1
    state = world.state_dict()["state"]
    assert state["reserve"] == 0 and state["counts"]["write_spent"] == 2
    assert state["counts"]["maintenance_spent"] == 1
    physical, random_state, weights = optimizer.medium.state_dict(), optimizer._rng.getstate(), p.detach().clone()
    assert world.begin().active is False
    assert world.publish().tick == 1
    optimizer.step(pay_write=world.pay_write)
    assert optimizer.medium.state_dict() == physical and torch.equal(p, weights)
    assert optimizer._rng.getstate() == random_state
    assert optimizer.costs["attempts"] == 2 and optimizer.costs["refused"] == 2
    assert world.state_dict()["state"]["tick"] == 2


def test_logarithmic_budget_powering_encloses_independent_exact_powers():
    from scripts.witness.validate_optimizer_content_bridge import dyadic_power_bounds
    for base in [F(0), F(1), F(1, 2), F(3, 7), F(65535, 65536)]:
        for horizon in [0, 1, 2, 3, 17, 127]:
            lower, upper = dyadic_power_bounds(base, horizon)
            assert lower <= base**horizon <= upper


def test_generic_material_columns_are_independent_of_native_byte_alphabet():
    law = LocalLaw(1, 257, 2, 2, tuple(i*i for i in range(515)), epsilon=0, coupling=12)
    body = LocalMedium(law, bytes(law.state_bytes))
    body._put(256, (1, 1, 1))
    medium = OpenMedium(body, ReservoirLaw(1, content_cost=16, chemical_base=2, stimulus=0),
                        opportunities_per_tick=4)
    parameter = torch.nn.Parameter(torch.empty(257, dtype=torch.float32))
    optimizer = MaterialOptimizer([parameter], medium, quantum=F(1), seed=11)
    assert parameter[-1].item() == 3 and torch.count_nonzero(parameter).item() == 1
    reads = []
    def force(edge):
        reads.append(edge)
        return F(-96)
    edits, exponent, bound, sign = prepare_gradient_event(medium, law.events + 2*256, 0, force)
    assert reads == [256] and edits == ((256, (1, 2, 1)),) and sign == 1
    assert exponent == bound
    after = _successor(medium, edits, sign)
    _, reverse, _, _ = prepare_gradient_event(after, medium.reverse(law.events + 2*256), 0, force)
    assert exponent - reverse == -2 * (_energy(after) - _energy(medium) - 96)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert optimizer.costs["attempts"] == 4
    world = PrefixReplayWorld(b"x", source_id="unit:wide-material-layout", reserve=10, capacity=10)
    before = world.state_dict()
    with pytest.raises(ValueError, match="native 256-byte"):
        medium.tick_window((), world)
    assert world.state_dict() == before


def test_billion_cell_square_layout_metadata_without_allocating_body():
    # A shape/arithmetic check only. It neither allocates a billion-cell body
    # nor measures its peak memory, execution speed or learning quality.
    width = 32768
    law = LocalLaw(width, width, 2, 2, tuple(i*i for i in range(2*width+1)),
                   epsilon=0, coupling=12)
    assert law.edges == 2**30 and law.bits == 5
    assert law.state_bytes == 5 * 2**27
    assert law.events == law.edges * (2*width + 4)


def _resource_optimizer(*, cache_totals=False):
    law = LocalLaw(2, 2, 2, 2, (0, 1, 4, 9, 16), epsilon=0, coupling=12)
    states = [(2, 2, 2), (0, 0, 0), (0, 0, 0), (2, 2, 2)]
    medium = OpenMedium(LocalMedium(law, _pack(law, states), cache_totals=cache_totals),
                        ReservoirLaw(0, content_cost=16, chemical_base=2, stimulus=0),
                        opportunities_per_tick=3)
    p = torch.nn.Parameter(torch.empty(4, dtype=torch.float64))
    return p, MaterialOptimizer([p], medium, quantum=F(1, 4), offset=F(-1, 2), seed=17)


def test_external_loss_preserves_material_and_updates_owned_cache_transactionally():
    for field in range(3):
        p, optimizer = _resource_optimizer(cache_totals=True)
        p.grad = torch.ones_like(p)
        gradient, costs, random_state = p.grad.clone(), deepcopy(optimizer.costs), optimizer._rng.getstate()
        for edge, bad_field in [(-1, field), (4, field), (True, field), (0, -1), (0, 3), (0, True)]:
            before = optimizer.state_dict()
            with pytest.raises(ValueError):
                optimizer.apply_external_loss(edge, bad_field)
            assert optimizer.state_dict() == before
        assert optimizer.apply_external_loss(0, field) is True
        expected = [2, 2, 2]
        expected[field] = 1
        assert optimizer.medium.body.get(0) == tuple(expected)
        assert p[0].item() == F(3, 4) and optimizer.medium.stock == 1
        assert optimizer.medium.released == 1
        body = optimizer.medium.body
        assert body.total(0, 0) == sum(body.get(e)[0] for e in [0, 1])
        assert body.total(1, 0) == sum(body.get(e)[1] for e in [0, 2])
        assert sum(sum(optimizer.medium.body.get(e)) for e in range(4)) + optimizer.medium.stock == 12
        before = optimizer.state_dict()
        assert optimizer.apply_external_loss(1, field) is False
        assert optimizer.state_dict() == before
        assert torch.equal(p.grad, gradient) and optimizer.costs == costs
        assert optimizer._rng.getstate() == random_state


def test_external_loss_runs_only_after_funding_before_native_proposal():
    p, optimizer = _resource_optimizer()
    before = optimizer.state_dict()
    with pytest.raises(TypeError):
        optimizer.step(before_attempt=7)
    assert optimizer.state_dict() == before
    order, payments = [], iter([True, True, False])
    def pay(_fee):
        order.append("pay")
        return next(payments)
    def loss():
        order.append("loss")
        assert optimizer.costs["fee_units"] == order.count("pay")
        optimizer.apply_external_loss(0, 0)
    def proposal(_rng):
        order.append("proposal")
        return 0  # P->Q is blocked because this Q field remains full.
    optimizer.medium.sample_proposal = proposal
    optimizer.step(pay_write=pay, before_attempt=loss)
    assert order == ["pay", "loss", "proposal", "pay", "loss", "proposal", "pay"]
    assert optimizer.costs["attempts"] == 2 and optimizer.costs["fee_units"] == 2
    assert optimizer.costs["refused"] == 1 and optimizer.medium.stock == 2
    assert optimizer.medium.body.get(0) == (0, 2, 2) and p[0].item() == F(1, 2)


def test_external_world_and_optimizer_resume_with_the_same_future_events():
    import random
    p, optimizer = _resource_optimizer()
    world_rng = random.Random(91)
    def callback(opt, rng):
        def loss():
            # Deliberately frequent deterministic interface coverage, not a
            # retention trial or the registered 1/1024 probability workload.
            edge, field = divmod(rng.randrange(12), 3)
            opt.apply_external_loss(edge, field)
        return loss
    for _ in range(2):
        optimizer.step(before_attempt=callback(optimizer, world_rng))
    snapshot, world_state = deepcopy(optimizer.state_dict()), world_rng.getstate()
    q, resumed = _resource_optimizer()
    resumed.load_state_dict(snapshot)
    other_rng = random.Random()
    other_rng.setstate(world_state)
    for _ in range(4):
        optimizer.step(before_attempt=callback(optimizer, world_rng))
        resumed.step(before_attempt=callback(resumed, other_rng))
        assert torch.equal(p, q) and optimizer.state_dict() == resumed.state_dict()
        assert world_rng.getstate() == other_rng.getstate()


@pytest.mark.parametrize("shape", [(2, 2), (2, 3), (3, 2)])
def test_scaled_pools_match_global_energy_reverse_odds_and_bounded_local_work(shape):
    law = LocalLaw(*shape, 2, 2, tuple(n*n for n in range(2*max(shape)+1)),
                   epsilon=0, coupling=12, pool_reference_size=2)
    assert 0 < law.max_potential_step < 8
    checked = 0
    for state in product(range(3), repeat=3):
        for background in ((0, 0, 0), (2, 2, 2), (2, 0, 1)):
            states = [state] + [background]*(law.edges-1)
            medium = OpenMedium(LocalMedium(law, _pack(law, states)),
                                ReservoirLaw(1, content_cost=16, chemical_base=2, stimulus=0))
            before, energy = medium.state_dict(), _energy(medium)
            for forces in ([F(0)]*law.edges, [F(2*e-3, 7) for e in range(law.edges)],
                           [F(9, 11)]*law.edges):
                for index in range(medium.events):
                    edits, e, b, sign = prepare_gradient_event(medium, index, 0, forces.__getitem__)
                    if e is None:
                        continue
                    after = _successor(medium, edits, sign)
                    back, er, br, sr = prepare_gradient_event(
                        after, medium.reverse(index), 0, forces.__getitem__)
                    reverted = _successor(after, back, sr)
                    work = sum(forces[j]*(sum(new)-sum(states[j])) for j, new in edits)
                    delta = _energy(after)-energy+work
                    assert e-er == -2*delta and e-b == min(0, -2*delta)
                    assert b == br == max(e, er)
                    assert reverted.body.payload == medium.body.payload and reverted.stock == medium.stock
                    checked += 1
            assert medium.state_dict() == before
    assert checked > 1000


def test_scaled_full_column_has_uniform_Q_repair_margin_as_size_grows():
    for size in (1, 2, 3, 8, 9, 64):
        law = LocalLaw(size, 1, 2, 2, tuple(n*n for n in range(2*size+1)),
                       epsilon=0, coupling=12, pool_reference_size=2)
        states = [(2, 2, 2)]*size
        medium = OpenMedium(LocalMedium(law, _pack(law, states)),
                            ReservoirLaw(0, content_cost=16, chemical_base=2, stimulus=0))
        edits, e, b, sign = prepare_gradient_event(medium, law.events+1, 0, lambda _: 0)
        assert e-b == -2*(26+F(2, size)) and sign == -1
        after = _successor(medium, edits, sign)
        _, er, br, sr = prepare_gradient_event(after, law.events, 0, lambda _: 0)
        assert er-br == 0 and sr == 1


def test_legacy_medium_snapshot_upgrades_only_to_the_unscaled_law():
    raw = LocalLaw(2, 3, 2, 2, tuple(n*n for n in range(7)), epsilon=0, coupling=12)
    states = [(2, 1, 2), (0, 1, 0), (1, 0, 0), (0, 0, 0), (1, 2, 0), (2, 2, 2)]
    body = LocalMedium(raw, _pack(raw, states))
    legacy = deepcopy(body.state_dict())
    legacy["version"] = 2
    del legacy["law"]["pool_reference_size"]
    original = deepcopy(legacy)
    clone = LocalMedium(raw, bytes(raw.state_bytes))
    clone.load_state_dict(legacy)
    assert clone.state_dict() == body.state_dict() and legacy == original
    scaled = LocalLaw(2, 3, 2, 2, raw.potential, epsilon=0, coupling=12, pool_reference_size=2)
    other = LocalMedium(scaled, bytes(scaled.state_bytes))
    before = other.state_dict()
    with pytest.raises(ValueError):
        other.load_state_dict(legacy)
    assert other.state_dict() == before
    for invalid in (-1, True, F(1, 2)):
        with pytest.raises(ValueError):
            LocalLaw(2, 3, 2, 2, raw.potential, pool_reference_size=invalid)


def test_scaled_optimizer_resumes_same_fractional_kernel_and_rejects_changed_pool_law():
    def make():
        law = LocalLaw(2, 3, 2, 2, tuple(n*n for n in range(7)),
                       epsilon=0, coupling=12, pool_reference_size=2)
        states = [(2, 1, 2), (0, 0, 0), (1, 2, 0), (0, 2, 1), (0, 0, 0), (2, 2, 2)]
        opened = OpenMedium(LocalMedium(law, _pack(law, states)),
                            ReservoirLaw(5, content_cost=16, chemical_base=2, stimulus=0),
                            opportunities_per_tick=24)
        p = torch.nn.Parameter(torch.zeros(6, dtype=torch.float64))
        return p, MaterialOptimizer([p], opened, quantum=F(1, 8), offset=F(-3, 8))
    p, optimizer = make()
    for _ in range(3):
        p.grad = torch.tensor([0., 1., -2., 2.**-80, 3., -1.], dtype=p.dtype)
        optimizer.step(before_attempt=lambda: optimizer.apply_external_loss(0, 1))
    snapshot = deepcopy(optimizer.state_dict())
    q, resumed = make()
    resumed.load_state_dict(snapshot)
    for step in range(3):
        p.grad = torch.tensor([step, -step, 1., -2., 0., 2.**-100], dtype=p.dtype)
        q.grad = p.grad.clone()
        optimizer.step()
        resumed.step()
        assert torch.equal(p, q) and optimizer.state_dict() == resumed.state_dict()
    before = deepcopy(resumed.state_dict())
    bad = deepcopy(before)
    bad["material_optimizer"]["medium"]["body"]["law"]["pool_reference_size"] = 0
    with pytest.raises(ValueError):
        resumed.load_state_dict(bad)
    assert resumed.state_dict() == before


def test_scaled_pools_reject_unadapted_executors_before_world_or_array_effects():
    from information_boltzmann.core.cgb_local_medium_ct import ContinuousOpenMedium
    law = LocalLaw(2, 2, 2, 2, (0, 1, 4, 9, 16), pool_reference_size=2)
    medium = OpenMedium(LocalMedium(law, bytes(law.state_bytes)), ReservoirLaw(1, stimulus=0))
    class UnusedWorld:
        def __getattr__(self, name):
            raise AssertionError(f"world was touched: {name}")
    before = medium.state_dict()
    with pytest.raises(ValueError, match="rational"):
        medium.body.tick(0, 0, UnusedWorld())
    with pytest.raises(ValueError, match="rational"):
        medium.tick_window((), UnusedWorld())
    with pytest.raises(ValueError, match="unscaled"):
        ContinuousOpenMedium(medium, tau=1, max_events=1, family_conductance=(1,)*5, seed=1)
    assert medium.state_dict() == before


def _fiber_medium(states, rows=2, labels=2, stock=2, opportunities=12):
    law = LocalLaw(rows, labels, 2, 2,
                   tuple(n*n for n in range(2*max(rows, labels)+1)),
                   epsilon=0, coupling=12, pool_reference_size=2)
    return OpenMedium(LocalMedium(law, _pack(law, states)),
                      ReservoirLaw(stock, content_cost=16, chemical_base=2, stimulus=0),
                      opportunities_per_tick=opportunities)


def _fiber_kernel_row(medium, edge, force):
    """Exact finite row for INTEGER powers; this is an arithmetic oracle."""
    old = medium.body.get(edge)
    row = {old: F(1)}
    for direction in (-1, 1):
        block = prepare_readout_fiber(medium, edge, direction, lambda _: force)
        for state, power, _ in block:
            assert power.denominator == 1
            rate = F(1, 2*len(block)*2**power.numerator)
            row[state] = row.get(state, F(0)) + rate
            row[old] -= rate
    return row


def test_readout_fiber_local_energy_matches_global_and_reverse_block():
    checked = 0
    for shape in ((1, 1), (2, 3), (3, 2)):
        size = shape[0]*shape[1]
        for cell in product(range(3), repeat=3):
            states = [cell] + [(i % 3, (i+1) % 3, (i+2) % 3) for i in range(size-1)]
            for stock in (0, 1):
                medium = _fiber_medium(states, *shape, stock=stock)
                before, energy = deepcopy(medium.state_dict()), _energy(medium)
                for direction in (-1, 1):
                    for force in (F(0), F(-96), F(2, 7), F(1, 2**200)):
                        reads = []
                        def observe(edge):
                            reads.append(edge)
                            return force
                        block = prepare_readout_fiber(medium, 0, direction, observe)
                        assert len(block) <= 13
                        assert reads == ([0] if block else [])
                        if not block:
                            continue
                        diffs = []
                        for state, power, delta in block:
                            assert abs(delta) <= 1 and sum(state)-sum(cell) == delta
                            after = _successor(medium, ((0, state),), delta)
                            diffs.append(_energy(after)-energy+force*delta)
                            assert after.stock >= 0
                        floor = min(diffs)
                        assert [power for _, power, _ in block] == [2*(v-floor) for v in diffs]
                        assert min(v for _, v, _ in block) == 0
                        for state, _, delta in block:
                            after = _successor(medium, ((0, state),), delta)
                            reverse_direction = -direction if delta else direction
                            reverse = prepare_readout_fiber(after, 0, reverse_direction, lambda _: force)
                            assert {s: p for s, p, _ in reverse} == {s: p for s, p, _ in block}
                            checked += 1
                assert medium.state_dict() == before
    assert checked == 9576


def test_readout_fiber_exact_rows_balance_and_stock_boundary():
    cells = list(product(range(3), repeat=3))
    for total in range(9):
        legal = [s for s in cells if sum(s) <= total]
        for force in (-35, 0, 23):
            rows, weights = {}, {}
            energies = {}
            for state in legal:
                medium = _fiber_medium([state], 1, 1, stock=total-sum(state))
                rows[state] = _fiber_kernel_row(medium, 0, force)
                energies[state] = _energy(medium)+force*sum(state)
                assert sum(rows[state].values()) == 1
                assert min(rows[state].values()) >= 0
                assert all(t in legal and abs(sum(t)-sum(state)) <= 1 for t in rows[state])
            floor = min(energies.values())
            for state in legal:
                power = 2*(energies[state]-floor)
                assert power.denominator == 1
                weights[state] = F(1, 2**power.numerator)
            for x in legal:
                for y in legal:
                    assert weights[x]*rows[x].get(y, 0) == weights[y]*rows[y].get(x, 0)


def test_readout_fiber_force_service_and_single_defect_regeneration():
    for state in product(range(3), repeat=3):
        for shape in ((1, 1), (2, 2), (1, 17), (17, 1)):
            size = shape[0]*shape[1]
            medium = _fiber_medium([state]+[(2, 2, 2)]*(size-1), *shape, stock=1)
            if sum(state):
                block = prepare_readout_fiber(medium, 0, -1, lambda _: 35)
                assert any(sum(s) == sum(state)-1 and power == 0 for s, power, _ in block)
                assert F(1, 2*len(block)) >= F(1, 26)
            if sum(state) < 6:
                block = prepare_readout_fiber(medium, 0, 1, lambda _: -23)
                assert any(sum(s) == sum(state)+1 and power == 0 for s, power, _ in block)
                assert F(1, 2*len(block)) >= F(1, 26)
            if sum(state) == 5:
                for force in (-1, 0, 1):
                    block = prepare_readout_fiber(medium, 0, 1, lambda _: force)
                    assert len(block) == 4
                    assert next(power for s, power, _ in block if s == (2, 2, 2)) == 0
                    assert F(1, 2*len(block)) == F(1, 8)


def test_readout_fiber_paid_order_denial_and_cost_bound():
    p = torch.nn.Parameter(torch.zeros(4))
    opt = FiberMaterialOptimizer([p], _fiber_medium([(2, 2, 2), (0, 0, 0)]*2),
                                 quantum=F(1, 8), seed=27)
    p.grad = torch.ones_like(p)
    before = deepcopy(opt.state_dict())
    calls = []
    opt.step(pay_write=lambda fee: calls.append(('refuse', fee)) or False,
             before_attempt=lambda: calls.append(('world',)))
    after = opt.state_dict()
    assert calls == [('refuse', 13)]
    assert after['material_optimizer']['medium'] == before['material_optimizer']['medium']
    assert after['material_optimizer']['rng'] == before['material_optimizer']['rng']
    assert opt.costs['attempts'] == opt.costs['candidate_energy_evals'] == 0
    def pay(fee):
        calls.append(('pay', fee, opt.costs['attempts']))
        return True
    def world():
        calls.append(('world', opt.costs['attempts'], opt.costs['fee_units']))
    calls.clear()
    opt.step(pay_write=pay, before_attempt=world)
    assert len(calls) == 24
    assert calls[:2] == [('pay', 13, 0), ('world', 1, 13)]
    assert opt.costs['fee_units'] == 13*opt.costs['attempts'] == 156
    assert 0 <= opt.costs['candidate_energy_evals'] <= 13*opt.costs['attempts']


def test_readout_fiber_checkpoint_full_continuation_and_clock_rejection():
    def make(cls=FiberMaterialOptimizer):
        p = torch.nn.Parameter(torch.zeros(4))
        return p, cls([p], _fiber_medium([(2, 1, 2), (0, 0, 0)]*2),
                      quantum=F(1, 8), seed=78)
    p, opt = make()
    p.grad = torch.tensor([0.25, -0.5, 1.0, -2.0])
    opt.step(before_attempt=lambda: opt.apply_external_loss(0, 0))
    snapshot = deepcopy(opt.state_dict())
    q, restored = make()
    restored.load_state_dict(snapshot)
    for _ in range(5):
        q.grad = p.grad.clone()
        opt.step(before_attempt=lambda: opt.apply_external_loss(2, 1))
        restored.step(before_attempt=lambda: restored.apply_external_loss(2, 1))
        assert opt.state_dict() == restored.state_dict()
        assert torch.equal(p, q)
    _, native = make(MaterialOptimizer)
    for receiver, bad in ((native, snapshot), (restored, native.state_dict())):
        before = deepcopy(receiver.state_dict())
        with pytest.raises(ValueError):
            receiver.load_state_dict(bad)
        assert receiver.state_dict() == before
    bad = deepcopy(snapshot)
    bad['material_optimizer']['costs']['candidate_energy_evals'] = 13*opt.costs['attempts']+1
    before = deepcopy(restored.state_dict())
    with pytest.raises(ValueError):
        restored.load_state_dict(bad)
    assert restored.state_dict() == before


def test_readout_fiber_has_same_carrier_and_no_parameter_moments():
    p = torch.nn.Parameter(torch.zeros(4))
    medium = _fiber_medium([(2, 2, 2), (0, 0, 0)]*2)
    bytes_before = (len(medium.body.payload), medium.body.cache_bytes)
    opt = FiberMaterialOptimizer([p], medium, quantum=F(1, 8), seed=3)
    assert opt.state == {} and opt.state_dict()['state'] == {}
    p.grad = torch.zeros_like(p)
    opt.step()
    assert (len(medium.body.payload), medium.body.cache_bytes) == bytes_before
    assert opt.state == {}
    assert set(opt.costs) == {'attempts', 'accepted', 'fee_units', 'random_bits', 'refused',
                              'coefficient_steps', 'max_rate_precision', 'candidate_energy_evals'}
    body_mass = sum(sum(medium.body.get(e)) for e in range(4))
    assert body_mass+medium.stock == medium.initial_content+medium.reservoir.source_initial
    assert medium.intake-medium.released == 2-medium.stock
    assert torch.equal(p, torch.tensor([float(parameter_value(medium, e, F(1, 8))) for e in range(4)]))


def test_readout_fiber_invalid_contracts_fail_before_mutation():
    law = LocalLaw(1, 1, 1, 1, (0, 0))
    medium = OpenMedium(LocalMedium(law, _pack(law, [(0, 0, 0)])), ReservoirLaw(1))
    p = torch.nn.Parameter(torch.tensor([17.0]))
    with pytest.raises(ValueError):
        FiberMaterialOptimizer([p], medium, quantum=F(1))
    assert p.item() == 17.0
    medium = _fiber_medium([(1, 1, 1)]*4)
    before = medium.state_dict()
    for direction in (0, True, 2):
        with pytest.raises(ValueError):
            prepare_readout_fiber(medium, 0, direction, lambda _: 0)
    with pytest.raises(TypeError):
        prepare_readout_fiber(medium, 0, 1, lambda _: 0.1)
    with pytest.raises(ValueError):
        prepare_readout_fiber(medium, 4, 1, lambda _: 0)
    assert medium.state_dict() == before


class _EnumeratedFiber(FiberMaterialOptimizer):
    """Previous executor as an independent all-candidate arithmetic oracle."""

    def _prepare_attempt(self):
        edge = self._rng.randrange(self.medium.body.law.edges)
        direction = 1 if self._rng.randrange(2) else -1
        block = prepare_readout_fiber(self.medium, edge, direction, self._force)
        self.costs["candidate_energy_evals"] += len(block)
        if not block:
            return (), None, 0, 0
        state, power, sign = block[self._rng.randrange(len(block))]
        return ((edge, state),), -power, 0, sign


def test_reduced_fiber_all_candidates_match_exact_enumeration():
    from information_boltzmann.conjugate_gradient_port import _prepare_reduced_fiber

    for shape, state, stock, force in product(
            ((1, 1), (2, 3), (3, 2)), product(range(3), repeat=3),
            (0, 1), (F(-100), F(0), F(100), F(1, 7))):
        size = shape[0]*shape[1]
        medium = _fiber_medium([state]+[(2, 2, 2)]*(size-1), *shape, stock=stock)
        before = medium.state_dict()
        for direction in (-1, 1):
            expected = prepare_readout_fiber(medium, 0, direction, lambda _: force)
            if not expected:
                assert _prepare_reduced_fiber(medium, 0, direction, lambda _: force,
                                             lambda _: pytest.fail("blocked draw")) == (None, 0, 0, 0)
            for index, answer in enumerate(expected):
                observations = []
                actual = _prepare_reduced_fiber(
                    medium, 0, direction, lambda e: observations.append(e) or force,
                    lambda n: index if n == len(expected) else pytest.fail("candidate count"))
                assert actual[:3] == answer
                assert 2 <= actual[3] <= 4 and observations == [0]
        assert medium.state_dict() == before


def test_reduced_fiber_full_trajectory_and_previous_checkpoint_continuation():
    def make(cls):
        p = torch.nn.Parameter(torch.zeros(4))
        return p, cls([p], _fiber_medium([(2, 1, 2), (0, 0, 0)]*2, opportunities=24),
                      quantum=F(1, 8), seed=781, energy_unit=F(1, 64))

    old_p, old = make(_EnumeratedFiber)
    new_p, new = make(FiberMaterialOptimizer)
    for t in range(12):
        gradient = torch.tensor([(-1 if (t+e)%3 else 1)*float(1+e) for e in range(4)])
        old_p.grad, new_p.grad = gradient.clone(), gradient.clone()
        previous = new.costs['candidate_energy_evals']
        for opt in (old, new):
            opt.step(before_attempt=lambda o=opt: o.apply_external_loss(t%4, t%3))
        assert new.costs['candidate_energy_evals']-previous <= 4*24
        assert torch.equal(old_p, new_p)
        a, b = deepcopy(old.state_dict()), deepcopy(new.state_dict())
        a['material_optimizer']['costs'].pop('candidate_energy_evals')
        b['material_optimizer']['costs'].pop('candidate_energy_evals')
        assert a == b
        if t == 4:
            # A historical v5 checkpoint retains its already paid and counted
            # work; only future evaluations use the proved reduction.
            new.load_state_dict(deepcopy(old.state_dict()))
            assert new.state_dict() == old.state_dict()
    assert new.costs['candidate_energy_evals'] < old.costs['candidate_energy_evals']


def test_reduced_fiber_unregistered_laws_keep_exact_enumerated_fallback():
    from dataclasses import replace

    base = _fiber_medium([(1, 1, 1)]*4).body.law
    for options in ({'coupling': 11}, {'epsilon': 1}, {'pool_reference_size': 0},
                    {'potential': tuple(0 for _ in base.potential)}):
        law = replace(base, **options)
        optimizers = []
        for cls in (FiberMaterialOptimizer, _EnumeratedFiber):
            medium = OpenMedium(LocalMedium(law, _pack(law, [(1, 1, 1)]*4)),
                                ReservoirLaw(2, content_cost=16, chemical_base=2, stimulus=0),
                                opportunities_per_tick=24)
            p = torch.nn.Parameter(torch.zeros(4))
            opt = cls([p], medium, quantum=F(1, 8), seed=2)
            assert opt._reduced_floor_law is None
            p.grad = torch.tensor([1., -2., 3., -4.])
            opt.step()
            optimizers.append(opt)
        assert optimizers[0].state_dict() == optimizers[1].state_dict()
