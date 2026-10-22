"""Constitutive tests, not a synthetic learning-capability experiment."""
import math

import numpy as np
import pytest

from information_boltzmann.cgb.local_learning import depletion_shift, local_learning_affinity
from information_boltzmann.cgb.material_fork import fork_material
from information_boltzmann.cgb.mobile_exchange import MaterialEvent
from scripts.ib.generative.compare_support_transport import same
from test_cgb_growing_material import growing


def writable_material():
    m = growing()
    # Use admitted thermal recovery with its original heat receipt; do not
    # clear storage arrays or manufacture unaccounted initial compositions.
    for event in list(m.events()):
        if event.kind == 'thermal' and event.sign == -1:
            m.apply_event(event)
    return m


def test_affinity_decomposes_all_admitted_internal_and_boundary_writes():
    m = writable_material()
    before = m.state_dict()
    count = 0
    for event in m.events():
        if event.kind in ('write', 'active') and event.sign == 1:
            value = local_learning_affinity(m, event)
            assert value.delta_g == m.event_delta(event)[1]
            assert value.actual_rate == m.event_rate(event)
            assert value.electrical_source != value.electrical_target
            count += 1
    assert count
    same(before, m.state_dict())


def test_consuming_shared_substrate_raises_threshold_without_an_extra_target():
    m = writable_material()
    event = next(e for e in m.events() if e.kind == 'write' and e.sign == 1 and m.charged[e.i] > 1)
    before = local_learning_affinity(m, event)
    n = fork_material(m)
    # An actually admitted active event has exactly the same P/D and charge
    # stoichiometry, leaving the write storage available for the comparison.
    active = MaterialEvent('active', event.i, event.j, event.edge, interface=event.interface)
    n.apply_event(active)
    after = local_learning_affinity(n, event)
    shift = after.half_attempt_voltage_drop - before.half_attempt_voltage_drop
    assert shift == pytest.approx(before.next_consumption_threshold_shift, abs=2e-14)
    assert shift > 0
    assert max(abs(float(v)) for v in n.balances().values()) < 1e-10
    # Separate the additional electrical feedback caused by the same event.
    z = m.params.charge_per_event
    assert after.delta_g-before.delta_g == pytest.approx(
        z*(shift+before.voltage_drop-after.voltage_drop), abs=2e-14)


def test_post_voltage_and_shared_composition_have_distinct_native_roles():
    m = writable_material()
    event = next(e for e in m.events() if e.kind == 'write' and e.sign == 1 and e.interface < 0)
    original = local_learning_affinity(m, event)
    n = fork_material(m)
    q = n.charge[event.j]
    du0, g0 = n.energy(), n.free_energy()
    n.charge[event.j] += .2*n.capacitance[event.j]
    n.electrode_charge -= n.charge[event.j]-q
    du, dg = n.energy()-du0, n.free_energy()-g0
    n.work_in += du
    n.entropy_balance += du-dg
    changed = local_learning_affinity(n, event)
    assert changed.delta_g-original.delta_g == pytest.approx(.2*m.params.charge_per_event)
    assert changed.half_attempt_voltage_drop == original.half_attempt_voltage_drop
    assert changed.actual_rate < original.actual_rate
    # The existing channel detects a signed voltage drop, not an assumed
    # pre/post coincidence or the entire V2 plasticity law.


def test_substrate_exhaustion_is_admission_not_an_infinite_physical_energy():
    assert depletion_shift(1, 0) is None
    assert depletion_shift(4, 2) == pytest.approx(math.log(16/9))
    for bad in ((0, 1), (1, -1), (2.5, 0)):
        with pytest.raises(ValueError):
            depletion_shift(*bad)
    m = writable_material()
    event = next(e for e in m.events() if e.kind == 'write' and e.sign == 1)
    with pytest.raises(ValueError):
        local_learning_affinity(m, MaterialEvent('write', event.i, event.j, 999))
