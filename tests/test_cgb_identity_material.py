"""Identity constitutive calibration, joint thermodynamics and full restore."""
from dataclasses import replace
import math

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss

from information_boltzmann.cgb.identity_material import IdentityMaterialProgram, IdentityMobileExchange
from information_boltzmann.cgb.mobile_exchange import MobileExchange, MobileExchangeParams, MaterialEvent
from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange


def program():
    return IdentityMaterialProgram(dict(code=np.array([[1., .2], [-.4, 1.], [.3, -.7]]),
        occurrence_affinity=np.array([[.1, .5], [-.2, .2]]),
        occurrence_out=np.array([.2, -.1]), occurrence_in=np.array([-.1, .3]),
        strength_affinity=np.array([[.2, -.3], [.4, .1]]),
        strength_out=np.array([.2, -.1, .3]), strength_in=np.array([-.1, .3, -.2])),
        ['a', 'b', 'c'], {'scope': 'numerical constitutive test only'})


def material(alpha=0., c=0, r=0, **params):
    return IdentityMobileExchange([[1., 1., 1.], [1.8, 1., 1.], [1.4, 1.7, 1.]],
        [3, 3, 3], [2, 2, 2], [3, 3, 3], [2, 2, 2], carried=2,
        end_capacity=4, pre=[0, 1], post=[1, 2], bound=c, trace=r,
        charge=[.1, -.2, .3], params=MobileExchangeParams(**params), port_nodes=[0, 2],
        program=program(), cell_type=[0, 1, 2], recognition_energy_fraction=alpha, seed=17)


@pytest.mark.parametrize('alpha', [0., .35, 1.])
@pytest.mark.parametrize('c,r', [(0, 0), (1, 1)])
def test_all_identity_events_match_joint_u_g_and_reverse_rates(alpha, c, r):
    m = material(alpha, c, r)
    for event in m.events():
        clone = IdentityMobileExchange.from_state_dict(m.state_dict())
        u, g = clone.energy(), clone.free_energy()
        du, dg = clone.event_delta(event)
        plus = clone.event_rate(event)
        count = len(clone.pre)
        clone.apply_event(event)
        assert abs(clone.energy()-u-du) < 5e-14
        assert abs(clone.free_energy()-g-dg) < 1e-13
        reverse = replace(event, sign=-event.sign,
            edge=count if event.kind == 'assembly' and event.sign > 0 else
                 -1 if event.kind == 'assembly' else event.edge)
        assert reverse in clone.events()
        assert abs(math.log(plus/clone.event_rate(reverse))+dg) < 2e-14
        for value in clone.balances().values():
            assert abs(float(value)) < 1e-12


def test_reference_odds_tilt_is_centered_identity_difference_not_an_extra_geometry_gate():
    xyz = [[1., 1., 1.], [1.8, 1., 1.], [1.4, 1.+np.sqrt(.48), 1.]]
    m = IdentityMobileExchange(xyz, [3]*3, [2]*3, [3]*3, [2]*3,
        program=program(), cell_type=[0, 1, 2], recognition_energy_fraction=0.)
    a, b = MaterialEvent('assembly', 0, 1), MaterialEvent('assembly', 0, 2)
    difference = m.event_delta(a)[1]-m.event_delta(b)[1]
    h = m.program.recognition_field(np.array([0, 0]), np.array([1, 2]))
    assert abs(difference+(h[0]-h[1])) < 1e-14


def test_coordinate_rotation_and_uniform_strength_gauge_leave_the_program_unchanged():
    p = program()
    angle = .7
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    arrays = {k: v.copy() for k, v in p.arrays.items()}
    arrays['code'] = arrays['code']@rot
    for name in ('occurrence_affinity', 'strength_affinity'):
        arrays[name] = rot.T@arrays[name]@rot
    arrays['occurrence_out'] = rot.T@arrays['occurrence_out']
    arrays['occurrence_in'] = rot.T@arrays['occurrence_in']
    arrays['strength_out'] += 7.
    q = IdentityMaterialProgram(arrays, p.types)
    a, b = np.indices((3, 3)).reshape(2, -1)
    np.testing.assert_allclose(p.recognition_field(a, b), q.recognition_field(a, b), atol=2e-15)
    np.testing.assert_allclose(p.transport_log_scale(a, b), q.transport_log_scale(a, b), atol=2e-15)
    assert abs(p.recognition_field(a, b).mean()) < 1e-15
    assert abs(p.transport_log_scale(a, b).mean()) < 1e-15


def test_reciprocal_polarized_bridges_each_pay_real_ends_and_material():
    m = material()
    e = MaterialEvent('assembly', 1, 0)
    assert e in m.events()
    before = m.material.sum()
    m.apply_event(e)
    assert {(0, 1), (1, 0)} <= set(zip(m.pre, m.post))
    assert m.material.sum() == before-1
    assert np.all(m.remaining_ends() >= 0)
    assert e not in m.events()
    assert float(m.balances()['material']) == 0.


def test_static_reference_free_energy_does_not_identify_energy_entropy_split():
    a, b = material(0., assembly_rate=.2, bulk_attempt=1.), material(1., assembly_rate=.2, bulk_attempt=1.)
    f0 = a._recognition_total()
    assert abs(a.free_energy()-b.free_energy()) < 1e-13
    assert abs((b.energy()-a.energy())-f0) < 1e-13
    for m in (a, b):
        m.apply_event(MaterialEvent('assembly', 0, 2))
        for _ in range(8):
            m.step(.5, np.array([3., -3.]))
    for name in MobileExchange._arrays+('pre', 'post'):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert a.reaction_counts == b.reaction_counts
    assert a.rng_reaction.bit_generator.state == b.rng_reaction.bit_generator.state
    assert a.rng_motion.bit_generator.state == b.rng_motion.bit_generator.state
    assert a.rng_field.bit_generator.state == b.rng_field.bit_generator.state
    df = a._recognition_total()-f0
    assert abs(b.heat_out-a.heat_out+df) < 1e-12
    assert abs(b.free_energy()-a.free_energy()) < 1e-12


def test_identity_transport_scales_both_ou_mean_and_thermal_variance():
    m = IdentityMobileExchange([[1., 1., 1.], [1.8, 1., 1.]], [2, 2], [1, 1], [3, 3], [1, 1],
        pre=[0], post=[1], program=program(), cell_type=[0, 1], recognition_energy_fraction=0.,
        capacitance=[2., 3.], charge=[1., -1.], unit_conductance=.3)
    g = m.bridge_conductance()[0]
    nodes, weights = hermgauss(9)
    old = m.charge.copy()
    class FixedNormal:
        def __init__(self, value): self.value = value
        def normal(self): return self.value
    sent = []
    for z in nodes*np.sqrt(2):
        m.charge[:] = old
        m.rng_field = FixedNormal(z)
        m._field(.7, None)
        sent.append(old[0]-m.charge[0])
    w = weights/np.sqrt(np.pi)
    mean = np.dot(w, sent)
    var = np.dot(w, (np.array(sent)-mean)**2)
    inv = 1/2+1/3
    assert abs(mean-(1/2+1/3)*(-np.expm1(-.7*g*inv))/inv) < 1e-15
    assert abs(var-(-np.expm1(-2*.7*g*inv))/inv) < 1e-15


def test_complete_program_and_three_rng_future_restore_without_source_files(tmp_path):
    a = material(0., assembly_rate=.1, bulk_attempt=1., write_rate=.1, recovery_rate=.1)
    for _ in range(3): a.step(.5, np.array([3., -3.]))
    save_exchange(tmp_path/'identity.npz', a, {'source_files_not_required_for_resume': True})
    b, provenance = load_exchange(tmp_path/'identity.npz')
    assert isinstance(b, IdentityMobileExchange)
    assert provenance['source_files_not_required_for_resume']
    for _ in range(5):
        a.step(.5, np.array([-3., 3.]))
        b.step(.5, np.array([-3., 3.]))
    for name in a._arrays+('pre', 'post'):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    for name in a._ledgers: assert getattr(a, name) == getattr(b, name)
    assert a.program.fingerprint() == b.program.fingerprint()
    assert a.rng_field.bit_generator.state == b.rng_field.bit_generator.state
    assert a.rng_motion.bit_generator.state == b.rng_motion.bit_generator.state
    assert a.rng_reaction.bit_generator.state == b.rng_reaction.bit_generator.state


def test_unknown_identity_and_changed_gene_fingerprint_are_rejected():
    m = material()
    with pytest.raises(ValueError):
        m.program.recognition_field(np.array([-1]), np.array([0]))
    state = m.state_dict()
    state['arrays']['gene_code'][0, 0] += .1
    with pytest.raises(ValueError, match='fingerprint'):
        IdentityMobileExchange.from_state_dict(state)
    with pytest.raises(ValueError):
        m.gene_code[0, 0] += .1


def test_disposable_type_pair_cache_is_bounded_and_never_changes_physical_state():
    p = program()
    arrays = {k: v.copy() for k, v in p.arrays.items()}
    arrays['code'] = np.tile(arrays['code'], (22, 1))[:65]
    arrays['strength_out'] = np.zeros(65)
    arrays['strength_in'] = np.zeros(65)
    p = IdentityMaterialProgram(arrays, [str(i) for i in range(65)])
    xyz = np.stack(np.meshgrid(*([np.arange(5)*.5+.5]*3), indexing='ij'), axis=-1).reshape(-1,3)[:65]
    m = IdentityMobileExchange(xyz, np.ones(65), np.ones(65), np.ones(65), np.ones(65),
        program=p, cell_type=np.arange(65), recognition_energy_fraction=0.)
    before = m.state_dict()
    original = m._pair_properties(0, 1)
    for i in range(65):
        for j in range(65): m._pair_properties(i, j)
    assert len(m._pair_cache) == 4096
    np.testing.assert_array_equal(m._pair_properties(0,1), original)
    after = m.state_dict()
    for name in m._arrays:
        np.testing.assert_array_equal(before['arrays'][name], after['arrays'][name])
    assert before['rng_reaction'] == after['rng_reaction']
    assert before['rng_motion'] == after['rng_motion']
    assert before['rng_field'] == after['rng_field']
