"""Whole-generator execution, not a synthetic learning-capability study."""
import copy
import itertools

import numpy as np
import pytest

from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.growing_material import GrowingIdentityMobileExchange
from information_boltzmann.cgb.material_fork import fork_material
from information_boltzmann.cgb.mobile_exchange import MobileExchange
from information_boltzmann.cgb.sensory_channel import frozen_field_channel
from scripts.ib.generative.compare_support_transport import same
from test_cgb_growing_material import growing


def symmetric():
    m = growing()
    m.domain_sampling = 'marginal'
    m.material_scheme = 'symmetric'
    return m


def test_material_palindrome_preserves_each_process_elapsed_time_and_source_intervals():
    m = symmetric()
    calls = []
    for name in ('_motion', '_reactions', '_field'):
        original = getattr(m, name)
        def record(dt, *args, name=name, original=original):
            calls.append((name, dt, getattr(m, '_reaction_clock_offset', None)))
            original(dt, *args)
        setattr(m, name, record)
    sources = []
    m.source_observer = sources.append
    m.step(2., np.array([3.]))
    assert [(name, dt) for name, dt, _ in calls] == [
        ('_motion', 1.), ('_reactions', 1.), ('_field', 2.), ('_reactions', 1.), ('_motion', 1.)]
    assert calls[1][2] == 0. and calls[3][2] == 1.
    assert [x['time_ms'] for x in sources] == [0., 1.]
    assert [x['dt_ms'] for x in sources] == [1., 1.]
    assert m.t == 2.
    assert max(abs(float(x)) for x in m.balances().values()) < 1e-11


@pytest.mark.parametrize('domain_enabled', [False, True])
def test_reaction_receipts_use_the_cumulative_reaction_clock_in_both_halves(domain_enabled):
    m = symmetric()
    m.domain_reactions_enabled = domain_enabled
    rows = []
    origin = []
    m.event_observer = lambda row: rows.append((origin[0], row))
    m.domain_observer = lambda row: rows.append((origin[0], row))
    original = m._reactions
    def reaction(dt):
        origin[:] = [m.t+m._reaction_clock_offset]
        original(dt)
    m._reactions = reaction
    for _ in range(10):
        m.step(10., np.array([3.]))
    assert rows
    assert {int(a/5.) % 2 for a, _ in rows} == {0, 1}
    for start, row in rows:
        assert start <= row['time_ms'] <= start+5.
    assert [row['time_ms'] for _, row in rows] == sorted(row['time_ms'] for _, row in rows)


@pytest.mark.parametrize('sampling', ['resolved', 'marginal'])
def test_new_schema_and_three_rng_full_continuation_are_exact(tmp_path, sampling):
    m = symmetric()
    m.domain_sampling = sampling
    for _ in range(4):
        m.step(1., np.array([3.]))
    state = m.state_dict()
    assert state['schema'] == 3 and state['domain_sampling'] == sampling
    path = tmp_path/'complete.npz'
    save_exchange(path, m, {'real_clock': m.t})
    restored, meta = load_exchange(path)
    fork = fork_material(m)
    assert meta['real_clock'] == m.t
    for _ in range(12):
        for candidate in (m, restored, fork):
            candidate.step(.5, np.array([-3.]))
    same(m.state_dict(), restored.state_dict())
    same(m.state_dict(), fork.state_dict())
    assert max(abs(float(x)) for x in m.balances().values()) < 1e-10


def test_partial_failed_step_cannot_be_forked_archived_or_restarted(tmp_path):
    m = symmetric()
    def interrupt(dt, voltage):
        with pytest.raises(ValueError, match='complete'):
            fork_material(m)
        with pytest.raises(ValueError, match='complete'):
            save_exchange(tmp_path/'partial.npz', m)
        raise RuntimeError('interrupted at an internal operator boundary')
    m._field = interrupt
    with pytest.raises(RuntimeError):
        m.step(1., np.array([3.]))
    with pytest.raises(ValueError, match='partial|failed'):
        m.step(1., np.array([3.]))
    with pytest.raises(ValueError, match='complete'):
        m.state_dict()
    assert not (tmp_path/'partial.npz').exists()


def test_modes_invalid_calls_and_unconditioned_gaussian_instrument_fail_before_mutation():
    m = symmetric()
    before = m.state_dict()
    for dt, source in [(0., None), (1., [float('nan')]), (1., [0., 0.])]:
        with pytest.raises(ValueError):
            m.step(dt, source)
        same(m.state_dict(), before)
    with pytest.raises(ValueError, match='mixture'):
        frozen_field_channel(m, [1.])
    same(m.state_dict(), before)
    for key, value in [('schema', 2), ('field_scheme', 'sequential_ou'),
                       ('material_scheme', 'sequential'), ('domain_sampling', 'missing')]:
        state = copy.deepcopy(before)
        state[key] = value
        with pytest.raises(ValueError):
            GrowingIdentityMobileExchange.from_state_dict(state)


def test_default_archive_and_legacy_execution_remain_explicit_and_unchanged():
    m = growing()
    state = m.state_dict()
    assert state['schema'] == 1 and 'material_scheme' not in state
    restored = GrowingIdentityMobileExchange.from_state_dict(state)
    assert restored.material_scheme == 'sequential'
    m.step(1., [3.]);restored.step(1., [3.])
    same(m.state_dict(), restored.state_dict())


def field_fixture():
    m = MobileExchange([[1., 1., 1.], [1.8, 1., 1.], [2.6, 1., 1.]],
        [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2],
        pre=[0, 1], post=[1, 2], trace=[0, 1], capacitance=[1., 2., 3.],
        port_nodes=[0, 2], world_capacitance=[2., 4.], port_conductance=[.3, .2],
        source_conductance=[.5, .4], unit_conductance=.3)
    m.field_scheme = 'symmetric_ou'
    m.material_scheme = 'symmetric'
    return m


class ConditionalNoise:
    def __init__(self, order, noise):
        self.order = np.asarray(order)
        self.noise = iter(noise)
        self.permutations = 0
    def permutation(self, length):
        assert length == len(self.order)
        self.permutations += 1
        return self.order
    def normal(self, size=None):
        return next(self.noise) if size is None else np.array([next(self.noise) for _ in range(size)])


def exact_conditional_channel(storage, order):
    # Impulses through the ACTUAL floating point OU implementation, not a
    # separately coded transition matrix or a Monte Carlo tolerance.
    draws = 12  # two sources (2 ports), two port halves, two edge halves
    def run(charge, noise):
        m = field_fixture()
        for name in ('pre', 'post', 'bound', 'trace', 'capacity'):
            setattr(m, name, getattr(m, name)[storage].copy())
        m.charge[:], m.world_charge[:] = charge[:3], charge[3:]
        rng = ConditionalNoise(order, noise)
        m.rng_field = rng
        m._field(2., np.zeros(2))
        assert rng.permutations == 1
        return np.r_[m.charge, m.world_charge]
    offset = run(np.zeros(5), np.zeros(draws))
    transition = np.array([run(x, np.zeros(draws))-offset for x in np.eye(5)]).T
    factor = np.array([run(np.zeros(5), x)-offset for x in np.eye(draws)]).T
    return transition, factor@factor.T


def test_each_random_edge_order_is_reciprocal_and_mixture_ignores_storage_labels():
    cov = np.diag([1., 2., 3., 2., 4.])
    kernels = {}
    for order in itertools.permutations(range(2)):
        a, noise = exact_conditional_channel(np.array([0, 1]), order)
        np.testing.assert_allclose(a@cov, cov@a.T, atol=3e-16)
        np.testing.assert_allclose(a@cov@a.T+noise, cov, atol=1e-15)
        kernels[order] = (a, noise)
    # Swapping storage swaps the two equally weighted mixture components.
    # Equality of moments alone would not prove equality of mixtures.
    for order in itertools.permutations(range(2)):
        a, noise = exact_conditional_channel(np.array([1, 0]), order)
        expected = kernels[tuple(1-i for i in order)]
        np.testing.assert_array_equal(a, expected[0])
        np.testing.assert_array_equal(noise, expected[1])
    assert not np.array_equal(kernels[(0, 1)][0], kernels[(1, 0)][0])
