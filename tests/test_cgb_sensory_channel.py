"""Numerical identities for the existing finite-temperature field solver."""
import numpy as np
import pytest

from information_boltzmann.cgb.mobile_exchange import MobileExchange
from information_boltzmann.cgb.sensory_channel import (
    field_channel, frozen_field_channel, byte_posterior, uniform_information_bound,
    source_innovation, source_path_posterior, uniform_binary_information)
from information_boltzmann.cgb.sensory_channel import exact_field_reference
from scripts.ib.generative.compare_support_transport import same


def panel():
    xyz = np.stack(np.meshgrid(*([np.array([1., 1.8])]*3), indexing='ij'), axis=-1).reshape(-1, 3)
    return MobileExchange(xyz, np.full(8, 3), np.full(8, 2), np.full(8, 4), np.full(8, 2),
        pre=[0, 1, 2, 3], post=[1, 3, 6, 7], bound=[0, 1, 0, 1], trace=[1, 0, 1, 0],
        charge=np.linspace(-.3, .3, 8), capacitance=np.linspace(.9, 1.2, 8),
        world_charge=np.linspace(-.1, .2, 8), world_capacitance=np.linspace(1.8, 2.2, 8),
        port_nodes=np.arange(8), seed=94)


class FixedNoise:
    def __init__(self, values):
        self.values = iter(values)

    def normal(self, size=None):
        if size is None:
            return next(self.values)
        return np.array([next(self.values) for _ in range(size)])


def run_field(state, source, noise):
    model = MobileExchange.from_state_dict(state)
    model.rng_field = FixedNoise(noise)
    model._field(2., source)
    return model.charge.copy()


def test_reverse_rows_equal_actual_ordered_ou_means_and_noise_covariance():
    model = panel()
    state = model.state_dict()
    channel = field_channel(model, 2.)
    count = 2*len(model.port_nodes)+len(model.pre)
    zero = np.zeros(count)
    mean = run_field(state, np.zeros(8), zero)
    np.testing.assert_allclose(mean, channel['offset'], atol=2e-16)
    for port in range(8):
        source = np.eye(8)[port]
        np.testing.assert_allclose(run_field(state, source, zero)-mean,
                                   channel['gain'][:, port], atol=2e-16)
    impulses = []
    for index in range(count):
        noise = np.zeros(count)
        noise[index] = 1.
        impulses.append(run_field(state, np.zeros(8), noise)-mean)
    factor = np.asarray(impulses).T
    np.testing.assert_allclose(factor@factor.T, channel['covariance'], atol=2e-16)
    same(model.state_dict(), state)
    assert channel['rhs_rows'] == 8
    assert channel['workspace_numeric_bytes'] == 8*(8*(model.n+8)+64+64)


def test_posterior_is_normalized_and_uniform_information_bound_obeys_capacity():
    model = panel()
    channel = field_channel(model, 2.)
    logp = byte_posterior(channel, model.charge[model.port_nodes])
    assert abs(np.exp(logp).sum()-1.) < 1e-14
    bound = uniform_information_bound(channel)
    assert 0 < bound < 8*np.log(2.)
    prior = np.arange(1., 257.)
    prior /= prior.sum()
    logp = byte_posterior(channel, model.charge[model.port_nodes], prior=prior)
    assert abs(np.exp(logp).sum()-1.) < 1e-14


def test_zero_source_coupling_removes_new_information_but_not_field_noise():
    model = panel()
    model.source_conductance[:] = 0.
    channel = field_channel(model, 2.)
    assert np.count_nonzero(channel['gain']) == 0
    assert np.trace(channel['covariance']) > 0
    assert uniform_information_bound(channel) == 0.
    prior = np.arange(1., 257.)
    prior /= prior.sum()
    np.testing.assert_allclose(byte_posterior(channel, model.charge, prior=prior), np.log(prior), atol=1e-14)


def test_degenerate_or_invalid_channels_fail_without_hidden_noise_floor():
    model = panel()
    with pytest.raises(ValueError):
        field_channel(model, 0.)
    with pytest.raises(ValueError):
        field_channel(model, 2., [0, 0])
    channel = field_channel(model, 2.)
    with pytest.raises(ValueError):
        byte_posterior(channel, np.zeros(8), prior=np.ones(256))
    channel['covariance'][:] = 0.
    with pytest.raises(np.linalg.LinAlgError):
        byte_posterior(channel, np.zeros(8))


def test_two_field_steps_and_joint_world_observation_match_all_noise_bases():
    model = panel()
    channel = frozen_field_channel(model, [1., 2.], np.arange(8), np.arange(8))
    state = model.state_dict()
    draws = 2*(16+len(model.pre))

    def run(source, noise):
        candidate = MobileExchange.from_state_dict(state)
        candidate.rng_field = FixedNoise(noise)
        candidate._field(1., source)
        candidate._field(2., source)
        return np.r_[candidate.charge, candidate.world_charge]

    mean = run(np.zeros(8), np.zeros(draws))
    np.testing.assert_allclose(channel['offset'], mean, atol=2e-16)
    for bit in range(8):
        np.testing.assert_allclose(run(np.eye(8)[bit], np.zeros(draws))-mean,
                                   channel['gain'][:, bit], atol=3e-16)
    noise_basis = []
    for i in range(draws):
        noise = np.zeros(draws)
        noise[i] = 1.
        noise_basis.append(run(np.zeros(8), noise)-mean)
    matrix = np.array(noise_basis).T
    np.testing.assert_allclose(matrix@matrix.T, channel['covariance'], atol=3e-16)


def test_source_instrument_and_mutated_record_leave_full_adaptive_future_exact():
    model = panel()
    sham = MobileExchange.from_state_dict(model.state_dict())
    innovations = []

    def read(record):
        innovations.append(source_innovation(model, record))
        record['after_charge'][:] = 999.  # observer receives copies

    model.source_observer = read
    for _ in range(2):
        model.step(2., np.array([3., -3.]*4))
        sham.step(2., np.array([3., -3.]*4))
    same(model.state_dict(), sham.state_dict())
    restored = MobileExchange.from_state_dict(model.state_dict())
    assert restored.source_observer is None
    assert len(innovations) == 2
    posterior = source_path_posterior(innovations)
    assert abs(np.exp(posterior['logposterior']).sum()-1.) < 1e-14
    # Check against direct likelihood multiplication over all 256 hypotheses.
    signs = 2*((np.arange(256)[:, None] >> np.arange(8)) & 1)-1
    score = np.zeros(256)
    for row in innovations:
        score -= .5*np.sum((row['observed']-3*signs*row['gain'])**2/row['variance'], axis=1)
    from scipy.special import logsumexp
    np.testing.assert_allclose(posterior['logposterior'], score-logsumexp(score), atol=4e-14)


def test_source_information_is_finite_noise_and_clock_derived():
    model = panel()
    record = dict(dt_ms=2., before_charge=model.world_charge.copy(), after_charge=model.world_charge.copy())
    innovation = source_innovation(model, record)
    result = source_path_posterior([innovation, innovation])
    rate = 2*model.source_conductance/model.world_capacitance
    expected = 2*9*model.world_capacitance*np.tanh(rate/2)
    np.testing.assert_allclose(result['snr'], expected, atol=5e-16)
    info = uniform_binary_information(expected)
    np.testing.assert_allclose(info, uniform_binary_information(expected, 64), atol=2e-8)
    assert np.all(info > 0) and np.all(info <= np.log(2.))
    assert np.all(info <= .5*np.log1p(expected))
    np.testing.assert_allclose(uniform_binary_information(np.zeros(8)), 0., atol=2e-16)
    zero = dict(observed=np.zeros(8), gain=np.zeros(8), variance=np.zeros(8))
    np.testing.assert_allclose(source_path_posterior([zero])['logposterior'], -8*np.log(2.), atol=1e-15)
    with pytest.raises(ValueError):
        source_path_posterior([])


def test_exact_small_reference_obeys_semigroup_and_thermal_covariance():
    model = panel()
    nodes, world = np.arange(8), np.arange(8)
    one = exact_field_reference(model, 1., nodes, world)
    two = exact_field_reference(model, 2., nodes, world)
    three = exact_field_reference(model, 3., nodes, world)
    a = two['state_transition']
    np.testing.assert_allclose(a@one['state_transition'], three['state_transition'], atol=3e-15)
    np.testing.assert_allclose(a@one['full_gain']+two['full_gain'], three['full_gain'], atol=2e-15)
    np.testing.assert_allclose(a@one['full_covariance']@a.T+two['full_covariance'],
                              three['full_covariance'], atol=4e-15)
    equilibrium = np.diag(np.r_[model.capacitance, model.world_capacitance])
    np.testing.assert_allclose(a@equilibrium@a.T+two['full_covariance'], equilibrium, atol=5e-15)
    coarse = frozen_field_channel(model, [3.], nodes, world)
    fine = frozen_field_channel(model, [.03]*100, nodes, world)
    assert np.linalg.norm(fine['gain']-three['gain']) < .02*np.linalg.norm(coarse['gain']-three['gain'])


def test_symmetric_solver_has_exact_conditional_noise_and_pays_each_half_step():
    model = panel()
    model.field_scheme = 'symmetric_ou'
    state = model.state_dict()
    assert state['schema'] == 3
    channel = frozen_field_channel(model, [2.], np.arange(8), np.arange(8))
    draws = 2*(16+len(model.pre))

    def run(source, noise):
        candidate = MobileExchange.from_state_dict(state)
        candidate.rng_field = FixedNoise(noise)
        candidate._field(2., source)
        for error in candidate.balances().values():
            assert abs(float(error)) < 2e-13
        return np.r_[candidate.charge, candidate.world_charge]

    zero = np.zeros(draws)
    mean = run(np.zeros(8), zero)
    np.testing.assert_allclose(channel['offset'], mean, atol=3e-16)
    for port in range(8):
        np.testing.assert_allclose(run(np.eye(8)[port], zero)-mean, channel['gain'][:, port], atol=3e-16)
    noise_basis = []
    for index in range(draws):
        noise = zero.copy()
        noise[index] = 1.
        noise_basis.append(run(np.zeros(8), noise)-mean)
    factor = np.array(noise_basis).T
    np.testing.assert_allclose(factor@factor.T, channel['covariance'], atol=4e-16)
    source_events = []
    model.source_observer = source_events.append
    model.step(2., np.array([3.,-3.]*4))
    assert model.t == 2.
    assert len(source_events) == 2 and sum(e['dt_ms'] for e in source_events) == 2.
    exact = exact_field_reference(model, 4., np.arange(8), np.arange(8))
    symmetric = frozen_field_channel(model, [2., 2.], np.arange(8), np.arange(8))
    model.field_scheme = 'sequential_ou'
    sequential = frozen_field_channel(model, [2., 2.], np.arange(8), np.arange(8))
    assert np.linalg.norm(symmetric['gain']-exact['gain']) < .05*np.linalg.norm(sequential['gain']-exact['gain'])
