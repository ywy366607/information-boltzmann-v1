"""Numerical identities for the existing finite-temperature field solver."""
import numpy as np
import pytest

from information_boltzmann.cgb.mobile_exchange import MobileExchange
from information_boltzmann.cgb.sensory_channel import (
    field_channel, byte_posterior, uniform_information_bound)
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
