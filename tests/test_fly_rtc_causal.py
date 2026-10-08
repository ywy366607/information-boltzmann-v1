"""Exact finite-horizon dependency selection and origin queue contracts."""
from dataclasses import fields

import pytest
import torch

from test_fly_rtc import individual
from information_boltzmann.core.fly_rtc_causal import CausalMotorForecaster
from information_boltzmann.core.fly_rtc_learning import (
    copy_physical, physical_options, quiet_teacher_step,
)


@pytest.mark.parametrize('horizon', [1, 2, 4, 7, 14])
@pytest.mark.parametrize('fused', [False, True])
def test_causal_domain_matches_every_motor_state_field(individual, horizon, fused):
    model, original = individual
    predictor = CausalMotorForecaster(model, horizon=horizon, fused_transmission=fused)
    with torch.no_grad():
        prediction = predictor.rollout(original, predictor.coefficients(model))
        actual = copy_physical(original)
        for tick in range(1, horizon+1):
            actual = quiet_teacher_step(model, actual, physical_options(model))
            for field in fields(prediction[tick]):
                if field.name == 'ring':
                    for a,b in zip(prediction[tick].ring, actual.ring):
                        torch.testing.assert_close(a, b[:, model.read_indices], rtol=1e-5, atol=2e-6)
                else:
                    name = 'h_mean' if field.name == 'mean' else field.name
                    torch.testing.assert_close(getattr(prediction[tick], field.name),
                        getattr(actual, name)[:, model.read_indices], rtol=1e-5, atol=2e-6)
        assert torch.equal(original.h, individual[1].h)
        assert not list(predictor.parameters())


def test_origins_outside_domain_preserve_pending_intermediate_arrivals(individual):
    model, original = individual
    predictor = CausalMotorForecaster(model, horizon=4)
    # Edge2->1 has delay3; 1->0->motor5 costs2+1. Node2 is
    # outside the horizon4 domain, but its old ring pulse can arrive at
    # intermediate1 on the next tick and affect motor5 by tick4.
    assert predictor.distances[2] == 6
    assert predictor.inverse[2] == -1 and predictor.inverse[1] >= 0
    physical = copy_physical(original)
    physical.ring[2][:, 2] = .9
    queue = predictor.known_queue(physical)
    assert queue[0, 0, predictor.inverse[1], 0] > 0
    forecast = predictor.rollout(physical, predictor.coefficients(model))
    actual = copy_physical(physical)
    with torch.no_grad():
        for _ in range(4):
            actual = quiet_teacher_step(model, actual, physical_options(model))
    torch.testing.assert_close(forecast[4].h, actual.h[:, model.read_indices], rtol=1e-5, atol=2e-6)


def test_horizon_guard_and_nested_anatomical_domains(individual):
    model, physical = individual
    small = CausalMotorForecaster(model, horizon=1)
    large = CausalMotorForecaster(model, horizon=14)
    assert set(small.indices.tolist()) <= set(large.indices.tolist())
    assert small.budget()['retained_neurons'] < model.n_neurons
    with pytest.raises(ValueError, match='compiled'):
        small.rollout(physical, small.coefficients(model), horizon=2)
    with pytest.raises(ValueError, match='Positive'):
        CausalMotorForecaster(model, horizon=0)
