"""Differentiable quiet branch, persistent clock and continuation contracts."""
import copy
from dataclasses import fields

import torch

from test_fly_rtc import individual
from information_boltzmann.core.fly_causal_rtc_learning import (
    CausalHorizonReadout, FlyCausalRTCLearner, exact_response_features,
)
from information_boltzmann.core.fly_rtc_causal import CausalMotorForecaster
from information_boltzmann.core.fly_rtc_learning import copy_physical, physical_options
from information_boltzmann.core.fly_rtc_surface import MotorSurfaceForecaster


def setup_model(model):
    del model.rtc_student
    model.causal_read = CausalHorizonReadout(3, horizon=14, key_dim=4)
    return model


def compare_states(a, b):
    for field in fields(a):
        x, y = getattr(a,field.name), getattr(b,field.name)
        if field.name == 'ring':
            for i,j in zip(x,y):
                torch.testing.assert_close(i,j)
        else:
            torch.testing.assert_close(x,y)


def test_differentiable_features_match_independent_causal_oracle(individual):
    model, state = individual
    setup_model(model)
    predictor = CausalMotorForecaster(model, horizon=14)
    origin = copy_physical(state)
    features = exact_response_features(model, state, physical_options(model), 14,
                                       checkpoint_branch=False)
    with torch.no_grad():
        states = predictor.rollout(state, predictor.coefficients(model))
        expected = torch.stack([MotorSurfaceForecaster.decode(model,s.h,s.mean) for s in states])
    torch.testing.assert_close(features, expected, rtol=1e-5, atol=2e-6)
    compare_states(state, origin)


def test_checkpoint_forward_and_joint_gradients_match(individual):
    model, state = individual
    setup_model(model)
    other = copy.deepcopy(model)
    plain = FlyCausalRTCLearner(model, copy_physical(state), horizon=4, checkpoint_branch=False)
    saved = FlyCausalRTCLearner(other, copy_physical(state), horizon=4, checkpoint_branch=True)
    ids, targets = torch.tensor([[1,2,3,4]]), torch.tensor([[2,3,4,5]])
    first = plain.forward_window(ids, targets)
    second = saved.forward_window(ids, targets)
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[2], second[2])
    compare_states(first[1], second[1])
    first[0].mean().backward()
    second[0].mean().backward()
    for (n,p),(m,q) in zip(model.named_parameters(), other.named_parameters()):
        assert n == m
        if p.requires_grad:
            assert p.grad is not None and q.grad is not None
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=2e-6)
    assert model.edge_weight_e.grad.abs().sum() > 0
    assert model.topographic_writer.proj_vis.weight.grad.abs().sum() > 0
    assert model.causal_read.horizon_bias.grad.abs().sum() > 0


def test_actual_clock_excludes_branch_ticks_and_resume_is_complete(individual):
    model, state = individual
    setup_model(model)
    learner = FlyCausalRTCLearner(model, state, horizon=4)
    learner.previous_token = 0
    scores, metrics = learner.observe([1,2,3,4])
    assert len(scores) == learner.events == learner.physical_ticks == 4
    assert learner.updates == 1 and learner.speculative_ticks == 16
    assert metrics['causal_speculative_ticks'] == 16
    saved = copy.deepcopy(learner.state_dict())
    resumed = FlyCausalRTCLearner(copy.deepcopy(model),copy_physical(learner.state),horizon=4)
    resumed.restore_learning_state(saved)
    a,_ = learner.observe([5,6])
    b,_ = resumed.observe([5,6])
    torch.testing.assert_close(torch.tensor(a),torch.tensor(b))
    compare_states(learner.state, resumed.state)
    assert resumed.speculative_ticks == learner.speculative_ticks == 24


def test_future_labels_do_not_enter_response_or_actual_body(individual):
    model, state = individual
    setup_model(model)
    learner = FlyCausalRTCLearner(model, state, horizon=4)
    ids = torch.tensor([[1,2]])
    a = learner.forward_window(ids, torch.tensor([[2,3]]))
    b = learner.forward_window(ids, torch.tensor([[7,8]]))
    torch.testing.assert_close(a[2],b[2])
    compare_states(a[1],b[1])
