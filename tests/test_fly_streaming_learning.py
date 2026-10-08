"""Deterministic streaming interfaces, not language-capability experiments."""
import copy

import numpy as np
import pytest
import torch

from information_boltzmann.core.fly_reservoir import (
    FlyReservoirLM, BiologicalTopographicWriter,
)
from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState
from information_boltzmann.core.fly_streaming_learning import (
    FlyStreamingLearner, predict_streaming_next,
)


def individual(tmp_path):
    from information_boltzmann.core.fly_streaming_observer import StreamingGraphObserver
    graph = tmp_path / 'graph.npz'
    np.savez(graph, neuron_body_ids=np.arange(6), edge_pre=np.arange(6),
        edge_post=np.roll(np.arange(6), 1), edge_weight=np.full(6, .1, dtype=np.float32),
        nt_sign=np.array([1, 1, 1, -1, -1, -1]),
        edge_delay=np.array([1, 2, 3, 4, 2, 1], dtype=np.int32),
        superclass_id=np.array([0, 0, 0, 1, 1, 1]),
        superclass_names=np.array(['cb_sensory', 'cb_motor']))
    partitions = tmp_path / 'parts.npz'
    np.savez(partitions, visual_idx=np.array([0]), chemo_idx=np.array([1]),
             mechano_idx=np.array([2]))
    model = FlyReservoirLM(graph, vocab_size=9, d_model=3, injection='sensory',
        read_surface='output', synapse_model='coba', use_alif=True, use_stp=True)
    model.topographic_writer = BiologicalTopographicWriter(3, partitions)
    model.injection_mode, model.input_proj = 'topographic', None
    model.streaming_observer = StreamingGraphObserver.from_model(
        model, latent_dim=4, sample_per_region=2, seed=11)
    h = torch.linspace(.01, .05, 6)[None]
    state = FlyPhysicalState(h, tuple(torch.full_like(h, .5) for _ in range(4)),
        torch.zeros_like(h), torch.zeros_like(h), torch.zeros_like(h),
        torch.ones_like(h), model.get_stp_params()[0].detach().expand_as(h).clone(),
        model.topographic_writer.a_adapt.clone(), torch.zeros_like(h))
    return model, state


def test_streaming_inference_matches_training_window_and_carries_tick(tmp_path):
    torch.manual_seed(11)
    model, state = individual(tmp_path)
    inference_model = copy.deepcopy(model)
    learner = FlyStreamingLearner(model, state)
    inference_physical = copy.deepcopy(state)
    inference_observer = inference_model.streaming_observer.initial_state(1, 'cpu', state.h.dtype)
    ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    targets = torch.tensor([[1, 2, 3, 4, 5, 6]])
    inference_logits = []
    for token in ids.unbind(1):
        logits, inference_physical, inference_observer = predict_streaming_next(
            inference_model, inference_physical, inference_observer, token)
        inference_logits.append(logits)
    expected = torch.nn.functional.cross_entropy(
        torch.cat(inference_logits), targets.flatten(), reduction='none')
    scores, physical, _ = learner.forward_window(ids, targets)
    torch.testing.assert_close(scores, expected)
    torch.testing.assert_close(physical.h, inference_physical.h)
    for name, value in learner.observer_state.state_dict().items():
        expected_value = inference_observer.state_dict()[name]
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, expected_value)
        else:
            assert value == expected_value


def test_streaming_observer_survives_updates_and_full_resume(tmp_path):
    torch.manual_seed(11)
    model, physical = individual(tmp_path)
    learner = FlyStreamingLearner(model, physical)
    learner.previous_token = 0
    old_adapter = [p.detach().clone() for p in model.streaming_observer.parameters()]
    learner.observe([1, 2, 3, 4, 5, 6, 7, 8])
    assert learner.events == learner.physical_ticks == 8
    assert learner.updates == 1
    assert not learner.state.h.requires_grad
    assert any(not torch.equal(before, after) for before, after
               in zip(old_adapter, model.streaming_observer.parameters()))
    saved = copy.deepcopy(learner.state_dict())
    resumed_model = copy.deepcopy(model)
    resumed = FlyStreamingLearner(resumed_model, copy.deepcopy(learner.state))
    resumed.restore_learning_state(saved)
    direct_scores, _ = learner.observe([0, 1, 2, 3])
    resumed_scores, _ = resumed.observe([0, 1, 2, 3])
    torch.testing.assert_close(torch.tensor(direct_scores), torch.tensor(resumed_scores))
    torch.testing.assert_close(learner.state.h, resumed.state.h)
    for name, parameter in learner.model.named_parameters():
        torch.testing.assert_close(parameter, dict(resumed.model.named_parameters())[name])


def test_no_implicit_observer_reset_or_extra_physical_ticks(tmp_path):
    model, state = individual(tmp_path)
    with pytest.raises(ValueError, match='continuing observer'):
        predict_streaming_next(model, state, None, torch.tensor([0]))
    with pytest.raises(ValueError, match='one physical tick'):
        FlyStreamingLearner(model, state, settle_ticks=1)
    model.use_read_gamma_trace = True
    with pytest.raises(ValueError, match='Gamma'):
        FlyStreamingLearner(model, state)


def test_cli_budget_and_calibration_are_explicit():
    from scripts.ib.train_fly_streaming_observer import parse_args
    args = parse_args([])
    assert args.window == 32 and args.additional_tokens == 100000
    with pytest.raises(SystemExit):
        parse_args(['--additional-tokens', '32'])
    args = parse_args(['--calibrate', '--additional-tokens', '32'])
    assert args.calibrate
