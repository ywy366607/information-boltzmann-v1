"""Numerical timing, continuation and learning-interface checks only."""
import copy
import gc
from dataclasses import fields

import pytest
import torch

from test_fly_bptt_learning import make_model, physical
from information_boltzmann.core.fly_bptt_learning import (
    advance_fly_input_event, FlyBPTTGraph, FlyPhysicalState, FlyBPTTLearner,
)
from information_boltzmann.core.fly_pipeline import (
    FlyPipelineLearner, begin_fly_prediction, commit_fly_observation,
    validate_pipeline_model, PROTOCOL,
)


def assert_state_equal(a, b, atol=0, rtol=0):
    for item in fields(a):
        va, vb = getattr(a, item.name), getattr(b, item.name)
        for ta, tb in zip(va if item.name == 'ring' else (va,),
                          vb if item.name == 'ring' else (vb,)):
            torch.testing.assert_close(ta, tb, atol=atol, rtol=rtol)


@pytest.mark.parametrize('center', [False, True])
def test_pipeline_matches_native_full_state_and_motor_read(tmp_path, center):
    torch.manual_seed(18)
    model = make_model(tmp_path, read_centering=center)
    validate_pipeline_model(model)
    state = physical(model)
    state.h = torch.linspace(-.03, .06, 6)[None]
    state.ring = tuple(torch.full_like(state.h, .2) for _ in range(4))
    for token_id in [1, 7, 2, 3, 4]:
        token = torch.tensor([token_id])
        logits, pending = begin_fly_prediction(model, state)
        native = advance_fly_input_event(model, state, token)
        result = commit_fly_observation(model, pending, token)
        assert_state_equal(native, result)
        source = native.h - native.h_mean if center else native.h
        torch.testing.assert_close(logits, model.read(source), atol=0, rtol=0)
        state = result


def test_current_observation_cannot_change_sealed_feature_but_next_tick_can(tmp_path):
    torch.manual_seed(5)
    model = make_model(tmp_path)
    old = physical(model)
    logits, pending = begin_fly_prediction(model, old)
    # Sensory neuron0 -> motor5 is a one-delay edge in this fixture.
    model.topographic_writer.proj_vis.weight.data[0].fill_(8.)
    model.embedding.weight.data[1].fill_(1.)
    model.embedding.weight.data[2].fill_(-1.)
    state_a = commit_fly_observation(model, pending, torch.tensor([1]))
    logits_b, pending_b = begin_fly_prediction(model, old)
    state_b = commit_fly_observation(model, pending_b, torch.tensor([2]))
    torch.testing.assert_close(logits, logits_b, atol=0, rtol=0)
    assert torch.equal(state_a.h[:, model.read_indices], state_b.h[:, model.read_indices])
    next_a, _ = begin_fly_prediction(model, state_a)
    next_b, _ = begin_fly_prediction(model, state_b)
    assert not torch.equal(next_a, next_b)
    grad = torch.autograd.grad(next_a.square().sum(),
                               model.topographic_writer.proj_vis.weight)[0]
    assert grad.abs().sum() > 0
    # The sealed prediction has no current-token writer dependency at all.
    current_grad = torch.autograd.grad(logits.sum(),
        model.topographic_writer.proj_vis.weight, allow_unused=True)[0]
    assert current_grad is None


def test_tick_commits_once_and_ring_is_old_until_commit(tmp_path):
    model = make_model(tmp_path)
    state = physical(model)
    old_ring = state.ring
    _, pending = begin_fly_prediction(model, state)
    assert state.ring is old_ring
    final = commit_fly_observation(model, pending, torch.tensor([1]))
    assert final.ring[1] is old_ring[0]
    assert final.ring[2] is old_ring[1]
    assert final.ring[3] is old_ring[2]
    with pytest.raises(RuntimeError, match='only once'):
        commit_fly_observation(model, pending, torch.tensor([2]))


def test_pipeline_refuses_silent_token_conditioned_predictor_hybrid(tmp_path):
    model = make_model(tmp_path)
    model.latent_predictor = torch.nn.Linear(3, 3)
    with pytest.raises(ValueError, match='token-conditioned'):
        FlyPipelineLearner(model, physical(model))


def test_fresh_decoder_clones_initialized_embedding_without_tying(tmp_path):
    torch.manual_seed(7)
    model = make_model(tmp_path)
    assert torch.equal(model.decoder.weight, model.embedding.weight)
    assert model.decoder.weight.data_ptr() != model.embedding.weight.data_ptr()
    assert model.decoder.weight.std() < .04
    with torch.no_grad():
        model.decoder.weight.add_(1.)
    saved = copy.deepcopy(model.state_dict())
    restored = make_model(tmp_path)
    restored.load_state_dict(saved)
    torch.testing.assert_close(restored.decoder.weight, model.decoder.weight, atol=0, rtol=0)


def test_pipeline_assimilates_every_target_once_and_continues_windows(tmp_path):
    torch.manual_seed(11)
    model = make_model(tmp_path)
    initial = physical(model)
    initial.ring = tuple(torch.full_like(initial.h, .2) for _ in range(4))
    learner = FlyPipelineLearner(model, initial)
    scores, metrics = learner.observe([1, 2, 3, 4, 5, 6, 7, 8])
    assert len(scores) == 8 and learner.physical_ticks == learner.events == 8
    assert learner.previous_token == 8 and learner.updates == 1
    assert metrics['writer_grad_norm'] > 0 and metrics['synapse_grad_norm'] > 0
    assert learner.state_dict()['prediction_protocol'] == PROTOCOL
    checkpoint = copy.deepcopy(learner.state_dict())
    restored = FlyPipelineLearner(copy.deepcopy(model), learner.state)
    restored.load_adam_state(checkpoint['optimizer'])
    restored.sgd.load_state_dict(checkpoint['sgd'])
    restored.load_edge_signs(checkpoint)
    for name in ('events', 'updates', 'physical_ticks', 'previous_token', 'ema'):
        setattr(restored, name, checkpoint[name])
    a, _ = learner.observe([0, 1, 2, 3])
    b, _ = restored.observe([0, 1, 2, 3])
    assert a == b
    assert_state_equal(learner.state, restored.state)
    assert learner.events == learner.physical_ticks == 12
    for pa, pb in zip(model.parameters(), restored.model.parameters()):
        torch.testing.assert_close(pa, pb, atol=0, rtol=0)


def test_windowed_decoder_is_identical_to_per_tick_sealed_logits(tmp_path):
    torch.manual_seed(8)
    model = make_model(tmp_path)
    initial = physical(model)
    initial.ring = tuple(torch.full_like(initial.h, .2) for _ in range(4))
    learner = FlyPipelineLearner(model, initial)
    tokens = torch.tensor([[1, 2, 3, 4]])
    actual, state, _ = learner.forward_window(tokens, tokens)
    reference, current = [], learner.state
    for token in tokens.unbind(1):
        logits, pending = begin_fly_prediction(model, current)
        reference.append(torch.nn.functional.cross_entropy(logits, token))
        current = commit_fly_observation(model, pending, token)
    torch.testing.assert_close(actual, torch.stack(reference), atol=1e-6, rtol=1e-6)
    assert_state_equal(state, current)


def test_pipeline_full_window_gradients_match_same_target_native_algebra(tmp_path):
    torch.manual_seed(15)
    model = make_model(tmp_path, read_centering=True)
    initial = physical(model)
    initial.h = torch.linspace(-.02, .03, 6)[None]
    initial.ring = tuple(torch.full_like(initial.h, .2) for _ in range(4))
    a = FlyPipelineLearner(model, initial)
    b = FlyBPTTLearner(copy.deepcopy(model), copy.deepcopy(initial))
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    sa, sta, _ = a.forward_window(tokens, tokens)
    sb, stb, _ = b.forward_window(tokens, tokens)
    torch.testing.assert_close(sa, sb, atol=0, rtol=0)
    assert_state_equal(sta, stb)
    sa.mean().backward()
    sb.mean().backward()
    named_b = dict(b.model.named_parameters())
    for name, pa in a.model.named_parameters():
        if pa.requires_grad:
            torch.testing.assert_close(pa.grad, named_b[name].grad, atol=1e-7, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_graph_pipeline_matches_eager_without_capture_updates(tmp_path):
    gc.collect()
    torch.manual_seed(8)
    model = make_model(tmp_path).cuda()
    cpu = physical(model)
    initial = FlyPhysicalState(**{k: tuple(t.cuda() for t in v) if k == 'ring'
        else v.cuda() for k, v in cpu.state_dict().items()})
    initial.ring = tuple(torch.full_like(initial.h, .2) for _ in range(4))
    a = FlyPipelineLearner(model, initial)
    b = FlyPipelineLearner(copy.deepcopy(model), copy.deepcopy(initial))
    b.runner = FlyBPTTGraph(b, window=8)
    assert b.events == b.updates == b.physical_ticks == 0
    assert_state_equal(a.state, b.state)
    for tokens in ([1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]):
        sa, _ = a.observe(tokens)
        sb, _ = b.observe(tokens)
        torch.testing.assert_close(torch.tensor(sa), torch.tensor(sb), atol=1e-6, rtol=1e-5)
        assert_state_equal(a.state, b.state, atol=1e-6, rtol=1e-5)
        for pa, pb in zip(a.model.parameters(), b.model.parameters()):
            torch.testing.assert_close(pa, pb, atol=1e-6, rtol=1e-5)
