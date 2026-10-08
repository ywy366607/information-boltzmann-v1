"""Known-delay numerical/interface checks; no learned task or capacity claim."""
import numpy as np
import pytest
import torch

from information_boltzmann.core.fly_reservoir import (
    BiologicalTopographicWriter, FlyReservoirLM,
)
from information_boltzmann.core.fly_bptt_learning import advance_fly_input_event
from information_boltzmann.core.fly_pipeline import (
    VERIFIED_PATH_HORIZON, begin_fly_prediction, commit_fly_observation,
    validate_pipeline_model, FlyPipelineLearner,
)
from test_fly_bptt_learning import physical
from test_fly_pipeline import assert_state_equal


def make_delay_paths(tmp_path, path_delays):
    """Disjoint paths share one sensory root; each motor has a known arrival.

    Weights and thresholds expose propagation without motor reset hiding the
    first arrival. These explicit fixture values are not production settings.
    """
    pre, post, delays, endpoints = [], [], [], []
    classes = [0, 0, 0]  # Three separate sensory partitions required by writer.
    for path in path_delays:
        previous = 0
        for delay in path:
            current = len(classes)
            pre.append(previous)
            post.append(current)
            delays.append(delay)
            classes.append(1)
            previous = current
        endpoints.append(previous)
        classes[previous] = 2
    graph = tmp_path / 'known_delay_graph.npz'
    np.savez(graph, neuron_body_ids=np.arange(len(classes)),
             edge_pre=np.array(pre), edge_post=np.array(post),
             edge_weight=np.full(len(pre), 4., dtype=np.float32),
             nt_sign=np.ones(len(classes)), edge_delay=np.array(delays),
             superclass_id=np.array(classes),
             superclass_names=np.array(['cb_sensory', 'cb_interneuron', 'cb_motor']))
    partitions = tmp_path / 'partitions.npz'
    np.savez(partitions, visual_idx=np.array([0]), chemo_idx=np.array([1]),
             mechano_idx=np.array([2]))
    width = len(endpoints)
    model = FlyReservoirLM(graph, vocab_size=3, d_model=width,
                           injection='sensory', read_surface='output',
                           synapse_model='coba', use_alif=True, use_stp=True)
    model.topographic_writer = BiologicalTopographicWriter(width, partitions)
    model.injection_mode, model.input_proj = 'topographic', None
    with torch.no_grad():
        model.embedding.weight.zero_()
        model.embedding.weight[1].fill_(1.)
        model.topographic_writer.gate_linear.weight.zero_()
        model.topographic_writer.gate_linear.bias.zero_()
        model.topographic_writer.proj_vis.weight.fill_(1. / width)
        model.topographic_writer.proj_chemo.weight.zero_()
        model.topographic_writer.proj_mech.weight.zero_()
        model.log_threshold[:] = torch.tensor([.1, .1, 1.5]).log()
        model.log_tau_s_e.fill_(torch.tensor(.5).log())
        model.output_read.weight.copy_(torch.eye(width))
    validate_pipeline_model(model)
    return model


def unfold(model, inputs, state=None, detach_at=None):
    state = physical(model) if state is None else state
    features = []
    for tick, token in enumerate(inputs):
        if tick == detach_at:
            state = state.detached()
        feature, pending = begin_fly_prediction(model, state, decode=False)
        features.append(feature)
        state = commit_fly_observation(model, pending, torch.tensor([token]))
    return torch.cat(features), state


def test_all_first_arrivals_one_through_fourteen_and_native_state_equivalence(tmp_path):
    paths = [[1] * d for d in range(1, VERIFIED_PATH_HORIZON + 1)]
    model = make_delay_paths(tmp_path, paths)
    signal, control = physical(model), physical(model)
    native = physical(model)
    first_arrivals = [None] * len(paths)
    for tick in range(VERIFIED_PATH_HORIZON + 1):
        feature, pending = begin_fly_prediction(model, signal, decode=False)
        reference, quiet = begin_fly_prediction(model, control, decode=False)
        delta = (feature - reference).abs().flatten()
        for i, path in enumerate(paths):
            if tick < sum(path):
                assert delta[i].item() == 0., (tick, i, delta[i].item())
            if delta[i] > 1e-7 and first_arrivals[i] is None:
                first_arrivals[i] = tick
        token = torch.tensor([1 if tick == 0 else 0])
        signal = commit_fly_observation(model, pending, token)
        control = commit_fly_observation(model, quiet, torch.tensor([0]))
        native = advance_fly_input_event(model, native, token)
        assert_state_equal(signal, native)
    assert first_arrivals == list(range(1, VERIFIED_PATH_HORIZON + 1))


@pytest.mark.parametrize('path', [[4, 4, 4, 2], [1] * 14])
def test_fourteen_tick_credit_reaches_writer_and_every_edge_inside_window(tmp_path, path):
    model = make_delay_paths(tmp_path, [path])
    model.edge_weight_e.requires_grad_(True)
    feature, _ = unfold(model, [1] + [0] * 14)
    source = model.topographic_writer.proj_vis.weight
    writer_grad, edge_grad = torch.autograd.grad(feature[14].sum(),
                                               (source, model.edge_weight_e))
    assert torch.isfinite(writer_grad).all() and writer_grad.abs().sum() > 0
    assert torch.isfinite(edge_grad).all() and torch.all(edge_grad.abs() > 0)


def test_window_detach_keeps_fourteen_tick_response_and_cuts_earlier_credit(tmp_path):
    model = make_delay_paths(tmp_path, [[1] * 14])
    # Independent source amplitude makes the historical input dependency
    # distinguishable from reuse of the same parameter after detachment.
    model.embedding.weight.requires_grad_(True)
    full, full_state = unfold(model, [1] + [0] * 14)
    source = model.embedding.weight
    full_grad = torch.autograd.grad(full[14].sum(), source)[0][1]
    split, split_state = unfold(model, [1] + [0] * 14, detach_at=7)
    split_grad = torch.autograd.grad(split[14].sum(), source)[0][1]
    torch.testing.assert_close(full, split, atol=0, rtol=0)
    assert_state_equal(full_state, split_state)
    assert full_grad.abs().sum() > 0 and torch.equal(split_grad, torch.zeros_like(split_grad))


def test_checkpoint_inflight_full_state_preserves_fourteen_tick_response(tmp_path):
    model = make_delay_paths(tmp_path, [[4, 4, 4, 2]])
    full, final = unfold(model, [1] + [0] * 14)
    prefix, midway = unfold(model, [1] + [0] * 6)
    payload = midway.state_dict()
    saved = tmp_path / 'inflight.pt'
    torch.save(payload, saved)
    loaded = torch.load(saved, weights_only=True)
    restored = type(midway)(**loaded)
    tail, resumed = unfold(model, [0] * 8, restored)
    torch.testing.assert_close(torch.cat((prefix, tail)), full, atol=0, rtol=0)
    assert_state_equal(resumed, final)


def test_actual_ce_window_receives_fourteen_tick_writer_credit_without_future_input(tmp_path):
    model = make_delay_paths(tmp_path, [[1] * d for d in range(1, 15)])
    learner = FlyPipelineLearner(model, physical(model))
    tokens = torch.tensor([[1] + [0] * 31])
    scores, _, _ = learner.forward_window(tokens, tokens)
    gradient = torch.autograd.grad(scores[14], model.topographic_writer.proj_vis.weight)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    changed = tokens.clone()
    changed[:, 14:] = 2
    other_scores, _, _ = learner.forward_window(changed, tokens)
    # Predictions through tick14 use only inputs0..13; changing the token
    # observed at14 and every later token cannot change the sealed score14.
    torch.testing.assert_close(scores[:15], other_scores[:15], atol=0, rtol=0)
