"""Circuit and continuation contracts, not visual/language capability tests."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState, FlyBPTTLearner
from information_boltzmann.core.fly_native_policy_credit import native_parameter_names
from information_boltzmann.core.fly_visual_physiology import attach_visual_physiology
from information_boltzmann.core.fly_activity_audit import instantaneous_release_metrics


def setup_model(tmp_path):
    path = tmp_path / 'visual.npz'
    np.savez(path, neuron_body_ids=np.arange(6), edge_pre=np.arange(6),
             edge_post=np.roll(np.arange(6), 1), edge_weight=np.full(6, .1, np.float32),
             nt_sign=np.array([1, -1, 1, 1, 1, 1]), edge_delay=np.ones(6, np.int32),
             superclass_id=np.array([0, 0, 0, 1, 1, 1]),
             superclass_names=np.array(['cb_sensory', 'cb_motor']))
    annotation = tmp_path / 'annotation.feather'
    pd.DataFrame({'bodyId': np.arange(6), 'type': ['APL', 'R1-R6', 'L1', 'Mi1', 'Tm3', 'T4a']}).to_feather(annotation)
    model = FlyReservoirLM(path, vocab_size=9, d_model=3, injection='sensory',
                           synapse_model='coba', use_alif=False, use_stp=False)
    model.attach_graded([0], gain=.4)
    return model, path, annotation


def state(model):
    h = torch.zeros(1, 6)
    return FlyPhysicalState(h, tuple(h.clone() for _ in range(4)), h.clone(), h.clone(),
                             h.clone(), torch.ones_like(h), torch.zeros_like(h), h.clone())


def test_visual_extension_preserves_existing_parameter_and_release(tmp_path):
    model, graph, ann = setup_model(tmp_path)
    old_gain = model.log_graded_gain
    v = torch.ones(1, 6)
    original = model.graded_release(v, torch.ones_like(v))
    counts = attach_visual_physiology(model, graph, ann)
    assert counts == {'L1': 1, 'Mi1': 1, 'R1-R6': 1, 'Tm3': 1}
    assert model.log_graded_gain is old_gain
    assert torch.equal(model.graded_release(v, torch.ones_like(v))[:, :1], original[:, :1])
    assert model.graded_mask[0, 5] == 0  # T4 is not blanket-converted.
    parameter = model.log_visual_graded_gain
    assert attach_visual_physiology(model, graph, ann) == counts
    assert model.log_visual_graded_gain is parameter


def test_visual_tonic_release_transmits_increments_and_decrements(tmp_path):
    model, graph, ann = setup_model(tmp_path)
    attach_visual_physiology(model, graph, ann)
    levels = [model.graded_release(torch.full((1, 6), x), torch.ones(1, 6))[0, 1]
              for x in (-.1, 0., .1)]
    assert 0 < levels[0] < levels[1] < levels[2]
    levels[-1].backward()
    assert torch.isfinite(model.log_visual_graded_gain.grad).all()
    assert model.log_visual_graded_gain.grad[0] > 0


def test_visual_cells_keep_membrane_and_release_without_spike_reset(tmp_path):
    model, graph, ann = setup_model(tmp_path)
    attach_visual_physiology(model, graph, ann)
    initial = state(model)
    h, ring, ge, gi = initial.h, initial.ring, initial.ge, initial.gi
    for _ in range(3):
        out = model.step(h, torch.tensor([0]), spike_ring=ring, ge=ge, gi=gi,
                         sensory_drive=torch.full_like(h, 10.0))
        h, spikes, ring, ge, gi = out[:5]
        assert torch.equal(spikes[:, 1:5], torch.zeros_like(spikes[:, 1:5]))
        assert (h[:, 1:5] > 0).all()
        assert (ring[0][:, 1:5] > 0).all()


def test_visual_activation_and_resume_preserve_existing_optimizer_moments(tmp_path):
    model, graph, ann = setup_model(tmp_path)
    names = ['output_read.weight', 'decoder.weight', 'read_norm.weight', 'log_graded_gain']
    first = FlyBPTTLearner(model, state(model), adam_names=names, lr_physiology=.005)
    for p in first.optimizer.param_groups:
        for tensor in p['params']:
            tensor.grad = torch.ones_like(tensor)
    first.optimizer.step()
    saved = copy.deepcopy(first.optimizer.state_dict())
    expected = copy.deepcopy(first.optimizer.state[model.log_graded_gain])
    attach_visual_physiology(model, graph, ann)
    second = FlyBPTTLearner(model, state(model), adam_names=names + ['log_visual_graded_gain'], lr_physiology=.005)
    second.load_adam_state(saved, newly_trainable=['log_visual_graded_gain'])
    for key, value in expected.items():
        torch.testing.assert_close(second.optimizer.state[model.log_graded_gain][key], value, atol=0, rtol=0)
    assert model.log_visual_graded_gain not in second.optimizer.state
    assert 'log_visual_graded_gain' in native_parameter_names(model)
    saved_model = copy.deepcopy(model.state_dict())
    resumed, _, _ = setup_model(tmp_path)
    attach_visual_physiology(resumed, graph, ann)
    FlyBPTTLearner(resumed, state(resumed), adam_names=names + ['log_visual_graded_gain'])
    resumed.load_state_dict(saved_model, strict=True)
    v = torch.randn(1, 6)
    torch.testing.assert_close(resumed.graded_release(v, torch.ones_like(v)),
                               model.graded_release(v, torch.ones_like(v)), atol=0, rtol=0)


@pytest.mark.parametrize('gain', [0, -1, float('nan')])
def test_invalid_visual_gain_is_rejected(tmp_path, gain):
    model, graph, ann = setup_model(tmp_path)
    with pytest.raises(ValueError):
        attach_visual_physiology(model, graph, ann, gain=gain)


def test_continuous_visual_release_is_not_counted_as_spikes():
    release = torch.tensor([[.1, .3, 1., 0.]])
    graded = torch.tensor([[1., 1., 0., 0.]])
    metrics = instantaneous_release_metrics(release, graded)
    assert metrics['firing_rate'] == .25
    assert metrics['transmission_active_fraction'] == .75
    assert metrics['spiking_cell_fraction_active'] == .5
