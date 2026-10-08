"""The numerical observer must detect changed decisions without changing them."""

import numpy as np
import pytest
import torch

import information_boltzmann.core.fly_bptt_learning as learning
from information_boltzmann.core.fly_reservoir import SpikeFn
from scripts.ib.fly_checkpoint_audit import CheckpointSpikeAudit


@pytest.mark.parametrize('noise,flips', [(0., 0), (.2, 1)])
def test_checkpoint_observer_compares_matching_same_run_contexts(noise, flips):
    x = torch.tensor([-.1, .3], requires_grad=True)
    observer = CheckpointSpikeAudit()
    original_checkpoint, original_spike = learning.checkpoint, SpikeFn.forward
    def function(value):
        if observer.active[1] == 'recompute':
            value = value + noise
        return SpikeFn.apply(value).square().sum()
    with observer:
        learning.checkpoint(function, x, use_reentrant=False).backward()
    report = observer.summary()
    assert report['coverage_complete']
    assert report['total_spike_flips'] == flips
    assert report['compared_neuron_decisions'] == 2
    assert report['voltage_max_abs_difference'] == pytest.approx(noise)
    assert torch.isfinite(x.grad).all()
    assert learning.checkpoint is original_checkpoint and SpikeFn.forward is original_spike


def test_audit_restores_hooks_when_forward_fails():
    original_checkpoint, original_spike = learning.checkpoint, SpikeFn.forward
    with pytest.raises(ValueError, match='deliberate'):
        with CheckpointSpikeAudit():
            raise ValueError('deliberate')
    assert learning.checkpoint is original_checkpoint and SpikeFn.forward is original_spike


def test_threshold_proxy_observer_detects_width_drift_without_spike_flips():
    x = torch.tensor([-.1, .3], requires_grad=True)
    observer = CheckpointSpikeAudit(track_proxy=True)
    def function(value):
        width = value.new_tensor(.2 if observer.active[1] == 'recompute' else .1)
        return SpikeFn.apply(value, width).square().sum()
    with observer:
        learning.checkpoint(function, x, use_reentrant=False).backward()
    report = observer.summary()
    assert report['total_spike_flips'] == 0
    assert report['proxy_compared_calls'] == 1
    assert report['proxy_max_abs_difference'] == pytest.approx(1.5)
    assert report['proxy_derivative_max_abs_difference'] > 0


def test_observing_the_same_graph_preserves_all_gradients():
    first = torch.tensor([-.2, .4], requires_grad=True)
    second = first.detach().clone().requires_grad_()
    def function(value):
        return (SpikeFn.apply(value) * value).sum()
    y1 = learning.checkpoint(function, first, use_reentrant=False)
    y1.backward()
    observer = CheckpointSpikeAudit()
    with observer:
        y2 = learning.checkpoint(function, second, use_reentrant=False)
        y2.backward()
    assert torch.equal(y1, y2) and torch.equal(first.grad, second.grad)
    assert observer.summary()['total_spike_flips'] == 0


def test_state_adjoint_observer_preserves_full_loss_and_parameter_gradients(tmp_path):
    import copy
    from test_fly_bptt_learning import make_model, physical
    from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner
    from scripts.ib.fly_adjoint_audit import FlyAdjointAudit
    torch.manual_seed(37)
    first = make_model(tmp_path, detach_reset=True)
    second = copy.deepcopy(first)
    learners = [FlyBPTTLearner(model, physical(model), settle_ticks=3,
                              use_ctm_loss=True, use_checkpointing=True)
                for model in (first, second)]
    ids = torch.tensor([[0, 1, 2]])
    targets = torch.tensor([[1, 2, 3]])
    output1 = learners[0].forward_window(ids, targets)[0]
    loss1 = learners[0].last_train_loss.mean()
    loss1.backward()
    observer = FlyAdjointAudit(second)
    with observer:
        output2 = learners[1].forward_window(ids, targets)[0]
        loss2 = learners[1].last_train_loss.mean()
        loss2.backward()
    assert torch.equal(output1, output2) and torch.equal(loss1, loss2)
    named2 = dict(second.named_parameters())
    for name, parameter in first.named_parameters():
        other = named2[name]
        assert (parameter.grad is None) == (other.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
    report = observer.summary()
    assert report['original_ticks'] == 12
    assert report['ticks_with_any_adjoint'] == 12
    assert all(info['finite'] for row in report['ticks'] for info in row['adjoints'].values())
    assert second.finish_coba_tick == observer._finish_tick


@pytest.mark.parametrize('path', ['active_only', 'silent_only'])
def test_pulse_credit_mask_uses_original_transmitted_support(path):
    from scripts.ib.fly_pulse_credit_audit import PulseCreditAudit
    observer = PulseCreditAudit(None, path)
    observer.ticks[(0, 0)] = {'hook_calls': 0}
    pulse = torch.tensor([0., 1., 0., 2.], requires_grad=True)
    active = np.array([False, True, False, True])
    pulse.register_hook(observer._mask_hook((0, 0), active))
    (pulse * torch.tensor([1., 2., 3., 4.])).sum().backward()
    expected = [0., 2., 0., 4.] if path == 'active_only' else [1., 0., 3., 0.]
    assert torch.equal(pulse.grad, torch.tensor(expected))
    assert observer.ticks[(0, 0)]['hook_calls'] == 1


@pytest.mark.parametrize('path', ['active_only', 'silent_only'])
def test_pulse_intervention_preserves_forward_loss_state_and_direct_read_gradients(tmp_path, path):
    import copy
    from test_fly_bptt_learning import make_model, physical
    from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner
    from scripts.ib.fly_pulse_credit_audit import PulseCreditAudit
    torch.manual_seed(37)
    first = make_model(tmp_path, detach_reset=True)
    second = copy.deepcopy(first)
    initial = physical(first)
    initial.h[0, 0] = 1.
    initial.ring = tuple(torch.full_like(initial.h, .5) for _ in range(4))
    learners = [FlyBPTTLearner(model, copy.deepcopy(initial), settle_ticks=3,
                              use_ctm_loss=True, use_checkpointing=True)
                for model in (first, second)]
    ids, targets = torch.tensor([[0, 1, 2]]), torch.tensor([[1, 2, 3]])
    output1, state1 = learners[0].forward_window(ids, targets)[:2]
    loss1 = learners[0].last_train_loss.mean()
    loss1.backward()
    observer = PulseCreditAudit(second, path)
    with observer:
        output2, state2 = learners[1].forward_window(ids, targets)[:2]
        loss2 = learners[1].last_train_loss.mean()
        loss2.backward()
    assert torch.equal(output1, output2) and torch.equal(loss1, loss2)
    for name, value in state1.state_dict().items():
        other = state2.state_dict()[name]
        tensors, others = (value, other) if name == 'ring' else ((value,), (other,))
        assert all(torch.equal(a, b) for a, b in zip(tensors, others))
    named2 = dict(second.named_parameters())
    changed = []
    for name, parameter in first.named_parameters():
        other = named2[name]
        if parameter.grad is not None and other.grad is not None:
            if name.startswith(('output_read.', 'decoder.', 'read_norm.')):
                torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
            elif not torch.equal(parameter.grad, other.grad):
                changed.append(name)
    assert changed, 'Intervention must actually change recurrent-path gradients'
    report = observer.summary()
    assert report['original_ticks'] == 12
    assert report['ticks_with_hook'] > 0
    assert second.finish_coba_tick == observer._finish_tick
