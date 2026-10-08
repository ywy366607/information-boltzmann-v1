"""Deterministic observer causality/continuation tests, not capability studies."""
from types import SimpleNamespace
import copy

import numpy as np
import torch

from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState
from information_boltzmann.core.fly_streaming_observer import (
    DelayRegionGraph, StreamingGraphObserver, StreamingObserverState,
)


def make_observer(delay=1):
    e = np.zeros((4, 2, 2), dtype=np.float32)
    e[delay - 1, 0, 1] = 1.0
    graph = DelayRegionGraph([0, 0, 1, 1], [1], e, np.zeros_like(e))
    return StreamingGraphObserver(graph, 3, latent_dim=5,
                                  sample_per_region=2, seed=11)


def physical(values=None):
    h = torch.zeros(1, 4) if values is None else values
    return FlyPhysicalState(h, tuple(torch.zeros_like(h) for _ in range(4)),
                            torch.zeros_like(h), torch.zeros_like(h),
                            torch.zeros_like(h), torch.ones_like(h),
                            torch.zeros_like(h), torch.zeros_like(h))


def test_actual_edge_delay_aggregation_and_signs():
    model = SimpleNamespace(
        synapse_model='coba', superclass_id=torch.tensor([0, 0, 1, 1]),
        read_indices=torch.tensor([3]), splits_e=(0, 0, 0, 1, 1),
        splits_i=(0, 1, 1, 1, 1), edge_pre_e=torch.tensor([0]),
        edge_post_e=torch.tensor([3]), edge_weight_e=torch.tensor([2.0]),
        edge_pre_i=torch.tensor([2]), edge_post_i=torch.tensor([3]),
        edge_weight_i=torch.tensor([1.0]))
    graph = DelayRegionGraph.from_model(model)
    pre, inhibitory, motor = (int(graph.region_ids[i]) for i in (0, 2, 3))
    assert graph.max_delay == 4
    torch.testing.assert_close(graph.a_e[2, pre, motor], torch.tensor(2 / 3))
    torch.testing.assert_close(graph.a_i[0, inhibitory, motor], torch.tensor(1 / 3))
    assert graph.a_e[0].count_nonzero() == 0
    assert graph.a_e.count_nonzero() == graph.a_i.count_nonzero() == 1


def test_three_tick_message_is_not_delivered_early():
    observer = make_observer(delay=3)
    history = observer.initial_state(1, 'cpu', torch.float32).history
    impulse = torch.zeros_like(history[0])
    impulse[:, 0] = 1
    for tick in range(3):
        current = impulse if tick == 0 else torch.zeros_like(impulse)
        history = torch.cat((current[None], history[:-1]))
        excitatory, inhibitory = observer.graph.incoming(history)
        assert inhibitory.count_nonzero() == 0
        if tick < 2:
            assert excitatory[:, 1].count_nonzero() == 0
        else:
            torch.testing.assert_close(excitatory[:, 1], torch.ones(1, 5))


def test_current_sensory_motor_forecast_requires_real_one_tick_edge():
    observer = make_observer()
    observer.motor_adapter.weight.data.fill_(0.2)
    h = torch.tensor([[0.3, 0.2, 0.0, 0.0]], requires_grad=True)
    state = observer.initial_state(1, 'cpu', torch.float32)
    output, _, _, _ = observer.step(physical(h), state)
    gradient = torch.autograd.grad(output.sum(), h)[0]
    assert gradient[:, :2].abs().sum() > 0
    observer.graph.a_e.zero_()
    output, _, _, _ = observer.step(physical(h), state)
    gradient = torch.autograd.grad(output.sum(), h)[0]
    assert gradient[:, :2].count_nonzero() == 0


def test_hidden_physical_ring_and_conductance_change_observation():
    observer = make_observer()
    baseline = physical()
    changed = physical()
    changed.ring[2][:, 0] = 0.8
    assert not torch.equal(observer.encode(baseline), observer.encode(changed))
    changed = physical()
    changed.ge[:, 0] = 0.2
    assert not torch.equal(observer.encode(baseline), observer.encode(changed))
    assert torch.equal(baseline.h, changed.h)


def test_observer_partition_and_checkpoint_forward_identity():
    observer = make_observer()
    observer.motor_adapter.weight.data.normal_(std=0.1)
    torch.manual_seed(3)
    sequence = [physical(torch.randn(1, 4) * 0.1) for _ in range(40)]

    def run(sizes):
        state = observer.initial_state(1, 'cpu', torch.float32)
        outputs, cursor = [], 0
        for size in sizes:
            for item in sequence[cursor:cursor + size]:
                out, _, state, _ = observer.step(item, state)
                outputs.append(out.detach())
            state = StreamingObserverState.from_state_dict(
                copy.deepcopy(state.detached().state_dict())).to('cpu')
            cursor += size
        return torch.cat(outputs), state

    full, end = run([40])
    for partition in ([16, 16, 8], [32, 8], [1] * 40):
        actual, continued = run(partition)
        torch.testing.assert_close(actual, full)
        for key, value in end.state_dict().items():
            if key == 'tick':
                assert value == continued.tick == 40
            else:
                torch.testing.assert_close(value, getattr(continued, key))


def test_detached_deadline_source_still_trains_current_predictor():
    observer = make_observer()
    state = observer.initial_state(1, 'cpu', torch.float32)
    _, _, state, metrics = observer.step(physical(torch.ones(1, 4) * 0.1), state)
    assert metrics['observer_scored_arrival'] == 0
    state = state.detached()
    _, loss, following, metrics = observer.step(
        physical(torch.ones(1, 4) * 0.3), state)
    loss.backward()
    assert observer.transition[-1].weight.grad.abs().sum() > 0
    assert state.pending_source.grad_fn is None
    assert following.tick == 2 and metrics['observer_scored_arrival'] == 1


def test_prequential_forecast_score_retains_original_issued_value():
    observer = make_observer()
    state = observer.initial_state(1, 'cpu', torch.float32)
    _, _, state, _ = observer.step(physical(torch.ones(1, 4) * 0.1), state)
    next_evidence = physical(torch.ones(1, 4) * 0.2)
    original_score = (state.issued_forecast - observer.encode(next_evidence)).square().mean()
    observer.transition[-1].bias.data.fill_(0.3)
    _, replay_loss, _, metrics = observer.step(next_evidence, state.detached())
    torch.testing.assert_close(metrics['observer_issued_forecast_mse'], original_score)
    assert not torch.isclose(replay_loss, original_score)


def test_adapter_zero_initialization_preserves_physical_read_scale():
    observer = make_observer()
    output, _, _, _ = observer.step(physical(torch.randn(1, 4)),
                                   observer.initial_state(1, 'cpu', torch.float32))
    assert output.count_nonzero() == 0
    assert not any('observation_projection' in name for name, _ in observer.named_parameters())
