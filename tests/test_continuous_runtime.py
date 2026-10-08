"""Causal ordering, concurrent deployment and retained temporal gradients."""
import copy
from concurrent.futures import TimeoutError
import math
import threading

import pytest
import torch

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime import ContinuousStream, LiveStream


def model():
    torch.manual_seed(443)
    return PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=8,
                                hidden=8, heads=2, queries=2).double()


def test_idle_flow_and_reads_share_persistent_state_without_per_token_duration():
    net = model()
    stream = ContinuousStream(net, max_step=0.01)
    stream.observe(0.0, torch.tensor([2]))
    first = stream.belief
    read = stream.read_at(0.025)
    assert read.state_time == 0.025
    assert stream.steps == 3
    assert (stream.belief.medium.field - first.medium.field).norm() > 0
    assert stream.belief.medium.conduction.abs().sum() > 0
    before = stream.belief
    stream.read_current()
    assert stream.belief is before
    stream.observe(0.025, torch.tensor([3]))
    assert stream.belief.medium.flux is before.medium.flux
    assert stream.belief.medium.conduction is before.medium.conduction
    assert stream.steps == 3  # A write is an event, not an assigned pondering loop.
    with pytest.raises(ValueError, match='Timestamp'):
        stream.observe(0.02, torch.tensor([4]))


def test_cached_and_quiet_execution_matches_original_equations_and_refreshes():
    net = model().eval()
    cached = ContinuousStream(net, max_step=0.01)
    reference = ContinuousStream(copy.deepcopy(net), max_step=0.01, cache_inference=False)
    with torch.no_grad():
        for stream in (cached, reference):
            stream.observe(0.0, torch.tensor([2]), training_terms=False)
            stream.advance_to(0.01)
            stream.advance_to(0.02)
            stream.observe(0.02, torch.tensor([3]), training_terms=False)
        torch.testing.assert_close(cached.belief.medium.field, reference.belief.medium.field)
        torch.testing.assert_close(cached.belief.medium.conduction, reference.belief.medium.conduction)
        assert cached.coefficient_preparations == 1
        assert cached.table_preparations == 1
        assert reference.coefficient_preparations == 2
        assert reference.table_preparations == 2
        net.medium.log_speed.weight.add_(0.01)
        cached.advance_to(0.03)
        assert cached.coefficient_preparations == 2
    state = cached.belief.medium
    regular, info = net.medium.advance(state, 0.01)
    quiet, empty = net.medium.advance(state, 0.01, prepared=net.medium.prepare_evolution(), diagnostics=False)
    assert empty == {} and info['energy_after'].isfinite().all()
    torch.testing.assert_close(regular.field, quiet.field)
    torch.testing.assert_close(regular.conduction, quiet.conduction)


def test_timestamped_training_retains_gradients_through_idle_time_and_targets_are_loss_only():
    net = model()
    ids = torch.tensor([[2, 3, 4]])
    targets = torch.tensor([[3, 4, 5]])
    observed, read = [0.0, 0.02, 0.05], [0.01, 0.03, 0.06]
    initial = net.initial_belief()
    initial = type(initial)(initial.medium.with_field(
        initial.medium.field.detach().requires_grad_()), initial.precision)
    loss, whole, info = net.forward_timestamped(ids, targets, observed, read,
                                               max_step=0.01, belief=initial)
    loss.backward()
    assert initial.medium.field.grad.norm() > 0
    for module in (net.medium.collision_rate, net.medium.conduction_plasticity, net.write_agent, net.readout):
        assert sum(p.grad.norm() for p in module.parameters() if p.grad is not None) > 0
    _, changed, _ = net.forward_timestamped(ids, torch.tensor([[7, 8, 9]]), observed, read, max_step=0.01)
    torch.testing.assert_close(whole.medium.field, changed.medium.field)
    torch.testing.assert_close(whole.medium.conduction, changed.medium.conduction)
    assert math.isclose(info['physical_time'], 0.06)
    with pytest.raises(ValueError, match='precede'):
        net.forward_timestamped(ids, targets, observed, [0.02, 0.03, 0.06], max_step=0.01)


def test_checkpoint_restores_full_persistent_state_and_long_time_precision(tmp_path):
    net = model()
    stream = ContinuousStream(net, max_step=0.01)
    stream.observe(0.0, torch.tensor([2]))
    stream.advance_to(0.02)
    path = tmp_path / 'live.pt'
    torch.save(stream.state_dict(), path)
    resumed = ContinuousStream.from_state_dict(net, torch.load(path, weights_only=True))
    stream.advance_to(0.03)
    resumed.advance_to(0.03)
    torch.testing.assert_close(resumed.belief.medium.field, stream.belief.medium.field)
    torch.testing.assert_close(resumed.belief.medium.conduction, stream.belief.medium.conduction)
    single = model().float()
    state = single.initial_belief()
    assert state.medium.elapsed.dtype == torch.float64
    shifted = type(state)(type(state.medium)(state.medium.field, state.medium.flux,
                         torch.tensor([1e8], dtype=torch.float64), state.medium.conduction), state.precision)
    later, _ = single.advance(shifted, 0.01)
    assert float(later.medium.elapsed - shifted.medium.elapsed) > 0.009


def test_compiled_evolution_entry_matches_uncached_forward():
    net = model()
    compiled = ContinuousStream(net, max_step=0.01, compile_backend='eager')
    normal = ContinuousStream(copy.deepcopy(net), max_step=0.01)
    for stream in (compiled, normal):
        stream.observe(0.0, torch.tensor([2]))
        stream.advance_to(0.015)
    torch.testing.assert_close(compiled.belief.medium.field, normal.belief.medium.field)
    torch.testing.assert_close(compiled.belief.medium.conduction, normal.belief.medium.conduction)


class ManualClock:
    def __init__(self):
        self.time = 0.0
    def __call__(self):
        return self.time


def wake(live, clock, timestamp):
    with live._condition:
        clock.time = timestamp
        live._condition.notify_all()


def test_live_inputs_and_reads_have_independent_clocks_and_idle_evolution():
    net = model().eval()
    clock = ManualClock()
    live = LiveStream(ContinuousStream(net, max_step=0.01), model_time_per_second=1,
                      clock=clock).start()
    try:
        assert live.submit_observation(torch.tensor([2]), timestamp=0).result(5) == 0
        future = live.request_read(timestamp=0.025)
        with pytest.raises(TimeoutError):
            future.result(0.001)
        wake(live, clock, 0.025)
        read = future.result(5)
        assert read.state_time == 0.025
        assert read.lag == 0
        assert live.stream.belief.medium.flux[0].norm() > 0
        current = live.request_read(decode=False).result(5)
        assert current.state_time == 0.025 and current.value.shape == (1, 8)
        bad = live.submit_observation(torch.tensor([3]), timestamp=0.01)
        with pytest.raises(ValueError, match='Late event'):
            bad.result(5)
    finally:
        live.close()


def test_nonblocking_submission_and_current_read_bypass_time_catchup():
    net = model().eval()
    clock = ManualClock()
    stream = ContinuousStream(net, max_step=0.01)
    live = LiveStream(stream, model_time_per_second=1, clock=clock)
    entered, release = threading.Event(), threading.Event()
    original = stream.advance_to
    def delayed(timestamp):
        if timestamp > 0:
            entered.set()
            assert release.wait(5)
        original(timestamp)
    stream.advance_to = delayed
    live.start()
    try:
        live.submit_observation(torch.tensor([2]), timestamp=0).result(5)
        wake(live, clock, 0.1)
        assert entered.wait(5)
        # Producers return Futures while the owner is busy.
        observed = live.submit_observation(torch.tensor([3]), timestamp=0.1)
        current = live.request_read(decode=False)
        assert not observed.done() and not current.done()
        release.set()
        result = current.result(5)
        assert result.state_time == 0.01 and result.lag > 0.08
        assert observed.result(5) == 0.1
    finally:
        release.set()
        live.close()


def test_worker_failure_resolves_active_future_instead_of_hanging():
    net = model().eval()
    stream = ContinuousStream(net, max_step=0.01)
    def fail(*args, **kwargs):
        raise RuntimeError('injected owner failure')
    stream.observe = fail
    live = LiveStream(stream, model_time_per_second=1, clock=ManualClock()).start()
    future = live.submit_observation(torch.tensor([2]), timestamp=0)
    with pytest.raises(RuntimeError, match='injected'):
        future.result(5)
    with pytest.raises(RuntimeError, match='failed'):
        live.close()
