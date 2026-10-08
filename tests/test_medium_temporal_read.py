"""Production temporal state/credit contracts; numerical checks, not capacity tests."""
import copy
from dataclasses import replace
import io
import math
import os

import numpy as np
import pytest
import torch

from information_boltzmann.core.plastic_ports import PlasticBelief, PlasticMediumPorts3D
from information_boltzmann.runtime.active_medium_training import ActiveMediumTrainer
from information_boltzmann.runtime.continuous import ContinuousStream
from information_boltzmann.runtime.optimization import make_medium_optimizer, medium_parameter_groups
from information_boltzmann.runtime.training import (
    CapturedPlasticChunk, belief_tensors, clone_belief, quiet_training_chunk)
from scripts.ib.train_plastic_conductance import pack_belief, unpack_belief


def net(device='cpu', dtype=torch.float64, **options):
    torch.manual_seed(709)
    return PlasticMediumPorts3D(vocab_size=29, shape=(4, 4, 4), channels=8,
        material_width=3, hidden=6, heads=2, queries=2, bath_type='conductance',
        activity_adaptation=True, short_term_plasticity=True, read_mode='temporal',
        temporal_rates=[math.log(2)/.005, math.log(2)/.02],
        temporal_frequencies=[0., math.pi/(16*.005)], **options).to(device=device, dtype=dtype)


def states_equal(a, b, atol=0., rtol=0.):
    aa, bb = belief_tensors(a), belief_tensors(b)
    assert len(aa) == len(bb)
    for x, y in zip(aa, bb):
        torch.testing.assert_close(x, y, atol=atol, rtol=rtol)


def test_history_integrates_only_on_advance_and_reads_are_idempotent():
    model = net()
    belief = model.initial_belief()
    history = belief.temporal
    belief, _ = model.assimilate(belief, torch.tensor([2]), diagnostics=False)
    assert belief.temporal is history
    belief, _ = model.advance(belief, .005, diagnostics=False)
    assert belief.temporal.value.abs().max() > 0
    torch.testing.assert_close(belief.temporal.elapsed, belief.medium.elapsed, atol=0, rtol=0)
    before = clone_belief(belief)
    a, _ = model.read(belief)
    b, diagnostics = model.read(belief, diagnostics=True)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    states_equal(belief, before)
    assert diagnostics['temporal_rates'].shape == (2,)
    zero, _ = model.advance(belief, 0., diagnostics=False)
    torch.testing.assert_close(zero.temporal.value, belief.temporal.value, atol=0, rtol=0)
    detached = belief.detach()
    assert detached.temporal.value.grad_fn is None
    states_equal(detached, belief)


def test_chunk_public_and_timestamped_execution_share_history_and_gradients():
    model = net()
    ids, targets = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])
    initial = model.initial_belief()
    loss, expected, _ = model(ids, targets, initial, event_duration=.005)
    gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    quiet_loss, actual, _ = quiet_training_chunk(model, ids, targets, initial, event_duration=.005)
    quiet_gradients = torch.autograd.grad(quiet_loss, tuple(model.parameters()), allow_unused=True)
    torch.testing.assert_close(loss, quiet_loss, atol=1e-12, rtol=1e-12)
    states_equal(actual, expected)
    for a, b in zip(gradients, quiet_gradients):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, atol=2e-11, rtol=2e-10)
    _, timestamped, _ = model.forward_timestamped(ids, targets, [0., .005, .01],
        [.005 - 1e-15, .01 - 1e-15, .015], max_step=.005)
    # Timestamped cuts differ by machine epsilon; same declared event samples.
    states_equal(actual, timestamped, atol=2e-11, rtol=2e-10)


def test_chunk_boundaries_preserve_values_and_temporal_only_brain_credit():
    model = net()
    ids, targets = torch.tensor([[1, 2, 3, 4]]), torch.tensor([[2, 3, 4, 5]])
    _, whole, _ = quiet_training_chunk(model, ids, targets, model.initial_belief(), event_duration=.005)
    _, first, _ = quiet_training_chunk(model, ids[:, :2], targets[:, :2],
                                     model.initial_belief(), event_duration=.005)
    _, split, _ = quiet_training_chunk(model, ids[:, 2:], targets[:, 2:], first.detach(), event_duration=.005)
    states_equal(whole, split)
    bank = model.temporal_readout.bank
    feature, _ = model.read(whole, decode=False)
    temporal = model.temporal_readout(whole.temporal)
    isolated = model.decode((feature - temporal).detach() + temporal)
    loss = torch.nn.functional.cross_entropy(isolated, targets[:, -1])
    groups = [[bank.log_rate], [bank.frequency], [model.readout.probe_coords],
              list(model.medium.parameters()), list(model.write_agent.parameters()),
              list(model.source.parameters())]
    params = [p for group in groups for p in group]
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    offset = 0
    for group in groups:
        current = grads[offset:offset + len(group)]
        assert sum(float(g.abs().sum()) for g in current if g is not None) > 0
        assert all(torch.isfinite(g).all() for g in current if g is not None)
        offset += len(group)


def test_probe_history_follows_sensor_identity_across_geometry_updates():
    model = net()
    state, _ = model.assimilate(model.initial_belief(), torch.tensor([3]), diagnostics=False)
    state, _ = model.advance(state, .005, diagnostics=False)
    value = state.temporal.value.detach().clone()
    with torch.no_grad():
        model.readout.probe_coords.add_(.01)
    model.read(state)
    torch.testing.assert_close(state.temporal.value, value, atol=0, rtol=0)
    evolved, _ = model.advance(state, .005, diagnostics=False)
    assert not torch.equal(evolved.temporal.value, value)
    torch.testing.assert_close(evolved.temporal.elapsed, evolved.medium.elapsed, atol=0, rtol=0)


def test_state_pack_and_live_runtime_resume_keep_complex_history_and_clock():
    model = net()
    with torch.no_grad():
        stream = ContinuousStream(model, max_step=.005)
        stream.observe(0., torch.tensor([2]))
        stream.advance_to(.015)
        saved = stream.state_dict()
        payload = io.BytesIO()
        torch.save(saved, payload)
        payload.seek(0)
        restored = ContinuousStream.from_state_dict(model, torch.load(payload, weights_only=True))
        packed = unpack_belief(pack_belief(stream.belief), 'cpu')
        states_equal(stream.belief, packed)
        states_equal(stream.belief, restored.belief)
        assert restored.belief.temporal.value.dtype == torch.complex128
        for value in (stream, restored):
            value.observe(.015, torch.tensor([4]))
            value.advance_to(.02)
        states_equal(stream.belief, restored.belief)
        torch.testing.assert_close(stream.read_current().value, restored.read_current().value, atol=0, rtol=0)
        bad = dict(saved, temporal=None)
        with pytest.raises(ValueError, match='history'):
            ContinuousStream.from_state_dict(model, bad)


def learner():
    model = net()
    return ActiveMediumTrainer(model, make_medium_optimizer(model, lr=2e-4),
        model.initial_belief(), carry_token=1, event_duration=.005,
        chunk_tokens=2, tokens_per_update=4)


def test_live_training_phase_changes_and_pending_optimizer_resume_are_exact():
    a, b = learner(), learner()
    prior = np.full(29, math.log(29))
    a.consume(torch.tensor([2, 3, 4, 5, 6]), phase='train_first_pass', prior_nll=prior)
    assert a.pending == 1 and a.optimizer_updates == 1
    b.model.load_state_dict(copy.deepcopy(a.model.state_dict()))
    b.optimizer.load_state_dict(copy.deepcopy(a.optimizer.state_dict()))
    b.belief = unpack_belief(pack_belief(a.belief), 'cpu')
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    left = a.consume(torch.tensor([7, 8, 9]), phase='fresh_B', prior_nll=prior)
    right = b.consume(torch.tensor([7, 8, 9]), phase='fresh_B', prior_nll=prior)
    assert left == right and a.summary() == b.summary()
    states_equal(a.belief, b.belief)
    torch.testing.assert_close(a.belief.temporal.elapsed, a.belief.medium.elapsed, atol=0, rtol=0)
    for x, y in zip(a.model.parameters(), b.model.parameters()):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    no_decay = medium_parameter_groups(a.model)[1]['param_names']
    assert 'temporal_readout.bank.log_rate' in no_decay
    assert 'temporal_readout.bank.frequency' in no_decay


def test_temporal_mode_requires_explicit_local_history_contract():
    with pytest.raises(ValueError, match='explicit'):
        PlasticMediumPorts3D(read_mode='temporal')
    with pytest.raises(ValueError, match='finite compact'):
        net(port_scope='global')
    model = net()
    empty = model.initial_belief()
    missing = PlasticBelief(empty.medium, empty.precision)
    with pytest.raises(ValueError, match='history'):
        model.read(missing)
    with pytest.raises(ValueError, match='history'):
        model.advance(missing, .005, diagnostics=False)
    bad_clock = replace(empty, temporal=replace(empty.temporal, elapsed=empty.temporal.elapsed + 1.))
    with pytest.raises(RuntimeError, match='clock'):
        model.read(bad_clock)
    with pytest.raises(RuntimeError, match='clock'):
        model.advance(bad_clock, .005, diagnostics=False)


def test_legacy_checkpoint_without_temporal_fields_keeps_existing_defaults():
    model = PlasticMediumPorts3D(vocab_size=29, shape=(4, 4, 4), channels=8,
                                 hidden=6, heads=2, queries=2).double()
    state = model.initial_belief()
    saved = pack_belief(state)
    saved.pop('temporal')
    assert unpack_belief(saved, 'cpu').temporal is None
    stream = ContinuousStream(model, max_step=.005)
    legacy = stream.state_dict()
    legacy['schema'] = 6
    legacy.pop('temporal')
    legacy.pop('temporal_contract')
    restored = ContinuousStream.from_state_dict(model, legacy)
    states_equal(state, restored.belief)
    assert model.read_mode == 'instantaneous'


@pytest.mark.skipif(os.environ.get('IB_ENABLE_RUNTIME_CUDA_TESTS') != '1',
                    reason='Opt-in bounded CUDA acceptance')
def test_temporal_cuda_graph_matches_chunk_gradients_updated_weights_and_ownership():
    model = net(device='cuda', dtype=torch.float32)
    reference = copy.deepcopy(model)
    ids, targets = torch.tensor([[1, 2]], device='cuda'), torch.tensor([[2, 3]], device='cuda')
    initial = model.initial_belief()
    capture = CapturedPlasticChunk(model, ids, targets, initial, event_duration=.005)
    actual = initial
    expected = clone_belief(initial)
    for version in range(2):
        capture.zero_grad()
        reference.zero_grad(set_to_none=True)
        if version:
            with torch.no_grad():
                model.temporal_readout.bank.log_rate.add_(.001)
                reference.temporal_readout.bank.log_rate.add_(.001)
                model.readout.probe_coords.add_(.001)
                reference.readout.probe_coords.add_(.001)
        _, actual, _ = capture.backward(ids, targets, actual)
        loss, expected, _, scores = quiet_training_chunk(reference, ids, targets,
            expected.detach(), event_duration=.005, return_token_nll=True)
        loss.backward()
        states_equal(actual, expected, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(capture.token_nll, scores, atol=3e-6, rtol=3e-5)
        for x, y in zip(model.parameters(), reference.parameters()):
            assert (x.grad is None) == (y.grad is None)
            if x.grad is not None:
                torch.testing.assert_close(x.grad, y.grad, atol=3e-6, rtol=3e-4)
    saved = actual.temporal.value.clone()
    capture.zero_grad()
    capture.backward(ids, targets, initial)
    torch.testing.assert_close(actual.temporal.value, saved, atol=0, rtol=0)


@pytest.mark.skipif(os.environ.get('IB_ENABLE_RUNTIME_CUDA_TESTS') != '1',
                    reason='Opt-in bounded CUDA acceptance')
def test_temporal_runtime_cuda_graph_retains_history():
    model = net(device='cuda', dtype=torch.float32)
    with torch.no_grad():
        graph = ContinuousStream(model, max_step=.005, compile_backend='cuda_graph')
        eager = ContinuousStream(model, max_step=.005)
        for stream in (graph, eager):
            stream.observe(0., torch.tensor([2]), training_terms=False)
            stream.advance_to(.015)
        states_equal(graph.belief, eager.belief, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(graph.read_current().value, eager.read_current().value,
                                   atol=3e-6, rtol=3e-5)
