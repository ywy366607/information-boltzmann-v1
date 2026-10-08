"""Numerical/continuation contracts only; no synthetic capability training."""
import copy
from dataclasses import fields

import numpy as np
import pytest
import torch

from information_boltzmann.core.fly_reservoir import FlyReservoirLM, BiologicalTopographicWriter
from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState, step_fly_physical_tick
from information_boltzmann.core.fly_rtc_student import TickResponseStudent
from information_boltzmann.core.fly_rtc_learning import (
    FlyRTCLearner, copy_physical, initial_student_state, predict_rtc_next,
    physical_options, quiet_teacher_step,
)
from information_boltzmann.runtime.fly_rtc import TickForecastBuffer, FlyRTCExecutor


@pytest.fixture
def individual(tmp_path):
    torch.manual_seed(11)
    path = tmp_path / 'interface_graph.npz'
    np.savez(path, neuron_body_ids=np.arange(6), edge_pre=np.arange(6),
             edge_post=np.roll(np.arange(6), 1), edge_weight=np.full(6, .1, dtype=np.float32),
             nt_sign=np.array([1, 1, 1, -1, -1, -1]),
             edge_delay=np.array([1, 2, 3, 4, 2, 1], dtype=np.int32),
             superclass_id=np.array([0, 0, 0, 1, 1, 1]),
             superclass_names=np.array(['cb_sensory', 'cb_motor']))
    parts = tmp_path / 'ports.npz'
    np.savez(parts, visual_idx=np.array([0]), chemo_idx=np.array([1]), mechano_idx=np.array([2]))
    model = FlyReservoirLM(path, vocab_size=9, d_model=3, injection='sensory',
                           read_surface='output', synapse_model='coba', use_alif=True, use_stp=True)
    model.topographic_writer = BiologicalTopographicWriter(3, parts)
    model.injection_mode, model.input_proj = 'topographic', None
    model.rtc_student = TickResponseStudent.from_model(
        model, latent_dim=4, sample_per_region=2, horizon=3)
    h = torch.linspace(.01, .05, 6)[None]
    state = FlyPhysicalState(h, tuple(torch.full_like(h, .5) for _ in range(4)),
                             torch.zeros_like(h), torch.zeros_like(h), torch.zeros_like(h),
                             torch.ones_like(h), model.get_stp_params()[0].detach().expand_as(h).clone(),
                             model.topographic_writer.a_adapt.clone(), torch.zeros_like(h))
    return model, state


def test_full_state_codec_includes_conductance_and_inflight_pulses(individual):
    model, state = individual
    codec = model.rtc_student.codec
    original = codec.encode(state)
    changed = copy_physical(state)
    changed.ge[:, 0] += .2
    assert not torch.equal(original, codec.encode(changed))
    changed = copy_physical(state)
    changed.ring[3][:, 0] += .2
    assert not torch.equal(original, codec.encode(changed))
    assert not list(codec.parameters())


def test_rollout_starts_from_true_observation_and_keeps_delay_history(individual):
    model, state = individual
    student = model.rtc_student
    history = initial_student_state(model, state).history
    history[0] = student.codec.encode(state)
    drafts, end = student.rollout(history)
    torch.testing.assert_close(drafts[0], history[0])
    assert drafts.shape == (4, 1, student.codec.graph.num_regions, 4)
    assert end.shape == history.shape
    assert student.codec.graph.a_e.shape[0] == 4
    assert student.codec.graph.a_i.count_nonzero() > 0
    with pytest.raises(ValueError, match='control'):
        student.rollout(history, controls=torch.zeros(1, *history.shape[1:]))


def test_query_reencoding_and_physical_copy_do_not_mutate_life(individual):
    model, state = individual
    before = copy_physical(state)
    student = model.rtc_student
    error = torch.ones_like(student.codec.encode(state))
    queried, work = student.perturb_copy(state, error, relative_radius=.1)
    assert work > 0
    assert not torch.equal(student.codec.encode(queried), student.codec.encode(state))
    for item in fields(state):
        left, right = getattr(state, item.name), getattr(before, item.name)
        if item.name == 'ring':
            for a, b in zip(left, right):
                torch.testing.assert_close(a, b)
                assert a.data_ptr() != getattr(queried, item.name)[0].data_ptr()
        else:
            torch.testing.assert_close(left, right)
    # Queried conductance/rings remain original full context, not an invented
    # inverse of the low-dimensional student's latent.
    torch.testing.assert_close(queried.ge, state.ge)
    following = quiet_teacher_step(model, queried, physical_options(model))
    assert torch.isfinite(student.codec.encode(following)).all()


def test_joint_update_query_replay_and_complete_resume(individual):
    model, state = individual
    learner = FlyRTCLearner(model, state, lambda_jepa=1)
    learner.previous_token = 0
    scores, metrics = learner.observe([1, 2, 3, 4])
    assert len(scores) == 4
    assert learner.events == learner.physical_ticks == learner.rtc_state.tick == 4
    assert metrics['rtc_query_ticks'] == model.rtc_student.horizon + 1
    assert metrics['rtc_speculative_ticks'] == 4 * model.rtc_student.horizon
    assert len(learner.rtc_state.replay) == 1
    assert not learner.rtc_state.history.requires_grad
    assert metrics['rtc_real_input_arrival_mse'] >= 0
    saved = copy.deepcopy(learner.state_dict())
    resumed = FlyRTCLearner(copy.deepcopy(model), copy_physical(learner.state), lambda_jepa=1)
    resumed.restore_learning_state(saved)
    a, _ = learner.observe([5, 6, 7, 8])
    b, _ = resumed.observe([5, 6, 7, 8])
    torch.testing.assert_close(torch.tensor(a), torch.tensor(b))
    torch.testing.assert_close(learner.state.h, resumed.state.h)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, dict(resumed.model.named_parameters())[name])
    learner.observe([0, 1])
    assert len(learner.rtc_state.replay) == 2
    assert learner.rtc_state.query_ticks == 3 * (model.rtc_student.horizon + 1)


def test_targets_cannot_modify_issued_readout_and_inference_matches(individual):
    model, physical = individual
    alternate = copy.deepcopy(model)
    learner = FlyRTCLearner(model, physical, lambda_jepa=0)
    ongoing = initial_student_state(alternate, physical)
    ids = torch.tensor([[0, 1, 2]])
    expected = []
    for token in ids.unbind(1):
        logits, physical, ongoing = predict_rtc_next(alternate, physical, ongoing, token)
        expected.append(logits)
    targets = torch.tensor([[1, 2, 3]])
    scores, final, features = learner.forward_window(ids, targets)
    torch.testing.assert_close(scores, torch.nn.functional.cross_entropy(
        torch.cat(expected), targets.flatten(), reduction='none'))
    torch.testing.assert_close(final.h, physical.h)
    fresh = FlyRTCLearner(copy.deepcopy(alternate), copy_physical(individual[1]), lambda_jepa=0)
    _, changed_final, changed_features = fresh.forward_window(ids, torch.tensor([[8, 8, 8]]))
    torch.testing.assert_close(features, changed_features)
    torch.testing.assert_close(final.h, changed_final.h)


def test_rollout_attention_has_task_gradient(individual):
    model, state = individual
    model.rtc_student.transition[-1].weight.data.normal_(std=.02)
    model.rtc_student.motor_adapter.weight.data.normal_(std=.1)
    learner = FlyRTCLearner(model, state, lambda_jepa=0)
    scores, _, _ = learner.forward_window(torch.tensor([[0, 1, 2]]), torch.tensor([[1, 2, 3]]))
    scores.mean().backward()
    for parameter in model.rtc_student.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert model.rtc_student.transition[-1].weight.grad.abs().sum() > 0


def test_rtc_commit_validation_and_new_input_suffix():
    cache = TickForecastBuffer(3)
    assert cache.publish(-1, torch.tensor([[1.], [2.], [3.]]), revision=0, input_cutoff=-1)
    key, spoken = cache.commit(0)
    error = cache.arrive(key, torch.tensor([9.]), same_input_history=True)
    assert error == 64
    torch.testing.assert_close(spoken, torch.tensor([1.]))
    revision = cache.new_input(0)
    assert not cache.publish(0, torch.tensor([[100.]]), revision=0, input_cutoff=-1)
    with pytest.raises(LookupError):
        cache.commit(1)
    assert cache.publish(0, torch.tensor([[4.], [5.]]), revision=revision, input_cutoff=0)
    key, spoken = cache.commit(1)
    assert cache.arrive(key, torch.tensor([7.]), same_input_history=False) is None
    restored = TickForecastBuffer.from_state_dict(copy.deepcopy(cache.state_dict()))
    _, next_value = restored.commit(2)
    torch.testing.assert_close(next_value, torch.tensor([5.]))
    assert restored.arrival_count == 1


def test_late_old_plan_cannot_overwrite_fresh_plan():
    cache = TickForecastBuffer(2)
    cache.publish(-1, torch.tensor([[1.], [2.]]), revision=0, input_cutoff=-1)
    old = cache.pending[1][0]
    cache.new_input(0)
    cache.publish(0, torch.tensor([[8.], [9.]]), revision=1, input_cutoff=0)
    assert cache.arrive(old, torch.tensor([99.]), same_input_history=True) == 97**2
    torch.testing.assert_close(cache.pending[1][1], torch.tensor([8.]))
    cache.new_model(1)
    assert cache.arrive(old, torch.tensor([99.]), same_input_history=True) is None
    assert not cache.publish(0, torch.tensor([[8.]]), revision=cache.revision,
                             input_cutoff=0, model_revision=0)
    assert not cache.pending


def test_async_teacher_validates_copy_and_does_not_advance_live_state(individual):
    model, physical = individual
    model.eval()
    history = initial_student_state(model, physical).history
    history[0] = model.rtc_student.codec.encode(physical)
    original = copy_physical(physical)
    executor = FlyRTCExecutor(model, origin_tick=0)
    try:
        audit = executor.stage(physical, history, 0, input_cutoff=0)
        assert audit is not None
        assert len(audit.result(timeout=10)) == model.rtc_student.horizon
        logits, key = executor.output(1)
        assert logits.shape == (1, model.decoder.out_features)
        assert key.target_tick == 1
        torch.testing.assert_close(physical.h, original.h)
        saved = copy.deepcopy(executor.state_dict())
        executor.restore_execution(saved)
        executor.output(2)
        executor.update_parameters(lambda: model.read_norm.weight.data.mul_(.99))
        assert executor.buffer.model_revision == 1
        assert not executor.buffer.pending
    finally:
        executor.close()


def test_cli_is_independent_and_has_explicit_query_budgets():
    from scripts.ib.train_fly_rtc_dagger import parse_args
    args = parse_args([])
    assert args.horizon == 14 and args.window == 32
    assert args.replay_capacity == 2 and args.query_radius == .1
    with pytest.raises(SystemExit):
        parse_args(['--query-radius', '-1'])
    with pytest.raises(SystemExit):
        parse_args(['--additional-tokens', '32'])
    assert parse_args(['--calibrate', '--additional-tokens', '32']).calibrate


def test_sensory_inbox_teacher_uses_exact_selected_drives_and_resumes(individual):
    from information_boltzmann.runtime.fly_rtc_flow import FlyRTCFlow
    model, physical = individual
    model.eval()
    flow = FlyRTCFlow(model, physical)
    try:
        tick = flow.submit(torch.tensor([0]))
        selected = flow.pending[0]
        reference = step_fly_physical_tick(
            model, copy_physical(physical), None, selected['drive'],
            selected['baseline'], physical_options(model))
        flow.output(tick)
        flow.synchronize()
        assert flow.anchor_tick == flow.tick == 1
        torch.testing.assert_close(flow.anchor_physical.h, reference.h)
        saved = copy.deepcopy(flow.state_dict())
        continued = FlyRTCFlow.restore(copy.deepcopy(model), saved)
        try:
            a_tick, b_tick = flow.submit(torch.tensor([1])), continued.submit(torch.tensor([1]))
            assert a_tick == b_tick == 2
            # Quiesce both to compare deterministic delivered teacher values;
            # a race may otherwise legitimately select draft vs arrived output.
            flow.synchronize()
            continued.synchronize()
            a, _ = flow.output(2)
            b, _ = continued.output(2)
            torch.testing.assert_close(a, b)
            torch.testing.assert_close(flow.anchor_physical.h, continued.anchor_physical.h)
        finally:
            continued.close()
    finally:
        flow.close()


def test_continuous_inbox_preserves_earlier_unconsumed_output(individual):
    from information_boltzmann.runtime.fly_rtc_flow import FlyRTCFlow
    model, physical = individual
    model.eval()
    flow = FlyRTCFlow(model, physical)
    try:
        flow.submit(torch.tensor([0]))
        flow.submit(torch.tensor([1]))
        a, a_key = flow.output(1)
        b, b_key = flow.output(2)
        assert a_key.target_tick == 1 and b_key.target_tick == 2
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        flow.synchronize()
        assert flow.buffer.pending
        assert flow.anchor_tick == 2
        assert min(flow.buffer.pending) == 3
        assert flow.buffer.revision >= 3  # two inputs plus arrival correction
    finally:
        flow.close()


def test_drafts_are_available_while_real_teacher_is_blocked(individual, monkeypatch):
    """A deterministic waiting test, independent of thread execution races."""
    from threading import Event
    import information_boltzmann.runtime.fly_rtc_flow as runtime

    model, physical = individual
    model.eval()
    # Nonconstant numerical forecasts make suffix correction observable. This
    # is a scheduler test, not evidence that this untrained predictor is good.
    model.rtc_student.transition[-1].weight.data.normal_(std=.02)
    model.rtc_student.motor_adapter.weight.data.normal_(std=.1)
    entered, release = Event(), Event()
    original_step = runtime.step_fly_physical_tick

    def blocked_step(*args, **kwargs):
        entered.set()
        if not release.wait(timeout=10):
            raise TimeoutError('Test must release its physical worker')
        return original_step(*args, **kwargs)

    monkeypatch.setattr(runtime, 'step_fly_physical_tick', blocked_step)
    flow = runtime.FlyRTCFlow(model, physical)
    try:
        flow.submit(torch.tensor([0]))
        assert entered.wait(timeout=5)
        flow.submit(torch.tensor([1]))
        first, _ = flow.output(1)
        second, _ = flow.output(2)
        assert not release.is_set()
        assert not flow.pending[0]['future'].done()
        assert flow.anchor_tick == 0 and flow.tick == 2
        assert torch.isfinite(first).all() and torch.isfinite(second).all()
        immutable = first.clone(), second.clone()
        old_revision = flow.buffer.revision
        # A full inbox has a declared bound rather than dropping inputs.
        flow.submit(torch.tensor([2]))
        with pytest.raises(BufferError, match='forecast budget'):
            flow.submit(torch.tensor([3]))
        flow.output(3)
        old_suffix = flow.buffer.pending[4][1].clone()
        release.set()
        flow.synchronize()
        assert flow.anchor_tick == flow.tick == 3
        assert flow.buffer.revision > old_revision
        assert min(flow.buffer.pending) == 4
        torch.testing.assert_close(first, immutable[0])
        torch.testing.assert_close(second, immutable[1])
        # Same checkpoint, same known input: correction also actually changes
        # numerical forecasts, not just their metadata.
        assert not torch.equal(old_suffix, flow.buffer.pending[4][1])
    finally:
        release.set()
        flow.close()
