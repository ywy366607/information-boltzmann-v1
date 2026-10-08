"""Numerical/interface checks; these are not capability experiments."""
import numpy as np
import copy
import gc
from dataclasses import fields
import pytest
import torch
from torch import nn

from information_boltzmann.core.fly_reservoir import FlyReservoirLM, BiologicalTopographicWriter
from information_boltzmann.core.fly_bptt_learning import (
    FlyPhysicalState, FlyBPTTLearner, FlyBPTTGraph,
    advance_fly_input_event, predict_fly_next,
)


def make_model(tmp_path, **kwargs):
    graph = tmp_path/'graph.npz'
    np.savez(graph, neuron_body_ids=np.arange(6), edge_pre=np.arange(6),
        edge_post=np.roll(np.arange(6), 1), edge_weight=np.full(6, .1, dtype=np.float32),
        nt_sign=np.array([1, 1, 1, -1, -1, -1]), edge_delay=np.ones(6, dtype=np.int32),
        superclass_id=np.array([0, 0, 0, 1, 1, 1]),
        superclass_names=np.array(['cb_sensory', 'cb_motor']))
    partitions = tmp_path/'parts.npz'
    np.savez(partitions, visual_idx=np.array([0]), chemo_idx=np.array([1]), mechano_idx=np.array([2]))
    model = FlyReservoirLM(graph, vocab_size=9, d_model=3, injection='sensory',
        read_surface='output', synapse_model='coba', use_alif=True, use_stp=True, **kwargs)
    model.topographic_writer = BiologicalTopographicWriter(3, partitions)
    model.injection_mode, model.input_proj = 'topographic', None
    return model


def physical(model):
    h = torch.zeros(1, model.n_neurons)
    u = model.get_stp_params()[0].detach().expand_as(h).clone()
    return FlyPhysicalState(h, tuple(h.clone() for _ in range(4)), h.clone(), h.clone(),
        h.clone(), torch.ones_like(h), u, model.topographic_writer.a_adapt.clone())


def test_functional_writer_retains_adaptation_history(tmp_path):
    torch.manual_seed(11)
    model = make_model(tmp_path)
    writer = model.topographic_writer
    a = writer.a_adapt.clone()
    h = torch.zeros(1, model.n_neurons)
    tokens = [torch.randn(1, 3, requires_grad=True) for _ in range(3)]
    for token in tokens:
        drive, a = writer.forward_with_state(token, h, a)
    gradients = torch.autograd.grad(drive.sum(), tokens)
    assert all(torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in gradients)
    assert torch.count_nonzero(drive[:, model.read_indices]) == 0
    assert torch.equal(writer.a_adapt, torch.zeros_like(writer.a_adapt))


def test_physical_detach_preserves_every_value():
    tensor = torch.randn(1, 5, requires_grad=True)*2
    original = FlyPhysicalState(tensor, (tensor,)*4, tensor, tensor, tensor, tensor, tensor, tensor)
    detached = original.detached()
    for key, value in original.state_dict().items():
        values = value if key == 'ring' else (value,)
        targets = getattr(detached, key) if key == 'ring' else (getattr(detached, key),)
        for actual, expected in zip(targets, values):
            torch.testing.assert_close(actual, expected)
            assert not actual.requires_grad


def test_bptt_joint_gradients_and_continuing_windows(tmp_path):
    torch.manual_seed(5)
    model = make_model(tmp_path)
    state = physical(model)
    state.h.copy_(torch.linspace(.01, .05, model.n_neurons)[None])
    state.ring = tuple(torch.ones_like(state.h)*.5 for _ in range(4))
    learner = FlyBPTTLearner(model, state, lr=1e-4, lr_synapse=1e-3)
    learner.previous_token = 0
    old_edge = model.edge_weight_e.clone()
    scores, metrics = learner.observe([1, 2, 3, 4, 5, 6, 7, 8])
    assert len(scores) == 8 and metrics['synapse_grad_norm'] > 0 and metrics['writer_grad_norm'] > 0
    assert torch.isfinite(torch.tensor(scores)).all()
    assert learner.updates == 1 and learner.events == 8 and learner.previous_token == 8
    assert not learner.state.h.requires_grad and not learner.state.baseline.requires_grad
    assert not torch.equal(old_edge, model.edge_weight_e)
    assert learner.state.baseline.abs().sum() > 0
    assert not model.embedding.weight.requires_grad
    assert not model.log_tau_fac.requires_grad
    prior_state = learner.state.h.clone()
    learner.observe([0, 1, 2, 3])
    assert learner.events == 12 and learner.updates == 2
    assert not torch.equal(prior_state, learner.state.h)


def test_uniform_adamw_has_all_trainable_parameters_and_continuation_metadata(tmp_path):
    torch.manual_seed(5)
    model = make_model(tmp_path)
    state = physical(model)
    state.ring = tuple(torch.ones_like(state.h)*.5 for _ in range(4))
    learner = FlyBPTTLearner(model, state)
    learner.previous_token = 0
    assert isinstance(learner.optimizer, torch.optim.AdamW)
    assert isinstance(learner.sgd, torch.optim.AdamW)
    groups = learner.optimizer.param_groups + learner.sgd.param_groups
    assert all(group['lr'] == 2e-4 for group in groups)
    optimized = [id(p) for group in groups for p in group['params']]
    assert len(optimized) == len(set(optimized))
    assert set(optimized) == {id(p) for p in learner.trainable}
    learner.observe([1, 2, 3, 4, 5, 6, 7, 8])
    assert all('exp_avg_sq' in learner.sgd.state[p]
               for p in learner.projections + learner.edges)
    assert learner.state_dict()['plasticity_optimizer_kind'] == 'adamw'


def test_decoder_lr_split_preserves_legacy_adam_moments(tmp_path):
    torch.manual_seed(5)
    model = make_model(tmp_path)
    state = physical(model)
    state.ring = tuple(torch.ones_like(state.h)*.5 for _ in range(4))
    first = FlyBPTTLearner(model, state)
    first.previous_token = 0
    first.observe([1, 2, 3, 4, 5, 6, 7, 8])
    legacy = copy.deepcopy(first.optimizer.state_dict())
    for group in legacy['param_groups']:
        group.pop('parameter_names')
    expected = {name: copy.deepcopy(first.optimizer.state[dict(model.named_parameters())[name]])
                for name in first.adam_names}
    second = FlyBPTTLearner(copy.deepcopy(model), first.state, lr=2e-4, lr_decoder=1e-4)
    second.load_adam_state(legacy)
    named = dict(second.model.named_parameters())
    for name, old in expected.items():
        actual = second.optimizer.state[named[name]]
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(actual[key], old[key], atol=0, rtol=0)
    for group in second.optimizer.param_groups:
        expected_lr = 1e-4 if group['parameter_names'] == ['decoder.weight'] else 2e-4
        assert group['lr'] == expected_lr
    # Named three-group checkpoints can also be loaded into an unchanged
    # two-group diagnostic optimizer without losing or swapping any moment.
    third = FlyBPTTLearner(copy.deepcopy(model), first.state)
    third.load_adam_state(second.optimizer.state_dict())
    for name, old in expected.items():
        actual = third.optimizer.state[dict(third.model.named_parameters())[name]]
        torch.testing.assert_close(actual['exp_avg'], old['exp_avg'], atol=0, rtol=0)


@pytest.mark.parametrize('settle_ticks', [0, 1])
@pytest.mark.parametrize('writer_clock', ['input', 'physical'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('learn_stp', [False, True])
def test_captured_joint_learning_matches_eager_without_warmup_updates(tmp_path, settle_ticks, writer_clock, learn_stp):
    # Previous parametrized graph/learner cycles must be released before a new
    # capture, rather than finalized by Python GC while capture is active.
    gc.collect()
    torch.manual_seed(7)
    model = make_model(tmp_path).cuda()
    other = copy.deepcopy(model)
    cpu = physical(model)
    state = FlyPhysicalState(**{key: tuple(t.cuda() for t in value) if key == 'ring'
        else value.cuda() for key, value in cpu.state_dict().items()})
    state.h.copy_(torch.linspace(.01, .05, model.n_neurons, device='cuda')[None])
    state.ring = tuple(torch.ones_like(state.h)*.5 for _ in range(4))
    first = FlyBPTTLearner(model, copy.deepcopy(state), lr=1e-4, settle_ticks=settle_ticks,
                         writer_baseline_clock=writer_clock, learn_stp=learn_stp)
    second = FlyBPTTLearner(other, copy.deepcopy(state), lr=1e-4, settle_ticks=settle_ticks,
                          writer_baseline_clock=writer_clock, learn_stp=learn_stp)
    first.previous_token = second.previous_token = 0
    before = {name: p.detach().clone() for name, p in other.named_parameters()}
    second.runner = FlyBPTTGraph(second, 8)
    for name, p in other.named_parameters():
        torch.testing.assert_close(p, before[name], atol=0, rtol=0)
    assert second.events == 0 and second.updates == 0
    for targets in ([1, 2, 3, 4, 5, 6, 7, 8], [0, 2, 4, 6, 8, 1, 3, 5]):
        a, _ = first.observe(targets)
        b, _ = second.observe(targets)
        torch.testing.assert_close(torch.tensor(a), torch.tensor(b), atol=2e-5, rtol=2e-5)
        for p, q in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)
        for key, value in first.state.state_dict().items():
            a_values = value if key == 'ring' else (value,)
            b_value = getattr(second.state, key)
            b_values = b_value if key == 'ring' else (b_value,)
            for a_value, b_value in zip(a_values, b_values):
                torch.testing.assert_close(a_value, b_value, atol=2e-6, rtol=2e-5)
    assert first.physical_ticks == second.physical_ticks == 16*(1+settle_ticks)


def test_quiet_propagation_connects_current_input_without_reinjection(tmp_path, monkeypatch):
    model = make_model(tmp_path)
    with torch.no_grad():
        model.embedding.weight.zero_()
        model.embedding.weight[0, 0] = 1.
        model.topographic_writer.gate_linear.weight.zero_()
        model.topographic_writer.gate_linear.bias.zero_()
        for projection in ('proj_vis', 'proj_chemo', 'proj_mech'):
            getattr(model.topographic_writer, projection).weight.zero_()
        # This fixture's one-delay sensory->motor edge is 0 -> 5.
        model.topographic_writer.proj_vis.weight[0, 0] = 1.
        model.output_read.weight.copy_(torch.eye(3))
    initial = physical(model)
    initial.h[:, model.read_indices] = .02
    token_a, token_b = torch.tensor([0]), torch.tensor([1])
    legacy_a = advance_fly_input_event(model, initial, token_a, settle_ticks=0)
    legacy_b = advance_fly_input_event(model, initial, token_b, settle_ticks=0)
    torch.testing.assert_close(legacy_a.h[:, model.read_indices],
                               legacy_b.h[:, model.read_indices], atol=0, rtol=0)
    calls, sources = [], []
    original_writer = model.topographic_writer.forward_with_state
    original_step = model.step

    def writer(*args):
        calls.append(1)
        return original_writer(*args)

    def tick(*args, **kwargs):
        sources.append(kwargs['sensory_drive'].detach().clone())
        return original_step(*args, **kwargs)

    monkeypatch.setattr(model.topographic_writer, 'forward_with_state', writer)
    monkeypatch.setattr(model, 'step', tick)
    fixed_a = advance_fly_input_event(model, initial, token_a, settle_ticks=1)
    fixed_b = advance_fly_input_event(model, initial, token_b, settle_ticks=1)
    assert len(calls) == 2 and len(sources) == 4
    assert torch.count_nonzero(sources[1]) == torch.count_nonzero(sources[3]) == 0
    assert not torch.equal(fixed_a.h[:, model.read_indices], fixed_b.h[:, model.read_indices])
    assert torch.count_nonzero(sources[0][:, model.read_indices]) == 0
    torch.testing.assert_close(fixed_a.baseline, legacy_a.baseline, atol=0, rtol=0)
    gradient = torch.autograd.grad(fixed_a.h[:, model.read_indices].sum(),
                                   model.topographic_writer.proj_vis.weight)[0]
    assert gradient.abs().sum() > 0


def test_event_timing_causality_generation_and_window_continuation(tmp_path):
    torch.manual_seed(43)
    model = make_model(tmp_path)
    initial = physical(model)
    initial.h.copy_(torch.linspace(.02, .07, 6)[None])
    initial.ring = tuple(torch.full_like(initial.h, .3) for _ in range(4))
    learner = FlyBPTTLearner(model, initial, settle_ticks=1)
    ids = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4]])
    scores, full_state, features = learner.forward_window(ids, targets)
    _, _, changed_targets = learner.forward_window(ids, targets.flip(1))
    torch.testing.assert_close(changed_targets, features, atol=0, rtol=0)
    _, _, changed_future = learner.forward_window(torch.tensor([[0, 1, 8, 7]]), targets)
    torch.testing.assert_close(changed_future[:2], features[:2], atol=0, rtol=0)
    logits, _ = predict_fly_next(model, initial, ids[:, 0], settle_ticks=1,
                                 writer_baseline_clock=learner.writer_baseline_clock)
    expected = torch.nn.functional.cross_entropy(logits, targets[:, 0], reduction='none')
    torch.testing.assert_close(scores[:1], expected)
    first, mid_state, first_features = learner.forward_window(ids[:, :2], targets[:, :2])
    learner.state = mid_state.detached()
    second, split_state, second_features = learner.forward_window(ids[:, 2:], targets[:, 2:])
    torch.testing.assert_close(torch.cat((first, second)), scores)
    torch.testing.assert_close(torch.cat((first_features, second_features)), features)
    for key, value in full_state.state_dict().items():
        actual = getattr(split_state, key)
        if key == 'ring':
            for a, b in zip(value, actual):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        else:
            torch.testing.assert_close(value, actual, atol=0, rtol=0)
    learner.previous_token = 3
    learner.observe([4, 5, 6, 7])
    assert learner.events == 4 and learner.physical_ticks == 8 and learner.updates == 1
    assert learner.state_dict()['settle_ticks'] == 1
    assert learner.state_dict()['writer_baseline_clock'] == 'input'


@pytest.mark.parametrize('quiet_ticks', [1, 3])
def test_packet_baseline_uses_observation_clock_and_preserves_archived_mode(tmp_path, quiet_ticks):
    model = make_model(tmp_path)
    initial = physical(model)
    initial.baseline.fill_(.4)
    token = torch.tensor([0])
    pulse = advance_fly_input_event(model, initial, token, settle_ticks=0)
    corrected = advance_fly_input_event(model, initial, token, settle_ticks=quiet_ticks,
                                        writer_baseline_clock='input')
    archived = advance_fly_input_event(model, initial, token, settle_ticks=quiet_ticks,
                                       writer_baseline_clock='physical')
    torch.testing.assert_close(corrected.baseline, pulse.baseline, atol=0, rtol=0)
    expected = pulse.baseline
    for _ in range(quiet_ticks):
        expected = model.topographic_writer.lambda_adapt * expected
    torch.testing.assert_close(archived.baseline, expected, atol=0, rtol=0)
    # The correction changes only the prediction baseline during this event;
    # physical dynamics still execute every quiet tick with exactly zero source.
    for key in ('h', 'ge', 'gi', 'b', 'x', 'u'):
        torch.testing.assert_close(getattr(corrected, key), getattr(archived, key), atol=0, rtol=0)
    for a, b in zip(corrected.ring, archived.ring):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    a_grad = torch.autograd.grad(corrected.baseline.sum(), model.topographic_writer.proj_vis.weight,
                                retain_graph=True)[0]
    p_grad = torch.autograd.grad(pulse.baseline.sum(), model.topographic_writer.proj_vis.weight)[0]
    torch.testing.assert_close(a_grad, p_grad, atol=0, rtol=0)


def test_writer_clock_requires_explicit_valid_semantics(tmp_path):
    model = make_model(tmp_path)
    state = physical(model)
    with pytest.raises(ValueError, match='writer_baseline_clock'):
        advance_fly_input_event(model, state, torch.tensor([0]), writer_baseline_clock='zero')
    with pytest.raises(ValueError, match='writer_baseline_clock'):
        FlyBPTTLearner(model, state, writer_baseline_clock='zero')


@pytest.mark.parametrize('legacy_names', [False, True])
def test_stp_activation_preserves_moments_state_and_has_actual_updates(tmp_path, legacy_names):
    from information_boltzmann.core.fly_bptt_learning import STP_PARAMETER_NAMES
    torch.manual_seed(5)
    model = make_model(tmp_path)
    with torch.no_grad():
        model.log_threshold.fill_(-5)
    first = FlyBPTTLearner(model, physical(model), settle_ticks=1)
    first.state.h.fill_(.03)
    first.state.ring = tuple(torch.full_like(first.state.h, .5) for _ in range(4))
    first.previous_token = 0
    first.observe([1, 2, 3, 4, 5, 6, 7, 8])
    saved = copy.deepcopy(first.optimizer.state_dict())
    if legacy_names:
        for group in saved['param_groups']:
            group.pop('parameter_names')
    second = FlyBPTTLearner(copy.deepcopy(first.model), first.state,
        adam_names=first.adam_names, learn_stp=True, settle_ticks=1)
    with pytest.raises(ValueError, match='coverage|ordering'):
        second.load_adam_state(saved)
    second.load_adam_state(saved, newly_trainable=STP_PARAMETER_NAMES)
    second.sgd.load_state_dict(copy.deepcopy(first.sgd.state_dict()))
    before = {name: p.detach().clone() for name, p in second.model.named_parameters()}
    named = dict(second.model.named_parameters())
    for name in first.adam_names:
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(second.optimizer.state[named[name]][key],
                first.optimizer.state[dict(first.model.named_parameters())[name]][key], atol=0, rtol=0)
    for name in STP_PARAMETER_NAMES:
        assert named[name].requires_grad and named[name] not in second.optimizer.state
    for key, value in first.state.state_dict().items():
        actual = getattr(second.state, key)
        for a, b in zip(value if key == 'ring' else (value,), actual if key == 'ring' else (actual,)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    second.previous_token = first.previous_token
    # Migration equality was checked above. Excite the numerical fixture with
    # available resources to test all three STP derivatives/updates directly.
    second.state.h.fill_(.03)
    second.state.x.fill_(1.)
    second.state.u.copy_(second.model.get_stp_params()[0].detach())
    second.state.baseline.zero_()
    second.state.ring = tuple(torch.full_like(second.state.h, .5) for _ in range(4))
    _, metrics = second.observe([1, 2, 3, 4, 5, 6, 7, 8]*4)
    for name in STP_PARAMETER_NAMES:
        assert metrics[f'{name}_grad_norm'] > 0
        assert not torch.equal(named[name], before[name])
        assert int(second.optimizer.state[named[name]]['step']) == 1
    assert second.state_dict()['learn_stp'] is True


def test_load_adam_state_reinit_parameter(tmp_path):
    torch.manual_seed(44)
    model = make_model(tmp_path, use_read_gamma_trace=True)
    initial = physical(model)
    learner = FlyBPTTLearner(model, initial, settle_ticks=1)
    # Perform an update so Adam states are populated
    ids = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4]])
    scores, _, _ = learner.forward_window(ids, targets)
    scores.sum().backward()
    learner.optimizer.step()
    saved = learner.optimizer.state_dict()
    gamma_param = dict(learner.model.named_parameters())['logit_read_gamma']
    assert gamma_param in learner.optimizer.state

    # Second learner resumes but reinitializes gamma: moments start at zero
    second_model = make_model(tmp_path, use_read_gamma_trace=True)
    second = FlyBPTTLearner(second_model, physical(second_model), settle_ticks=1)
    second.load_adam_state(saved, newly_trainable={'logit_read_gamma'})
    second_gamma = dict(second.model.named_parameters())['logit_read_gamma']
    assert second_gamma not in second.optimizer.state


@pytest.mark.parametrize("settle_ticks,window,temporal_read", [(3, 5, False), (14, 32, False), (3, 5, True)])
@pytest.mark.parametrize("detach_reset", [False, True])
@pytest.mark.parametrize("surrogate_mode", ['absolute', 'threshold'])
@pytest.mark.parametrize("transmission_mode", ['atomic', 'incoming'])
def test_checkpointing_equivalence(tmp_path, settle_ticks, window, temporal_read, detach_reset, surrogate_mode, transmission_mode):
    torch.manual_seed(42)
    m1 = make_model(tmp_path, read_centering=temporal_read,
                    use_read_gamma_trace=temporal_read, detach_reset=detach_reset,
                    surrogate_mode=surrogate_mode, transmission_mode=transmission_mode)
    m2 = copy.deepcopy(m1)

    s1 = physical(m1)
    s1.h.fill_(.02)
    s1.ring = tuple(torch.full_like(s1.h, .05) for _ in range(4))
    # Track input-state derivatives as well as parameter derivatives. These
    # leaf states represent the beginning of the one unbroken BPTT window.
    for item in fields(s1):
        value = getattr(s1, item.name)
        if isinstance(value, tuple):
            value = tuple(t.detach().clone().requires_grad_() for t in value)
        else:
            value = value.detach().clone().requires_grad_()
        setattr(s1, item.name, value)
    s2 = copy.deepcopy(s1)

    learner1 = FlyBPTTLearner(m1, s1, settle_ticks=settle_ticks, learn_stp=True,
                            use_ctm_loss=True, use_checkpointing=False)
    learner2 = FlyBPTTLearner(m2, s2, settle_ticks=settle_ticks, learn_stp=True,
                            use_ctm_loss=True, use_checkpointing=True)

    ids = (torch.arange(window) % 9)[None]
    targets = ((torch.arange(window) + 1) % 9)[None]

    learner1.state, learner2.state = s1, s2
    scores1, state1, feat1 = learner1.forward_window(ids, targets)
    scores2, state2, feat2 = learner2.forward_window(ids, targets)

    assert torch.allclose(scores1, scores2, atol=1e-5), f"Scores mismatch: max diff {(scores1 - scores2).abs().max()}"
    assert torch.allclose(feat1, feat2, atol=1e-5), f"Features mismatch: max diff {(feat1 - feat2).abs().max()}"
    for item in fields(state1):
        v1, v2 = getattr(state1, item.name), getattr(state2, item.name)
        tensors1 = v1 if isinstance(v1, tuple) else (v1,)
        tensors2 = v2 if isinstance(v2, tuple) else (v2,)
        assert len(tensors1) == len(tensors2)
        for t1, t2 in zip(tensors1, tensors2):
            torch.testing.assert_close(t1, t2, atol=1e-5, rtol=1e-5)

    loss1 = learner1.last_train_loss.mean()
    loss2 = learner2.last_train_loss.mean()
    torch.testing.assert_close(loss1, loss2, atol=1e-5, rtol=1e-5)
    loss1.backward()
    loss2.backward()

    # Verify gradients match across all trainable parameters
    named1 = dict(m1.named_parameters())
    named2 = dict(m2.named_parameters())
    for name, p1 in named1.items():
        p2 = named2[name]
        assert (p1.grad is None) == (p2.grad is None), name
        if p1.grad is not None:
            torch.testing.assert_close(p1.grad, p2.grad, atol=1e-5, rtol=1e-5)
    for item in fields(s1):
        v1, v2 = getattr(s1, item.name), getattr(s2, item.name)
        tensors1 = v1 if isinstance(v1, tuple) else (v1,)
        tensors2 = v2 if isinstance(v2, tuple) else (v2,)
        for t1, t2 in zip(tensors1, tensors2):
            assert (t1.grad is None) == (t2.grad is None), item.name
            if t1.grad is not None:
                torch.testing.assert_close(t1.grad, t2.grad, atol=1e-5, rtol=1e-5)
    assert s1.h.grad is not None, "Checkpoint must preserve the whole-window state credit path"


