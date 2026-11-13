"""Local program/derivative contracts only; no capability training."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.ib_local.pretrained_gdn_experience import LayerExperienceRecall, LocalExperience


def memory():
    torch.manual_seed(41)
    wrapper = LayerExperienceRecall(6, capacity=2, latent_dim=3, topk=2, read_interval=1)
    wrapper.memory.clock.fill_(512)
    wrapper.reactivation_mode = 'complete_attention'
    wrapper.capture_full_window = True
    for i in range(2):
        x = torch.randn(6)
        wrapper.memory.commit_episode(x, x+1, x+2, x+3, wrapper.encoder, source_time=64+i)
        wrapper.capsules[i] = LocalExperience(torch.randn(1, 65, 6), None,
            int(wrapper.memory.read_credit_version[i]), torch.randn(1, 63, 6))
    return wrapper


def test_complete_sequence_changes_with_query_and_reaches_tail_parameter():
    wrapper = memory()
    query = torch.randn(2, 6, requires_grad=True)
    gain = nn.Parameter(torch.ones(128))
    def replay(capsule):
        return capsule.complete_inputs * gain[None, :, None]
    value = wrapper.retrieve(query, replay)
    assert not torch.allclose(value[0], value[1])
    value.square().sum().backward()
    assert gain.grad[65:].norm() > 0
    assert query.grad.norm() > 0
    assert all(x.complete_inputs.grad_fn is None for x in wrapper.capsules)


def test_held_keeps_value_and_query_encoder_gradients_but_cuts_native():
    wrapper = memory()
    query = torch.randn(2, 6, requires_grad=True)
    gain = nn.Parameter(torch.tensor(.7))
    replay = lambda capsule: torch.tanh(gain * capsule.complete_inputs)
    full = wrapper.retrieve(query, replay)
    full_grad = torch.autograd.grad(full.square().sum(), (gain, query, wrapper.encoder.weight))
    wrapper.detach_replay = True
    held = wrapper.retrieve(query, replay)
    held_grad = torch.autograd.grad(held.square().sum(), (gain, query, wrapper.encoder.weight), allow_unused=True)
    torch.testing.assert_close(full, held, atol=0, rtol=0)
    assert full_grad[0].abs() > 0 and held_grad[0] is None
    assert held_grad[1].norm() > 0 and held_grad[2].norm() > 0
    torch.testing.assert_close(full_grad[1], held_grad[1])
    torch.testing.assert_close(full_grad[2], held_grad[2])


def test_attach_executes_every_stored_position_and_future_query_cannot_change_prefix():
    wrapper = memory()
    lengths = []
    class Program(nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = nn.Parameter(torch.ones(128))
        def forward(self, hidden_states, **kwargs):
            lengths.append(hidden_states.shape[1])
            return ((hidden_states * self.gain[None, :hidden_states.shape[1], None]).cumsum(1),)
    program = Program()
    model = SimpleNamespace(model=SimpleNamespace(layers=[program]))
    nn.init.normal_(wrapper.inlet.projection.weight, std=.1)
    x = torch.randn(1, 4, 6)
    with wrapper.attach(model, layer=0, capture=False):
        a = program(x)[0]
    changed = x.clone()
    changed[:, 2:] += 10
    with wrapper.attach(model, layer=0, capture=False):
        b = program(changed)[0]
    torch.testing.assert_close(a[:, :2], b[:, :2], atol=0, rtol=0)
    assert lengths == [128, 128, 4, 128, 128, 4]


def test_legacy_capsule_is_not_padded_and_mode_round_trips():
    wrapper = memory()
    wrapper.capsules[0].continuation = None
    lengths = []
    def replay(capsule):
        lengths.append(capsule.complete_inputs.shape[1])
        return capsule.complete_inputs
    wrapper.retrieve(torch.randn(1, 6), replay)
    assert sorted(lengths) == [65, 128]
    saved = copy.deepcopy(wrapper.state_dict())
    restored = memory()
    restored.load_state_dict(saved)
    assert restored.reactivation_mode == 'complete_attention'
    del saved['_extra_state']['reactivation_mode']
    restored.load_state_dict(saved)
    assert restored.reactivation_mode == 'legacy_endpoint'


def test_complete_reader_rejects_silently_truncated_callback():
    wrapper = memory()
    with pytest.raises(ValueError, match='all actual65..128'):
        wrapper.retrieve(torch.randn(1, 6), lambda capsule: capsule.inputs)


def test_old_cut_uses_latest_participation_not_midpoint_key_age():
    wrapper = memory()
    # key at64; full span ends127. At time220, midpoint is156 old but
    # latest input is only93 old: do not cut the mixed-age old program.
    wrapper.memory.clock.fill_(220)
    wrapper.memory.birth[:] = 64
    query = torch.randn(1, 6)
    gain = nn.Parameter(torch.tensor(.7))
    replay = lambda capsule: gain * capsule.complete_inputs
    full = wrapper.retrieve(query, replay)
    full_grad, = torch.autograd.grad(full.sum(), gain)
    wrapper.detach_replay_older_than = 128
    mixed = wrapper.retrieve(query, replay)
    mixed_grad, = torch.autograd.grad(mixed.sum(), gain)
    torch.testing.assert_close(full, mixed, atol=0, rtol=0)
    torch.testing.assert_close(full_grad, mixed_grad)
    wrapper.query_clock_offsets = torch.tensor([100])
    old = wrapper.retrieve(query, replay)
    old_grad, = torch.autograd.grad(old.sum(), gain, allow_unused=True)
    assert old_grad is None or old_grad == 0


def test_frozen_response_cache_never_caches_the_current_query_result():
    from scripts.ib_local.evaluate_pretrained_gdn_ruler import frozen_decode_replays
    wrapper = memory()
    learner = SimpleNamespace(memories={0: wrapper})
    a, b = torch.randn(1,6), torch.randn(1,6)
    calls = []
    def replay(capsule):
        calls.append(id(capsule))
        return capsule.complete_inputs.tanh()
    with torch.no_grad():
        expected_a = wrapper.retrieve(a, replay)
        expected_b = wrapper.retrieve(b, replay)
        calls.clear()
        with frozen_decode_replays(learner):
            actual_a = wrapper.retrieve(a, replay)
            actual_b = wrapper.retrieve(b, replay)
    torch.testing.assert_close(actual_a, expected_a, atol=0, rtol=0)
    torch.testing.assert_close(actual_b, expected_b, atol=0, rtol=0)
    assert not torch.allclose(actual_a, actual_b)
    assert len(calls) == 2 and len(set(calls)) == 2


def test_migration_adds_no_parameters_and_does_not_reset_existing_state():
    from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
    wrapper = memory()
    wrapper.reactivation_mode = 'legacy_endpoint'
    wrapper.capture_full_window = False
    holder = SimpleNamespace(memories={0: wrapper})
    holder.enable_full_window_participation = lambda: PretrainedExperienceLearner.enable_full_window_participation(holder)
    parameters = [(id(p), p.detach().clone()) for p in wrapper.parameters()]
    clock = wrapper.memory.clock.clone()
    slots = list(wrapper.capsules)
    PretrainedExperienceLearner.enable_complete_reactivation(holder)
    assert wrapper.reactivation_mode == 'complete_attention' and wrapper.capture_full_window
    assert slots == wrapper.capsules
    torch.testing.assert_close(wrapper.memory.clock, clock, atol=0, rtol=0)
    for (identity, value), actual in zip(parameters, wrapper.parameters()):
        assert identity == id(actual)
        torch.testing.assert_close(value, actual, atol=0, rtol=0)


def test_disk_roundtrip_preserves_next_read_and_next_committed_record(tmp_path):
    wrapper = memory()
    path = tmp_path/'local_read.pt'
    torch.save(wrapper.state_dict(),path)
    restored = memory()
    restored.load_state_dict(torch.load(path,map_location='cpu',weights_only=False))
    query = torch.randn(2,6)
    replay = lambda capsule:capsule.complete_inputs.tanh()
    torch.testing.assert_close(wrapper.retrieve(query,replay),restored.retrieve(query,replay),atol=0,rtol=0)
    incoming = torch.randn(1,128,6)
    committed = []
    for value in (wrapper,restored):
        value.current=dict(inputs=incoming,source=incoming,response=incoming.tanh(),boundary=None,capture=True)
        committed.append(value.commit())
    assert committed[0] == committed[1]
    torch.testing.assert_close(wrapper.retrieve(query,replay),restored.retrieve(query,replay),atol=0,rtol=0)
