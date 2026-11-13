"""CPU wiring tests for the opt-in learner; no pretrained model or CUDA load."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from scripts.ib_local.native_s_eligibility import NativeSEligibility
from scripts.ib_local.native_s_learner import (
    NativeSEligibilityLearner, apply_master_credit, mismatch_credit,
)
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner


def optimizer_fixture():
    p = nn.Parameter(torch.tensor([1., 2.], dtype=torch.float16))
    master = nn.Parameter(p.detach().float().clone())
    opt = torch.optim.AdamW([master], lr=.01, weight_decay=0., foreach=False)
    return SimpleNamespace(native=[('layer1.a_proj.weight', p)], masters=[master],
                           extras=[], null_parameters=[], optimizer=opt)


def test_zero_addition_matches_original_optimizer_including_none_semantics():
    a, b = optimizer_fixture(), optimizer_fixture()
    for learner in (a, b):
        learner.native[0][1].grad = torch.tensor([.2, .3], dtype=torch.float16)
    PretrainedExperienceLearner.apply_gradients(a)
    apply_master_credit(b, {'layer1.a_proj.weight': np.zeros(2, np.float32)})
    torch.testing.assert_close(a.masters[0], b.masters[0], rtol=0, atol=0)
    for learner in (a, b):
        learner.native[0][1].grad = None
    before = a.masters[0].detach().clone()
    PretrainedExperienceLearner.apply_gradients(a)
    apply_master_credit(b, {'layer1.a_proj.weight': np.zeros(2, np.float32)})
    torch.testing.assert_close(a.masters[0], before, rtol=0, atol=0)
    torch.testing.assert_close(a.masters[0], b.masters[0], rtol=0, atol=0)
    assert b.optimizer.state[b.masters[0]]['step'].item() == 1


def test_historical_addition_is_fp32_before_the_single_clip_and_adam():
    l = optimizer_fixture(); p = l.native[0][1]
    p.grad = torch.tensor([.25, .25], dtype=torch.float16)
    steps = []
    step = l.optimizer.step
    l.optimizer.step = lambda: (steps.append(1), step())[1]
    report = apply_master_credit(l, {'layer1.a_proj.weight': np.full(2, 1e-6, np.float32)})
    torch.testing.assert_close(l.masters[0].grad, torch.full((2,), .250001), rtol=0, atol=1e-8)
    assert steps == [1] and report['optimizer_steps'] == 1
    l = optimizer_fixture()
    l.native[0][1].grad = None
    report = apply_master_credit(l, {'layer1.a_proj.weight': np.array([3., 4.], np.float32)})
    assert report['combined_preclip_norm'] == 5
    torch.testing.assert_close(l.masters[0].grad, torch.tensor([.6, .8]), atol=3e-7, rtol=0)


def test_family_mismatch_preserves_layer_tensor_and_weight_row_norms():
    raw = {'layer1.a_proj.weight': np.arange(12, dtype=np.float32).reshape(3, 4),
           'layer12.A_log': np.arange(4, dtype=np.float32)}
    wrong = mismatch_credit(raw)
    assert wrong.keys() == raw.keys()
    for name, value in raw.items():
        np.testing.assert_allclose(np.linalg.norm(wrong[name]), np.linalg.norm(value))
        assert not np.array_equal(wrong[name], value)
    np.testing.assert_array_equal(np.linalg.norm(wrong['layer1.a_proj.weight'], axis=1),
                                  np.linalg.norm(raw['layer1.a_proj.weight'], axis=1))


def test_bad_master_receiver_is_not_silently_accepted():
    l = optimizer_fixture()
    with pytest.raises(ValueError, match='names'):
        apply_master_credit(l, {'unknown': np.ones(1)})
    with pytest.raises(ValueError, match='shape'):
        apply_master_credit(l, {'layer1.a_proj.weight': np.ones(3)})


class TinyConv(nn.Module):
    def forward(self, x, **kwargs):
        return x.tanh(), None


class TinyAttention(nn.Module):
    def __init__(self, layer, detach_state=False):
        super().__init__()
        self.layer = layer
        self.a_proj = nn.Linear(2, 1, bias=False)
        self.b_proj = nn.Linear(2, 1, bias=False)
        self.k_conv1d, self.v_conv1d = TinyConv(), TinyConv()
        self.A_log = nn.Parameter(torch.tensor([-1.]))
        self.dt_bias = nn.Parameter(torch.tensor([.1]))
        self.head_k_dim = self.head_v_dim = 2
        self.num_heads = self.num_v_heads = 1
        self.use_short_conv, self.mode = True, 'chunk'
        self.detach_state = detach_state

    def forward(self, hidden_states, past_key_values=None, use_cache=True):
        self.k_conv1d(hidden_states); self.v_conv1d(hidden_states)
        a, b = self.a_proj(hidden_states), self.b_proj(hidden_states)
        state = past_key_values.states[self.layer]['recurrent_state']
        contribution = (state.detach() if self.detach_state else state).sum((1, 2))
        return hidden_states + contribution[:, None, :] + (a + b) * torch.tensor([.1, -.1])


class TinyModel(nn.Module):
    def __init__(self, detach_layer12=False):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module() for _ in range(13)])
        for layer in (1, 12):
            self.model.layers[layer].attn = TinyAttention(layer, detach_layer12 and layer == 12)

    def forward(self, x, past_key_values=None, use_cache=True):
        for layer in (1, 12):
            x = self.model.layers[layer].attn(x, past_key_values=past_key_values, use_cache=use_cache)
        return SimpleNamespace(logits=x, past_key_values=past_key_values)


def tiny_adapter(detach_layer12=False):
    learner = NativeSEligibilityLearner.__new__(NativeSEligibilityLearner)
    learner.model = TinyModel(detach_layer12)
    learner.layers = (1, 12)
    learner.memories = {l: SimpleNamespace(_replaying=False) for l in learner.layers}
    learner.native = [(f'layer{l}.{n}', p) for l in learner.layers
                      for n, p in learner.model.model.layers[l].attn.named_parameters()]
    states = [{} for _ in range(13)]
    for l in learner.layers:
        states[l] = {'recurrent_state': torch.randn(1, 1, 2, 2)}
    learner.cache = SimpleNamespace(states=states, _seen_tokens=69)
    learner.updates = 3
    learner.eligibility = None
    learner._native_capture_started = False
    learner._native_actual_count = 0
    learner._native_window_version = None
    learner._native_extra = None
    learner._last_native_s = learner._empty_report()
    learner.enable_native_s_eligibility(block_size=16)
    return learner


def test_old_partial_window_is_not_backfilled_and_replay_is_not_actual_capture():
    l = tiny_adapter()
    with l.factual_capture(69):
        pass
    assert l.eligibility.pending is None
    l.cache._seen_tokens = 128
    x = torch.randn(1, 2, 2)
    with l.factual_capture(0):
        for layer in l.layers:
            l.memories[layer]._replaying = True
            l.model.model.layers[layer].attn(x, past_key_values=l.cache)
            l.memories[layer]._replaying = False
            l.model.model.layers[layer].attn(x, past_key_values=l.cache)
    assert l.eligibility.pending['counts'] == {1: 2, 12: 2}
    for layer in l.layers:
        packet = l.eligibility.pending['chunks'][layer][0]
        assert packet['branch'] == 'recurrent'
        np.testing.assert_allclose(packet['x'], x[0].numpy())
    assert l._temporary_bytes() == l.eligibility.estimated_temporary_bytes


def test_real_current_ce_reaches_private_s_without_accumulating_other_gradients():
    l = tiny_adapter(); l.eligibility.code.U[0, 0] = 1; l.eligibility.code.V[0] = 1
    target = torch.tensor([[0, 1]])
    with l._teaching_context(target, None):
        output = l.model(torch.randn(1, 2, 2), past_key_values=l.cache)
    assert l._last_native_s['teaching_available']
    assert l._last_native_s['old_credit_norm'] > 0
    assert all(p.grad is None for p in l.model.parameters())
    F = torch.nn.functional
    F.cross_entropy(output.logits.reshape(-1, 2), target.reshape(-1)).backward()
    assert any(p.grad is not None for p in l.model.parameters())


def test_unreachable_s_is_an_admission_failure_not_zero_credit():
    l = tiny_adapter(detach_layer12=True)
    with pytest.raises(RuntimeError, match='cannot reach'):
        with l._teaching_context(torch.tensor([[0, 1]]), None):
            l.model(torch.randn(1, 2, 2), past_key_values=l.cache)
