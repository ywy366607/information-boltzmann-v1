"""Conditional derivatives and credit continuity, not capability studies."""
import copy
from dataclasses import replace

import pytest
import torch

from test_fly_dan_release import setup
from information_boltzmann.core.fly_dan_release import DAN_RELEASE_STATE_FIELDS, advance_dan_release
from information_boltzmann.core.fly_dan_credit import (
    DAN_CREDIT_FIELDS, DAN_CREDIT_PARAMETERS, credit_state, advance_credit, carried_gradients,
)
from information_boltzmann.core.fly_carried_credit import open_window, apply_carried_credit, attach_carried_credit
from information_boltzmann.core.fly_kc_terminals import terminal_state


def prepared(tmp_path):
    m, s, _ = setup(tmp_path)
    m = m.double()
    s = type(s)(**{k: tuple(t.double() for t in v) if k == 'ring' else v.double()
                  for k, v in s.state_dict().items()})
    m.dan_carried_credit_enabled = True
    attach_carried_credit(m, ('dopamine',))
    with torch.no_grad():
        m.dan_local_log_transport.fill_(-.4)
        m.dan_local_log_affinity.fill_(.2)
        m.dan_local_log_off.fill_(-1.3)
        m.dan_local_logit_calcium_decay.fill_(.3)
        m.dan_local_logit_resource_recovery.fill_(1.1)
        m.dan_local_log_release_rate.fill_(-.7)
        m.superclass_id[2] = m.superclass_id[0]  # two DANs share a gain parameter
    s = replace(s, **dict(zip(DAN_CREDIT_FIELDS, credit_state(m, s))))
    return m, s


def tick(m, s, i):
    kc = terminal_state(m, s)
    kc[3] = s.h.new_tensor([[.7, .1, .4], [.2, 0, .8], [.3, .6, 0], [0, .1, .2]])
    ring = list(s.ring)
    ring[0] = s.h.new_tensor([[i % 2, 0., (i+1) % 2, 0., 0., 0.]])
    values = advance_dan_release(m, s, ring, kc)
    sensitivity = advance_credit(m, s, ring, kc, values)
    return replace(s, **dict(zip(DAN_RELEASE_STATE_FIELDS, values)),
                   **dict(zip(DAN_CREDIT_FIELDS, sensitivity)))


def objective(s):
    return sum(((i+1)*getattr(s, n)*torch.arange(1, getattr(s, n).numel()+1,
                dtype=s.h.dtype).reshape(getattr(s, n).shape)).sum()
               for i, n in enumerate(DAN_RELEASE_STATE_FIELDS))


def test_full_spatial_tangents_match_unrolled_autograd_and_chunk_contraction(tmp_path):
    m, start = prepared(tmp_path)
    full = start
    for i in range(8):
        full = tick(m, full, i)
    exact = torch.autograd.grad(objective(full), [getattr(m, n) for n in DAN_CREDIT_PARAMETERS])
    end = replace(full, **{n: getattr(full, n).detach().requires_grad_() for n in DAN_RELEASE_STATE_FIELDS})
    G = torch.autograd.grad(objective(end), [getattr(end, n) for n in DAN_RELEASE_STATE_FIELDS])
    tangents = carried_gradients(m, dict(zip(DAN_RELEASE_STATE_FIELDS, G)), full.state_dict())
    for n, value in zip(DAN_CREDIT_PARAMETERS, exact):
        torch.testing.assert_close(tangents[n], value, rtol=2e-7, atol=2e-9)
    state = start
    for i in range(5):
        state = tick(m, state, i).detached()
    state, leaves, history = open_window(state, m.carried_credit)
    for i in range(5, 8):
        state = tick(m, state, i)
    objective(state).backward()
    diagnostics = apply_carried_credit(m, leaves, history)
    assert diagnostics['dan_local_logit_resource_recovery'] > 0
    for n, value in zip(DAN_CREDIT_PARAMETERS, exact):
        torch.testing.assert_close(getattr(m, n).grad, value, rtol=2e-7, atol=2e-9)


def test_history_roundtrip_and_fail_closed_partial_state(tmp_path):
    m, s = prepared(tmp_path)
    s = tick(m, s, 0).detached()
    path = tmp_path/'history.pt'
    torch.save(s.state_dict(), path)
    restored = type(s)(**torch.load(path, weights_only=True))
    a, b = tick(m, s, 1), tick(m, restored, 1)
    for name in DAN_CREDIT_FIELDS + DAN_RELEASE_STATE_FIELDS:
        torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)
    with pytest.raises(ValueError, match='Incomplete'):
        credit_state(m, replace(s, dan_credit_packets=torch.empty(0)))


def test_one_token_checkpoint_recompute_carries_and_restores_credit(tmp_path):
    from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner
    m, s, _ = setup(tmp_path)
    m.dan_carried_credit_enabled = True
    attach_carried_credit(m, ('dopamine',))
    a = FlyBPTTLearner(m, copy.deepcopy(s), settle_ticks=3, use_checkpointing=False)
    b = FlyBPTTLearner(copy.deepcopy(m), copy.deepcopy(s), settle_ticks=3, use_checkpointing=True)
    a.previous_token = b.previous_token = 0
    for word in (1, 0):
        la, _ = a.observe([word]); lb, _ = b.observe([word])
        torch.testing.assert_close(torch.tensor(la), torch.tensor(lb))
        for name in DAN_CREDIT_FIELDS + DAN_RELEASE_STATE_FIELDS:
            torch.testing.assert_close(getattr(a.state, name), getattr(b.state, name))
    restored = type(s)(**a.state_dict()['physical'])
    for name in DAN_CREDIT_FIELDS:
        torch.testing.assert_close(getattr(restored, name), getattr(a.state, name), rtol=0, atol=0)


@pytest.mark.parametrize('host', [False, True])
def test_native_action_score_includes_past_chemistry_once_without_touching_CE_slots(tmp_path, host):
    from types import SimpleNamespace
    from information_boltzmann.core.fly_native_policy_credit import native_policy_credit
    m, s = prepared(tmp_path)
    s = tick(m, s, 0).detached()
    opened, leaves, history = open_window(s, m.carried_credit)
    # The timing score depends only on opening resources: there is no direct
    # current-window material path to hide a missing historical contraction.
    hazard = opened.dan_local_resources.mean().sigmoid().reshape(1)
    m.native_motor_port_enabled = False
    obj = SimpleNamespace(model=m, _credit=(leaves, history), consequence_state={}, native_credit_cpu=host)
    score = hazard[0].log()
    G = torch.autograd.grad(score, tuple(leaves.values()), allow_unused=True, retain_graph=True)
    expected = carried_gradients(m, dict(zip(leaves, G)), history)
    events = [{'kind':'clock', 'ticks':1},
              {'kind':'timing', 'start':0, 'length':1, 'first':0, 'tail_ticks':0}]
    native_policy_credit(obj, events, hazard, torch.zeros(1, m.decoder.in_features, dtype=torch.double), [1])
    E = obj.consequence_state['native_credit']['E']
    assert expected['dan_local_log_release_rate'].abs().sum() > 0
    for name, value in expected.items():
        torch.testing.assert_close(E[name].cpu(), value.cpu())
        assert getattr(m, name).grad is None
    assert all(t.grad is None for t in leaves.values())
