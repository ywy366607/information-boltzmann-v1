"""Exact source-resolved filter derivatives, cuts and native action credit."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_fly_dan_credit import prepared, tick
from test_fly_dan_release import setup
from information_boltzmann.core.fly_dan_release import DAN_RELEASE_STATE_FIELDS, dan_arrival
from information_boltzmann.core.fly_dan_credit import DAN_CREDIT_FIELDS, DAN_CREDIT_PARAMETERS, carried_gradients
from information_boltzmann.core.fly_dan_receiver_credit import (
    attach_receiver_credit, receiver_state, advance_receiver_credit, edge_arrival_credit,
)
from information_boltzmann.core.fly_carried_credit import open_window, apply_carried_credit, attach_carried_credit


def configuration(tmp_path):
    m, s = prepared(tmp_path)
    # Two different sources converge on one recipient; their material history
    # must remain distinguishable even though forward concentration is summed.
    m.dan_edge_post[1] = 3
    m.danl_out_2_post[0] = 3
    m.danl_out_2_target[0] = 3
    attach_receiver_credit(m)
    return m, replace(s, dan_credit_dopamine=receiver_state(m, s))


def advance(m, s, i):
    after = tick(m, s, i)
    values = [getattr(after, n) for n in DAN_RELEASE_STATE_FIELDS + DAN_CREDIT_FIELDS]
    arrival = dan_arrival(m, values)
    f, slow = m.dopamine_logit_fast.sigmoid(), m.dopamine_logit_slow.sigmoid()
    conc = f*s.dopamine[:1] + (1-f)*arrival
    dopamine = torch.cat((conc, slow*s.dopamine[1:] + (1-slow)*conc), 0)
    return replace(after, dopamine=dopamine, dan_credit_dopamine=advance_receiver_credit(m, s, values))


def loss(state):
    return (state.dopamine*state.h.new_tensor([[1.,2.,3.,4.,5.,6.], [6.,5.,4.,3.,2.,1.]])).sum()


def test_converging_sources_true_delays_and_full_unroll_match_cut_gradient(tmp_path):
    m, s = configuration(tmp_path)
    end = s
    for i in range(9):
        end = advance(m, end, i)
    exact = torch.autograd.grad(loss(end), [getattr(m, n) for n in DAN_CREDIT_PARAMETERS])
    cut = s
    for i in range(6):
        cut = advance(m, cut, i).detached()
    cut, leaves, start = open_window(cut, m.carried_credit)
    for i in range(6, 9):
        cut = advance(m, cut, i)
    loss(cut).backward()
    G = {n: t.grad for n, t in leaves.items()}
    full_correction = carried_gradients(m, G, start)
    m.dan_receiver_credit_enabled = False
    partial_correction = carried_gradients(m, G, start)
    m.dan_receiver_credit_enabled = True
    assert (full_correction['dan_local_log_release_rate'] - partial_correction['dan_local_log_release_rate']).abs().sum() > 1e-5
    apply_carried_credit(m, leaves, start)
    for n, expected in zip(DAN_CREDIT_PARAMETERS, exact):
        torch.testing.assert_close(getattr(m, n).grad, expected, rtol=3e-7, atol=2e-10)
    # Lag1 maps two physical sites onto one edge; lag2 onto the other edge.
    packets = torch.zeros(4, 7, 3, dtype=torch.double)
    packets[0, 0, 0] = 2.
    packets[1, 1, 2] = 3.
    arrival = edge_arrival_credit(m, packets)
    torch.testing.assert_close(arrival[0], torch.tensor([.2, 0.], dtype=torch.double))
    torch.testing.assert_close(arrival[1], torch.tensor([0., .9], dtype=torch.double))


@pytest.mark.parametrize('host', [False, True])
@pytest.mark.parametrize('decision', ['timing', 'content'])
def test_past_receiver_history_reaches_action_score_without_rewriting_CE_gradients(tmp_path, host, decision):
    from information_boltzmann.core.fly_native_policy_credit import native_policy_credit
    m, s = configuration(tmp_path)
    for i in range(4):
        s = advance(m, s, i).detached()
    opened, leaves, start = open_window(s, m.carried_credit)
    m.native_motor_port_enabled = False
    hazard = opened.dopamine[0, 3].sigmoid().reshape(1)
    reads = torch.zeros(1, m.decoder.in_features, dtype=torch.double)
    reads[:, 0] = opened.dopamine[0, 3]
    if decision == 'timing':
        score = hazard[0].log()
        event = {'kind':'timing','start':0,'length':1,'first':0,'tail_ticks':0}
    else:
        score = m.decoder(m.read_norm(reads)).float().log_softmax(-1)[0, 0]
        event = {'kind':'content','tick':0,'word':0,'temperature':1.0,'tail_ticks':0}
    G = torch.autograd.grad(score, tuple(leaves.values()), allow_unused=True, retain_graph=True)
    expected = carried_gradients(m, dict(zip(leaves, G)), start)
    obj = SimpleNamespace(model=m, _credit=(leaves, start), consequence_state={}, native_credit_cpu=host)
    native_policy_credit(obj, [{'kind':'clock','ticks':1}, event], hazard, reads, [1])
    assert expected['dan_local_log_release_rate'].abs().sum() > 0
    for n, value in expected.items():
        torch.testing.assert_close(obj.consequence_state['native_credit']['E'][n].cpu(), value.cpu())
        assert getattr(m, n).grad is None
    assert all(t.grad is None for t in leaves.values())


def test_partial_or_wrong_identity_history_rejected_and_source_off_decays_not_erases(tmp_path):
    m, s = configuration(tmp_path)
    with pytest.raises(ValueError, match='Malformed'):
        receiver_state(m, replace(s, dan_credit_dopamine=torch.zeros(2,7,3)))
    old = torch.ones_like(s.dan_credit_dopamine)
    s = replace(s, dan_credit_dopamine=old)
    m.dan_local_source_enabled = False
    evolved = advance_receiver_credit(m, s, None)
    assert torch.all(evolved > 0) and torch.all(evolved < old)


def test_one_token_recompute_and_complete_credit_roundtrip(tmp_path):
    from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner
    m, s, _ = setup(tmp_path)
    m.dan_carried_credit_enabled = True
    attach_carried_credit(m, ('dopamine',))
    attach_receiver_credit(m)
    a = FlyBPTTLearner(m, copy.deepcopy(s), settle_ticks=3, use_checkpointing=False)
    b = FlyBPTTLearner(copy.deepcopy(m), copy.deepcopy(s), settle_ticks=3, use_checkpointing=True)
    a.previous_token = b.previous_token = 0
    for word in (1, 0):
        la,_ = a.observe([word]); lb,_ = b.observe([word])
        torch.testing.assert_close(torch.tensor(la), torch.tensor(lb))
        torch.testing.assert_close(a.state.dan_credit_dopamine, b.state.dan_credit_dopamine)
    path = tmp_path/'credit.pt'
    torch.save(a.state_dict()['physical'], path)
    restored = type(s)(**torch.load(path, weights_only=True))
    torch.testing.assert_close(restored.dan_credit_dopamine, a.state.dan_credit_dopamine, rtol=0, atol=0)
