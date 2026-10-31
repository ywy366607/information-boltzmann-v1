"""Sensory terminal returns: actual contacts, local shunting, site delivery, exact point limit."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from information_boltzmann.core.fly_sensory_terminals import (
    attach_sensory_terminals, advance_sensory_terminals, sensory_delivery, terminal_values,
    SENSORY_TERMINAL_PARAMETER_NAMES,
)

N = 6  # cells 0,1 sensory; 2 inhibitory interneuron; 3 excitatory; 4,5 targets


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_neurons = N
        # e list: sensory 0->4, sensory 1->5, 3->4 ; i list: 2->5
        self.register_buffer('edge_pre_e', torch.tensor([0, 1, 3]))
        self.register_buffer('edge_post_e', torch.tensor([4, 5, 4]))
        self.edge_weight_e = nn.Parameter(torch.tensor([.3, .5, .7]))
        self.register_buffer('edge_pre_i', torch.tensor([2]))
        self.register_buffer('edge_post_i', torch.tensor([5]))
        self.edge_weight_i = nn.Parameter(torch.tensor([.4]))

    def get_decay_rates(self):
        return (torch.full((1, N), .9), torch.full((1, N), .5), torch.full((1, N), .6))

    def get_conductance_gains(self):
        return (torch.full((1, N), 2.), torch.full((1, N), 3.))


def archive(tmp_path):
    # cell 0: nodes 0-1-2 (chain), sites at nodes 0 and 2; cell 1: nodes 3-4, site at node 4
    data = dict(
        cell_index=np.array([0, 1]), cell_body_ids=np.array([100, 101]),
        volume=np.ones(5), u=np.array([1, 2, 4], np.int32), v=np.array([0, 1, 3], np.int32),
        g=np.ones(3), owner=np.array([0, 0, 0, 1, 1], np.int32),
        density=np.array([3, 3, 3, 2, 2], np.float32),
        site_node=np.array([0, 2, 4], np.int32), site_owner=np.array([0, 0, 1], np.int32),
        # inhibitory cell 2 -> node 2 of cell 0 (lag 1); sensory cell 1's site 2 -> node 1 of cell 0 (lag 2, excitatory)
        in_node=np.array([2, 1], np.int32), in_source=np.array([2, 1], np.int32),
        in_site=np.array([-1, 2], np.int32), in_delay=np.array([1, 2], np.int8),
        in_kind=np.array([1, 0], np.int8), in_weight=np.array([.0065, .0018], np.float32),
        # edge e0 (0->4) leaves from sites 0 and 1 (lag 1), edge e1 (1->5) from site 2 (lag 3)
        out_edge=np.array([0, 0, 1], np.int32), out_channel=np.array([0, 0, 0], np.int8),
        out_site=np.array([0, 1, 2], np.int32), out_delay=np.array([1, 1, 3], np.int8),
        out_share=np.array([.5, .5, 1.], np.float32),
        verify_e_edge=np.array([0, 1], np.int32), verify_i_edge=np.array([], np.int32),
        verify_e_pre=np.array([0, 1]), verify_e_post=np.array([4, 5]),
        verify_i_pre=np.array([], np.int64), verify_i_post=np.array([], np.int64))
    path = tmp_path / 'st.npz'
    np.savez(path, **data)
    return path


def setup(tmp_path):
    model = Tiny()
    attach_sensory_terminals(model, archive(tmp_path))
    ring = tuple(torch.zeros(1, N) for _ in range(4))
    state = SimpleNamespace(h=torch.zeros(1, N), ring=ring, sensory_terminal_signal=torch.empty(0),
                            sensory_terminal_packets=torch.empty(0))
    return model, state


def point_delivery(model, ring):
    out = torch.zeros(1, N)
    for lag, (e,) in ((1, (0,)), (3, (1,))):
        out[0, model.edge_post_e[e]] += model.edge_weight_e[e] * ring[lag - 1][0, model.edge_pre_e[e]]
    return out


def test_point_limit_exact_without_terminal_input(tmp_path):
    model, state = setup(tmp_path)
    ring = tuple(torch.rand(1, N) for _ in range(4))
    state.ring = ring
    _, packets = terminal_values(model, state)
    ge, gi = sensory_delivery(model, packets)
    torch.testing.assert_close(ge, point_delivery(model, ring).detach())
    assert torch.count_nonzero(gi) == 0
    model.sensory_terminal_input_enabled = False
    signal, packets = advance_sensory_terminals(model, state, ring[0] * 0 + 1.)
    assert torch.count_nonzero(signal) == 0
    torch.testing.assert_close(packets[0], torch.ones(3))
    torch.testing.assert_close(packets[1:], torch.stack([ring[k][0, [0, 0, 1]] for k in range(3)]))


def test_local_inhibition_shunts_nearby_sites_and_all_materials_learn(tmp_path):
    model, state = setup(tmp_path)
    ring = [torch.zeros(1, N) for _ in range(4)]
    ring[0][0, 2] = 1.  # the inhibitory interneuron fired one tick ago (lag 1)
    state.ring = tuple(ring)
    signal, packets = advance_sensory_terminals(model, state, torch.ones(1, N))
    assert signal[1, 2] > signal[1, 1] > signal[1, 0] > 0      # spreads from node 2 along the tree
    assert signal[1, 3] == signal[1, 4] == 0                   # never crosses into another cell
    assert torch.count_nonzero(signal[0]) == 0
    assert packets[0, 1] < packets[0, 0] < 1                   # the site at node 2 is shunted most
    assert packets[0, 2] == 1                                   # other cell untouched
    params = [getattr(model, n) for n in SENSORY_TERMINAL_PARAMETER_NAMES]
    grads = torch.autograd.grad(packets[0].sum(), params)
    assert all(bool(torch.isfinite(g).all()) for g in grads)
    assert grads[0][1, 0] != 0 and grads[1][1, 0] != 0          # inhibitory transport and gain of cell 0
    model.sensory_terminal_gating_enabled = False
    _, ungated = advance_sensory_terminals(model, state, torch.ones(1, N))
    torch.testing.assert_close(ungated[0], torch.ones(3))


def test_native_source_reads_its_actual_release_site(tmp_path):
    model, state = setup(tmp_path)
    signal, packets = terminal_values(model, state)
    packets = packets.clone()
    packets[1, 2] = 1.  # cell 1's site released two ticks ago
    state.sensory_terminal_signal, state.sensory_terminal_packets = signal, packets
    new_signal, _ = advance_sensory_terminals(model, state, torch.zeros(1, N))
    assert new_signal[0, 1] > 0 and torch.count_nonzero(new_signal[1]) == 0
    packets[1, 2] = 0.
    packets[1, 0] = 5.  # a different site of another cell is not this contact's source
    new_signal, _ = advance_sensory_terminals(model, state, torch.zeros(1, N))
    assert torch.count_nonzero(new_signal) == 0


def test_refuses_mismatched_anatomy_and_edge_order(tmp_path):
    model = Tiny()
    path = archive(tmp_path)
    saved = {'sensory_terminal_log_gain': torch.zeros(2, 2), 'sensory_terminal_digest': torch.zeros(32, dtype=torch.uint8)}
    with pytest.raises(ValueError, match='anatomy differs'):
        attach_sensory_terminals(model, path, saved=saved)
    model = Tiny()
    model.edge_post_e = torch.tensor([5, 4, 4])
    with pytest.raises(ValueError, match='edge order'):
        attach_sensory_terminals(model, path)


@pytest.mark.parametrize('checkpointing', [False, True])
def test_learner_carries_terminal_state_and_trains_its_materials(tmp_path, checkpointing):
    from test_fly_bptt_learning import make_model, physical
    from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner
    from information_boltzmann.core.fly_native_policy_credit import native_parameter_names
    from information_boltzmann.core.fly_native_motor_port import checkpoint_model_weights
    model = make_model(tmp_path)
    pre, post = model.edge_pre_e.numpy(), model.edge_post_e.numpy()
    out_edge = int(np.flatnonzero(pre == 0)[0])
    path = tmp_path / 'terminals.npz'
    np.savez(path, cell_index=np.array([0]), cell_body_ids=np.array([0]),
             volume=np.ones(3), u=np.array([1, 2], np.int32), v=np.array([0, 1], np.int32), g=np.ones(2),
             owner=np.zeros(3, np.int32), density=np.full(3, 3, np.float32),
             site_node=np.array([0, 2], np.int32), site_owner=np.zeros(2, np.int32),
             in_node=np.array([2, 0], np.int32), in_source=np.array([3, 1], np.int32),
             in_site=np.array([-1, -1], np.int32), in_delay=np.array([1, 1], np.int8),
             in_kind=np.array([1, 0], np.int8), in_weight=np.array([.5, .5], np.float32),
             out_edge=np.array([out_edge, out_edge], np.int32), out_channel=np.zeros(2, np.int8),
             out_site=np.array([0, 1], np.int32), out_delay=np.ones(2, np.int8),
             out_share=np.full(2, .5, np.float32),
             verify_e_edge=np.array([out_edge], np.int32), verify_i_edge=np.array([], np.int32),
             verify_e_pre=pre[[out_edge]], verify_e_post=post[[out_edge]],
             verify_i_pre=np.array([], np.int64), verify_i_post=np.array([], np.int64))
    attach_sensory_terminals(model, path)
    model.sensory_terminal_enabled = True
    state = physical(model)
    learner = FlyBPTTLearner(model, state, use_checkpointing=checkpointing, settle_ticks=3)
    learner.previous_token = 0
    learner.observe([1, 2, 3])
    assert learner.state.sensory_terminal_signal.shape == (2, 3)
    assert learner.state.sensory_terminal_packets.shape == (4, 2)
    assert not learner.state.sensory_terminal_signal.requires_grad
    assert set(SENSORY_TERMINAL_PARAMETER_NAMES) <= set(native_parameter_names(model))
    assert 'sensory_terminal_digest' in checkpoint_model_weights(model)
    for name in SENSORY_TERMINAL_PARAMETER_NAMES:
        assert getattr(model, name) in learner.optimizer.state
