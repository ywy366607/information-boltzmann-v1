"""Finite-volume conservation, adjoints and real learner interface contracts."""
import copy
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

from test_fly_mb_terminal import setup as mb_setup
from information_boltzmann.core.fly_apl_cable import (
    CableGeometry, diffuse, attach_apl_cable, cable_state,
    advance_apl_cable, apl_mb_arrival, apply_apl_readout,
)
from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner, FlyPhysicalState
from information_boltzmann.core.fly_mb_terminal import step_mb_terminal
from information_boltzmann.core.fly_native_motor_port import checkpoint_model_weights
from information_boltzmann.core.fly_native_policy_credit import native_parameter_names


def geometry():
    return CableGeometry([1., 2., 1., 1.], [0, 1], [1, 2], [1., 1.], [0, 0, 0, 1])


def test_direct_cable_preserves_mass_constant_voltage_and_disconnected_native_roots():
    g = geometry()
    value = torch.tensor([[1., 0., 0., 0.]], dtype=torch.double)
    result = diffuse(value, torch.zeros(2, dtype=torch.double), g)
    assert result[0, 0] > result[0, 1] > result[0, 2] > 0 and result[0, 3] == 0
    torch.testing.assert_close((result * torch.tensor(g.area)).sum(), value.sum(), rtol=0, atol=1e-14)
    torch.testing.assert_close(diffuse(torch.ones_like(value), torch.zeros(2), g), torch.ones_like(value))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_direct_cable_gradient_matches_dense_implicit_operator(device):
    g = geometry()
    x = torch.tensor([[.4, -.2, .7, .3]], dtype=torch.double, device=device, requires_grad=True)
    material = torch.tensor([.2, -.1], dtype=torch.double, device=device, requires_grad=True)
    y = diffuse(x, material, g)
    first = torch.tensor(g.first, device=device)
    second = torch.tensor(g.second, device=device)
    incidence = torch.zeros((2, 4), dtype=torch.double, device=device)
    incidence[torch.arange(2), first] = 1.
    incidence[torch.arange(2), second] = -1.
    area = torch.tensor(g.area, device=device)
    weights = torch.exp(material[0]) * torch.tensor(g.conductance, device=device)
    matrix = torch.diag(area) + incidence.T @ torch.diag(weights) @ incidence
    expected = torch.linalg.solve(matrix, (area * x).T).T
    torch.testing.assert_close(y, expected, rtol=1e-12, atol=1e-12)
    actual_grad = torch.autograd.grad(y.square().sum(), (x, material), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), (x, material))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)


def setup(tmp_path):
    model, current, _, _, _ = mb_setup(tmp_path)
    model.register_buffer('graded_index', torch.tensor([1]), persistent=False)
    model.register_buffer('graded_mask', torch.tensor([[0., 1., 0., 0., 0., 0.]]), persistent=False)
    model.register_buffer('graded_floor', torch.zeros_like(current.h), persistent=False)
    model.log_graded_gain = nn.Parameter(torch.tensor([-2.]))
    # Three actual discrete cable volumes for one graded source; no input edges
    # in this numerical fixture. Two differently located output paths.
    data = dict(node_ids=np.array([[1, 10], [1, 11], [1, 12]]), node_owner=np.zeros(3, dtype=np.int64),
        area=np.array([1., 2., 1.]), cable_u=np.array([0, 1]), cable_v=np.array([1, 2]),
        cable_g=np.array([1., 1.]), apl_index=np.array([1]))
    for channel, edges, pre, post in [('e', [], [], []), ('i', [2], [1], [4])]:
        for name, value in [('edge', edges), ('pre', pre), ('post', post)]:
            data[f'verify_{channel}_{name}'] = np.array(value, dtype=np.int64)
    for route in ['in_e', 'in_i', 'out_i', 'out_mb']:
        used = route.startswith('out')
        values = {'edge': [2], 'pre': [1], 'post': [4], 'delay': [2],
                  'node': [0 if route == 'out_i' else 2], 'pre_node': [-1], 'fraction': [1.]}
        for name, value in values.items():
            data[f'{route}_{name}'] = np.asarray(value if used else [], dtype=np.float64 if name == 'fraction' else np.int64)
    path = tmp_path / 'geometry.npz'
    np.savez(path, **data)
    path.with_suffix('.json').write_text(json.dumps({'nodes': 3}), encoding='utf8')
    attach_apl_cable(model, path)
    current = replace(current, apl_voltage=torch.tensor([[.2, .1, 0.]]),
        apl_ge=torch.zeros(1, 3), apl_gi=torch.zeros(1, 3),
        apl_packets=torch.tensor([[.1, .2, .3]]).expand(4, -1).clone())
    return model, current, path


def test_local_release_replaces_global_packet_without_double_counting(tmp_path):
    model, state, _ = setup(tmp_path)
    state = replace(state, ring=tuple(p.index_fill(1, torch.tensor([1]), 900.) for p in state.ring))
    step = advance_apl_cable(model, state, torch.zeros_like(state.h))
    # Native delayed fast path sees node0=.1; MB axon path sees node2=.3.
    assert float(step[0][0, 4]) == pytest.approx(.1 * .4)
    mb = apl_mb_arrival(model, state).reshape(3, 2)
    assert float(mb[2, 1]) == pytest.approx(.3 * .4 * .6)
    _, gaba, _, _ = step_mb_terminal(model, state)
    rho = model.get_decay_rates()[2].expand_as(state.h)[0, 4]
    assert float(gaba[2, 1]) == pytest.approx(float((1 - rho) * mb[2, 1]))
    model.apl_output_enabled = False
    off = advance_apl_cable(model, state, torch.zeros_like(state.h))
    assert torch.count_nonzero(off[0]) == 0 and torch.count_nonzero(off[-1][0]) == 0
    assert torch.count_nonzero(apl_mb_arrival(model, state)) == 0


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_cable_state_gradient_and_learner_checkpoint_recomputation_match(tmp_path, device):
    torch.manual_seed(4)
    model, state, _ = setup(tmp_path)
    model = model.to(device)
    state = FlyPhysicalState(**{k: tuple(p.to(device) for p in v) if k == 'ring' else v.to(device) for k, v in state.state_dict().items()})
    a = FlyBPTTLearner(model, copy.deepcopy(state), use_checkpointing=False, settle_ticks=3)
    b = FlyBPTTLearner(copy.deepcopy(model), copy.deepcopy(state), use_checkpointing=True, settle_ticks=3)
    a.previous_token = b.previous_token = 0
    for targets in ([1, 2, 3], [4, 5, 6]):
        p, _ = a.observe(targets)
        q, _ = b.observe(targets)
        torch.testing.assert_close(torch.tensor(p), torch.tensor(q))
        for field in ['apl_voltage', 'apl_ge', 'apl_gi', 'apl_packets']:
            torch.testing.assert_close(getattr(a.state, field), getattr(b.state, field))
    assert 'apl_log_axial' in native_parameter_names(a.model)
    assert a.optimizer.state[a.model.apl_log_axial]['exp_avg'].abs().sum() > 0
    saved = checkpoint_model_weights(a.model)
    assert 'apl_cable_node_ids' in saved and 'apl_cable_digest' in saved
    cloned = copy.deepcopy(a.model)  # A sparse-factor cache must not prevent a full copy.
    assert cloned.apl_geometry._cached is None


def test_nonempty_incompatible_spatial_state_and_anatomical_identity_are_rejected(tmp_path):
    model, state, path = setup(tmp_path)
    with pytest.raises(ValueError, match='incompatible shape'):
        cable_state(model, replace(state, apl_voltage=torch.ones(1, 2)))
    saved = checkpoint_model_weights(model)
    saved['apl_cable_digest'] = saved['apl_cable_digest'] ^ 1
    other, _, _ = setup(tmp_path)
    with pytest.raises(ValueError, match='identity/digest'):
        attach_apl_cable(other, path, saved=saved)
