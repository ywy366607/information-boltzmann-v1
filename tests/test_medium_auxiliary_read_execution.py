"""Exact auxiliary execution contracts on a structural, temporal individual.

These are CPU numerical tests, not learning/capability experiments.
"""
import copy
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.active_medium_training import ActiveMediumTrainer
from information_boltzmann.runtime.optimization import make_medium_optimizer
from information_boltzmann.runtime.training import (
    _writer_auxiliary_gradients, belief_tensors, clone_belief, training_event,
)


def candidate(*, begin_window=True):
    # Same constitutive candidate as test_medium_structural_integration, with a
    # smaller content dimension only; all persistent mechanisms stay enabled.
    torch.manual_seed(193)
    net = PlasticMediumPorts3D(
        vocab_size=13, shape=(2, 2, 2), channels=4, material_width=3,
        hidden=4, heads=1, queries=1, anisotropic_transport=True,
        bath_type='conductance', short_term_plasticity=True, activity_adaptation=True,
        material_reference_shape=None, read_mode='temporal', temporal_rates=[1., 4.],
        temporal_frequencies=[0., 3.], temporal_time_reference=.02,
        intrinsic_time_reference=.023, solver_max_step=.01, observer_max_step=.02,
        structure_options=dict(resource_density=4., speed_reference=4., structure_time=1.,
            prior_std=.2, initial_std=.2, maintenance_supply=2., initial_dual=.1),
    ).double()
    if begin_window:
        q = net.medium.structural_posterior
        q.begin_window(torch.randn_like(q.mean))
    return net


def writer_parameters(net):
    return tuple(p for name, p in net.learning_named_parameters()
                 if name.startswith(('source.', 'write_agent.')))


def assert_state_equal(left, right):
    a, b = belief_tensors(left), belief_tensors(right)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_omitted_expression_keeps_complete_temporal_trajectory_without_read():
    net = candidate()
    initial = net.initial_belief()
    table = F.normalize(net.source.embedding.weight, dim=-1)
    with torch.no_grad():
        prepared = net.medium.prepare_evolution()
        left, right = initial, initial
        for token in (2, 3, 5, 7):
            token = torch.tensor([token])
            full = training_event(net, left, token, table, prepared, .005, 1, False,
                                  read_feature=True)
            with patch.object(net, 'read', side_effect=AssertionError('unused read executed')):
                omitted = training_event(net, right, token, table, prepared, .005, 1, False,
                                         read_feature=False)
            assert full[4] is not None and omitted[4] is None
            assert_state_equal(full[0], omitted[0])
            assert_state_equal(full[1], omitted[1])
            torch.testing.assert_close(full[2]['_write_free_energy'],
                                       omitted[2]['_write_free_energy'], atol=0, rtol=0)
            left, right = full[1], omitted[1]
    assert left.medium.elapsed.item() > 4 * .005
    assert left.temporal.value.norm() > 0
    torch.testing.assert_close(left.medium.elapsed, left.temporal.elapsed, atol=0, rtol=0)
    assert_state_equal(initial, net.initial_belief())


@pytest.mark.parametrize('checkpointed,tokens', [(False, 4), (True, 32)])
def test_all_writer_auxiliary_gradients_match_with_unused_read_removed(checkpointed, tokens):
    net = candidate()
    ids = (torch.arange(tokens) % 12 + 1).reshape(1, -1)
    initial = net.initial_belief()
    initial_copy = clone_belief(initial)
    # Preserve None for genuinely unused leaves, including through event VJP.
    unused = torch.nn.Parameter(torch.tensor(.7, dtype=torch.float64))
    parameters = writer_parameters(net) + (unused,)
    q = net.medium.structural_posterior
    posterior_before = {k: v.clone() for k, v in q.state_dict().items()}
    rng_before = torch.random.get_rng_state().clone()
    with patch.object(net, 'read', wraps=net.read) as reader:
        full = _writer_auxiliary_gradients(net, ids, initial, .005, 1,
            checkpointed, False, parameters, checkpoint_granularity='event', read_feature=True)
        assert reader.call_count >= tokens
    with patch.object(net, 'read', side_effect=AssertionError('unused read executed')):
        omitted = _writer_auxiliary_gradients(net, ids, initial, .005, 1,
            checkpointed, False, parameters, checkpoint_granularity='event')
    assert full[-1] is None and omitted[-1] is None
    assert any(g is not None and g.norm() > 0 for g in omitted[:-1])
    for a, b in zip(full, omitted):
        assert (a is None) == (b is None)
        if a is not None:
            assert a.device.type == b.device.type == 'cpu'
            assert not a.requires_grad and not b.requires_grad
            torch.testing.assert_close(a, b, atol=2e-11, rtol=2e-9)
    assert_state_equal(initial, initial_copy)
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, atol=0, rtol=0)
    for name, value in q.state_dict().items():
        torch.testing.assert_close(value, posterior_before[name], atol=0, rtol=0, msg=name)
    assert all(p.grad is None for p in parameters)


def test_omitted_auxiliary_read_preserves_pending_window_checkpoint_semantics():
    # The trainer owns both sampling and the physical start timestamp.
    a = candidate(begin_window=False)
    oa = make_medium_optimizer(a, lr=2e-4)
    la = ActiveMediumTrainer(a, oa, a.initial_belief(), carry_token=1,
        event_duration=.005, chunk_tokens=1, tokens_per_update=2,
        activation_checkpointing=True, checkpoint_granularity='event')
    prior = torch.ones(13)
    la.consume(torch.tensor([2]), phase='train_first_pass', prior_nll=prior)
    assert la.pending == 1 and a.medium.structural_posterior.window_active
    saved_pending = {name: None if p.grad is None else p.grad.clone()
                     for name, p in a.named_parameters()}
    b = copy.deepcopy(a)
    ob = make_medium_optimizer(b, lr=2e-4, saved_state=copy.deepcopy(oa.state_dict()))
    lb = ActiveMediumTrainer(b, ob, clone_belief(la.belief), carry_token=1,
        event_duration=.005, chunk_tokens=1, tokens_per_update=2,
        activation_checkpointing=True, checkpoint_granularity='event')
    lb.load_state_dict(copy.deepcopy(la.state_dict()))
    for name, p in b.named_parameters():
        expected = saved_pending[name]
        assert (p.grad is None) == (expected is None), name
        if expected is not None:
            torch.testing.assert_close(p.grad, expected, atol=0, rtol=0, msg=name)
    ra = la.consume(torch.tensor([3]), phase='fresh_B', prior_nll=prior)
    rb = lb.consume(torch.tensor([3]), phase='fresh_B', prior_nll=prior)
    assert ra == rb and la.optimizer_updates == lb.optimizer_updates == 1
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name], atol=0, rtol=0, msg=name)
    assert_state_equal(la.belief, lb.belief)
    assert a.medium.structural_posterior.evidence_events == 2
    assert a.medium.structural_posterior.windows_committed == 1
