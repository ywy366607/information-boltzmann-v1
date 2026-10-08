"""Expression scaling, optimizer policy and continuation of the actual 3D graph."""
import copy

import pytest
import torch
from torch.nn import functional as F

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.optimization import (
    load_medium_branch_weights, make_medium_optimizer, medium_parameter_groups)
from information_boltzmann.runtime.training import belief_tensors


def make_model(**kwargs):
    torch.manual_seed(72)
    return PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=8,
        hidden=8, bath_type='conductance', activity_adaptation=True,
        short_term_plasticity=True, **kwargs).double()


def test_decoder_normalizes_final_feature_and_has_finite_joint_gradients():
    model = make_model()
    feature = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    original = model.decode(feature)
    # RMSNorm epsilon is machine epsilon; compare a 20x change away from zero.
    torch.testing.assert_close(model.decode(20 * feature), original, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(model.read_norm(feature).square().mean(-1),
                               torch.ones(2, 3, dtype=torch.float64), rtol=2e-12, atol=2e-12)
    assert torch.autograd.gradcheck(model.decode, (feature,))
    F.cross_entropy(original.flatten(0, 1), torch.tensor([1, 2, 3, 4, 5, 6])).backward()
    for p in (model.read_norm.weight, model.decoder.weight, model.decoder.bias):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
    # A learned nonunit gain can still change expression scale; observe it.
    with torch.no_grad():
        model.read_norm.weight.fill_(2)
    torch.testing.assert_close(model.read_norm(feature).square().mean(-1),
                               torch.full((2, 3), 4., dtype=torch.float64))
    assert torch.isfinite(model.decode(torch.zeros_like(feature))).all()


def test_expression_norm_preserves_all_persistent_physical_state():
    old = make_model(pre_decoder_norm=False)
    new = make_model()
    load_medium_branch_weights(new, old.state_dict())
    ids, targets = torch.tensor([[1, 2]]), torch.tensor([[2, 3]])
    _, before, _ = old(ids, targets, event_duration=.005)
    _, after, _ = new(ids, targets, event_duration=.005)
    for a, b in zip(belief_tensors(before), belief_tensors(after)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    raw, _ = new.read(after, decode=False)
    logits, _ = new.read(after)
    torch.testing.assert_close(logits, new.decode(raw))
    new.zero_grad()
    loss, _, _ = new(ids, targets, event_duration=.005)
    loss.backward()
    assert new.read_norm.weight.grad.norm() > 0
    assert new.medium.collision_rate[-1].weight.grad.norm() > 0
    assert new.write_agent.chart_gate[-1].weight.grad.norm() > 0


def test_weight_decay_policy_covers_every_parameter_once_and_preserves_physical_maps():
    model = make_model()
    groups = medium_parameter_groups(model, .01)
    all_ids = [id(p) for g in groups for p in g['params']]
    assert len(all_ids) == len(set(all_ids)) == len(list(model.learning_named_parameters()))
    omitted = {n for n, p in model.named_parameters() if id(p) not in set(all_ids)}
    assert omitted and all(n.startswith('source.') and n not in (
        'source.embedding.weight', 'source.channel_scale') for n in omitted)
    decay = dict(zip(groups[0]['param_names'], groups[0]['params']))
    protected = dict(zip(groups[1]['param_names'], groups[1]['params']))
    for name in ('decoder.weight', 'source.embedding.weight', 'readout.merge.weight',
                 'medium.collision_rate.2.weight'):
        assert name in decay
    for name in ('read_norm.weight', 'decoder.bias', 'readout.head_log_scale',
                 'readout.probe_coords', 'medium.material.coefficients', 'medium.log_speed.weight',
                 'medium.conductance_response.log_parameters.weight',
                 'medium.conduction_plasticity.log_rate.weight',
                 'medium.short_term_plasticity.parameters_map.weight',
                 'write_agent.observation_precision.0.weight'):
        assert name in protected
    optimizer = make_medium_optimizer(model, lr=.1)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()
    for n, p in model.named_parameters():
        expected = before[n] * .999 if n in decay else before[n]
        torch.testing.assert_close(p, expected, atol=0, rtol=0)
    with pytest.raises(ValueError):
        medium_parameter_groups(model, float('nan'))


@pytest.mark.parametrize('legacy', [False, True])
def test_optimizer_resume_preserves_moments_parameter_binding_and_next_update(legacy):
    model = make_model(pre_decoder_norm=not legacy)
    original = (torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.037)
                if legacy else make_medium_optimizer(model, lr=1e-4, weight_decay=.037))
    for p in model.parameters():
        p.grad = torch.full_like(p, .1)
    original.step()
    replica = copy.deepcopy(model)
    restored = make_medium_optimizer(replica, lr=9e-4, saved_state=copy.deepcopy(original.state_dict()))
    for source, target in zip(model.parameters(), replica.parameters()):
        source.grad = torch.full_like(source, -.2)
        target.grad = source.grad.clone()
    original.step()
    restored.step()
    for source, target in zip(model.parameters(), replica.parameters()):
        torch.testing.assert_close(source, target, rtol=0, atol=0)
        assert bool(original.state[source]) == bool(restored.state[target])
        if not original.state[source]:
            continue
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(original.state[source][key], restored.state[target][key], rtol=0, atol=0)


def test_named_legacy_resume_keeps_disconnected_parameters_and_moments():
    model = make_model()
    names, parameters = zip(*model.named_parameters())
    old = torch.optim.AdamW([{'params': parameters, 'param_names': list(names),
                             'group_name': 'historical_all_named'}], lr=1e-4)
    for p in parameters:
        p.grad = torch.ones_like(p)
    old.step()
    replica = copy.deepcopy(model)
    restored = make_medium_optimizer(replica, lr=1e-3, saved_state=copy.deepcopy(old.state_dict()))
    assert restored.param_groups[0]['param_names'] == list(names)
    for p, q in zip(model.parameters(), replica.parameters()):
        p.grad = torch.full_like(p, .2)
        q.grad = p.grad.clone()
    old.step()
    restored.step()
    for p, q in zip(model.parameters(), replica.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)


def test_branch_upgrade_accepts_only_missing_final_norm():
    old, new = make_model(pre_decoder_norm=False), make_model()
    saved = old.state_dict()
    load_medium_branch_weights(new, saved)
    torch.testing.assert_close(new.read_norm.weight, torch.ones(8, dtype=torch.float64))
    del saved['decoder.weight']
    with pytest.raises(ValueError, match='decoder.weight'):
        load_medium_branch_weights(new, saved)
