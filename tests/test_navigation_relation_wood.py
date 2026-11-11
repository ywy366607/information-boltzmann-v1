"""Finite actual-evidence relation contracts; not a language capability test."""
import torch

from scripts.ib_local.navigation_relation_wood import local_relation, configure_relation_wood
from scripts.ib_local.navigation_momentum_wood import MomentumWoodEngine
from scripts.ib_local.navigation_internal_rollout import next_boundary_prediction
from test_navigation_internal_rollout import populate
from test_navigation_momentum_wood import configured


def test_relation_uses_compatible_observations_and_current_encoder_gradients():
    e = configured(); populate(e); configure_relation_wood(e)
    source = torch.randn(2, 16, requires_grad=True)
    action = torch.randn(2, 16, requires_grad=True)
    value, receipt = local_relation(e.model, source, action)
    assert receipt['valid'].all() and value.norm() > 0
    gradients = torch.autograd.grad(value.square().sum(), (source, e.model.encoder.weight))
    assert all(g.norm() > 0 for g in gradients)
    e.model.memory.internal_valid.zero_()
    empty, info = local_relation(e.model, source, action)
    assert not info['valid'].any() and torch.equal(empty, torch.zeros_like(empty))


def test_actual_receipt_rejects_recycled_and_self_target_slots():
    e = configured(); populate(e); configure_relation_wood(e)
    source, action = torch.randn(1, 16), torch.randn(1, 16)
    _, info = local_relation(e.model, source, action)
    slots, stamps, valid = (info[k].clone() for k in ('slots','stamps','valid'))
    e.model.memory.last_write[slots[0, 0]] += 1
    excluded = slots[:, 1].clone()
    _, now = local_relation(e.model, source, action, exclude=excluded,
                            receipt=(slots, stamps, valid))
    assert not now['valid'][0, 0] and not now['valid'][0, 1]
    assert torch.equal(now['slots'], slots)


def test_relation_shadow_read_does_not_write_reservoir():
    e = configured(); populate(e); configure_relation_wood(e)
    before = {k:v.clone() for k,v in e.model.memory.state_dict().items() if torch.is_tensor(v)}
    local_relation(e.model, torch.randn(2, 16), torch.randn(2, 16))
    assert all(torch.equal(v, e.model.memory.state_dict()[k]) for k,v in before.items())


def test_hypothetical_evidence_uses_same_solve_and_does_not_write():
    e = configured(); populate(e); configure_relation_wood(e)
    source, action = torch.randn(2, 16), torch.randn(2, 16)
    observed = (torch.randn(2, 16), torch.randn(2, 16), torch.randn(2, 16))
    before = {k:v.clone() for k,v in e.model.memory.state_dict().items() if torch.is_tensor(v)}
    baseline, _ = local_relation(e.model, source, action)
    omitted, _ = local_relation(e.model, source, action,
        extra=(*observed, torch.zeros(2, dtype=torch.bool)))
    shadow, packet = local_relation(e.model, source, action,
        extra=(*observed, torch.ones(2, dtype=torch.bool)))
    torch.testing.assert_close(omitted, baseline)
    assert (shadow-baseline).norm() > 0 and packet['extra_coefficient'].shape == (2,1)
    assert all(torch.equal(v, e.model.memory.state_dict()[k]) for k,v in before.items())


def test_shadow_evidence_matches_same_observation_added_to_selected_bank():
    e = configured(); populate(e); configure_relation_wood(e)
    m = e.model.memory
    m.internal_valid[7] = False
    source, action = torch.randn(1, 16), torch.randn(1, 16)
    extra_s, extra_a, extra_y = torch.randn(1, 16), torch.randn(1, 16), torch.randn(1, 16)
    shadow, receipt = local_relation(e.model, source, action,
        extra=(extra_s, extra_a, extra_y, torch.ones(1, dtype=torch.bool)))
    with torch.no_grad():
        m.source[7].copy_(extra_s[0]); m.action[7].copy_(extra_a[0]); m.internal_next[7].copy_(extra_y[0])
        m.count[7] = 1; m.occupied[7] = True; m.internal_valid[7] = True
        m.last_write[7] = 9; m.internal_stamp[7] = 9
    m.topk += 1
    augmented = (torch.cat((receipt['slots'], torch.tensor([[7]])), -1),
                 torch.cat((receipt['stamps'], torch.tensor([[9]])), -1),
                 torch.cat((receipt['valid'], torch.ones(1,1,dtype=torch.bool)), -1))
    actual, _ = local_relation(e.model, source, action, receipt=augmented)
    torch.testing.assert_close(shadow, actual, rtol=1e-5, atol=1e-6)


def test_encoded_future_uses_same_relation_update_as_raw_observation():
    e = configured(); populate(e); configure_relation_wood(e)
    s, a, y = torch.randn(2,16), torch.randn(2,16), torch.randn(2,16)
    valid = torch.ones(2, dtype=torch.bool)
    raw, _ = local_relation(e.model, s, a, extra=(s,a,y,valid))
    encoded, _ = local_relation(e.model, s, a,
        extra=(s,a,e.model.encoder(y),valid), extra_encoded=True)
    torch.testing.assert_close(raw, encoded)


def test_joint_real_label_replaces_instead_of_duplicates_momentum_example(monkeypatch):
    e = configured(); populate(e); configure_relation_wood(e)
    s, a, target = torch.randn(16), torch.randn(16), torch.randn(16)
    read = torch.randn(e.model.momentum_to_boundary.in_features)
    _, recorded = local_relation(e.model, s[None], a[None])
    e.internal_pending = dict(slot=7, momentum_read=read,
        relation_slots=recorded['slots'][0], relation_stamps=recorded['stamps'][0],
        relation_valid=recorded['valid'][0])
    e._internal_label = (s,a,target)
    label = e.model.encoder(target[None])
    base = next_boundary_prediction(e.model, s[None], a[None]) + e.model.momentum_to_boundary(read[None])
    p_loss = torch.nn.functional.mse_loss(base, label)
    monkeypatch.setattr(MomentumWoodEngine, '_outcome_loss',
        lambda self, *args: (self.model.encoder.weight.sum()*0 + 1 + p_loss, {}))
    result, diag = e._outcome_loss(None, None, None)
    assert diag['relation_samples'] == 1
    correction, _ = local_relation(e.model, s[None], a[None], exclude=torch.tensor([7]),
        receipt=(recorded['slots'], recorded['stamps'], recorded['valid']))
    expected = 1 + torch.nn.functional.mse_loss(base+correction, label)
    torch.testing.assert_close(result, expected)
