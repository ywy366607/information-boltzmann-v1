"""Finite interfaces, provenance and derivatives; no capability claims."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.train_gdn_fast_table import TableGDN
from scripts.ib_local.language_navigation_kanli_v4b import attach_imagination
from scripts.ib_local.language_navigation_kanli_v4m import WoodObservationBody
from scripts.ib_local.language_navigation_kanli_v4p import ActionEvidence
from scripts.ib_local.navigation_direct_read import DirectReadGainEngine
from scripts.ib_local.navigation_internal_rollout import (
    InternalRolloutEngine, configure_internal_rollout, continuation,
    next_boundary_prediction, next_boundary_encoded)


def engine():
    base = GatedDeltaNetLM(256, 16, 2, 2, 'decay_then_delta', 32, 4, 32.,
                          decay_init='log_timescale')
    model = attach_imagination(WoodObservationBody(TableGDN(base, True),
        latent_dim=8, capacity=8, short=1, long=4, topk=4, history_size=4, max_chunk=4))
    # Mirror the existing action-conditioned F, without initializing a world.
    model.kanli.outcome[0] = torch.nn.Linear(19, 8)
    model.wood_evidence = ActionEvidence()
    m = model.memory
    m.register_buffer('wood_option', torch.arange(8) % 3)
    m.register_buffer('wood_target_count', torch.ones(8, dtype=torch.long))
    e = object.__new__(DirectReadGainEngine)
    e.model, e.wood_decay = model, .95
    e.device = torch.device('cpu')
    e.replay_rng, e.replay_batch = np.random.default_rng(17), 2
    configure_internal_rollout(e, 'observe')
    return e


def populate(e):
    m = e.model.memory
    with torch.no_grad():
        m.source.normal_(); m.action.normal_(); m.internal_next.normal_()
        m.occupied.fill_(True); m.count.fill_(1); m.last_write.fill_(7)
        m.internal_stamp.fill_(7); m.internal_valid.fill_(True)
        e.model.wood_evidence.seen.fill_(10)
        e.model.wood_evidence.base_error.fill_(1)
        e.model.wood_evidence.raw_error.fill_(.1)
    return torch.ones(8, dtype=torch.bool)


def test_migration_preserves_all_parameters_rng_and_has_fixed_storage():
    e = engine()
    params = list(e.model.parameters())
    state = torch.get_rng_state().clone()
    old = copy.deepcopy(e.model.state_dict())
    configure_internal_rollout(e, 'real')
    assert list(map(id, params)) == list(map(id, e.model.parameters()))
    assert torch.equal(state, torch.get_rng_state())
    for k, v in old.items():
        if torch.is_tensor(v):
            assert torch.equal(v, e.model.state_dict()[k])
    m = e.model.memory
    assert m.internal_next.numel() == m.capacity*m.width
    assert not m.internal_valid.any() and e.internal_pending is None
    assert e.model.long_prediction_loss(torch.randn(2, 16)).item() == 0


def test_encoded_query_is_exact_legacy_read_with_same_gradients():
    e = engine(); populate(e)
    model, m = e.model, e.model.memory
    x = torch.randn(3, 16, requires_grad=True)
    a = m.read(x, model.encoder)
    b = m.read_encoded(model.encoder(x), model.encoder)
    assert torch.equal(a.slots, b.slots)
    torch.testing.assert_close(a.value, b.value, rtol=0, atol=0)
    da = torch.autograd.grad(a.value.sum(), (x, model.encoder.weight), retain_graph=True)
    db = torch.autograd.grad(b.value.sum(), (x, model.encoder.weight))
    for u, v in zip(da, db):
        torch.testing.assert_close(u, v, rtol=0, atol=0)


def test_imagination_reads_without_mutation_and_rejects_incompatible_records():
    e = engine(); valid = populate(e)
    m, model = e.model.memory, e.model
    before = copy.deepcopy(model.state_dict()); rng = torch.get_rng_state().clone()
    q = model.encoder(torch.randn(3, 16))
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        v, packet = continuation(model, q, reward_valid=valid)
    assert packet['support'].all() and torch.isfinite(v).all()
    assert torch.equal(rng, torch.get_rng_state())
    for k, old in before.items():
        if torch.is_tensor(old):
            assert torch.equal(old, model.state_dict()[k])
    for change in ('merged', 'recycled', 'different_reward', 'unobserved'):
        with torch.no_grad():
            m.count.fill_(1); m.internal_stamp.copy_(m.last_write); m.internal_valid.fill_(True)
            mask = valid.clone()
            if change == 'merged': m.count.fill_(2)
            if change == 'recycled': m.internal_stamp.sub_(1)
            if change == 'different_reward': mask.zero_()
            if change == 'unobserved': m.internal_valid.zero_()
            out, p = continuation(model, q, reward_valid=mask)
            assert not p['support'].any() and not out.any()


def test_transition_learning_reaches_shared_encoder_and_existing_head(monkeypatch):
    e = engine(); populate(e)
    e._internal_label = None
    monkeypatch.setattr(DirectReadGainEngine, '_outcome_loss',
        lambda self, *args: (self.model.encoder.weight.sum()*0, {}))
    loss, d = e._outcome_loss(None, None, None)
    loss.backward()
    assert d['internal_transition_samples'] == 2
    assert e.model.encoder.weight.grad.norm() > 0
    assert e.model.long_predictor[0].weight.grad.norm() > 0
    assert not e.model.memory.internal_next.requires_grad


def test_equal_candidate_values_do_not_create_roundoff_credit():
    e = engine(); mask = populate(e)
    with torch.no_grad():
        e.model.wood_evidence.raw_error.fill_(1)
        e.model.wood_evidence.mean[:, 3] = .47565949
        q = torch.randn(20, 8)
        value, _ = continuation(e.model, q, reward_valid=mask)
    assert torch.equal(value, value[0].expand_as(value))


def test_only_real_next_boundary_fills_slot_and_scores_old_coordinate(monkeypatch):
    e = engine(); populate(e)
    target = torch.randn(16)
    e.recurrent, e.pending, e.table = None, 4, None
    e.decision = SimpleNamespace(start=lambda _: None, step=lambda *args: (None, target))
    monkeypatch.setattr('scripts.ib_local.navigation_internal_rollout.peek', lambda *args: None)
    e.model.memory.internal_valid.zero_()
    e._capture_boundary()
    assert not e.model.memory.internal_valid.any()
    old_encoder = e.model.encoder.weight.detach().clone()
    e.internal_pending = dict(slot=2, stamp=7, source=target, action=target,
        encoder_weight=old_encoder, next_latent=torch.nn.functional.linear(target, old_encoder))
    with torch.no_grad(): e.model.encoder.weight.add_(1)
    e._capture_boundary()
    assert e._internal_pre_error == 0
    assert e.model.memory.internal_valid.sum() == 1
    assert torch.equal(e.model.memory.internal_next[2], target)
    e.model.memory.internal_valid.zero_(); e.model.memory.last_write[2] += 1
    e._capture_boundary()
    assert e._internal_label is None and not e.model.memory.internal_valid.any()


def test_observe_mode_and_empty_calibration_deliver_no_new_choice_offset():
    e = engine()
    e.precision = lambda: 0.
    e.episodic_wood_mode = 'real'
    e._direct_read_offset = torch.tensor([.1, -.1, 0.])
    e._internal_offset = torch.tensor([.4, -.2, .8])
    available = torch.tensor([True, True, False])
    v = torch.ones(3)
    assert torch.equal(e._choice_offset(v, available), e._direct_read_offset)
    e.internal_mode = 'real'
    torch.testing.assert_close(e._choice_offset(v, available), torch.tensor([.4, -.4, 0.]))


def test_tag_state_roundtrip_and_imagined_state_remains_differentiable():
    e = engine(); mask = populate(e)
    saved = copy.deepcopy(e.model.state_dict())
    restored = engine()
    restored.model.load_state_dict(saved, strict=True)
    q = next_boundary_prediction(e.model, torch.randn(3, 16), torch.randn(3, 16))
    v, _ = continuation(e.model, q, reward_valid=mask)
    grad = torch.autograd.grad(v.sum(), e.model.long_predictor[0].weight)[0]
    assert grad.norm() > 0
    assert torch.equal(restored.model.memory.internal_next, e.model.memory.internal_next)
    # A next imagined state needs no inverse encoder or token reconstruction.
    third = next_boundary_encoded(e.model, q, q + .1)
    assert third.shape == q.shape
    assert torch.autograd.grad(third.sum(), q)[0].norm() > 0


def test_quiescent_world_save_borrows_seen_but_snapshot_still_copies():
    from scripts.ib_local.language_navigation_kanli_v4h import ReplyTopicWorld
    from scripts.ib_local.mother_reply_world import MotherReplyWorld
    data = np.frombuffer(b'abcd '*100, dtype=np.uint8).copy()
    world = MotherReplyWorld(ReplyTopicWorld(data, window=64, buckets=8, per_bucket=2), object())
    world.transcript = [('child', 'abcd'), ('mother', 'abcd ef')]
    snapshot = world.state_dict(); saved = world.checkpoint_state_dict()
    assert np.shares_memory(saved['seen'], world.seen)
    assert not np.shares_memory(snapshot['seen'], world.seen)
    assert saved.keys() == snapshot.keys()
    world.seen[:] = True; world.transcript.clear()
    assert not snapshot['seen'].any()
    assert len(snapshot['mother_transcript']) == len(saved['mother_transcript']) == 2


def test_mode_intervention_is_explicit_and_preserves_migration_provenance():
    from scripts.ib_local.run_mother_continuation import internal_mode_config
    cfg={'internal_rollout':dict(mode='real',migrated_at_cycle=2304,
                                initial_roundoff_repair={'cycle':2309})}
    with pytest.raises(ValueError):
        internal_mode_config(cfg,'observe',2309)
    changed=internal_mode_config(cfg,'observe',2309,True)
    assert changed['migrated_at_cycle']==2304
    assert changed['initial_roundoff_repair']=={'cycle':2309}
    assert len(changed['mode_interventions'])==1
    assert cfg['internal_rollout']['mode']=='real'
    assert internal_mode_config({'internal_rollout':changed},'observe',2317)==changed
