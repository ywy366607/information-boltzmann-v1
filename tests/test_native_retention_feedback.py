"""Versioned finite-use feedback on the existing native co-evolving graph."""
import copy

import pytest
import torch

from scripts.ib_local.native_retention_feedback import (
    configure, language_loss, measure, settle, without_identity,
)
from scripts.ib_local.past_present_coevolution import configure_coevolution
from test_past_present_coevolution import engine, populated, SPEC


def body(mode='apply'):
    _, e = engine()
    configure_coevolution(e, SPEC)
    populated(e)
    e.model.memory.consolidation_enabled = True
    e.model.memory.consolidation_rule = 'utility'
    configure(e, mode)
    return e


def packet(e):
    ids = torch.tensor([[11, 12, 13, 14, 15, 16]])
    targets = ids+1
    lp = torch.full((1, 6, 256), -5.54517744448)
    e.model.memory.begin_read_credit()
    logits = e.model.predict(ids, e.recurrent, lp)[0]
    return ids, targets, lp, dict(logits=logits)


def test_finite_replay_preserves_live_graph_rng_state_and_receipts():
    e = body()
    ids, targets, lp, diag = packet(e)
    m, f = e.model.memory, e.model.body.coevolution
    before = {k:v.clone() for k,v in m.state_dict().items() if torch.is_tensor(v)}
    valid, state, rng = f.valid, e.recurrent.clone(), torch.random.get_rng_state()
    receipts = len(m._read_participation)
    measure(e, ids, targets, e.recurrent, lp, diag)
    assert f.valid is valid
    assert not hasattr(m, '_retention_excluded')
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(state, e.recurrent)
    assert len(m._read_participation) == receipts
    for name,value in before.items():
        assert torch.equal(value, m.state_dict()[name]), name
    # The diagnostic intervention must not invalidate the real graph.
    language_loss(diag['logits'], targets).backward()
    assert e.model.body.coevolution.bind.weight.grad.norm() > 0
    assert len(diag['retention_feedback']['observations']) == SPEC['slots']
    settle(e, diag)
    assert int(m.retention_utility_steps.sum()) == SPEC['slots']
    with pytest.raises(RuntimeError):
        settle(e, diag)


def test_scoped_panel_matches_plain_factual_and_each_cut():
    e = body()
    ids, targets, lp, diag = packet(e)
    plain = dict(logits=diag['logits'])
    e.decision.optimize = False
    measure(e, ids, targets, e.recurrent, lp, plain)
    e.decision.optimize = True
    measure(e, ids, targets, e.recurrent, lp, diag)
    for key in ('factual_language_loss', 'observations', 'replay_logit_max_error'):
        assert diag['retention_feedback'][key] == plain['retention_feedback'][key]
    assert not hasattr(e.model.memory, '_decode_projection_cache')


def test_withdrawal_cuts_same_identity_without_refilling_other_candidates():
    e = body()
    m, f = e.model.memory, e.model.body.coevolution
    s = int(f.slots_id[0]); v = int(f.versions[0])
    source = m.source[:2]
    with torch.no_grad():
        base = m.read(source, e.model.encoder)
        with without_identity(e.model, s, v):
            cut = m.read(source, e.model.encoder)
            assert not bool(f.valid[f.slots_id == s].any())
        assert torch.equal(base.slots, cut.slots)
        assert not bool(cut.weights[cut.slots == s].any())
        assert bool((cut.weights.sum(-1)-1).abs().max() < 1e-6)
        with without_identity(e.model, s, v+1):
            stale = m.read(source, e.model.encoder)
        assert torch.equal(base.weights, stale.weights)


def test_observe_keeps_retention_and_apply_rejects_stale_version():
    e = body('observe')
    ids, targets, lp, diag = packet(e)
    measure(e, ids, targets, e.recurrent, lp, diag)
    settle(e, diag)
    assert not bool(e.model.memory.retention_utility_steps.any())
    configure(e, 'apply')
    receipt = copy.deepcopy(diag['retention_feedback'])
    receipt.update(mode='apply', settled=False)
    first = receipt['observations'][0]
    e.model.memory.read_credit_version[first['slot']] += 1
    settle(e, dict(retention_feedback=receipt))
    assert receipt['stale'] == 1
    assert receipt['records'] == SPEC['slots']-1


def test_external_mask_never_pays_self_utterance():
    logits = torch.randn(1, 6, 256)
    target = torch.arange(6)[None]
    mask = torch.tensor([[0, 0, 1, 1, 1, 1]])
    changed = logits.clone(); changed[:, :2] = torch.randn_like(changed[:, :2])*100
    assert torch.equal(language_loss(logits,target,mask),language_loss(changed,target,mask))


def test_native_world_cycle_settles_before_next_episode(tmp_path):
    import numpy as np
    from scripts.ib_local.native_navigation_body import component_config, create
    from scripts.ib_local.language_navigation_kanli_v4h import ReplyTopicWorld
    from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
    from test_native_navigation_body import spec
    s = spec(); cfg = component_config(s, {})
    world = ReplyTopicWorld(np.arange(8192).astype(np.uint8), window=4096, seed=s['seed'])
    e, _ = create(cfg, world, table=RetrievedCreditTable(orders=(0,), heads=1, slots_log2=1),
                  causal_storage=tmp_path/'causal')
    try:
        configure_coevolution(e, SPEC)
        configure(e, 'apply')
        rows = [e.step(torch.randn(s['latent_dim'], 4)) for _ in range(2)]
        assert bool(e.model.memory.retention_utility_steps.any())
        assert all(torch.isfinite(e.model.memory.retention_utility_first))
        assert e.model._causal_archive.db.execute('PRAGMA foreign_key_check').fetchall() == []
    finally:
        e.model._causal_archive.close()


def test_off_has_no_forward_or_retention_effect():
    e = body('off')
    measure(e, None, None, None, None, {})
    settle(e, {})
    assert not bool(e.model.memory.retention_utility_steps.any())
