"""Disk identity, real mechanism gradients and continuation; no capability tasks."""
import copy
import gzip
import sqlite3

import pytest
import torch

from scripts.ib_local.causal_archive import attach_archive, sha256
from scripts.ib_local.causal_relation_program import Evidence, Port, mechanism, real_evidence
from test_causal_relation_program import configured


def populated():
    engine = configured()
    m, bank = engine.model.memory, engine.model.causal_relations
    state = bank.bind(engine.model, m.source[:1])
    return engine, state


def test_migration_shares_attachment_preserves_model_rng_and_bounded_disk(tmp_path):
    e, state = populated()
    original = copy.deepcopy(e.model.state_dict())
    rng = torch.get_rng_state().clone()
    a = attach_archive(e, tmp_path, episode_capacity=32, relation_capacity=32)
    try:
        assert a.stats()['episodes'] == int(e.model.memory.occupied.sum())
        assert a.stats()['relations'] == 1
        assert a.stats()['record_bytes'] > 0
        assert torch.equal(rng, torch.get_rng_state())
        for k, v in original.items():
            if torch.is_tensor(v):
                assert torch.equal(v, e.model.state_dict()[k])
        assert a.db.execute('PRAGMA foreign_key_check').fetchall() == []
        a.sync()
        assert a.stats()['episodes'] == int(e.model.memory.occupied.sum())
        assert e.model.memory.words.data_ptr() == e.model.retrieval.words.data_ptr()
    finally:
        a.close()


def test_cold_evidence_survives_hot_reuse_and_drives_same_mechanism(tmp_path):
    e, state = populated()
    model, m, bank = e.model, e.model.memory, e.model.causal_relations
    source, action = m.source[:1].clone(), m.action[:1].clone()
    m.question_valid[0] = True
    m.question_stamp[0] = m.last_write[0]
    m.question_support[0].fill_(True)
    m.question_stamps[0].fill_(-123)  # expired parents, but a real anchor
    before = mechanism(model, state, source, action)['mean'].detach().clone()
    a = attach_archive(e, tmp_path)
    try:
        expected = int(state.valid.sum())
        e.prev = e.internal_pending = None
        # All old workset addresses are reused with distinct versions. Their
        # exact disk evidence must remain separate from the new occupants.
        with torch.no_grad():
            m.last_write.add_(100)
            m.read_credit_version.add_(1)
            m.source.add_(7)
            m.action.sub_(3)
        a.sync()
        assert not len(real_evidence(model, state, Port.MOTHER))
        # Remove the relation itself as well as its evidence from the hot
        # cache. Its persistent identity must survive the cache slot reuse.
        with torch.no_grad():
            bank.identity[state.index] = 0
            bank.valid[state.index].zero_()
        assert a.warm_relation(state.identity)
        current = bank.state(int(torch.nonzero(bank.identity == state.identity)[0]))
        assert len(real_evidence(model, current, Port.MOTHER)) == expected
        after = mechanism(model, current, source, action)['mean']
        torch.testing.assert_close(after, before, atol=1e-6, rtol=1e-6)
        after.sum().backward()
        assert model.encoder.weight.grad.norm() > 0
        assert a.stats()['loaded_episodes'] == expected
        anchor = current.slots[current.valid][0]
        assert bool(m.question_valid[anchor])
        assert a.db.execute('PRAGMA foreign_key_check').fetchall() == []
    finally:
        a.close()


def test_paging_protects_delayed_episode_and_old_read_receipts(tmp_path):
    e, state = populated()
    m = e.model.memory
    e.prev = dict(slot=0)
    e.internal_pending = None
    a = attach_archive(e, tmp_path)
    try:
        with torch.no_grad():
            m.last_write.add_(100)
            m.read_credit_version.add_(1)
        a.sync()
        protected = m.source[0].clone()
        old_version = int(m.read_credit_version[0])
        a.warm_relation(state.identity)
        assert torch.equal(protected, m.source[0]) and int(m.read_credit_version[0]) == old_version
        m._read_participation.append(object())
        with pytest.raises(RuntimeError, match='unsettled'):
            a.warm_relation(state.identity)
        m._read_participation.clear()
    finally:
        a.close()


def test_relation_search_uses_current_encoder_without_loading_whole_bank(tmp_path):
    e, state = populated()
    a = attach_archive(e, tmp_path)
    try:
        assert a.best_relation(e.model.causal_relations.condition[state.index:state.index+1], 1)[1] == state.identity
        # More than the hot relation count can persist. This adds no neural
        # model or fabricated observation: duplicate conditions are storage fixtures.
        row = a.db.execute('SELECT generation,version,condition,touched FROM relations LIMIT 1').fetchone()
        with a.db:
            for i in range(100, 100+e.model.causal_relations.capacity+1):
                a.db.execute('INSERT INTO relations VALUES (?,?,?,?,?)', (i, *row))
        assert a.stats()['relations'] > e.model.causal_relations.capacity
        assert a.best_relation(e.model.memory.source[:1], 2)[1] == state.identity
    finally:
        a.close()


def test_real_observation_after_cold_recall_revises_same_relation(tmp_path):
    e, state = populated()
    m, bank = e.model.memory, e.model.causal_relations
    a = attach_archive(e, tmp_path)
    try:
        a.warm_relation(state.identity)
        current = bank.state(state.index)
        slot = int(current.slots[current.valid][0])
        fact = Evidence(m.source[slot:slot+1], m.action[slot:slot+1], m.mother_next[slot:slot+1],
                        Port.MOTHER, observed=True, slot=slot, stamp=int(m.last_write[slot]))
        assert bank.commit(e.model, current, fact)
        a.sync()
        assert a.db.execute('SELECT version FROM relations WHERE id=?', (state.identity,)).fetchone()[0] == current.version+1
        assert not bank.commit(e.model, current, fact)
    finally:
        a.close()


def test_snapshot_reloads_private_copy_and_checks_hash_and_hot_identity(tmp_path):
    e, state = populated()
    a = attach_archive(e, tmp_path/'work')
    try:
        manifest = a.checkpoint(tmp_path/'saved')
        before = sha256(manifest['snapshot'])
        other, _ = populated()
        other.model.load_state_dict(copy.deepcopy(e.model.state_dict()))
        b = attach_archive(other, tmp_path/'restored', manifest=manifest)
        try:
            assert a.stats()['episodes'] == b.stats()['episodes']
            assert a.db.execute('SELECT * FROM evidence ORDER BY relation_id,ordinal').fetchall() == b.db.execute(
                'SELECT * FROM evidence ORDER BY relation_id,ordinal').fetchall()
            assert b.path != a.path
            b.db.execute('UPDATE relations SET version=version+1')
            b.db.commit()
            assert sha256(manifest['snapshot']) == before
        finally:
            b.close()
        wrong = dict(manifest, sha256='0'*64)
        with pytest.raises(ValueError, match='identity mismatch'):
            attach_archive(other, tmp_path/'bad', manifest=wrong)
        other.model.memory.last_write.add_(1)
        with pytest.raises(ValueError, match='workset'):
            attach_archive(other, tmp_path/'wrongbody', manifest=manifest)
    finally:
        a.close()


def test_record_quota_is_enforced_without_dangling_foreign_keys(tmp_path):
    e, _ = populated()
    with pytest.raises(RuntimeError, match='budget full'):
        attach_archive(e, tmp_path, episode_capacity=1)


def test_entire_joint_cycle_and_disk_next_update_restore(tmp_path):
    from test_joint_world_cycle import complete_interface_engine, assert_same_saved_values
    from scripts.ib_local.mother_checkpoint import save
    from scripts.ib_local.navigation_checkpoint import restore
    torch.set_num_threads(1)
    e = complete_interface_engine()
    a = attach_archive(e, tmp_path/'work')
    b = None
    try:
        first = e.step(torch.randn(8, 4))
        assert first['causal_program']['evidence_committed']
        assert first['causal_archive']['episodes'] == 1
        config = {}
        save(e, tmp_path/'saved', 1, config)
        with gzip.open(tmp_path/'saved/last.pt.gz', 'rb') as stream:
            saved = torch.load(stream, map_location='cpu', weights_only=False)
        assert saved['config']['causal_archive']['sha256'] == sha256(config['causal_archive']['snapshot'])
        expected = e.step(torch.randn(8, 4))
        expected_state = copy.deepcopy(e.model.state_dict())
        expected_optimizer = copy.deepcopy(e.optimizer.state_dict())
        other = complete_interface_engine()
        restore(other, saved)
        b = attach_archive(other, tmp_path/'restored', manifest=saved['config']['causal_archive'])
        actual = other.step(torch.randn(8, 4))
        assert actual['choice'] == expected['choice']
        assert actual['metal_g'] == expected['metal_g']
        assert actual['policy_replay_max_error'] == expected['policy_replay_max_error']
        assert actual['causal_program']['evidence_committed'] == expected['causal_program']['evidence_committed']
        assert_same_saved_values(expected_state, other.model.state_dict())
        assert_same_saved_values(expected_optimizer, other.optimizer.state_dict())
    finally:
        a.close()
        if b is not None:
            b.close()


def test_hot_only_restore_rejects_live_archive_instead_of_dropping_cold_history(tmp_path):
    from scripts.ib_local.navigation_checkpoint import snapshot, restore
    e, _ = populated()
    a = attach_archive(e, tmp_path)
    try:
        with pytest.raises(RuntimeError, match='disk'):
            snapshot(e)
        with pytest.raises(RuntimeError, match='disk'):
            restore(e, {})
    finally:
        a.close()


def test_failed_body_save_keeps_old_archive_binding_and_snapshot(tmp_path, monkeypatch):
    from scripts.ib_local import mother_checkpoint
    e, _ = populated()
    a = attach_archive(e, tmp_path/'work')
    try:
        old = a.checkpoint(tmp_path/'saved')
        config = dict(causal_archive=old)
        def failed(*args, **kwargs):
            raise OSError('injected checkpoint failure')
        monkeypatch.setattr(mother_checkpoint.navigation_checkpoint, 'save', failed)
        with pytest.raises(OSError, match='injected'):
            mother_checkpoint.save(e, tmp_path/'saved', 1, config)
        assert config['causal_archive'] == old
        assert sha256(old['snapshot']) == old['sha256']
    finally:
        a.close()


def test_old_loader_refuses_to_silently_discard_external_archive(tmp_path):
    from scripts.ib_local.run_causal_water_reply import build
    checkpoint = tmp_path/'archive.pt.gz'
    with gzip.open(checkpoint, 'wb') as stream:
        torch.save(dict(config=dict(causal_archive={'schema': 1})), stream)
    with pytest.raises(ValueError, match='external causal archive'):
        build(checkpoint, device='cpu')


def test_fixed_validation_does_not_copy_or_write_live_archive(tmp_path):
    import numpy as np
    from test_joint_world_cycle import complete_interface_engine
    from scripts.ib_local.run_language_navigation_v2 import validate
    e = complete_interface_engine()
    a = attach_archive(e, tmp_path)
    try:
        memory = e.model.memory
        before = copy.deepcopy(e.model.state_dict())
        digest = sha256(a.path)
        score = validate(e.model, e.table, np.arange(130, dtype=np.uint8), chunk=64)
        assert np.isfinite(score)
        assert e.model.memory is memory
        assert sha256(a.path) == digest
        for k, v in before.items():
            if torch.is_tensor(v):
                assert torch.equal(v, e.model.state_dict()[k])
    finally:
        a.close()
