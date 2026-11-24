"""Real event identity survives compression of the hot experience prototype.

The original bounded SQLite episode store holds the observed event; the hot
slot may still merge normally. Prediction, finite U, payment, shared unknowns
and delayed next-boundary learning all refer to this same immutable identity.
This explicit experimental mode uses the existing archive quotas and model.
"""
from __future__ import annotations

import copy
import numpy as np
import torch


def configure(engine, mode):
    if mode not in ('compressed', 'identified'):
        raise ValueError(mode)
    if getattr(engine, 'causal_use_arm', None) != 'C2':
        raise ValueError('requires finite-U settlement')
    if getattr(engine.model, 'joint_relation_update_revision', 0) < 4:
        raise ValueError('requires persistent canonical fact identities')
    old = getattr(engine, 'event_credit_mode', None)
    if old is not None:
        if old != mode:
            raise ValueError('change event-credit arm only at a common source boundary')
        return
    from scripts.ib_local.navigation_direct_read import gain_state
    from scripts.ib_local.language_navigation_kanli_v4p import ActionEvidence
    engine.event_credit_migration = dict(kind=engine.settlement_kind,
        calibration=copy.deepcopy(engine.causal_use_calibration),
        value_evidence=copy.deepcopy(engine.causal_use_value_evidence.state_dict()),
        baseline=engine.baseline, critic_error=engine.critic_error,
        gain_states={k:copy.deepcopy(getattr(engine,k)) for k in
                     ('read_gain_state','internal_gain_state','shadow_gain_state')})
    engine.event_credit_mode = mode
    engine.settlement_kind = 'finite_event_targeted_use_v3'
    engine.causal_use_calibration = dict(count=0, cross=0., second=0.)
    engine.causal_use_value_evidence = ActionEvidence(engine.wood_decay).to(engine.device)
    object.__setattr__(engine.model, '_causal_use_value_evidence', engine.causal_use_value_evidence)
    engine.baseline, engine.critic_error, engine.adv_ms = 0., None, None
    for k in ('read_gain_state','internal_gain_state','shadow_gain_state'):
        setattr(engine,k,gain_state(engine.wood_decay))


def enabled(engine):
    return getattr(engine, 'event_credit_mode', '') == 'identified'


@torch.no_grad()
def capture(engine, *, source, action, short, long, words, source_time, slot, merged):
    """Persist only actual observed vectors, after hot allocation, before U.

    Individual hot episodes keep their existing identity. A merged prototype
    gets a separate cold event with raw fields from this event (never averages).
    The record has the existing schema; no physical GDN/optimizer snapshot.
    """
    engine._identified_event_fact = None
    if not enabled(engine) or engine._mother_boundary_target is None or not merged:
        return
    from scripts.ib_local.shared_unknown_bindings import _evidence
    archive, m = engine.model._causal_archive, engine.model.memory
    data = {k:np.zeros(tuple(spec['shape']), dtype=np.dtype(spec['dtype']))
            for k,spec in archive.schema.items()}
    for name,value in dict(source=source, action=action, successor=short,
                           long_successor=long, mother_next=engine._mother_boundary_target).items():
        data[name][...] = value.detach().cpu().numpy()
    stamp = int(m.clock)
    for name,value in dict(count=1, occupied=True, last_write=stamp,
                           mother_stamp=stamp, mother_valid=True, birth=source_time,
                           last_touch=stamp, read_credit_version=1, wood_option=-1,
                           option=engine._pending_record['choice']).items():
        if name in data:
            data[name][...] = value
    if '@words' in data:
        size = min(len(words), data['@words'].size)
        data['@words'].reshape(-1)[:size] = np.frombuffer(words, dtype=np.uint8)[:size]
        data['@word_count'][...] = size
    payload = b''.join(data[k].tobytes() for k,_,_,_ in archive.layouts)
    with archive.db:
        archive._room('episodes')
        eid = archive.db.execute('INSERT INTO episodes(origin_slot,stamp,payload,touched) '
            'VALUES (?,?,?,?)', (-1, stamp, payload, stamp)).lastrowid
    engine._identified_event_fact = _evidence(archive, eid)[0]


def current(engine):
    return getattr(engine, '_identified_event_fact', None) if enabled(engine) else None


@torch.no_grad()
def finish_labels(engine, row):
    """Attach this event's actual paid label, rather than a prototype average."""
    fact = current(engine)
    if fact is None:
        return
    archive = engine.model._causal_archive
    eid = int(fact.fact_id.split(':')[1])
    payload = archive.db.execute('SELECT payload FROM episodes WHERE id=?',(eid,)).fetchone()[0]
    data = archive._unpack(payload)
    choice = engine._pending_record['choice']
    values = dict(wood_target=row['wood_target'], wood_target_count=1, wood_option=choice,
        wood_baseline=engine._forecast['base'][choice], progress=row['metal_g'],
        progress_valid=True, feeling=engine._pending_record['feeling'],
        causal_evidence=engine._episodic_decision_read['features'][choice],
        causal_evidence_valid=True)
    from scripts.ib_local.navigation_settlement import settlement_code
    values.update({'$'+k:settlement_code(engine.settlement_kind)
                   for k in ('settlement_versions','wood_settlement_versions')})
    for k,v in values.items():
        if k in data:
            data[k][...] = v.detach().cpu().numpy() if torch.is_tensor(v) else v
    payload = b''.join(data[k].tobytes() for k,_,_,_ in archive.layouts)
    with archive.db:
        archive.db.execute('UPDATE episodes SET payload=? WHERE id=?',(payload,eid))


def valid_cold_fact(model, evidence):
    """Commit guard: byte-exact actual SSD event, not an imagined observation."""
    if not enabled(model._causal_archive.engine) or evidence.slot >= 0:
        return False
    if not evidence.fact_id.startswith('episode:'):
        return False
    from scripts.ib_local.shared_unknown_bindings import _evidence
    actual = _evidence(model._causal_archive, int(evidence.fact_id.split(':')[1]))[0]
    return (actual.stamp == evidence.stamp and actual.kind == evidence.kind and
            all(torch.equal(getattr(actual,k),getattr(evidence,k))
                for k in ('source','action','outcome')))


@torch.no_grad()
def complete_boundary(engine, pending, target):
    """Attach the later actual boundary to its own event, despite a merged cache."""
    if not enabled(engine):
        return None
    identity = pending.get('causal_program', {}).get('episode_identity', '')
    if not identity:
        return None
    archive = engine.model._causal_archive
    eid = int(identity.split(':')[1])
    row = archive.db.execute('SELECT origin_slot,payload FROM episodes WHERE id=?',(eid,)).fetchone()
    if row is None or row[0] >= 0:
        return None
    data = archive._unpack(row[1])
    if int(data['count']) != 1 or not bool(data['mother_valid']):
        raise ValueError('delayed boundary must name an actual individual event')
    data['internal_next'][...] = target.detach().cpu().numpy()
    data['internal_stamp'][...] = data['last_write']
    data['internal_valid'][...] = True
    payload = b''.join(data[k].tobytes() for k,_,_,_ in archive.layouts)
    with archive.db:
        archive.db.execute('UPDATE episodes SET payload=? WHERE id=?',(payload,eid))
    # archive.prepare() may have paged this cold event into another hot slot
    # before the boundary. Update that exact cache identity as well, so the
    # ordinary sync cannot overwrite the newly completed port with stale data.
    m = engine.model.memory
    for key,mapping in archive.hot.items():
        if mapping['id'] == eid and archive._signature(int(key)) == mapping['signature']:
            hot = int(key)
            m.internal_next[hot].copy_(target)
            m.internal_stamp[hot] = m.last_write[hot]
            m.internal_valid[hot] = True
    # Decode directly: an earlier preview cache can hold the pre-boundary view.
    from scripts.ib_local.shared_unknown_bindings import _read_evidence
    fact = _read_evidence(archive,eid)[0]
    return fact.source.reshape(-1), fact.action.reshape(-1), target
