"""Persistent typed questions in the SAME bounded causal archive.

A question fixes its known premises and leaves one port open. Recalling it
re-evaluates those premises in the current encoder, rather than treating a
new context with a similar desired outcome as the same inverse problem.
Evidence is actual episode identity, not a stored gradient or imagined fact.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
import torch

from scripts.ib_local.causal_relation_program import (
    Evidence, Port, Variable, infer_role, real_evidence, revise, relation_features)


def _archive(engine):
    archive = getattr(engine.model, '_causal_archive', None)
    if archive is None:
        raise ValueError('shared unknowns require the continuing disk causal archive')
    return archive


def configure_shared_unknowns(engine):
    if not hasattr(engine, 'generative_relation_schema'):
        raise ValueError('requires the existing complete generative cycle')
    archive = _archive(engine)
    if getattr(engine,'shared_unknown_schema',1) != 1:
        raise ValueError('unknown shared-binding schema')
    archive.db.executescript('''
        CREATE TABLE IF NOT EXISTS unknown_bindings (
            id TEXT PRIMARY KEY, signature TEXT UNIQUE NOT NULL,
            payload TEXT NOT NULL, version INTEGER NOT NULL, touched INTEGER NOT NULL,
            anchor_episode INTEGER REFERENCES episodes(id));
        CREATE TABLE IF NOT EXISTS unknown_evidence (
            unknown_id TEXT NOT NULL REFERENCES unknown_bindings(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL, episode_id INTEGER NOT NULL REFERENCES episodes(id),
            PRIMARY KEY(unknown_id,ordinal));
        CREATE INDEX IF NOT EXISTS unknown_episode ON unknown_evidence(episode_id);
        CREATE TABLE IF NOT EXISTS unknown_turns (
            episode_id INTEGER PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE,
            unknown_id TEXT NOT NULL REFERENCES unknown_bindings(id) ON DELETE CASCADE,
            question_version INTEGER NOT NULL);
    ''')
    engine.shared_unknown_schema = 1


def _payload(role, source, action, target, anchor):
    raw = lambda x: None if x is None else x.detach().cpu().reshape(-1).tolist()
    value = dict(role=role.value, source=raw(source), action=raw(action),
                 target=raw(target), anchor=anchor,
                 subject='continuing_body', receiver='mother')
    # Only KNOWN premises determine equality. Inverse action problems in
    # different conditions stay distinct, even when their desired end is equal.
    known = dict(role=role.value, subject=value['subject'], receiver=value['receiver'])
    for name in ('source', 'action', 'target'):
        if name != {Port.MOTHER:'target', Port.ACTION:'action', Port.CONDITION:'source'}[role]:
            known[name] = value[name]
    if role != Port.MOTHER:
        known['anchor'] = anchor
    signature = hashlib.sha256(json.dumps(known, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return signature, value


def _row(archive, identity):
    row = archive.db.execute('SELECT id,payload,version FROM unknown_bindings WHERE id=?',
                             (identity,)).fetchone()
    return None if row is None else dict(identity=row[0], payload=json.loads(row[1]), version=row[2])


def _evidence(archive, eid):
    from scripts.ib_local.preview_runtime import read
    return read(archive.engine.model, 'episode', eid,
                lambda: _read_evidence(archive, eid), anchors=(archive,))


def _read_evidence(archive, eid):
    ep_cache = getattr(archive, '_episode_payload_cache', None)
    entry = None if ep_cache is None else ep_cache.get(eid)
    if entry is None:
        row = archive.db.execute('SELECT payload FROM episodes WHERE id=?', (eid,)).fetchone()
        if row is None:
            raise ValueError('shared unknown refers to a released fact')
        data = archive._unpack(row[0])
        if (int(data['count']) != 1 or not bool(data['mother_valid']) or
                int(data['mother_stamp']) != int(data['last_write'])):
            raise ValueError('merged or unlabelled episode cannot constrain an unknown')
        m = archive.engine.model.memory
        internal = bool(data['internal_valid']) and int(data['internal_stamp']) == int(data['last_write'])
        names = ['source', 'action', 'mother_next']+(['internal_next'] if internal else [])
        # The SQLite payload is already one immutable episode. Transfer its
        # equally shaped fields together instead of synchronizing CUDA five times.
        packed = torch.as_tensor(np.stack([data[name] for name in names]),
                                 device=m.source.device, dtype=m.source.dtype)
        tensors = {name: t.reshape(1, -1) for name, t in zip(names, packed.unbind(0))}
        entry = (tensors, int(data['last_write']), internal)
        if ep_cache is not None:
            if len(ep_cache) >= 1024:
                ep_cache.pop(next(iter(ep_cache)))
            ep_cache[eid] = entry
    tensors, stamp, internal = entry
    hot = archive.hot_slot_for(eid)
    common = dict(source=tensors['source'], action=tensors['action'], observed=True,
                  slot=hot, stamp=stamp, fact_id=f'episode:{eid}')
    result = [Evidence(outcome=tensors['mother_next'], kind=Port.MOTHER, **common)]
    if internal:
        result.append(Evidence(outcome=tensors['internal_next'], kind=Port.DECISION,
                               mother=tensors['mother_next'], **common))
    return result


def _supports(archive, identity):
    ue_cache = getattr(archive, '_unknown_evidence_cache', None)
    eids = None if ue_cache is None else ue_cache.get(identity)
    if eids is None:
        rows = archive.db.execute('SELECT episode_id FROM unknown_evidence WHERE unknown_id=? '
                                  'ORDER BY ordinal', (identity,)).fetchall()
        eids = tuple(eid for (eid,) in rows)
        if ue_cache is not None:
            if len(ue_cache) >= 1024:
                ue_cache.pop(next(iter(ue_cache)))
            ue_cache[identity] = eids
    return tuple(e for eid in eids for e in _evidence(archive, eid))


def _seed(archive, state):
    if getattr(archive.engine.model,'joint_relation_update_revision',1)>=3:
        from scripts.ib_local.relation_evidence import mechanism_evidence
        ids=list(dict.fromkeys(int(e.fact_id.split(':')[1]) for e in
            mechanism_evidence(archive.engine.model,state,Port.MOTHER)
            if e.observed and e.fact_id.startswith('episode:')))
        return ids[:len(state.valid)]
    ids=[]
    for slot in real_evidence(archive.engine.model,state,Port.MOTHER).tolist():
        mapping=archive.hot.get(str(slot))
        if mapping and archive._signature(slot)==mapping['signature'] and mapping['id'] not in ids:
            ids.append(mapping['id'])
    return ids[:len(state.valid)]


def retain_constraints(model, old, incoming, budget, *, use=None):
    """Same D-optimal release as local relations; incoming evidence survives U.

    The mother/decision ports of an episode consume ONE fact budget. The
    result contains old constraints only; incoming evidence stays in extra.
    Both real revision and imagination call this function through revise().
    """
    from scripts.ib_local.preview_runtime import read, tensor_key
    key = lambda e: e.fact_id or f'event:{e.slot}:{e.stamp}'
    identity = (tuple((id(e), tensor_key(e.source), tensor_key(e.action),
                       tensor_key(e.outcome)) for e in old),
                tuple(sorted({key(e) for e in incoming})), budget, use)
    return read(model, 'retained_support', identity,
                lambda: _retain_constraints(model, old, incoming, budget, use=use),
                anchors=(old, incoming))


def _retain_constraints(model, old, incoming, budget, *, use=None):
    key = lambda e: e.fact_id or f'event:{e.slot}:{e.stamp}'
    new_keys = {key(e) for e in incoming}
    kept = [e for e in old if key(e) not in new_keys]
    if len({key(e) for e in kept} | new_keys) <= budget:
        return tuple(kept)
    representatives = {key(e): e for e in kept if e.kind == Port.MOTHER}
    if not representatives:
        raise ValueError('incoming constraints exceed the fixed evidence budget')
    names = list(representatives)
    values = list(representatives.values())
    with torch.no_grad():
        enc = model.encoder
        x_all = relation_features(model, enc(torch.cat([e.source for e in values])),
                                  enc(torch.cat([e.action for e in values])),
                                  enc(torch.cat([e.outcome for e in values]))).cpu()
        ridge = getattr(model, 'causal_relation_ridge', .1)
        from scripts.ib_local.relation_retention_use import select
        alive = list(range(len(names)))
        evicted = set()
        remaining = len({key(e) for e in kept} | new_keys)
        while remaining > budget:
            if not alive:
                raise ValueError('incoming constraints exceed the fixed evidence budget')
            cur_names = [names[i] for i in alive]
            x = x_all[alive]
            index, _ = select(x, cur_names, use, ridge)
            evicted.add(cur_names[index])
            alive.pop(index)
            remaining -= 1
    return tuple(e for e in kept if key(e) not in evicted)


def _with_constraints(archive, state, question):
    supports=_supports(archive,question['identity'])
    canonical=getattr(archive.engine.model,'joint_relation_update_revision',1)>=3
    organization = None
    if getattr(archive.engine.model,'joint_relation_update_schema',None)==1:
        from scripts.ib_local.relation_retention_use import load
        own = load(archive.engine.model,f'relation:{state.identity}:{state.generation}')
        organization = replace(state,binding=None,constraints=supports if canonical else (),extra=(),
                               evidence_bound=canonical,
                               organization=None,variable_uses=(),retention_use=own)
    from scripts.ib_local.relation_retention_use import load, with_prior
    from scripts.ib_local.relation_evidence import evidence_key
    use = with_prior(archive.engine.model, load(archive.engine.model, question['identity']),
                     None if organization is None else organization.retention_use,
                     [evidence_key(f) for f in supports])
    return replace(state, binding=Variable(question['identity'], Port(question['payload']['role'])),
                   constraints=supports,organization=organization,evidence_bound=canonical,
                   retention_use=use)


def _protected(engine):
    result = set()
    pending = getattr(engine, 'internal_pending', None)
    if pending and 'causal_program' in pending:
        binding = getattr(pending['causal_program']['state'], 'binding', None)
        if binding is not None:
            result.add(binding.identity)
    current = getattr(engine, '_shared_question_receipt', None)
    if current is not None:
        result.add(current['identity'])
    return result


def _open(engine, state, payload, signature):
    archive, db = _archive(engine), _archive(engine).db
    existing = db.execute('SELECT id FROM unknown_bindings WHERE signature=?', (signature,)).fetchone()
    clock = int(engine.model.memory.clock)
    if existing:
        with db:
            db.execute('UPDATE unknown_bindings SET touched=? WHERE id=?', (clock,existing[0]))
        return _row(archive, existing[0])
    # Identity lives on SSD, not in the 64-relation CUDA workset. Reuse the
    # existing cold relation quota so old questions need not expire after
    # merely filling the hot bank once.
    capacity = archive.budgets['relations']
    with db:
        if db.execute('SELECT COUNT(*) FROM unknown_bindings').fetchone()[0] >= capacity:
            candidates = db.execute('SELECT id FROM unknown_bindings ORDER BY touched,rowid').fetchall()
            victim = next((r[0] for r in candidates if r[0] not in _protected(engine)), None)
            if victim is None:
                raise RuntimeError('all unknown slots are pinned by pending real participation')
            db.execute('DELETE FROM unknown_bindings WHERE id=?', (victim,))
            archive._unknown_evidence_cache.pop(victim, None)
            archive._retention_cache.pop(victim, None)
            archive._relation_vars_cache.clear()
        counter = archive._meta('next_unknown_identity', 1)
        identity = f'unknown:{counter}'
        archive._set_meta('next_unknown_identity', counter+1)
        anchor = payload['anchor']
        eid = int(anchor.split(':')[1]) if anchor.startswith('episode:') else None
        db.execute('INSERT INTO unknown_bindings VALUES (?,?,?,?,?,?)',
            (identity,signature,json.dumps(payload,allow_nan=False),0,clock,eid))
        # Seed with the relation's EXISTING real evidence, never predictions.
        ids=_seed(archive,state)
        db.executemany('INSERT INTO unknown_evidence VALUES (?,?,?)',
                       ((identity,i,eid) for i,eid in enumerate(ids[:len(state.valid)])))
    archive._unknown_evidence_cache[identity] = tuple(ids[:len(state.valid)])
    return _row(archive, identity)


_DEFAULT_RECEIPT = object()


def candidates(engine, state, source, initial_action, target_raw, anchor, *,
               retained_packet=_DEFAULT_RECEIPT, state_transform=None):
    """Current three open roles plus one question recalled by the EXISTING read.

    The selector learns whether to resume an old question. No fixed persistence
    schedule, question bonus or source-location information is introduced.
    """
    archive, model, m = _archive(engine), engine.model, engine.model.memory
    proposals=[]
    for role in (Port.MOTHER,Port.ACTION,Port.CONDITION):
        if role != Port.MOTHER and target_raw is None:
            continue
        signature,payload=_payload(role,source,initial_action if role==Port.MOTHER else source,
                                    None if role==Port.MOTHER else target_raw,anchor)
        found=archive.db.execute('SELECT id FROM unknown_bindings WHERE signature=?',(signature,)).fetchone()
        proposals.append(dict(signature=signature,payload=payload,
            question=None if found is None else _row(archive,found[0]),recalled=False))
    packet=(getattr(engine,'_retained_question_receipt',None)
            if retained_packet is _DEFAULT_RECEIPT else retained_packet)
    if packet is not None:
        mapping=archive.hot.get(str(packet['anchor']))
        if (mapping is not None and int(m.last_write[packet['anchor']])==packet['stamp'] and
                archive._signature(packet['anchor'])==mapping['signature']):
            row=archive.db.execute('SELECT u.id,u.signature FROM unknown_bindings u '
                'JOIN unknown_turns t ON t.unknown_id=u.id WHERE t.episode_id=?',
                (mapping['id'],)).fetchone()
            if row and all(p['signature']!=row[1] for p in proposals):
                q=_row(archive,row[0])
                proposals.append(dict(signature=row[1],payload=q['payload'],question=q,recalled=True))
    for proposal in proposals:
        q,payload=proposal['question'],proposal['payload']
        from scripts.ib_local.relation_retention_use import load
        bound_state=(_with_constraints(archive,state,q) if q is not None else
            replace(state,binding=Variable('pending:'+proposal['signature'],Port(payload['role'])),
                constraints=tuple(e for eid in _seed(archive,state) for e in _evidence(archive,eid)),
                retention_use=load(model,'pending:'+proposal['signature'])))
        if getattr(model,'joint_relation_update_schema',None)==1:
            canonical=getattr(model,'joint_relation_update_revision',1)>=3
            bound_state=replace(bound_state,organization=replace(state,binding=None,
                constraints=bound_state.constraints if canonical else (),extra=(),
                evidence_bound=canonical,organization=None,variable_uses=()),evidence_bound=canonical)
        from scripts.ib_local.relation_retention_use import with_prior
        from scripts.ib_local.relation_evidence import evidence_key
        bound_state = replace(bound_state, retention_use=with_prior(model, bound_state.retention_use,
            load(model, f'relation:{state.identity}:{state.generation}'),
            [evidence_key(f) for f in bound_state.constraints]))
        if state_transform is not None:
            bound_state=state_transform(proposal,bound_state)
        raw=lambda name: source.new_tensor(payload[name]).reshape(1,-1)
        role=Port(payload['role']);s,a=raw('source'),raw('action')
        slots=real_evidence(model,state,Port.MOTHER)
        if role==Port.MOTHER:
            y=infer_role(model,bound_state,open_role=role,condition=s,action=a,
                         ridge=engine.relation_ridge)['mean']
        else:
            y=model.encoder(raw('target'))
            facts=[e for e in bound_state.constraints if e.kind==Port.MOTHER]
            old_s=(torch.cat([e.source for e in facts]) if facts else m.source[slots])
            old_a=(torch.cat([e.action for e in facts]) if facts else m.action[slots])
            options=(torch.cat((s,s+old_a-old_s))[:m.topk]
                     if role==Port.ACTION else torch.cat((s,old_s))[:m.topk])
            inferred=infer_role(model,bound_state,open_role=role,condition=s,action=a,
                                candidates=options,observed=y,ridge=engine.relation_ridge)
            inferred_value=(inferred['log_weights'].exp()[:,None]*options).sum(0,keepdim=True)
            if role==Port.ACTION:a=inferred_value
            else:s=inferred_value
        proposal.update(state=bound_state,role=role,source=s,action=a,target=y)
    return proposals


def select(engine, proposal, state):
    archive=_archive(engine)
    q=_open(engine,state,proposal['payload'],proposal['signature'])
    if getattr(engine.model,'joint_relation_update_schema',None)==1:
        from scripts.ib_local.joint_relation_update import register_use
        register_use(engine,state,q)
    bound_state=_with_constraints(archive,state,q)
    engine._shared_question_receipt=dict(identity=q['identity'],version=q['version'],
        role=proposal['role'].value,recalled=proposal['recalled'],
        relation_identity=state.identity,signature=proposal['signature'])
    return bound_state


def commit(engine, row):
    """Actual participation supplies a conditional constraint, NOT an answer.

    In particular, performing an action does not prove it achieves a previously
    desired outcome. The posterior must predict the actually observed result.
    """
    archive=_archive(engine); state=engine._program_question_state
    identity=state.binding.identity
    if not row['causal_program']['evidence_committed']:
        return dict(identity=identity,committed=False,reason='real episode not individually valid')
    archive.sync()
    committed_identity = row['causal_program'].get('committed_episode_identity')
    eid=(int(committed_identity.split(':')[1]) if committed_identity else
         archive.hot[str(row['episode_slot'])]['id'])
    evidence=_evidence(archive,eid)[0]
    updated=revise(state,evidence,model=engine.model)
    ids=list(dict.fromkeys(int(e.fact_id.split(':')[1]) for e in
        updated.constraints+updated.extra if e.observed and e.fact_id.startswith('episode:')))
    if len(ids)>len(state.valid):raise RuntimeError('shared unknown exceeded evidence budget')
    from scripts.ib_local.relation_retention_use import commit as commit_use, filtered
    with archive.db:
        archive.db.execute('DELETE FROM unknown_evidence WHERE unknown_id=?',(identity,))
        archive.db.executemany('INSERT INTO unknown_evidence VALUES (?,?,?)',
                              ((identity,i,eid) for i,eid in enumerate(ids)))
        archive.db.execute('UPDATE unknown_bindings SET version=version+1 WHERE id=?',(identity,))
        archive.db.execute('INSERT OR REPLACE INTO unknown_turns VALUES (?,?,?)',
                           (eid,identity,engine._shared_question_receipt['version']))
        commit_use(engine.model,getattr(state,'retention_use',None),
                   filtered(getattr(updated,'retention_use',None),[f'episode:{x}' for x in ids]))
    archive._unknown_evidence_cache[identity] = tuple(ids)
    archive._relation_vars_cache.clear()
    q=_row(archive,identity)
    receipt=dict(identity=identity,committed=True,version_before=engine._shared_question_receipt['version'],
                version_after=q['version'],support=len(ids),observed_episode=eid,
                meaning='observed mechanism constraint, not equality to an inverse-query answer')
    if getattr(updated,'retention_use',None) is not None:
        receipt['retention_use']=dict(trace=updated.retention_trace,
            scope=updated.retention_use.scope,revision=updated.retention_use.revision,
            moments=list(filtered(updated.retention_use,[f'episode:{x}' for x in ids]).moments))
    if getattr(engine.model,'joint_relation_update_schema',None)==1:
        receipt['dependent_relations']=archive.db.execute(
            'SELECT COUNT(*) FROM relation_variables WHERE unknown_id=?',(identity,)).fetchone()[0]
        receipt['propagation']='new unknown version; dependents re-execute on next use; pending snapshots unchanged'
    return receipt
