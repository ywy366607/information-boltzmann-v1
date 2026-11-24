"""One bounded fact view for prediction, organization, replay and commit."""
from __future__ import annotations

from dataclasses import replace
import torch


def evidence_key(evidence):
    return evidence.fact_id or f'event:{evidence.slot}:{evidence.stamp}'


def excludes_evidence(model,evidence,*,exclude=-1,exclude_fact=''):
    """Stable identity wins; a bare cache address needs its current event tag."""
    if getattr(model,'joint_relation_update_revision',1)<4:
        return ((exclude>=0 and evidence.slot==exclude) or
                bool(exclude_fact and evidence.fact_id==exclude_fact))
    if exclude_fact:return evidence.fact_id==exclude_fact
    if exclude<0 or evidence.slot!=exclude:return False
    m=model.memory
    return (bool(m.occupied[exclude]) and int(m.last_write[exclude])==evidence.stamp)


def pending_fact_identity(model,pending):
    """Use the actually committed event, never a coincidentally reused slot."""
    recorded=pending.get('causal_program',{}).get('episode_identity')
    if recorded:return recorded
    archive=model._causal_archive;m=model.memory;slot=pending['slot']
    mapping=archive.hot.get(str(slot))
    if (mapping is None or not bool(m.occupied[slot]) or int(m.last_write[slot])!=pending['stamp']
            or archive._signature(slot)!=mapping['signature']):
        raise ValueError('delayed outcome has no matching committed fact identity')
    return f"episode:{mapping['id']}"


def mechanism_evidence(model,state,kind,*,exclude=-1,exclude_fact=''):
    from scripts.ib_local.causal_relation_program import Evidence,real_evidence
    m=model.memory
    if state.evidence_bound or state.binding is not None or state.variable_uses:
        facts=state.constraints
    else:
        archive=getattr(model,'_causal_archive',None)
        hot=[]
        for slot in real_evidence(model,state,kind).tolist():
            mapping=None if archive is None else archive.hot.get(str(slot))
            identity=(f"episode:{mapping['id']}" if mapping is not None and
                archive._signature(slot)==mapping['signature'] else '')
            hot.append(Evidence(m.source[slot:slot+1],m.action[slot:slot+1],
                m.mother_next[slot:slot+1] if kind.value=='mother' else m.internal_next[slot:slot+1],
                kind,True,slot=slot,stamp=int(m.last_write[slot]),fact_id=identity,
                mother=None if kind.value=='mother' else m.mother_next[slot:slot+1]))
        facts=tuple(hot)
    selected={}
    for evidence in facts+state.extra:
        if evidence.kind!=kind or excludes_evidence(model,evidence,exclude=exclude,exclude_fact=exclude_fact):continue
        selected[(evidence_key(evidence),evidence.kind)]=evidence
    facts = tuple(selected.values())
    if getattr(model, 'identity_preview_revision', 0) >= 1:
        from scripts.ib_local.identity_preview import fact_identity
        facts = tuple(sorted(facts, key=fact_identity))
    return facts


def fact_reference(evidence):
    return evidence.fact_id if evidence.observed and evidence.fact_id else None


def restrict_support(model,state,references):
    """A functional split/share retains exactly its chosen finite facts."""
    from scripts.ib_local.shared_unknown_bindings import _evidence
    wanted={ref for ref in references if ref is not None}
    extra_keys={evidence_key(e) for e in state.extra}
    existing={evidence_key(e) for e in state.constraints}
    facts=tuple(e for e in state.constraints if evidence_key(e) in wanted)
    for ref in sorted(wanted-existing-extra_keys):
        if not ref.startswith('episode:'):raise ValueError('unknown evidence identity')
        facts+=tuple(_evidence(model._causal_archive,int(ref.split(':')[1])))
    from scripts.ib_local.relation_retention_use import filtered
    return replace(state,constraints=facts,evidence_bound=True,
        retention_use=filtered(getattr(state,'retention_use',None),wanted|extra_keys))


def state_episode_ids(model,state):
    from scripts.ib_local.causal_relation_program import Port
    ids=[]
    for e in mechanism_evidence(model,state,Port.MOTHER):
        if not e.observed or e.encoded or not e.fact_id.startswith('episode:'):
            raise ValueError('only individually identified real facts may commit')
        identity=int(e.fact_id.split(':')[1])
        if identity not in ids:ids.append(identity)
    if len(ids)>len(state.valid):raise RuntimeError('canonical support exceeded finite budget')
    return ids


@torch.no_grad()
def install_episode_ids(bank,model,index,identities):
    """SSD IDs are authoritative; hot addresses are only a validated cache."""
    if len(identities)>bank.evidence_budget:raise RuntimeError('too many canonical facts')
    archive=model._causal_archive;m=model.memory
    wanted=set(identities)
    mapping={v['id']:int(k) for k,v in archive.hot.items()
             if v['id'] in wanted and bool(m.occupied[int(k)]) and archive._signature(int(k))==v['signature']}
    bank.support_episode_ids[index].zero_();bank.valid[index].zero_()
    for j,eid in enumerate(identities):
        if archive.db.execute('SELECT 1 FROM episodes WHERE id=?',(eid,)).fetchone() is None:
            raise ValueError('canonical evidence does not exist')
        bank.support_episode_ids[index,j]=eid
        if eid in mapping:
            slot=mapping[eid]
            bank.slots[index,j]=slot;bank.stamps[index,j]=m.last_write[slot];bank.valid[index,j]=True


@torch.no_grad()
def configure_support_ids(engine):
    bank=engine.model.causal_relations
    if hasattr(bank,'support_episode_ids'):return
    archive=getattr(engine.model,'_causal_archive',None)
    if archive is not None:archive.sync()
    bank.register_buffer('support_episode_ids',torch.zeros_like(bank.slots))
    if archive is None:return
    for i in torch.nonzero(bank.identity>0).flatten().tolist():
        rows=archive.db.execute('SELECT ordinal,episode_id FROM evidence WHERE relation_id=? ORDER BY ordinal',
                               (int(bank.identity[i]),)).fetchall()
        for ordinal,eid in rows:bank.support_episode_ids[i,ordinal]=eid
