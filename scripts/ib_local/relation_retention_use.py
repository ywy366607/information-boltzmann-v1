"""Scope-bound delayed use, shared by real and encoded finite evidence U.

An immutable snapshot belongs to one relation generation or shared unknown.
Old evidence chooses the current release. The arrived MOTHER label then
updates surviving moments for later use. Only explicit real commits write SQL.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict
import copy
import json
import math

import torch


@dataclass(frozen=True)
class UseSnapshot:
    scope: str
    revision: int = 0
    mode: str = 'observe'
    moments: tuple = ()  # (episode identity, first, second, count)
    last_event: str = ''
    betas: tuple = (.9, .999)
    eps: float = 1e-8
    prior_scope: str = ''
    prior_revision: int = 0
    prior_moments: tuple = ()  # read-only parent evidence, separate from local observations


def with_prior(model, snapshot, parent, identities):
    """Read a same-fact parent only where this unknown has no local evidence."""
    if snapshot is None:
        return snapshot
    if getattr(model, 'retention_scope_prior_mode', None) != 'relation':
        return (replace(snapshot, prior_scope='', prior_revision=0, prior_moments=())
                if snapshot.prior_scope else snapshot)
    if parent is None or not parent.scope.startswith('relation:'):
        return replace(snapshot, prior_scope='', prior_revision=0, prior_moments=())
    if parent.betas != snapshot.betas or parent.eps != snapshot.eps:
        raise ValueError('parent and local retention moments require the same numerical units')
    wanted = set(identities)
    return replace(snapshot, prior_scope=parent.scope, prior_revision=parent.revision,
                   prior_moments=tuple(r for r in parent.moments if r[0] in wanted))


def effective_rows(snapshot):
    """Specific observed evidence wins; parent records supply a cold prior."""
    if snapshot is None:
        return {}
    rows = {r[0]: r[1:] for r in snapshot.prior_moments if r[3] > 0}
    rows.update({r[0]: r[1:] for r in snapshot.moments if r[3] > 0})
    return rows


def _bias_values(snapshot, identities):
    rows = effective_rows(snapshot)
    values = []
    for identity in identities:
        first, second, count = rows.get(identity,(0.,0.,0))
        if not count:
            values.append(0.); continue
        b1,b2 = snapshot.betas
        f = first/max(1-b1**count,snapshot.eps)
        s = second/max(1-b2**count,snapshot.eps)
        values.append(math.tanh(f/(math.sqrt(max(0.,s))+snapshot.eps)))
    return values


def biases(snapshot, identities, like):
    return like.new_tensor(_bias_values(snapshot, identities))


def priorities(x, identities, snapshot, ridge):
    gram = x@x.T
    eye = torch.eye(len(x),device=x.device,dtype=x.dtype)
    h = (gram@torch.linalg.solve(gram+ridge*eye,eye)).diag()
    if snapshot is not None and snapshot.mode == 'apply':
        b = biases(snapshot,identities,h)
        return h, h*b.exp()
    return h, h


def select(x, identities, snapshot, ridge):
    gram = x@x.T
    eye = torch.eye(len(x),device=x.device,dtype=x.dtype)
    h = (gram@torch.linalg.solve(gram+ridge*eye,eye)).diag()
    b_vals = _bias_values(snapshot, identities)
    b_cpu = torch.tensor(b_vals, dtype=h.dtype)
    apply_bias = snapshot is not None and snapshot.mode == 'apply'
    if apply_bias:
        p = h * b_cpu.to(device=h.device).exp()
        h_list, p_list = torch.stack((h, p)).detach().cpu().tolist()
    else:
        h_list = h.detach().cpu().tolist()
        p_list = h_list
    # Stable identity breaks exact numeric ties, not physical presentation order.
    order = (list(range(len(identities))) if snapshot is None else
             sorted(range(len(identities)),key=lambda i:identities[i]))
    index = order[min(range(len(order)), key=lambda idx: p_list[order[idx]])]
    geom_index = order[min(range(len(order)), key=lambda idx: h_list[order[idx]])]
    return index, dict(identities=list(identities), geometric=h_list,
        priority=p_list, biases=b_cpu.tolist(),
        released=identities[index], geometric_released=identities[geom_index],
        consumed_revision=0 if snapshot is None else snapshot.revision,
        **({} if snapshot is None or not snapshot.prior_scope else dict(
            prior_scope=snapshot.prior_scope, prior_revision=snapshot.prior_revision,
            prior_used=[identity for identity in identities
                if identity not in {r[0] for r in snapshot.moments if r[3] > 0}
                and identity in {r[0] for r in snapshot.prior_moments if r[3] > 0}])))


@torch.no_grad()
def contributions(model, facts, incoming, ridge):
    """Batched exact leave-one-fact-out Student-t predictive contribution.

    Facts/incoming are encoded role vectors (not tokens); all labels share the
    current coordinates and prior noise. The incoming label is absent from fits.
    This evaluates one MOTHER port, not action/organization total usefulness.
    """
    from scripts.ib_local.causal_relation_program import relation_features,posterior,log_score
    if not facts:
        return {}
    s,a,y = (torch.cat([getattr(f,n) for f in facts]) for n in ('source','action','outcome'))
    base = model.long_predictor(torch.cat((s,a),-1))
    base = base+model.mother_from_boundary(base)
    x, residual = relation_features(model,s,a), y-base
    q = relation_features(model,incoming.source,incoming.action)
    qb = model.long_predictor(torch.cat((incoming.source,incoming.action),-1))
    qb = qb+model.mother_from_boundary(qb)
    noise = model.relation_basis.noise(mother=True)
    d = posterior(x,residual,q,ridge,noise_prior=noise)
    d['mean'] = d['mean']+qb
    full = log_score(d,incoming.outcome).mean()
    n = len(facts)
    indices = torch.arange(n,device=x.device).expand(n,n)
    keep = indices[~torch.eye(n,device=x.device,dtype=torch.bool)].reshape(n,n-1)
    xx,yy = x[keep],residual[keep]
    g = xx@xx.transpose(-1,-2)
    eye = torch.eye(n-1,device=x.device,dtype=x.dtype)
    factor = torch.linalg.lu_factor(g+ridge*eye)
    cross = q@xx.transpose(-1,-2)
    solved = torch.linalg.lu_solve(*factor, torch.cat((yy, cross.transpose(-1,-2)), -1))
    coefficient, projected = solved.split((yy.shape[-1], 1), -1)
    leverage = ((q.square().sum(-1)-(cross*projected.transpose(-1,-2)).sum(-1))/ridge).clamp_min(0)
    alpha = 1+(n-1)/2
    rss = (yy.square().sum(-2)-(yy*(g@coefficient)).sum(-2)).clamp_min(0)
    scale = (noise+.5*rss)/alpha
    cut = dict(mean=(cross@coefficient).squeeze(-2)+qb,
        scale2=(1+leverage)*scale,df=2*alpha)
    values = (full-log_score(cut,incoming.outcome)).cpu().tolist()
    return {f.identity:float(v) for f,v in zip(facts,values)}


def advance(snapshot, kept_ids, benefits, event_identity):
    if snapshot is None:
        return None
    old = {r[0]:r[1:] for r in snapshot.moments}
    rows = []
    repeated = snapshot.last_event == event_identity
    for identity in sorted(set(kept_ids)):
        first,second,count = old.get(identity,(0.,0.,0))
        if identity in benefits and not repeated:
            value = benefits[identity]
            if not math.isfinite(value):
                raise FloatingPointError('nonfinite relation-use contribution')
            b1,b2 = snapshot.betas
            first,second,count = b1*first+(1-b1)*value, b2*second+(1-b2)*value**2,count+1
        rows.append((identity,first,second,count))
    return replace(snapshot,moments=tuple(rows),
        revision=snapshot.revision+(not repeated),last_event=event_identity)


def filtered(snapshot, identities):
    if snapshot is None:
        return None
    wanted = set(identities)
    return replace(snapshot,moments=tuple(r for r in snapshot.moments if r[0] in wanted),
                   prior_moments=tuple(r for r in snapshot.prior_moments if r[0] in wanted))


def active(model):
    return getattr(model,'relation_retention_use_mode',None) in ('observe','apply')


def load(model, scope):
    if not active(model):
        return None
    from scripts.ib_local.preview_runtime import read
    return read(model,'retention_scope',scope,lambda:_load(model,scope),anchors=(model,))


def _load(model, scope):
    archive = model._causal_archive
    rcache = getattr(archive, '_retention_cache', None)
    mode = model.relation_retention_use_mode
    if rcache is not None:
        cached = rcache.get((scope, mode))
        if cached is not None:
            return cached
    db = archive.db
    kind,identity,*_ = scope.split(':')
    table,key = ('relation_retention','relation_id') if kind == 'relation' else ('unknown_retention','unknown_id')
    actual = int(identity) if kind == 'relation' else scope
    row = db.execute(f'SELECT payload FROM {table} WHERE {key}=?',(actual,)).fetchone()
    data = None if row is None else json.loads(row[0])
    if data is not None and data['scope'] == scope:
        data['moments'] = tuple(tuple(r) for r in data['moments'])
        data['prior_moments'] = tuple(tuple(r) for r in data.get('prior_moments', ()))
        data['betas'] = tuple(data['betas'])
        result = replace(UseSnapshot(**data),mode=mode)
    else:
        m = model.memory
        result = UseSnapshot(scope,mode=mode,
                             betas=tuple(m.read_credit_betas),eps=m.read_credit_eps)
    if rcache is not None:
        if len(rcache) >= 1024:
            rcache.pop(next(iter(rcache)))
        rcache[(scope, mode)] = result
    return result


def commit(model, before, after):
    if before is None or after is None:
        return
    if before.scope != after.scope or len(after.moments)>model.causal_relations.evidence_budget:
        raise ValueError('scope or finite relation-use budget mismatch')
    current = _load(model,before.scope)
    if current == after:
        return  # identical recomputation is idempotent
    if current.revision != before.revision:
        raise RuntimeError('stale scope snapshot cannot pay delayed retention use')
    kind,identity,*_ = after.scope.split(':')
    table,key = ('relation_retention','relation_id') if kind == 'relation' else ('unknown_retention','unknown_id')
    actual = int(identity) if kind == 'relation' else after.scope
    archive = model._causal_archive
    archive.db.execute(f'INSERT OR REPLACE INTO {table} ({key},payload) VALUES (?,?)',
        (actual,json.dumps(asdict(after),allow_nan=False)))
    rcache = getattr(archive, '_retention_cache', None)
    if rcache is not None:
        for k in [k for k in rcache if k[0] == after.scope]:
            del rcache[k]


def revise_raw(model,state,evidence):
    """Pure select-old / update-later U for bound raw relations."""
    from scripts.ib_local.epistemic_value_reference import encode_fact
    from scripts.ib_local.relation_evidence import mechanism_evidence,evidence_key
    from scripts.ib_local.causal_relation_program import Port
    snapshot = getattr(state,'retention_use',None)
    # Canonical old facts include any preceding branch observation.
    facts = mechanism_evidence(model,state,Port.MOTHER)
    new_ids = {evidence_key(e) for e in state.extra+(evidence,)}
    from scripts.ib_local.shared_unknown_bindings import retain_constraints
    kept = retain_constraints(model,state.constraints,state.extra+(evidence,),len(state.valid),use=snapshot)
    all_kept = tuple(kept)+state.extra+(evidence,)
    from scripts.ib_local.learnable_precision_scales import effective_ridge
    ridge = effective_ridge(model, getattr(model,'causal_relation_ridge',.1))
    benefits = {}
    if snapshot is not None and evidence.kind == Port.MOTHER and evidence_key(evidence) not in {evidence_key(f) for f in facts}:
        benefits = contributions(model,tuple(encode_fact(model,f) for f in facts),encode_fact(model,evidence),
                                 ridge)
    updated = (advance(snapshot,{evidence_key(f) for f in all_kept},benefits,evidence_key(evidence))
               if evidence.kind == Port.MOTHER else snapshot)
    trace = dict(scope=None if snapshot is None else snapshot.scope,
        revision=0 if snapshot is None else snapshot.revision,
        old_ids=[evidence_key(f) for f in facts],retained=sorted({evidence_key(f) for f in all_kept}),
        released=sorted({evidence_key(f) for f in facts}-{evidence_key(f) for f in all_kept}),
        consumed=[] if snapshot is None else list(snapshot.moments),benefits=benefits)
    if snapshot is not None and snapshot.prior_scope:
        trace.update(prior_scope=snapshot.prior_scope, prior_revision=snapshot.prior_revision,
                     prior_consumed=list(snapshot.prior_moments))
    if snapshot is not None:
        candidates=tuple(f for f in facts if evidence_key(f) not in new_ids)
        if len({evidence_key(f) for f in candidates}|new_ids)>len(state.valid):
            from scripts.ib_local.causal_relation_program import relation_features
            from scripts.ib_local.preview_runtime import read, tensor_key
            cand_ids = [evidence_key(f) for f in candidates]
            def _sel():
                with torch.no_grad():
                    enc_c = [encode_fact(model, f) for f in candidates]
                    x = relation_features(model,
                        torch.cat([e.source for e in enc_c]),
                        torch.cat([e.action for e in enc_c]),
                        torch.cat([e.outcome for e in enc_c])).detach().cpu()
                    r_f = getattr(ridge, '_cached_float', None) if torch.is_tensor(ridge) else float(ridge)
                    if r_f is None:
                        r_f = float(ridge.detach())
                    return select(x, cand_ids, snapshot, r_f)[1]
            sel_key = (tuple((tensor_key(f.source), tensor_key(f.action), tensor_key(f.outcome)) for f in candidates),
                       snapshot, getattr(ridge, '_cached_float', None) if torch.is_tensor(ridge) else ridge)
            trace['selection'] = read(model, 'retention_trace_selection', sel_key, _sel, anchors=candidates)
    return kept,updated,trace


def configure(engine, mode, *, schema=2):
    if schema not in (1,2):
        raise ValueError('unknown relation retention transition schema')
    if mode not in ('observe','apply') or getattr(engine,'event_credit_mode',None) != 'identified':
        raise ValueError('relation use requires identified actual events')
    old = getattr(engine,'relation_retention_use_mode',None)
    old_schema=getattr(engine,'relation_retention_use_schema',1)
    if old is not None and old != mode:
        raise ValueError('restore a common source for another relation retention arm')
    if old is not None and schema < old_schema:
        raise ValueError('restore a legacy checkpoint to reproduce its original transition')
    db = engine.model._causal_archive.db
    db.executescript('''CREATE TABLE IF NOT EXISTS relation_retention (
        relation_id INTEGER PRIMARY KEY REFERENCES relations(id) ON DELETE CASCADE,
        payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS unknown_retention (
        unknown_id TEXT PRIMARY KEY REFERENCES unknown_bindings(id) ON DELETE CASCADE,
        payload TEXT NOT NULL);''')
    engine.model.relation_retention_use_mode = mode
    engine.model.relation_retention_use_schema = schema
    if old is not None and old_schema == schema:
        engine.relation_retention_use_schema = schema
        return
    from scripts.ib_local.navigation_direct_read import gain_state
    from scripts.ib_local.language_navigation_kanli_v4p import ActionEvidence
    engine.relation_retention_use_migration = dict(kind=engine.settlement_kind,
        from_schema=None if old is None else old_schema,to_schema=schema,
        previous_migration=copy.deepcopy(getattr(engine,'relation_retention_use_migration',None)),
        calibration=copy.deepcopy(engine.causal_use_calibration),
        value_evidence=copy.deepcopy(engine.causal_use_value_evidence.state_dict()),
        baseline=engine.baseline,critic_error=engine.critic_error,adv_ms=engine.adv_ms,
        gain_states={k:copy.deepcopy(getattr(engine,k)) for k in
            ('read_gain_state','internal_gain_state','shadow_gain_state')})
    engine.relation_retention_use_mode = mode
    engine.relation_retention_use_schema = schema
    engine.settlement_kind = 'finite_relation_use_v4' if schema == 1 else 'finite_relation_use_v5'
    engine.causal_use_calibration = dict(count=0,cross=0.,second=0.)
    engine.causal_use_value_evidence = ActionEvidence(engine.wood_decay).to(engine.device)
    object.__setattr__(engine.model,'_causal_use_value_evidence',engine.causal_use_value_evidence)
    engine.baseline,engine.critic_error,engine.adv_ms = 0.,None,None
    for k in ('read_gain_state','internal_gain_state','shadow_gain_state'):
        setattr(engine,k,gain_state(engine.wood_decay))


@torch.no_grad()
def begin_fresh(engine, state, source, action):
    """Freeze later-use validation BEFORE world arrival and all event updates."""
    if not active(engine.model):
        return None
    from scripts.ib_local.causal_relation_program import mechanism,Port
    from scripts.ib_local.relation_evidence import mechanism_evidence,evidence_key
    def freeze(d):
        return {k:(v.detach().clone() if torch.is_tensor(v) else v)
                for k,v in d.items() if k in ('mean','scale2','df')}
    full=freeze(mechanism(engine.model,state,source,action,ridge=engine.relation_ridge))
    old=mechanism_evidence(engine.model,state,Port.MOTHER)
    cut={evidence_key(f):freeze(mechanism(engine.model,state,source,action,
        ridge=engine.relation_ridge,exclude_fact=evidence_key(f))) for f in old}
    return dict(full=full,cut=cut,encoder=engine.model.encoder.weight.detach().clone(),
        bias=None if engine.model.encoder.bias is None else engine.model.encoder.bias.detach().clone(),
        scope=None if state.retention_use is None else state.retention_use.scope,
        consumed=[] if state.retention_use is None else list(state.retention_use.moments),
        **({} if state.retention_use is None or not state.retention_use.prior_scope else dict(
            prior_scope=state.retention_use.prior_scope,
            prior_revision=state.retention_use.prior_revision,
            prior_consumed=list(state.retention_use.prior_moments))))


@torch.no_grad()
def finish_fresh(engine, frozen, target):
    if frozen is None or target is None:
        return None
    from torch.nn import functional as F
    from scripts.ib_local.causal_relation_program import log_score
    y=F.linear(target.reshape(1,-1),frozen['encoder'],frozen['bias'])
    full=float(log_score(frozen['full'],y).mean())
    return dict(scope=frozen['scope'],score=full,
        mse=float((frozen['full']['mean']-y).square().mean()),
        benefits={key:full-float(log_score(d,y).mean()) for key,d in frozen['cut'].items()},
        consumed=frozen['consumed'],
        **{k: frozen[k] for k in ('prior_scope', 'prior_revision', 'prior_consumed') if k in frozen},
        timing='prediction and encoder frozen before actual response and learning; label used once on arrival')
