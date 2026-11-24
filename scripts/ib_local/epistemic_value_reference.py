"""Encoded reference for the body's existing finite-U predictive utility.

The state is (bounded identified evidence, held-out query scope), in the
CURRENT body's latent coordinates. This exposes the state needed by a value
query without reconstructing bytes, a GDN boundary, or an old autograd graph.
All learned operators, posterior, noise and quadrature are the production ones.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from scripts.ib_local.causal_relation_program import (
    Port, log_score, response_hypotheses, posterior, relation_features,
)


@dataclass(frozen=True)
class LatentFact:
    identity: str
    phase_identity: str
    source: torch.Tensor
    action: torch.Tensor
    outcome: torch.Tensor


@dataclass(frozen=True)
class LearningContext:
    facts: tuple[LatentFact, ...]
    queries: tuple[LatentFact, ...]
    budget: int
    retention_use: object = None
    prediction_shift: torch.Tensor | None = None


def encode_fact(model, fact):
    from scripts.ib_local.identity_preview import fact_identity
    from scripts.ib_local.relation_evidence import evidence_key
    from scripts.ib_local.preview_runtime import encoder_read, tensor_key
    identity, phase = evidence_key(fact), fact_identity(fact)
    key = (identity, phase, fact.encoded,
           *(tensor_key(t) for t in (fact.source, fact.action, fact.outcome)))
    return encoder_read(model.encoder, 'preview_encoded_fact', key,
        lambda: LatentFact(identity, phase, model.encoder(fact.source), model.encoder(fact.action),
                          fact.outcome if fact.encoded else model.encoder(fact.outcome)),
        anchors=(fact,))


def from_body(engine, state, queries, *, momentum=None):
    from scripts.ib_local.relation_evidence import mechanism_evidence
    facts = tuple(encode_fact(engine.model, f) for f in
                  mechanism_evidence(engine.model, state, Port.MOTHER))
    if len({f.identity for f in facts}) != len(facts):
        raise ValueError('finite epistemic state requires unique real fact identities')
    return LearningContext(facts, tuple(encode_fact(engine.model, q) for q in queries),
                           len(state.valid),getattr(state,'retention_use',None),
                           None if momentum is None else engine.model.momentum_to_boundary(momentum))


def response(model, context, source, action, *, ridge, exclude='', query_shift=None):
    """Same mother posterior as mechanism(), now with an explicit latent query."""
    from scripts.ib_local.learnable_precision_scales import effective_ridge
    ridge = effective_ridge(model, ridge)
    facts = tuple(f for f in context.facts if not exclude or f.identity != exclude)
    width = source.shape[-1]
    if facts:
        s = torch.cat([f.source for f in facts])
        a = torch.cat([f.action for f in facts])
        y = torch.cat([f.outcome for f in facts])
        old_base = model.long_predictor(torch.cat((s, a), -1))
        old_base = old_base+model.mother_from_boundary(old_base)
        x, residual = relation_features(model, s, a), y-old_base
    else:
        x = relation_features(model, source[:0], action[:0])
        residual = source.new_empty(0, width)
    base = model.long_predictor(torch.cat((source, action), -1))
    if query_shift is not None:
        base=base+query_shift
    base = base+model.mother_from_boundary(base)
    q = relation_features(model, source, action)
    d = posterior(x, residual, q, ridge, noise_prior=model.relation_basis.noise(mother=True))
    d.update(mean=d['mean']+base, base=base, query=q,
             fact_identities=tuple(f.phase_identity for f in facts))
    return d


def scores(model, context, *, ridge):
    from scripts.ib_local.prospective_decision import at_terminal
    if at_terminal():
        return batched_scores(model, context, ridge=ridge)
    return torch.stack([log_score(response(model, context, q.source, q.action,
        ridge=ridge, exclude=q.identity), q.outcome).mean() for q in context.queries])


def batched_scores(model, context, *, ridge):
    """Same own-fact-excluded Student-t scores, one batched kernel solve."""
    from scripts.ib_local.learnable_precision_scales import effective_ridge
    ridge = effective_ridge(model, ridge)
    if not context.queries:
        return model.encoder.weight.new_empty(0)
    if not context.facts:
        qs=torch.cat([q.source for q in context.queries]);qa=torch.cat([q.action for q in context.queries])
        target=torch.cat([q.outcome for q in context.queries])
        return log_score(response(model,context,qs,qa,ridge=ridge),target)
    facts=context.facts;queries=context.queries
    s,a,y=(torch.cat([getattr(f,n) for f in facts]) for n in ('source','action','outcome'))
    prior=model.long_predictor(torch.cat((s,a),-1));prior=prior+model.mother_from_boundary(prior)
    x=relation_features(model,s,a);residual=y-prior
    keep=x.new_tensor([[q.identity!=f.identity for f in facts] for q in queries])
    xx=x[None]*keep[...,None];yy=residual[None]*keep[...,None]
    qs,qa,target=(torch.cat([getattr(q,n) for q in queries]) for n in ('source','action','outcome'))
    query=relation_features(model,qs,qa)
    cross=(query[:,None,:]@xx.transpose(-1,-2))
    gram=xx@xx.transpose(-1,-2)
    kernel=gram+ridge*torch.eye(len(facts),device=x.device,dtype=x.dtype)
    factor=torch.linalg.lu_factor(kernel)
    solved=torch.linalg.lu_solve(*factor,torch.cat((yy,cross.transpose(-1,-2)),-1))
    coefficient,projected=solved.split((y.shape[-1],1),-1)
    mean=(cross@coefficient).squeeze(-2)
    leverage=((query.square().sum(-1)-(cross*projected.transpose(-1,-2)).sum((-1,-2)))/ridge).clamp_min(0)
    alpha=1+keep.sum(-1)/2
    rss=(yy.square().sum(-2)-(yy*(gram@coefficient)).sum(-2)).detach().clamp_min(0)
    noise=(model.relation_basis.noise(mother=True)+.5*rss)/alpha[:,None]
    base=model.long_predictor(torch.cat((qs,qa),-1));base=base+model.mother_from_boundary(base)
    return log_score(dict(mean=mean+base,scale2=(1+leverage[:,None])*noise,
                          df=2*alpha[:,None]),target)


def batched_context_scores(model, contexts, *, ridge):
    """Batch independent held-out scopes; each keeps its own facts and df.

    This is the original marginal PAYMENT score, even for joint predictive
    training. Padded rows are zero observations with zero evidence count.
    """
    from scripts.ib_local.learnable_precision_scales import effective_ridge
    ridge = effective_ridge(model, ridge)
    if not contexts:
        return []
    groups = []
    # Bound temporary B x evidence x latent storage on the small GPU.
    for start in range(0, len(contexts), 8):
        batch = contexts[start:start+8]
        active = [c for c in batch if c.queries]
        if not active:
            groups.extend(model.encoder.weight.new_empty(0) for _ in batch)
            continue
        n = max(1, max(len(c.facts) for c in active))
        # Deduplicate shared `kept` LatentFacts across the batch of contexts so
        # long_predictor / mother_from_boundary / relation_features run ONCE per
        # unique fact instead of repeating the same 7 facts 8-32 times.
        from scripts.ib_local.preview_runtime import tensor_key
        uniq_facts = []
        fact_idx_map = {}
        ctx_fact_ids = []
        for c in active:
            idxs = []
            for f in c.facts:
                fk = (tensor_key(f.source), tensor_key(f.action), tensor_key(f.outcome))
                pos = fact_idx_map.get(fk)
                if pos is None:
                    pos = len(uniq_facts)
                    fact_idx_map[fk] = pos
                    uniq_facts.append(f)
                idxs.append(pos)
            ctx_fact_ids.append(idxs)
        if uniq_facts:
            us = torch.cat([f.source for f in uniq_facts])
            ua = torch.cat([f.action for f in uniq_facts])
            uy = torch.cat([f.outcome for f in uniq_facts])
            uprior = model.long_predictor(torch.cat((us, ua), -1))
            uprior = uprior + model.mother_from_boundary(uprior)
            ux = relation_features(model, us, ua)
            ures = uy - uprior
        else:
            ref0 = active[0].queries[0].source
            ux = relation_features(model, ref0[:0], ref0[:0])
            ures = ref0[:0]
        xx, yy, masks, queries, targets = [], [], [], [], []
        for c, idxs in zip(active, ctx_fact_ids):
            if idxs:
                idx_t = torch.tensor(idxs, device=ux.device, dtype=torch.long)
                x, residual = ux[idx_t], ures[idx_t]
            else:
                x, residual = ux[:0], ures[:0]
            x = torch.nn.functional.pad(x, (0, 0, 0, n-len(c.facts)))
            residual = torch.nn.functional.pad(residual, (0, 0, 0, n-len(c.facts)))
            keep = x.new_tensor([[q.identity != f.identity for f in c.facts]
                                 + [False]*(n-len(c.facts)) for q in c.queries])
            xx.append(x[None]*keep[..., None]); yy.append(residual[None]*keep[..., None])
            masks.append(keep)
            queries.extend(c.queries); targets.append(len(c.queries))
        x, y, keep = torch.cat(xx), torch.cat(yy), torch.cat(masks)
        qs, qa, target = (torch.cat([getattr(q, k) for q in queries])
                          for k in ('source', 'action', 'outcome'))
        query = relation_features(model, qs, qa)
        cross = query[:, None]@x.transpose(-1, -2)
        gram = x@x.transpose(-1, -2)
        kernel = gram+ridge*torch.eye(n, device=x.device, dtype=x.dtype)
        factor = torch.linalg.lu_factor(kernel)
        solved = torch.linalg.lu_solve(*factor, torch.cat((y, cross.transpose(-1, -2)), -1))
        coefficient, projected = solved.split((y.shape[-1], 1), -1)
        correction = (cross@coefficient).squeeze(-2)
        leverage = ((query.square().sum(-1)-(cross*projected.transpose(-1, -2)).sum((-1, -2)))/ridge).clamp_min(0)
        alpha = 1+keep.sum(-1)/2
        rss = (y.square().sum(-2)-(y*(gram@coefficient)).sum(-2)).detach().clamp_min(0)
        noise = (model.relation_basis.noise(mother=True)+.5*rss)/alpha[:, None]
        base = model.long_predictor(torch.cat((qs, qa), -1))
        base = base+model.mother_from_boundary(base)
        scored = log_score(dict(mean=correction+base, scale2=(1+leverage[:, None])*noise,
                                df=2*alpha[:, None]), target)
        split = iter(scored.split(targets))
        groups.extend(next(split) if c.queries else scored[:0] for c in batch)
    return groups


def _kept_support(model, context, identity, ridge):
    kept = [f for f in context.facts if f.identity != identity]
    while len(kept)+1 > context.budget:
        s, a, y = (torch.cat([getattr(f, name) for f in kept])
                   for name in ('source', 'action', 'outcome'))
        x = relation_features(model, s, a, y)
        from scripts.ib_local.relation_retention_use import select
        index,_=select(x,[f.identity for f in kept],getattr(context,'retention_use',None),ridge)
        del kept[index]
    return kept


def revise(model, context, incoming, *, ridge):
    """Full shadow transition, including retention credit used by future releases."""
    kept = _kept_support(model, context, incoming.identity, ridge)
    from scripts.ib_local.relation_retention_use import contributions,advance
    snapshot=getattr(context,'retention_use',None)
    benefit=(contributions(model,context.facts,incoming,ridge) if snapshot is not None and
             incoming.identity not in {f.identity for f in context.facts} else {})
    use=advance(snapshot,[f.identity for f in kept]+[incoming.identity],benefit,incoming.identity)
    # Same canonical lexicographic fact-identity presentation as production.
    return replace(context, facts=tuple(sorted((*kept, incoming), key=lambda f:f.phase_identity)),
                   retention_use=use)


@torch.no_grad()
def expected(model, context, source, actions, *, ridge, distribution=None):
    """The SAME expected local use, with (R,Q) supplied rather than discarded."""
    if not context.queries:
        return actions.new_zeros(len(actions))
    source = source.expand_as(actions)
    d = (response(model, context, source, actions, ridge=ridge,query_shift=context.prediction_shift)
         if distribution is None else distribution)
    bank = model.causal_relations
    _, hs, _ = response_hypotheses(model, d)
    from scripts.ib_local.joint_response import enabled
    batch = getattr(model, '_preview_batch_enabled', enabled(model))
    before = None if batch else scores(model, context, ridge=ridge)
    values = []
    after_contexts = []
    # All samples have the same new identity. Release uses the OLD evidence
    # and its use snapshot; the immediate held-out score reads only the new
    # fact set. No after-context here is carried into a subsequent revision.
    identity = 'event:-1:-1'
    if any(f.identity == identity for f in context.facts):
        identity = 'shadow:'+str(sum(f.identity.startswith(('event:', 'shadow:')) for f in context.facts))
    kept = _kept_support(model, context, identity, ridge)
    for i in range(len(actions)):
        sample_values = []
        for h in range(hs.shape[1]):
            y = hs[i:i+1, h]
            f = LatentFact(identity, 'mother:0:'+identity,
                           source[i:i+1], actions[i:i+1], y)
            after = replace(context, facts=tuple(sorted((*kept, f), key=lambda fact:fact.phase_identity)),
                            retention_use=None)
            if batch:
                after_contexts.append(after)
            else:
                sample_values.append((scores(model, after, ridge=ridge)-before).mean())
        if not batch:
            values.append(torch.stack(sample_values).mean())
    if batch:
        before, *after = batched_context_scores(model, [context, *after_contexts], ridge=ridge)
        return torch.stack([(s-before).mean() for s in after]).reshape(len(actions), hs.shape[1]).mean(-1)
    return torch.stack(values)


@torch.no_grad()
def continuation(model, context, latent, *, reward_valid, ridge, calibration=1.):
    """Original bounded recalled-action prior, valued with explicit learning state."""
    m = model.memory
    eligible = (m.occupied & reward_valid & m.internal_valid & (m.count == 1) &
        (m.internal_stamp == m.last_write) & (m.wood_option >= 0) &
        (m.wood_target_count == m.count))
    read = m.read_encoded(latent, model.encoder, eligible=eligible)
    values, detail = [], []
    for i in range(len(latent)):
        valid = eligible[read.slots[i]]
        slots = read.slots[i][valid]
        if not len(slots):
            values.append(latent.new_zeros(())); detail.append(dict(slots=[], gains=[])); continue
        old_s = model.encoder(m.source[slots])
        old_a = model.encoder(m.action[slots])
        z = latent[i:i+1]
        gains = 10*calibration*expected(model, context, z,
            z+old_a-old_s, ridge=ridge)
        prior = read.weights[i][valid]
        weights = (gains+prior.clamp_min(1e-30).log()).softmax(-1)
        reference = gains[0]
        values.append(reference+(weights*(gains-reference)).sum())
        detail.append(dict(slots=slots.tolist(), gains=gains.tolist(), weights=weights.tolist()))
    return torch.stack(values), detail


def configure(engine, *, schema=1, word_execution='historical'):
    """Explicit experimental route; persistent marker plus runtime reattachment."""
    if getattr(engine, 'causal_use_arm', None) != 'C2':
        raise ValueError('epistemic continuation requires the actual finite-U settlement')
    if getattr(engine.model, 'identity_preview_revision', 0) != 1:
        raise ValueError('identity-coupled quadrature must already be restored')
    if schema not in (1,2,3,4,5,6):
        raise ValueError('unknown epistemic continuation contract')
    if word_execution not in ('historical','native_anchor') or (word_execution!='historical' and schema not in (3,4,5,6)):
        raise ValueError('native self execution requires schema3')
    if schema>=2 and not hasattr(engine,'bfss_momenta'):
        raise ValueError('momentum continuation requires the actual persistent P')
    if schema>=3 and getattr(engine,'gain_units_schema',0)!=1:
        raise ValueError('native decision continuation requires repaired actor gain units')
    if schema in (4,5,6) and (word_execution!='native_anchor' or getattr(engine.model,'joint_relation_update_schema',None)!=1):
        raise ValueError('distribution continuation requires native self execution and coordinated output')
    engine.epistemic_value_schema = schema
    engine.epistemic_word_execution = word_execution
    object.__setattr__(engine.model, '_epistemic_value_engine', engine)


@torch.no_grad()
def branch_value(model, state, latent, *, momentum=None, source=None, action=None, mother=None,
                 decision_distribution=None):
    """Value the next action after U, retaining the branch's learning state.

    Q follows the next local finite-U query policy: the bounded retained
    facts, including the response known within this hypothetical branch.
    Existing current-query payment keeps its own pre-world frozen Q.
    """
    from scripts.ib_local.causal_use_value import queries, calibrated
    engine = model._epistemic_value_engine
    if getattr(engine,'epistemic_value_schema',1)>=2 and momentum is None:
        raise ValueError('momentum-conditioned continuation requires its known P read')
    if getattr(engine,'epistemic_value_schema',1)>=3:
        from scripts.ib_local.prospective_decision import value,at_terminal
        if not at_terminal():
            if source is None or action is None or mother is None:
                raise ValueError('native decision continuation requires its originating typed transition')
            if engine.epistemic_value_schema in (4,5,6):
                if decision_distribution is None:
                    raise ValueError('successor continuation requires the complete decision distribution')
                from scripts.ib_local.successor_distribution import value as distribution_value
                return distribution_value(engine,state,source,action,mother,decision_distribution,momentum)
            return value(engine,state,source,action,mother,latent,momentum)
    context = from_body(engine, state, queries(engine, state),
        momentum=momentum if getattr(engine,'epistemic_value_schema',1)>=2 else None)
    context = replace(context, queries=context.facts)
    value, _ = continuation(model, context, latent,
        reward_valid=engine.reward_label_valid(wood=True), ridge=engine.relation_ridge,
        calibration=float(calibrated(engine, 1.)))
    return value
