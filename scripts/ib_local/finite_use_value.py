"""Score the body's bounded U, including the evidence it actually releases.

Wood integrates over its existing response hypotheses. Metal substitutes the
observed source, action and response in the same U. Old queries stay held out
and both scores use one current encoder, preventing coordinate drift from
being paid as evidence utility.
"""
from dataclasses import replace

import torch

from scripts.ib_local.causal_relation_program import (
    Evidence, Port, log_score, mechanism, response_hypotheses, revise,
)
from scripts.ib_local.causal_use_transport import clone_fact
from scripts.ib_local.relation_evidence import mechanism_evidence
from scripts.ib_local import run_navigation_residual_wood as _legacy_derived
_legacy_derived._DERIVED.add('_finite_use_last_expected')


def score_scope(engine, state, facts):
    values = []
    for fact in facts:
        d = mechanism(engine.model, state, fact.source, fact.action,
                      exclude_fact=fact.fact_id, ridge=engine.relation_ridge)
        values.append(log_score(d, engine.model.encoder(fact.outcome)).mean())
    return torch.stack(values) if values else engine.model.encoder.weight.new_empty(0)


def score_state(state):
    # The bound predictor consumes these constraints. Organization is updated
    # by the production U at commit, but does not enter this port's score.
    return replace(state, organization=None, variable_uses=())


@torch.no_grad()
def expected(engine, state, source, actions, distribution, facts):
    """Signed net utility for each action under the SAME finite revision rule."""
    state = score_state(state)
    if not facts:
        return actions.new_zeros(len(actions))
    from scripts.ib_local import joint_response, epistemic_value_reference as ev
    batch = getattr(engine.model, '_preview_batch_enabled', joint_response.enabled(engine.model))
    before = None if batch else score_scope(engine, state, facts)
    bank = engine.model.causal_relations
    _, hypotheses, _ = response_hypotheses(engine.model, distribution)
    gains = []
    after_states = []
    for b in range(len(actions)):
        samples = []
        for h in range(hypotheses.shape[1]):
            y = hypotheses[b:b+1, h]
            incoming = Evidence(source[b:b+1], actions[b:b+1], y,
                                Port.MOTHER, observed=False, encoded=True)
            updated = revise(state, incoming, model=engine.model)
            if batch:
                after_states.append(updated)
            else:
                samples.append((score_scope(engine, updated, facts)-before).mean())
        if not batch:
            gains.append(torch.stack(samples).mean())
    if batch:
        contexts = [ev.from_body(engine, st, facts) for st in [state, *after_states]]
        before, *after = ev.batched_context_scores(engine.model, contexts, ridge=engine.relation_ridge)
        result = torch.stack([(v-before).mean() for v in after]).reshape(len(actions), hypotheses.shape[1]).mean(-1)
    else:
        result = torch.stack(gains)
    cache = getattr(engine, '_finite_use_last_expected', None)
    if not isinstance(cache, dict):
        cache = {}
        engine._finite_use_last_expected = cache
    if len(cache) >= 4:
        cache.pop(next(iter(cache)))
    from scripts.ib_local.learnable_precision_scales import effective_ridge
    ridge_val = float(effective_ridge(engine.model, engine.relation_ridge))
    cache[id(distribution)] = dict(
        state_key=(state.identity, state.generation, state.version,
                   None if state.binding is None else state.binding.identity,
                   tuple(f.fact_id for f in state.constraints),
                   tuple(f.fact_id for f in state.extra)),
        facts_key=tuple(f.fact_id for f in facts),
        source_val=source.detach().clone(),
        actions_val=actions.detach().clone(),
        clock=int(engine.model.memory.clock),
        encoder_version=engine.model.encoder.weight._version,
        ridge=ridge_val,
        values=result.detach().clone(),
    )
    return result


@torch.no_grad()
def begin(engine, record, distribution, facts):
    organization=engine._program_question_state.organization
    orig_state = score_state(engine._program_question_state)
    state = replace(orig_state, constraints=tuple(clone_fact(f) for f in orig_state.constraints),
                    extra=tuple(clone_fact(f) for f in orig_state.extra))
    selected = record['choice']
    facts = tuple(clone_fact(f) for f in facts)
    cache = getattr(engine, '_finite_use_last_expected', None)
    cached = cache.get(id(distribution)) if isinstance(cache, dict) else None
    engine._finite_use_last_expected = None
    expected_val = None
    if cached is not None:
        s_key = (orig_state.identity, orig_state.generation, orig_state.version,
                 None if orig_state.binding is None else orig_state.binding.identity,
                 tuple(f.fact_id for f in orig_state.constraints),
                 tuple(f.fact_id for f in orig_state.extra))
        src = engine._internal_source.reshape(1, -1)
        act = engine._internal_action[selected:selected+1]
        from scripts.ib_local.learnable_precision_scales import effective_ridge
        if (cached['state_key'] == s_key and
                cached['facts_key'] == tuple(f.fact_id for f in facts) and
                cached['clock'] == int(engine.model.memory.clock) and
                cached['encoder_version'] == engine.model.encoder.weight._version and
                cached['ridge'] == float(effective_ridge(engine.model, engine.relation_ridge)) and
                selected < len(cached['values']) and
                cached['source_val'].shape[0] > selected and
                cached['actions_val'].shape[0] > selected and
                torch.equal(cached['source_val'][selected:selected+1], src) and
                torch.equal(cached['actions_val'][selected:selected+1], act)):
            expected_val = float(cached['values'][selected])
    if expected_val is None:
        # The action-conditioned distribution is sliced only on its query axis;
        # fitted evidence and posterior factors retain their original dimensions.
        d = dict(distribution)
        for key in ('query', 'mean', 'base', 'scale2', 'leverage'):
            if key in d:
                d[key] = d[key][selected:selected+1]
        value = expected(engine, state, engine._internal_source.reshape(1, -1),
                         engine._internal_action[selected:selected+1], d, facts)
        expected_val = float(value[0])
    engine.causal_use_pending['finite'] = dict(state=state, facts=facts,
        expected=expected_val, before_action=score_scope(engine, state, facts).tolist())
    from scripts.ib_local.relation_retention_use import begin_fresh
    engine.causal_use_pending['finite']['fresh']=begin_fresh(engine,state,
        engine._internal_source.reshape(1,-1),engine._internal_action[selected:selected+1])
    engine.causal_use_pending['finite']['fresh_organization']=(None if organization is None else
        begin_fresh(engine,organization,engine._internal_source.reshape(1,-1),
                    engine._internal_action[selected:selected+1]))
    if getattr(engine.model, 'reference_use_enabled', False):
        from scripts.ib_local.reference_use import begin as begin_references
        engine.causal_use_pending['finite']['fresh_references'] = begin_references(
            engine, engine._reference_use_boundary, engine._internal_source.reshape(1, -1),
            engine._internal_action[selected:selected+1])
        engine._reference_use_boundary = None


@torch.no_grad()
def observed(engine, receipt, slot, merged):
    """After actual episode allocation, before downstream reward consumers."""
    p = receipt['finite']
    target = engine._mother_boundary_target
    from scripts.ib_local.identified_event_credit import current
    event = current(engine)
    if (merged and event is None) or target is None:
        return [], dict(committed=False, reason='episode has no individual response constraint')
    memory = engine.model.memory
    # Allocate the persisted identity before scoring. Later boundary/credit
    # fields update this same record; solver ordering now matches the commit.
    archive = engine.model._causal_archive
    archive.sync()
    identity = f"episode:{archive.hot[str(slot)]['id']}"
    incoming = event if event is not None else Evidence(memory.source[slot:slot+1], memory.action[slot:slot+1],
        target.reshape(1, -1), Port.MOTHER, True, slot=slot,
        stamp=int(memory.last_write[slot]), fact_id=identity)
    state = p['state']
    updated = revise(state, incoming, model=engine.model)
    before = score_scope(engine, state, p['facts'])
    after = score_scope(engine, updated, p['facts'])
    comparisons = [dict(query=f.fact_id, query_stamp=f.stamp, before=float(a),
        after=float(b), contribution=float(b-a), expected=p['expected'])
        for f, a, b in zip(p['facts'], before, after)]
    supports = [f.fact_id for f in mechanism_evidence(engine.model, updated, Port.MOTHER)]
    engine._finite_use_commit_pending = dict(state=state, facts=p['facts'],
        before=before.clone(), after=after.clone(), supports=supports,
        source=incoming.source.clone(), action=incoming.action.clone(), outcome=incoming.outcome.clone(),
        retention_use=getattr(updated,'retention_use',None))
    from scripts.ib_local.relation_retention_use import finish_fresh
    reference_report = {}
    if 'fresh_references' in p:
        from scripts.ib_local.reference_use import settle
        reference_report['reference_use'] = settle(p['fresh_references'], target, incoming.fact_id)
    return comparisons, dict(committed=True, supports=supports,
        fresh_use=finish_fresh(engine,p.get('fresh'),target),
        fresh_organization=finish_fresh(engine,p.get('fresh_organization'),target),
        **reference_report,
        representation_shift=(float((before-before.new_tensor(p['before_action'])).mean())
                              if len(before) else 0.))


@torch.no_grad()
def verify_commit(engine, revision):
    """Bind the paid U to its actual SSD identity before the next actor update."""
    if getattr(engine, 'causal_use_arm', None) != 'C2':
        return
    p = getattr(engine, '_finite_use_commit_pending', None)
    if p is None:
        if revision['committed']:
            raise RuntimeError('finite U committed without its settlement receipt')
        return
    if not revision['committed']:
        raise RuntimeError('paid finite U did not commit')
    from scripts.ib_local.shared_unknown_bindings import _supports, _evidence
    archive = engine.model._causal_archive
    incoming = _evidence(archive, revision['observed_episode'])[0]
    for name in ('source', 'action', 'outcome'):
        torch.testing.assert_close(getattr(incoming, name), p[name], rtol=0, atol=0)
    supports = _supports(archive, revision['identity'])
    actual_ids = {f.fact_id for f in supports if f.kind == Port.MOTHER}
    expected_ids = set(p['supports'])
    if actual_ids != expected_ids:
        raise RuntimeError('paid finite U and persisted evidence differ')
    if p.get('retention_use') is not None:
        from scripts.ib_local.relation_retention_use import _load,filtered
        saved=_load(engine.model,p['retention_use'].scope)
        if saved != filtered(p['retention_use'],actual_ids):
            raise RuntimeError('paid finite U and persisted retention state differ')
    after = score_scope(engine, replace(p['state'], constraints=supports, extra=()), p['facts'])
    torch.testing.assert_close(after, p['after'], rtol=2e-5, atol=3e-6)
    error = float((after-p['after']).abs().max()) if len(after) else 0.
    engine.causal_use_last['finite_U'].update(verified=True, observed_episode=revision['observed_episode'],
        committed_score_max_error=error, supports=sorted(actual_ids))
    del engine._finite_use_commit_pending
