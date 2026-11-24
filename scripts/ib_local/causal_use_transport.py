"""Measure valuation across the actual finite U and its committed unknown.

The query identities are fixed before action. End-of-event comparisons use
one current encoder and one target, and exclude the query's own fact in every
fit. This observer never revises the live archive or changes a reward.
"""
from dataclasses import replace

import torch

from scripts.ib_local.causal_relation_program import Port, mechanism, log_score
from scripts.ib_local.relation_evidence import mechanism_evidence


def clone_fact(fact):
    return replace(fact, **{name: value.detach().clone() for name in
        ('source', 'action', 'outcome', 'mother')
        if torch.is_tensor(value := getattr(fact, name))})


@torch.no_grad()
def begin(engine):
    if not getattr(engine, 'causal_use_transport_audit', False):
        return
    state = engine._program_question_state
    if state.binding is None:
        raise ValueError('transport audit requires the actual shared unknown binding')
    facts = tuple(clone_fact(f) for f in mechanism_evidence(engine.model, state, Port.MOTHER))
    receipt = engine.causal_use_pending
    selected = {q['identity']: q for q in receipt['queries']}
    engine._use_transport_pending = dict(
        state=replace(state, constraints=facts, extra=(), evidence_bound=True,
                      organization=None, variable_uses=()),
        queries=tuple(f for f in facts if f.fact_id in selected),
        initial_scores={key: float(log_score(q['before'], q['target']).mean())
                        for key, q in selected.items()},
        predicted_x=receipt['incoming_x'].clone(),
        prediction_event=receipt['prediction_event'])


@torch.no_grad()
def finish(engine, row):
    if not getattr(engine, 'causal_use_transport_audit', False):
        return
    if 'use_transport' in row:
        return
    pending = engine._use_transport_pending
    state = pending['state']
    archive = engine.model._causal_archive
    from scripts.ib_local.shared_unknown_bindings import _supports, _evidence
    revision = row['shared_unknown']['revision']
    if not revision['committed']:
        row['use_transport'] = dict(committed=False, revision=revision)
        del engine._use_transport_pending
        return
    assert revision['identity'] == state.binding.identity
    incoming = _evidence(archive, revision['observed_episode'])[0]
    committed = _supports(archive, state.binding.identity)
    bounded = replace(state, constraints=committed)
    appended = replace(state, extra=(incoming,))
    old_ids = {f.fact_id for f in state.constraints}
    kept_ids = {f.fact_id for f in committed if f.kind == Port.MOTHER}
    values = []
    for fact in pending['queries']:
        target = engine.model.encoder(fact.outcome)
        scores = []
        supports = []
        for scope in (state, appended, bounded):
            d = mechanism(engine.model, scope, fact.source, fact.action,
                          exclude_fact=fact.fact_id, ridge=engine.relation_ridge)
            scores.append(float(log_score(d, target).mean()))
            supports.append([f.fact_id for f in mechanism_evidence(
                engine.model, scope, Port.MOTHER, exclude_fact=fact.fact_id)])
            assert fact.fact_id not in supports[-1]
        old, append, actual = scores
        values.append(dict(query=fact.fact_id, before_action=pending['initial_scores'][fact.fact_id],
            old_support_now=old, appended_now=append, committed_now=actual,
            append_gain=append-old, release_effect=actual-append, actual_U_gain=actual-old,
            representation_shift=old-pending['initial_scores'][fact.fact_id],
            supports=supports))
    actual_d = mechanism(engine.model, state, incoming.source, incoming.action,
                         ridge=engine.relation_ridge)
    count = len(values)
    row['use_transport'] = dict(schema=1, committed=True, unknown=state.binding.identity,
        measurement_event=('finite_U_commit_before_value_fit' if
                           getattr(engine, 'causal_use_arm', None) == 'C2' else 'end_of_event'),
        prediction_event=pending['prediction_event'], settlement_event=engine.autonomous_event_clock,
        incoming=incoming.fact_id, removed=sorted(old_ids-kept_ids), retained=sorted(old_ids&kept_ids),
        budget=len(state.valid), committed_count=len(kept_ids), queries=values,
        mean={key: sum(q[key] for q in values)/count if count else None
              for key in ('append_gain', 'release_effect', 'actual_U_gain', 'representation_shift')},
        decomposition_residual=max((abs(q['actual_U_gain']-q['append_gain']-q['release_effect'])
                                    for q in values), default=0.),
        incoming_feature_change=float((actual_d['query']-pending['predicted_x']).norm()),
        coordinate_contract='three end-of-event scores share current parameters and target; '
                            'before-action score is separately labelled representation shift')
    del engine._use_transport_pending
