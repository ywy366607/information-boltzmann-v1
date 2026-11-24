"""Compare same-event response hypotheses through a read-only successor path."""
from pathlib import Path
from time import perf_counter

import torch

from scripts.ib_local.causal_relation_program import Evidence, Port, mechanism, revise
from scripts.ib_local.finite_use_value import score_scope
from scripts.ib_local.preview_runtime import preview_scope
from scripts.ib_local.prospective_decision import boundary, preview_bind
from scripts.ib_local.relation_evidence import mechanism_evidence
from scripts.ib_local.successor_distribution import training_distribution


def controls(prior, post, observed, chosen, alternative):
    return dict(prior=prior[alternative], copy_factual=observed,
                full_residual=observed+(prior[alternative]-prior[chosen]),
                native_U=post[alternative])


def supports(model, state):
    return [fact.fact_id for fact in mechanism_evidence(model, state, Port.MOTHER)]


@torch.no_grad()
def capture(engine, source, actions, momentum):
    return (source.detach().clone(), actions.detach().clone(),
            None if momentum is None else momentum.detach().clone(),
            int(engine.model.memory.clock))


def downstream(engine, before, queries, source, action, momentum, response):
    model, ridge = engine.model, engine.relation_ridge
    response = response[None]
    imagined = Evidence(source, action, response, Port.MOTHER, observed=False, encoded=True)
    branch = revise(before, imagined, model=model)
    mediated = mechanism(model, branch, source, action, kind=Port.DECISION,
                         mother=response, ridge=ridge)
    direct = mechanism(model, branch, source, action, kind=Port.DECISION,
                       momentum=momentum, ridge=ridge)
    successor = training_distribution(model, branch, mediated, direct)
    raw, _ = boundary(engine, branch, source, action, response, successor['mean'])
    addressed, address = preview_bind(model, raw, overlay=branch)
    scores = score_scope(engine, addressed, queries)
    return dict(successor=successor['mean'][0].tolist(), address=address,
                addressed_supports=supports(model, addressed),
                old_query_score=float(scores.mean()) if len(scores) else None)


@torch.no_grad()
def settle(engine, context, before, after, incoming, chosen, probabilities):
    """Keep the actual U; propagate each alternate only inside its own branch."""
    started = perf_counter()
    model, ridge = engine.model, engine.relation_ridge
    source, actions, momentum, clock = context
    observed = model.encoder(incoming.outcome)[0]
    alternatives = []
    from scripts.ib_local.causal_use_value import queries
    # Fixed old queries across all methods. The current observation and future
    # labels cannot enter this panel.
    panel = queries(engine, before)
    panel = [q for q in panel if q.fact_id != incoming.fact_id]
    with preview_scope(model):
        pre = mechanism(model, before, source, actions, ridge=ridge, momentum=momentum)
        post = mechanism(model, after, source, actions, ridge=ridge, momentum=momentum)
        for i, probability in enumerate(probabilities):
            if i == chosen or probability <= 0:
                continue
            p = None if momentum is None else momentum[i:i+1]
            means = controls(pre['mean'], post['mean'], observed, chosen, i)
            follow = ({k: downstream(engine, before, panel, source[i:i+1], actions[i:i+1], p, y)
                       for k, y in means.items()}
                      if getattr(engine, 'epistemic_value_schema', 0) in (4, 5, 6) else None)
            alternatives.append(dict(option=i, **{k: v.tolist() for k, v in means.items()},
                                     successor_use=follow))
    report = dict(chart='same post-learning parameters; actual functional revision',
                  observed_episode=incoming.fact_id, chosen=chosen, observed=observed.tolist(),
                  prediction_clock=clock, settlement_clock=int(model.memory.clock),
                  supports_before=list(pre['fact_identities']),
                  supports_after=list(post['fact_identities']),
                  query_identities=[q.fact_id for q in panel], alternatives=alternatives,
                  revision=after.retention_trace)
    directory = getattr(engine, 'counterfactual_response_capsules', None)
    if directory is not None:
        # This optional measurement artifact never enters the live archive.
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{int(model.memory.clock)}_{incoming.fact_id.replace(':', '_')}.pt"
        with path.open('xb') as stream:
            torch.save(report, stream)
        report['capsule'] = str(path)
    report['seconds'] = perf_counter()-started
    return report
