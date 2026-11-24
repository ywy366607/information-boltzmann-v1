"""Value an acquired response by its use on another contact with the same problem."""
from dataclasses import replace
import torch
from scripts.ib_local.causal_relation_program import Port, mechanism, log_score


@torch.no_grad()
def acquisition_quality(model, state, background, acquired, later, ridge):
    """Inputs are raw MOTHER facts. The caller binds problem signatures and forks.

    Separate calls per branch prevent branch-local episode numbers from being
    mistaken for the same evidence. No text, navigation or provider interface.
    """
    common = tuple(background)
    for f in (*common, acquired, later):
        if f.kind != Port.MOTHER or f.encoded or not f.observed:
            raise ValueError('requires actual raw response evidence')
    if len(common)+1 > len(state.valid) or later.stamp <= acquired.stamp:
        raise ValueError('requires bounded support and a later contact')
    def score(facts):
        s = replace(state, constraints=facts, extra=(), organization=None, variable_uses=(), evidence_bound=True)
        d = mechanism(model,s,later.source,later.action,ridge=ridge)
        return float(log_score(d, model.encoder(later.outcome)).mean())
    without, full = score(common), score(common+(acquired,))
    return dict(without=without, with_evidence=full, benefit=full-without)
