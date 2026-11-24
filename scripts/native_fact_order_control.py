"""Actual fact-row permutation control, retaining all references and contents.

Only an exactly matching eight-fact observed support is reordered at the
shared fact-view boundary. Prediction, organization and action proposals all
consume this same view. No operator, learner or random draw is added.
"""
from __future__ import annotations

import functools
import sys
from dataclasses import replace


def install(observer, specification):
    from scripts.ib_local import relation_evidence as re
    original = re.mechanism_evidence
    target = specification['target_order']
    assert len(target) == len(set(target))
    calls, undo = [], []

    @functools.wraps(original)
    def reordered(model, state, kind, *, exclude=-1, exclude_fact=''):
        if specification.get('boundary') == 'stored_fact_presentation':
            revised_state = replace(state, constraints=tuple(reversed(state.constraints)),
                                    extra=tuple(reversed(state.extra)))
            if len(state.constraints) > 1 or len(state.extra) > 1:
                receipt = dict(event=observer.clock, external_cycle=observer.external_cycle,
                    update_clock=observer.update_clock, relation=[state.identity,state.generation],
                    port=kind.value, contents_preserved=True, registered_references_unchanged=True,
                    boundary='stored_fact_presentation', items=len(state.constraints))
                calls.append(receipt)
                observer.record('actual_fact_order_control',receipt)
            return original(model,revised_state,kind,exclude=exclude,exclude_fact=exclude_fact)
        facts=original(model,state,kind,exclude=exclude,exclude_fact=exclude_fact)
        keys = [e.fact_id for e in facts]
        if (len(facts) == len(target) and set(keys) == set(target)
                and all(e.observed for e in facts) and keys != target):
            indexed = {e.fact_id:e for e in facts}
            revised = tuple(indexed[k] for k in target)
            receipt = dict(event=observer.clock, external_cycle=observer.external_cycle,
                update_clock=observer.update_clock, original_order=keys, target_order=target,
                relation=[state.identity,state.generation],port=kind.value,contents_preserved=True,
                registered_references_unchanged=True)
            calls.append(receipt)
            observer.record('actual_fact_order_control',receipt)
            return revised
        return facts

    for module in list(sys.modules.values()):
        if module is not None and getattr(module,'__name__','').startswith('scripts.ib_local'):
            if getattr(module,'mechanism_evidence',None) is original:
                undo.append(module)
                setattr(module,'mechanism_evidence',reordered)

    def close():
        for module in undo:
            setattr(module,'mechanism_evidence',original)
    return calls,close
