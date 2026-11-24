"""The body's existing finite selectors, with scoped conditional execution.

Normal execution makes exactly the original categorical draw. An explicit
private audit may fix a named choice while consuming that same draw, so all
other sampling retains its original RNG position. Nothing is persisted here.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch


@dataclass
class ChoiceTrace:
    overrides: dict = field(default_factory=dict)
    events: list = field(default_factory=list)


_trace = ContextVar('native_finite_choice_trace', default=None)


@contextmanager
def trace_choices(overrides=None):
    trace = ChoiceTrace(dict(overrides or {}))
    token = _trace.set(trace)
    try:
        yield trace
    finally:
        _trace.reset(token)


def choose(probability, *, stage):
    """Preserve the original draw and permit one scoped conditional choice."""
    sampled = int(torch.multinomial(probability, 1))
    trace = _trace.get()
    if trace is None:
        return sampled
    if any(e['stage'] == stage for e in trace.events):
        raise RuntimeError(f'duplicate finite-choice stage: {stage}')
    selected = trace.overrides.get(stage, sampled)
    if not 0 <= selected < len(probability) or not bool(probability[selected] > 0):
        raise ValueError('conditional choice must have positive actual probability')
    trace.events.append(dict(stage=stage, sampled=sampled, selected=selected,
                             probability=probability.detach().clone()))
    return selected


def tree_expectation(question_probability, proposal_probabilities,
                     action_probabilities, action_values):
    """Contract the actual question -> proposal -> action finite tree.

    The supplied action values are the original finite-U expected payment in
    Wood units, at each leaf's actual Q, endpoint, P and original actor view.
    Draft byte randomness is conditioned on the recorded common draws.
    """
    if not (len(question_probability) == len(proposal_probabilities) ==
            len(action_probabilities) == len(action_values)):
        raise ValueError('matched question branches required')
    by_question, by_proposal, leaf_mass = [], [], []
    for p, actors, values in zip(proposal_probabilities, action_probabilities, action_values):
        if not (len(p) == len(actors) == len(values)):
            raise ValueError('matched proposal branches required')
        local = torch.stack([(a*q).sum() for a, q in zip(actors, values)])
        by_proposal.append(local)
        by_question.append((p*local).sum())
        leaf_mass.append(torch.stack([w*a.sum() for w, a in zip(p, actors)]).sum())
    by_question = torch.stack(by_question)
    return dict(expected=(question_probability*by_question).sum(),
                by_question=by_question, by_proposal=by_proposal,
                total_mass=(question_probability*torch.stack(leaf_mass)).sum())
