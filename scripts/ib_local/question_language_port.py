"""A scoped question is an input to the shared transition, not a byte prior.

The last self-heard byte still carries the intention into the recurrent state.
Its output already predicts the interlocutor: direct speech bias stops there.
No role label, answer, artificial question reward or extra recurrent state.
"""
from contextlib import contextmanager

import torch
from torch import nn


def enabled(model):
    return hasattr(model, 'question_to_hidden')


def condition_hidden(model, hidden):
    q = getattr(model, '_question_context', None)
    if q is None or not enabled(model):
        return hidden
    gain, shift = model.question_to_hidden(q).chunk(2, -1)
    # Existing layer gates now see content AND intention. The multiplicative
    # term also distinguishes contents before any recurrent propagation.
    rms = hidden.square().mean(-1, keepdim=True).add(1e-6).rsqrt()
    delta = hidden*rms*gain.tanh() + shift
    mask = getattr(model, '_question_input_mask', None)
    if mask is not None:
        if mask.shape != hidden.shape[:-1]:
            raise ValueError('question input scope must match actual input positions')
        delta = delta*mask[..., None]
    return hidden + delta


def condition_logits(model, logits):
    q = getattr(model, '_question_context', None)
    if q is None:
        return logits
    delta = model.question_to_logits(q)[None, None, :]
    mask = getattr(model, '_question_output_mask', None)
    if mask is not None:
        if mask.shape != logits.shape[:-1]:
            raise ValueError('question output scope must match predicted positions')
        delta = delta*mask[..., None]
    return logits + delta


@contextmanager
def observation_scope(model, own, length):
    """Scopes refer to input and predicted byte positions, respectively.

    Input 0 is the carried pending byte; input own is the final self-heard
    byte (or the pending byte for silence). Inputs after it are world bytes.
    Outputs 0..own-1 are speech; output own predicts the first world byte.
    """
    if not enabled(model):
        yield
        return
    if not 0 <= own < length:
        raise ValueError('a scoped action must be followed by a real observation')
    names = ('_question_input_mask', '_question_output_mask')
    previous = [getattr(model, name, None) for name in names]
    old_budget=getattr(model,'_expression_remaining',None)
    positions = torch.arange(length, device=model.encoder.weight.device)[None]
    model._question_input_mask = (positions <= own).to(model.encoder.weight.dtype)
    model._question_output_mask = (positions < own).to(model.encoder.weight.dtype)
    if hasattr(model,'expression_budget_to_hidden'):
        model._expression_remaining=((1-positions/128).clamp_min(0)*(positions<=own)*(own>0)).to(model.encoder.weight.dtype)
    try:
        yield
    finally:
        for name, value in zip(names, previous):
            setattr(model, name, value)
        model._expression_remaining=old_budget


def configure_content_question(engine):
    """Explicit additive migration; preserve all old weights and Adam moments.

    Zero initial residual preserves speech at migration. Listening immediately
    gets its correct output scope. Actual response loss learns the new port
    through the self-heard prefix and carried state; policy credit uses the
    same port for every sampled draft. This is not a separate language model.
    """
    model = engine.model
    if not hasattr(model, 'question_to_logits'):
        raise ValueError('requires the inherited question-conditioned body')
    if not enabled(model):
        w = model.encoder.weight
        with torch.random.fork_rng(devices=[w.device] if w.is_cuda else []):
            port = nn.Linear(model.encoder.out_features, 2*model.encoder.in_features,
                             bias=False, device=w.device, dtype=w.dtype)
        nn.init.zeros_(port.weight)
        model.question_to_hidden = port
        added = list(port.parameters())
        engine.optimizer.add_param_group(dict(params=added, lr=engine.optimizer.param_groups[0]['lr']))
        engine.params.extend(added)
        engine.groups = (*engine.groups, added)
    model._question_input_mask = model._question_output_mask = None
    engine.content_question_schema = 1
    return engine
