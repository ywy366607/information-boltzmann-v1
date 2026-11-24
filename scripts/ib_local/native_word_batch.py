"""Bounded, read-only batch execution of the ORIGINAL known-word programs.

Each trajectory starts at the same physical anchor with the current intention.
Padding affects only positions after its requested endpoint. No batch state,
word receipt, table mutation, or autograd graph enters the living individual.
"""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

import numpy as np
import torch


request = ContextVar('native_word_endpoint_request', default=None)


class EndpointReady(Exception):
    def __init__(self, features):
        self.features = features


@contextmanager
def endpoint_scope(model, lengths, *, raw=False):
    if torch.is_grad_enabled():
        raise ValueError('endpoint batching is an inference-only private branch')
    token = request.set((model, lengths, raw))
    try:
        yield
    finally:
        request.reset(token)


def table_payload(engine, words):
    """Only table probabilities are shared across distinct question contexts."""
    from scripts.ib_local.preview_runtime import read
    from scripts.ib_local.table_branch import table_branch
    from scripts.ib_local.language_action import peek
    table = engine.table
    def compute():
        raw = np.frombuffer(bytes([engine.pending])+words, dtype=np.uint8)
        with table_branch(table) as branch:
            if words:
                _, probabilities, _ = branch.run_credit(raw, max_window=len(words))
            else:
                probabilities = np.empty((0, 256), dtype=np.float32)
            final = peek(branch, int(raw[-1]))
        return raw.astype(np.int64), np.log(np.maximum(
            np.concatenate((probabilities, final[None])), 1e-6)).astype(np.float32)
    return read(engine.model, 'native_word_table_payload',
                (id(table), int(table.history), engine.pending, words), compute, anchors=(table,))


@torch.no_grad()
def endpoints(engine, words, *, batch_size=4):
    """Return one physical feature per payload; leave all engine state intact."""
    if (getattr(engine, 'native_factorized_endpoints', False) and
            getattr(engine.model, '_question_input_mask', None) is None):
        return factorized_endpoints(engine, words, batch_size=batch_size)
    return _endpoints(engine, words, batch_size=batch_size)


@torch.no_grad()
def factorized_endpoints(engine, words, *, batch_size=4):
    """Share the physical prefix across questions; apply the actual port last.

    The endpoint request exits at the experience-layer INPUT. Question FiLM
    and the budget enter at this very port; all earlier causal layers depend
    only on the immutable physical anchor and the known byte/table prefix.
    """
    from scripts.ib_local.preview_runtime import read_many, tensor_key
    from scripts.ib_local.question_language_port import condition_hidden
    from scripts.ib_local.variable_expression import budget_scope, condition_budget
    words = list(dict.fromkeys(bytes(w) for w in words))
    model = engine.model
    signature = (tuple(tensor_key(p) for p in model.parameters()), tensor_key(engine.recurrent),
                 id(engine.table), int(engine.table.history), engine.pending)
    keys = [(signature, word) for word in words]

    def compute(missing):
        raw = _endpoints(engine, [key[1] for key in missing], batch_size=batch_size, raw=True)
        return [raw[key[1]] for key in missing]

    raw = read_many(model, 'native_word_physical_prefix', keys, compute,
                    anchors=(engine.recurrent, engine.table))
    # Preserve the original per-payload gate and empty-word budget convention.
    result = {}
    for word, feature in zip(words, raw):
        conditioned = condition_hidden(model, feature[None])
        remaining = 1-len(word)/engine.max_utterance if word else 0.
        with budget_scope(model, remaining):
            conditioned = condition_budget(model, conditioned)
        result[word] = conditioned[0].detach().clone()
    return result


@torch.no_grad()
def _endpoints(engine, words, *, batch_size=4, raw=False):
    from scripts.ib_local.variable_expression import budget_scope
    model = engine.model
    words = list(dict.fromkeys(bytes(w) for w in words))
    if batch_size < 1:
        raise ValueError('positive bounded word batch required')
    result = {}
    # Refresh the same fixed fact/workspace projections once for this group.
    if hasattr(engine.decision, '_refresh_projections'):
        engine.decision._refresh_projections()
    context = getattr(engine.decision, '_decode_context', nullcontext)
    for start in range(0, len(words), batch_size):
        group = words[start:start+batch_size]
        payload = [table_payload(engine, w) for w in group]
        lengths = np.asarray([len(p[0]) for p in payload], dtype=np.int64)
        width = int(lengths.max())
        ids = np.zeros((len(group), width), dtype=np.int64)
        logp = np.zeros((len(group), width, 256), dtype=np.float32)
        for i, (tokens, probabilities) in enumerate(payload):
            ids[i, :len(tokens)] = tokens
            logp[i, :len(tokens)] = probabilities
        ids = torch.as_tensor(ids, device=engine.device)
        logp = torch.as_tensor(logp, device=engine.device)
        remaining = (1-torch.arange(width, device=engine.device, dtype=logp.dtype)[None]/engine.max_utterance).expand(len(group), -1).clone()
        # Original known_film uses no remaining-budget input for empty words.
        remaining[[not w for w in group]] = 0.
        recurrent = engine.recurrent.expand(-1, len(group), -1).contiguous()
        with context(), budget_scope(model, remaining), endpoint_scope(model, lengths, raw=raw):
            try:
                model.predict_observed(ids, recurrent, logp, decode=False)
            except EndpointReady as ready:
                feature = ready.features
            else:
                raise RuntimeError('original experience feature port was not reached')
        for word, value in zip(group, feature):
            result[word] = value[None].detach().clone()
    return result
