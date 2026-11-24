"""Execute known words without sampling or a Python per-byte neural loop.

The table still reads-before-writes every actual byte, using its normal Numba
recurrence. A known payload is one causal teacher-forced forward. A draft
already executed in the same frozen utterance scope needs no second forward.
"""
import numpy as np
import torch
from contextlib import nullcontext

from scripts.ib_local.language_action import peek
from scripts.ib_local.variable_expression import budget_scope


@torch.no_grad()
def known_film(engine, words, table):
    words = bytes(words)
    raw = np.frombuffer(bytes([engine.pending])+words, dtype=np.uint8)
    engine.decision.start(engine.recurrent)
    if words:
        _, probabilities, _ = table.run_credit(raw, max_window=len(words))
    else:
        probabilities = np.empty((0, 256), dtype=np.float32)
    final = peek(table, int(raw[-1]))
    probs = np.concatenate((probabilities, final[None]))
    ids = torch.as_tensor(raw.astype(np.int64), device=engine.device)[None]
    logp = torch.as_tensor(np.log(np.maximum(probs, 1e-6)), device=engine.device)[None]
    remaining = (1-torch.arange(len(raw), device=engine.device, dtype=logp.dtype)[None]/engine.max_utterance
                 if words else None)
    context = getattr(engine.decision, '_decode_context', nullcontext)
    with context(), budget_scope(engine.model, remaining):
        result = engine.model.predict_observed(ids, engine.recurrent, logp, decode=False)
    endpoint = result[3][-1].detach().clone()
    engine.decision.state.copy_(result[1])
    engine._draft_actions.append(endpoint.reshape(-1))
    engine._last_expression_receipt = dict(stops=[], final_probability=final,
        reason='reused_payload', length=len(words), cap=engine.max_utterance)
    remember(engine, words, list(probabilities), endpoint)
    return words, [], list(probabilities), 0.


def remember(engine, words, probabilities, endpoint):
    cache = getattr(engine, '_utterance_payload_cache', None)
    if cache is not None:
        cache[bytes(words)] = dict(probabilities=np.asarray(probabilities, np.float32).reshape(-1, 256),
            feature=endpoint.detach().clone(), state=engine.decision.state.detach().clone())


def deliver(engine, words, record):
    """Commit actual self-hearing; reuse only a same-scope executed draft."""
    cache = getattr(engine, '_utterance_payload_cache', None)
    receipt = None if cache is None else cache.get(bytes(words))
    if receipt is None:
        return None
    words = bytes(words)
    if words:
        raw = np.frombuffer(bytes([engine.pending])+words, np.uint8)
        _, probabilities, _ = engine.table.run_credit(raw, max_window=len(words))
        if not np.array_equal(probabilities, receipt['probabilities']):
            raise RuntimeError('frozen draft table differs from actual self-hearing')
        record['probs'].extend(probabilities)
    engine.decision.state.copy_(receipt['state'])
    record['truncated'] = False
    engine.model._expression_remaining = None
    ending = getattr(engine, '_selected_handover', {}) if record.get('choice') == 1 else {}
    record['ending_reason'] = ending.get('reason', 'reused_payload')
    record['truncated'] = record['ending_reason'] == 'budget'
    return words
