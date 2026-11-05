"""Read the body's witness text with the current fast table, without writes.

Both before and after the interaction use exactly the same stored text and
context convention. The two readings use the corresponding fast-table states,
so their score difference includes both neural and fast-table changes.
"""
from __future__ import annotations

import numba
import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.language_action import _peek
from scripts.ib_local.language_navigation_v2 import FLOOR
from scripts.ib_local.ngram_fast_table import MASK64


@numba.njit(cache=True)
def _read_sequence(ids, rows, count, tag, err, orders, heads, slots, prior_weight, err_floor):
    out = np.empty((len(ids), 256), np.float32)
    history = np.uint64(0)
    for i in range(len(ids)):
        out[i] = _peek(ids[i], history, rows, count, tag, err, orders, heads, slots,
                       prior_weight, err_floor)
        history = ((history << np.uint64(8)) | np.uint64(ids[i])) & MASK64
    return out


def frozen_table_logp(table, ids):
    """Read without modifying table. ids are the actual concatenated witness IDs."""
    ids = np.ascontiguousarray(ids, dtype=np.uint8)
    probs = _read_sequence(ids, table.rows, table.count, table.tag, table.err,
        np.asarray(table.orders, np.int64), table.heads, 1 << table.slots_log2,
        table.prior_weight, table.err_floor)
    return np.log(np.maximum(probs, FLOOR))


@torch.no_grad()
def current_body_witness_nll(engine):
    if not engine.w_ids:
        return 0.
    ids, targets = (torch.cat(x, 1) for x in (engine.w_ids, engine.w_tgt))
    table_logp = torch.as_tensor(frozen_table_logp(engine.table, ids[0].cpu().numpy()),
                                 device=ids.device)[None]
    state = engine.model.initial_state(1, device=engine.device)
    total = 0.
    for i in range(0, ids.shape[1], 512):
        logits, state, _ = engine.model.predict(ids[:, i:i+512], state, table_logp[:, i:i+512])
        total += float(F.cross_entropy(logits[0], targets[0,i:i+512], reduction='sum'))
    return total / ids.shape[1]
