"""Finite-index real-text retrieval world, not an interactive human dialogue.

A byte utterance affects where a fixed OWT source is read. The world sees the
utterance only; it never reads the body's hidden state or unseen alternatives.
The continuation after the retrieved passage is an untrained generalization
probe. Empty or unmatched utterances continue from the world's existing cursor.
"""
from __future__ import annotations

import numba
import numpy as np


@numba.njit(cache=True, inline='always')
def _hash4(a, b, c, d):
    x = np.uint64(a) | (np.uint64(b) << np.uint64(8)) | (np.uint64(c) << np.uint64(16)) | (np.uint64(d) << np.uint64(24))
    x ^= x >> np.uint64(16)
    x *= np.uint64(0x7feb352d)
    x ^= x >> np.uint64(15)
    return x


@numba.njit(cache=True)
def _build_index(data, buckets, positions):
    # Retain a fixed number of genuine positions per bucket. Verification at
    # retrieval rejects hash collisions. -1 means empty; deterministic order.
    mask = buckets-1
    for i in range(len(data)-3):
        h = int(_hash4(data[i], data[i+1], data[i+2], data[i+3]) & np.uint64(mask))
        for j in range(positions.shape[1]):
            if positions[h, j] < 0:
                positions[h, j] = i
                break


@numba.njit(cache=True)
def _select(query, data, positions, required, fallback):
    mask = len(positions)-1
    best, best_score = -1, 0
    for i in range(len(query)-3):
        h = int(_hash4(query[i], query[i+1], query[i+2], query[i+3]) & np.uint64(mask))
        for j in range(positions.shape[1]):
            p = int(positions[h, j])
            if p < 0 or p+4+required > len(data):
                continue
            if data[p] != query[i] or data[p+1] != query[i+1] or data[p+2] != query[i+2] or data[p+3] != query[i+3]:
                continue
            # Match additional utterance bytes around this 4-gram, without
            # looking at the continuation that will become the response.
            score = 4
            left = min(i, p)
            for n in range(1, left+1):
                if query[i-n] != data[p-n]: break
                score += 1
            right = min(len(query)-i-4, len(data)-p-4)
            for n in range(right):
                if query[i+4+n] != data[p+4+n]: break
                score += 1
            if score > best_score:
                # Start AFTER the matched query span, no duplicate as response.
                end = p+4
                for n in range(right):
                    if query[i+4+n] != data[p+4+n]: break
                    end += 1
                if end+required <= len(data):
                    best, best_score = end, score
    if best < 0:
        if fallback+required > len(data):
            return -1, 0
        return fallback, 0
    return best, best_score


class LanguageNavigationWorld:
    def __init__(self, data, buckets=1 << 17, per_bucket=8, cursor=0):
        if data.dtype != np.uint8 or data.ndim != 1 or buckets <= 0 or buckets & (buckets-1):
            raise ValueError('world requires 1D bytes and power-of-two buckets')
        if len(data) >= np.iinfo(np.int32).max or per_bucket < 1 or not 0 <= cursor <= len(data):
            raise ValueError('invalid bounded index or cursor')
        self.data = data
        self.positions = np.full((buckets, per_bucket), -1, dtype=np.int32)
        _build_index(data, buckets, self.positions)
        self.cursor = int(cursor)
        # Environment-side evaluation provenance, not a reward/food variable
        # in the body. Fixed by corpus size, independent of elapsed history.
        self.seen = np.zeros(len(data), dtype=np.bool_)

    def mark_seen(self, start, end):
        if not 0 <= start <= end <= len(self.data):
            raise ValueError('invalid source-position range')
        self.seen[start:end] = True

    def state_dict(self):
        return dict(cursor=self.cursor, seen=self.seen.copy())

    def load_state_dict(self, state):
        if (not 0 <= state['cursor'] <= len(self.data) or
                state['seen'].shape != self.seen.shape or state['seen'].dtype != np.bool_):
            raise ValueError('incompatible world provenance state')
        self.cursor = int(state['cursor'])
        self.seen[:] = state['seen']

    def take(self, utterance, response_bytes, holdout_bytes):
        query = np.frombuffer(utterance, np.uint8)
        required = response_bytes+holdout_bytes
        if response_bytes <= 0 or holdout_bytes <= 0:
            raise ValueError('response and heldout budgets must be positive')
        start, matched = _select(query, self.data, self.positions, required, self.cursor)
        if start < 0:
            raise EOFError('world exhausted, no silent wrap')
        end = start+required
        # Count exposure before this event; response and probe are disjoint.
        # Same text at a different position can still be familiar.
        prior_response = int(self.seen[start:start+response_bytes].sum())
        prior_holdout = int(self.seen[start+response_bytes:end].sum())
        self.mark_seen(start,end)
        self.cursor = end
        return (self.data[start:start+response_bytes].copy(),
                self.data[start+response_bytes:end].copy(),
                dict(start=int(start), end=int(end), match=int(matched),
                     response_previously_seen_bytes=prior_response,
                     holdout_previously_seen_bytes=prior_holdout,
                     holdout_position_fresh=(prior_holdout == 0),
                     source='matched' if matched else 'continuation'))
