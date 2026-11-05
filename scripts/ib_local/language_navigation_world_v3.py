"""Broad indexed OWT world with bounded, reproducible reservoir sampling.

Only the world sees its corpus and RNG. The exposure bitmap records evaluation
provenance; it never chooses a response. A retrieval is not a human dialogue.
"""
from __future__ import annotations

import numba
import numpy as np

from scripts.ib_local.language_navigation_world import LanguageNavigationWorld, _hash4


@numba.njit(cache=True)
def reservoir_index(data, positions, seed):
    """Uniform reservoir over occurrences in each hash bucket, collisions included."""
    np.random.seed(seed)
    counts = np.zeros(len(positions), dtype=np.int64)
    mask = len(positions) - 1
    slots = positions.shape[1]
    for i in range(len(data) - 3):
        h = int(_hash4(data[i], data[i + 1], data[i + 2], data[i + 3]) & np.uint64(mask))
        counts[h] += 1
        n = counts[h]
        if n <= slots:
            positions[h, n - 1] = i
        else:
            pick = np.random.randint(0, n)
            if pick < slots:
                positions[h, pick] = i


@numba.njit(cache=True)
def random_best(query, data, positions, required, fallback, seed):
    """Uniformly choose among encountered maximum-score hits, not unique spans."""
    np.random.seed(seed)
    mask = len(positions) - 1
    best, score_best, ties = -1, 0, 0
    for i in range(len(query) - 3):
        h = int(_hash4(query[i], query[i+1], query[i+2], query[i+3]) & np.uint64(mask))
        for j in range(positions.shape[1]):
            p = int(positions[h, j])
            if p < 0 or p + 4 + required > len(data):
                continue
            if data[p] != query[i] or data[p+1] != query[i+1] or data[p+2] != query[i+2] or data[p+3] != query[i+3]:
                continue
            score = 4
            for n in range(1, min(i, p) + 1):
                if query[i-n] != data[p-n]:
                    break
                score += 1
            end = p + 4
            for n in range(min(len(query)-i-4, len(data)-p-4)):
                if query[i+4+n] != data[p+4+n]:
                    break
                score += 1
                end += 1
            if end + required > len(data):
                continue
            if score > score_best:
                best, score_best, ties = end, score, 1
            elif score == score_best:
                ties += 1
                if np.random.randint(0, ties) == 0:
                    best = end
    if best < 0:
        return (fallback, 0) if fallback + required <= len(data) else (-1, 0)
    return best, score_best


class BroadLanguageWorld(LanguageNavigationWorld):
    def __init__(self, data, buckets=1 << 17, per_bucket=8, cursor=0, seed=11):
        if data.dtype != np.uint8 or data.ndim != 1 or buckets <= 0 or buckets & (buckets - 1):
            raise ValueError('world requires 1D bytes and power-of-two buckets')
        if len(data) >= np.iinfo(np.int32).max or per_bucket < 1 or not 0 <= cursor <= len(data):
            raise ValueError('invalid bounded index or cursor')
        self.data = data
        self.positions = np.full((buckets, per_bucket), -1, dtype=np.int32)
        self.seed = int(seed)
        reservoir_index(data, self.positions, self.seed)
        self.rng = np.random.default_rng(self.seed + 1)
        self.cursor = int(cursor)
        self.seen = np.zeros(len(data), dtype=np.bool_)

    def state_dict(self):
        state = super().state_dict()
        state.update(seed=self.seed, rng=self.rng.bit_generator.state)
        return state

    def load_state_dict(self, state):
        if state['seed'] != self.seed:
            raise ValueError('world index seed mismatch')
        super().load_state_dict(state)
        self.rng.bit_generator.state = state['rng']

    def take(self, utterance, response_bytes, holdout_bytes):
        if response_bytes <= 0 or holdout_bytes <= 0:
            raise ValueError('response and heldout budgets must be positive')
        required = response_bytes + holdout_bytes
        query = np.frombuffer(utterance, np.uint8)
        draw = int(self.rng.integers(0, 2**32, dtype=np.uint64))
        start, score = random_best(query, self.data, self.positions, required, self.cursor, draw)
        if start < 0:
            raise EOFError('world exhausted, no silent wrap')
        end = start + required
        response_prior = int(self.seen[start:start+response_bytes].sum())
        holdout_prior = int(self.seen[start+response_bytes:end].sum())
        self.mark_seen(start, end)
        self.cursor = end
        return (self.data[start:start+response_bytes].copy(),
                self.data[start+response_bytes:end].copy(),
                dict(start=int(start), end=int(end), match=int(score),
                     response_previously_seen_bytes=response_prior,
                     holdout_previously_seen_bytes=holdout_prior,
                     holdout_position_fresh=(holdout_prior == 0),
                     source='matched' if score else 'continuation'))


class SequentialResponseWorld(BroadLanguageWorld):
    """Control: the body may speak and self-hear, but response follows cursor."""
    def take(self, utterance, response_bytes, holdout_bytes):
        return super().take(b'', response_bytes, holdout_bytes)


class StreamingLanguageWorld(BroadLanguageWorld):
    """Expose only consumed bytes so the body may act at any event boundary."""
    def act(self, utterance, required=1):
        if required <= 0:
            raise ValueError('required continuation must be positive')
        query = np.frombuffer(utterance, np.uint8)
        if not len(query):
            return dict(start=self.cursor, match=0, source='continuation')
        draw = int(self.rng.integers(0, 2**32, dtype=np.uint64))
        start, score = random_best(query, self.data, self.positions, required, self.cursor, draw)
        if start < 0:
            raise EOFError('world exhausted, no silent wrap')
        self.cursor = int(start)
        return dict(start=int(start), match=int(score), source='matched' if score else 'continuation')

    def observe(self, count=1):
        if count <= 0:
            raise ValueError('positive observation length required')
        start, end = self.cursor, self.cursor + count
        if end > len(self.data):
            raise EOFError('world exhausted, no silent wrap')
        previously_seen = int(self.seen[start:end].sum())
        result = self.data[start:end].copy()
        self.mark_seen(start, end)
        self.cursor = end
        return result, dict(start=start, end=end, previously_seen=previously_seen,
                            position_fresh=(previously_seen == 0))
