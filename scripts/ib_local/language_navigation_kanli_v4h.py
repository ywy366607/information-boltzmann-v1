"""V4 world change: the world takes up the body's words (接话) (Claude, 2026-11-04).

The action-effect probe (results/.../analysis/action_effect_probe.py) found
that neither the body nor wood had learned what its own words do to its
input.
* After its own words, the body predicts the first 32 reply bytes only
  0.004 nats/byte better than after another utterance.
* F's own action beats another action in only 53% of cycles.
The cause is the world. TopicWorld hangs the reply on the longest 4+ byte
fragment found anywhere in the utterance (random among ties) and reads on
from that fragment's end, so the reply's content barely depends on what was
said. What speech changed was mostly novelty, and metal learned that.

User, 11-04: "模型应该学到自己的动作会产生什么影响，对世界的影响也会反过来改变
自己的输入". Approved ("做吧"): the world takes up the END of the utterance.
Among the indexed occurrences of the utterance's last 4 bytes (the same
uniform reservoir of 8 per hash bucket), the world picks the one whose
preceding text matches the utterance furthest back (random among ties), and
continues from there. So the reply continues the body's own words. Speech
whose ending occurs nowhere in the index is not taken up, and the topic
stays. Nothing else in the world changes: bounded topic window, cyclic
reading, same response and holdout sizes.
"""
from __future__ import annotations

import numba
import numpy as np

from scripts.ib_local.language_navigation_kanli_v4 import TopicWorld
from scripts.ib_local.language_navigation_world import _hash4


@numba.njit(cache=True)
def suffix_best(query, data, positions, required, seed):
    """Occurrence of the query's last 4 bytes with the longest backward agreement."""
    np.random.seed(seed)
    n = len(query)
    if n < 4:
        return -1, 0
    mask = len(positions) - 1
    h = int(_hash4(query[n-4], query[n-3], query[n-2], query[n-1]) & np.uint64(mask))
    best, score_best, ties = -1, 0, 0
    for j in range(positions.shape[1]):
        p = int(positions[h, j])
        if p < 0 or p + 4 + required > len(data):
            continue
        if data[p] != query[n-4] or data[p+1] != query[n-3] or data[p+2] != query[n-2] or data[p+3] != query[n-1]:
            continue
        score = 4
        for k in range(1, min(n-4, p)+1):
            if query[n-4-k] != data[p-k]:
                break
            score += 1
        if score > score_best:
            best, score_best, ties = p+4, score, 1
        elif score == score_best:
            ties += 1
            if np.random.randint(0, ties) == 0:
                best = p+4
    return best, score_best


class ReplyTopicWorld(TopicWorld):
    def take(self, utterance, response_bytes, holdout_bytes):
        required = response_bytes+holdout_bytes
        if required > self.window:
            raise ValueError('one turn must fit the topic window')
        score = 0
        if utterance:
            query = np.frombuffer(utterance, np.uint8)
            draw = int(self.rng.integers(0, 2**32, dtype=np.uint64))
            start, score = suffix_best(query, self.data, self.positions, self.window, draw)
            if score > 0:
                self.anchor, self.offset = int(start), 0
                self.moves += 1
        positions = self.anchor + (self.offset + np.arange(required)) % self.window
        self.offset = (self.offset+required) % self.window
        prior_response = int(self.seen[positions[:response_bytes]].sum())
        prior_holdout = int(self.seen[positions[response_bytes:]].sum())
        self.seen[positions] = True
        self.cursor = int(positions[-1])+1
        return (self.data[positions[:response_bytes]].copy(), self.data[positions[response_bytes:]].copy(),
                dict(start=int(positions[0]), anchor=self.anchor, match=int(score),
                     response_previously_seen_bytes=prior_response,
                     holdout_previously_seen_bytes=prior_holdout,
                     holdout_position_fresh=(prior_holdout == 0),
                     source='matched' if score else 'topic'))
