"""World-side exact suffix navigation, with deterministic local tie-breaking.

No model state, reward, exposure flag or unseen alternative is consulted.
The current world cursor resolves equally specific addresses. Common short
fragments therefore retain local continuity instead of drawing a distant topic.
"""
from __future__ import annotations

import numba


@numba.njit(cache=True)
def stable_suffix(query, data, required, cursor):
    """Longest actual suffix; nearest continuation wins ties, then lower offset.

    Scan the finite source without a second in-memory copy or a sampled index.
    Return (start, match_bytes, equally_best_occurrences, nearest_distance).
    A four-byte suffix is a lookup primitive, not a global random jump.
    """
    n = len(query)
    if n < 4:
        return -1, 0, 0, -1
    best, score_best, count, distance_best = -1, 0, 0, len(data) + 1
    for p in range(len(data) - required - 3):
        if (data[p] != query[n-4] or data[p+1] != query[n-3] or
                data[p+2] != query[n-2] or data[p+3] != query[n-1]):
            continue
        score = 4
        for k in range(1, min(n-4, p) + 1):
            if query[n-4-k] != data[p-k]:
                break
            score += 1
        start = p + 4
        distance = abs(start - cursor)
        if score > score_best:
            best, score_best, count, distance_best = start, score, 1, distance
        elif score == score_best:
            count += 1
            if distance < distance_best or (distance == distance_best and start < best):
                best, distance_best = start, distance
    if best < 0:
        return -1, 0, 0, -1
    return best, score_best, count, distance_best
