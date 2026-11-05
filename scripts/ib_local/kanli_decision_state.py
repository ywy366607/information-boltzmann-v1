"""Decision-time topic state for TopicWorld records (Claude, 2026-11-04).

Each record's `fresh` flag describes the turn AFTER the action. This module
instead replays the world's seen set from the records and asks, before each
choice, whether a silent turn would read fresh holdout positions.

Returns one state per record: None (no previous turn), True (new) or
False (exhausted), together with the number of replay mismatches against the
recorded `holdout_previously_seen_bytes`.
"""
import numpy as np


def decision_states(records, window=4096, response=480, holdout=480):
    seen, anchor, offset, states, mismatches = set(), None, None, [], 0
    span = np.arange(response+holdout)
    for r in records:
        w = r['world']
        if anchor is None:
            states.append(None)
        else:
            pos = anchor+(offset+span) % window
            states.append(not any(int(p) in seen for p in pos[response:]))
        anchor, offset = w['anchor'], w['start']-w['anchor']
        pos = anchor+(offset+span) % window
        mismatches += sum(int(p) in seen for p in pos[response:]) != w['holdout_previously_seen_bytes']
        seen.update(int(p) for p in pos)
        offset = (offset+response+holdout) % window
    return states, mismatches
