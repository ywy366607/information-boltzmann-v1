"""Replay recorded real utterances against old and broad world indexes offline.

The individual does not adapt. This measures a changed retrieval kernel, not
its eventual effect on trained language ability or navigation.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np

from scripts.ib_local.language_navigation_world import LanguageNavigationWorld
from scripts.ib_local.language_navigation_world_v3 import BroadLanguageWorld


def replay(world, utterances, response, heldout):
    intervals, fresh, matches, starts = [], 0, 0, []
    for utterance in utterances:
        try:
            _, _, info = world.take(utterance, response, heldout)
        except EOFError:
            break
        intervals.append((info['start'], info['end']))
        starts.append(info['start'])
        fresh += info['holdout_position_fresh']
        matches += info['source'] == 'matched'
    total = len(intervals) * (response + heldout)
    end, unique = -1, 0
    for lo, hi in sorted(intervals):
        if hi > end:
            unique += hi - max(lo, end)
            end = hi
    counts = collections.Counter(starts)
    return dict(events=len(intervals), source_bytes=len(world.data),
                unique_positions=unique, repeated_position_fraction=1-unique/total if total else None,
                fresh_holdout_fraction=fresh/len(intervals) if intervals else None,
                matched_fraction=matches/len(intervals) if intervals else None,
                distinct_starts=len(counts), max_start_repeats=max(counts.values()) if counts else None,
                start_range=[min(starts), max(starts)] if starts else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prior audit')
    cfg = json.loads((args.run/'config.json').read_text())
    raw_log = (args.run/'metrics.jsonl').read_bytes()
    if not raw_log.endswith(b'\n'):
        raise ValueError('use a completed run')
    rows = [json.loads(line) for line in raw_log.splitlines() if line.strip()]
    utterances = [bytes.fromhex(row['utterance_hex']) for row in rows]
    root = Path('data/ib_owt_10000')
    offset = json.loads((root/'tokenizer.json').read_text())['offset']
    train = np.load(root/'train.npy', mmap_mode='r')
    lo, hi = cfg['source_raw_bounds']
    def source(stop):
        raw = np.asarray(train[lo:stop])
        return (raw[raw >= offset] - offset).astype(np.uint8)
    old = source(hi)
    if hashlib.sha256(old.tobytes()).hexdigest() != cfg['source_byte_sha256']:
        raise ValueError('original source mismatch')
    full = source(len(train))
    reports = {}
    for name, world in (
        ('old_same_source', LanguageNavigationWorld(old)),
        ('broad_same_source', BroadLanguageWorld(old, seed=11)),
        ('broad_larger_source', BroadLanguageWorld(full, seed=11)),
    ):
        reports[name] = replay(world, utterances, cfg['response_bytes'], cfg['holdout_bytes'])
    result = dict(scope='fixed logged utterances; no model adaptation, not a trained comparison',
                  log_sha256=hashlib.sha256(raw_log).hexdigest(),
                  original_source_sha256=hashlib.sha256(old.tobytes()).hexdigest(),
                  larger_source_sha256=hashlib.sha256(full.tobytes()).hexdigest(),
                  larger_raw_bounds=[lo, len(train)], seed=11, reports=reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
