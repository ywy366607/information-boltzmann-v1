"""Read-only addressing audit using actual recorded utterances and real corpus.

This measures the world mapping, not a learned navigation capability. No Torch,
model checkpoint, API request, training, or live world state is loaded.
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path
import statistics
import time

import numba
import numpy as np

from scripts.ib_local.text_navigation import stable_suffix


def original_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queries', type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve the previous audit')
    root = Path(__file__).resolve().parents[2]
    # The saved initial config predates the source append. Use the current
    # atomic metadata, without loading the large Torch checkpoint.
    config = json.loads((args.run/'last.metadata.json').read_text(encoding='utf-8'))['config']
    tail = collections.deque(maxlen=128)
    with (args.run/'metrics.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            try:
                tail.append(json.loads(line))
            except ValueError:
                continue
    selected, seen = [], set()
    for row in reversed(tail):
        utterance = bytes.fromhex(row.get('utterance_hex', ''))
        if len(utterance) >= 4 and utterance not in seen:
            selected.append(row)
            seen.add(utterance)
        if len(selected) == args.queries:
            break
    selected.reverse()
    if not selected:
        raise ValueError('No actual utterances available')
    tokenizer = json.loads((root/'data/ib_owt_10000/tokenizer.json').read_text())
    tokens = np.load(root/'data/ib_owt_10000/train.npy', mmap_mode='r')
    lo, hi = config['source_raw_bounds']
    chunks = []
    for start in range(lo, hi, 1_000_000):
        chunk = tokens[start:min(start+1_000_000, hi)]
        chunks.append((chunk[chunk >= tokenizer['offset']]-tokenizer['offset']).astype(np.uint8))
    prefix = np.concatenate(chunks)
    del chunks, tokens
    expected = config['source']['world_source_sha256']
    if hashlib.sha256(prefix).hexdigest() != expected:
        raise ValueError('Original world source mismatch')
    manifest = config['natural_text_corpus']
    extra = np.fromfile(manifest['train_path'], dtype=np.uint8)
    if hashlib.sha256(extra).hexdigest() != manifest['train_sha256']:
        raise ValueError('Appended source mismatch')
    data = np.concatenate((prefix, extra))
    del prefix, extra
    namespace = dict(numba=numba, np=np)
    original_function(root/'scripts/ib_local/language_navigation_world.py', '_hash4', namespace)
    reservoir = original_function(root/'scripts/ib_local/language_navigation_world_v3.py', 'reservoir_index', namespace)
    old = original_function(root/'scripts/ib_local/language_navigation_kanli_v4h.py', 'suffix_best', namespace)
    positions = np.full((1 << 17, 8), -1, np.int32)
    reservoir(data, positions, config['seed'])
    records = []
    required = config.get('topic_window', 4096)
    stable_suffix(np.frombuffer(b'test', np.uint8), data[:8192], required, 0)
    for row in selected:
        query = np.frombuffer(bytes.fromhex(row['utterance_hex']), np.uint8)
        before = row.get('world_before', {})
        cursor = int(before.get('anchor', 0)) + int(before.get('offset', 0))
        repeated = [tuple(map(int, old(query, data, positions, required, seed))) for seed in range(16)]
        started = time.perf_counter()
        start, score, ties, distance = map(int, stable_suffix(query, data, required, cursor))
        seconds = time.perf_counter()-started
        last4 = bytes(query[-4:])
        records.append(dict(cycle=row['cycle'], utterance_hex=row['utterance_hex'],
            cursor=cursor, old_match=repeated[0][1],
            old_distinct_landings=len({s for s, _ in repeated if s >= 0}),
            old_mean_distance=statistics.mean(abs(s-cursor) for s, _ in repeated if s >= 0)
                if repeated[0][0] >= 0 else None,
            stable_start=start, stable_match=score, stable_best_occurrences=ties,
            stable_distance=distance, stable_lookup_seconds=seconds,
            last4_hex=last4.hex()))
    matched = [r for r in records if r['old_match']]
    result = dict(scope='Read-only world mapping audit; actual utterances, no model learning',
        source_bytes=len(data), cycles=[selected[0]['cycle'], selected[-1]['cycle']],
        queries=len(records), old_matched=len(matched),
        old_multivalued=sum(r['old_distinct_landings'] > 1 for r in matched),
        better_context_matches=sum(r['stable_match'] > r['old_match'] for r in records),
        old_four_byte_matches=sum(r['old_match'] == 4 for r in records),
        stable_four_byte_matches=sum(r['stable_match'] == 4 for r in records),
        stable_mean_seconds=statistics.mean(r['stable_lookup_seconds'] for r in records),
        stable_max_seconds=max(r['stable_lookup_seconds'] for r in records),
        records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
