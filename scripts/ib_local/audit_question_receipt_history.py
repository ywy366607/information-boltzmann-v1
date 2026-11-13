"""Inspect versioned pre-draft relation receipts in a committed event log.

Receipt overlap is evidence of reusing an episode, not question identity.
The diagnostic does not load or change the individual or its world.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def summarize(rows):
    receipts = []
    for row in rows:
        q = row['question_context']
        if len(q['slots']) != len(q['stamps']):
            raise ValueError('receipt slots and stamps have different lengths')
        # The current log contains only the support count, not a valid mask.
        # Do not guess identities for partially valid receipts.
        if q['support'] != len(q['slots']):
            raise ValueError('partial receipt lacks a logged validity mask')
        receipts.append(set(zip(q['slots'], q['stamps'])))
    counts = collections.Counter(item for receipt in receipts for item in receipt)
    overlap = [len(a & b) for a, b in zip(receipts, receipts[1:])]
    return {
        'cycles': [row['cycle'] for row in rows],
        'support': [len(receipt) for receipt in receipts],
        'adjacent_overlap': overlap,
        'adjacent_with_overlap': sum(n > 0 for n in overlap),
        'unique_versioned_episodes': len(counts),
        'episodes_recalled_multiple_times': sum(n > 1 for n in counts.values()),
        'max_recall_count': max(counts.values(), default=0),
        'question_evidence_samples': dict(collections.Counter(
            str(row.get('question_evidence_samples')) for row in rows)),
        'meaning': 'Shared episode receipt is not proof of a shared open question.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite an existing audit')
    rows = [json.loads(line) for line in args.log.read_text(encoding='utf-8').splitlines() if line.strip()]
    result = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
