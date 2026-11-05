"""Summarize available registered navigation arms without inventing missing results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = ('fresh_active_merge', 'fresh_random_merge', 'fresh_silence_merge',
        'fresh_active_fifo', 'fresh_active_no_latent')


def inspect(root: Path, name: str) -> dict:
    arm = root / name
    if not (arm / 'metrics.jsonl').exists():
        return {'status': 'not_started'}
    raw = (arm / 'metrics.jsonl').read_bytes()
    if not raw.endswith(b'\n'):
        raw = raw.rsplit(b'\n', 1)[0]
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows:
        return {'status': 'starting'}
    initial = json.loads((arm / 'initial.json').read_text())
    complete = (arm / 'summary.json').exists()
    validations = [{'cycle': row['cycle'], 'nll': row['validation_nll']}
                   for row in rows if 'validation_nll' in row]
    fresh = [row for row in rows if row['world'].get('holdout_position_fresh', False)]
    result = {'status': 'complete' if complete else 'running_or_stopped',
              'cycles': len(rows), 'updates': rows[-1]['optimizer_updates'],
              'initial_validation_nll': initial['validation_nll'], 'validation': validations,
              'external_bytes': sum(row['external_bytes'] for row in rows),
              'self_bytes': sum(row['self_bytes'] for row in rows),
              'fresh_position_fraction': len(fresh) / len(rows),
              'mean_fresh_position_progress': sum(row['progress'] for row in fresh) / len(fresh) if fresh else None,
              'merge_count': rows[-1]['merge_count'], 'evict_count': rows[-1]['evict_count'],
              'last_read_gain': rows[-1]['read_gain']}
    if complete:
        result['run_summary'] = json.loads((arm / 'summary.json').read_text())
    revisit = root / (name + '_revisit') / 'summary.json'
    if revisit.exists():
        result['revisit'] = json.loads(revisit.read_text())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'scope': 'one seed, fixed budget, finite indexed OWT world; not semantic novelty',
              'arms': {name: inspect(args.root, name) for name in ARMS}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
