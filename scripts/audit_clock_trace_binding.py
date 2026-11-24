"""Bind the clock algebra to actual non-overlapping complete-body dispatches.

This is a realized trace clock binding; sampled events retain their phase and
within_external tags. Actual arrival-to-arrival intervals span learning,
settlement, quiet motion, action and world arrival in their original order.
"""
import argparse
import json
from pathlib import Path

import torch

from scripts.clock_credit_jet import compose, mark


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    raw = a.trace.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raw = raw[:raw.rfind(b'\n')+1]
    rows = [json.loads(line) for line in raw.decode('utf8').splitlines()]
    one = torch.ones(1,1,dtype=torch.double)
    total = mark(one,torch.zeros_like(one))
    intervals, parts, since, count = [], {}, 0., {}
    for row in rows:
        event = row['event']
        elapsed = float(row['dispatch_seconds'])
        assert elapsed >= 0
        total = compose(total,mark(one,one*elapsed))
        since += elapsed
        phase = event.get('phase',event['kind'])
        label = phase+(':within_learning' if event.get('within_external') else '')
        parts[label] = parts.get(label,0.)+elapsed
        count[label] = count.get(label,0)+1
        if event['kind'] == 'handoff' and phase == 'world_arrived':
            intervals.append(dict(kind='checkpoint_to_arrival' if not intervals else 'arrival_to_arrival',
                seconds=since, event=event['event']))
            since = 0.
    expected = sum(row['dispatch_seconds'] for row in rows)
    error = abs(float(total.clock)-expected)
    assert error < 1e-9
    result = dict(schema=1, source=str(a.trace), dispatches=len(rows),
        total_dispatch_seconds=expected, clock_jet_reconstruction_error=error,
        phase_seconds=parts, phase_dispatches=count, arrivals=intervals,
        pending_after_last_arrival_seconds=since,
        mass_convention='unit weight on this realized event path; clock binding only',
        credit_convention='credit operators are supplied directional derivatives; trace clock binding uses zero credit',
        world_calls_added=0, optimizer_updates_added=0)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('arrivals','mass_convention','credit_convention')},ensure_ascii=False))


if __name__ == '__main__':
    main()
