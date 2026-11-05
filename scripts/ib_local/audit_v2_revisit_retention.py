"""Compare actual latent slots before and after one real-text continuation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from scripts.ib_local.language_action import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('before', type=Path)
    p.add_argument('after', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    first = torch.load(args.before, map_location='cpu', weights_only=False)['model']
    last = torch.load(args.after, map_location='cpu', weights_only=False)['model']
    def get(model, name):
        return model['memory.' + name]
    occupied = get(first, 'occupied')
    old_birth = get(first, 'birth')
    same = occupied & get(last, 'occupied') & (old_birth == get(last, 'birth'))
    result = dict(before=str(args.before), after=str(args.after),
                  before_sha256=digest(args.before), after_sha256=digest(args.after),
                  occupied_before=int(occupied.sum()), same_birth_slots_after=int(same.sum()),
                  event_delta=int(get(last, 'clock')-get(first, 'clock')),
                  merged_delta=int(get(last, 'merged_total')-get(first, 'merged_total')),
                  admitted_delta=int(get(last, 'admitted_total')-get(first, 'admitted_total')),
                  evicted_delta=int(get(last, 'evicted_total')-get(first, 'evicted_total')),
                  scope='Slot birth survival only; not semantic forgetting or proof of useful retrieval.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
