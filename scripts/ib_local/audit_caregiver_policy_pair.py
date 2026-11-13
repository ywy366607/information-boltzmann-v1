"""Compare caregiver policies on identical saved history and real child turns.

This is a read-only fork of the world: no body, optimizer, world or actual
transcript is advanced. Inputs are prior actual utterances, not model answers.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import torch

from scripts.ib_local.mother_reply_world import StrictToddlerMother


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--events', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite a prior result')
    with gzip.open(args.checkpoint, 'rb') as stream:
        snapshot = torch.load(stream, map_location='cpu', weights_only=False)
    history = snapshot['world']['mother_transcript']
    horizon = snapshot['world']['mother_history']
    rows = [json.loads(line) for line in args.events.read_text(encoding='utf8').splitlines() if line]
    selected = [next(row for row in rows if row['cycle'] == cycle) for cycle in (2961,2963,2964,2967)]
    providers = {policy: StrictToddlerMother(policy=policy, shared_cache=args.cache)
                 for policy in ('responsive', 'responsive_v2')}
    results = []
    for row in selected:
        child = row['world']['child_text'] or '<quiet_turn/>'
        turns = [tuple(item) for item in history[-2*horizon:]] + [('child', child)]
        answers = {name: mother.reply(turns) for name, mother in providers.items()}
        results.append(dict(source_cycle=row['cycle'], child=child, answers=answers))
    out = dict(checkpoint=str(args.checkpoint), cycle=int(snapshot['cycle']),
               method='same saved transcript and actual historical utterances; no body/world update',
               results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(out,ensure_ascii=True))


if __name__ == '__main__':
    main()
