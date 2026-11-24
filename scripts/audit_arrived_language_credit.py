"""H=2 endpoint credit from an independent ordinary-train probe, without learning."""
import argparse
import hashlib
import json
from pathlib import Path
from secrets import randbelow

import numpy as np
import torch

from scripts.ib_local.delayed_action_contrast import arrived_contrast
from scripts.ib_local.wood_policy_measurement import score


def train_probe(world, size, draw=randbelow):
    manifest = world.source_manifest
    # extended_source verified this immutable train append on restore. Do not
    # query cursor/seen or select from bytes obtained by the chosen action.
    train = world.data[manifest['prefix_bytes']:]
    if len(train) != manifest['train_bytes'] or not 0 < size < len(train):
        raise ValueError('invalid train probe window')
    start = draw(len(train)-size)
    return train[start:start+size+1].copy(), start


def payment(row, endpoint_score):
    habit = torch.tensor(row['habit_logits'], dtype=torch.float64)
    offset = torch.tensor(row['imagination_offset'], dtype=torch.float64)
    legal = habit > -1e3
    p = torch.tensor(row['choice_probability'], dtype=torch.float64)
    torch.testing.assert_close((habit+offset).masked_fill(~legal, -torch.inf).softmax(0), p,
                               atol=1e-7, rtol=1e-6)
    options = torch.nonzero(legal).flatten().tolist()
    chosen = options.index(('silence','generate','reuse').index(row['choice']))
    p, blind = p[legal].tolist(), habit[legal].softmax(0).tolist()
    return dict(legal=options, chosen=options[chosen], production=p, without_offset=blind,
                contrast=arrived_contrast(p, blind, chosen, endpoint_score))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--natural-run', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
    from scripts.ib_local.natural_text_world import NaturalTextWorld
    torch.set_num_threads(1)
    body = args.natural_run/'body'
    result = json.loads((args.natural_run/'result.json').read_text(encoding='utf8'))
    with (args.natural_run/'events.jsonl').open(encoding='utf8') as stream:
        rows = [r for line in stream if (r:=json.loads(line))['autonomous_event']['kind']=='external']
    first, last = rows[-2:]
    assert result['status']=='completed' and first['cycle']+1==last['cycle']==result['cycle']
    opts = continuation_parser().parse_args(['--checkpoint',str(body/'last.pt.gz'),'--output',str(body),
        '--resume-output','--world-source','text','--autonomous-events','--optimizer-policy','budgeted',
        '--cycles','1','--table-storage',str(args.work/'table'),'--causal-storage',str(args.work/'causal')])
    loaded = prepare_continuation(opts, resume_output=True)
    e = loaded.engine
    try:
        assert loaded.cycle==last['cycle'] and loaded.source_sha==result['checkpoint_sha256']
        assert isinstance(e.world, NaturalTextWorld)
        raw, start = train_probe(e.world, 2048)
        y = -score(e, [raw])['mean_nll']
        report = dict(action_cycle=first['cycle'], endpoint_cycle=loaded.cycle, horizon=2,
            checkpoint_sha256=loaded.source_sha, train_sha256=e.world.source_manifest['train_sha256'],
            start_in_train=start, probe_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
            target_bytes=len(raw)-1, endpoint_log_score=y, units='nats/byte',
            **payment(first,y), forecast='zero', reward_writes=0, world_calls=0, optimizer_updates=0)
        with args.output.open('x', encoding='utf8') as out:
            json.dump(report,out,indent=2,allow_nan=False); out.write('\n')
        print(json.dumps(report))
    finally:
        e.model._causal_archive.close()


if __name__=='__main__':
    main()
