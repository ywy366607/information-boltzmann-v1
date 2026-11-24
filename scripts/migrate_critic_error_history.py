"""Re-score the inherited critic's error memory using the recorded TD history.

This is a checkpoint-boundary migration: only engine.critic_error changes.
The body, world, learned parameters, optimizer, traces and RNG are preserved.
"""
import argparse
import gc
import gzip
import json
import math
from pathlib import Path

import torch

from scripts.audit_critic_repair_history import audit
from scripts.audit_repaired_mother_loop import SEGMENTS
from scripts.ib_local.audit_joint_restore import fingerprints
from scripts.ib_local.causal_archive import sha256
from scripts.ib_local.mother_checkpoint import ChunkedWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--additional-segment', action='append', type=Path, default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    base = SEGMENTS[0].parent
    history = audit((*SEGMENTS, *args.additional_segment),
                    previous_segments=(base/'admission', base/'learning'))
    if (history['replay_max_abs_squared_error'] > 1e-6 or
            history['recomputed_history']['target_replay_max_abs_error'] > 1e-7):
        raise ValueError('recorded TD or saved error statistic does not reconstruct')
    source_hash = sha256(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    before = fingerprints(args.checkpoint)
    with gzip.open(args.checkpoint, 'rb') as stream:
        state = torch.load(stream, map_location='cpu', weights_only=False)
    if state['cycle'] != history['last']:
        raise ValueError('history must end at the source checkpoint event boundary')
    engine = state['engine']
    old = float(engine['critic_error'])
    if not math.isclose(math.sqrt(old), history['saved_rms'], rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError('saved critic error disagrees with the last committed event')
    revised = history['recomputed_history']['bias_corrected_rms']**2
    if not math.isfinite(revised) or revised <= 0:
        raise ValueError('history yielded invalid critic error')
    engine['critic_error'] = revised
    state['config'] = dict(state['config'], critic_error_history_migration={
        'source_sha256': source_hash, 'cycle': state['cycle'],
        'history_first': history['recomputed_history']['first'],
        'history_last': history['last'], 'old': old, 'new': revised,
        'method': 'bias-corrected .98 EMA of recorded errors with corrected TD units'})
    tmp = args.output/'last.tmp.gz'
    with gzip.open(tmp, 'wb', compresslevel=1) as stream:
        torch.save(state, ChunkedWriter(stream))
    checkpoint = args.output/'last.pt.gz'
    tmp.replace(checkpoint)
    del state
    gc.collect()
    after = fingerprints(checkpoint)
    changed = [key for key in before['engine_fields']
               if before['engine_fields'][key] != after['engine_fields'].get(key)]
    preserved = {k: before[k] == after.get(k)
                 for k in before if k not in ('engine', 'engine_fields')}
    passed = (changed == ['critic_error'] and all(preserved.values()) and
              before['engine_fields'].keys() == after['engine_fields'].keys() and
              source_hash == sha256(args.checkpoint))
    report = {'source': str(args.checkpoint.resolve()), 'source_sha256': source_hash,
        'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': sha256(checkpoint),
        'cycle': history['last'], 'old_critic_error': old, 'new_critic_error': revised,
        'history': history, 'changed_engine_fields': changed, 'preserved': preserved,
        'mother_calls': 0, 'optimizer_updates': 0, 'passed': passed}
    (args.output/'migration.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'passed': passed, 'cycle': history['last'], 'changed': changed,
                      'old_rms': math.sqrt(old), 'new_rms': math.sqrt(revised)}))
    if not passed:
        raise RuntimeError('migration changed state outside the declared statistic')


if __name__ == '__main__':
    main()
