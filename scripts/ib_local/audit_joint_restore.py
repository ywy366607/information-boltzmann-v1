"""Reload a committed mother-world body and compare all declared carried state.

Uses the actual continuation constructor and disk serializer. It never takes a
world step, calls the mother, or updates parameters. Configuration provenance
may change on restore; physical/learning/world/RNG state must not.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local.checkpoint_digest import state_digest
from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation


def canonical(value):
    # Device restoration can compact a read-only receipt slice. Its storage
    # stride is not a different historical value. Keep dtype, shape and every
    # element, but compare both checkpoint layouts in contiguous order.
    if torch.is_tensor(value):
        return value.contiguous()
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if isinstance(value, np.random.Generator):
        return dict(generator=type(value.bit_generator).__name__, state=value.bit_generator.state)
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(canonical(v) for v in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(value, **{f.name: canonical(getattr(value, f.name))
                                            for f in dataclasses.fields(value) if f.init})
    return value


def fingerprints(path):
    with gzip.open(path, 'rb') as stream:
        state = torch.load(stream, map_location='cpu', weights_only=False)
    result = {key: state_digest(canonical(value))['sha256']
              for key, value in state.items() if key != 'config'}
    result['engine_fields'] = {key: state_digest(canonical(value))['sha256']
                              for key, value in state['engine'].items()}
    del state
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--table-storage', type=Path, required=True)
    parser.add_argument('--compare-with', type=Path,
                        help='compare an already reconstructed checkpoint without loading a body again')
    args = parser.parse_args()
    torch.set_num_threads(1)
    before = fingerprints(args.checkpoint)
    if args.compare_with is None:
        options = continuation_parser().parse_args([
            '--checkpoint', str(args.checkpoint), '--output', str(args.output),
            '--cycles', '1', '--table-storage', str(args.table_storage)])
        loaded = prepare_continuation(options)
        cycle = loaded.cycle
        del loaded
        gc.collect()
        torch.cuda.empty_cache()
        comparison = args.output/'last.pt.gz'
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        cycle = None
        comparison = args.compare_with
    after = fingerprints(comparison)
    same = {key: before[key] == after.get(key) for key in before}
    report = dict(cycle=cycle, same=same, passed=all(same.values()) and before.keys() == after.keys(),
                  comparison='all declared values, shapes, dtypes; storage strides canonicalized; config provenance excluded',
                  changed_engine_fields=[key for key in before['engine_fields']
                      if before['engine_fields'][key] != after['engine_fields'].get(key)],
                  before=before, after=after, mother_calls=0, optimizer_steps=0,
                  scope='full CUDA constructor and mother-world restore; no new event')
    (args.output/'restore_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(cycle=cycle, passed=report['passed'], same=same)), flush=True)
    if not report['passed']:
        raise RuntimeError('restored carried state differs')


if __name__ == '__main__':
    main()
