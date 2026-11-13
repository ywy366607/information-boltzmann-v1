"""Bind a verified SSD archive to an unchanged complete body checkpoint."""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import shutil
from pathlib import Path

import torch

from scripts.ib_local.audit_joint_restore import fingerprints
from scripts.ib_local.causal_archive import sha256
from scripts.ib_local.mother_checkpoint import ChunkedWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--audit-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    audit = json.loads(args.audit_report.read_text(encoding='utf-8'))
    origin = sha256(args.checkpoint)
    if not audit['passed'] or audit['source_sha256'] != origin:
        raise ValueError('archive audit does not belong to this source checkpoint')
    manifest = dict(audit['manifest'])
    if sha256(manifest['snapshot']) != manifest['sha256']:
        raise ValueError('verified archive snapshot changed')
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output/('causal_archive_'+manifest['sha256']+'.sqlite')
    shutil.copyfile(manifest['snapshot'], destination)
    manifest['snapshot'] = str(destination.resolve())
    before = fingerprints(args.checkpoint)
    with gzip.open(args.checkpoint, 'rb') as stream:
        state = torch.load(stream, map_location='cpu', weights_only=False)
    state['config'] = dict(state['config'], causal_archive=manifest,
        causal_archive_migration=dict(source_sha256=origin, cycle=state['cycle'],
            kind='storage-only; all declared body/world/optimizer/RNG values retained',
            archive_audit=str(args.audit_report.resolve())))
    with gzip.open(args.output/'last.tmp.gz', 'wb', compresslevel=1) as stream:
        torch.save(state, ChunkedWriter(stream))
    (args.output/'last.tmp.gz').replace(args.output/'last.pt.gz')
    cycle = state['cycle']
    del state
    gc.collect()
    after = fingerprints(args.output/'last.pt.gz')
    same = {key: before[key] == after.get(key) for key in before}
    report = dict(source=str(args.checkpoint.resolve()), source_sha256=origin,
        checkpoint=str((args.output/'last.pt.gz').resolve()),
        checkpoint_sha256=sha256(args.output/'last.pt.gz'), archive=manifest,
        cycle=cycle, same=same, passed=all(same.values()) and before.keys()==after.keys(),
        source_unchanged=origin == sha256(args.checkpoint), mother_calls=0, optimizer_updates=0)
    (args.output/'migration.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['passed'] or not report['source_unchanged']:
        raise RuntimeError('archive/body migration changed physical or learning history')


if __name__ == '__main__':
    main()
