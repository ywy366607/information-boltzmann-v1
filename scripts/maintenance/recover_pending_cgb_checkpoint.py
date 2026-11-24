"""Publish one fully serialized event after a failed atomic rename.

Requires a successful frozen audit of the exact temporary checkpoint. Does not
replay an event, train, edit weights, or rewrite historical observations.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch

from scripts.ib_local.checkpoint_commit import publish_checkpoint


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    args = parser.parse_args()
    temporary, target = args.body/'last.tmp.gz', args.body/'last.pt.gz'
    metadata_path = args.body/'last.metadata.json'
    prior = json.loads(metadata_path.read_text(encoding='utf-8'))
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    new_hash = digest(temporary)
    if new_hash != audit['checkpoint_sha256'] or digest(target) != prior['checkpoint_sha256']:
        raise ValueError('recovery source changed')
    with gzip.open(temporary,'rb') as stream:
        saved = torch.load(stream,map_location='cpu',weights_only=False)
    if saved['cycle'] != prior['cycle']+1 or saved['cycle'] != audit['cycle']:
        raise ValueError('only the next complete audited event may be published')
    for field in ('model','optimizer','engine','table','world','torch_rng','cuda_rng','numpy_rng','python_rng'):
        if field not in saved:
            raise ValueError(f'missing complete state: {field}')
    archive = saved['config'].get('causal_archive')
    if archive and (not Path(archive['snapshot']).is_file() or digest(Path(archive['snapshot'])) != archive['sha256']):
        raise ValueError('bound causal archive is missing or changed')
    metadata = dict(schema=1,checkpoint_sha256=new_hash,cycle=saved['cycle'],config=saved['config'],
        mother_world={k:v for k,v in saved['world'].items() if k.startswith('mother_')})
    record = dict(previous_cycle=prior['cycle'],recovered_cycle=saved['cycle'],
        previous_sha256=prior['checkpoint_sha256'],recovered_sha256=new_hash,
        meaning='complete event serialized before rename failure; no replay or reset',
        telemetry='event did not reach metrics publication; preserved as a separate recovery record')
    # All validation precedes the authoritative atomic publication.
    pending_metadata = args.body/'last.metadata.tmp.json'
    pending_metadata.write_text(json.dumps(metadata,ensure_ascii=False)+'\n',encoding='utf-8')
    recovery = args.body/f'recovered_commit_{saved["cycle"]}.json'
    if recovery.exists():
        raise FileExistsError('preserve prior recovery record')
    recovery.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    publish_checkpoint(temporary,target)
    publish_checkpoint(pending_metadata,metadata_path)
    print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':
    main()
