"""Recompress identical serialized bytes, preserving the complete individual."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from scripts.ib_local.checkpoint_commit import publish_checkpoint


def digest(path, decompress=False):
    h=hashlib.sha256()
    opener=gzip.open if decompress else open
    with opener(path,'rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body',type=Path,required=True)
    parser.add_argument('--level',type=int,choices=range(1,10),default=6)
    args=parser.parse_args()
    target=args.body/'last.pt.gz'
    metadata_path=args.body/'last.metadata.json'
    metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
    before=digest(target)
    if before!=metadata['checkpoint_sha256']:raise ValueError('checkpoint metadata mismatch')
    temporary=args.body/'last.recompress.tmp.gz'
    if temporary.exists():raise FileExistsError('preserve unfinished transcode')
    raw_hash=hashlib.sha256()
    with gzip.open(target,'rb') as source,gzip.open(temporary,'wb',compresslevel=args.level) as output:
        for block in iter(lambda:source.read(1048576),b''):
            raw_hash.update(block);output.write(block)
    if digest(temporary,decompress=True)!=raw_hash.hexdigest():raise ValueError('serialized bytes changed')
    record=dict(cycle=metadata['cycle'],before_sha256=before,after_sha256=digest(temporary),
        uncompressed_sha256=raw_hash.hexdigest(),before_bytes=target.stat().st_size,
        after_bytes=temporary.stat().st_size,compression_level=args.level,
        meaning='identical uncompressed Torch archive; no state, weight, optimizer or RNG changes')
    if digest(target)!=before:raise RuntimeError('checkpoint owner changed during transcode')
    metadata['checkpoint_sha256']=record['after_sha256']
    pending=args.body/'last.metadata.tmp.json'
    pending.write_text(json.dumps(metadata,ensure_ascii=False)+'\n',encoding='utf-8')
    report=args.body/f'recompression_{metadata["cycle"]}.json'
    if report.exists():raise FileExistsError('preserve prior transcode report')
    report.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    publish_checkpoint(temporary,target)
    publish_checkpoint(pending,metadata_path)
    print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':main()
