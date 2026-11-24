"""Read the same complete body via gzip seeking and bounded disk staging."""
import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import torch

from scripts.audit_native_throughput import compare
from scripts.ib_local.compressed_checkpoint import checkpoint_stream


def sha(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8*1024**2):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--spool', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    before = sha(args.checkpoint)
    started = time.perf_counter()
    with gzip.open(args.checkpoint, 'rb') as stream:
        original = torch.load(stream, map_location='cpu', weights_only=False)
    original_seconds = time.perf_counter()-started
    # Force only the admission decision, so this measures the low-RAM path
    # even if sufficient RAM happens to be free during this standalone audit.
    from scripts.ib_local import compressed_checkpoint
    admission = compressed_checkpoint.available_memory
    compressed_checkpoint.available_memory = lambda: 0
    started = time.perf_counter()
    try:
        with checkpoint_stream(args.checkpoint, spool_directory=args.spool) as stream:
            staged = torch.load(stream, map_location='cpu', weights_only=False)
    finally:
        compressed_checkpoint.available_memory = admission
    staged_seconds = time.perf_counter()-started
    result = dict(source=str(args.checkpoint), sha256=before, original_seconds=original_seconds,
        staged_seconds=staged_seconds, speedup=original_seconds/staged_seconds,
        complete_state=compare(original, staged), source_unchanged=before == sha(args.checkpoint),
        staging_empty=not any(args.spool.iterdir()), optimizer_updates=0, world_calls=0)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result))
    assert result['complete_state']['different'] == 0 and not result['complete_state']['shape_differences']
    assert result['source_unchanged'] and result['staging_empty']


if __name__ == '__main__':
    main()
