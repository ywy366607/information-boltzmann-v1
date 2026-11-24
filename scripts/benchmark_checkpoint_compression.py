"""Recompress identical full-checkpoint bytes and verify standard-gzip restore."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

from isal import igzip_threaded


def digest(stream):
    result = hashlib.sha256()
    while block := stream.read(8 * 1024**2):
        result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    with args.checkpoint.open('rb') as stream:
        source_hash = digest(stream)
    rows = []
    # Only this invocation's new temporary directory is released on exit.
    with tempfile.TemporaryDirectory(prefix='compression-', dir=args.work) as temporary:
        root = Path(temporary)
        original = root / 'original.pt'
        with gzip.open(args.checkpoint, 'rb') as src, original.open('wb') as dst:
            shutil.copyfileobj(src, dst, 1024**2)
        with original.open('rb') as stream:
            expected = digest(stream)
        uncompressed_bytes = original.stat().st_size
        for backend, workers in [('gzip', 1), ('isal', 1), ('isal', 4)]:
            output = root / f'{backend}{workers}.gz'
            start = time.perf_counter()
            if backend == 'gzip':
                writer = gzip.open(output, 'wb', compresslevel=1)
            else:
                writer = igzip_threaded.open(output, 'wb', compresslevel=1,
                                             threads=workers, block_size=1024**2)
            with original.open('rb') as src, writer as dst:
                shutil.copyfileobj(src, dst, 1024**2)
            seconds = time.perf_counter() - start
            # Actual historical reader, including its CRC/trailer verification.
            with gzip.open(output, 'rb') as stream:
                actual = digest(stream)
            assert actual == expected
            rows.append(dict(backend=backend, workers=workers, seconds=seconds,
                             compressed_bytes=output.stat().st_size,
                             restored_bytes_sha256=actual))
    with args.checkpoint.open('rb') as stream:
        assert digest(stream) == source_hash
    result = dict(source=str(args.checkpoint), source_sha256=source_hash,
                  uncompressed_bytes=uncompressed_bytes,
                  rows=rows, byte_exact_restore=True, source_unchanged=True,
                  optimizer_updates=0, world_calls=0)
    args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
