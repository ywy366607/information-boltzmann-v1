"""Download one public object with parallel HTTP range requests, resumably (fixed-size parts)."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading
import time

import requests

_local = threading.local()


def part(url, path, index, start, end):
    done = path.with_name(path.name + f'.part{index}.ok')
    if done.exists():
        return index
    session = getattr(_local, 's', None) or requests.Session()
    _local.s = session
    for attempt in range(8):
        try:
            r = session.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=(15, 300))
            if r.status_code != 206 or len(r.content) != end - start + 1:
                raise IOError('bad range')
            with open(path, 'r+b') as f:
                f.seek(start)
                f.write(r.content)
            done.touch()
            return index
        except (requests.RequestException, IOError):
            time.sleep(min(60, 2 ** attempt))
    raise IOError(f'part {index} failed')


def main(a):
    size = int(requests.head(a.url, timeout=30).headers['Content-Length'])
    if not a.output.exists() or a.output.stat().st_size != size:
        with open(a.output, 'wb') as f:
            f.truncate(size)
    parts = [(i, s, min(size, s + a.part_mb * 2 ** 20) - 1) for i, s in enumerate(range(0, size, a.part_mb * 2 ** 20))]
    started, finished = time.time(), 0
    with ThreadPoolExecutor(a.workers) as pool:
        futures = [pool.submit(part, a.url, a.output, *p) for p in parts]
        for f in as_completed(futures):
            f.result()
            finished += 1
            if finished % 10 == 0 or finished == len(parts):
                print(json.dumps({'parts': finished, 'of': len(parts),
                                  'MiB_per_s': round(finished * a.part_mb / (time.time() - started), 2)}), flush=True)
    for i, _, _ in parts:
        a.output.with_name(a.output.name + f'.part{i}.ok').unlink(missing_ok=True)
    print(json.dumps({'status': 'complete', 'bytes': size}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--url', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--part-mb', type=int, default=16)
    main(p.parse_args())
