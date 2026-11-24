"""Fetch 16 MiB warm Chinese storybook corpus (RobinChen2001/TinyStories-Zh-2M) via direct parquet shard.

Uses 1 single HTTP download (no Datasets Server 429 limits), extracts clean everyday
Chinese stories & mother-child dialogues with exact SHA-256 deduplication and a
deterministic `mod 32 == 0` held-out validation split, then removes the temporary parquet.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import certifi
import pyarrow.parquet as pq
import requests


OUT_DIR = Path('D:/information-boltzmann-v1/data/cgb_chinese_stories_20261124').resolve()
PARQUET_URL = 'https://huggingface.co/datasets/RobinChen2001/TinyStories-Zh-2M/resolve/main/data/train-00000-of-00003.parquet'
TARGET_TRAIN_BYTES = 16 * 1024 * 1024  # 16 MiB warm Chinese stories (~22,000 stories!)
TARGET_VAL_BYTES = 512 * 1024          # 512 KiB held-out story validation


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_parquet = Path('D:/tinystories_zh_shard0.tmp.parquet')
    t0 = time.perf_counter()

    print('[STORIES] Downloading TinyStories-Zh shard 0 (direct HTTPS)...', flush=True)
    session = requests.Session()
    session.verify = certifi.where()
    with session.get(PARQUET_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        downloaded = 0
        with tmp_parquet.open('wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (50 * 1024 * 1024) < 1024 * 1024:
                        print(f'  downloaded {downloaded / 1024 / 1024:.1f} MiB ({time.perf_counter()-t0:.1f}s)', flush=True)

    print(f'[STORIES] Downloaded {tmp_parquet.stat().st_size/1024/1024:.1f} MiB in {time.perf_counter()-t0:.1f}s; extracting clean stories...', flush=True)
    pf = pq.ParquetFile(tmp_parquet)
    seen_hashes: set[str] = set()
    train_chunks: list[bytes] = []
    val_chunks: list[bytes] = []
    train_bytes = 0
    val_bytes = 0
    scanned = 0
    accepted = 0

    for batch in pf.iter_batches(batch_size=2048, columns=['text']):
        for val in batch.column(0).to_pylist():
            scanned += 1
            if not isinstance(val, str):
                continue
            text = val.strip()
            if text.endswith('结束。'):
                text = text[:-3].strip()
            if len(text) < 80:
                continue
            doc_bytes = (text + '\n\n').encode('utf-8')
            digest = hashlib.sha256(doc_bytes).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            accepted += 1
            if int(digest[:8], 16) % 32 == 0:
                if val_bytes < TARGET_VAL_BYTES:
                    val_chunks.append(doc_bytes)
                    val_bytes += len(doc_bytes)
            else:
                if train_bytes < TARGET_TRAIN_BYTES:
                    train_chunks.append(doc_bytes)
                    train_bytes += len(doc_bytes)
            if train_bytes >= TARGET_TRAIN_BYTES and val_bytes >= TARGET_VAL_BYTES:
                break
        if train_bytes >= TARGET_TRAIN_BYTES and val_bytes >= TARGET_VAL_BYTES:
            break

    del pf
    try:
        tmp_parquet.unlink()
    except OSError:
        pass

    train_raw = b''.join(train_chunks)
    val_raw = b''.join(val_chunks)
    train_path = OUT_DIR / 'train.bin'
    val_path = OUT_DIR / 'validation.bin'
    train_path.write_bytes(train_raw)
    val_path.write_bytes(val_raw)

    manifest = dict(
        schema=1,
        dataset='RobinChen2001/TinyStories-Zh-2M',
        shard='data/train-00000-of-00003.parquet',
        partition='SHA256(story) mod 32 == 0 held out; exact story deduplication',
        content='warm, emotional, everyday Chinese children storybooks & family dialogues (zero politics/news)',
        rows_scanned=scanned,
        stories_accepted=accepted,
        train_path=str(train_path),
        train_bytes=len(train_raw),
        train_sha256=hashlib.sha256(train_raw).hexdigest(),
        validation_path=str(val_path),
        validation_bytes=len(val_raw),
        validation_sha256=hashlib.sha256(val_raw).hexdigest(),
    )
    (OUT_DIR / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'DONE: {accepted} warm Chinese stories -> train={len(train_raw)/1024/1024:.2f}MiB ({manifest["train_sha256"][:12]}) '
          f'val={len(val_raw)/1024:.1f}KiB ({manifest["validation_sha256"][:12]}) in {time.perf_counter()-t0:.1f}s',
          flush=True)


if __name__ == '__main__':
    main()
