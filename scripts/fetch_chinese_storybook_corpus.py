"""Fetch clean natural Chinese text (HuggingFaceFW/fineweb-2 cmn_Hani) via official Datasets Server API.

Matches the exact partition & provenance discipline of `data/cgb_natural_text_20261115`:
  - Official HuggingFace Datasets Server rows API (`certifi`-verified HTTPS),
  - Exact SHA-256 document deduplication,
  - Deterministic `SHA256(doc) % 32 == 0` held-out validation split (`validation.bin`),
  - Natural Chinese web text only (no SFT/instruction templating).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import certifi
import requests


OUT_DIR = Path('D:/information-boltzmann-v1/data/cgb_chinese_natural_20261124').resolve()
DATASET = 'HuggingFaceFW/fineweb-2'
CONFIG = 'cmn_Hani'
SPLIT = 'train'
TARGET_TRAIN_BYTES = 10 * 1024 * 1024  # 10 MiB clean UTF-8 Chinese natural text
TARGET_VAL_BYTES = 256 * 1024          # >= 256 KiB held-out Chinese validation


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.verify = certifi.where()

    seen_hashes: set[str] = set()
    train_chunks: list[bytes] = []
    val_chunks: list[bytes] = []
    train_bytes = 0
    val_bytes = 0
    offset = 0
    rows_scanned = 0
    accepted = 0
    batch_len = 100

    t0 = time.perf_counter()
    while train_bytes < TARGET_TRAIN_BYTES or val_bytes < TARGET_VAL_BYTES:
        r = None
        for attempt in range(4):
            try:
                r = session.get(
                    'https://datasets-server.huggingface.co/rows',
                    params=dict(dataset=DATASET, config=CONFIG, split=SPLIT,
                                offset=offset, length=batch_len),
                    timeout=20,
                )
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1.5 * (attempt + 1))
        if r is None or r.status_code != 200:
            raise RuntimeError(f'Datasets Server fetch failed at offset={offset}: {getattr(r, "status_code", None)}')

        data = r.json()
        rows = data.get('rows', [])
        if not rows:
            break
        offset += len(rows)
        rows_scanned += len(rows)

        for item in rows:
            if item.get('truncated_cells'):
                continue
            row = item.get('row') or {}
            text = (row.get('text') or '').strip()
            score = float(row.get('language_score') or 0.0)
            if len(text) < 120 or score < 0.85:
                continue
            doc_bytes = (text + '\n\n').encode('utf-8')
            digest = hashlib.sha256(doc_bytes).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            accepted += 1
            if int(digest[:8], 16) % 32 == 0:
                val_chunks.append(doc_bytes)
                val_bytes += len(doc_bytes)
            else:
                if train_bytes < TARGET_TRAIN_BYTES:
                    train_chunks.append(doc_bytes)
                    train_bytes += len(doc_bytes)

        if rows_scanned % 500 == 0:
            print(f'  scanned={rows_scanned} accepted={accepted} '
                  f'train={train_bytes/1024/1024:.2f}MiB val={val_bytes/1024:.1f}KiB '
                  f'({time.perf_counter()-t0:.1f}s)', flush=True)

    train_raw = b''.join(train_chunks)
    val_raw = b''.join(val_chunks)
    train_path = OUT_DIR / 'train.bin'
    val_path = OUT_DIR / 'validation.bin'
    train_path.write_bytes(train_raw)
    val_path.write_bytes(val_raw)

    manifest = dict(
        schema=1,
        dataset=DATASET,
        config=CONFIG,
        split=SPLIT,
        retrieval='official dataset viewer rows API; reject truncated cells; language_score >= 0.85',
        partition='SHA256(document) mod 32 == 0 held out; exact document deduplication',
        content='natural Simplified Chinese web text; no SFT/instruction conversion',
        rows_scanned=rows_scanned,
        documents_accepted=accepted,
        train_path=str(train_path),
        train_bytes=len(train_raw),
        train_sha256=hashlib.sha256(train_raw).hexdigest(),
        validation_path=str(val_path),
        validation_bytes=len(val_raw),
        validation_sha256=hashlib.sha256(val_raw).hexdigest(),
    )
    (OUT_DIR / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'DONE: train={len(train_raw)/1024/1024:.2f}MiB ({manifest["train_sha256"][:12]}) '
          f'val={len(val_raw)/1024:.1f}KiB ({manifest["validation_sha256"][:12]}) in {time.perf_counter()-t0:.1f}s',
          flush=True)


if __name__ == '__main__':
    main()
