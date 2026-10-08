"""Extend pinned OWT streams while preserving every existing token prefix.

New documents use the original NFC/SHA256 split and are deduplicated against
all existing splits. The GPT-2 tokenizer is unchanged. No training replay or
epoch wrap is used to manufacture the requested token budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import ssl
import unicodedata

import numpy as np
import tiktoken


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/ib_owt_gpt2'))
    parser.add_argument('--output', type=Path, default=Path('data/ib_owt_gpt2_31m'))
    parser.add_argument('--min-train-tokens', type=int, default=31000000)
    parser.add_argument('--min-validation-tokens', type=int, default=200000)
    args = parser.parse_args()
    manifest = json.loads((args.source/'manifest.json').read_text())
    if manifest['tokenizer'] != 'gpt2':
        raise ValueError('A verified GPT-2 stream is required')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new output to preserve dataset provenance')
    args.output.mkdir(parents=True, exist_ok=True)
    seen = {row['sha256'] for row in manifest['documents']}
    last_index = max(row['source_index'] for row in manifest['documents'])
    records = list(manifest['documents'])
    counts, streams = {}, {}
    for split in ('train', 'validation', 'test'):
        source = args.source/f'{split}.npy'
        if digest(source) != manifest['files'][source.name]:
            raise ValueError(f'Source checksum mismatch: {split}')
        values = np.load(source, mmap_mode='r')
        counts[split] = len(values)
        target = (args.output/f'{split}.bin').open('wb')
        for start in range(0, len(values), 1048576):
            np.asarray(values[start:start+1048576], dtype=np.uint16).tofile(target)
        streams[split] = target
    tokenizer = tiktoken.get_encoding('gpt2')
    if tokenizer.n_vocab != manifest['vocab_size']:
        raise ValueError('Tokenizer vocabulary changed')
    try:
        ssl.create_default_context()
    except ssl.SSLError:
        import certifi
        from functools import partial
        ssl.create_default_context = partial(ssl.create_default_context, cafile=certifi.where())
    from datasets import load_dataset
    consumed = -1
    used_shards = []
    print('Existing prefixes copied; reading pinned OWT after source index', last_index, flush=True)
    try:
        complete = False
        for shard in manifest['parquet_shards']:
            used_shards.append(shard)
            url = f"hf://datasets/Skylion007/openwebtext@{manifest['revision']}/{shard}"
            rows = load_dataset('parquet', data_files={'train': [url]}, split='train', streaming=True)
            for row in rows:
                consumed += 1
                if consumed <= last_index:
                    continue
                text = unicodedata.normalize('NFC', row['text']).replace('\r\n', '\n').strip()
                key = hashlib.sha256(text.encode('utf-8')).hexdigest()
                if not text or key in seen:
                    continue
                seen.add(key)
                bucket = int(key, 16) % 100
                split = 'train' if bucket < 98 else 'validation' if bucket == 98 else 'test'
                ids = tokenizer.encode_ordinary(text)
                ids.append(tokenizer.eot_token)
                np.asarray(ids, dtype=np.uint16).tofile(streams[split])
                counts[split] += len(ids)
                records.append(dict(sha256=key, split=split, source_index=consumed))
                if len(records) % 1000 == 0:
                    for handle in streams.values():
                        handle.flush()
                    progress = dict(source_index=consumed, documents=len(records), token_counts=counts)
                    (args.output/'preparation_progress.json').write_text(json.dumps(progress, indent=2))
                    print(progress, flush=True)
                if counts['train'] >= args.min_train_tokens and counts['validation'] >= args.min_validation_tokens:
                    complete = True
                    break
            if complete:
                break
        if not complete:
            raise ValueError('Pinned dataset exhausted before requested fresh-data capacity')
    finally:
        for handle in streams.values():
            handle.close()
    files = {}
    for split, count in counts.items():
        temporary = args.output/f'{split}.bin'
        source = np.memmap(temporary, mode='r', dtype=np.uint16)
        if len(source) != count:
            raise ValueError(f'Written token count mismatch: {split}')
        path = args.output/f'{split}.npy'
        array = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint16, shape=(count,))
        for start in range(0, count, 1048576):
            array[start:start+1048576] = source[start:start+1048576]
        array.flush()
        del source, array
        temporary.unlink()
        files[path.name] = digest(path)
    shutil.copyfile(args.source/'tokenizer.json', args.output/'tokenizer.json')
    files['tokenizer.json'] = digest(args.output/'tokenizer.json')
    expanded = {**manifest, 'documents': records, 'token_counts': counts, 'files': files,
        'extension_parent': str(args.source),
        'extension_parent_manifest_sha256': digest(args.source/'manifest.json'),
        'preserved_prefix_token_counts': manifest['token_counts'],
        'new_source_range': [last_index+1, consumed+1],
        'shards_read_for_extension': used_shards,
        'deduplication': 'NFC document SHA256 across all existing and newly consumed splits',
        'stream_policy': 'append only; preserved tokenizer, split assignment and token prefixes'}
    (args.output/'manifest.json').write_text(json.dumps(expanded, indent=2)+'\n', encoding='utf-8')
    print('COMPLETED', counts, flush=True)


if __name__ == '__main__':
    main()
