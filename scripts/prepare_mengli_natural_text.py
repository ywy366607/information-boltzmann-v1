"""Download a bounded, document-disjoint real FineWeb-Edu text supply."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import requests


def prepare(output, train_bytes=16*1024*1024, validation_bytes=256*1024):
    output.mkdir(parents=True, exist_ok=False)
    dataset = 'HuggingFaceFW/fineweb-edu'
    session = requests.Session()
    response = session.get(f'https://huggingface.co/api/datasets/{dataset}', timeout=60)
    response.raise_for_status()
    revision = response.json()['sha']
    counts = dict(train=0, validation=0)
    seen = set()
    offset = 0
    with (output/'train.bin').open('wb') as train, (output/'validation.bin').open('wb') as valid, (output/'documents.jsonl').open('w', encoding='utf-8') as records:
        while counts['train'] < train_bytes or counts['validation'] < validation_bytes:
            for attempt in range(4):
                response = session.get('https://datasets-server.huggingface.co/rows', params=dict(
                    dataset=dataset, config='sample-10BT', split='train', offset=offset, length=100), timeout=90)
                if response.status_code == 200:
                    break
                if attempt == 3:
                    response.raise_for_status()
                time.sleep(2*(attempt+1))
            page = response.json()
            if not page.get('rows'):
                raise RuntimeError('source exhausted before the declared budget')
            for item in page['rows']:
                row = item['row']
                text = row.get('text')
                if not text or item.get('truncated_cells'):
                    continue
                raw = (text+'\n\n').encode('utf-8')
                digest = hashlib.sha256(raw).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                split = 'validation' if int(digest[:8],16) % 32 == 0 else 'train'
                target = train_bytes if split == 'train' else validation_bytes
                if counts[split] >= target:
                    continue
                stream = train if split == 'train' else valid
                start = stream.tell()
                stream.write(raw)
                records.write(json.dumps(dict(row=item['row_idx'], split=split, start=start, bytes=len(raw),
                    sha256=digest, id=row.get('id'), url=row.get('url')), ensure_ascii=False)+'\n')
                counts[split] += len(raw)
            offset += len(page['rows'])
            print(json.dumps(dict(rows_scanned=offset, **counts)), flush=True)
    final = session.get(f'https://huggingface.co/api/datasets/{dataset}', timeout=60)
    final.raise_for_status()
    if final.json()['sha'] != revision:
        raise RuntimeError('dataset changed during download; do not admit this supply')
    result = dict(schema=1, dataset=dataset, revision=revision, config='sample-10BT', split='train',
        retrieval='official dataset viewer rows API; reject truncated cells; pinned start/end revision',
        partition='SHA256(document) mod 32 == 0 held out; exact document deduplication',
        content='natural educational web text; no instruction-answer conversion or SFT', rows_scanned=offset)
    for split in counts:
        p = output/f'{split}.bin'
        result.update({f'{split}_path':str(p.resolve()),f'{split}_bytes':p.stat().st_size,
                       f'{split}_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    prepare(args.output)
