"""Download real reading sources and build a small caregiver-reviewed shelf.

The shelf supplies optional material, never a timed or exclusive subject schedule.
Full original files and source hashes remain in the local, ignored data directory.
"""
import hashlib
import json
from pathlib import Path
import re

import requests

ROOT = Path(__file__).resolve().parents[3]


def main():
    shelf = ROOT / 'data/harness/reading'
    shelf.mkdir(parents=True, exist_ok=True)
    books = {}
    sources = []
    for number, title in ((21, 'Three hundred Aesop’s fables'),
                          (11, "Alice's Adventures in Wonderland"), (24623, 'Mother Goose')):
        url = f'https://www.gutenberg.org/cache/epub/{number}/pg{number}.txt'
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        raw = response.content
        (shelf / f'pg{number}.txt').write_bytes(raw)
        text = raw.decode('utf-8-sig').replace('\r\n', '\n')
        books[number] = text
        sources.append({'id': number, 'title': title, 'url': url,
                        'catalog': f'https://www.gutenberg.org/ebooks/{number}',
                        'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                        'license_note': 'See the retained Project Gutenberg source and license.'})
    entries = []
    for title in ('The Crow and the Pitcher', 'The North Wind and the Sun', 'The Hare and the Tortoise',
                  'The Oak and the Reeds', 'The Travelers and the Plane-Tree'):
        match = re.search(r'^' + re.escape(title) + r'\n', books[21], re.M)
        if not match:
            raise ValueError(f'Missing reviewed fable: {title}')
        story = re.split(r'\n{4,}', books[21][match.end():].strip(), maxsplit=1)[0]
        entries.append({'id': f'fable_{len(entries)}', 'title': title, 'strand': 'causal_exploration',
                        'source': 21, 'text': ' '.join(story.split()),
                        'use': 'Share a short coherent part, then listen. Discuss causes or compare possibilities only when relevant.'})
    for title in ('Pat a cake, pat a cake', 'Rain, rain, go to Spain', 'One, two, three, four, five',
                  'Twinkle, twinkle, little star'):
        paragraphs = re.split(r'\n\s*\n', books[24623])
        stanza = next((p for p in paragraphs if title.lower() in p.lower() and len(p.split()) > 10), None)
        if stanza:
            entries.append({'id': f'rhyme_{len(entries)}', 'title': title,
                            'strand': 'quantity_structure' if title.startswith('One') else 'expression',
                            'source': 24623, 'text': ' '.join(stanza.split()),
                            'use': 'A rhyme can be enjoyed or varied; never require copying its last word.'})
    alice = books[11]
    start = alice.index('Alice was beginning to get very tired')
    paragraphs = re.split(r'\n\s*\n', alice[start:])
    for i, paragraph in enumerate(paragraphs[:12]):
        entries.append({'id': f'alice_{i}', 'title': f'Alice opening, passage {i+1}',
                        'strand': 'language', 'source': 11, 'text': ' '.join(paragraph.split()),
                        'use': 'Read or clearly retell a short part; story events are fictional. Preserve the passage context.'})
    plan = {'version': 1, 'sources': sources, 'entries': entries,
            'policy': 'Optional interleaved reading resources. Ling reviews fit and revisits; no clock or stage restricts access.'}
    (shelf / 'shelf.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'sources': len(sources), 'reviewed_entries': len(entries),
                      'words_available': sum(len(b.split()) for b in books.values()),
                      'path': str(shelf / 'shelf.json')}))


if __name__ == '__main__':
    main()
