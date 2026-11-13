"""Freeze prior mother-world fork replies for research-side validation.

The previous 48-state fork was a research sidecar, not the current online
mother interaction. Preserve ordering and provenance; never feed this file
to the live world or use it for gradient updates.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('scratch/claude_mother_fork_48_20261105.json'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = args.source.read_bytes()
    rows = json.loads(data)
    replies = [fork['mother'] for row in rows for fork in row['forks'].values()
               if isinstance(fork,dict) and isinstance(fork.get('mother'),str) and fork['mother']]
    text = ('\n'.join(replies)+'\n').encode('utf-8')
    args.output.mkdir(parents=True)
    (args.output/'mother_validation.bin').write_bytes(text)
    report = {'source':str(args.source), 'source_sha256':hashlib.sha256(data).hexdigest(),
              'text_sha256':hashlib.sha256(text).hexdigest(), 'bytes':len(text),
              'replies':len(replies), 'unique_replies':len(set(replies)),
              'scope':'existing 11-05 v4q mother-fork replies; fixed research-side validation',
              'caveat':'old real provider outputs, not a representative external language benchmark'}
    (args.output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))


if __name__ == '__main__':
    main()
