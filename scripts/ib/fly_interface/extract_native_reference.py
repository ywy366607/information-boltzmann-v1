"""Extract small native hypotheses from a complete retained archive, with provenance."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_native_policy_credit import native_parameter_names


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.decompressing.pt')
    if args.output.exists() or temporary.exists():
        raise FileExistsError('Reference extraction never overwrites a previous artifact')
    digest = hashlib.sha256()
    with args.archive.open('rb') as raw:
        for block in iter(lambda: raw.read(4 * 1024**2), b''):
            digest.update(block)
    with gzip.open(args.archive, 'rb') as source, temporary.open('wb') as target:
        shutil.copyfileobj(source, target, length=4 * 1024**2)
    saved = torch.load(temporary, mmap=True, map_location='cpu', weights_only=False)
    if saved.get('format') != 'fly-bptt-v1':
        raise ValueError('Reference source is not a complete retained learner')
    metadata = SimpleNamespace(named_parameters=lambda: (
        (name, SimpleNamespace(requires_grad=True)) for name in saved['model']))
    names = native_parameter_names(metadata)
    result = {'format': 'fly-native-reference-v1', 'bptt_train_tokens': saved['bptt_train_tokens'],
              'parameters': {name: saved['model'][name].clone() for name in names},
              'provenance': {'archive': str(args.archive.resolve()), 'sha256': digest.hexdigest(),
                             'source_format': saved['format'], 'native_names': list(names)}}
    torch.save(result, args.output)
    del saved
    # The exact, explicitly named decompression temporary is ours; all retained
    # organism archives remain untouched.
    temporary.unlink()
    args.output.with_suffix('.json').write_text(json.dumps(result['provenance'], indent=2), encoding='utf8')
    print(json.dumps({'path': str(args.output), 'trained': result['bptt_train_tokens'],
                      'bytes': args.output.stat().st_size, 'native_tensors': len(names)}))


if __name__ == '__main__':
    main()
