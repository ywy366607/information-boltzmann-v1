"""Invoke unmodified official RULER generators without POSIX shell quoting.

User-authorized RULER assessment; third-party code and generated data stay out
of the repository. Short lengths/subsets are explicitly not the standard score.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def sha256(path):
    # Checkpoints share a memory budget with the running learner. Hashing
    # must not temporarily allocate another complete checkpoint in RAM.
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('official', 'model', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--tasks', nargs='+', default=['niah_single_1', 'niah_multikey_1', 'vt'])
    p.add_argument('--lengths', type=int, nargs='+', default=[1024, 2048, 8192])
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--seed', type=int, default=71107)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output/'manifest.json'
    if target.exists():
        raise FileExistsError('preserve generated benchmark provenance')
    script_dir = args.official/'scripts/data'
    spec = importlib.util.spec_from_file_location('ruler_data_constants', script_dir/'synthetic/constants.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    configs = yaml.safe_load((args.official/'scripts/synthetic.yaml').read_text())
    # The upstream generators retry forever below their minimum haystack /
    # built-in demonstration sizes. Refuse those unsupported combinations.
    for task in args.tasks:
        if task == 'vt' or configs[task]['args'].get('type_haystack') == 'essay':
            if min(args.lengths) < 1024:
                raise ValueError('official VT/fixed essay minimum needs >=1024 with this tokenizer')
    revision = subprocess.check_output(['git','-C',str(args.official),'rev-parse','HEAD'], text=True).strip()
    manifest = dict(status='running',official_revision=revision, tokenizer=str(args.model), seed=args.seed,
        samples=args.samples, template='base + official answer_prefix',
        scope='configured short-distance admission and length curves; not standard 13-task RULER score',
        data=[])
    for length in args.lengths:
        for task in args.tasks:
            config = configs[task]
            family = config['task']
            base = module.TASKS[family]
            script = script_dir/'synthetic'/f'{family}.py'
            out = args.output/str(length)
            command = [sys.executable, str(script), '--save_dir', str(out), '--save_name',task,
                '--subset','test','--tokenizer_path',str(args.model),'--tokenizer_type','hf',
                '--max_seq_length',str(length),'--tokens_to_generate',str(base['tokens_to_generate']),
                '--num_samples',str(args.samples),'--random_seed',str(args.seed),
                '--template',base['template']+base.get('answer_prefix','')]
            for k,v in config['args'].items():
                command.extend(['--'+k,str(v)])
            # Windows receives an argv list, not prepare.py's shell-interpolated template.
            subprocess.run(command, check=True, timeout=300, env=os.environ.copy())
            data = out/task/'test.jsonl'
            manifest['data'].append(dict(task=task,family=family,length=length,path=str(data),
                sha256=sha256(data), generator_sha256=sha256(script),
                max_new_tokens=base['tokens_to_generate']))
            target.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    manifest['status'] = 'complete'
    target.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
