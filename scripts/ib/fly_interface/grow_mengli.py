"""Chunked continuation of one mother-world individual, with explicit pause and checkpoint accounting."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def checkpoint_tokens(path):
    """Keep the supervisor lightweight: release the CPU-only loader before starting the CUDA learner."""
    code = ("import json,sys,torch; "
            "saved=torch.load(sys.argv[1],map_location='cpu',mmap=True,weights_only=False); "
            "print(json.dumps({'trained':int(saved['bptt_train_tokens'])}))")
    result = subprocess.check_output([sys.executable, '-c', code, str(path)], text=True)
    return int(json.loads(result.strip())['trained'])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--chunk-tokens', type=int, default=8192)
    parser.add_argument('--chunks', type=int, default=1)
    args = parser.parse_args()
    if args.chunk_tokens <= 0 or args.chunk_tokens % 32:
        parser.error('chunk-tokens must be a positive multiple of the 32-token training window')
    root = Path(__file__).resolve().parents[3]
    status = args.output / 'status'
    plan_path = args.output / 'chunk_plan.json'
    def note(text):
        with status.open('a', encoding='utf-8') as handle:
            handle.write(datetime.now().strftime('%m-%d %H:%M ') + text + '\n')
        print(text, flush=True)

    for _ in range(args.chunks):
        if (args.output / 'STOP').exists():
            note('mother-world continuation stopped at STOP; no new chunk launched')
            return
        trained = checkpoint_tokens(args.checkpoint_dir / 'last.pt')
        plan = json.loads(plan_path.read_text(encoding='utf-8')) if plan_path.exists() else None
        if plan is None or plan.get('status') == 'completed':
            text = status.read_text(encoding='utf-8') if status.exists() else ''
            done = [int(x) for x in re.findall(r'chunk (\d+) done:', text)]
            chunk = max(done, default=0) + 1
            plan = {'chunk': chunk, 'start_tokens': trained, 'target_tokens': trained + args.chunk_tokens,
                    'status': 'running', 'chunk_tokens': args.chunk_tokens}
        remaining = int(plan['target_tokens']) - trained
        if remaining <= 0:
            raise RuntimeError('Unfinished chunk plan is inconsistent with the saved checkpoint')
        plan['status'] = 'running'
        write_json(plan_path, plan)
        c = int(plan['chunk'])
        note(f'chunk {c} start: mother-world {trained}->{plan["target_tokens"]}, continuous state, ABA/AG deferred')
        command = [sys.executable, '-u', str(root / 'scripts/ib/train_fly_bptt_stream.py'),
                   '--resume', str(args.checkpoint_dir / 'last.pt'), '--extend-budget',
                   '--additional-tokens', str(remaining), '--output', str(args.output),
                   '--checkpoint-dir', str(args.checkpoint_dir), '--score-log', str(args.output / f'scores_c{c}.jsonl'),
                   '--skip-eval', '--checkpoint-every-tokens', '1024',
                   '--mother-world', 'deepseek', '--mother-speak-act', '--mother-credit', 'epistemic',
                   '--mother-credit-ticks', '64', '--mother-credit-tau', '32',
                   '--mother-voice', '--mother-voice-level', '1.5',
                   '--compartment-dendritic', '--no-eye-binary', '--noise-rate', '0',
                   '--log-word-dopamine', '--stream-pace-mean', '8', '--stream-pace-min', '4']
        with (args.output / 'train.log').open('a', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        progress = json.loads((args.output / 'progress.json').read_text(encoding='utf-8'))
        if result.returncode or progress['status'] != 'completed':
            plan['status'] = 'paused' if progress['status'] == 'paused' else 'failed'
            write_json(plan_path, plan)
            note(f'chunk {c} {plan["status"]}: {progress.get("bptt_train_tokens")} tokens, exit={result.returncode}')
            return
        saved_tokens = checkpoint_tokens(args.checkpoint_dir / 'last.pt')
        if saved_tokens != plan['target_tokens']:
            raise RuntimeError('Completed training was not saved at the chunk endpoint')
        plan['status'] = 'completed'
        write_json(plan_path, plan)
        note(f'chunk {c} done: mother-world tokens={saved_tokens}; chunk_words={saved_tokens - plan["start_tokens"]}; '
             f'speed={progress.get("speed_tokens_per_sec", 0):.3f}; ABA/AG deferred')


if __name__ == '__main__':
    main()
