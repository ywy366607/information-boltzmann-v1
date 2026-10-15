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
    parser.add_argument('--extra', default='', help='Extra trainer switches for every chunk (they are inherited anyway)')
    parser.add_argument('--probes', action=argparse.BooleanOptionalAction, default=True,
                        help='After each chunk: dopamine cells alive (alpha3_probe), surprise -> DA (live_da_report), '
                             'act timing (live_act_report); one summary line each in the status')
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
                   '--log-word-dopamine'] + args.extra.split()     # the word interval is a profile switch: inherited
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
        if args.probes:
            run_probes(root, args, c, note)


def run_probes(root, args, c, note):
    """The brain's own circuits after each chunk (diagnostics on the saved state; the brain is not frozen for any gate):
    the dopamine cells alive, surprise -> phasic DA on the chunk's own words, the acts' timing within interval."""
    fi = root / 'scripts/ib/fly_interface'
    out = args.output
    env = {**__import__('os').environ, 'PYTHONIOENCODING': 'utf-8', 'FLY_REUSE_TRANSMISSION': '0'}

    def run(cmd, log):
        with open(log, 'w', encoding='utf-8') as handle:
            return subprocess.run(cmd, cwd=root, stdout=handle, stderr=subprocess.STDOUT, env=env).returncode

    if run([sys.executable, '-u', str(fi / 'alpha3_probe.py'), str(args.checkpoint_dir), str(out / f'alpha3_c{c}.json')],
           out / f'alpha3_c{c}.txt') == 0:
        r = json.loads((out / f'alpha3_c{c}.json').read_text(encoding='utf-8'))
        note(f'c{c} dopamine cells: rate {json.dumps({k: round(v, 4) for k, v in r["rate_per_tick"].items()})}; '
             f'v/theta {json.dumps({k: round(v[1], 3) for k, v in r["median_v_over_theta"].items()})}')
    else:
        note(f'c{c} alpha3 probe FAILED')
    da = subprocess.run([sys.executable, str(fi / 'live_da_report.py'), str(out / f'scores_c{c}.jsonl')], cwd=root,
                        capture_output=True, text=True, env=env)
    note(f'c{c} (2) surprise~DA: {da.stdout.strip()[:600] or "FAILED"}')
    # gate 2 at the compartments (da_info_probe): does the ordinary words' dopamine carry anything (surprisal, word
    # identity, decline over repeats), and how much of the memory it writes per word
    if run([sys.executable, '-u', str(fi / 'da_info_probe.py'), str(args.checkpoint_dir), str(out), '256',
            str(out / f'da_info_c{c}.json')], out / f'da_info_c{c}.txt') == 0:
        r = json.loads((out / f'da_info_c{c}.json').read_text(encoding='utf-8'))
        note(f'c{c} (2b) compartment dopamine: surprisal rho {r["spearman drive~surprisal"]}, identity eta2 '
             f'{r["identity eta2 (>=3 occurrences)"]}, repeat trend {r["repeat trend (spearman occurrence~relative drive)"]}, '
             f'edges written per word (median) {r["edges written share"][1]}')
    else:
        note(f'c{c} da_info probe FAILED')
    trace = out / 'ai_trace.jsonl'
    if trace.exists():
        act = subprocess.run([sys.executable, str(fi / 'live_act_report.py'), str(trace)], cwd=root,
                             capture_output=True, text=True, env=env)
        note(f'c{c} (3) acts: {act.stdout.strip().splitlines()[-1][:900] if act.stdout.strip() else "FAILED"}')


if __name__ == '__main__':
    main()
