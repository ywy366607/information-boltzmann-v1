"""Run the approved 512 natural-text rounds as four complete continuations."""
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
SOURCE = Path('D:/cgb_speech_only_2605_20261121/body/last.pt.gz')
FLAGS = ['--native-decision', '--native-word-execution', '--successor-distribution',
         '--joint-successor', '--batch-native-words', '--batch-preview-scores',
         '--amortized-decision', '--latent-action-planner', '--persistent-rhythm',
         'coupled', '--root-native-correction', 'observe', '--retention-scope-prior', 'off']


def write(path, data):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    pending.replace(path)


def summary(log, seen):
    rows = [r for line in log.open(encoding='utf8') if (r := json.loads(line)).get('cycle')]
    samples = []
    counts = dict(revisited=0, spoke_on_revisit=0, different_from_previous_on_revisit=0,
                  handover_on_revisit=0, matching_prior_evidence_scored=0)
    for r in rows:
        q = r['shared_unknown']['signature']
        old = seen.get(q)
        words = r['utterance_hex']
        revisited = r['shared_unknown']['recalled']
        counts['revisited'] += int(revisited)
        counts['spoke_on_revisit'] += int(bool(words) and revisited)
        counts['different_from_previous_on_revisit'] += int(bool(words) and old is not None and words != old[0])
        counts['handover_on_revisit'] += int(revisited and r['expression']['reason'] == 'handover')
        if old is not None:
            fresh = r['causal_use']['finite_U'].get('fresh_use') or {}
            value = fresh.get('benefits', {}).get(old[1])
            if value is not None:
                samples.append(value)
        seen[q] = (words, r['causal_program'].get('committed_episode_identity'))
    counts['matching_prior_evidence_scored'] = len(samples)
    return dict(**counts, evidence_gain=None if not samples else dict(min=min(samples),
                median=statistics.median(samples), mean=statistics.mean(samples), max=max(samples)))


def main(root, scratch):
    root.mkdir(parents=True, exist_ok=False)
    scratch.mkdir(parents=True, exist_ok=False)
    status = root/'growth_status.json'
    start, source = 2613, SOURCE
    parts = []
    seen = {}
    for end in range(2741, 3126, 128):
        work = root/str(end)
        if shutil.disk_usage(root).free < 1024**3 or shutil.disk_usage(scratch).free < 5*1024**3:
            write(status, dict(status='stopped_disk_headroom', cycle=start, parts=parts))
            return
        write(status, dict(status='running', cycle=start, segment_end=end, target=3125, parts=parts))
        command = [sys.executable, '-u', '-m', 'scripts.run_arrived_language_actor',
                   '--arrived-mode', 'observe', '--work', str(work), '--checkpoint', str(source),
                   '--cycles', '128', '--evaluate-every', '128', '--table-storage', str(scratch/'table'/str(end)),
                   '--world-activations', str(scratch/'world'), '--policy-activations', str(scratch/'policy'), *FLAGS]
        with (root/f'{end}.log').open('x', encoding='utf8') as log:
            result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write(status, dict(status='failed', cycle=start, segment_end=end, exit_code=result.returncode, parts=parts))
            return
        r = json.loads((work/'result.json').read_text(encoding='utf8'))
        if (r['status'], r['cycle'], r['cycles'], r['provider_calls']) != ('completed', end, 128, 0):
            write(status, dict(status='incomplete', cycle=start, segment_end=end, result=r, parts=parts))
            return
        start, source = end, Path(r['checkpoint'])
        parts.append(dict(cycle=end, checkpoint=str(source), updates=r['updates'],
                          bytes=r['external_bytes'], **summary(work/'events.jsonl', seen)))
    write(status, dict(status='completed', cycle=start, parts=parts))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
