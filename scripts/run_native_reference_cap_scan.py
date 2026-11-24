"""Serial native 6/7/8 reference-write scan with passive Figure 5 R receipts."""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from unittest.mock import patch

from scripts.measure_native_relation_percolation import build, sha
from scripts.run_speech_growth import FLAGS, REPO, write


CAPS = (8, 7, 6)


def snapshot(engine):
    db = engine.model._causal_archive.db
    count, = db.execute('SELECT COUNT(*) FROM relations').fetchone()
    rows = db.execute('SELECT relation_id,unknown_id,port,generation,seen_version '
                      'FROM relation_variables ORDER BY relation_id,unknown_id').fetchall()
    graph = build(dict(rows=rows, receipt=dict(causal_archive=dict(relations=count))))
    numerator = len(graph['largest'])
    return dict(R=numerator/count, numerator=numerator, denominator=count,
                edges=len(graph['witnesses']), references=len(rows), giant=sorted(graph['largest']))


def worker(args):
    import torch
    from scripts import run_arrived_language_actor as actor, run_native_causal_use_value as use
    from scripts.ib_local import autonomous_cycle, run_mother_continuation as continuation

    prepare, dispatch = continuation.prepare_continuation, autonomous_cycle.dispatch

    def prepared(opts, **kwargs):
        opts.relation_reference_write_cap = args.cap
        loaded = prepare(opts, **kwargs)
        if not loaded.config.get('speech_only') or loaded.config['world_source'] != 'text':
            raise ValueError('reference scan requires the native two-action text body')
        write(args.work/'initial_topology.json', snapshot(loaded.engine))
        return loaded

    def observed(engine):
        row = dispatch(engine)
        if row['autonomous_event']['kind'] == 'external':
            row['reference_topology'] = snapshot(engine)
        return row

    def material(engine):
        return torch.load(args.material, map_location=engine.device, weights_only=False)

    temp = args.temp/str(args.cap)
    command = ['--arrived-mode', 'observe', '--work', str(args.work),
               '--checkpoint', str(args.checkpoint), '--cycles', str(args.cycles),
               '--evaluate-every', str(args.cycles), '--table-storage', str(temp/'table'),
               '--world-activations', str(temp/'world'), '--policy-activations', str(temp/'policy'), *FLAGS]
    with (patch.object(continuation, 'prepare_continuation', prepared),
          patch.object(autonomous_cycle, 'dispatch', observed),
          patch.object(use, 'fixed_queries', material),
          patch.object(sys, 'argv', [sys.argv[0], *command])):
        actor.main()


def summarize(work, cap, source_sha, cycles):
    result = json.loads((work/'result.json').read_text(encoding='utf8'))
    rows = [r for line in (work/'events.jsonl').open(encoding='utf8')
            if (r := json.loads(line)).get('cycle')]
    config = json.loads((work/'body/config.json').read_text(encoding='utf8'))
    assert result['status'] == 'completed' and result['cycles'] == len(rows) == cycles
    assert result['provider_calls'] == 0 and result['initial_source_sha256'] == source_sha
    assert config['relation_reference_write_cap'] == cap
    assert [r['cycle'] for r in rows] == list(range(result['origin_cycle']+1, result['cycle']+1))
    trajectory = [dict(cycle=r['cycle'], choice=r['choice'], utterance_hex=r['utterance_hex'],
                       relation=r['causal_program']['roles'], question=r['shared_unknown'],
                       evidence_ids=r['joint_relation_update']['evidence_contract']['episode_ids'],
                       **r['reference_topology']) for r in rows]
    tail = [r['R'] for r in trajectory[cycles//2:]]
    return dict(cap=cap, updates=result['updates'], external_bytes=result['external_bytes'],
                checkpoint=result['checkpoint'], initial=json.loads((work/'initial_topology.json').read_text()),
                tail=dict(min=min(tail), median=statistics.median(tail), mean=statistics.mean(tail),
                          max=max(tail), distinct=len(set(tail))), trajectory=trajectory)


def scan(args):
    args.work.mkdir(parents=True, exist_ok=False)
    args.temp.mkdir(parents=True, exist_ok=False)
    status = args.work/'scan_status.json'
    source_sha = sha(args.checkpoint)
    arms = {}
    for cap in CAPS:
        assert shutil.disk_usage(args.temp).free >= 5*1024**3, 'insufficient activation storage'
        assert shutil.disk_usage(args.work).free >= 1024**3, 'insufficient checkpoint storage'
        write(status, dict(status='running', cap=cap, completed=list(arms), rounds_per_arm=args.cycles))
        work = args.work/str(cap)
        command = [sys.executable, '-u', '-m', 'scripts.run_native_reference_cap_scan', '--cap', str(cap),
                   '--checkpoint', str(args.checkpoint), '--material', str(args.material),
                   '--work', str(work), '--temp', str(args.temp), '--cycles', str(args.cycles)]
        with (args.work/f'{cap}.log').open('x', encoding='utf8') as log:
            process = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            write(status, dict(status='failed', cap=cap, completed=list(arms), exit_code=process.returncode))
            raise SystemExit(process.returncode)
        arms[cap] = summarize(work, cap, source_sha, args.cycles)
        write(args.work/f'trajectory_{cap}.json', arms[cap])
    assert all(arm['initial'] == arms[8]['initial'] for arm in arms.values())
    report = dict(status='completed', source_checkpoint=str(args.checkpoint), source_sha256=source_sha,
                  rounds_per_arm=args.cycles, observation_start=args.cycles//2+1, arms=arms)
    write(args.work/'result.json', report)
    write(status, dict(status='completed', summaries={b: a['tail'] for b, a in arms.items()}))
    print(json.dumps({b: a['tail'] for b, a in arms.items()}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('checkpoint', 'material', 'work', 'temp'):
        p.add_argument('--'+flag, type=Path, required=True)
    p.add_argument('--cycles', type=int, default=64)
    p.add_argument('--cap', type=int, choices=CAPS)
    args = p.parse_args()
    (worker if args.cap is not None else scan)(args)


if __name__ == '__main__':
    main()
