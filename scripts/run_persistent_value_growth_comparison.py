"""Serial, source-frozen complete-life comparison of persistent residual refinement."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def summarize(work, rounds):
    result = load(work/'run/result.json')
    readings = load(work/'independent_scores.json')['readings']
    events = [json.loads(line) for line in (work/'run/events.jsonl').read_text(encoding='utf8').splitlines()]
    external = [r for r in events if r['autonomous_event']['kind'] == 'external']
    assert result['cycles'] == rounds and len(external) == rounds
    assert result['updates'] == sum(r['updates'] for r in external) == 4*rounds
    assert result['provider_calls'] == 0
    timing = [json.loads(line) for line in (work/'dispatch_timing.jsonl').read_text(encoding='utf8').splitlines()]
    seconds = sum(r['dispatch_seconds'] for r in timing)
    gains = {key:dict(mean=readings['0']['mean_nll']-value['mean_nll'],
        segments=[a-b for a,b in zip(readings['0']['segment_nll'],value['segment_nll'])])
        for key,value in readings.items() if key != '0'}
    return dict(cycle=result['cycle'], rounds=rounds, updates=result['updates'],
        external_bytes=result['external_bytes'], self_bytes=result['self_bytes'],
        independent_gain=gains, dispatch_seconds=seconds,
        mean_payment=sum(r['metal_g'] for r in external)/rounds,
        old_fact_evaluation=result['evaluations'],
        choices=[r['choice'] for r in external],
        native_teachers=sum(r['causal_use']['native_amortization']['action_planner']['teacher_count'] for r in external),
        finite_U_verified=all(r['causal_use']['finite_U']['verified'] for r in external),
        max_policy_replay_error=max(r['policy_replay_max_error'] for r in external),
        mean_gain_per_minute=gains[str(rounds)]['mean']*60/seconds,
        refinement=None if not (work/'persistent_residual_runtime.json').exists() else
            load(work/'persistent_residual_runtime.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--temporary', type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    args.temporary.mkdir(parents=True, exist_ok=False)
    source = Path('scratch/cgb_wood_policy_use_16rounds_20261120/natural/run/body/last.pt.gz').resolve()
    source_sha = digest(source)
    assert source_sha == 'd673ffaa3168b153fcd853ad6af76f9567fa73336dc44ed129ebdd780e31a729'
    validation = Path('data/cgb_natural_text_20261115/validation.bin').resolve()
    validation_sha = digest(validation)
    assert validation_sha == 'b09c3879960cab5b5bbe3d3282059d472d7ee140925c58d7550b7fedebfbee93'
    starts = [40960, 114688, 196608, 245760]
    old_starts = [0, 95629, 191258, 286887, 32768, 65536, 131072, 229376,
                  16384, 81920, 163840, 262144]
    assert all(s+2049 <= validation.stat().st_size for s in starts)
    assert all(s+2049 <= old or old+2049 <= s for s in starts for old in old_starts)
    protocol = dict(schema=1, source=str(source), source_sha256=source_sha, source_cycle=2601,
        natural_rounds=8, arms=['original', 'refined'], max_world_rounds=16,
        max_optimizer_updates=64, observation_rounds=[], prefix_paths={},
        score_rounds=[2, 4, 8], record_actions=True, record_dispatch_timing=True,
        intervention='root persistent scalar second residual fit throughout refined life',
        candidate='fixed two fits; original ridge, features, sources, teacher budget, reward and update rules',
        validation=dict(path=str(validation), sha256=validation_sha, starts=starts,
            bytes_per_segment=2048, total_scored_bytes=8192, used_in_training=False),
        comparison='same complete source and RNG; both sample normally and retain all original learning',
        endpoint='difference of final independent improvement; 2/4/8 trajectory, old facts, choices and cost',
        unit='two complete living trajectories; materials are readouts of each trajectory',
        stop='8 real rounds per arm; stop and preserve on failure; no automatic retry or extra arm')
    write(args.work/'protocol.json', protocol)
    files = [*Path('scripts/ib_local').glob('*.py'), Path(__file__),
        Path('scripts/run_wood_policy_use_worker.py'), Path('scripts/run_native_preview_restore_comparison.py'),
        Path('scripts/native_value_residual_filter.py'), Path('scripts/native_policy_activation_storage.py')]
    hashes = {str(path.resolve()):digest(path) for path in files}
    write(args.work/'source_hashes.json', hashes)
    assert shutil.disk_usage(args.work).free > 1.9*1024**3
    assert shutil.disk_usage(args.temporary).free > 2.0*1024**3
    state = dict(status='running', pid=os.getpid(), started=datetime.now(timezone.utc).isoformat(),
        stage='original', completed=[], charged_world_rounds=0, results={})
    write(args.work/'pipeline_status.json', state)
    try:
        for arm in protocol['arms']:
            state.update(stage=arm, charged_world_rounds=state['charged_world_rounds']+8)
            command = [sys.executable, '-m', 'scripts.run_wood_policy_use_worker', '--mode', 'natural',
                '--checkpoint', str(source), '--work', str(args.work/arm),
                '--temporary', str(args.temporary/arm), '--protocol', str(args.work/'protocol.json')]
            if arm == 'refined':
                command.append('--persistent-residual-refinement')
            with (args.work/f'{arm}.log').open('w', encoding='utf8') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                state['child_pid'] = child.pid
                write(args.work/'pipeline_status.json', state)
                assert child.wait() == 0, f'{arm} failed; see {arm}.log'
            assert digest(source) == source_sha
            assert all(digest(path) == expected for path,expected in hashes.items()), 'active runtime changed'
            state['results'][arm] = summarize(args.work/arm, 8)
            state['completed'].append(arm)
            write(args.work/'pipeline_status.json', state)
        old, new = state['results']['original'], state['results']['refined']
        difference = {str(k):new['independent_gain'][str(k)]['mean']-old['independent_gain'][str(k)]['mean']
                      for k in protocol['score_rounds']}
        report = dict(protocol=protocol, arms=state['results'], refined_minus_original=difference,
            next='use whole-life benefit and cost to choose the next persistent-value experiment')
        write(args.work/'report.json', report)
        state.update(status='completed', stage='finished', child_pid=None,
            completed_at=datetime.now(timezone.utc).isoformat())
        write(args.work/'pipeline_status.json', state)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        write(args.work/'pipeline_status.json', state)
        raise


if __name__ == '__main__':
    main()
