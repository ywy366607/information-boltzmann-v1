"""Same-source batching admission followed by continuous real-feedback learning.

Four initial cycles compare runtime with the calibrated reference; three
eight-cycle continuations keep the same individual, material, and live banks.
Every segment ends in a full checkpoint and restore audit. Paid calls stay zero.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temp.replace(path)


def summary(work, segments, reference, restore):
    from statistics import mean
    from scripts.summarize_joint_response import read_arm
    from scripts.audit_native_value_transition import summarize
    run, rows, metrics = read_arm(work/'joint')
    metrics['seconds'] = sum(segment['seconds'] for segment in segments)
    metrics['use_sum_per_minute'] = sum(metrics['actual_use'])*60/metrics['seconds']
    metrics['old_gain_per_minute'] = (metrics['old_query_after']-metrics['old_query_before'])*60/metrics['seconds']
    trend = []
    offset = 0
    for segment in segments:
        selected = rows[offset:offset+segment['cycles']]
        pairs = summarize(selected)
        teachers = [t for row in selected for t in row['causal_use']['native_amortization']['action_planner']['teacher_checks']]
        receipts = [row['causal_use']['native_amortization']['action_planner'] for row in selected]
        rank_pairs = [r['teacher_checks'] for r in receipts if len(r['teacher_checks'])==2]
        trend.append(dict(segment=segment['stage'], cycles=[r['cycle'] for r in selected],
            mean_actual_use=mean(r['metal_g'] for r in selected),
            actual_policy_discrepancy_MSE=mean(p['components']['transport']**2 for p in pairs),
            realized_future_MSE=mean(p['error']**2 for p in pairs),
            teacher_MSE=mean(t['error']**2 for t in teachers) if teachers else None,
            successor_condition_rank_agreement=mean(
                (pair[0]['predicted']-pair[1]['predicted'])*(pair[0]['native']-pair[1]['native']) > 0
                for pair in rank_pairs) if rank_pairs else None,
            real_selected_prequential_MSE=mean((r['world_prediction']-r['world_target'])**2
                for r in receipts if r['world_prediction'] is not None and r['world_target'] is not None),
            teacher_checks=teachers,
            profile=segment['profile'], seconds=segment['seconds']))
        offset += segment['cycles']
    checks = dict(completed=run['status']=='completed', origin=run['origin_cycle']==2493,
        source=run['initial_source_sha256']==reference['source_sha256'],
        original_U=all(r['causal_use']['finite_U']['verified'] for r in rows if r['causal_use']['finite_U']['committed']),
        original_payment=all(r['causal_use']['reward_version']=='finite_relation_use_v5' for r in rows),
        world_labels=all(r['causal_use']['native_amortization']['action_planner']['new_world_label']==
            r['causal_use']['finite_U']['committed'] for r in rows),
        bounded_banks=all(r['causal_use']['native_amortization']['action_planner']['records'] <=
            r['causal_use']['native_amortization']['action_planner']['capacity'] for r in rows),
        teacher_budget=all(0 <= r['causal_use']['native_amortization']['action_planner']['teacher_count'] <= 2 for r in rows),
        natural_text=run['provider_calls']==0, full_restore=restore['valid'],
        original_queries=run['evaluations'][0]==reference['initial_evaluation'])
    return dict(valid=all(checks.values()), checks=checks, run=dict(cycle=run['cycle'], cycles=run['cycles'],
        updates=run['updates'], external_bytes=run['external_bytes']), metrics=metrics,
        segments=trend, evaluations=run['evaluations'], admission=reference.get('admission'),
        calibrated_reference=reference['metrics'], restore=restore,
        units='actual_use is original finite-U/v5; independent old facts retain fixed identities',
        purpose='latent candidate ranking learns from sparse native teachers AND actual world feedback')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--continuation-segments', type=int, default=3)
    parser.add_argument('--resume', action='store_true', help='continue completed segments after a recorded failure')
    parser.add_argument('--activation-storage', type=Path,
                        help='scratch volume for automatically deleted world/policy backward files')
    args = parser.parse_args()
    if args.activation_storage is not None:
        args.activation_storage = args.activation_storage.resolve()
        args.activation_storage.mkdir(parents=True, exist_ok=True)
        assert shutil.disk_usage(args.activation_storage).free >= 6*1024**3, 'need 6 GiB for measured activation peaks'
    assert args.continuation_segments >= 0
    work = args.work.resolve()
    assert shutil.disk_usage(work.parent).free >= (2 if args.resume else 3)*1024**3, 'need checkpoint replacement headroom'
    source = Path('scratch/cgb_momentum_value_2489_20261119/carry/body/last.pt.gz').resolve()
    source_sha = '7cccc744d4549eca2662727cba6b4c900609393243b9fb557dee6a2f17498107'
    assert digest(source)==source_sha
    if args.resume:
        old_status = json.loads((work/'pipeline_status.json').read_text(encoding='utf8'))
        assert old_status['status']=='failed', 'resume only a recorded failed pipeline'
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        shutil.copy2(work/'pipeline_status.json', work/f'interrupted_status_{stamp}.json')
        shutil.copy2(work/'source_hashes.json', work/f'interrupted_sources_{stamp}.json')
        # The saved result/checkpoint form one committed boundary. Keep the
        # entire interrupted log, then replay only events through that boundary.
        last = json.loads((work/'joint/result.json').read_text(encoding='utf8'))
        event_path = work/'joint/events.jsonl'
        shutil.copy2(event_path, work/f'interrupted_events_{stamp}.jsonl')
        lines = event_path.read_text(encoding='utf8').splitlines(keepends=True)
        cut = next(i+1 for i in range(len(lines)-1, -1, -1)
                   if json.loads(lines[i]).get('cycle')==last['cycle'])
        event_path.write_text(''.join(lines[:cut]), encoding='utf8')
        evaluations = json.loads((work/'joint/evaluations.json').read_text(encoding='utf8'))
        write(work/'joint/evaluations.json', [r for r in evaluations if r['cycle'] <= last['cycle']])
    else:
        work.mkdir(exist_ok=False)
    files = [*Path('scripts/ib_local').glob('*.py'), Path(__file__),
             Path('scripts/run_native_preview_restore_comparison.py'),
             Path('scripts/audit_native_relation_use_restore.py'),
             Path('scripts/summarize_joint_response.py'), Path('scripts/audit_native_value_transition.py')]
    hashes = {str(p.resolve()): digest(p) for p in files}
    write(work/'source_hashes.json', hashes)
    state = dict(status='running', pid=os.getpid(), started=datetime.now(timezone.utc).isoformat(),
                 source_sha256=source_sha, plan=[4]+[8]*args.continuation_segments)
    def status(**values):
        state.update(values); write(work/'pipeline_status.json', state)
    def run(stage, module, argv):
        with (work/(stage+'.log')).open('w', encoding='utf8') as log:
            child = subprocess.Popen([sys.executable, '-m', module, *argv], stdout=log, stderr=subprocess.STDOUT)
            status(stage=stage, child_pid=child.pid)
            code = child.wait()
        assert code==0, f'{stage} failed: {code}; see {stage}.log'
        assert all(digest(Path(p))==h for p,h in hashes.items()), 'runtime source changed'
    from scripts.summarize_joint_response import read_arm
    old, _, old_metrics = read_arm(Path('scratch/cgb_latent_action_loop_2493_20261120_calibrated_gain/joint'))
    reference = dict(source_sha256=old['initial_source_sha256'], initial_evaluation=old['evaluations'][0], metrics=old_metrics)
    segments = []
    if args.resume:
        state['resumed_from'] = old_status
        for index, cycles in enumerate(state['plan']):
            stage = 'admission' if index==0 else f'learning_{index}'
            path, audit = work/(stage+'_result.json'), work/(stage+'_restore.json')
            if not path.exists() or not audit.exists():
                break
            result = json.loads(path.read_text(encoding='utf8'))
            assert json.loads(audit.read_text(encoding='utf8'))['valid']
            segments.append(dict(stage=stage, cycles=cycles, seconds=result['invocation']['seconds'],
                                 profile=result['runtime_profile']))
        previous_report = json.loads(Path('results/published/cgb_latent_planner_learning_2493_20261120.json').read_text(encoding='utf8'))
        reference['admission'] = previous_report['admission']
    report_path = Path('results/published/cgb_latent_planner_learning_2493_20261120.json')
    try:
        for index, cycles in enumerate(state['plan']):
            if index < len(segments):
                continue
            assert shutil.disk_usage(work.parent).free >= 2*1024**3, 'need replacement headroom'
            stage = 'admission' if index==0 else f'learning_{index}'
            log_stage = stage+(f'_retry_{stamp}' if args.resume else '')
            run(log_stage, 'scripts.run_native_preview_restore_comparison', [
                '--checkpoint', str(source), '--work', str(work/'joint'), '--cycles', str(cycles),
                '--native-decision','--native-word-execution','--successor-distribution',
                '--joint-successor','--batch-native-words','--batch-preview-scores',
                '--amortized-decision','--latent-action-planner','--profile-runtime',
                *(['--world-activations',str(args.activation_storage/'world'),
                   '--policy-activations',str(args.activation_storage/'policy')]
                  if args.activation_storage is not None else []),
                *(['--resume-work'] if index else [])])
            result = json.loads((work/'joint/result.json').read_text(encoding='utf8'))
            segments.append(dict(stage=stage, cycles=cycles, seconds=result['invocation']['seconds'],
                                 profile=result['runtime_profile']))
            write(work/(stage+'_result.json'), result)
            run(stage+'_restore', 'scripts.audit_native_relation_use_restore',
                ['--work', str(work/'joint'), '--output', str(work/(stage+'_restore.json'))])
            restored = json.loads((work/(stage+'_restore.json')).read_text(encoding='utf8'))
            if index==0:
                _, _, metric = read_arm(work/'joint')
                reference['admission'] = dict(metrics=metric, speedup=old_metrics['seconds']/metric['seconds'],
                    same_material=result['evaluations'][0]==old['evaluations'][0],
                    old_fact_gain_retained=(metric['old_query_after']-metric['old_query_before'])/
                        (old_metrics['old_query_after']-old_metrics['old_query_before']))
            report = summary(work, segments, reference, restored)
            write(report_path, report)
            assert report['valid'], report['checks']
            assert digest(source)==source_sha
            status(completed_segments=len(segments), cycle=result['cycle'], cycles=result['cycles'],
                   updates=result['updates'], external_bytes=result['external_bytes'], report=str(report_path))
        status(status='completed', stage='completed', source_unchanged=True)
    except Exception as error:
        status(status='failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
