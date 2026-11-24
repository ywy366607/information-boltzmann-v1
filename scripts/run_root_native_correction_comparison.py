"""Same-2553 complete learners: observe versus apply paid root residuals."""
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
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temp.replace(path)


def summary(work):
    from statistics import mean
    from scripts.summarize_joint_response import read_arm
    arms, runs, checks = {}, {}, {}
    for name in ('observe', 'apply'):
        run, rows, metrics = read_arm(work/name)
        stages = [json.loads((work/f'{name}_{i}_result.json').read_text(encoding='utf8')) for i in range(3)]
        metrics['seconds'] = sum(r['invocation']['seconds'] for r in stages)
        metrics['actual_use_sum_per_minute'] = sum(metrics['actual_use'])*60/metrics['seconds']
        patches = [r['causal_use']['epistemic_value']['root_native_correction'] for r in rows]
        metrics['root_patches'] = patches
        metrics['mean_absolute_critic_patch'] = mean(abs(v) for r in patches for v in r['applied_critic_delta'])
        metrics['positive_U_cycles'] = sum(r['metal_g'] > 0 for r in rows)
        metrics['policy_replay_max_error'] = max(r['policy_replay_max_error'] for r in rows)
        ys = [r['actual'] for r in metrics['pairs']]
        metrics['zero_prediction_MSE'] = mean(y*y for y in ys)
        restores = [json.loads((work/f'{name}_{i}_restore.json').read_text(encoding='utf8')) for i in range(3)]
        checks[name+'_complete'] = (run['cycle'], run['cycles'], run['updates']) == (2569, 16, 64)
        checks[name+'_restore'] = all(r['valid'] for r in restores)
        checks[name+'_natural'] = run['provider_calls'] == 0 and run['external_bytes'] == 15360
        checks[name+'_finite_U'] = all(r['causal_use']['finite_U']['verified'] for r in rows)
        checks[name+'_teacher_budget'] = all(r['teacher_count'] <= 2 for r in patches)
        checks[name+'_root_bound'] = all(r['contributions'] and all(c['options'] for c in r['contributions']) for r in patches)
        checks[name+'_mode'] = all(r['mode'] == name for r in patches)
        checks[name+'_interleaved'] = rows[-1]['persistent_rhythm']['interleave']
        arms[name], runs[name] = metrics, run
    checks['same_source'] = runs['observe']['initial_source_sha256'] == runs['apply']['initial_source_sha256']
    checks['same_initial_queries'] = runs['observe']['evaluations'][0] == runs['apply']['evaluations'][0]
    checks['observe_no_patch'] = arms['observe']['mean_absolute_critic_patch'] == 0
    checks['apply_live_patch'] = arms['apply']['mean_absolute_critic_patch'] > 0
    report = dict(valid=all(checks.values()), checks=checks, source_cycle=2553, arms=arms,
        scope='same full persistent/interleaved learner and root-only native teacher scope; replace evaluated successor mass before actor choice versus learn-only labels')
    write(Path('results/published/cgb_root_native_correction_2553_20261120.json'), report)
    assert report['valid'], checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--activation-storage', type=Path, required=True)
    a = p.parse_args()
    work, temporary = a.work.resolve(), a.activation_storage.resolve()
    source = Path('scratch/cgb_persistent_rhythm_2537_20261120/coupled/body/last.pt.gz').resolve()
    expected = digest(source)
    assert shutil.disk_usage(work.parent).free >= 3*1024**3, 'two complete body work copies'
    temporary.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(temporary).free >= 6*1024**3, 'measured backward-storage headroom'
    work.mkdir(exist_ok=False)
    files = [*Path('scripts/ib_local').glob('*.py'), Path(__file__),
             Path('scripts/run_native_preview_restore_comparison.py'),
             Path('scripts/audit_native_relation_use_restore.py'),
             Path('scripts/summarize_joint_response.py'), Path('scripts/audit_native_value_transition.py')]
    hashes = {str(f.resolve()): digest(f) for f in files}
    write(work/'source_hashes.json', hashes)
    state = dict(status='running', pid=os.getpid(), started=datetime.now(timezone.utc).isoformat(),
                 source=str(source), source_sha256=expected, plan=[4, 8, 4], cycles_per_arm=16)
    def update(**fields):
        state.update(fields)
        write(work/'pipeline_status.json', state)
    def run(stage, module, arguments):
        with (work/(stage+'.log')).open('w', encoding='utf8') as log:
            child = subprocess.Popen([sys.executable, '-m', module, *arguments], stdout=log, stderr=subprocess.STDOUT)
            update(stage=stage, child_pid=child.pid)
            code = child.wait()
        assert code == 0, f'{stage} failed ({code}); inspect {stage}.log'
        assert all(digest(f) == h for f, h in hashes.items()), 'runtime changed during comparison'
        assert digest(source) == expected, 'source body changed'
    try:
        for name in ('observe', 'apply'):
            for i, cycles in enumerate(state['plan']):
                assert shutil.disk_usage(work).free >= 2*1024**3, 'checkpoint headroom'
                assert shutil.disk_usage(temporary).free >= 6*1024**3, 'activation headroom'
                stage = f'{name}_{i}'
                run(stage, 'scripts.run_native_preview_restore_comparison', [
                    '--checkpoint', str(source), '--work', str(work/name), '--cycles', str(cycles),
                    '--native-decision', '--native-word-execution', '--successor-distribution', '--joint-successor',
                    '--batch-native-words', '--batch-preview-scores', '--amortized-decision', '--latent-action-planner',
                    '--persistent-rhythm', 'coupled', '--root-native-correction', name,
                    '--world-activations', str(temporary/'world'), '--policy-activations', str(temporary/'policy'),
                    '--table-storage', str(temporary/name/'table'), '--profile-runtime',
                    *(['--resume-work'] if i else [])])
                result = json.loads((work/name/'result.json').read_text(encoding='utf8'))
                assert result['status'] == 'completed'
                write(work/(stage+'_result.json'), result)
                run(stage+'_restore', 'scripts.audit_native_relation_use_restore', [
                    '--work', str(work/name), '--output', str(work/(stage+'_restore.json')),
                    '--table-storage', str(temporary/name/'restore_table')])
                update(completed_arm=name, completed_segment=i, cycle=result['cycle'])
        summary(work)
        update(status='completed', stage='completed', report='results/published/cgb_root_native_correction_2553_20261120.json')
    except Exception as error:
        update(status='failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
