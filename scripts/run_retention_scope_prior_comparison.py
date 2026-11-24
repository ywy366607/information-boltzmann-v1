"""Complete learners with independent local retention or identity-bound parent priors."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from statistics import mean

from scripts.run_root_native_correction_comparison import digest, write


def summarize(work):
    from scripts.summarize_joint_response import read_arm
    arms, runs, checks = {}, {}, {}
    for name in ('off', 'relation'):
        run, rows, metrics = read_arm(work/name)
        transport = [r['use_transport'] for r in rows]
        metrics['append_use_mean'] = mean(r['mean']['append_gain'] for r in transport)
        metrics['release_use_mean'] = mean(r['mean']['release_effect'] for r in transport)
        metrics['positive_U_cycles'] = sum(r['metal_g'] > 0 for r in rows)
        metrics['actual_use_sum_per_minute'] = sum(metrics['actual_use'])*60/metrics['seconds']
        fresh = [r['causal_use']['finite_U']['fresh_use'] for r in rows]
        metrics['cold_cycles'] = sum(not any(m[3] > 0 for m in f['consumed']) for f in fresh)
        metrics['prior_read_cycles'] = sum(any(m[3] > 0 for m in f.get('prior_consumed', [])) for f in fresh)
        metrics['actual_prior_selection'] = [r['shared_unknown']['revision']['retention_use']['trace'].get('selection') for r in rows]
        metrics['policy_replay_max_error'] = max(r['policy_replay_max_error'] for r in rows)
        restore = json.loads((work/f'{name}_restore.json').read_text(encoding='utf8'))
        checks[name+'_complete'] = (run['cycle'], run['cycles'], run['updates']) == (2585, 16, 64)
        checks[name+'_restore'] = restore['valid']
        checks[name+'_natural'] = run['provider_calls'] == 0 and run['external_bytes'] == 15360
        checks[name+'_finite_U'] = all(r['causal_use']['finite_U']['verified'] for r in rows)
        checks[name+'_native_budget'] = all(r['causal_use']['native_amortization']['action_planner']['teacher_count'] <= 2 for r in rows)
        checks[name+'_rhythm'] = rows[-1]['persistent_rhythm']['interleave']
        checks[name+'_unchanged_root_mode'] = all(r['causal_use']['epistemic_value']['root_native_correction']['mode'] == 'observe' for r in rows)
        arms[name], runs[name] = metrics, run
    checks['same_source'] = runs['off']['initial_source_sha256'] == runs['relation']['initial_source_sha256']
    checks['same_initial_queries'] = runs['off']['evaluations'][0] == runs['relation']['evaluations'][0]
    checks['control_no_parent_read'] = arms['off']['prior_read_cycles'] == 0
    checks['candidate_live_parent_read'] = arms['relation']['prior_read_cycles'] > 0
    report = dict(valid=all(checks.values()), checks=checks, source_cycle=2569, arms=arms,
        scope='same persistent/interleaved full learner; parent relation moments read by same fact identity only where the typed unknown has no own observations')
    output = Path('results/published/cgb_retention_scope_prior_2569_20261120.json')
    write(output, report)
    assert report['valid'], checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--activation-storage', type=Path, required=True)
    a = p.parse_args()
    work, temporary = a.work.resolve(), a.activation_storage.resolve()
    previous = Path('scratch/cgb_root_native_correction_2553_20261120/observe')
    old = json.loads((previous/'result.json').read_text(encoding='utf8'))
    assert old['status'] == 'completed' and old['cycle'] == 2569
    source = (previous/'body/last.pt.gz').resolve()
    expected = digest(source)
    # One 16-cycle segment per arm reduces rolling working/archive copies.
    # Fast tables are embedded in every full checkpoint. Each of the four
    # sequential child processes gets one private UUID mmap directory, bounding
    # new table work copies to four instead of twelve in a 3-segment protocol.
    assert shutil.disk_usage(work.parent).free >= 3*1024**3, 'two independent full body checkpoints'
    temporary.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(temporary).free >= 6*1024**3, 'measured backward headroom'
    work.mkdir(exist_ok=False)
    files = [*Path('scripts/ib_local').glob('*.py'), Path(__file__),
             Path('scripts/run_root_native_correction_comparison.py'),
             Path('scripts/run_native_preview_restore_comparison.py'),
             Path('scripts/audit_native_relation_use_restore.py'),
             Path('scripts/summarize_joint_response.py'), Path('scripts/audit_native_value_transition.py')]
    hashes = {str(f.resolve()): digest(f) for f in files}
    write(work/'source_hashes.json', hashes)
    state = dict(status='running', pid=os.getpid(), started=datetime.now(timezone.utc).isoformat(),
                 source=str(source), source_sha256=expected, cycles_per_arm=16,
                 work_table_root=str(temporary/'table'), table_ownership='four sequential private UUID copies; full state restored from immutable checkpoint')
    def update(**fields):
        state.update(fields); write(work/'pipeline_status.json', state)
    def run(stage, module, arguments):
        with (work/(stage+'.log')).open('w', encoding='utf8') as log:
            child = subprocess.Popen([sys.executable, '-m', module, *arguments], stdout=log, stderr=subprocess.STDOUT)
            update(stage=stage, child_pid=child.pid)
            code = child.wait()
        assert code == 0, f'{stage} failed ({code}); inspect {stage}.log'
        assert all(digest(f) == h for f, h in hashes.items()), 'runtime changed during comparison'
        assert digest(source) == expected, 'source body changed'
    try:
        for name in ('off', 'relation'):
            assert shutil.disk_usage(work).free >= 2*1024**3, 'body checkpoint headroom'
            assert shutil.disk_usage(temporary).free >= 6*1024**3, 'backward headroom'
            run(name, 'scripts.run_native_preview_restore_comparison', [
                '--checkpoint', str(source), '--work', str(work/name), '--cycles', '16',
                '--native-decision', '--native-word-execution', '--successor-distribution', '--joint-successor',
                '--batch-native-words', '--batch-preview-scores', '--amortized-decision', '--latent-action-planner',
                '--persistent-rhythm', 'coupled', '--root-native-correction', 'observe',
                '--retention-scope-prior', name, '--table-storage', str(temporary/'table'),
                '--world-activations', str(temporary/'world'), '--policy-activations', str(temporary/'policy'),
                '--profile-runtime'])
            run(name+'_restore', 'scripts.audit_native_relation_use_restore', [
                '--work', str(work/name), '--output', str(work/f'{name}_restore.json'),
                '--table-storage', str(temporary/'table')])
            update(completed_arm=name, cycle=2585)
        summarize(work)
        update(status='completed', stage='completed', report='results/published/cgb_retention_scope_prior_2569_20261120.json')
    except Exception as error:
        update(status='failed', error=str(error)); raise


if __name__ == '__main__':
    main()
