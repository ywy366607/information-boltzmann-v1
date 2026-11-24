"""Serial same-source full-life comparison of native teacher allocation.

Both arms start at the completed 2521 individual, preserve its learned banks,
and continue natural-text life with identical native/real settlement budgets.
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
        for block in iter(lambda:f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temp.replace(path)


def summarize(work, output):
    from scripts.summarize_joint_response import read_arm
    arms, runs, rows = {}, {}, {}
    for name in ('control','competing'):
        run, events, metrics = read_arm(work/name)
        segments = [json.loads((work/f'{name}_{i}_result.json').read_text(encoding='utf8')) for i in range(2)]
        metrics['seconds'] = sum(r['invocation']['seconds'] for r in segments)
        metrics['use_sum_per_minute'] = sum(metrics['actual_use'])*60/metrics['seconds']
        metrics['fixed_old_gain'] = metrics['old_query_after']-metrics['old_query_before']
        metrics['fixed_old_gain_per_minute'] = metrics['fixed_old_gain']*60/metrics['seconds']
        runs[name], rows[name], arms[name] = run, events, metrics
    selected = [r['causal_use']['native_amortization']['action_planner'] for r in rows['competing']]
    restore = {name:json.loads((work/f'{name}_1_restore.json').read_text(encoding='utf8'))
               for name in arms}
    checks = dict(completed=all(r['status']=='completed' and r['cycles']==16 and r['cycle']==2537 for r in runs.values()),
        same_source=len({r['initial_source_sha256'] for r in runs.values()})==1,
        same_queries=runs['control']['evaluations'][0]==runs['competing']['evaluations'][0],
        restored=all(r['valid'] and r['cycle']==2537 for r in restore.values()),
        natural=all(r['provider_calls']==0 for r in runs.values()),
        finite_U=all(r['causal_use']['finite_U']['verified'] for arm in rows.values() for r in arm
                     if r['causal_use']['finite_U']['committed']),
        same_payment=all(r['causal_use']['reward_version']=='finite_relation_use_v5' for arm in rows.values() for r in arm),
        teacher_budget=all(r['causal_use']['native_amortization']['action_planner']['teacher_count']<=2
                           for arm in rows.values() for r in arm),
        root_coverage=all(r['teacher_selection']['covered_roots']==r['teacher_selection']['available_roots'] for r in selected),
        distinct_roots=all(len(set(r['teacher_selection']['selected_roots']))==r['teacher_count'] for r in selected))
    result = dict(valid=all(checks.values()), checks=checks, arms=arms, restore=restore,
        source_sha256=runs['control']['initial_source_sha256'], source_cycle=2521,
        policy='control is novelty/diversity, candidate covers competing actual root actions; same two native calls',
        interpretation='compare real finite-U, fixed old facts, actual next-policy/future errors and time; teacher residuals reflect selected-condition distributions')
    write(output,result)
    assert result['valid'], checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    args=parser.parse_args();work=args.work.resolve()
    assert shutil.disk_usage(work.parent).free>=9*1024**3, 'two work copies plus measured temporary activation peaks'
    source=Path('scratch/cgb_latent_planner_learning_2493_20261120/joint/body/last.pt.gz').resolve()
    expected='5947465a2a45d541a8c27b829aad062a3213c8064ddd693b53ef0dfe93e32830'
    assert digest(source)==expected
    work.mkdir(exist_ok=False)
    files=[*Path('scripts/ib_local').glob('*.py'),Path(__file__),Path('scripts/run_native_preview_restore_comparison.py'),
           Path('scripts/audit_native_relation_use_restore.py'),Path('scripts/summarize_joint_response.py'),
           Path('scripts/audit_native_value_transition.py')]
    hashes={str(p.resolve()):digest(p) for p in files};write(work/'source_hashes.json',hashes)
    status=dict(status='running',pid=os.getpid(),source_sha256=expected,
                started=datetime.now(timezone.utc).isoformat(),cycles_per_arm=16)
    def state(**fields):
        status.update(fields);write(work/'pipeline_status.json',status)
    def run(stage,module,arguments):
        with (work/(stage+'.log')).open('w',encoding='utf8') as log:
            child=subprocess.Popen([sys.executable,'-m',module,*arguments],stdout=log,stderr=subprocess.STDOUT)
            state(stage=stage,child_pid=child.pid)
            code=child.wait()
        assert code==0,f'{stage} failed: {code}'
        assert all(digest(Path(p))==h for p,h in hashes.items()),'runtime changed during comparison'
        assert digest(source)==expected,'2521 source changed'
    try:
        for name in ('competing','control'):
            for segment in range(2):
                assert shutil.disk_usage(work).free>=6*1024**3,'temporary activation headroom'
                stage=f'{name}_{segment}'
                run(stage,'scripts.run_native_preview_restore_comparison',[
                    '--checkpoint',str(source),'--work',str(work/name),'--cycles','8',
                    '--native-decision','--native-word-execution','--successor-distribution','--joint-successor',
                    '--batch-native-words','--batch-preview-scores','--amortized-decision','--latent-action-planner',
                    '--teacher-selection','competing_actions' if name=='competing' else 'novelty_diversity',
                    '--world-activations',str(work/'temporary/world'),'--policy-activations',str(work/'temporary/policy'),
                    '--profile-runtime',*(['--resume-work'] if segment else [])])
                result=json.loads((work/name/'result.json').read_text(encoding='utf8'))
                write(work/(stage+'_result.json'),result)
                run(stage+'_restore','scripts.audit_native_relation_use_restore',[
                    '--work',str(work/name),'--output',str(work/(stage+'_restore.json'))])
                state(completed_arm=name,completed_segment=segment,cycle=result['cycle'])
        output=Path('results/published/cgb_competing_action_teachers_2521_20261120.json')
        summarize(work,output)
        state(status='completed',stage='completed',report=str(output),source_unchanged=True)
    except Exception as error:
        state(status='failed',error=str(error))
        raise


if __name__=='__main__':
    main()
