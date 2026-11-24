"""Three complete continuations: nested planning, persistent port, and coupling."""
import argparse
from collections import Counter
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
        for block in iter(lambda: stream.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    temp.replace(path)


def summary(work):
    from scripts.summarize_joint_response import read_arm
    arms, runs, checks = {}, {}, {}
    for name in ('control','port','coupled'):
        run, rows, metrics = read_arm(work/name)
        stages = [json.loads((work/f'{name}_{i}_result.json').read_text(encoding='utf8')) for i in range(3)]
        metrics['seconds'] = sum(r['invocation']['seconds'] for r in stages)
        metrics['actual_use_sum_per_minute'] = sum(metrics['actual_use'])*60/metrics['seconds']
        metrics['old_gain_per_minute'] = (metrics['old_query_after']-metrics['old_query_before'])*60/metrics['seconds']
        events = [json.loads(s) for s in (work/name/'events.jsonl').read_text(encoding='utf8').splitlines()]
        metrics['event_kinds'] = dict(Counter(r['autonomous_event']['kind'] for r in events))
        metrics['within_external_choices'] = sum(r['autonomous_event'].get('within_external',False) for r in events)
        metrics['within_external_quiet'] = sum(r['autonomous_event'].get('within_external',False)
            and r['autonomous_event']['kind']=='internal' for r in events)
        metrics['last_rhythm'] = rows[-1].get('persistent_rhythm')
        metrics['world_activation_high_water'] = max(r['world_activation_bytes'] for r in rows)
        restore = json.loads((work/f'{name}_2_restore.json').read_text(encoding='utf8'))
        checks[name+'_complete'] = run['status']=='completed' and run['cycle']==2553 and run['cycles']==16
        checks[name+'_restore'] = restore['valid'] and restore['cycle']==2553
        checks[name+'_U'] = all(r['causal_use']['finite_U']['verified'] for r in rows if r['causal_use']['finite_U']['committed'])
        checks[name+'_labels'] = all(r['causal_use']['native_amortization']['action_planner']['new_world_label']
                                    for r in rows if r['causal_use']['finite_U']['committed'])
        checks[name+'_native_budget'] = all(r['causal_use']['native_amortization']['action_planner']['teacher_count']<=2 for r in rows)
        checks[name+'_natural'] = run['provider_calls']==0 and run['external_bytes']==15360
        arms[name], runs[name] = metrics, run
    checks['same_source'] = len({r['initial_source_sha256'] for r in runs.values()})==1
    checks['same_initial_queries'] = runs['control']['evaluations'][0]==runs['port']['evaluations'][0]==runs['coupled']['evaluations'][0]
    checks['live_quiet_coupling'] = arms['coupled']['within_external_quiet']>0
    checks['separate_port_control'] = arms['port']['within_external_choices']==0
    report = dict(valid=all(checks.values()),checks=checks,arms=arms,source_cycle=2537,
        scope='same complete natural-text learner; persistent conditional programs versus nested recomputation, then interleaved S/P learning events',
        interpretation='quality and time are evaluated together; continuation values are re-encoded learned predictions; source individual is preserved')
    write(Path('results/published/cgb_persistent_rhythm_2537_20261120.json'),report)
    assert report['valid'],checks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--activation-storage',type=Path,required=True)
    a=p.parse_args(); work=a.work.resolve(); temporary=a.activation_storage.resolve()
    source=Path('scratch/cgb_competing_action_teachers_2521_20261120/control/body/last.pt.gz').resolve()
    expected='8e86b636f3085daa98f72468ccdfbc7f9ca2425317ef9c7f4456907ed3a4d52c'
    assert digest(source)==expected
    assert shutil.disk_usage(work.parent).free>=5*1024**3,'three work copies and rolling checkpoints'
    temporary.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(temporary).free>=6*1024**3,'measured backward-storage headroom'
    work.mkdir(exist_ok=False)
    files=[*Path('scripts/ib_local').glob('*.py'),Path(__file__),Path('scripts/run_native_preview_restore_comparison.py'),
           Path('scripts/audit_native_relation_use_restore.py'),Path('scripts/summarize_joint_response.py'),
           Path('scripts/audit_native_value_transition.py')]
    hashes={str(f.resolve()):digest(f) for f in files};write(work/'source_hashes.json',hashes)
    state=dict(status='running',pid=os.getpid(),started=datetime.now(timezone.utc).isoformat(),
               source_sha256=expected,plan=[4,8,4],cycles_per_arm=16,activation_storage=str(temporary))
    def update(**fields):state.update(fields);write(work/'pipeline_status.json',state)
    def run(stage,module,arguments):
        with (work/(stage+'.log')).open('w',encoding='utf8') as log:
            child=subprocess.Popen([sys.executable,'-m',module,*arguments],stdout=log,stderr=subprocess.STDOUT)
            update(stage=stage,child_pid=child.pid)
            code=child.wait()
        assert code==0,f'{stage} failed ({code}); inspect {stage}.log'
        assert all(digest(f)==h for f,h in hashes.items()),'runtime changed during comparison'
        assert digest(source)==expected,'source checkpoint changed'
    try:
        for name in ('coupled','port','control'):
            for i,cycles in enumerate(state['plan']):
                assert shutil.disk_usage(work).free>=2*1024**3,'body checkpoint headroom'
                assert shutil.disk_usage(temporary).free>=6*1024**3,'activation headroom'
                stage=f'{name}_{i}'
                run(stage,'scripts.run_native_preview_restore_comparison',[
                    '--checkpoint',str(source),'--work',str(work/name),'--cycles',str(cycles),
                    '--native-decision','--native-word-execution','--successor-distribution','--joint-successor',
                    '--batch-native-words','--batch-preview-scores','--amortized-decision','--latent-action-planner',
                    '--world-activations',str(temporary/'world'),'--policy-activations',str(temporary/'policy'),
                    '--table-storage',str(temporary/name/'table'),
                    '--profile-runtime',*(['--persistent-rhythm',name] if name!='control' else []),
                    *(['--resume-work'] if i else [])])
                result=json.loads((work/name/'result.json').read_text(encoding='utf8'))
                assert result['status']=='completed'
                write(work/(stage+'_result.json'),result)
                run(stage+'_restore','scripts.audit_native_relation_use_restore',[
                    '--work',str(work/name),'--output',str(work/(stage+'_restore.json')),
                    '--table-storage',str(temporary/name/'restore_table')])
                update(completed_arm=name,completed_segment=i,cycle=result['cycle'])
        summary(work)
        update(status='completed',stage='completed',report='results/published/cgb_persistent_rhythm_2537_20261120.json')
    except Exception as error:
        update(status='failed',error=str(error));raise


if __name__=='__main__': main()
