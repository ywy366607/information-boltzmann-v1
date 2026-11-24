"""Serial 2x2 actual internal-preview / learned-dynamics feedback experiment."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.run_native_relation_mode_use import ROOT, prepare_private, sha, write
from scripts.analyze_native_relation_mode_use import compare_choices


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--release-observer-workspace',action='store_true')
    p.add_argument('--private-storage-root',type=Path)
    args=p.parse_args()
    args.previous=args.previous.resolve();args.work=args.work.resolve()
    old=json.loads((args.previous/'plan.json').read_text(encoding='utf8'))
    args.task=Path(old['checkpoint']['path']).parent.parent
    args.work.mkdir(exist_ok=True)
    plan=copy.deepcopy(old)
    base=next(a for a in old['arms'] if a['arm']=='baseline')
    modes={'baseline':(False,False),'bridge_cut':(True,False),
           'matched_cut':(False,True),'order_only':(True,True)}
    plan['arms']=[]
    for name,(preview,feedback) in modes.items():
        row=copy.deepcopy(base)
        row.update(arm=name,inner_loop=dict(hold_preview=preview,hold_feedback=feedback))
        plan['arms'].append(row)
    plan['experiment']='2x2 preview temporal contrast / dynamics parameter commit'
    plan['release_observer_workspace']=args.release_observer_workspace
    if args.private_storage_root:plan['private_storage_root']=str(args.private_storage_root.resolve())
    plan['legacy_directory_labels']={'baseline':'live/live','bridge_cut':'held-preview/live-feedback',
        'matched_cut':'live-preview/held-feedback','order_only':'held/held'}
    plan['training']='all original updates, S/P dynamics, endogenous scheduling, navigation, real text and persistent revision'
    plan['runtime_sources']={name:sha(ROOT/name) for name in (
        'scripts/native_inner_loop_intervention.py','scripts/native_topology_observer.py',
        'scripts/run_native_relation_mode_use.py','scripts/native_observation_memory.py')}
    plan['proof']=dict(path='lean/CGBAxioms/reviews/NativeInnerLoopActual2401.lean',
                       sha256=sha(ROOT/'lean/CGBAxioms/reviews/NativeInnerLoopActual2401.lean'))
    if (args.work/'plan.json').exists():
        previous_plan=json.loads((args.work/'plan.json').read_text(encoding='utf8'))
        if previous_plan != plan:
            assert args.release_observer_workspace
            for key in ('arms','checkpoint','archive','identity_preview_revision'):
                assert previous_plan[key]==plan[key]
            old_manifest=args.work/'plan_before_workspace_release.json'
            if not old_manifest.exists():write(old_manifest,previous_plan)
            plan['workspace_revision_from']=str(old_manifest)
            write(args.work/'plan.json',plan)
    else:write(args.work/'plan.json',plan)
    if not (args.work/'anchor.pt.gz').exists():os.link(args.previous/'anchor.pt.gz',args.work/'anchor.pt.gz')
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[key]='1'
    env['PYTORCH_CUDA_ALLOC_CONF']='backend:native,garbage_collection_threshold:0.4'
    previous=json.loads((args.previous/'baseline/result.json').read_text(encoding='utf8'))
    results={}
    for arm in modes:
        directory,_=prepare_private(args,arm)
        if not (directory/'result.json').exists():
            with (directory/'stdout.log').open('w',encoding='utf8') as out,(directory/'stderr.log').open('w',encoding='utf8') as err:
                child=subprocess.Popen([sys.executable,'-u','-m','scripts.run_native_relation_mode_use',
                    '--task',str(args.task),'--work',str(args.work),'--worker','--arm',arm,'--cycles','8'],
                    cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                write(args.work/'status.json',dict(status='running',arm=arm,meaning=plan['legacy_directory_labels'][arm],
                    worker_pid=child.pid,completed=list(results)))
                code=child.wait()
            if code:
                write(args.work/'status.json',dict(status='failed',arm=arm,exit_code=code,stderr=str(directory/'stderr.log')))
                raise RuntimeError(f'{arm} failed; inspect {directory / "stderr.log"}')
        result=json.loads((directory/'result.json').read_text(encoding='utf8'))
        assert result['actual_optimizer_calls']==32 and result['provider_calls']==0
        assert result['before']==previous['before']
        if arm=='baseline':
            assert result['after']==previous['after'], 'passive port instrumentation must preserve complete carried state'
        results[arm]=result
        print(json.dumps(dict(arm=arm,seconds=result['seconds'],status='completed')),flush=True)
    assert sha(plan['checkpoint']['path'])==plan['checkpoint']['sha256']
    assert sha(plan['archive']['path'])==plan['archive']['sha256']
    summary=dict(status='completed',main_source_hashes_preserved=True,baseline_carried_state_equals_prior=True,
        plan=plan,arms={arm:dict(seconds=r['seconds'],updates=r['actual_optimizer_calls'],
            intervention=plan['legacy_directory_labels'][arm],changes=compare_choices(results['baseline'],r),
            receipts=r['inner_loop_interventions'],carried=r['after']) for arm,r in results.items()})
    write(args.work/'summary.json',summary)
    write(ROOT/'results/published/cgb_native_inner_loop_use_2401_20261117.json',summary)
    write(args.work/'status.json',dict(status='completed'))


if __name__=='__main__':main()
