"""Repeat the actual largest-group experiment with identity-coupled preview.

Serial private continuations reuse the durable 2401 state and actual graph
certificate. The order arm permutes storage presentation before shared reading.
"""
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
    p = argparse.ArgumentParser()
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    args = p.parse_args()
    args.work = args.work.resolve()
    args.previous = args.previous.resolve()
    old = json.loads((args.previous/'plan.json').read_text(encoding='utf8'))
    args.task = Path(old['checkpoint']['path']).parent.parent
    args.work.mkdir(exist_ok=True)
    plan = copy.deepcopy(old)
    plan['identity_preview_revision'] = 1
    plan['layer_streamed_observation'] = True
    plan['baseline_reused_from'] = None
    plan['arms'] = [a for a in old['arms'] if a['arm'] in ('baseline','bridge_cut','matched_cut')]
    order = copy.deepcopy(next(a for a in plan['arms'] if a['arm']=='baseline'))
    order.update(arm='order_only', fact_order_only=dict(target_order=[], boundary='stored_fact_presentation'))
    plan['arms'].append(order)
    plan['runtime_sources'] = {name:sha(ROOT/name) for name in (
        'scripts/ib_local/identity_preview.py','scripts/ib_local/causal_relation_program.py',
        'scripts/ib_local/navigation_causal_program.py','scripts/ib_local/relation_evidence.py',
        'scripts/ib_local/generative_relations.py',
        'scripts/native_fact_order_control.py','scripts/run_native_relation_mode_use.py',
        'scripts/native_observation_memory.py')}
    if (args.work/'plan.json').exists():
        assert json.loads((args.work/'plan.json').read_text(encoding='utf8')) == plan
    else:
        write(args.work/'plan.json', plan)
    if not (args.work/'anchor.pt.gz').exists():
        os.link(args.previous/'anchor.pt.gz', args.work/'anchor.pt.gz')
    assert sha(args.work/'anchor.pt.gz') == plan['checkpoint']['sha256']
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        env[name] = '1'
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:native,garbage_collection_threshold:0.4'
    results = {}
    for arm in ('baseline','order_only','bridge_cut','matched_cut'):
        directory, _ = prepare_private(args, arm)
        if not (directory/'result.json').exists():
            with (directory/'stdout.log').open('w', encoding='utf8') as out, (directory/'stderr.log').open('w', encoding='utf8') as err:
                child = subprocess.Popen([sys.executable,'-u','-m','scripts.run_native_relation_mode_use',
                    '--task',str(args.task),'--work',str(args.work),'--worker','--arm',arm,'--cycles','8'],
                    cwd=ROOT, env=env, stdout=out, stderr=err,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                write(args.work/'status.json',dict(status='running',arm=arm,worker_pid=child.pid,completed=list(results)))
                code = child.wait()
            if code:
                write(args.work/'status.json',dict(status='failed',arm=arm,exit_code=code,stderr=str(directory/'stderr.log')))
                raise RuntimeError(f'{arm} exit {code}: {directory / "stderr.log"}')
        result = json.loads((directory/'result.json').read_text(encoding='utf8'))
        assert result['status']=='completed' and result['actual_optimizer_calls']==32
        results[arm] = result
        assert result['before']==results['baseline']['before']
        if arm=='order_only':
            assert result['fact_order_interventions']
            assert result['after']==results['baseline']['after'], 'whole carried state must agree after order-only presentation'
        print(json.dumps(dict(arm=arm,status='completed',seconds=result['seconds']),ensure_ascii=False),flush=True)
    assert sha(plan['checkpoint']['path']) == plan['checkpoint']['sha256']
    assert sha(plan['archive']['path']) == plan['archive']['sha256']
    summary = dict(status='completed',plan=plan,main_checkpoint_and_archive_unchanged=True,complete_starts_equal=True,
        whole_carried_order_equivalence=True,
        results={name:dict(seconds=r['seconds'],optimizer_calls=r['actual_optimizer_calls'],
            final_R=r['final_R'],read_counts=r['read_counts'],provider_calls=r['provider_calls'],
            changes=compare_choices(results['baseline'],r)) for name,r in results.items()})
    write(args.work/'summary.json',summary)
    write(ROOT/'results/published/cgb_native_identity_preview_2401_20261117.json',summary)
    write(args.work/'status.json',dict(status='completed',summary=str(args.work/'summary.json')))


if __name__=='__main__':
    main()
