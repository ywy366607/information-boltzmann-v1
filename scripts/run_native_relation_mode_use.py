"""Whole-loop causal use of the measured four-relation component.

Three serial, complete learner branches start at the same durable body. Only
actual registered references are withdrawn once; subsequent native learning,
selection, sharing, autonomous events and navigation proceed without clamps.
No checkpoint, main archive, pending scope, world rule or objective is edited.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')
    tmp.replace(path)


def sql_rows(path):
    with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True) as db:
        return db.execute('SELECT relation_id,unknown_id,port,generation,seen_version '
                          'FROM relation_variables ORDER BY relation_id,unknown_id').fetchall()


def plan(args):
    from scripts.measure_native_relation_percolation import build, export_lean, read_frames
    measured_path = ROOT/'results/published/cgb_native_relation_percolation_2401_20261117.json'
    measured = json.loads(measured_path.read_text(encoding='utf8'))
    frame = read_frames(args.task/'work'/measured['frames'][-1]['segment'])[-1]
    graph = build(frame)
    # Identified from the completed measurement, before any intervention outcome.
    discoveries = measured['actual_identity_reconnections']
    assert len(discoveries) == 1
    group = set(map(tuple, discoveries[0]['scopes']))
    cut = set()
    for u, supports in graph['supports'].items():
        inside, outside = supports & group, supports - group
        if not inside or not outside:
            continue
        # Remove the smaller side of each actual witness. Ties retain all
        # group incidences. No within-group shared witness is removed.
        side = inside if len(inside) < len(outside) else outside
        cut.update((v, u) for v in side)
    cut = sorted(cut)
    assert all(not (sum(v in group for v in graph['supports'][u]) > 1 and v in group)
               for v, u in cut)
    main_db = next((args.task/'body').glob('causal_archive_*.sqlite'))
    with sqlite3.connect(main_db.resolve().as_uri()+'?mode=ro', uri=True) as db:
        touched = {( (r, g), (u,p) ): t for r,u,p,g,t in db.execute(
            'SELECT relation_id,unknown_id,port,generation,touched FROM relation_variables')}
        fact_counts = dict(db.execute('SELECT unknown_id,COUNT(*) FROM unknown_evidence GROUP BY unknown_id'))
    forbidden_unknowns = {u for _,u in cut}
    candidates = [k for k in sorted(graph['dependencies']) if k[0] not in group
                  and k[1] not in forbidden_unknowns and len(graph['supports'][k[1]]) > 1]
    controls, matches = [], []
    for target in cut:
        tv, tu = target
        def cost(k):
            v,u = k
            if u[1] != tu[1]:
                return float('inf')
            # Match actual reference type, support size, live graph degree,
            # fact budget and observed-use age; no outcome enters matching.
            return (abs(math.log(len(graph['supports'][u])/len(graph['supports'][tu]))) +
                abs(math.log((1+len(graph['neighbors'][v]))/(1+len(graph['neighbors'][tv])))) +
                abs(fact_counts[u[0]]-fact_counts[tu[0]]) +
                abs(math.log((1+abs(touched[k]))/(1+abs(touched[target])))))
        chosen = min((k for k in candidates if k not in controls), key=lambda k:(cost(k),k))
        assert math.isfinite(cost(chosen))
        controls.append(chosen)
        matches.append(dict(target=target, control=chosen, cost=cost(chosen),
            target_support_size=len(graph['supports'][tu]),
            control_support_size=len(graph['supports'][chosen[1]]),
            target_fact_count=fact_counts[tu[0]], control_fact_count=fact_counts[chosen[1][0]]))
    assert Counter(u[1] for _,u in cut) == Counter(u[1] for _,u in controls)
    frames, graphs = [], []
    for name, removed in [('baseline', []), ('bridge_cut', cut), ('matched_cut', controls)]:
        rows = [row for row in frame['rows'] if ((row[0],row[3]),(row[1],row[2])) not in removed]
        changed = dict(frame, rows=rows)
        g = build(changed)
        graphs.append(g)
        frames.append(dict(arm=name, removed=removed, R=len(g['largest'])/g['archive_count'],
            R_numerator=len(g['largest']), archive_count=g['archive_count'],
            pair_edges=len(g['witnesses']), edges_removed=len(graph['witnesses'])-len(g['witnesses']),
            components_meeting_group=[c for c in g['components'] if group.intersection(c)]))
        if name == 'bridge_cut':
            assert any(set(c) == group for c in g['components'])
            before_internal = {e for e in graph['witnesses'] if set(e) <= group}
            after_internal = {e for e in g['witnesses'] if set(e) <= group}
            assert before_internal == after_internal
    proof = ROOT/'lean/CGBAxioms/reviews/NativeRelationModeUseActual2401.lean'
    export_lean(proof, graphs)
    result = subprocess.run(['lake','env','lean',str(proof.relative_to(ROOT/'lean/CGBAxioms'))],
        cwd=ROOT/'lean/CGBAxioms',capture_output=True,text=True,encoding='utf8',errors='replace')
    (args.work/'measurement_mapping.lean.log').write_text(result.stdout+result.stderr,encoding='utf8')
    if result.returncode:
        raise RuntimeError('actual intervention graph certificate failed')
    result = dict(status='mapped_and_lean_checked', main_cycle=2401,
        identified_group=sorted(group), arms=frames, matches=matches,
        matching='equal reference count and port histogram; support size, fact count, graph degree and age matched before outcomes',
        intervention='once-only withdrawal of actual SQL uses; all native recovery/rebinding remains enabled',
        pending_policy='saved pending scopes and their old evidence are not modified',
        source_report=dict(path=str(measured_path),sha256=sha(measured_path)),
        proof=dict(path=str(proof),sha256=sha(proof),exit_code=0),
        checkpoint=dict(path=str(args.task/'body/last.pt.gz'),sha256=sha(args.task/'body/last.pt.gz')),
        archive=dict(path=str(main_db),sha256=sha(main_db)),
        external_events_per_arm=args.cycles, maximum_native_polls_per_arm=512,
        training='full inherited optimizer, autonomy, navigation, actual text feedback, settlement and persistent revision',
        main_body_will_be_changed=False)
    write(args.work/'plan.json',result)
    return result


def prepare_private(args, name):
    directory = args.work/name
    directory.mkdir(exist_ok=True)
    body = directory/'body'
    body.mkdir(exist_ok=True)
    # Read-only use of the immutable snapshot. No checkpoint writer is called
    # by this worker; the shared inode is never opened for writing.
    anchor = args.work/'anchor.pt.gz'
    if not anchor.exists():
        shutil.copy2(args.task/'body/last.pt.gz',anchor)
        assert sha(anchor) == sha(args.task/'body/last.pt.gz')
    if not (body/'last.pt.gz').exists():
        os.link(anchor,body/'last.pt.gz')
    for filename in ('last.metadata.json','config.json'):
        shutil.copy2(args.task/'body'/filename,body/filename)
    return directory, body


def worker(args):
    import numpy as np
    import torch
    from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
    from scripts.ib_local.autonomous_cycle import dispatch
    from scripts.ib_local.event_journal import rng_digest
    from scripts.audit_spectrum_runtime import carried, digest
    from scripts.native_topology_observer import NativeTopologyObserver
    from scripts.measure_native_relation_percolation import build
    specification = json.loads((args.work/'plan.json').read_text(encoding='utf8'))
    if specification.get('layer_streamed_observation'):
        from scripts.native_observation_memory import install as install_observation_memory
        install_observation_memory(release_workspace=specification.get('release_observer_workspace', False))
    directory, body = args.work/args.arm, args.work/args.arm/'body'
    storage = (Path(specification['private_storage_root'])/args.arm
               if specification.get('private_storage_root') else directory)
    storage.mkdir(parents=True, exist_ok=True)
    status_path = directory/'status.json'
    started = time.perf_counter()
    write(status_path,dict(status='loading',pid=os.getpid(),arm=args.arm))
    opts = continuation_parser().parse_args(['--checkpoint',str(body/'last.pt.gz'),
        '--output',str(body),'--resume-output','--cycles',str(args.cycles),
        '--world-source','text','--autonomous-events','--optimizer-policy','budgeted',
        '--table-storage',str(storage/'table'),'--causal-storage',str(storage/'causal')])
    torch.set_num_threads(1)
    loaded = prepare_continuation(opts,resume_output=True)
    e = loaded.engine
    if specification.get('release_policy_workspace', False):
        import gc
        original_policy_grad = e._policy_grad
        def policy_grad(*a, **kw):
            gc.collect()
            torch.cuda.empty_cache()
            if specification.get('disk_policy_activations', False):
                from scripts.native_policy_activation_storage import policy_storage
                with policy_storage(e.params, storage/'activation_workspace') as usage:
                    result = original_policy_grad(*a, **kw)
                observer.record('policy_activation_storage', usage)
                return result
            return original_policy_grad(*a, **kw)
        e._policy_grad = policy_grad
    torch.cuda.empty_cache()
    loaded.life_card = None
    assert loaded.cycle == specification['main_cycle']
    before = carried(e)
    archive = e.model._causal_archive
    rows = archive.db.execute('SELECT relation_id,unknown_id,port,generation,seen_version '
                              'FROM relation_variables ORDER BY relation_id,unknown_id').fetchall()
    before['registered_uses'] = digest(rows)
    before['fast_table'] = digest({k:np.asarray(getattr(e.table,k)) for k in
                                  ('rows','count','tag','err','epoch','last_write')})
    write(directory/'before.json',before)
    rule = next(a for a in specification['arms'] if a['arm'] == args.arm)
    e.model.identity_preview_revision = specification.get('identity_preview_revision', 0)
    affected = []
    with archive.db:
        for (rid,gen),(uid,port) in rule['removed']:
            count = archive.db.execute('DELETE FROM relation_variables WHERE relation_id=? '
                'AND generation=? AND unknown_id=? AND port=?',(rid,gen,uid,port)).rowcount
            assert count == 1
            affected.append((rid,gen,uid,port))
    original_pending = digest(e.internal_pending)
    assert original_pending == before['pending']
    initial_rows = archive.db.execute('SELECT relation_id,unknown_id,port,generation,seen_version '
                                     'FROM relation_variables ORDER BY relation_id,unknown_id').fetchall()
    removed_keys = {((rid,gen),(uid,port)) for rid,gen,uid,port in affected}
    assert initial_rows == [r for r in rows if ((r[0],r[3]),(r[1],r[2])) not in removed_keys]
    graph = build(dict(rows=initial_rows,receipt=dict(causal_archive=dict(relations=283))))
    assert len(graph['largest']) == rule['R_numerator']
    observer = NativeTopologyObserver(e,directory/'observations',raw_anchor_limit=0)
    reads, uses = [], Counter()
    # The already-admitted observer records the original loaded-state signal.
    # This added recorder reads no tensor, invokes no model, draws no RNG and
    # delegates every operation unchanged.
    original_record = observer.record
    def record(kind,data):
        if kind == 'relation_signal' and data['operation'] == 'relation_variables':
            state = data['loaded_state']
            rid,gen = state['identity'],state['generation']
            uses[str((rid,gen))] += 1
            reads.append(dict(event=observer.clock,external_cycle=observer.external_cycle,
                update_clock=observer.update_clock,relation=[rid,gen],
                unknowns=[dict(identity=v.identity,port=v.port.value,version=version,seen=seen)
                          for v,version,seen in state['variable_uses']],
                facts=[x.fact_id for x in state['constraints']]))
        return original_record(kind,data)
    observer.record = record
    fact_order_interventions, undo_order = [], lambda: None
    if rule.get('fact_order_only'):
        from scripts.native_fact_order_control import install
        fact_order_interventions, undo_order = install(observer, rule['fact_order_only'])
    pending_interventions, undo_pending = [], lambda: None
    if args.arm in ('pending_mode_cut','pending_matched'):
        from scripts.native_pending_mode_intervention import install
        pending_spec = json.loads((args.work/'pending_input_lift.json').read_text(encoding='utf8'))
        pending_interventions, undo_pending = install(observer,args.arm,pending_spec)
    inner_loop_interventions, undo_inner_loop = [], lambda: None
    if 'inner_loop' in rule:
        from scripts.native_inner_loop_intervention import install
        inner_loop_interventions, undo_inner_loop = install(observer, **rule['inner_loop'])
    transport_interventions, undo_transport = [], lambda: None
    if 'transport_curl' in rule:
        from scripts.native_transport_curl_intervention import install
        transport_interventions, undo_transport = install(observer, rule['transport_curl'], specification['transport_lift'])
    receipts, rng_checks, cycle = [], [], loaded.cycle
    update_norms = []
    original_step = e.optimizer.step
    def optimizer_step(*a,**kw):
        result = original_step(*a,**kw)
        # Current complete learner's actual gradient fields, not a diagnostic update.
        gradients = [p.grad.detach() for p in e.params if p.grad is not None]
        update_norms.append(dict(parameters_with_grad=len(gradients),
            gradient_sq_sum=float(torch.stack([g.square().sum() for g in gradients]).sum())))
        return result
    e.optimizer.step = optimizer_step
    try:
        for poll in range(512):
            current_rng = rng_digest()
            observer.begin_event(cycle)
            rng_checks.append(current_rng == rng_digest())
            row = dispatch(e)
            current_rng = rng_digest()
            observer.end_event(row)
            rng_checks.append(current_rng == rng_digest())
            receipts.append({k:v for k,v in row.items() if k != 'films'})
            if row['autonomous_event']['kind'] == 'external':
                cycle += 1
                torch.cuda.empty_cache()
                write(status_path,dict(status='running',pid=os.getpid(),arm=args.arm,
                    external_events=cycle-loaded.cycle,cycle=cycle,events=poll+1,
                    seconds=time.perf_counter()-started,actual_optimizer_calls=len(update_norms),
                    cuda_allocated_mib=torch.cuda.memory_allocated()/1048576,
                    cuda_reserved_mib=torch.cuda.memory_reserved()/1048576))
                if cycle-loaded.cycle == args.cycles:
                    break
        else:
            raise RuntimeError('native internal-event budget reached; no forced world release')
    finally:
        e.optimizer.step = original_step
        undo_pending()
        undo_order()
        undo_transport()
        undo_inner_loop()
        observer.close()
    assert all(rng_checks)
    archive.sync()
    after = carried(e)
    final_rows = archive.db.execute('SELECT relation_id,unknown_id,port,generation,seen_version '
                                   'FROM relation_variables ORDER BY relation_id,unknown_id').fetchall()
    target = set(map(tuple,specification['identified_group']))
    restored = [r for r in affected if archive.db.execute('SELECT 1 FROM relation_variables '
        'WHERE relation_id=? AND generation=? AND unknown_id=? AND port=?',r).fetchone()]
    final_archive = archive.db.execute('SELECT COUNT(*) FROM relations').fetchone()[0]
    final_graph = build(dict(rows=final_rows,receipt=dict(causal_archive=dict(relations=final_archive))))
    final = dict(status='completed',arm=args.arm,start_cycle=loaded.cycle,private_final_cycle=cycle,
        seconds=time.perf_counter()-started,actual_optimizer_calls=len(update_norms),
        update_participation=update_norms,before=before,after=after,
        receipts=receipts,read_calls=reads,read_counts=dict(uses),
        identified_group_read_calls=sum(tuple(r['relation']) in target for r in reads),
        affected_scope_read_calls=sum(any(r['relation']==list(a[:2]) for a in affected) for r in reads),
        withdrawn_references=affected,natively_restored_references=restored,
        pending_mode_interventions=pending_interventions,
        fact_order_interventions=fact_order_interventions,
        inner_loop_interventions=inner_loop_interventions,
        transport_curl_interventions=transport_interventions,
        final_R=len(final_graph['largest'])/final_archive,
        final_group_components=[g for g in final_graph['components'] if target.intersection(g)],
        observer_rng_equal=all(rng_checks),provider_calls=sum(r.get('world',{}).get('provider_calls',0) for r in receipts),
        main_body_changed=False,source_sha=loaded.source_sha)
    write(directory/'result.json',final)
    write(status_path,{k:final[k] for k in ('status','arm','private_final_cycle','seconds','actual_optimizer_calls',
        'identified_group_read_calls','affected_scope_read_calls','observer_rng_equal','provider_calls')})
    archive.close()


def run(args):
    args.task, args.work = args.task.resolve(), args.work.resolve()
    args.work.mkdir(parents=True,exist_ok=args.resume)
    if args.resume:
        specification = json.loads((args.work/'plan.json').read_text(encoding='utf8'))
        assert specification['checkpoint']['sha256'] == sha(args.task/'body/last.pt.gz')
        assert specification['external_events_per_arm'] == args.cycles
    else:
        write(args.work/'status.json',dict(status='preparing_measurement_mapping',pid=os.getpid()))
        specification = plan(args)
    environment = os.environ.copy()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        environment[key] = '1'
    environment['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:native,garbage_collection_threshold:0.4'
    before_hash = specification['checkpoint']['sha256']
    collected = {}
    for arm in ('baseline','bridge_cut','matched_cut'):
        prior = args.work/arm/'result.json'
        if prior.exists():
            collected[arm] = json.loads(prior.read_text(encoding='utf8'))
            assert collected[arm]['status'] == 'completed'
            continue
        directory,body = prepare_private(args,arm)
        command = [sys.executable,'-u','-m','scripts.run_native_relation_mode_use',
            '--worker','--task',str(args.task),'--work',str(args.work),'--arm',arm,'--cycles',str(args.cycles)]
        with (directory/'stdout.log').open('w',encoding='utf8') as out, (directory/'stderr.log').open('w',encoding='utf8') as err:
            child = subprocess.Popen(command,cwd=ROOT,env=environment,stdout=out,stderr=err,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            write(args.work/'status.json',dict(status='running',arm=arm,worker_pid=child.pid,
                completed_arms=list(collected),external_events_per_arm=args.cycles))
            code = child.wait()
        if code:
            write(args.work/'status.json',dict(status='failed',arm=arm,exit_code=code,
                stderr=str(directory/'stderr.log'),completed_arms=list(collected)))
            raise RuntimeError(f'{arm} exited {code}; preserve branch evidence')
        collected[arm] = json.loads((directory/'result.json').read_text(encoding='utf8'))
        baseline = collected['baseline']['before']
        assert collected[arm]['before'] == baseline, 'complete causal starts must match'
    assert sha(args.task/'body/last.pt.gz') == before_hash
    assert sha(specification['archive']['path']) == specification['archive']['sha256']
    summary = dict(status='completed',work=str(args.work),plan=specification,
        complete_starts_equal=True,main_checkpoint_and_archive_unchanged=True,
        arms={name:{k:r[k] for k in ('seconds','actual_optimizer_calls','private_final_cycle',
            'identified_group_read_calls','affected_scope_read_calls','final_R','observer_rng_equal','provider_calls')}
            for name,r in collected.items()},
        result_paths={name:str(args.work/name/'result.json') for name in collected})
    write(args.work/'summary.json',summary)
    write(args.work/'status.json',dict(status='completed',summary=str(args.work/'summary.json')))
    print(json.dumps(summary,ensure_ascii=False))


def run_pending(args):
    """Use the previously measured old pending mode after the structural arm.

    Baseline's first N recorded events are the exact normal baseline prefix;
    no endpoint-state comparison to its longer eight-event endpoint is made.
    """
    args.task,args.work = args.task.resolve(),args.work.resolve()
    from scripts.native_pending_mode_intervention import prepare
    specification = json.loads((args.work/'plan.json').read_text(encoding='utf8'))
    assert sha(args.task/'body/last.pt.gz') == specification['checkpoint']['sha256']
    baseline = json.loads((args.work/'baseline/result.json').read_text(encoding='utf8'))
    assert len([r for r in baseline['receipts'] if r['autonomous_event']['kind']=='external']) >= args.cycles
    mode_path = args.work/'pending_input_lift.json'
    if not mode_path.exists():
        prepare(mode_path)
    for name in ('pending_mode_cut','pending_matched'):
        if not any(a['arm']==name for a in specification['arms']):
            specification['arms'].append(dict(specification['arms'][0],arm=name))
    write(args.work/'plan.json',specification)
    environment=os.environ.copy()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        environment[key]='1'
    environment['PYTORCH_CUDA_ALLOC_CONF']='backend:native,garbage_collection_threshold:0.4'
    collected={}
    for name in ('pending_mode_cut','pending_matched'):
        result_path=args.work/name/'result.json'
        if not result_path.exists():
            directory,body=prepare_private(args,name)
            command=[sys.executable,'-u','-m','scripts.run_native_relation_mode_use','--worker',
                '--task',str(args.task),'--work',str(args.work),'--arm',name,'--cycles',str(args.cycles)]
            write(args.work/'pending_status.json',dict(status='running',arm=name,external_events_per_arm=args.cycles))
            with (directory/'stdout.log').open('w',encoding='utf8') as out,(directory/'stderr.log').open('w',encoding='utf8') as err:
                result=subprocess.run(command,cwd=ROOT,env=environment,stdout=out,stderr=err,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if result.returncode:
                raise RuntimeError(f'{name} exited {result.returncode}')
        current=json.loads(result_path.read_text(encoding='utf8'))
        assert current['status']=='completed' and current['before']==baseline['before']
        assert current['private_final_cycle']-current['start_cycle']==args.cycles
        assert len(current['pending_mode_interventions'])==2
        collected[name]=str(result_path)
    assert sha(args.task/'body/last.pt.gz')==specification['checkpoint']['sha256']
    assert sha(specification['archive']['path'])==specification['archive']['sha256']
    write(args.work/'pending_summary.json',dict(status='completed',complete_starts_equal=True,
        external_events_per_arm=args.cycles,original_baseline_prefix=True,
        main_checkpoint_and_archive_unchanged=True,result_paths=collected))
    write(args.work/'pending_status.json',dict(status='completed',summary=str(args.work/'pending_summary.json')))


def run_order_control(args):
    """No-reference-withdrawal control for the completed largest-group test."""
    specification=json.loads((args.work/'plan.json').read_text(encoding='utf8'))
    baseline=json.loads((args.work/'baseline/result.json').read_text(encoding='utf8'))
    cut=json.loads((args.work/'bridge_cut/result.json').read_text(encoding='utf8'))
    original_order=list(dict.fromkeys(baseline['read_calls'][0]['facts']))
    target_order=list(dict.fromkeys(cut['read_calls'][0]['facts']))
    assert set(original_order)==set(target_order) and original_order!=target_order
    assert args.cycles==specification['external_events_per_arm']
    name='order_only'
    if not any(a['arm']==name for a in specification['arms']):
        specification['arms'].append(dict(specification['arms'][0],arm=name,
            fact_order_only=dict(target_order=target_order,
                boundary='shared mechanism_evidence view, including prediction, organization and proposals')))
        write(args.work/'plan.json',specification)
    path=args.work/name/'result.json'
    if not path.exists():
        directory,_=prepare_private(args,name)
        environment=os.environ.copy()
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
            environment[key]='1'
        environment['PYTORCH_CUDA_ALLOC_CONF']='backend:native,garbage_collection_threshold:0.4'
        command=[sys.executable,'-u','-m','scripts.run_native_relation_mode_use','--worker',
            '--task',str(args.task),'--work',str(args.work),'--arm',name,'--cycles',str(args.cycles)]
        write(args.work/'order_control_status.json',dict(status='running',arm=name,cycles=args.cycles))
        with (directory/'stdout.log').open('w',encoding='utf8') as out,(directory/'stderr.log').open('w',encoding='utf8') as err:
            child=subprocess.run(command,cwd=ROOT,env=environment,stdout=out,stderr=err,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if child.returncode:
            raise RuntimeError('order-only control failed; preserve evidence')
    result=json.loads(path.read_text(encoding='utf8'))
    assert result['status']=='completed' and result['before']==baseline['before']
    assert result['private_final_cycle']-result['start_cycle']==args.cycles
    assert not result['withdrawn_references'] and result['fact_order_interventions']
    assert sha(args.task/'body/last.pt.gz')==specification['checkpoint']['sha256']
    assert sha(specification['archive']['path'])==specification['archive']['sha256']
    write(args.work/'order_control_status.json',dict(status='completed',complete_starts_equal=True,
        main_checkpoint_and_archive_unchanged=True,interventions=len(result['fact_order_interventions'])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--cycles',type=int,default=8)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--pending-only',action='store_true')
    p.add_argument('--order-control',action='store_true')
    p.add_argument('--arm',choices=('baseline','bridge_cut','matched_cut','pending_mode_cut','pending_matched','order_only'))
    args = p.parse_args()
    if args.worker:
        try:
            worker(args)
        except Exception as error:
            write(args.work/args.arm/'status.json',dict(status='failed',error=repr(error),pid=os.getpid()))
            raise
    elif args.order_control:
        run_order_control(args)
    elif args.pending_only:
        run_pending(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
