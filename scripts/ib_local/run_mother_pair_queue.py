"""Alternating bounded mother-world continuation with persistent checkpoints.

One process/GPU at a time. A failure or STOP never starts a later arm; each
child saves the actual complete individual after each committed turn. This is
a 512-turn first stage, not a claim of convergence or a full no-loop control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/cgb_mother_continuation_20261110'


def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def worker_modes(comparison, mode):
    return (['--mode','real','--internal-rollout',mode,'--allow-internal-mode-change']
            if comparison=='internal' else ['--mode',mode])


def mother_context_args(context, endpoint, output, mode):
    if context is None:
        if endpoint is not None:
            raise ValueError('a caregiver endpoint requires --mother-context dsh')
        return []
    options=['--mother-context',context]
    if context=='dsh':
        from scripts.ib_local.mengli_harness.mengli_world_client import branch_for_output
        options+=['--mother-session',branch_for_output(Path(output)/mode)]
        if endpoint is not None:
            options+=['--mother-harness-url',endpoint]
    elif endpoint is not None:
        raise ValueError('a caregiver endpoint requires --mother-context dsh')
    return options


def prune_batch_checkpoint(previous, output, segment_cycles, record, status):
    """Only transient checkpoints made by THIS queue; keep each segment end."""
    if record['cycles'] != segment_cycles or previous is None:
        return
    previous=previous.resolve(); output=output.resolve()
    if output not in previous.parents or previous.name!='last.pt.gz':
        return
    # Every segment's last batch is retained. Others have already been fully
    # inherited by a successfully saved successor, with logs left in place.
    completed=next((x for x in reversed(status['completed'][:-1])
                    if x.get('output')==str(previous.parent)),None)
    if completed is None or completed.get('segment_end'):
        return
    if previous.exists():
        status.setdefault('retired_checkpoints',[]).append(
            dict(path=str(previous),sha256=file_hash(previous),reason='complete successor saved'))
        previous.unlink()


def write(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--segments',type=int,default=7)
    p.add_argument('--segment-cycles',type=int,default=64)
    p.add_argument('--real-checkpoint',type=Path)
    p.add_argument('--blocked-checkpoint',type=Path)
    p.add_argument('--real-first-cycles',type=int)
    p.add_argument('--blocked-first-cycles',type=int)
    p.add_argument('--batch-cycles',type=int,default=8,
                   help='restart the worker from its full checkpoint at bounded intervals')
    p.add_argument('--comparison',choices=('episodic','internal'),default='episodic')
    p.add_argument('--mother-context',choices=('segmented','dsh'))
    p.add_argument('--mother-harness-url')
    p.add_argument('--shared-source',type=Path,
                   help='same full source for real/observe internal continuation comparison')
    p.add_argument('--prune-intermediate',action='store_true',
                   help='keep segment endpoints and latest, remove superseded own batch snapshots')
    args=p.parse_args()
    if args.output.exists() or args.segments<1 or args.segment_cycles<1 or args.batch_cycles<1:
        raise ValueError('new output and positive bounded stages required')
    args.output.mkdir(parents=True)
    stop=args.output/'STOP'
    shared=BASE/'mother_replies.sqlite'
    validation=BASE/'validation/mother_validation.bin'
    modes=('real','observe') if args.comparison=='internal' else ('real','blocked')
    if args.comparison=='internal' and args.shared_source is None:
        raise ValueError('internal comparison requires one explicit shared source')
    checkpoints={mode:(args.shared_source.resolve() if args.shared_source else
                      BASE/f'pair_{mode}_navigating_1815/last.pt.gz') for mode in modes}
    for mode in checkpoints:
        override=getattr(args, f'{mode}_checkpoint',None)
        if override is not None:
            checkpoints[mode]=override.resolve()
        first=getattr(args, f'{mode}_first_cycles',None)
        if first is not None and first < 0:
            raise ValueError('first segment cycles cannot be negative')
    prior=json.loads((BASE/'pair_real_navigating_1815/config.json').read_text(encoding='utf-8'))
    files=list(dict.fromkeys([Path(path) for path in prior['sources']]+[
           Path(__file__),ROOT/'scripts/ib_local/run_mother_continuation.py',ROOT/'scripts/ib_local/mother_checkpoint.py',
           ROOT/'scripts/ib_local/mother_reply_world.py',ROOT/'scripts/ib_local/evaluate_mother_continuation.py',
           ROOT/'scratch/claude_mother_world_20261105.py',ROOT/'information_boltzmann/core/fly_mother_world.py']))
    if args.comparison=='internal':
        files=list(dict.fromkeys(files+list((ROOT/'scripts/ib_local').glob('*.py'))))
    if args.mother_context=='dsh':
        harness=ROOT/'scripts/ib_local/mengli_harness'
        files=list(dict.fromkeys(files+list(harness.glob('*.py'))+list(harness.glob('*.mjs'))))
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    status=dict(status='running',stage='starting',start_cycle='see checkpoint metadata',segments=args.segments,
        segment_cycles=args.segment_cycles, completed=[],checkpoints={k:str(v) for k,v in checkpoints.items()},
        source_hashes=hashes,shared_cache=str(shared),stop_file=str(stop),
        goal='continuous mother-world learning and paired experience-to-choice control',
        comparison=args.comparison,initial_source_hashes={k:file_hash(v) for k,v in checkpoints.items()},
        caveat='single seed; no convergence or complete no-loop claim')
    write(args.output/'status.json',status)
    try:
        for seg in range(1,args.segments+1):
            for mode in modes:
                if stop.exists():
                    status.update(status='stopped',stage='between_segments')
                    write(args.output/'status.json',status); return
                if any(hashlib.sha256(path.read_bytes()).hexdigest()!=hashes[str(path)] for path in files):
                    raise RuntimeError('running source changed; no automatic continuation')
                first=getattr(args,f'{mode}_first_cycles',None)
                remaining=first if seg == 1 and first is not None else args.segment_cycles
                batch=0
                while remaining:
                    batch+=1
                    count=min(args.batch_cycles,remaining)
                    out=args.output/f'{mode}_{seg:02d}_{batch:02d}'
                    status.update(stage='training',active_mode=mode,active_segment=seg,active_batch=batch,active_output=str(out))
                    write(args.output/'status.json',status)
                    command=[sys.executable,'-m','scripts.ib_local.run_mother_continuation',
                        '--checkpoint',str(checkpoints[mode]),'--output',str(out),
                        '--shared-cache',str(shared),'--stop-file',str(stop),
                        '--cycles',str(count)]+worker_modes(args.comparison,mode)+mother_context_args(
                            args.mother_context,args.mother_harness_url,args.output,mode)
                    with (args.output/f'{mode}_{seg:02d}_{batch:02d}.log').open('w',encoding='utf-8') as log:
                        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise RuntimeError(f'{mode} segment {seg} batch {batch} exited {result.returncode}; see its log')
                    record=json.loads((out/'completed.json').read_text(encoding='utf-8'))
                    previous=checkpoints[mode]
                    checkpoints[mode]=out/'last.pt.gz'
                    status['checkpoints'][mode]=str(checkpoints[mode])
                    remaining-=record['cycles']
                    status['completed'].append(dict(mode=mode,segment=seg,batch=batch,
                        output=str(out.resolve()),segment_end=(remaining==0),**record))
                    if args.prune_intermediate:
                        prune_batch_checkpoint(previous,args.output,count,record,status)
                    if record['status']!='complete':
                        status.update(status='stopped',stage='event_boundary')
                        write(args.output/'status.json',status); return
                    write(args.output/'status.json',status)
                # After both sides have advanced, evaluate their fixed OWT and
                # earlier mother-language sidecars sequentially.
                write(args.output/'status.json',status)
            for mode in modes:
                if stop.exists():
                    status.update(status='stopped',stage='before_validation')
                    write(args.output/'status.json',status); return
                dest=args.output/f'eval_{mode}_{seg:02d}.json'
                status.update(stage='validation',active_mode=mode,active_segment=seg,active_output=str(dest))
                write(args.output/'status.json',status)
                command=[sys.executable,'-m','scripts.ib_local.evaluate_mother_continuation',
                         '--checkpoint',str(checkpoints[mode]),'--mother-validation',str(validation),
                         '--output',str(dest)]
                with (args.output/f'eval_{mode}_{seg:02d}.log').open('w',encoding='utf-8') as log:
                    result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'{mode} validation {seg} exited {result.returncode}; see its log')
            if args.comparison=='internal':
                result=subprocess.run([sys.executable,'-m','scripts.ib_local.analyze_internal_rollout',
                    '--root',str(args.output)],cwd=ROOT)
                if result.returncode:
                    raise RuntimeError('internal continuation analysis failed')
        status.update(status='complete',stage='first_stage_complete')
    except Exception as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}')
        write(args.output/'status.json',status)
        raise
    write(args.output/'status.json',status)


if __name__=='__main__':
    main()
