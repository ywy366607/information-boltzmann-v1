"""Resume the frozen whole-body comparison from committed event boundaries.

This controller changes no experiment runtime. Each uninterrupted segment
keeps its original checkpoint and log; summaries join consecutive events.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)


def read_rows(directory):
    path=directory/'metrics.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def checkpoint_cycle(path):
    # A short-lived CPU reader avoids retaining Torch/table memory beside the
    # actual GPU learner. Only a fully committed checkpoint can be resumed.
    code=('import gzip,sys,torch; '
          'f=gzip.open(sys.argv[1],"rb"); '
          's=torch.load(f,map_location="cpu",weights_only=False); '
          'print(int(s["cycle"]))')
    return int(subprocess.check_output([sys.executable,'-c',code,str(path)],text=True).strip())


def check_runtime(manifest):
    actual={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path('scripts/ib_local').glob('*.py'))}
    if actual!=manifest:
        raise RuntimeError('experiment runtime differs from the frozen manifest')


def segments(root,arm):
    base=root/arm
    return ([base] if base.exists() else [])+sorted(root.glob(arm+'_resume_[0-9][0-9]'))


def validate_rows(rows,origin):
    if any(row['cycle']!=origin+i+1 for i,row in enumerate(rows)):
        raise RuntimeError('committed metrics are not a consecutive life history')


def run(root):
    from scripts.ib_local.run_causal_program_pair import summary
    status=json.loads((root/'status.json').read_text(encoding='utf-8'))
    manifest=json.loads((root/'runtime_manifest.json').read_text(encoding='utf-8'))
    check_runtime(manifest)
    source=Path(status['source']); origin=checkpoint_cycle(source)
    budget=int(status['cycles_per_arm'])
    status.pop('error',None)
    status.pop('failed_at',None)
    status.update(status='running',controller_pid=os.getpid(),resumed_at=time.time(),
                  continuity='only exact committed checkpoints; all old segments retained')
    env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    for arm in ('joint_program','inherited'):
        paths=segments(root,arm)
        rows=[row for path in paths for row in read_rows(path)]
        validate_rows(rows,origin)
        if len(rows)>budget: raise RuntimeError('arm exceeds its registered budget')
        status.update(active_arm=arm,committed_cycles=len(rows),origin_cycle=origin)
        write(root/'status.json',status)
        if len(rows)<budget:
            checkpoint=paths[-1]/'last.pt.gz' if paths else source
            actual=checkpoint_cycle(checkpoint)
            if actual!=origin+len(rows):
                raise RuntimeError('checkpoint/metric event mismatch; refuse to skip or repeat experience')
            if (root/'STOP').exists():
                status['status']='stopped';write(root/'status.json',status);return
            output=root/(arm+f'_resume_{len(paths):02d}') if paths else root/arm
            extra=[] if paths or arm=='inherited' else ['--causal-program']
            cmd=[sys.executable,'-m','scripts.ib_local.run_mother_continuation',
                 '--checkpoint',str(checkpoint),'--output',str(output),
                 '--cycles',str(budget-len(rows)),
                 '--shared-cache',str(root/'mother_cache.json'),
                 '--stop-file',str(root/'STOP'),*extra]
            status.update(active_segment=str(output),resume_checkpoint=str(checkpoint),
                          remaining_cycles=budget-len(rows))
            write(root/'status.json',status)
            with (root/(output.name+'.log')).open('w',encoding='utf-8') as log:
                result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{arm} exited with code {result.returncode}; committed state retained')
            paths.append(output);rows.extend(read_rows(output));validate_rows(rows,origin)
        check_runtime(manifest)
        combined=root/(arm+'_combined');combined.mkdir(exist_ok=True)
        (combined/'metrics.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        status['arms'][arm]=dict(segments=[str(p) for p in paths],
            final_checkpoint=str(paths[-1]/'last.pt.gz'),cycles=len(rows),summary=summary(combined))
        write(root/'status.json',status)
        if len(rows)<budget:
            status['status']='stopped';write(root/'status.json',status);return
    status.update(status='complete',finished_at=time.time())
    for key in ('active_arm','active_segment','remaining_cycles'):
        status.pop(key,None)
    write(root/'status.json',status);write(root/'comparison.json',status)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args.output)
    except Exception as error:
        path=args.output/'status.json'
        status=json.loads(path.read_text(encoding='utf-8'))
        status.update(status='failed',error=f'{type(error).__name__}: {error}',failed_at=time.time())
        write(path,status)
        raise


if __name__=='__main__':main()
