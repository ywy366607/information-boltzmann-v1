"""Serial real admission, full restore, then continuous whole-body learning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cycles',type=int,default=512)
    args=p.parse_args()
    if args.cycles<3:raise ValueError('whole study includes two committed admission events')
    root=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=False)
    files=['scripts/ib_local/'+name+'.py' for name in ('generative_relations',
        'navigation_generative_cycle','causal_relation_program','navigation_causal_program',
        'navigation_joint_cycle','causal_archive','run_causal_water_reply',
        'run_mother_continuation','admit_generative_cycle','audit_joint_restore',
        'navigation_checkpoint','mother_checkpoint','world_learning_cycle')]
    files.append('scripts/run_generative_joint_study.py')
    fingerprints={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
    status=dict(status='running',pid=os.getpid(),planned_cycles=args.cycles,
                checkpoint=str(args.checkpoint.resolve()),source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                code_sha256=fingerprints,started_at=time.time())
    def write():
        temporary=args.output/'pipeline.json.tmp'
        temporary.write_text(json.dumps(status,indent=2),encoding='utf-8')
        temporary.replace(args.output/'pipeline.json')
    write()
    (args.output/'preregistration.json').write_text(json.dumps(dict(
        source_checkpoint=status['checkpoint'],source_sha256=status['source_sha256'],
        total_real_events=args.cycles,admission_events=2,learning_events=args.cycles-2,
        whole_body='inherit all parameters, optimizer, S/P, facts, world and pending credit',
        generators=['bind','execute','revise'],
        grammar=[['mother_response','ambient_transport','existing_continuation'],
                 ['body_transition','existing_continuation']],
        open_roles=['mother','action','condition'],
        observation='same actual mother and ambient stream; original world_stream settlement',
        numerical_exit='two real committed events and complete independent state restore',
        evaluation='prequential real predictions and later reuse by relation ID; source-separated progress; independent sidecars every 64 events',
        benefit_claim='requires matched whole-loop control and sufficient learning, not admission activity',
        stop='interface/numerical/restore failure or explicit STOP; preserve committed prefix'),indent=2),encoding='utf-8')
    phases=[('admission',['-m','scripts.ib_local.admit_generative_cycle',
        '--checkpoint',str(args.checkpoint),'--output',str(args.output/'admission')]),
        ('restore',['-m','scripts.ib_local.audit_joint_restore',
        '--checkpoint',str(args.output/'admission/last.pt.gz'),'--output',str(args.output/'restore'),
        '--table-storage',str(args.output/'restore_table')]),
        ('learning',['-m','scripts.ib_local.run_mother_continuation',
        '--checkpoint',str(args.output/'admission/last.pt.gz'),'--output',str(args.output/'learning'),
        '--cycles',str(args.cycles-2),'--validate-every','64','--validation-bytes','8192',
        '--mother-harness-url','http://127.0.0.1:8902','--stop-file',str(args.output/'STOP')])]
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    for phase, command in phases:
        if (args.output/'STOP').exists():
            status.update(status='stopped',phase=phase);write();return
        if any(hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha for name,sha in fingerprints.items()):
            status.update(status='failed',error='executing source changed');write();return
        status.update(phase=phase);write()
        with (args.output/(phase+'.log')).open('w',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-u',*command],cwd=root,env=env,
                                   stdout=log,stderr=subprocess.STDOUT)
            status.update(child_pid=child.pid);write()
            code=child.wait()
        if code:
            status.update(status='failed',exit_code=code);write();return
        if phase=='admission':
            result=json.loads((args.output/'admission/admission.json').read_text(encoding='utf-8'))
            if not result.get('event_admission_passed'):
                status.update(status='failed',error='real event admission incomplete');write();return
        if phase=='restore':
            result=json.loads((args.output/'restore/restore_audit.json').read_text(encoding='utf-8'))
            if not result['passed']:
                status.update(status='failed',error='complete restore differs');write();return
    final=json.loads((args.output/'learning/completed.json').read_text(encoding='utf-8'))
    status.update(status=final['status'],final_cycle=final['final_cycle'],ended_at=time.time());write()


if __name__=='__main__':main()
