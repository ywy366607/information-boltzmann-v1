"""Serialize complete-body optimizer comparisons on one small GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--body', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase-events', type=int, default=32)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    from scripts.run_guarded_mother_continuation import resources, stop_reason
    paths = sorted([*(root/'scripts/ib_local').glob('*.py'),
                    *(root/'information_boltzmann').glob('*.py'),Path(__file__)])
    def code_digest():
        h = hashlib.sha256()
        for path in paths:
            h.update(path.relative_to(root).as_posix().encode()); h.update(path.read_bytes())
        return h.hexdigest()
    code = code_digest()
    with zipfile.ZipFile(a.output/'execution_sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in paths: archive.write(path,path.relative_to(root).as_posix())
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_MODULE_LOADING='LAZY')
    status = dict(status='running', body=str(a.body), phase_events=a.phase_events,
                  started_at=time.time(), main_body_written=False, completed_arms=[], code_sha=code,
                  source_control='actual availability, not only cursor')
    def publish():
        temporary = a.output/'status.tmp.json'
        temporary.write_text(json.dumps(status,indent=2)+'\n')
        temporary.replace(a.output/'status.json')
    for arm in ('budgeted','adamw','global_anneal'):
        if (a.output/'STOP').exists():
            status.update(status='stopped_between_arms'); publish(); return
        while True:
            reading = resources(a.output)
            status.update(status='waiting_resources', resources=reading); publish()
            if (reading['available_commit_mib'] >= 3328 and reading['system_disk_free_mib'] >= 512
                    and reading['output_disk_free_mib'] >= 1024): break
            if (a.output/'STOP').exists():
                status.update(status='stopped_before_arm'); publish(); return
            time.sleep(10)
        if code_digest() != code:
            status.update(status='source_changed_requires_review'); publish(); return
        command = [sys.executable,'-u','-m','scripts.ib_local.audit_optimizer_complete_loop',
            '--body',str(a.body),'--output',str(a.output),'--arm',arm,
            '--phase-events',str(a.phase_events),'--stop-file',str(a.output/'STOP_EVENT')]
        status.update(status='running', arm=arm, command=command); publish()
        with (a.output/f'{arm}.log').open('w') as log:
            child = subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            status['child_pid'] = child.pid; publish()
            while child.poll() is None:
                reading = resources(a.output); status['resources'] = reading
                reason = stop_reason(reading)
                if reason is not None:
                    (a.output/'STOP_EVENT').write_text(reason)
                    status['resource_stop_reason'] = reason
                publish(); time.sleep(2)
            exit_code = child.returncode
        if exit_code:
            status.update(status='failed',exit_code=exit_code); publish(); return
        status['completed_arms'].append(arm); publish()
    reports = {arm:json.loads((a.output/arm/'report.json').read_text()) for arm in status['completed_arms']}
    summary = dict(schema=1, scope='real complete-body optimizer/relearning comparison; no dialogue capability verdict',
        source_sha=next(iter(reports.values()))['source_sha'], arms=reports,
        main_body_written=False, next='compare source exposure, real prediction/action/credit and local reopening before main adoption')
    (a.output/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    status.update(status='complete',ended_at=time.time()); publish()


if __name__ == '__main__': main()
