"""Run the registered H14 and H0 continuous individuals sequentially."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results')
    args = parser.parse_args()
    common = ['--backend','exact','--horizon','14','--learn-stp',
              '--window','32','--lr','0.0002','--lr-decoder','0.00002',
              '--lr-sensory','0.0002','--lr-synapse','0.0002',
              '--additional-tokens','100000','--vram-limit-mib','3900']
    status_path = args.results/'published/fly_causal_rtc_joint_execution_20261007.json'
    status = dict(status='running', started=time.time(), arms=[],
                  execution='sequential GPU jobs; calibrated full-life resume')

    def save():
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
        temporary.replace(status_path)

    for horizon in (14,0):
        output = args.results/f'fly_causal_rtc_h{horizon}_joint_100k'
        calibration = json.loads((output/'calibration.json').read_text(encoding='utf-8'))
        parity = json.loads((output/'cuda_checkpoint_parity.json').read_text(encoding='utf-8'))
        memory = calibration['windows_memory']
        if (max(calibration['cuda_peak_reserved_mib'], calibration['cuda_peak_allocated_mib']) > 3900
                or memory is None or memory['dedicated_bytes']/2**20 > 3900
                or not math.isfinite(calibration['pre_update_nll'])
                or not parity['actual_pulses_match'] or parity['forward_max_abs_error'] > 2e-5
                or not (output/'last.pt').exists()):
            raise RuntimeError(f'H{horizon} resource gate/checkpoint missing')
        for gradient in parity['gradients'].values():
            if (not math.isfinite(gradient['difference_norm'])
                    or gradient['difference_norm'] > 1e-6+2e-3*gradient['norm']):
                raise RuntimeError(f'H{horizon} checkpoint gradient gate failed')
        record = dict(horizon=horizon,status='running',output=str(output),started=time.time())
        status['arms'].append(record)
        save()
        command = [sys.executable,'-X','utf8',str(ROOT/'scripts/ib/train_fly_rtc_dagger.py'),
                   *common,'--read-horizon',str(horizon),'--output',str(output),
                   '--resume',str(output/'last.pt')]
        with (output/'training.log').open('a',encoding='utf-8') as log:
            process = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            record['pid'] = process.pid
            save()
            code = process.wait()
        progress = json.loads((output/'progress.json').read_text(encoding='utf-8'))
        record.update(exit_code=code,status=progress['status'],ended=time.time())
        if code or progress['status'] != 'completed':
            status['status'] = 'stopped'
            save()
            return code or 1
        save()
    status.update(status='completed',ended=time.time())
    save()
    return 0


if __name__ == '__main__':
    sys.exit(main())
