"""Audit a real medium training-window calibration, including dedicated memory.

Pass the regular train_medium_active_stream arguments after --audit-report.
This wrapper requires --calibrate-only: updates belong to a disposable birth,
never to an existing research individual, and no checkpoints are written.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from information_boltzmann.runtime import training
from information_boltzmann.core.plastic_medium import PlasticMedium3D
from scripts.ib.audit_medium_long_interval import DedicatedMemorySampler
from scripts.ib.train_medium_active_stream import main as train_main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-report', type=Path, required=True)
    parser.add_argument('--audit-backward-gc-interval', type=int, default=0)
    args, remaining = parser.parse_known_args()
    args_gc_interval = args.audit_backward_gc_interval
    if '--calibrate-only' not in remaining or '--resume' in remaining:
        parser.error('A disposable --calibrate-only birth is required')
    cap = int(remaining[remaining.index('--vram-limit-mib') + 1]) if '--vram-limit-mib' in remaining else 3900
    monitor = DedicatedMemorySampler(cap * 2**20)
    monitor.thread.start()
    original_event = training.training_event
    original_substeps = PlasticMedium3D._fused_substeps
    report = {'arguments': remaining, 'event_function_calls': 0, 'physical_step_calls': 0,
              'memory_breadcrumbs': [],
              'purpose': 'full real-data BPTT execution/resources, not capability evidence'}
    started = time.perf_counter()
    last_bucket = -1

    def measured_substeps(self, operation, *values, **kwargs):
        def measured_operation(*args, **options):
            nonlocal last_bucket
            if (torch._C._current_graph_task_id() != -1
                    and args_gc_interval and report['physical_step_calls'] % args_gc_interval == 0):
                before_gc = torch.cuda.memory_allocated() / 2**20
                gc.collect()
                after_gc = torch.cuda.memory_allocated() / 2**20
                report.setdefault('backward_gc', []).append({
                    'physical_step_calls': report['physical_step_calls'],
                    'before_mib': before_gc, 'after_mib': after_gc})
            allocated = torch.cuda.memory_allocated() / 2**20
            bucket = int(allocated / 128)
            if bucket > last_bucket:
                last_bucket = bucket
                entry = {'completed_events': report['event_function_calls'],
                         'physical_step_calls': report['physical_step_calls'],
                         'allocated_mib': allocated,
                         'seconds': time.perf_counter() - started}
                report['memory_breadcrumbs'].append(entry)
                print(f'CALIBRATION physical-call {entry["physical_step_calls"]}, '
                      f'events {entry["completed_events"]}: {allocated:.1f} MiB', flush=True)
            result = operation(*args, **options)
            report['physical_step_calls'] += 1
            return result
        return original_substeps(self, measured_operation, *values, **kwargs)

    def measured_event(*values, **kwargs):
        result = original_event(*values, **kwargs)
        report['event_function_calls'] += 1
        count = report['event_function_calls']
        if count == 1 or count % 16 == 0:
            print(f'CALIBRATION event-call {count}: allocated '
                  f'{torch.cuda.memory_allocated()/2**20:.1f} MiB, '
                  f'{time.perf_counter()-started:.1f}s elapsed', flush=True)
        monitor.check()
        return result

    training.training_event = measured_event
    PlasticMedium3D._fused_substeps = measured_substeps
    sys.argv = [sys.argv[0], *remaining]
    try:
        train_main()
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        training.training_event = original_event
        PlasticMedium3D._fused_substeps = original_substeps
        report['seconds'] = time.perf_counter() - started
        report['memory'] = monitor.finish()
        if torch.cuda.is_initialized():
            report['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
            report['peak_reserved_mib'] = torch.cuda.max_memory_reserved() / 2**20
        output = Path(remaining[remaining.index('--output') + 1])
        progress = output / 'progress.json'
        if progress.exists():
            report['final_progress'] = json.loads(progress.read_text(encoding='utf-8'))
        args.audit_report.parent.mkdir(parents=True, exist_ok=True)
        args.audit_report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
