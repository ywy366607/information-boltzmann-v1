"""Matched actual persistent-evolution execution benchmark, no capability claim.

CUDA uses a small allocator budget and a total dedicated-memory guard.
This script does not stop other jobs or launch training. Model state is identical
across reference, cached and captured paths; warm-up is excluded from timing.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from information_boltzmann.core.plastic_medium import MediumState
from information_boltzmann.core.plastic_ports import PlasticBelief, PlasticMediumPorts3D
from information_boltzmann.runtime import ContinuousStream
from information_boltzmann.runtime.gpu_memory import windows_gpu_memory, total_gpu_memory


class ReferenceStream(ContinuousStream):
    def _coefficients(self):
        self.coefficient_preparations += 1
        return None

    def __init__(self, model, **options):
        super().__init__(model, **options)
        self._kernel = lambda state, duration, _: model.medium.advance(state, duration)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--shape', type=int, nargs=3, default=(8, 8, 4))
    parser.add_argument('--channels', type=int, default=128)
    parser.add_argument('--bath-type', choices=('quadratic', 'conductance'), default='quadratic')
    parser.add_argument('--step-duration', type=float, required=True)
    parser.add_argument('--ticks', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--allocator-fraction', type=float, default=0.12)
    parser.add_argument('--allow-driver-shared-baseline', action='store_true',
                        help='Legacy option; shared usage is recorded rather than rejected by default')
    parser.add_argument('--require-zero-shared-growth', action='store_true',
                        help='Reproduce the historical strict shared-memory policy')
    parser.add_argument('--vram-limit-gib', type=float, default=4.0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.ticks, args.warmup, args.threads) < 1:
        parser.error('Positive ticks, warmup and threads required')
    if not math.isfinite(args.vram_limit_gib) or args.vram_limit_gib <= 0:
        parser.error('Positive finite dedicated VRAM limit required')
    torch.set_num_threads(args.threads)
    torch.manual_seed(449)
    report = {'purpose': 'actual dynamics execution only; no language/capability benchmark',
              'shape': args.shape, 'channels': args.channels, 'device': args.device,
              'bath_type': args.bath_type,
              'ticks': args.ticks, 'warmup': args.warmup, 'threads': args.threads,
              'step_duration': args.step_duration, 'pytorch': torch.__version__,
              'allocator_fraction': args.allocator_fraction,
              'allow_driver_shared_baseline': args.allow_driver_shared_baseline,
              'require_zero_shared_growth': args.require_zero_shared_growth,
              'vram_limit_bytes': int(args.vram_limit_gib * 2**30)}

    def publish():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2))

    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.allocator_fraction)
        torch.cuda.init()
        before = windows_gpu_memory()
        report['windows_gpu_memory_before'] = before
        if args.require_zero_shared_growth and (before is None or (
                before['shared_bytes'] > 0 and not args.allow_driver_shared_baseline)):
            report['status'] = 'stopped_before_benchmark_shared_memory_not_zero_or_unavailable'
            publish()
            return
    net = PlasticMediumPorts3D(vocab_size=17, shape=tuple(args.shape), channels=args.channels,
                               bath_type=args.bath_type).eval().to(args.device)
    initial = net.initial_belief()
    state = replace(initial.medium, field=torch.randn_like(initial.medium.field),
                    flux=tuple(torch.randn_like(x) for x in initial.medium.flux))
    initial = PlasticBelief(state, initial.precision)
    variants = [('reference_monitored', ReferenceStream, None),
                ('cached_quiet', ContinuousStream, None)]
    if args.device == 'cuda':
        variants.append(('cached_cuda_graph', ContinuousStream, 'cuda_graph'))
        torch.cuda.reset_peak_memory_stats()
    outputs, measurements = {}, {}
    with torch.no_grad():
        for name, cls, backend in variants:
            stream = cls(net, max_step=args.step_duration, belief=initial,
                         compile_backend=backend)
            for index in range(args.warmup):
                stream.advance_to((index + 1) * args.step_duration)
            if args.device == 'cuda':
                torch.cuda.synchronize()
            preparations = stream.coefficient_preparations
            start = time.perf_counter()
            for index in range(args.ticks):
                stream.advance_to((args.warmup + index + 1) * args.step_duration)
            if args.device == 'cuda':
                torch.cuda.synchronize()
            seconds = (time.perf_counter() - start) / args.ticks
            outputs[name] = tuple(x.detach().cpu().clone() for x in (
                stream.belief.medium.field, *stream.belief.medium.flux,
                stream.belief.medium.conduction, stream.belief.medium.receptors) if x is not None)
            measurements[name] = {'seconds_per_tick': seconds,
                                  'coefficient_preparations_during_measurement':
                                      stream.coefficient_preparations - preparations,
                                  'physical_time_per_wall_second': args.step_duration / seconds}
            if args.device == 'cuda':
                memory = windows_gpu_memory()
                measurements[name]['windows_gpu_memory'] = memory
                total = total_gpu_memory()
                measurements[name]['total_gpu_memory'] = total
                growth = None if memory is None or before is None else memory['shared_bytes'] - before['shared_bytes']
                measurements[name]['shared_growth_bytes'] = growth
                vram_limit_reached = total['used_bytes'] >= min(total['total_bytes'], report['vram_limit_bytes'])
                if vram_limit_reached or (args.require_zero_shared_growth and (growth is None or growth > 0)):
                    report['measurements'] = measurements
                    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                    report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                    report['status'] = ('stopped_dedicated_vram_limit' if vram_limit_reached
                                        else 'stopped_shared_memory_growth')
                    publish()
                    return
            # Release graph pool before the next independent variant.
            del stream
    reference = outputs['reference_monitored']
    for name, values in outputs.items():
        error = max(float((x - y).abs().max()) for x, y in zip(values, reference))
        measurements[name]['maximum_state_difference'] = error
        measurements[name]['speedup_over_reference'] = (
            measurements['reference_monitored']['seconds_per_tick'] / measurements[name]['seconds_per_tick'])
        if error > 2e-5:
            raise RuntimeError(f'{name} changes persistent evolution: {error}')
    report['measurements'] = measurements
    report['status'] = 'completed'
    if args.device == 'cuda':
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        report['windows_gpu_memory_after'] = windows_gpu_memory()
        after = report['windows_gpu_memory_after']
        report['shared_growth_bytes'] = None if after is None or before is None else after['shared_bytes'] - before['shared_bytes']
        if args.require_zero_shared_growth and (after is None or report['shared_growth_bytes'] > 0):
            report['status'] = 'shared_memory_guard_failed_after_measurement'
    publish()


if __name__ == '__main__':
    main()
