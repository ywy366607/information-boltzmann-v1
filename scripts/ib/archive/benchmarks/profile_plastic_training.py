"""Measure real captured module forward intervals and total backward, in OWT.

Backward is reported as a whole; it is never inferred from eager proportions.
External event instrumentation adds overhead, so use the regular execution
benchmark to report production throughput.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.training import clone_belief, quiet_training_chunk
from information_boltzmann.runtime.gpu_memory import total_gpu_memory
from information_boltzmann.runtime.execution_cache import configure_execution_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replays', type=int, default=20)
    parser.add_argument('--fusion', choices=('none', 'medium', 'all'), default='none')
    parser.add_argument('--compile-cache-dir', type=Path, default=Path('scratch/compiler_cache'))
    parser.add_argument('--data', type=Path, default=Path('data/ib_owt_gpt2/train.npy'))
    args = parser.parse_args()
    configure_execution_cache(args.compile_cache_dir)
    if args.replays < 1:
        parser.error('Positive replay count required')
    torch.set_num_threads(1)
    torch.manual_seed(449)
    torch.cuda.set_per_process_memory_fraction(0.4)
    net = PlasticMediumPorts3D(bath_type='conductance', activity_adaptation=True,
                              short_term_plasticity=True,
                              medium_execution='native' if args.fusion == 'none' else 'fused',
                              port_execution='fused' if args.fusion == 'all' else 'native').cuda().train()
    tokens = torch.from_numpy(np.array(np.load(args.data, mmap_mode='r')[:9], dtype=np.int64)).cuda()[None]
    ids, targets = tokens[:, :-1], tokens[:, 1:]
    state = clone_belief(net.initial_belief())
    duration = torch.tensor(0.005, device='cuda', dtype=torch.float64)
    intervals = []
    def instrument(function, name):
        def wrapped(*inputs, **kwargs):
            if not torch.cuda.is_current_stream_capturing():
                return function(*inputs, **kwargs)
            begin = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            begin.record()
            result = function(*inputs, **kwargs)
            end.record()
            intervals.append((name, begin, end))
            return result
        return wrapped
    for owner, key, name in ((net, 'assimilate', 'write'), (net.medium, 'advance', 'medium'),
                              (net, 'read', 'read'), (net.decoder, 'forward', 'decode'),
                              (net.medium, 'prepare_evolution', 'prepare')):
        setattr(owner, key, instrument(getattr(owner, key), name))
    def operation():
        return quiet_training_chunk(net, ids, targets, state, event_duration=duration)
    current = torch.cuda.current_stream()
    warm = torch.cuda.Stream()
    warm.wait_stream(current)
    with torch.cuda.stream(warm):
        for _ in range(3):
            net.zero_grad(set_to_none=True)
            operation()[0].backward()
    current.wait_stream(warm)
    torch.cuda.synchronize()
    active = tuple(p for p in net.parameters() if p.grad is not None)
    def zero():
        for p in active:
            p.grad.zero_()
    zero()
    starts = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(3)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        starts[0].record()
        loss, output, _ = operation()
        starts[1].record()
        loss.backward()
        starts[2].record()
    rows = []
    for _ in range(args.replays):
        zero()
        graph.replay()
        torch.cuda.synchronize()
        row = {'forward_ms': starts[0].elapsed_time(starts[1]),
               'backward_ms': starts[1].elapsed_time(starts[2])}
        for name, begin, end in intervals:
            row[name + '_forward_ms'] = row.get(name + '_forward_ms', 0.) + begin.elapsed_time(end)
        rows.append(row)
    memory = total_gpu_memory()
    if memory['used_bytes'] >= min(4 * 2**30, memory['total_bytes']):
        raise RuntimeError('Dedicated GPU memory limit reached')
    report = {'purpose': 'actual captured forward attribution; total backward only',
              'fusion': args.fusion, 'chunk_tokens': 8, 'replays': args.replays,
              'median_chunk': {key: statistics.median(row[key] for row in rows) for key in rows[0]},
              'total_gpu_memory': memory, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'note': 'External events add instrumentation overhead; not production update throughput.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
