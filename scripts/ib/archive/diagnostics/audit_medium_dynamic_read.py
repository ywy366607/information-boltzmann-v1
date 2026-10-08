"""Read-only migration, gradient and optional CUDA-graph cost calibration.

No optimizer exists. Uses the completed individual's actual physical state and
next real OWT event. Timing covers the read/decoder, not whole training updates.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.nn import functional as F

from information_boltzmann.core.plastic_ports import PlasticBelief, PlasticMediumPorts3D
from information_boltzmann.runtime.optimization import initialize_dynamic_read_branch
from scripts.ib.train_plastic_conductance import unpack_belief


def digest(model):
    result = hashlib.sha256()
    for name, parameter in model.named_parameters():
        result.update(name.encode())
        result.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def graph_pair(model, belief, target, backward):
    """Warm and capture the same native read, with coefficients inside capture."""
    def execute():
        logits, _ = model.read(belief)
        if backward:
            F.cross_entropy(logits, target).backward()
        return logits
    current = torch.cuda.current_stream()
    warm = torch.cuda.Stream()
    warm.wait_stream(current)
    with torch.cuda.stream(warm):
        for _ in range(3):
            model.zero_grad(set_to_none=False)
            execute()
    current.wait_stream(warm)
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=False)
    reference = execute().detach().clone()
    reference_grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()
                       if p.grad is not None} if backward else {}
    model.zero_grad(set_to_none=False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = execute()
    model.zero_grad(set_to_none=False)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, reference, atol=2e-6, rtol=2e-6)
    if backward:
        for name, p in model.named_parameters():
            if name in reference_grads:
                torch.testing.assert_close(p.grad, reference_grads[name], atol=2e-5, rtol=2e-5)
    return graph, captured


def timed(graph, model, count, backward):
    if backward:
        model.zero_grad(set_to_none=False)
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(count):
        # At fixed parameters, backward graph accumulates gradients only. No
        # optimizer step or physical state advance is performed.
        graph.replay()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--replays', type=int, default=30)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(709)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    constructor = dict(saved['config']['constructor'])
    constructor.update(medium_execution='native', port_execution='native')
    old = PlasticMediumPorts3D(**constructor).train()
    old.load_state_dict(saved['model'], strict=True)
    new = PlasticMediumPorts3D(**constructor, read_mode='dynamic').train()
    initialize_dynamic_read_branch(new, saved['model'])
    before_hash = digest(new)
    belief = unpack_belief(saved['belief'], 'cpu')
    # The saved field ends BEFORE assimilating the pending carry token; its
    # issued prediction therefore targets carry, not the following corpus token.
    target = torch.tensor([int(saved['learner']['carry_token'])])
    original, _ = old.read(belief)
    migrated, _ = new.read(belief)
    torch.testing.assert_close(migrated, original, atol=0, rtol=0)
    F.cross_entropy(migrated, target).backward()
    gradient_norms = {name: float(getattr(new.readout, name).weight.grad.norm())
                      for name in ('motion_policy', 'motion_keys', 'motion_merge')}
    assert all(np.isfinite(value) and value > 0 for value in gradient_norms.values())
    new.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    cpu_timing = {}
    with torch.no_grad():
        for label, model in (('instantaneous', old), ('dynamic', new)):
            left = time.perf_counter()
            for _ in range(args.replays):
                model.read(belief)
            cpu_timing[label] = (time.perf_counter() - left) * 1000 / args.replays
    report = dict(checkpoint=str(args.checkpoint.resolve()), optimizer_updates=0,
        mode='weights-only explicit migration; fixed real physical state',
        old_parameters=sum(p.numel() for p in old.parameters()),
        dynamic_parameters=sum(p.numel() for p in new.parameters()),
        exact_migration=True, new_projection_ce_gradient_norms=gradient_norms,
        cpu_read_decode_ms=cpu_timing, cpu_calibration_seconds=time.perf_counter() - t0,
        timing_scope='one native read and vocabulary decode at fixed state; includes coefficient preparation',
        source_hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in map(Path, (
                'information_boltzmann/core/plastic_medium.py',
                'information_boltzmann/core/readout_probes.py',
                'information_boltzmann/core/plastic_ports.py',
                'information_boltzmann/core/conductance_response.py',
                'information_boltzmann/runtime/optimization.py',
                'information_boltzmann/runtime/continuous.py',
                'information_boltzmann/runtime/training.py',
                'scripts/ib/audit_medium_dynamic_read.py'))})
    if args.device == 'cuda':
        # The concurrent job is left untouched. Cap THIS process at 900MiB,
        # including graphs; combined device memory must stay below 3900MiB.
        from information_boltzmann.runtime.gpu_memory import total_gpu_memory
        baseline = total_gpu_memory()['used_bytes'] / 2**20
        if baseline > 2900:
            raise RuntimeError('Insufficient GPU headroom for read-only calibration')
        torch.cuda.set_per_process_memory_fraction(900 / 4096)
        old.cuda(); new.cuda()
        belief = unpack_belief(saved['belief'], 'cuda')
        target = target.cuda()
        report['gpu_other_job_mib_before'] = baseline
        report['gpu_graphs'] = {}
        for backward in (False, True):
            records = {}
            with torch.set_grad_enabled(backward):
                first, _ = graph_pair(old, belief, target, backward)
                second, _ = graph_pair(new, belief, target, backward)
            # Alternate order across rounds to reduce concurrent-load bias.
            samples = {'instantaneous': [], 'dynamic': []}
            for index in range(4):
                arms = [('instantaneous', first, old), ('dynamic', second, new)]
                for label, graph, model in arms if index % 2 == 0 else reversed(arms):
                    samples[label].append(timed(graph, model, args.replays, backward))
            for label, values in samples.items():
                records[label] = dict(median_ms=float(np.median(values)), round_ms=values)
            records['native_graph_output_gradient_parity'] = True
            report['gpu_graphs']['read_decode_backward' if backward else 'read_decode_forward'] = records
            del first, second
            old.zero_grad(set_to_none=True); new.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            memory = total_gpu_memory()['used_bytes'] / 2**20
            if memory >= 3900:
                raise RuntimeError(f'Combined GPU use {memory}MiB exceeded calibration cap')
        report['gpu_allocated_peak_mib'] = torch.cuda.max_memory_allocated() / 2**20
        report['gpu_reserved_peak_mib'] = torch.cuda.max_memory_reserved() / 2**20
        report['gpu_combined_mib_after'] = total_gpu_memory()['used_bytes'] / 2**20
        report['gpu_timing_limit'] = 'Concurrent GPU workload; read-only graphs, no fused compile, no full BPTT32 capture'
    report['all_candidate_parameters_unchanged'] = before_hash == digest(new)
    assert report['all_candidate_parameters_unchanged']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
