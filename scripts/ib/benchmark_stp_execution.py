"""Captured STP component cost and numerical equivalence, no capability claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from information_boltzmann.core.short_term_plasticity import LocalShortTermPlasticity
from information_boltzmann.runtime.gpu_memory import total_gpu_memory


def original_step(state, field, flux, duration, coefficients):
    """Pre-optimization expression, retained here only as a reference."""
    signals = []
    for axis, current in enumerate(flux):
        neighbor = torch.roll(field, -1, axis + 1)
        moving = current.square().mean(-1)
        total = moving + 0.5 * (field.square().mean(-1) + neighbor.square().mean(-1))
        signals.append(moving / total.clamp_min(torch.finfo(field.dtype).eps))
    rates, baseline = coefficients
    recovery, closing, activity_rate = rates.unbind(-1)
    r = torch.stack(signals, -1) * activity_rate[None]
    u0 = baseline[None]
    x, u = state.unbind(-1)
    rate = closing[None] + u0 * r
    target = u0 * (closing[None] + r) / rate
    relax = LocalShortTermPlasticity._relax
    half = relax(u, target, rate, 0.5 * duration)
    depletion = recovery[None] + half * r
    next_x = relax(x, recovery[None] / depletion, depletion, duration)
    next_u = relax(half, target, rate, 0.5 * duration)
    return torch.stack((next_x, next_u), -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=100)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('Positive repeat count required')
    torch.set_num_threads(1)
    torch.manual_seed(827)
    torch.cuda.set_per_process_memory_fraction(0.4)
    rule = LocalShortTermPlasticity(8).cuda()
    field = torch.randn(1, 8, 8, 4, 128, device='cuda', requires_grad=True)
    flux = tuple(torch.randn_like(field).requires_grad_() for _ in range(3))
    state = torch.rand(1, 8, 8, 4, 3, 2, device='cuda', requires_grad=True)
    rates = torch.rand(8, 8, 4, 3, 3, device='cuda').add(0.1).requires_grad_()
    baseline = torch.rand(8, 8, 4, 3, device='cuda').mul(0.8).add(0.1).requires_grad_()
    dt = torch.full((1, 1, 1, 1, 1), 0.0025, device='cuda', requires_grad=True)
    direction = torch.randn_like(state)
    inputs = (state, field, *flux, dt, rates, baseline)
    report = {'purpose': 'isolated captured STP component execution, not whole-update attribution',
              'shape': [8, 8, 4], 'channels': 128, 'repeats': args.repeats, 'variants': {}}
    reference_output = original_step(state, field, flux, dt, (rates, baseline))
    reference_grads = torch.autograd.grad(reference_output, inputs, direction)
    reference_output = reference_output.detach()
    for name, function in (('original', original_step), ('shared_energy_native', rule.native_step),
                           ('fused', rule.forward)):
        def operation():
            result = function(state, field, flux, dt, (rates, baseline))
            (result * direction).sum().backward()
            return result
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                for tensor in inputs:
                    tensor.grad = None
                operation()
        torch.cuda.current_stream().wait_stream(warm)
        torch.cuda.synchronize()
        def zero():
            for tensor in inputs:
                tensor.grad.zero_()
        zero()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = operation()
        zero()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(result, reference_output, atol=3e-7, rtol=3e-6)
        error = 0.
        for tensor, expected in zip(inputs, reference_grads):
            torch.testing.assert_close(tensor.grad, expected, atol=3e-6, rtol=3e-5)
            error = max(error, float((tensor.grad - expected).abs().max()))
        memory = total_gpu_memory()
        if memory['used_bytes'] >= min(4 * 2**30, memory['total_bytes']):
            raise RuntimeError('Dedicated GPU memory limit reached')
        times = []
        for _ in range(3):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(args.repeats):
                graph.replay()
            end.record()
            end.synchronize()
            times.append(begin.elapsed_time(end) / args.repeats)
        report['variants'][name] = {'captured_forward_backward_ms': min(times),
                                    'timings_ms': times, 'max_gradient_error': error,
                                    'max_state_error': float((result.detach() - reference_output).abs().max()),
                                    'total_gpu_memory': memory}
        del graph, result
    report['interpretation'] = 'Component timing only; production speed measured separately on real OWT.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
