"""Real OWT update timing with bounded CUDA allocation and total VRAM monitoring.

This measures execution, not capability. All fast-state values persist between
BPTT chunks. Memory-counter queries are excluded from execution timings.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.training import (
    CapturedPlasticChunk, belief_tensors, quiet_training_chunk)
from information_boltzmann.runtime.gpu_memory import windows_gpu_memory, total_gpu_memory
from information_boltzmann.runtime.execution_cache import configure_execution_cache


class VramLimitExceeded(RuntimeError):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ib_owt_gpt2/train.npy'))
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--chunk-tokens', type=int, default=8)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--updates', type=int, default=1)
    parser.add_argument('--warmup-updates', type=int, default=0)
    parser.add_argument('--backend', choices=('eager', 'cuda_graph'), default='eager')
    parser.add_argument('--write-exchange', choices=('global', 'contact_mode'),
                        default='contact_mode')
    parser.add_argument('--port-scope', choices=('global', 'compact'), default='compact')
    parser.add_argument('--activity-adaptation', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--short-term-plasticity', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--medium-execution', choices=('native', 'fused'), default='fused')
    parser.add_argument('--port-execution', choices=('native', 'fused'), default='fused')
    parser.add_argument('--compile-cache-dir', type=Path, default=Path('scratch/compiler_cache'))
    parser.add_argument('--event-duration', type=float, required=True,
                        help='Explicit model-time cadence for execution measurement only')
    parser.add_argument('--substeps', type=int, default=1)
    parser.add_argument('--allocator-fraction', type=float, default=0.30)
    parser.add_argument('--vram-limit-gib', type=float, default=4.0)
    parser.add_argument('--initialize-kernel-driver-baseline', action='store_true',
                        help='Initialize scalar CUDA kernel and CPU monitoring transfer before model allocation')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    execution_cache = configure_execution_cache(args.compile_cache_dir)
    if min(args.tokens, args.chunk_tokens, args.substeps, args.updates) < 1 or args.offset < 0 or args.warmup_updates < 0:
        parser.error('Positive tokens, chunks and substeps required; offset >= 0')
    if not math.isfinite(args.vram_limit_gib) or args.vram_limit_gib <= 0:
        parser.error('Positive finite dedicated VRAM limit required')
    if args.backend == 'cuda_graph' and args.tokens % args.chunk_tokens:
        parser.error('Graph measurement requires an integer number of fixed-shape chunks')
    if not math.isfinite(args.event_duration) or args.event_duration <= 0:
        parser.error('Positive finite event duration required')
    torch.set_num_threads(1)
    torch.manual_seed(449)
    data = np.load(args.data, mmap_mode='r')
    total_updates = args.updates + args.warmup_updates
    sample = np.array(data[args.offset:args.offset + args.tokens * total_updates + 1], dtype=np.int64)
    if len(sample) != args.tokens * total_updates + 1:
        parser.error('Insufficient real corpus data')
    model = PlasticMediumPorts3D(vocab_size=50257, shape=(8, 8, 4), channels=128,
                                bath_type='conductance', write_exchange=args.write_exchange,
                                port_scope=args.port_scope, activity_adaptation=args.activity_adaptation,
                                short_term_plasticity=args.short_term_plasticity,
                                medium_execution=args.medium_execution, port_execution=args.port_execution)
    parameter_count = sum(p.numel() for p in model.parameters())
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    report = {
        'purpose': 'real OWT update execution measurement only',
        'architecture': model.architecture, 'shape': [8, 8, 4], 'channels': 128,
        'write_exchange': args.write_exchange,
        'port_scope': args.port_scope,
        'activity_adaptation': args.activity_adaptation,
        'short_term_plasticity': args.short_term_plasticity,
        'medium_execution': args.medium_execution,
        'port_execution': args.port_execution,
        'execution_cache': execution_cache,
        'write_port_radius': (model.write_agent.local_ports.physical_radius if args.port_scope == 'compact' else None),
        'read_port_radius': (model.readout.physical_radius if args.port_scope == 'compact' else None),
        'vocab_size': 50257, 'parameters': parameter_count,
        'parameter_bytes': parameter_bytes,
        'parameter_gradient_adam_moment_floor_bytes': 4 * parameter_bytes,
        'tokens_per_update': args.tokens, 'bptt_chunk_tokens': args.chunk_tokens,
        'event_duration': args.event_duration, 'substeps': args.substeps,
        'precision': 'FP32', 'backend': args.backend + ', quiet diagnostics',
        'warmup_updates': args.warmup_updates, 'requested_measured_updates': args.updates,
        'completed_updates': 0, 'completed_tokens': 0,
        'data': str(args.data), 'data_offset': args.offset,
        'allocator_fraction': args.allocator_fraction,
        'pytorch': torch.__version__, 'measurements': [], 'memory_samples': [], 'update_timings': [],
        'memory_policy': 'total dedicated VRAM limit; shared usage recorded, growth permitted',
        'vram_limit_bytes': int(args.vram_limit_gib * 2**30),
    }

    def publish():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    torch.cuda.set_per_process_memory_fraction(args.allocator_fraction)
    torch.cuda.init()
    torch.cuda.synchronize()
    if args.initialize_kernel_driver_baseline:
        # Separate no-model controls establish first-kernel and first scalar
        # GPU-to-CPU monitoring-transfer overhead (2 MiB each on this host).
        # Record the raw delta, never absorb allocations from the model itself.
        context_memory = windows_gpu_memory()
        scalar = torch.ones(1).cuda()
        result = scalar + scalar
        torch.cuda.synchronize()
        kernel_memory = windows_gpu_memory()
        finite = bool(torch.isfinite(result).all())
        torch.cuda.synchronize()
        monitoring_memory = windows_gpu_memory()
        report['no_model_driver_initialization_control'] = {
            'operation': 'scalar addition then one finite boolean copied to CPU',
            'before': context_memory, 'after_kernel': kernel_memory,
            'after_monitoring_transfer': monitoring_memory,
            'monitoring_value': finite}
        del scalar, result
        torch.cuda.empty_cache()
    baseline = windows_gpu_memory()
    report['cuda_initialization_memory'] = baseline
    print(json.dumps({'parameters': parameter_count, 'cuda_baseline': baseline}), flush=True)
    torch.cuda.reset_peak_memory_stats()

    def guard(stage):
        torch.cuda.synchronize()
        try:
            memory = windows_gpu_memory()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            # Optional per-process counters can fail while another job samples
            # WMI. The mandatory total dedicated-VRAM guard remains active.
            memory = None
            report.setdefault('unavailable_process_counters', []).append(
                {'stage': stage, 'error_type': type(error).__name__})
        total_memory = total_gpu_memory()
        snapshot = {'stage': stage, 'windows': memory, 'total_gpu': total_memory,
                    'allocated_bytes': torch.cuda.memory_allocated(),
                    'reserved_bytes': torch.cuda.memory_reserved(),
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        snapshot['shared_growth_bytes'] = (None if memory is None or baseline is None else
                                          memory['shared_bytes'] - baseline['shared_bytes'])
        report['memory_samples'].append(snapshot)
        publish()
        if total_memory['used_bytes'] >= min(total_memory['total_bytes'], report['vram_limit_bytes']):
            raise VramLimitExceeded(f'Total dedicated VRAM reached limit at {stage}')

    def timed(stage, operation):
        torch.cuda.synchronize()
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        start_event.record()
        result = operation()
        end_event.record()
        torch.cuda.synchronize()
        report['measurements'].append({'stage': stage,
                                       'wall_seconds': time.perf_counter() - start,
                                       'cuda_seconds': start_event.elapsed_time(end_event) / 1000})
        return result

    try:
        model = model.cuda().train()
        ids = torch.from_numpy(sample).cuda()[None]
        belief = model.initial_belief()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
        optimizer.zero_grad(set_to_none=True)
        guard('model_and_initial_state')

        capture = None
        if args.backend == 'cuda_graph':
            start = time.perf_counter()
            capture = CapturedPlasticChunk(
                model, ids[:, :args.chunk_tokens], ids[:, 1:args.chunk_tokens + 1], belief,
                event_duration=args.event_duration, substeps=args.substeps,
                loss_scale=args.chunk_tokens / args.tokens)
            torch.cuda.synchronize()
            report['capture_setup_seconds'] = time.perf_counter() - start
            guard('training_graph_captured')
            # Audit two DIFFERENT chunks, including accumulated parameter grads.
            capture.zero_grad()
            expected, expected_losses = belief, []
            # Compare against the unfused operators at the full production
            # shape as well, rather than only replay versus fused eager.
            previous_execution = model.medium.execution_backend, model.port_execution
            model.medium.execution_backend, model.port_execution = 'native', 'native'
            try:
                for offset in (0, args.chunk_tokens):
                    loss, expected, _ = quiet_training_chunk(
                        model, ids[:, offset:offset + args.chunk_tokens],
                        ids[:, offset + 1:offset + args.chunk_tokens + 1], expected,
                        event_duration=args.event_duration, substeps=args.substeps)
                    (loss * (args.chunk_tokens / args.tokens)).backward()
                    expected_losses.append(float(loss.detach()))
                    expected = expected.detach()
            finally:
                model.medium.execution_backend, model.port_execution = previous_execution
            gradients = {name: p.grad.detach().cpu().clone()
                         for name, p in model.named_parameters() if p.grad is not None}
            expected_values = tuple(x.detach().cpu().clone() for x in belief_tensors(expected))
            capture.zero_grad()
            actual, actual_losses = belief, []
            for offset in (0, args.chunk_tokens):
                loss, actual, _ = capture.backward(
                    ids[:, offset:offset + args.chunk_tokens],
                    ids[:, offset + 1:offset + args.chunk_tokens + 1], actual)
                actual_losses.append(float(loss))
            max_gradient_error = 0.0
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    gradient = parameter.grad.detach().cpu()
                    torch.testing.assert_close(gradient, gradients[name], atol=3e-6, rtol=3e-4)
                    max_gradient_error = max(max_gradient_error,
                                             float((gradient - gradients[name]).abs().max()))
            max_state_error = 0.0
            for value, reference in zip(belief_tensors(actual), expected_values):
                value = value.detach().cpu()
                torch.testing.assert_close(value, reference, atol=3e-6, rtol=3e-5)
                max_state_error = max(max_state_error, float((value - reference).abs().max()))
            torch.testing.assert_close(torch.tensor(actual_losses), torch.tensor(expected_losses))
            report['graph_equivalence'] = {'two_chunk_gradient_accumulation': 'passed',
                                          'reference': 'native medium and native write/read',
                                          'max_state_absolute_error': max_state_error,
                                          'max_gradient_absolute_error': max_gradient_error,
                                          'eager_losses': expected_losses, 'graph_losses': actual_losses}
            del gradients, expected_values, expected, actual, loss
            capture.zero_grad()
            guard('training_graph_equivalence_checked')

        for update in range(total_updates):
            if capture is None:
                optimizer.zero_grad(set_to_none=True)
            else:
                capture.zero_grad()
            measurement_start = len(report['measurements'])
            for relative_index in range(0, args.tokens, args.chunk_tokens):
                index = update * args.tokens + relative_index
                count = min(args.chunk_tokens, args.tokens - relative_index)
                input_ids = ids[:, index:index + count]
                targets = ids[:, index + 1:index + count + 1]
                if capture is None:
                    loss, next_belief, nll = timed(f'forward_{index}', lambda: quiet_training_chunk(
                        model, input_ids, targets, belief, event_duration=args.event_duration,
                        substeps=args.substeps))
                    guard(f'forward_{index}')
                    timed(f'backward_{index}', lambda: (loss * (count / args.tokens)).backward())
                else:
                    loss, next_belief, nll = timed(f'graph_forward_backward_{index}', lambda:
                                                  capture.backward(input_ids, targets, belief))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite loss')
                report['completed_tokens'] += count
                report['last_chunk_nll'] = float(nll.detach())
                belief = next_belief.detach()
                del loss, nll, next_belief
                guard(f'backward_{index}')
                print(f'Backward complete: {report["completed_tokens"]}/{args.tokens * total_updates} tokens', flush=True)

            def step():
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                              error_if_nonfinite=True)
                optimizer.step()
                return float(gradient_norm)

            report['unclipped_gradient_norm'] = timed(f'adamw_step_{update}', step)
            report['completed_updates'] += 1
            guard(f'adamw_step_{update}_with_moments_allocated')
            phases = report['measurements'][measurement_start:]
            report['update_timings'].append({
                'update': update, 'warmup': update < args.warmup_updates,
                'execution_wall_seconds': sum(x['wall_seconds'] for x in phases),
                'execution_cuda_seconds': sum(x['cuda_seconds'] for x in phases),
                'phases_wall_seconds': {key: sum(x['wall_seconds'] for x in phases
                                                if x['stage'].startswith(key))
                                       for key in ('forward', 'backward', 'graph_forward_backward', 'adamw')}})
        report['status'] = 'completed'
        if model.activity_adaptation:
            with torch.no_grad():
                from dataclasses import replace
                response = model.medium.conductance_response
                material = model.medium.material_field()
                c = response.coefficients(material)
                state = belief.medium
                native = response.opening_rates(state.field, state.flux,
                                                replace(c, adaptation_gain=None), state.receptors)
                adapted = response.opening_rates(state.field, state.flux, c, state.receptors)
                gain, _, metric = model.medium.conduction_plasticity.coefficients(material)
                evidence = model.medium.conduction_plasticity.evidence(state.field, state.flux, metric)
                changed = model.medium.conduction_plasticity.adapted_evidence(
                    state.field, state.flux, metric, gain, response.inhibition_history(state.receptors))
                snapshot = model.port_snapshot(belief)
                report['activity_feedback_execution'] = {
                    'interpretation': 'real-corpus execution and engagement only; no capability conclusion',
                    'processing_activity_mean': float(response.processing_activity(state.field, state.flux).mean()),
                    'inhibitory_gate_mean': float(state.receptors[..., 1, :].mean()),
                    'inhibitory_opening_mean_increment': float((adapted[..., 1, :] - native[..., 1, :]).mean()),
                    'excitatory_opening_mean_ratio': float((adapted[..., 0, :] / native[..., 0, :]).mean()),
                    'routing_evidence_mean_change': float((changed - evidence).abs().mean()),
                    'local_port_snapshot': {key: value.cpu().tolist() for key, value in snapshot.items()
                                            if key not in ('write_port_weights', 'read_port_weights')},
                }
        measured = [x for x in report['update_timings'] if not x['warmup']]
        report['measured_seconds_per_update'] = sum(x['execution_wall_seconds'] for x in measured) / len(measured)
    except VramLimitExceeded as error:
        report['status'] = 'stopped_dedicated_vram_limit'
        report['error'] = str(error)
    except torch.cuda.OutOfMemoryError as error:
        report['status'] = 'stopped_allocator_budget_exceeded'
        report['error'] = str(error)
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        report['execution_wall_seconds_excluding_memory_queries'] = sum(
            x['wall_seconds'] for x in report['measurements'])
        report['execution_cuda_seconds'] = sum(x['cuda_seconds'] for x in report['measurements'])
        publish()
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
