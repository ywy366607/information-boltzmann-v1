"""Real-corpus execution and task-gradient audit, without capability claims.

CPU is the default so inspecting an untrained medium cannot evict another
individual's GPU allocations. No checkpoint is produced by this audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.evaluation import field_energy_statistics
from information_boltzmann.runtime.training import belief_tensors, quiet_training_chunk


def gradient_groups(model):
    """Separate task gradients from the writer's auxiliary objective."""
    groups = {}
    for name, parameter in model.named_parameters():
        group = name.split('.')[0]
        values = groups.setdefault(group, {'parameters': 0, 'with_gradient': 0,
                                          'nonzero_gradient': 0, 'norm_squared': 0.0,
                                          'finite': True})
        values['parameters'] += parameter.numel()
        if parameter.grad is not None:
            grad = parameter.grad.detach()
            values['with_gradient'] += parameter.numel()
            values['nonzero_gradient'] += int(torch.count_nonzero(grad))
            values['norm_squared'] += float(grad.double().square().sum())
            values['finite'] &= bool(torch.isfinite(grad).all())
    for values in groups.values():
        values['gradient_norm'] = math.sqrt(values.pop('norm_squared'))
        values['gradient_rms'] = values['gradient_norm'] / math.sqrt(values['parameters'])
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/ib_owt_gpt2'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-tokens', type=int, default=32)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--event-duration', type=float, required=True)
    parser.add_argument('--seed', type=int, default=449)
    args = parser.parse_args()
    if (args.chunk_tokens < 1 or args.offset < 0 or
            not math.isfinite(args.event_duration) or args.event_duration <= 0):
        parser.error('Positive chunk/duration and nonnegative corpus offset required')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    data = np.load(args.data_dir / 'train.npy', mmap_mode='r')
    sample = np.array(data[args.offset:args.offset + args.chunk_tokens + 1], dtype=np.int64)
    if len(sample) != args.chunk_tokens + 1:
        parser.error('Insufficient corpus; no wrapping')
    ids = torch.from_numpy(sample)[None]
    constructor = dict(vocab_size=50257, shape=(8, 8, 4), channels=128,
                       bath_type='conductance', write_exchange='contact_mode',
                       port_scope='compact', activity_adaptation=True,
                       short_term_plasticity=True, pre_decoder_norm=True)
    model = PlasticMediumPorts3D(**constructor).train()
    initial = model.initial_belief()
    started = time.perf_counter()
    loss, evolved, nll = quiet_training_chunk(
        model, ids[:, :-1], ids[:, 1:], initial, event_duration=args.event_duration)
    forward_seconds = time.perf_counter() - started
    started = time.perf_counter()
    nll.backward(retain_graph=True)
    task_groups = gradient_groups(model)
    task_gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()
                      if p.grad is not None}
    model.zero_grad(set_to_none=True)
    auxiliary = loss - nll
    auxiliary.backward()
    auxiliary_groups = gradient_groups(model)
    dot = task_sq = auxiliary_sq = 0.0
    for name, parameter in model.named_parameters():
        task = task_gradients.get(name)
        if task is not None:
            task_sq += float(task.double().square().sum())
        if parameter.grad is not None:
            auxiliary_sq += float(parameter.grad.double().square().sum())
            if task is not None:
                dot += float((task.double() * parameter.grad.double()).sum())
    backward_seconds = time.perf_counter() - started
    report = {
        'purpose': 'untrained real-OWT interface/gradient readiness, not performance evaluation',
        'architecture': model.architecture, 'constructor': constructor,
        'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'physical_state_bytes': sum(t.numel() * t.element_size() for t in belief_tensors(initial)),
        'device': 'cpu', 'optimizer_updates': 0, 'tokens': args.chunk_tokens,
        'data_offset': args.offset, 'event_duration': args.event_duration,
        'initialization': 'random learned embeddings and decoder; zero decoder bias; no GPT-2 transfer',
        'nll': float(nll.detach()), 'write_objective': float(auxiliary.detach()),
        'task_gradient_groups': task_groups, 'write_gradient_groups': auxiliary_groups,
        'task_write_gradient_cosine': dot / math.sqrt(task_sq * auxiliary_sq)
            if task_sq > 0 and auxiliary_sq > 0 else None,
        'evolved_field': field_energy_statistics(evolved.medium.field),
        'full_state_finite': all(bool(torch.isfinite(t).all()) for t in belief_tensors(evolved)),
        'full_state_elapsed': float(evolved.medium.elapsed[0]),
        'cpu_forward_seconds': forward_seconds, 'cpu_two_objective_backward_seconds': backward_seconds,
        'gpu_speed': None, 'gpu_speed_note': 'CPU timings do not estimate production CUDA throughput',
        'manifest_sha256': hashlib.sha256((args.data_dir / 'manifest.json').read_bytes()).hexdigest(),
        'source_hashes': {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in (
            'scripts/ib/audit_medium_training_readiness.py',
            'information_boltzmann/core/plastic_ports.py',
            'information_boltzmann/runtime/training.py')},
        'unigram_representability': 'Decoder bias can equal log train-only frequencies while decoder weight is zero; no architecture-imposed NLL=7 floor.',
    }
    if not report['full_state_finite'] or any(not g['finite'] for g in task_groups.values()):
        raise FloatingPointError('Training readiness check encountered nonfinite state/task gradients')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
