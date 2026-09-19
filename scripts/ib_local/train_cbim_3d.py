"""Train the resolution-independent 3D CBIM truncation on OpenWebText."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from scripts.ib_local.cbim_cuda_graph import CBIMGraphTrainer
from scripts.ib_local.cbim_operator3d import CBIMOperator3D


def atomic_json(path, data, required=False):
    """A browser briefly locking a monitor file must not stop training."""
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    payload = json.dumps(data, allow_nan=False)
    for attempt in range(60):
        try:
            temp.write_text(payload, encoding='utf-8')
            os.replace(temp, path)
            return True
        except PermissionError:
            time.sleep(.05 * min(attempt + 1, 4))
    if required:
        raise PermissionError(f'Unable to update required file: {path}')
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/ib_owt_gpt2'))
    parser.add_argument('--output', type=Path,
                        default=Path('results/cbim_operator3d_3000'))
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--shape', type=int, nargs=3, default=(4, 4, 4))
    parser.add_argument('--channels', type=int, default=128)
    parser.add_argument('--validate-every', type=int, default=250)
    parser.add_argument('--validation-tokens', type=int, default=4096)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    args.shape = tuple(args.shape)

    torch.set_num_threads(2)
    torch.manual_seed(11)
    torch.cuda.set_per_process_memory_fraction(.48)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'last.pt').exists() and args.resume is None:
        raise ValueError('Existing run requires --resume or a new output directory')
    train = np.load(args.data / 'train.npy', mmap_mode='r')
    valid = np.load(args.data / 'validation.npy', mmap_mode='r')
    if args.steps * args.tokens + 1 > len(train):
        raise ValueError('Training stream is shorter than requested budget')
    if args.validation_tokens % args.tokens:
        raise ValueError('validation-tokens must be divisible by tokens')

    source_files = [
        'scripts/ib_local/cbim_operator3d.py',
        'scripts/ib_local/cbim_cuda_graph.py',
        __file__,
    ]
    config = {
        'architecture': 'CBIM-operator3d-v1',
        'data': str(args.data), 'output': str(args.output),
        'steps': args.steps, 'tokens': args.tokens,
        'shape': args.shape, 'channels': args.channels,
        'grid_points': math.prod(args.shape), 'queries': 4,
        'scatter_sweeps': 1, 'precision': 'FP32', 'seed': 11,
        'lr': 3e-4, 'validate_every': args.validate_every,
        'validation_tokens': args.validation_tokens,
        'stopping': 'fixed 3000-update budget; convergence not established',
        'manifest': json.loads((args.data / 'manifest.json').read_text()),
        'source_sha256': {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in source_files
        },
    }
    atomic_json(args.output / 'config.json', config, required=True)
    atomic_json(args.output / 'progress.json', {
        'status': 'capturing', 'step': 0, 'target_steps': args.steps})

    model = CBIMOperator3D(shape=args.shape, d=args.channels).cuda()
    runner = CBIMGraphTrainer(model, tokens=args.tokens)
    step, best = 0, float('inf')
    if args.resume:
        saved = torch.load(args.resume, map_location='cuda', weights_only=False)
        for key in ('architecture', 'shape', 'channels', 'tokens'):
            if saved['config'][key] != config[key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
        model.load_state_dict(saved['model'])
        current = runner.optimizer.state_dict()
        for key, state in saved['optimizer']['state'].items():
            for name, value in state.items():
                current['state'][key][name].copy_(value)
        runner.state.copy_(saved['state'])
        step, best = saved['step'], saved['best_validation_nll']

    def batch(data, offset):
        return tuple(torch.as_tensor(
            np.array(data[offset + shift:offset + shift + args.tokens]),
            dtype=torch.long, device='cuda')[None] for shift in (0, 1))

    def save(name):
        temp = args.output / f'{name}.{os.getpid()}.tmp'
        torch.save({
            'model': model.state_dict(),
            'optimizer': runner.optimizer.state_dict(),
            'state': runner.state.detach(), 'step': step,
            'events': step * args.tokens,
            'best_validation_nll': best, 'config': config,
        }, temp)
        os.replace(temp, args.output / name)

    def log(row):
        with (args.output / 'metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row), flush=True)

    @torch.no_grad()
    def validate():
        state = torch.zeros_like(runner.state)
        total = 0.
        for offset in range(0, args.validation_tokens + args.tokens, args.tokens):
            ids, targets = batch(valid, offset)
            loss, state, _ = model(ids, targets, state)
            if offset:
                total += float(loss) * args.tokens
        return total / args.validation_tokens

    try:
        if args.resume is None:
            best = validate()
            log({'kind': 'validation', 'step': 0, 'validation_nll': best})
            save('BBest.pt')
            save('last.pt')
        while step < args.steps:
            ids, targets = batch(train, step * args.tokens)
            torch.cuda.synchronize()
            started = time.perf_counter()
            loss, state, diagnostics = runner.step(ids, targets)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            step += 1
            loss_value = float(loss)
            if not math.isfinite(loss_value):
                raise FloatingPointError('Non-finite training loss')
            if elapsed > 1.5 and step > 2:
                raise RuntimeError(f'Training step exceeded 1.5 seconds: {elapsed:.3f}')

            if step == 1 or step % 10 == 0 or step == args.steps:
                field = state.detach()
                dispersion = model.transport.dispersion().detach()
                row = {
                    'kind': 'train', 'step': step,
                    'events': step * args.tokens, 'nll': loss_value,
                    'seconds': elapsed,
                    'energy': float(diagnostics['final_energy']),
                    'local_projection_rate': float(
                        diagnostics['local_projection_rate']),
                    'grad_norm': float(runner.grad_norm),
                    'variance_x': float(field.var(1, unbiased=False).mean()),
                    'variance_y': float(field.var(2, unbiased=False).mean()),
                    'variance_z': float(field.var(3, unbiased=False).mean()),
                    'dispersion_abs_mean': float(dispersion.abs().mean()),
                    'dispersion_abs_max': float(dispersion.abs().max()),
                    'allocated_mib': torch.cuda.memory_allocated() / 2**20,
                    'reserved_mib': torch.cuda.memory_reserved() / 2**20,
                }
                log(row)
                atomic_json(args.output / 'progress.json', {
                    'status': 'running', 'target_steps': args.steps, **row})
                # Channel-averaged 3D scalar volume is sufficient for browser slices.
                atomic_json(args.output / 'live_state.json', {
                    **row, 'volume': field[0].mean(-1).cpu().tolist()})

            if step % args.validate_every == 0 or step == args.steps:
                score = validate()
                improved = score < best
                if improved:
                    best = score
                log({'kind': 'validation', 'step': step,
                     'validation_nll': score, 'best_validation_nll': best})
                if improved:
                    save('BBest.pt')
                save('last.pt')
                save(f'age_{step:06d}.pt')
        atomic_json(args.output / 'progress.json', {
            'status': 'complete', 'step': step,
            'target_steps': args.steps, 'best_validation_nll': best})
    except BaseException as error:
        atomic_json(args.output / 'progress.json', {
            'status': 'failed', 'step': step, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
