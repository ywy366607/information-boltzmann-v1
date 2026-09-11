"""One persistent OWT individual; update-count budgets and longitudinal records."""
from __future__ import annotations

import argparse
from collections import deque
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fine_grain.information_boltzmann import InformationBoltzmann
from fine_grain.information_boltzmann.streaming import StreamRunner
from fine_grain.information_boltzmann.diagnostics import conditional_response


class TensorBudget(dict):
    """Accumulate detached accounting tensors without per-substep CPU waits."""
    defer_cpu = True


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def atomic_checkpoint(runner, path, metadata):
    temporary = path.with_suffix('.tmp.pt')
    runner.save(temporary, metadata)
    os.replace(temporary, path)


def energy(model, state):
    return .5 * (state.v.square() + model.force.kappa * state.x.square()).sum(-1).mean()


@torch.no_grad()
def event_observables(model, state):
    gamma = model.force.damping(state.x)
    values = {'energy': energy(model, state), **state.moments(),
              'gamma_mean': gamma.mean(), 'gamma_std': gamma.std(unbiased=False),
              'gamma_min': gamma.min(), 'gamma_max': gamma.max()}
    for axis in range(state.x.shape[-1]):
        values[f'mean_x_{axis}'] = state.x[:, axis].mean()
        values[f'mean_v_{axis}'] = state.v[:, axis].mean()
    return values


@torch.no_grad()
def validate(model, tokens, bos, length=1024, burn_in=128):
    """Separate frozen validation stream; no updates, no caller RNG consumption."""
    replica = copy.deepcopy(model)
    runner = StreamRunner(replica, bos, seed=99173)
    losses = []
    for index, token in enumerate(tokens[:length + burn_in]):
        runner.predict()
        loss = runner.observe(int(token))
        if index >= burn_in:
            losses.append(loss)
    return {'nll': float(np.mean(losses)), 'scored_tokens': len(losses),
            'burn_in_tokens': burn_in, 'unit': 'byte_token',
            'frozen': True, 'validation_seed': 99173}


@torch.no_grad()
def checkpoint_response(model, runner, future, length):
    """Frozen copy of this individual's phase; true continuation token alignment."""
    replica = copy.deepcopy(model)
    sequence = [runner.current_token, *map(int, future[:max(0, length-1)])]
    generator = torch.Generator(device=runner.state.x.device)
    generator.set_state(runner.generator.get_state())
    result = conditional_response(replica, runner.state.detach(), sequence, generator,
                                  epsilon=1e-3, burn_in=0)
    result['interpretation'] = 'finite_common_noise_phase_response_not_history_memory_or_criticality'
    result['epsilon'] = 1e-3
    result['tokens'] = len(sequence)
    return result


def temporal_windows(rows):
    """Age-local empirical ACF; no fitted exponent or stationarity assumption."""
    result = []
    for start in range(0, len(rows), 4096):
        window = rows[start:start+4096]
        if len(window) < 256:
            continue
        item = {'start_event': window[0]['event'], 'end_event': window[-1]['event'],
                'status': 'descriptive_age_local_correlations_not_scaling_law'}
        for key in ('ce', 'energy', 'gamma_mean', 'mean_x_0', 'mean_v_0'):
            a = np.asarray([r[key] for r in window], dtype=np.float64)
            a = a - a.mean()
            power = float(np.dot(a, a))
            if power <= 1e-24:
                item[key] = {'acf': None, 'reason': 'no_resolvable_variance'}
                continue
            fft = np.fft.rfft(a, n=2*len(a))
            corr = np.fft.irfft(fft * fft.conj())[:len(a)] / power
            item[key] = {'acf': {str(lag): float(corr[lag]) for lag in
                                (1, 2, 4, 8, 16, 32, 64, 128) if lag < len(a)//4},
                         'first_half_mean': float(np.mean([r[key] for r in window[:len(a)//2]])),
                         'second_half_mean': float(np.mean([r[key] for r in window[len(a)//2:]]))}
            # The periodogram is descriptive: input, explicit sin/cos clock,
            # updates every 16 events and aging may all create spectral structure.
            tapered = np.fft.rfft(a * np.hanning(len(a)))
            item[key]['frequency_cycles_per_event'] = np.fft.rfftfreq(len(a))[1:].tolist()
            item[key]['windowed_power'] = (np.abs(tapered[1:])**2/len(a)).tolist()
        result.append(item)
    return result


def gpu_sampler(output, stop):
    with (output / 'gpu.jsonl').open('a', encoding='utf-8', buffering=1) as handle:
        while not stop.is_set():
            try:
                line = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                                                '--format=csv,noheader,nounits'], text=True, timeout=5).strip()
                handle.write(json.dumps({'wall_time': time.time(), 'gpu_csv': line})+'\n')
            except (OSError, subprocess.SubprocessError):
                pass
            stop.wait(5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=5000, help='Total optimizer updates, including resumed history')
    p.add_argument('--resume', type=Path)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--checkpoint-every', type=int, default=100)
    p.add_argument('--diagnose-every', type=int, default=500)
    p.add_argument('--validation-tokens', type=int, default=1024)
    p.add_argument('--response-tokens', type=int, default=256)
    p.add_argument('--profile', action='store_true', help='Real OWT runtime calibration, not capability evidence')
    args = p.parse_args()
    torch.set_num_threads(1)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    manifest = json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
    if manifest['source'] != 'openwebtext' or len(manifest['documents']) < 1000:
        raise ValueError('This entry point requires the prepared real OWT subset')
    if (config['data']['revision'] != manifest['revision'] or not manifest['revision']
            or config['data']['vocab_size'] != manifest['vocab_size']
            or config['data']['tokenizer'] != manifest['tokenizer']):
        raise ValueError('Dataset revision/tokenizer/vocabulary mismatch')
    for name in ('train.npy', 'validation.npy', 'tokenizer.json'):
        if digest(args.data/name) != manifest['files'][name]:
            raise ValueError(f'Dataset checksum mismatch: {name}')
    settings = config['train']
    interval = settings['update_every_real_tokens']
    if interval != settings['tbptt_length'] or settings['batch_size'] != 1:
        raise ValueError('One individual, explicit matching online update and backpropagation window required')
    train = np.load(args.data/'train.npy', mmap_mode='r')
    validation = np.load(args.data/'validation.npy', mmap_mode='r')
    target = args.updates * interval
    if not 0 < target <= len(train) or args.validation_tokens+128 > len(validation):
        raise ValueError('Invalid budget or insufficient data; no stream wrapping')
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError('Use a new run directory or explicitly resume the individual')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.cuda.set_per_process_memory_fraction(.75)
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(config['seed'])
    model = InformationBoltzmann.from_config(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                                 weight_decay=settings['weight_decay'], foreach=args.device == 'cuda')
    runner = StreamRunner(model, manifest['bos_token'], config['seed'], optimizer,
                          interval, settings['gradient_clip'])
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), *(root/'fine_grain/information_boltzmann').glob('*.py')]
    metadata = {'config': config, 'manifest_sha256': digest(args.data/'manifest.json'),
                'split': 'train', 'source_hashes': {f.relative_to(root).as_posix(): digest(f) for f in sources},
                'runtime': {'torch': str(torch.__version__), 'cuda': torch.version.cuda}}
    if args.resume:
        old = runner.load(args.resume)
        if runner.updates == 0:
            raise ValueError('Birth snapshot lacks the initialization autograd graph; resume an updated checkpoint')
        if old != metadata:
            raise ValueError('Exact continuation requires identical config, data, runtime and source')
    if runner.updates >= args.updates:
        raise ValueError('Requested update horizon already reached')
    write_json(args.output/'run.json', {**metadata, 'pid': os.getpid(), 'requested_updates': args.updates,
                                      'parameters': sum(p.numel() for p in model.parameters()),
                                      'device': args.device, 'profile': args.profile})
    # Archive the executed source/config separately from mutable working files.
    if not args.resume:
        for file in sources:
            destination = args.output/'source'/file.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(file.read_bytes())
        atomic_checkpoint(runner, args.output/'birth.pt', metadata)
    stop = threading.Event()
    sampler = threading.Thread(target=gpu_sampler, args=(args.output, stop), daemon=True)
    if args.device == 'cuda':
        sampler.start()
    started, initial_events = time.perf_counter(), runner.events
    # Disk stores the lifetime; RAM retains a bounded age-local observation window.
    all_rows, pending_rows, pending_tensors, validations = deque(maxlen=4096), [], [], []
    maximum_residual = 0.
    if args.resume:
        # Roll back only logs later than the exact checkpoint cursor.
        path = args.output/'events.jsonl'
        if path.exists():
            temporary = path.with_suffix('.resume.tmp')
            with path.open(encoding='utf-8') as old_log, temporary.open('w', encoding='utf-8') as new_log:
                for line in old_log:
                    r = json.loads(line)
                    if r['event'] <= runner.events:
                        new_log.write(line)
                        all_rows.append(r)
                        maximum_residual = max(maximum_residual, abs(r['accounting_residual']))
            os.replace(temporary, path)
        path = args.output/'updates.jsonl'
        if path.exists():
            temporary = path.with_suffix('.resume.tmp')
            with path.open(encoding='utf-8') as old_log, temporary.open('w', encoding='utf-8') as new_log:
                for line in old_log:
                    if json.loads(line)['update'] <= runner.updates:
                        new_log.write(line)
            os.replace(temporary, path)
        for path in sorted(args.output.glob('validation_*.json')):
            value = json.loads(path.read_text())
            if value['update'] <= runner.updates:
                validations.append(value)
        # Keep records from the abandoned future explicitly outside the active history.
        for pattern in ('validation_*.json', 'response_*.json', 'temporal_*.json', 'age_*.pt'):
            for path in args.output.glob(pattern):
                suffix = path.stem.rsplit('_', 1)[-1]
                if suffix.isdigit() and int(suffix) > runner.updates:
                    archive = args.output/'superseded_on_resume'
                    archive.mkdir(exist_ok=True)
                    os.replace(path, archive/f'{time.time_ns()}_{path.name}')
    try:
        with (args.output/'events.jsonl').open('a', encoding='utf-8', buffering=1) as log, \
                (args.output/'updates.jsonl').open('a', encoding='utf-8', buffering=1) as update_log:
            if not args.profile and not args.resume:
                baseline = validate(model, validation, manifest['bos_token'], args.validation_tokens)
                baseline['update'] = 0
                validations.append(baseline)
                write_json(args.output/'validation_000000.json', baseline)
            for cursor in range(runner.events, target):
                if not runner.loss_terms:
                    warmup = max(1, settings.get('warmup_updates', 250))
                    lr = settings['learning_rate'] * min(1., (runner.updates+1)/warmup)
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                before = energy(model, runner.state).detach()
                budget = TensorBudget()
                runner.predict(budget=budget)
                # Record before optimizer update: physical accounting uses one
                # parameter version. kappa is fixed; updates do not jump x/v/E.
                values = event_observables(model, runner.state)
                predicted = (budget.get('drive_work', 0)+budget.get('trap_work', 0)
                             -budget.get('deterministic_damping_loss', 0)
                             +budget.get('ou_fluctuation_energy', 0)
                             +budget.get('drift_potential_change', 0)
                             +budget.get('collision_energy_error', 0))
                values.update(budget)
                values['energy_change'] = values['energy']-before
                values['accounting_residual'] = values['energy_change']-predicted
                values['parameter_update_energy_jump'] = before.new_zeros(())
                keys = list(values)
                pending_tensors.append(torch.stack([values[k].detach() for k in keys]))
                token = int(train[cursor])
                ce = runner.observe(token)
                pending_rows.append({'event': runner.events, 'update_before_event': (runner.events-1)//interval,
                                     'phase_time': runner.state.time, 'ce': ce, 'token_id': token})
                if runner.loss_terms:
                    continue
                tensor_rows = torch.stack(pending_tensors).cpu().tolist()
                for row, numbers in zip(pending_rows, tensor_rows):
                    row.update(zip(keys, numbers))
                    if not all(math.isfinite(v) for v in row.values()):
                        raise FloatingPointError('Nonfinite event observation')
                    log.write(json.dumps(row, allow_nan=False)+'\n')
                all_rows.extend(pending_rows)
                maximum_residual = max(maximum_residual, max(abs(r['accounting_residual']) for r in pending_rows))
                peak = torch.cuda.max_memory_allocated()/2**20 if args.device == 'cuda' else 0.
                status = {'status': 'running', 'update': runner.updates, 'target_updates': args.updates,
                          'event': runner.events, 'window_nll': float(np.mean([r['ce'] for r in pending_rows])),
                          'cumulative_prequential_nll': runner.total_nll/runner.events,
                          'energy': pending_rows[-1]['energy'], 'gamma_mean': pending_rows[-1]['gamma_mean'],
                          'gradient_norm_before_clip': runner.last_gradient_norm, 'learning_rate': lr,
                          'peak_allocated_mb': peak, 'seconds': time.perf_counter()-started,
                          'new_tokens_per_second': (runner.events-initial_events)/(time.perf_counter()-started),
                          'max_abs_accounting_residual': max(abs(r['accounting_residual']) for r in pending_rows)}
                pending_rows, pending_tensors = [], []
                write_json(args.output/'progress.json', status)
                update_log.write(json.dumps(status, allow_nan=False)+'\n')
                if runner.updates % 10 == 0 or runner.updates == args.updates:
                    print(json.dumps(status), flush=True)
                checkpoint_due = args.checkpoint_every and runner.updates % args.checkpoint_every == 0
                diagnostic_due = not args.profile and args.diagnose_every and runner.updates % args.diagnose_every == 0
                if checkpoint_due or diagnostic_due or runner.updates in (1, args.updates):
                    atomic_checkpoint(runner, args.output/'last.pt', metadata)
                if diagnostic_due or (not args.profile and runner.updates == args.updates):
                    atomic_checkpoint(runner, args.output/f'age_{runner.updates:06d}.pt', metadata)
                    val = validate(model, validation, manifest['bos_token'], args.validation_tokens)
                    val.update(update=runner.updates, event=runner.events)
                    validations.append(val)
                    write_json(args.output/f'validation_{runner.updates:06d}.json', val)
                    if args.response_tokens:
                        try:
                            response = checkpoint_response(model, runner, train[runner.events:], args.response_tokens)
                        except FloatingPointError as error:
                            response = {'status': 'unresolved_measurement', 'error': str(error),
                                        'interpretation': 'No response-rate estimate; main individual continues'}
                        write_json(args.output/f'response_{runner.updates:06d}.json', response)
                    write_json(args.output/f'temporal_{runner.updates:06d}.json', temporal_windows(list(all_rows)))
            status.update(status='budget_complete_not_convergence_claim', validations=validations,
                          max_abs_accounting_residual=maximum_residual,
                          seconds=time.perf_counter()-started,
                          peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20 if args.device == 'cuda' else 0.)
            write_json(args.output/'summary.json', status)
            write_json(args.output/'progress.json', status)
            write_json(args.output/'temporal_windows.json', temporal_windows(list(all_rows)))
    except BaseException as error:
        write_json(args.output/'failure.json', {'error': str(error), 'type': type(error).__name__,
                                              'events': runner.events, 'updates': runner.updates,
                                              'recovery': 'Resume last.pt; never reset phase or silently change configuration'})
        raise
    finally:
        stop.set()
        if sampler.is_alive():
            sampler.join(timeout=6)


if __name__ == '__main__':
    main()
