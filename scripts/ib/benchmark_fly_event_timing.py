"""Measure event-timing execution on matched mature OWT checkpoint forks.

This is a numerical/resource calibration, not primary active evaluation or a
capability comparison. No checkpoint, production state or cursor is overwritten.
Each timing uses the same saved weights, physical state and optimizer moments.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState, FlyBPTTLearner, FlyBPTTGraph


def measure(saved, settle_ticks, windows, window, vram_limit, profile=False):
    cfg, old = saved['config'], saved['learner']
    torch.cuda.reset_peak_memory_stats()
    model = FlyReservoirLM(ROOT/cfg['graph'], vocab_size=50257,
        d_model=cfg['d_model'], injection='topographic', read_surface='output',
        synapse_model='coba', use_alif=True, use_stp=True,
        decoder_bias=cfg['decoder_bias']).cuda()
    with torch.no_grad():
        model.load_state_dict({key: value for key, value in saved['model'].items()
                              if key not in ('edge_weight_e', 'edge_weight_i')}, strict=True)
        for key in ('edge_weight_e', 'edge_weight_i'):
            getattr(model, key).copy_(saved['model'][key])
    state = FlyPhysicalState(**{key: tuple(t.cuda() for t in value) if key == 'ring'
                              else value.cuda() for key, value in old['physical'].items()})
    learner = FlyBPTTLearner(model, state, lr=cfg['lr'], lr_decoder=cfg.get('lr_decoder'),
        lr_synapse=cfg['lr_synapse'], lr_sensory=cfg['lr_sensory'],
        adam_names=old['adam_names'], plasticity_optimizer=old['plasticity_optimizer_kind'],
        settle_ticks=settle_ticks, writer_baseline_clock=old.get('writer_baseline_clock', 'physical'))
    learner.load_adam_state(old['optimizer'])
    learner.sgd.load_state_dict(old['sgd'])
    for key in ('events', 'updates', 'previous_token', 'ema'):
        setattr(learner, key, old[key])
    learner.physical_ticks = old.get('physical_ticks', old['events'])
    learner.latent_window.copy_(old['latent_window'])
    before_events, before_ticks, before_updates = learner.events, learner.physical_ticks, learner.updates
    setup_start = time.perf_counter()
    learner.runner = FlyBPTTGraph(learner, window)
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter()-setup_start
    data = np.load(ROOT/cfg['data']/'train.npy', mmap_mode='r')
    cursor = saved['train_cursor']
    rows = []
    for index in range(windows):
        targets = np.asarray(data[cursor+1+index*window:cursor+1+(index+1)*window], dtype=np.int64)
        started = time.perf_counter()
        scores, gradient = learner.observe(targets)
        torch.cuda.synchronize()
        duration = time.perf_counter()-started
        peak = torch.cuda.max_memory_allocated()/2**20
        reserved = torch.cuda.memory_reserved()/2**20
        if max(peak, reserved) > vram_limit:
            raise MemoryError(f'{max(peak, reserved):.1f} MiB exceeded {vram_limit} MiB')
        rows.append({'window_index': index, 'seconds': duration,
                     'numeric_preupdate_nll': float(np.mean(scores)),
                     'grad_norm_before_clip': gradient['grad_norm_before_clip']})
        print(f'settle={settle_ticks}: window {index+1}, {duration:.3f}s, peak {peak:.0f} MiB', flush=True)
    median = statistics.median(row['seconds'] for row in rows)
    result = {'settle_ticks': settle_ticks, 'physical_ticks_per_input': 1+settle_ticks,
              'windows': rows, 'median_seconds_per_window': median,
              'tokens_per_second': window/median, 'setup_seconds': setup_seconds,
              'vram_peak_mib': torch.cuda.max_memory_allocated()/2**20,
              'vram_reserved_mib': torch.cuda.memory_reserved()/2**20,
              'input_events_executed': learner.events-before_events,
              'physical_ticks_executed': learner.physical_ticks-before_ticks,
              'optimizer_updates_executed': learner.updates-before_updates}
    if profile:
        targets = np.asarray(data[cursor+1+windows*window:cursor+1+(windows+1)*window], dtype=np.int64)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profiler:
            learner.observe(targets)
            torch.cuda.synchronize()
        kernels = [event for event in profiler.events()
                   if event.device_type == torch.autograd.DeviceType.CUDA]
        grouped = {}
        for event in kernels:
            row = grouped.setdefault(event.name, {'name': event.name, 'calls': 0, 'cuda_us': 0.})
            row['calls'] += 1
            row['cuda_us'] += event.device_time_total
        result['profile_scope'] = 'one additional diagnostic update; excluded from timed windows'
        result['cuda_kernels'] = sorted(grouped.values(), key=lambda row: row['cuda_us'], reverse=True)[:30]
        print(json.dumps({'settle_ticks': settle_ticks, 'top_kernels': result['cuda_kernels'][:8]}), flush=True)
    learner.runner = None
    del learner, model, state
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('results/q8_fly_bptt32_adamw_continuous_100k/last.pt'))
    parser.add_argument('--windows', type=int, default=3)
    parser.add_argument('--window', type=int, default=32)
    parser.add_argument('--vram-limit-mib', type=float, default=3900)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--output', type=Path,
                        default=Path('results/published/fly_event_timing_execution.json'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    report = {'scope': 'matched checkpoint forks; numerical/resource calibration only; production stays paused',
              'checkpoint': str(args.checkpoint), 'checkpoint_bptt_tokens': saved['bptt_train_tokens'],
              'checkpoint_train_cursor': saved['train_cursor'], 'window_tokens': args.window,
              'arms': [measure(saved, ticks, args.windows, args.window, args.vram_limit_mib, args.profile)
                       for ticks in (0, 1)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
