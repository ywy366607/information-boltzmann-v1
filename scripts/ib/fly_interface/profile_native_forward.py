"""Where the time of one frozen native word goes (no exposure, no learning).

Restores the full individual exactly as the trainer does, then advances one word
on the saved pending clock under cProfile (and, on CUDA, synchronized wall time
per native mechanism). Reports the top functions by own time and cumulative time,
wall seconds per tick, and peak memory. Diagnostic only; the source is untouched.
"""
from __future__ import annotations

import argparse
import cProfile
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import pstats
import sys
import time

import psutil
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.ib.fly_interface.native_causal_audit import restore_native, pending_queue_tape, clone_state
from information_boltzmann.core.fly_native_generative import _preserve_rollout
from information_boltzmann.core import fly_bptt_learning as fl

TIMED = ('advance_kc_terminals', 'advance_sensory_terminals', 'advance_mb_terminals', 'advance_apl_cable',
         'advance_dan_release', 'sensory_delivery', 'step_fly_physical_tick', 'checkpoint_native_spatial')


def run(checkpoint: Path, output: Path, device: str, ticks: int | None):
    output.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process()
    started = time.perf_counter()
    learner, saved, args, changed = restore_native(checkpoint, output, frozen=True, device=device)
    restore_seconds = time.perf_counter() - started
    if changed:
        raise ValueError('Profile must not migrate the learning algorithm')
    model = learner.model
    tensors = [(f'buffer:{n}', b) for n, b in model.named_buffers()] + [(f'param:{n}', q) for n, q in model.named_parameters()]
    by_dtype = {}
    for _, t in tensors:
        by_dtype[str(t.dtype)] = by_dtype.get(str(t.dtype), 0) + t.numel() * t.element_size()
    largest = sorted(((n, t.numel() * t.element_size(), str(t.dtype)) for n, t in tensors), key=lambda r: -r[1])[:40]
    memory = {'model_mib_by_dtype': {k: round(v / 2 ** 20, 1) for k, v in by_dtype.items()},
              'largest_model_tensors_mib': [(n, round(b / 2 ** 20, 1), d) for n, b, d in largest]}
    tape = pending_queue_tape(saved, 1, 1)[0]
    job = saved['learner'].get('native_context_state', {}).get('pending_teaching') or {}
    source = job.get('source', tape['forced']['sources'][0])
    duration = int(ticks or tape['forced']['intervals'][0])
    wall = {name: 0.0 for name in TIMED}
    calls = {name: 0 for name in TIMED}
    originals = {}

    def wrap(name, fn):
        def timed(*a, **k):
            if device == 'cuda':
                torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn(*a, **k)
            if device == 'cuda':
                torch.cuda.synchronize()
            wall[name] += time.perf_counter() - t
            calls[name] += 1
            return out
        return timed

    for name in TIMED:
        if hasattr(fl, name):
            originals[name] = getattr(fl, name)
            setattr(fl, name, wrap(name, originals[name]))
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    rss_before = proc.memory_info().rss
    profiler = cProfile.Profile()
    try:
        with _preserve_rollout(model), torch.no_grad():
            current = clone_state(learner.state)
            current = fl.prepare_sensory_expectation(model, current, learner.previous_token, source)
            t = time.perf_counter()
            profiler.enable()
            out = fl.advance_fly_token_stream(
                model, current, torch.tensor([int(learner.previous_token)], device=current.h.device),
                [duration], ticks=duration, hold_input=learner.hold_input,
                writer_baseline_clock=learner.writer_baseline_clock, fresh=True, sensory_on=True,
                voice=tape['last_voice'], executed_action=source == 'self_hearing',
                reward_ticks=fl.reward_ticks_left(model, tape['last_reward'], 0),
                ai_g=learner.ai_g.detach().clone(), ai_tape=[False] * duration,
                read_every=True, eye_forced=None, update_expectation=False)
            if device == 'cuda':
                torch.cuda.synchronize()
            profiler.disable()
            word_seconds = time.perf_counter() - t
    finally:
        for name, fn in originals.items():
            setattr(fl, name, fn)
    def table(key, n=45):
        buffer = io.StringIO()
        pstats.Stats(profiler, stream=buffer).sort_stats(key).print_stats(n)
        return buffer.getvalue()
    own, cumulative = table('tottime'), table('cumulative')
    (output / 'profile_tottime.txt').write_text(own, encoding='utf-8')
    (output / 'profile_cumulative.txt').write_text(cumulative, encoding='utf-8')
    report = {'at': datetime.now(timezone.utc).isoformat(), 'checkpoint': str(checkpoint), 'device': device,
              'ticks': duration, 'source': source, 'restore_seconds': restore_seconds,
              'word_seconds': word_seconds, 'seconds_per_tick': word_seconds / duration,
              'mechanism_wall_seconds': {k: round(v, 3) for k, v in wall.items() if calls[k]},
              'mechanism_calls': {k: v for k, v in calls.items() if v},
              'rss_gb_before_word': rss_before / 1e9, 'rss_gb_after_word': proc.memory_info().rss / 1e9,
              'cuda_peak_mib': (torch.cuda.max_memory_allocated() / 2 ** 20) if device == 'cuda' else None,
              'memory': memory, 'scope': 'Frozen single-word timing; no learning, no exposure'}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=1), flush=True)
    print(own[:6000])


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--ticks', type=int, default=None)
    a = p.parse_args()
    run(a.checkpoint, a.output, a.device, a.ticks)
