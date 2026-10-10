"""Where does the champion's time and GPU memory go?  (diagnostic; fixed weights for inference, one real window for training)

Inference: adaptive admission, no grad, real validation tokens after a short burn-in.
Training: one BPTT window through the champion's own learner (checkpoint recompute + backward), no optimizer side effects
kept (the learner is a throwaway fork in memory; nothing is saved).
Timers (CUDA-synchronized, so they add a small overhead): synaptic transmission forward / backward, the rest of each
physical tick, the adaptive stop decision (host sync), read + decoder.  Also counts, per tick, the fraction of neurons that
spike and the fraction of edges whose presynaptic neuron spiked: the upper bound on what event-driven transmission saves.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import information_boltzmann.core.triton_synapse as ts  # noqa: E402
import information_boltzmann.core.fly_bptt_learning as fl  # noqa: E402
from information_boltzmann.core.fly_reservoir import FlyReservoirLM  # noqa: E402
from scripts.ib.inspect_fly_bptt_credit import move_state  # noqa: E402

T = defaultdict(float)
N = defaultdict(int)
ACT = {'spike_frac': [], 'edge_frac': []}


def timed(name, fn):
    def wrapper(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize(); T[name] += time.perf_counter() - t; N[name] += 1
        return out
    return wrapper


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, default=Path('E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt'))
    ap.add_argument('--tokens', type=int, default=32)
    ap.add_argument('--burn-in', type=int, default=8)
    ap.add_argument('--train', action='store_true', help='also profile one training window')
    args = ap.parse_args()
    dev = 'cuda'
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    cfg, L = saved['config'], saved['learner']
    model = FlyReservoirLM(ROOT / cfg['graph'], vocab_size=50257, d_model=cfg['d_model'], injection='topographic',
                           read_surface='output', synapse_model='coba', use_alif=True, use_stp=True,
                           decoder_bias=cfg['decoder_bias'], detach_reset=cfg.get('detach_reset', False),
                           surrogate_mode=cfg.get('surrogate_mode', 'absolute'),
                           transmission_mode=cfg.get('transmission_mode', 'atomic'))
    edges = ('edge_weight_e', 'edge_weight_i')
    model.load_state_dict({k: v for k, v in saved['model'].items() if k not in edges}, strict=False)
    model = model.to(dev)
    buffers = dict(model.named_buffers())
    with torch.no_grad():
        for k in edges:
            buffers[k].copy_(saved['model'][k])
    out_degree = torch.bincount(torch.cat([buffers['edge_pre_e'].long(), buffers['edge_pre_i'].long()]),
                                minlength=model.n_neurons).float()
    total_edges = float(out_degree.sum())

    # instrument
    ts.IncomingDelayedTransmission.forward = staticmethod(timed('transmission_forward', ts.IncomingDelayedTransmission.forward))
    ts.IncomingDelayedTransmission.backward = staticmethod(timed('transmission_backward', ts.IncomingDelayedTransmission.backward))
    original_tick = fl.step_fly_physical_tick
    def tick(model_, current, *a, **k):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = original_tick(model_, current, *a, **k)
        torch.cuda.synchronize(); T['tick_total'] += time.perf_counter() - t; N['tick_total'] += 1
        with torch.no_grad():
            s = (out.ring[0] > 0).float().flatten()
            ACT['spike_frac'].append(float(s.mean()))
            ACT['edge_frac'].append(float((s * out_degree).sum() / total_edges))
        return out
    fl.step_fly_physical_tick = tick

    val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
    cursor = int(saved['val_cursor'])
    count = args.burn_in + args.tokens
    targets = np.array(val[cursor:cursor + count], dtype=np.int64)
    inputs = np.concatenate(([L['previous_token']], targets[:-1]))
    clock = L.get('writer_baseline_clock', 'physical')
    kw = dict(min_settle_ticks=int(L.get('min_settle_ticks', 3)), max_settle_ticks=int(L.get('settle_ticks', 14)) or 14,
              flux_baseline=float(L.get('flux_baseline', 0.048)), writer_baseline_clock=clock)

    # ---------- inference ----------
    model.eval().requires_grad_(False)
    state = move_state(fl.FlyPhysicalState(**L['physical']), dev)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(),
                    conductance_gains=model.get_conductance_gains(), alif_params=model.get_alif_params(),
                    stp_params=model.get_stp_params())
        for i, tok in enumerate(inputs):
            if i == args.burn_in:
                torch.cuda.synchronize(); T.clear(); N.clear(); ACT['spike_frac'].clear(); ACT['edge_frac'].clear()
                start = time.perf_counter()
            token = torch.tensor([int(tok)], device=dev)
            state, z = fl.advance_fly_token_adaptive(model, state, token, [None], **kw, **opts)
            torch.cuda.synchronize(); t = time.perf_counter()
            logits = model.decoder(model.read_norm(z))
            F.cross_entropy(logits.float(), torch.tensor([int(targets[i])], device=dev))
            torch.cuda.synchronize(); T['read_decode'] += time.perf_counter() - t
        torch.cuda.synchronize(); wall = time.perf_counter() - start
    ticks = N['tick_total']
    print(f'=== inference: {args.tokens} tokens, {ticks} ticks ({ticks / args.tokens:.2f}/token), {wall:.2f} s '
          f'({args.tokens / wall:.2f} tok/s), peak GPU {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB')
    trans = T['transmission_forward']
    print(f'  transmission forward {trans:.3f} s ({100 * trans / wall:.0f}%), other tick work {T["tick_total"] - trans:.3f} s '
          f'({100 * (T["tick_total"] - trans) / wall:.0f}%), read+decode {T["read_decode"]:.3f} s ({100 * T["read_decode"] / wall:.0f}%), '
          f'remaining (stop decision, writer, host) {wall - T["tick_total"] - T["read_decode"]:.3f} s')
    print(f'  per tick: spiking neurons {100 * np.mean(ACT["spike_frac"]):.2f}% , edges with a spiking presynaptic neuron '
          f'{100 * np.mean(ACT["edge_frac"]):.2f}% (max {100 * np.max(ACT["edge_frac"]):.2f}%)')

    if not args.train:
        return
    # ---------- training window ----------
    T.clear(); N.clear(); ACT['spike_frac'].clear(); ACT['edge_frac'].clear()
    model.train().requires_grad_(True)
    names = L['adam_names']
    learner = fl.FlyBPTTLearner(model, move_state(fl.FlyPhysicalState(**L['physical']), dev), adam_names=names, lr=2e-4,
                                plasticity_optimizer='adamw', settle_ticks=int(L.get('settle_ticks', 14)),
                                writer_baseline_clock=clock, use_ctm_loss=bool(L.get('use_ctm_loss', True)),
                                use_checkpointing=True, adaptive_admission=True,
                                min_settle_ticks=kw['min_settle_ticks'], flux_baseline=kw['flux_baseline'])
    learner.previous_token = int(L['previous_token'])
    window = torch.as_tensor(targets[args.burn_in:args.burn_in + 32], device=dev)
    learner.observe(window.cpu().numpy())              # warm-up window (compilation, allocator); not timed
    T.clear(); N.clear()
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    start = time.perf_counter()
    learner.observe(window.cpu().numpy())
    torch.cuda.synchronize(); wall = time.perf_counter() - start
    fwd, bwd = T['transmission_forward'], T['transmission_backward']
    print(f'=== training window: 32 tokens, {N["tick_total"]} tick calls (incl. recompute), {wall:.2f} s, '
          f'peak GPU {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB')
    print(f'  transmission forward {fwd:.3f} s ({100 * fwd / wall:.0f}%) over {N["transmission_forward"]} calls, '
          f'backward {bwd:.3f} s ({100 * bwd / wall:.0f}%) over {N["transmission_backward"]} calls, '
          f'other tick work {T["tick_total"] - fwd:.3f} s, rest (backward of the rest, optimizer, host) {wall - T["tick_total"] - bwd:.3f} s')


if __name__ == '__main__':
    main()
