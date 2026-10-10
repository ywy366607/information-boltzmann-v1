"""Causal audit of the CTM-S14 champion with adaptive admission (fixed weights, paired, no reset).

Every arm starts from the same saved full physical life, runs the same real unscored burn-in on the fresh validation
stream, then scores the same next tokens through the champion's own read path (adaptive admission: terminal settled
motor latent -> read_norm -> decoder). Arms change exactly one mechanism during burn-in and scoring:
  full              adaptive admission as trained (3..14 quiet ticks)
  fixed_K           K quiet ticks for every token (K = 14, 6, 3, 0); terminal read
  no_inhibition     inhibitory edge weights zeroed
  no_edges          all connectome edges zeroed (sensory drive cannot reach the motor surface)
  no_alif           ALIF adaptation strength beta = 0
  static_stp        STP facilitation/recovery instantaneous (u = u0, x = 1 every tick)
  no_memory         before each token the state is reset to the scoring-start state (no carried history)
  rollback_edges    learned connectome weights restored to their birth values (graph file); readout/decoder stay learned
  rollback_physiology  learned thresholds, membrane/synaptic time constants, ALIF and conductance gains restored to birth
  rollback_brain    both rollbacks together
Reports mean NLL, paired delta vs full with 128-token block means, and the mean quiet ticks used.
This is a mechanism diagnostic of one checkpoint, not active evaluation or a training comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_reservoir import FlyReservoirLM  # noqa: E402
from information_boltzmann.core.fly_bptt_learning import (  # noqa: E402
    FlyPhysicalState, advance_fly_token_adaptive, advance_fly_input_event, extract_fly_motor_latent)
from scripts.ib.inspect_fly_bptt_credit import move_state  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, default=Path('E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt'))
    ap.add_argument('--burn-in', type=int, default=256)
    ap.add_argument('--score-tokens', type=int, default=768)
    ap.add_argument('--arms', default='full,fixed_14,fixed_6,fixed_3,fixed_0,no_inhibition,no_edges,no_alif,static_stp,no_memory')
    ap.add_argument('--out', type=Path, default=Path('results/q8_fly_ctm_settle14_100k/champion_causality.json'))
    args = ap.parse_args()
    torch.set_num_threads(4)
    dev = 'cuda'
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    cfg, L = saved['config'], saved['learner']
    model = FlyReservoirLM(ROOT / cfg['graph'], vocab_size=50257, d_model=cfg['d_model'], injection='topographic',
                           read_surface='output', synapse_model='coba', use_alif=True, use_stp=True,
                           decoder_bias=cfg['decoder_bias'], detach_reset=cfg.get('detach_reset', False),
                           surrogate_mode=cfg.get('surrogate_mode', 'absolute'),
                           transmission_mode=cfg.get('transmission_mode', 'atomic'))
    edges = ('edge_weight_e', 'edge_weight_i')
    result = model.load_state_dict({k: v for k, v in saved['model'].items() if k not in edges}, strict=False)
    if result.unexpected_keys or [k for k in result.missing_keys if not k.startswith(('graph_observer', 'latent_predictor'))]:
        raise ValueError(f'checkpoint/model mismatch: {result}')
    model = model.to(dev).eval().requires_grad_(False)
    tensors = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    for name in edges:                                       # trained connectome weights (non-persistent buffers)
        if tensors[name].shape != saved['model'][name].shape:
            raise ValueError(f'{name} shape mismatch')
        tensors[name].copy_(saved['model'][name])
    initial = FlyPhysicalState(**L['physical'])
    val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
    cursor = int(saved['val_cursor'])
    count = args.burn_in + args.score_tokens
    targets = np.array(val[cursor:cursor + count], dtype=np.int64)
    if len(targets) != count:
        raise ValueError('Fresh validation stream exhausted')
    inputs = np.concatenate(([L['previous_token']], targets[:-1]))
    clock = L.get('writer_baseline_clock', 'physical')
    min_ticks, flux = int(L.get('min_settle_ticks', 3)), float(L.get('flux_baseline', 0.048))
    max_ticks = int(L.get('settle_ticks', 14)) or 14
    report = {'scope': __doc__.split('\n')[0], 'checkpoint': str(args.checkpoint), 'bptt_train_tokens': int(saved['bptt_train_tokens']),
              'validation_cursor': cursor, 'burn_in': args.burn_in, 'score_tokens': args.score_tokens, 'arms': {}}
    reference = None
    physiology = ['log_threshold', 'log_tau_m', 'log_tau_s_e', 'log_tau_s_i', 'log_beta', 'log_tau_a', 'log_g_e', 'log_g_i']
    birth = None
    if any(a.startswith('rollback') for a in args.arms.split(',')):
        torch.manual_seed(0)                              # birth values: graph-file edges and constructor physiology
        fresh = FlyReservoirLM(ROOT / cfg['graph'], vocab_size=50257, d_model=cfg['d_model'], injection='topographic',
                               read_surface='output', synapse_model='coba', use_alif=True, use_stp=True,
                               decoder_bias=cfg['decoder_bias'], detach_reset=cfg.get('detach_reset', False),
                               surrogate_mode=cfg.get('surrogate_mode', 'absolute'),
                               transmission_mode=cfg.get('transmission_mode', 'atomic'))
        fb = {**dict(fresh.named_parameters()), **dict(fresh.named_buffers())}
        birth = {k: fb[k].detach().clone() for k in list(edges) + physiology}
        del fresh, fb

    def params(arm):
        rates, thr = model.get_decay_rates(), model.get_thresholds()
        gains, alif, stp = model.get_conductance_gains(), model.get_alif_params(), model.get_stp_params()
        if arm == 'no_alif':
            alif = (alif[0], torch.zeros_like(alif[1]))
        if arm == 'static_stp':
            stp = (stp[0], torch.zeros_like(stp[1]), torch.zeros_like(stp[2]), stp[3])
        return dict(base_rates=rates, thresholds=thr, conductance_gains=gains, alif_params=alif, stp_params=stp)

    with torch.no_grad():
        for arm in args.arms.split(','):
            originals = {}
            rollback = {'rollback_edges': list(edges), 'rollback_physiology': physiology,
                        'rollback_brain': list(edges) + physiology}.get(arm, [])
            for name in rollback:
                originals[name] = tensors[name].detach().clone()
                tensors[name].copy_(birth[name].to(tensors[name]))
            if arm in ('no_inhibition', 'no_edges'):
                for name in (['edge_weight_i'] if arm == 'no_inhibition' else ['edge_weight_e', 'edge_weight_i']):
                    originals[name] = tensors[name].clone(); tensors[name].zero_()
            options = params(arm)
            state = move_state(initial, dev)
            anchor = None
            scores, ticks = [], []
            start = time.perf_counter()
            for i, tok in enumerate(inputs):
                token = torch.tensor([int(tok)], device=dev)
                if arm == 'no_memory' and i >= args.burn_in:
                    if anchor is None:
                        anchor = move_state(state, dev)
                    state = move_state(anchor, dev)
                if arm.startswith('fixed_'):
                    k = int(arm.split('_')[1])
                    state = advance_fly_input_event(model, state, token, settle_ticks=k, writer_baseline_clock=clock, **options)
                    z = extract_fly_motor_latent(model, state)
                    used = k
                else:
                    box = [None]
                    state, z = advance_fly_token_adaptive(model, state, token, box, min_settle_ticks=min_ticks,
                                                          max_settle_ticks=max_ticks, flux_baseline=flux,
                                                          writer_baseline_clock=clock, **options)
                    used = box[0]
                if i >= args.burn_in:
                    logits = model.decoder(model.read_norm(z))
                    scores.append(float(F.cross_entropy(logits.float(), torch.tensor([int(targets[i])], device=dev))))
                    ticks.append(used)
            for name, value in originals.items():
                tensors[name].copy_(value)
            s = np.array(scores)
            if reference is None:
                reference = s.copy()
            d = s - reference
            report['arms'][arm] = {'nll': float(s.mean()), 'delta_vs_full': float(d.mean()),
                                   'delta_blocks_128': [float(d[j:j + 128].mean()) for j in range(0, len(d), 128)],
                                   'mean_quiet_ticks': float(np.mean(ticks)), 'seconds': time.perf_counter() - start}
            print(arm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in report['arms'][arm].items()
                        if k != 'delta_blocks_128'}, flush=True)
            args.out.write_text(json.dumps(report, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
