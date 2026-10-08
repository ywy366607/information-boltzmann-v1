"""Measure actual AdamW updates on a short continuation of real OWT.

This calibrates optimizer displacement/resources, not language capability.
Weights, physical state and existing Adam moments come from one completed
individual. Newly Adam-optimized edge/projection moments start at zero.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState, FlyBPTTLearner, FlyBPTTGraph


def displacement(parameter, before, gradient, *, old_sgd_lr=None, weight_decay=0., edge_bounds=False):
    """Bound temporaries to 262,144 coordinates; measure represented updates."""
    p, a, g = parameter.detach().flatten(), before.flatten(), gradient.detach().flatten()
    powers = torch.zeros(4, device=p.device, dtype=torch.float64)
    counts = torch.zeros(4, device=p.device, dtype=torch.int64)
    for start in range(0, p.numel(), 262144):
        stop = start + 262144
        x, b, grad = p[start:stop], a[start:stop], g[start:stop]
        delta = x-b
        powers[0] += b.square().sum(dtype=torch.float64)
        powers[1] += delta.square().sum(dtype=torch.float64)
        powers[2] += grad.square().sum(dtype=torch.float64)
        counts[0] += (delta != 0).sum()
        counts[1] += (grad != 0).sum()
        if old_sgd_lr is not None:
            direction = grad.add(b, alpha=weight_decay) if weight_decay else grad
            # Match torch SGD's add_ operation, including float32 rounding.
            candidate = b.clone().add_(direction, alpha=-old_sgd_lr)
            if edge_bounds:
                candidate.clamp_(0., 5.)
            represented = candidate-b
            powers[3] += represented.square().sum(dtype=torch.float64)
            counts[2] += (represented != 0).sum()
            counts[3] += ((grad != 0) & (represented == 0)).sum()
    w2, d2, g2, old2 = powers.cpu().tolist()
    changed, nonzero, old_changed, old_lost = counts.cpu().tolist()
    out = {'elements': p.numel(), 'weight_l2': w2**.5, 'update_l2': d2**.5,
        'relative_l2_update': (d2/max(w2,1e-30))**.5,
        'clipped_gradient_l2': g2**.5, 'fraction_changed': changed/p.numel(),
        'fraction_nonzero_gradient': nonzero/p.numel()}
    if old_sgd_lr is not None:
        out.update(old_sgd_lr=old_sgd_lr, old_sgd_update_l2=old2**.5,
            old_sgd_fraction_changed=old_changed/p.numel(),
            old_sgd_nonzero_gradient_but_no_represented_update_fraction=old_lost/max(nonzero,1),
            adamw_to_old_sgd_update_l2_ratio=(d2/old2)**.5 if old2 else None)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('results/q8_fly_bptt32_continuous_100k/last.pt'))
    parser.add_argument('--output', type=Path, default=Path('results/fly_bptt32_adamw_update_scale'))
    parser.add_argument('--updates', type=int, default=8)
    parser.add_argument('--window', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    args = parser.parse_args()
    if min(args.updates, args.window, args.lr) <= 0:
        parser.error('Updates, window and learning rate must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(3900/4096)
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    cfg, old = saved['config'], saved['learner']
    if saved['format'] != 'fly-bptt-v1':
        raise ValueError('Complete BPTT lifecycle checkpoint required')
    model = FlyReservoirLM(ROOT/cfg['graph'], vocab_size=50257, d_model=cfg['d_model'],
        injection='topographic', read_surface='output', synapse_model='coba',
        use_alif=True, use_stp=True, decoder_bias=cfg['decoder_bias'])
    with torch.no_grad():
        model.load_state_dict({k:v for k,v in saved['model'].items()
            if k not in ('edge_weight_e','edge_weight_i')}, strict=True)
        for name in ('edge_weight_e','edge_weight_i'):
            getattr(model,name).copy_(saved['model'][name])
    model = model.cuda()
    state = FlyPhysicalState(**{k:tuple(t.cuda() for t in v) if k=='ring' else v.cuda()
        for k,v in old['physical'].items()})
    learner = FlyBPTTLearner(model, state, lr=args.lr, adam_names=old['adam_names'],
                           settle_ticks=old.get('settle_ticks', 0),
                           writer_baseline_clock=old.get('writer_baseline_clock', 'physical'))
    learner.load_adam_state(old['optimizer'])
    for group in learner.optimizer.param_groups:
        group.update(lr=args.lr, fused=True)
    for value in learner.optimizer.state.values():
        if isinstance(value.get('step'),torch.Tensor): value['step']=value['step'].cuda()
    if old.get('plasticity_optimizer_kind','sgd') == 'adamw':
        learner.sgd.load_state_dict(old['sgd'])
        for group in learner.sgd.param_groups: group.update(lr=args.lr, fused=True)
    for name in ('events','updates','previous_token','ema'):
        setattr(learner,name,old[name])
    learner.physical_ticks = old.get('physical_ticks', old['events'])
    learner.latent_window.copy_(old['latent_window'])
    cursor = saved['train_cursor']
    report = {'scope':'short actual-update/resource calibration, not capability/convergence evidence',
        'checkpoint':str(args.checkpoint), 'starting_train_cursor':cursor,
        'starting_events':learner.events, 'starting_updates':learner.updates,
        'lr':args.lr,'window':args.window,'planned_updates':args.updates,
        'optimizer':'AdamW for every currently trainable parameter',
        'existing_adam_moments':'retained','new_plasticity_moments':'zero if source used SGD',
        'fixed_parameters':cfg['fixed_parameters'], 'rows':[]}
    del old, saved
    gc.collect()
    torch.cuda.empty_cache()
    learner.runner = FlyBPTTGraph(learner,args.window)
    named = {n:p for n,p in model.named_parameters() if p.requires_grad}
    originals = {n:p.detach().clone() for n,p in named.items()}
    before = {n:p.detach().clone() for n,p in named.items()}
    gradients = {}
    def capture_gradients(optimizer, positional, keywords):
        gradients.update({n:p.grad for n,p in named.items()})
    hook = learner.optimizer.register_step_pre_hook(capture_gradients)
    train = np.load(ROOT/cfg['data']/'train.npy',mmap_mode='r')
    setup_peak = torch.cuda.max_memory_allocated()/2**20
    try:
        for step in range(args.updates):
            with torch.no_grad():
                for name,p in named.items(): before[name].copy_(p)
            targets = np.array(train[cursor+1:cursor+args.window+1],dtype=np.int64)
            if len(targets)!=args.window: raise ValueError('Fresh training stream exhausted')
            torch.cuda.synchronize()
            started = time.perf_counter()
            scores, metrics = learner.observe(targets)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            per_parameter = {}
            with torch.no_grad():
                for name,p in named.items():
                    old_lr = 1e-5 if name.startswith('edge_weight_') else (
                        1e-4 if name.startswith('topographic_writer.proj_') else None)
                    per_parameter[name] = displacement(p,before[name],gradients[name],
                        old_sgd_lr=old_lr, weight_decay=1e-4 if 'proj_' in name else 0.,
                        edge_bounds=name.startswith('edge_weight_'))
            cursor += args.window
            row = {'update':step+1,'train_cursor':cursor,'pre_update_nll':float(np.mean(scores)),
                'training_seconds':elapsed,'training_tokens_per_second':args.window/elapsed,
                **metrics,'parameters':per_parameter,
                'allocated_mib':torch.cuda.memory_allocated()/2**20,
                'reserved_mib':torch.cuda.memory_reserved()/2**20,
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
            report['rows'].append(row)
            (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
            print({k:v for k,v in row.items() if k!='parameters'},flush=True)
            for name in ('edge_weight_e','edge_weight_i','topographic_writer.proj_chemo.weight','output_read.weight'):
                print(name,per_parameter[name],flush=True)
            if max(row['reserved_mib'],row['peak_allocated_mib'])>3900:
                raise MemoryError('Calibration exceeded the 3900 MiB dedicated CUDA limit')
        with torch.no_grad():
            report['cumulative_displacements'] = {name:displacement(p,originals[name],gradients[name]) for name,p in named.items()}
        report.update(status='completed',actual_updates=args.updates,actual_targets=args.updates*args.window,
            ending_events=learner.events,ending_updates=learner.updates,setup_peak_mib=setup_peak)
    finally:
        hook.remove()
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    main()
