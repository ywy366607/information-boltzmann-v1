"""Paired local AdamW step diagnostics on a saved continuing individual.

This is an optimizer diagnostic, not a capability evaluation. All alternatives
use identical physical state, targets, clipped gradients and saved Adam moments.
No production weights, state or optimizer moments are modified.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import (
    FlyPhysicalState, FlyBPTTLearner, FlyBPTTGraph,
)


def clone_physical(state):
    return FlyPhysicalState(**{
        k: tuple(t.detach().clone() for t in v) if k == 'ring'
        else v.detach().clone() for k, v in state.state_dict().items()})


def parameter_group(name):
    if name.startswith('edge_weight_'):
        return 'synapse'
    if name.startswith('topographic_writer.'):
        return 'writer'
    if name.startswith(('output_read.', 'read_norm.', 'decoder.')):
        return 'read_decoder'
    return 'physiology'


def tensor_statistics(weight, gradient, original=None):
    """Avoid materializing full-size float64 parameter or gradient copies."""
    w, g = weight.detach().flatten(), gradient.detach().flatten()
    totals = torch.zeros(5, dtype=torch.float64, device=w.device)
    active = torch.zeros((), dtype=torch.int64, device=w.device)
    for start in range(0, w.numel(), 262144):
        end = start + 262144
        x, y = w[start:end], g[start:end]
        totals[0] += x.double().square().sum()
        totals[1] += y.double().square().sum()
        active += (y != 0).sum()
        if original is not None:
            old = original.flatten()[start:end].to(w.device)
            delta = (x-old).double()
            totals[2] += delta.square().sum()
            totals[3] += (y.double()*delta).sum()
            totals[4] += old.double().square().sum()
    w2, g2, d2, gd, old2 = totals.cpu().tolist()
    count = int(active.item())
    result = dict(elements=w.numel(), active_elements=count,
                  weight_l2=math.sqrt(w2), gradient_l2=math.sqrt(g2),
                  gradient_rms=math.sqrt(g2/w.numel()),
                  active_gradient_rms=math.sqrt(g2/max(count, 1)),
                  nonzero_gradient_fraction=count/w.numel())
    if original is not None:
        result.update(update_l2=math.sqrt(d2),
                      relative_l2_update=math.sqrt(d2/max(old2, 1e-30)),
                      predicted_nll_reduction=-gd,
                      descent_cosine=-gd/math.sqrt(max(g2*d2, 1e-60)))
    return result


def aggregate(records):
    grouped = {}
    for name, row in records.items():
        key = parameter_group(name)
        out = grouped.setdefault(key, dict(elements=0, active_elements=0,
                                          weight_squared=0., gradient_squared=0.))
        out['elements'] += row['elements']
        out['active_elements'] += row['active_elements']
        out['weight_squared'] += row['weight_l2']**2
        out['gradient_squared'] += row['gradient_l2']**2
        if 'update_l2' in row:
            out['update_squared'] = out.get('update_squared', 0.)+row['update_l2']**2
            out['predicted_nll_reduction'] = out.get('predicted_nll_reduction', 0.)+row['predicted_nll_reduction']
    for row in grouped.values():
        row.update(gradient_l2=math.sqrt(row['gradient_squared']),
                   gradient_rms=math.sqrt(row['gradient_squared']/row['elements']),
                   active_gradient_rms=math.sqrt(row['gradient_squared']/max(row['active_elements'], 1)),
                   nonzero_gradient_fraction=row['active_elements']/row['elements'],
                   gradient_to_weight_l2=math.sqrt(row['gradient_squared']/max(row['weight_squared'], 1e-30)))
    return grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/fly_bptt32_adamw_lr_diagnostic'))
    parser.add_argument('--windows', type=int, default=3)
    parser.add_argument('--window', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0002)
    args = parser.parse_args()
    if min(args.windows, args.window, args.lr) <= 0:
        parser.error('Budgets and rate must be positive')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(3900/4096)
    args.output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    cfg, old = saved['config'], saved['learner']
    if old.get('plasticity_optimizer_kind') != 'adamw':
        raise ValueError('This diagnostic requires the mature AdamW continuation')
    model = FlyReservoirLM(ROOT/cfg['graph'], vocab_size=50257,
        d_model=cfg['d_model'], injection='topographic', read_surface='output',
        synapse_model='coba', use_alif=True, use_stp=True, decoder_bias=cfg['decoder_bias'])
    model.load_state_dict({k:v for k,v in saved['model'].items()
                          if k not in ('edge_weight_e', 'edge_weight_i')}, strict=True)
    for name in ('edge_weight_e', 'edge_weight_i'):
        getattr(model, name).copy_(saved['model'][name])
    model = model.cuda()
    state = FlyPhysicalState(**{k:tuple(t.cuda() for t in v) if k == 'ring' else v.cuda()
                               for k,v in old['physical'].items()})
    learner = FlyBPTTLearner(model, state, adam_names=old['adam_names'], lr=args.lr,
                           lr_decoder=cfg.get('lr_decoder'),
                           settle_ticks=old.get('settle_ticks', 0),
                           writer_baseline_clock=old.get('writer_baseline_clock', 'physical'))
    named = {n:p for n,p in model.named_parameters() if p.requires_grad}
    optimizers = (learner.optimizer, learner.sgd)
    base_weights = saved['model']

    def restore(scale=1., only=None):
        with torch.no_grad():
            for name,p in named.items():
                p.copy_(base_weights[name])
        for opt, key in zip(optimizers, ('optimizer', 'sgd')):
            if key == 'optimizer':
                learner.load_adam_state(old[key])
            else:
                opt.load_state_dict(old[key])
            for group in opt.param_groups:
                base_rate = (cfg.get('lr_decoder') or args.lr) if (
                    group.get('parameter_names') == ['decoder.weight']) else args.lr
                group.update(fused=True, lr=base_rate*scale)
            for entry in opt.state.values():
                if isinstance(entry.get('step'), torch.Tensor):
                    entry['step'] = entry['step'].cuda()
        if only is not None:
            for name,p in named.items():
                # A missing gradient skips both Adam moments and weight decay.
                p.grad = gradients[name] if parameter_group(name) == only else None
        else:
            for name,p in named.items():
                if name in gradients:
                    p.grad = gradients[name]

    gradients = {}
    restore()
    learner.runner = FlyBPTTGraph(learner, args.window)
    train = np.load(ROOT/cfg['data']/'train.npy', mmap_mode='r')
    cursor, previous = saved['train_cursor'], old['previous_token']
    report = dict(scope='paired finite-step diagnostic; not active-learning capability evaluation',
        checkpoint=str(args.checkpoint), train_cursor=cursor,
        saved_events=old['events'], saved_optimizer_updates=old['updates'],
        lr=args.lr, windows=args.windows, window=args.window,
        state_policy='same saved continuing physical state, then original-weight rollout; no cold/reset state',
        optimizer_policy='same saved Adam moments and same clipped gradient per alternative',
        next_window_policy='same original-weight post-window physical state, as retained by actual learner before updating parameters',
        gradient_policy='existing ATan surrogate; finite-step response may differ from its linear prediction',
        source_checkpoint_modified=False, rows=[])
    arms = [('half', .5, None), ('current', 1., None), ('double', 2., None)]
    arms += [(key+'_only', 1., key) for key in ('writer', 'synapse', 'read_decoder', 'physiology')]

    def evaluate(initial, ids, targets):
        learner.state = initial
        with torch.no_grad():
            scores, _, _ = learner.forward_window(ids, targets)
        return float(scores.mean())

    for index in range(args.windows):
        restore()
        initial = clone_physical(state)
        values = np.array(train[cursor+1:cursor+2*args.window+1], dtype=np.int64)
        if len(values) != 2*args.window:
            raise ValueError('Fresh real OWT stream exhausted')
        ids = torch.as_tensor(np.concatenate(([previous], values[:args.window-1]))[None], device='cuda')
        targets = torch.as_tensor(values[:args.window][None], device='cuda')
        next_ids = torch.as_tensor(values[args.window-1:2*args.window-1][None], device='cuda')
        next_targets = torch.as_tensor(values[args.window:][None], device='cuda')
        learner.state = initial
        scores, baseline_state, _ = learner.runner.replay(ids, targets)
        baseline_loss = float(scores.detach().mean())
        baseline_state = clone_physical(baseline_state)
        gradients = {name:p.grad for name,p in named.items()}
        raw = {name:tensor_statistics(p, p.grad) for name,p in named.items()}
        norm = float(torch.nn.utils.clip_grad_norm_(learner.trainable, 1., error_if_nonfinite=True))
        baseline_next = evaluate(baseline_state, next_ids, next_targets)
        row = dict(window=index+1, train_cursor=cursor, before_nll=baseline_loss,
                   next_before_nll=baseline_next, grad_norm_before_clip=norm,
                   raw_gradient_parameters=raw, raw_gradient_groups=aggregate(raw), arms={})
        print('window', index+1, 'baseline', baseline_loss, 'next', baseline_next, flush=True)
        for label, scale, only in arms:
            restore(scale, only)
            for opt in optimizers:
                opt.step()
            with torch.no_grad():
                for weight in learner.edges:
                    weight.clamp_(0., 5.)
                stats = {name:tensor_statistics(p, gradients[name], base_weights[name])
                         for name,p in named.items()}
            after = evaluate(initial, ids, targets)
            next_after = evaluate(baseline_state, next_ids, next_targets)
            predicted = sum(s['predicted_nll_reduction'] for s in stats.values())
            actual = baseline_loss-after
            result = dict(lr=args.lr*scale, only=only, after_nll=after,
                          actual_nll_reduction=actual,
                          predicted_nll_reduction=predicted,
                          actual_over_predicted=actual/predicted if abs(predicted)>1e-15 else None,
                          next_after_nll=next_after,
                          next_nll_reduction=baseline_next-next_after,
                          parameters=stats, groups=aggregate(stats))
            row['arms'][label] = result
            print(label, {k:v for k,v in result.items() if k not in ('parameters','groups')}, flush=True)
            if max(torch.cuda.memory_reserved(), torch.cuda.max_memory_allocated())/2**20 > 3900:
                raise MemoryError('Diagnostic exceeded 3900 MiB CUDA budget')
        report['rows'].append(row)
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        state = baseline_state
        previous = int(values[args.window-1])
        cursor += args.window
    report.update(status='completed', peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  reserved_mib=torch.cuda.memory_reserved()/2**20)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
