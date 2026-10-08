"""One saved-Adam OWT update on a complete lifecycle fork; source is untouched.

Numerical/optimizer calibration only. Following-window scores use the same
pre-update physical state and differ only in the updated weights. No evaluation
labels train the fork, and no model checkpoints are created by this audit.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner, FlyBPTTGraph, FlyPhysicalState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    saved = torch.load(args.checkpoint, mmap=True, map_location='cpu', weights_only=False)
    if saved['format'] != 'fly-bptt-v1':
        raise ValueError('A complete BPTT lifecycle checkpoint is required')
    cfg, old = saved['config'], saved['learner']
    model = FlyReservoirLM(ROOT/cfg['graph'], vocab_size=50257, d_model=cfg['d_model'],
        injection='topographic', read_surface='output', synapse_model='coba',
        use_alif=True, use_stp=True, decoder_bias=cfg['decoder_bias'],
        read_centering=old.get('read_centering', cfg.get('read_centering', False))).cuda()
    with torch.no_grad():
        for name in ('edge_weight_e', 'edge_weight_i'):
            getattr(model, name).copy_(saved['model'][name])
    model.load_state_dict({name: tensor for name, tensor in saved['model'].items()
                          if name not in ('edge_weight_e', 'edge_weight_i')}, strict=True)
    model.dan_plastic_lr = old.get('dan_plastic_lr', cfg.get('dan_plastic_lr', 0.0))
    initial = FlyPhysicalState(**{
        k: tuple(t.cuda() for t in v) if k == 'ring' else v.cuda()
        for k, v in old['physical'].items()})
    learner = FlyBPTTLearner(model, initial, adam_names=old['adam_names'],
        lr=cfg['lr'], lr_synapse=cfg['lr_synapse'], lr_sensory=cfg['lr_sensory'],
        lr_decoder=cfg.get('lr_decoder'), plasticity_optimizer=old['plasticity_optimizer_kind'],
        settle_ticks=old.get('settle_ticks', 0),
        writer_baseline_clock=old.get('writer_baseline_clock', 'physical'),
        learn_stp=old.get('learn_stp', False))
    learner.load_adam_state(old['optimizer'])
    learner.sgd.load_state_dict(old['sgd'])
    learner.load_edge_signs(old)
    for key in ('events', 'updates', 'previous_token', 'ema', 'physical_ticks'):
        if key in old:
            setattr(learner, key, old[key])
    learner.latent_window.copy_(old['latent_window'])
    initial = learner.state
    data = np.load(ROOT/cfg['data']/'train.npy', mmap_mode='r')
    cursor, width = saved['train_cursor'], cfg['window']
    labels = torch.as_tensor(np.array(data[cursor+1:cursor+width+1], dtype=np.int64), device='cuda')
    fresh = torch.as_tensor(np.array(data[cursor+width+1:cursor+2*width+1], dtype=np.int64), device='cuda')
    ids = torch.cat((labels.new_tensor([learner.previous_token]), labels[:-1]))[None]
    fresh_ids = torch.cat((labels[-1:], fresh[:-1]))[None]
    with torch.no_grad():
        before_fit, terminal_before, _ = learner.forward_window(ids, labels[None])
        fit_before = float(before_fit.mean())
        learner.state = terminal_before.detached()
        fresh_before = float(learner.forward_window(fresh_ids, fresh[None])[0].mean())
    learner.state = initial
    print('Capturing full 32-event lifecycle fork...', flush=True)
    learner.runner = FlyBPTTGraph(learner, width)
    torch.cuda.synchronize()
    start = time.perf_counter()
    scores, gradient_metrics = learner.observe(labels)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    continuing = learner.state
    learner.state = initial
    with torch.no_grad():
        fit_after = float(learner.forward_window(ids, labels[None])[0].mean())
        learner.state = terminal_before.detached()
        fresh_after = float(learner.forward_window(fresh_ids, fresh[None])[0].mean())
        # Crossed finite-update controls identify which side causes a local
        # improvement/interference. They perform no additional optimizer step.
        head_names = {'output_read.weight', 'read_norm.weight', 'decoder.weight', 'decoder.bias'}
        new_head = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n in head_names}
        new_body = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                    if p.requires_grad and n not in head_names}
        for name, parameter in model.named_parameters():
            if name in head_names:
                parameter.copy_(saved['model'][name])
        learner.state = terminal_before.detached()
        body_only = float(learner.forward_window(fresh_ids, fresh[None])[0].mean())
        for name, parameter in model.named_parameters():
            if name in head_names:
                parameter.copy_(new_head[name])
            elif parameter.requires_grad:
                parameter.copy_(saved['model'][name])
        learner.state = terminal_before.detached()
        head_only = float(learner.forward_window(fresh_ids, fresh[None])[0].mean())
        named = dict(model.named_parameters())
        for name in new_head:
            named[name].copy_(saved['model'][name])
        component_controls = {}
        component_updates = {}
        body_sets = {
            'transmission_edges': set(learner.edge_names),
            'sensory_writer': {n for n in new_body if n.startswith('topographic_writer.')},
            'cell_biophysics': {n for n in new_body if not n.startswith('topographic_writer.')
                                and n not in learner.edge_names},
        }
        for label, names in body_sets.items():
            displacement_sq = reference_sq = 0.0
            for name in names:
                named[name].copy_(new_body[name])
                delta = new_body[name].double()-saved['model'][name].double()
                displacement_sq += float(delta.square().sum())
                reference_sq += float(saved['model'][name].double().square().sum())
            learner.state = terminal_before.detached()
            component_controls[label] = float(learner.forward_window(fresh_ids, fresh[None])[0].mean())
            component_updates[label] = {
                'update_norm': displacement_sq**.5,
                'relative_parameter_update_norm': (displacement_sq/max(reference_sq, 1e-30))**.5,
            }
            for name in names:
                named[name].copy_(saved['model'][name])
        # Return the fork itself to the exact full-update weights/state.
        for name, value in {**new_body, **new_head}.items():
            named[name].copy_(value)
    learner.state = continuing
    peak = torch.cuda.max_memory_allocated()/2**20
    if peak >= 3900:
        raise RuntimeError(f'Calibration exceeded registered allocation limit: {peak:.1f} MiB')
    payload = {
        'scope': 'one numerical update on a lifecycle fork; fitted replay is not validation; '
                 'fresh same-state scores diagnose this update only, not long-run capability',
        'checkpoint': str(args.checkpoint.resolve()), 'train_cursor': cursor,
        'fit_range': [cursor+1, cursor+width+1], 'following_range': [cursor+width+1, cursor+2*width+1],
        'optimizer_updates': 1, 'physical_ticks_per_event': 1+learner.settle_ticks,
        'read_centering': model.read_centering, 'dan_plastic_lr': learner.dan_plastic_lr,
        'learn_stp': learner.learn_stp,
        'fit_nll_before': fit_before, 'fit_nll_after': fit_after,
        'fit_improvement': fit_before-fit_after,
        'following_same_state_nll_before': fresh_before,
        'following_same_state_nll_after': fresh_after,
        'following_same_state_improvement': fresh_before-fresh_after,
        'following_same_state_body_update_only_nll': body_only,
        'following_same_state_head_update_only_nll': head_only,
        'following_same_state_component_only_nll': component_controls,
        'actual_body_update_scales': component_updates,
        'eager_capture_max_score_error': float(np.max(np.abs(np.asarray(scores)-before_fit.cpu().numpy()))),
        'captured_update_seconds': elapsed, 'tokens_per_second': width/elapsed,
        'peak_allocated_mib': peak, 'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
        'gradients': gradient_metrics, 'dan_wiring_trainable': model.dan_edge_weight.requires_grad,
        'all_continuing_state_finite': all(torch.isfinite(t).all().item()
            for v in continuing.state_dict().values() for t in (v if isinstance(v, tuple) else (v,))),
        'source_checkpoint_modified': False, 'model_checkpoints_written': 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
