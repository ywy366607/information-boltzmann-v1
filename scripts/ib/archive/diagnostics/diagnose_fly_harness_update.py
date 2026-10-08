"""Preregistered finite-update diagnosis; never a capability benchmark.

The live fork takes exactly the registered number of consecutive saved-Adam
updates. Matched counterfactuals do not update optimizers, feed their scores
back, or replace the live physical state. Fixed-spike replay is an instrument
for differentiating a smooth branch, not an alternative learner.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core import fly_reservoir as reservoir
from information_boltzmann.core.fly_bptt_learning import (
    FlyBPTTGraph, FlyBPTTLearner, FlyPhysicalState,
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def json_scalar(value):
    """Retain finite NumPy numerical values as standard JSON scalars."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported diagnostic value: {type(value).__name__}')


def clone_state(state):
    return FlyPhysicalState(**{
        k: tuple(t.detach().clone() for t in v) if k == 'ring'
        else v.detach().clone() for k, v in state.state_dict().items()})


def parameter_snapshot(named, path):
    """Disk-backed snapshots; neither host commit nor graph VRAM duplicates."""
    total = sum(p.numel() for p in named.values())
    storage = np.memmap(path, mode='w+', dtype=np.float32, shape=(total,))
    offset, values = 0, {}
    for name, parameter in named.items():
        if parameter.dtype != torch.float32:
            raise ValueError('Registered snapshot precision is FP32')
        flat = parameter.detach().reshape(-1)
        for start in range(0, flat.numel(), 65536):
            chunk = flat[start:start+65536].cpu().numpy()
            storage[offset+start:offset+start+len(chunk)] = chunk
        values[name] = torch.from_numpy(storage[offset:offset+flat.numel()]).reshape(parameter.shape)
        offset += flat.numel()
    storage.flush()
    return values, storage


def restore_parameters(named, snapshot, selected=None, fraction=1.0,
                       origin=None, chunk_size=65536):
    """Restore disk-mapped FP32 values with bounded interpolation staging."""
    with torch.no_grad():
        for name, parameter in named.items():
            use_snapshot = selected is None or name in selected
            if not use_snapshot and origin is None:
                continue
            source = snapshot[name] if use_snapshot else origin[name]
            destination = parameter.view(-1)
            flat_source = source.view(-1)
            interpolate = use_snapshot and origin is not None and fraction != 1
            base = origin[name].view(-1) if interpolate else None
            for start in range(0, parameter.numel(), chunk_size):
                stop = start + chunk_size
                value = flat_source[start:stop]
                if interpolate:
                    value = base[start:stop] + fraction * (value - base[start:stop])
                destination[start:stop].copy_(value)


@contextmanager
def observe_gradient_instrument():
    """Read gradients after the learner's unchanged clipping call."""
    original = torch.nn.utils.clip_grad_norm_
    captured = {}

    def instrument(parameters, *args, **kwargs):
        parameters = list(parameters)
        norm = original(parameters, *args, **kwargs)
        for parameter in parameters:
            if parameter.grad is not None:
                captured[id(parameter)] = parameter.grad.detach()
        return norm

    torch.nn.utils.clip_grad_norm_ = instrument
    try:
        yield captured
    finally:
        torch.nn.utils.clip_grad_norm_ = original


@contextmanager
def spike_instrument(record=None):
    """Record actual hard branches or hold their values fixed for a replay."""
    original = reservoir.SpikeFn
    spikes = []
    cursor = [0]

    class Instrument:
        @staticmethod
        def apply(margin):
            index = cursor[0]
            cursor[0] += 1
            if record is None:
                spike = original.apply(margin)
                spikes.append(spike.detach().bool().clone())
                return spike
            if index >= len(record):
                raise ValueError('Fixed branch has fewer ticks than replay')
            return record[index].to(margin.dtype)

    reservoir.SpikeFn = Instrument
    try:
        yield spikes
        if record is not None and cursor[0] != len(record):
            raise ValueError('Fixed branch tick count differs from replay')
    finally:
        reservoir.SpikeFn = original


def branch_distance(a, b):
    if len(a) != len(b):
        raise ValueError('Branch lengths differ')
    per_tick = [int((x != y).sum()) for x, y in zip(a, b)]
    changed = sum(per_tick)
    total = sum(x.numel() for x in a)
    return {'changed_spikes': changed, 'total_decisions': total,
            'changed_fraction': changed / max(total, 1),
            'changed_per_physical_tick': per_tick,
            'first_divergent_tick': next((i for i, v in enumerate(per_tick) if v), None)}


def directional_summary(gradient, old, new, names, gradient_divisor=1.0):
    """Stream small chunks to CPU; FP32 unscale then float64 products."""
    dot = gg = dd = rr = absolute_products = 0.0
    changed_coordinates = 0
    for name in sorted(names):
        a, b = old[name].reshape(-1), new[name].reshape(-1)
        g = gradient.get(name)
        g = None if g is None else g.reshape(-1)
        for offset in range(0, a.numel(), 65536):
            before = a[offset:offset+65536].cpu().double()
            delta = b[offset:offset+65536].cpu().double() - before
            dd += float(delta.square().sum())
            rr += float(before.square().sum())
            changed_coordinates += int(delta.count_nonzero())
            if g is not None:
                piece = g[offset:offset+65536].cpu()
                if gradient_divisor != 1:
                    piece = piece / gradient_divisor  # FP32 unscale, as registered.
                piece = piece.double()
                dot += float((piece * delta).sum())
                absolute_products += float((piece * delta).abs().sum())
                gg += float(piece.square().sum())
    return {'gradient_dot_actual_displacement': dot, 'gradient_norm': gg**.5,
            'displacement_norm': dd**.5,
            'direction_cosine': dot / max((gg*dd)**.5, 1e-30),
            'relative_displacement_norm': (dd/max(rr, 1e-30))**.5,
            'changed_coordinates': changed_coordinates,
            'float64_accumulation_floor': 64*np.finfo(np.float64).eps*absolute_products}


def state_distance(a, b):
    values = {}
    for name, x in a.state_dict().items():
        y = getattr(b, name)
        xs, ys = (x, y) if name == 'ring' else ((x,), (y,))
        total = sum(u.numel() for u in xs)
        square = sum(float((u-v).double().square().sum()) for u, v in zip(xs, ys))
        maximum = max((float((u-v).abs().max()) for u, v in zip(xs, ys) if u.numel()), default=0.)
        values[name] = {'rms_difference': (square/max(total, 1))**.5,
                        'max_absolute_difference': maximum}
        if name == 'ring':
            values[name]['slots'] = [
                {'rms_difference': float((u-v).double().square().mean())**.5,
                 'max_absolute_difference': float((u-v).abs().max())}
                for u, v in zip(xs, ys)]
    return values


def load_fork(checkpoint):
    saved = torch.load(checkpoint, mmap=True, map_location='cpu', weights_only=False)
    if saved['format'] != 'fly-bptt-v1':
        raise ValueError('Complete lifecycle checkpoint required')
    cfg, life = saved['config'], saved['learner']
    required = {'physical', 'optimizer', 'sgd', 'adam_names', 'plasticity_optimizer_kind',
                'events', 'updates', 'previous_token', 'ema', 'physical_ticks',
                'latent_window', 'settle_ticks', 'writer_baseline_clock', 'learn_stp'}
    if required - life.keys():
        raise ValueError(f'Missing lifecycle keys: {required-life.keys()}')
    p = life['physical']
    if any(k not in p for k in ('h', 'ring', 'ge', 'gi', 'b', 'x', 'u', 'baseline', 'h_mean')):
        raise ValueError('Incomplete physical state')
    if len(p['ring']) != 4 or p['h_mean'].shape != p['h'].shape:
        raise ValueError('No baseline/read-mean or ring fallback allowed in this audit')
    model = reservoir.FlyReservoirLM(
        ROOT/cfg['graph'], vocab_size=50257, d_model=cfg['d_model'],
        injection='topographic', read_surface='output', synapse_model='coba',
        use_alif=True, use_stp=True, decoder_bias=cfg['decoder_bias'],
        read_centering=life.get('read_centering', cfg.get('read_centering', False))).cuda()
    with torch.no_grad():
        for name in ('edge_weight_e', 'edge_weight_i'):
            getattr(model, name).copy_(saved['model'][name])
    model.load_state_dict({k: v for k, v in saved['model'].items()
                          if k not in ('edge_weight_e', 'edge_weight_i')}, strict=True)
    model.dan_plastic_lr = life.get('dan_plastic_lr', 0.0)
    if model.dan_plastic_lr != 0:
        raise ValueError('Preregistered diagnosis requires DAN disabled')
    physical = FlyPhysicalState(**{
        k: tuple(t.cuda() for t in v) if k == 'ring' else v.cuda()
        for k, v in life['physical'].items()})
    learner = FlyBPTTLearner(
        model, physical, adam_names=life['adam_names'], lr=cfg['lr'],
        lr_synapse=cfg['lr_synapse'], lr_sensory=cfg['lr_sensory'],
        lr_decoder=cfg.get('lr_decoder'),
        plasticity_optimizer=life['plasticity_optimizer_kind'],
        settle_ticks=life.get('settle_ticks', 0),
        writer_baseline_clock=life.get('writer_baseline_clock', 'physical'),
        learn_stp=life.get('learn_stp', False))
    learner.load_adam_state(life['optimizer'])
    learner.sgd.load_state_dict(life['sgd'])
    learner.load_edge_signs(life)
    for key in ('events', 'updates', 'previous_token', 'ema', 'physical_ticks'):
        if key in life:
            setattr(learner, key, life[key])
    learner.latent_window.copy_(life['latent_window'])
    torch.set_rng_state(saved['rng_cpu'])
    cuda_rng = saved['rng_cuda']
    if isinstance(cuda_rng, torch.Tensor):
        torch.cuda.set_rng_state(cuda_rng)
    else:
        torch.cuda.set_rng_state_all(cuda_rng)
    # Check requested Adam learning rates exactly match saved group rates.
    old_rates = {n: g['lr'] for g in life['optimizer']['param_groups']
                 for n in g.get('parameter_names', [])}
    for group in learner.optimizer.param_groups:
        for name in group['parameter_names']:
            if name in old_rates and group['lr'] != old_rates[name]:
                raise ValueError(f'Learning rate changed for {name}')
    return saved, learner


def gate_registry(directory):
    required = ('hypotheses.json', 'necessity.json', 'blind_predictions.json',
                'preregistration.json', 'plan_review.json', 'execution_lock.json')
    documents = {name: json.loads((directory/name).read_text(encoding='utf-8'))
                 for name in required}
    if documents['plan_review.json'].get('status') != 'approved':
        raise ValueError('Independent plan approval required before execution')
    if documents['blind_predictions.json']['provenance']['measurement_access']:
        raise ValueError('Blinded predictions were exposed to measurements')
    lock = documents['execution_lock.json']
    if digest(__file__) != lock['script_sha256']:
        raise ValueError('Diagnostic code differs from locked reviewed version')
    for name, expected in lock['artifact_sha256'].items():
        if digest(directory/name) != expected:
            raise ValueError(f'Preregistration changed: {name}')
    return documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    registered = gate_registry(args.registry)
    prereg = registered['preregistration.json']
    windows = int(prereg['execution']['consecutive_windows'])
    if windows != 4 or prereg['execution']['window_events'] != 32:
        raise ValueError('This implementation is the reviewed four-window, 32-event design')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    source_stamp = (args.checkpoint.stat().st_size, args.checkpoint.stat().st_mtime_ns)
    source_hash = digest(args.checkpoint)
    saved, learner = load_fork(args.checkpoint)
    cfg, model = saved['config'], learner.model
    cursor, width = saved['train_cursor'], cfg['window']
    if width != 32:
        raise ValueError('Saved window differs from preregistration')
    data = np.load(ROOT/cfg['data']/'train.npy', mmap_mode='r')
    if int(data[cursor]) != learner.previous_token:
        raise ValueError('Saved previous token differs from consumed stream cursor')
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    head = {'output_read.weight', 'read_norm.weight', 'decoder.weight', 'decoder.bias'} & named.keys()
    body = named.keys() - head
    groups = {'full': set(named), 'body': set(body), 'head': set(head),
              'edges': set(learner.edge_names),
              'writer': {n for n in body if n.startswith('topographic_writer.')},
              'cells': {n for n in body if not n.startswith('topographic_writer.')
                        and n not in learner.edge_names}}
    result = {'scope': 'four finite consecutive updates; mechanism diagnosis, no capability claim',
              'checkpoint': str(args.checkpoint.resolve()),
              'preregistration_sha256': digest(args.registry/'preregistration.json'),
              'code_sha256': digest(__file__), 'train_cursor': cursor, 'windows': [],
              'source_checkpoint_sha256': source_hash,
              'updates_at_start': learner.updates, 'events_at_start': learner.events,
              'physical_ticks_at_start': learner.physical_ticks,
              'clipping_configuration': {'max_grad_norm': learner.max_grad_norm,
                  'provenance': 'historical learner constructor default 1.0; checkpoint omitted this field'},
              'source_checkpoint_modified': False, 'model_checkpoints_written': 0}
    ledger = {'eager_diagnostic_window_forwards': 0,
              'diagnostic_fixed_branch_backwards': 0,
              'graph_warmup_capture_forwards': 0, 'graph_warmup_capture_backwards': 0,
              'live_graph_forwards': 0, 'live_graph_backwards': 0,
              'live_eager_forwards': 0, 'live_eager_backwards': 0}

    def weights(snapshot, selected=None, fraction=1.0, origin=None):
        restore_parameters(named, snapshot, selected, fraction, origin)

    def forward(state, ids, labels, fixed=None):
        ledger['eager_diagnostic_window_forwards'] += 1
        learner.state = state
        with torch.no_grad(), spike_instrument(fixed) as record:
            scores, terminal, _ = learner.forward_window(ids, labels[None])
        return scores.detach().cpu().tolist(), clone_state(terminal), record

    for iteration in range(windows):
        start_time = time.perf_counter()
        print(f'Window {iteration+1}/{windows}: matched controls and one live update', flush=True)
        beginning = clone_state(learner.state)
        old_path, new_path = args.registry/'theta_old.bin', args.registry/'theta_new.bin'
        old, old_storage = parameter_snapshot(named, old_path)
        offset = cursor + iteration*width
        labels = torch.as_tensor(np.array(data[offset+1:offset+width+1], dtype=np.int64), device='cuda')
        fresh = torch.as_tensor(np.array(data[offset+width+1:offset+2*width+1], dtype=np.int64), device='cuda')
        ids = torch.cat((labels.new_tensor([learner.previous_token]), labels[:-1]))[None]
        fresh_ids = torch.cat((labels[-1:], fresh[:-1]))[None]
        base_fit, old_terminal, fit_spikes = forward(beginning, ids, labels)
        base_follow, _, follow_spikes = forward(old_terminal, fresh_ids, fresh)
        repeat_fit, repeat_terminal, repeat_spikes = forward(beginning, ids, labels)
        repeat_follow, _, repeat_follow_spikes = forward(old_terminal, fresh_ids, fresh)
        third_fit, _, third_spikes = forward(beginning, ids, labels)
        third_follow, _, third_follow_spikes = forward(old_terminal, fresh_ids, fresh)
        learner.state = beginning
        if iteration == 0:
            learner.runner = FlyBPTTGraph(learner, width)
            ledger['graph_warmup_capture_forwards'] += 3
            ledger['graph_warmup_capture_backwards'] += 3
        with observe_gradient_instrument() as captured:
            scores, metrics = learner.observe(labels)
        mode = 'graph' if iteration == 0 else 'eager'
        ledger[f'live_{mode}_forwards'] += 1
        ledger[f'live_{mode}_backwards'] += 1
        live_state = clone_state(learner.state)
        factor = min(1.0, learner.max_grad_norm / (metrics['grad_norm_before_clip'] + 1e-6))
        name_by_id = {id(p): n for n, p in named.items()}
        surrogate = {name_by_id[parameter_id]: gradient
                     for parameter_id, gradient in captured.items()}
        if set(surrogate) != set(named):
            raise RuntimeError('Actual observe gradient coverage is incomplete')
        surrogate_direction = {
            g: directional_summary(surrogate, old, {n: p.detach() for n, p in named.items()},
                                   names, gradient_divisor=factor)
            for g, names in groups.items()}
        surrogate_names = sorted(surrogate)
        norm_error = abs(surrogate_direction['full']['gradient_norm'] -
                         metrics['grad_norm_before_clip'])
        if norm_error > 64*np.finfo(np.float32).eps*max(metrics['grad_norm_before_clip'], 1.):
            raise RuntimeError('Instrumented gradient norm differs from actual clipping norm')
        capture_error = max(abs(a-b) for a, b in zip(scores, base_fit))
        capture_state = state_distance(live_state, old_terminal)
        identity_state = state_distance(repeat_terminal, old_terminal)
        capture_floor = max(
            max(abs(a-b) for a, b in zip(base_fit, repeat_fit)),
            max(abs(a-b) for a, b in zip(base_fit, third_fit)),
            64*np.finfo(np.float32).eps*max(base_fit))
        if capture_error > capture_floor:
            raise RuntimeError('Actual observe fit exceeds preregistered identity precision floor')
        for name, values in capture_state.items():
            xs = old_terminal.ring if name == 'ring' else (getattr(old_terminal, name),)
            scale = max((float(x.abs().max()) for x in xs if x.numel()), default=0.)
            floor = max(identity_state[name]['max_absolute_difference'],
                        64*np.finfo(np.float32).eps*max(scale, 1.))
            if values['max_absolute_difference'] > floor:
                raise RuntimeError(f'Captured terminal mismatch in {name}')
        # Release graph pool before constructing the fixed-branch gradient.
        learner.runner = None
        del surrogate
        captured.clear()
        learner.state = live_state
        gc.collect()
        torch.cuda.empty_cache()
        new, new_storage = parameter_snapshot(named, new_path)
        entry = {'iteration': iteration, 'fit_range': [offset+1, offset+width+1],
                 'actual_execution_mode': mode,
                 'capture_comparison_performed': iteration == 0,
                 'instrument_recovered_gradient_norm_error': norm_error,
                 'following_range': [offset+width+1, offset+2*width+1],
                 'base_fit_nll': float(np.mean(base_fit)),
                 'base_following_nll': float(np.mean(base_follow)),
                 'repeat_max_score_error': max(abs(a-b) for a, b in zip(base_fit, repeat_fit)),
                 'repeat_branches': branch_distance(fit_spikes, repeat_spikes),
                 'third_repeat_branches': branch_distance(fit_spikes, third_spikes),
                 'following_repeat_max_score_error': max(
                     max(abs(a-b) for a, b in zip(base_follow, repeat_follow)),
                     max(abs(a-b) for a, b in zip(base_follow, third_follow))),
                 'following_repeat_branches': branch_distance(follow_spikes, repeat_follow_spikes),
                 'following_third_repeat_branches': branch_distance(follow_spikes, third_follow_spikes),
                 'capture_max_score_error': capture_error if iteration == 0 else None,
                 'actual_observe_max_score_error': capture_error, 'metrics': metrics,
                 'capture_terminal_state_difference': capture_state,
                 'identity_terminal_state_difference': identity_state,
                 'captured_gradient_coverage': surrogate_names,
                 'parameter_partition': {g: sorted(names) for g, names in groups.items()},
                 'controls': {}, 'surrogate_direction': surrogate_direction}
        for label, names in groups.items():
            weights(new, selected=names, origin=old)
            fit, _, changed_fit = forward(beginning, ids, labels)
            follow, _, changed_follow = forward(old_terminal, fresh_ids, fresh)
            fixed_fit, _, _ = forward(beginning, ids, labels, fixed=fit_spikes)
            entry['controls'][label] = {
                'fit_nll': float(np.mean(fit)), 'following_nll': float(np.mean(follow)),
                'fixed_branch_fit_nll': float(np.mean(fixed_fit)),
                'fit_spikes': branch_distance(fit_spikes, changed_fit),
                'following_spikes': branch_distance(follow_spikes, changed_follow),
                'fit_per_event_nll': fit, 'following_per_event_nll': follow}
        # Small-step branch diagnostic, not an optimizer or LR sweep.
        entry['body_fraction_controls'] = []
        for fraction in prereg['execution']['body_displacement_fractions']:
            weights(new, selected=groups['body'], origin=old, fraction=fraction)
            hard, _, changed = forward(beginning, ids, labels)
            fixed, _, _ = forward(beginning, ids, labels, fixed=fit_spikes)
            entry['body_fraction_controls'].append({
                'fraction': fraction, 'hard_fit_nll': float(np.mean(hard)),
                'fixed_fit_nll': float(np.mean(fixed)),
                'spikes': branch_distance(fit_spikes, changed)})
        # Counterfactual consistency audit: recompute the prefix from its exact
        # OLD start with new weights. It does not replace the live old terminal.
        weights(new)
        _, counter_terminal, _ = forward(beginning, ids, labels)
        counter_score, _, _ = forward(counter_terminal, fresh_ids, fresh)
        entry['following_from_recomputed_prefix_nll'] = float(np.mean(counter_score))
        entry['old_new_terminal_state_difference'] = state_distance(old_terminal, counter_terminal)
        weights(old)
        old_new_state_score, _, _ = forward(counter_terminal, fresh_ids, fresh)
        entry['state_parameter_factorial'] = {
            'old_parameters_old_terminal': float(np.mean(base_follow)),
            'new_parameters_old_terminal': entry['controls']['full']['following_nll'],
            'old_parameters_recomputed_new_terminal': float(np.mean(old_new_state_score)),
            'new_parameters_recomputed_new_terminal': float(np.mean(counter_score))}
        weights(new, selected=groups['body'], origin=old)
        _, body_counter_terminal, _ = forward(beginning, ids, labels)
        body_new_state_score, _, _ = forward(body_counter_terminal, fresh_ids, fresh)
        weights(old)
        body_old_new_state_score, _, _ = forward(body_counter_terminal, fresh_ids, fresh)
        entry['body_state_parameter_factorial'] = {
            'old_parameters_old_terminal': float(np.mean(base_follow)),
            'body_parameters_old_terminal': entry['controls']['body']['following_nll'],
            'old_parameters_recomputed_body_terminal': float(np.mean(body_old_new_state_score)),
            'body_parameters_recomputed_body_terminal': float(np.mean(body_new_state_score))}
        entry['old_body_terminal_state_difference'] = state_distance(old_terminal, body_counter_terminal)
        # Exact derivative of the current selected smooth branch, NOT of a
        # discontinuous spike crossing. Absent derivatives count as zero.
        weights(old)
        learner.state = beginning
        model.zero_grad(set_to_none=True)
        # Keep leaf gradient accumulation on the same nonlegacy stream as
        # capture. A default-stream diagnostic backward between captures can
        # make the next capture require an illegal legacy-stream dependency.
        caller_stream = torch.cuda.current_stream()
        fixed_stream = torch.cuda.graph.default_capture_stream
        if fixed_stream is None:
            raise RuntimeError('Graph capture stream was not initialized')
        fixed_stream.wait_stream(caller_stream)
        with torch.cuda.stream(fixed_stream), spike_instrument(fit_spikes):
            fixed_scores, _, _ = learner.forward_window(ids, labels[None])
            ledger['eager_diagnostic_window_forwards'] += 1
            fixed_scores.mean().backward()
            ledger['diagnostic_fixed_branch_backwards'] += 1
        caller_stream.wait_stream(fixed_stream)
        physical = {n: p.grad.detach() for n, p in named.items() if p.grad is not None}
        entry['fixed_branch_derivative'] = {
            g: directional_summary(physical, old, new, names) for g, names in groups.items()}
        entry['fixed_branch_missing_gradient_names'] = sorted(set(named)-physical.keys())
        identity, identity_terminal, _ = forward(beginning, ids, labels, fixed=fit_spikes)
        entry['fixed_branch_identity_max_score_error'] = max(abs(a-b) for a, b in zip(base_fit, identity))
        if entry['fixed_branch_identity_max_score_error'] > capture_floor:
            raise RuntimeError('Fixed-spike reference differs from actual baseline')
        entry['fixed_branch_identity_terminal_difference'] = state_distance(identity_terminal, old_terminal)
        for name, values in entry['fixed_branch_identity_terminal_difference'].items():
            xs = old_terminal.ring if name == 'ring' else (getattr(old_terminal, name),)
            scale = max((float(x.abs().max()) for x in xs if x.numel()), default=0.)
            floor = max(identity_state[name]['max_absolute_difference'],
                        64*np.finfo(np.float32).eps*max(scale, 1.))
            if values['max_absolute_difference'] > floor:
                raise RuntimeError(f'Fixed-spike reference terminal differs in {name}')
        entry['score_numerical_floor'] = max(
            entry['repeat_max_score_error'], capture_error,
            entry['following_repeat_max_score_error'],
            entry['fixed_branch_identity_max_score_error'],
            32*np.finfo(np.float32).eps*max(max(base_fit), max(base_follow)))
        model.zero_grad(set_to_none=True)
        del physical
        weights(new)
        learner.state = live_state
        entry['elapsed_diagnostic_seconds'] = time.perf_counter()-start_time
        result['windows'].append(entry)
        result['execution_ledger'] = dict(ledger)
        peak = torch.cuda.max_memory_allocated()/2**20
        if peak >= 3900 or torch.cuda.max_memory_reserved()/2**20 >= 3900:
            raise RuntimeError(f'Registered memory ceiling exceeded: {peak:.1f} MiB')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False,
                                         default=json_scalar), encoding='utf-8')
        print(f'Window {iteration+1}: fit {entry["base_fit_nll"]:.5f}, '
              f'body {entry["controls"]["body"]["fit_nll"]:.5f}; '
              f'fresh {entry["base_following_nll"]:.5f} -> '
              f'{entry["controls"]["full"]["following_nll"]:.5f}; '
              f'peak {peak:.0f} MiB', flush=True)
        del old, new, fit_spikes, follow_spikes, repeat_spikes, third_spikes
        del repeat_follow_spikes, third_follow_spikes
        gc.collect()
        old_storage._mmap.close()
        new_storage._mmap.close()
        del old_storage, new_storage
        for path in (old_path, new_path):
            if path.resolve().parent != args.registry.resolve():
                raise ValueError('Scratch snapshot escaped registry')
            path.unlink()
    result.update(updates_at_end=learner.updates, events_at_end=learner.events,
                  physical_ticks_at_end=learner.physical_ticks,
                  actual_optimizer_updates=learner.updates-result['updates_at_start'],
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                  source_checkpoint_modified=source_stamp != (
                      args.checkpoint.stat().st_size, args.checkpoint.stat().st_mtime_ns))
    result['execution_ledger'].update(
        unique_live_events=windows*width, unique_lookahead_events=width,
        diagnostic_replayed_events=ledger['eager_diagnostic_window_forwards']*width,
        graph_placeholder_warmup_events=ledger['graph_warmup_capture_forwards']*width,
        total_forward_physical_ticks=(ledger['eager_diagnostic_window_forwards']+
            ledger['graph_warmup_capture_forwards']+ledger['live_graph_forwards']+
            ledger['live_eager_forwards'])*
            width*(1+learner.settle_ticks))
    if learner.updates-result['updates_at_start'] != windows or \
            learner.events-result['events_at_start'] != windows*width or \
            learner.physical_ticks-result['physical_ticks_at_start'] != windows*width*(1+learner.settle_ticks):
        raise RuntimeError('Live counters differ from registered cadence')
    # Preserve an isolated full diagnostic fork for post-result lineage review.
    # This is not a candidate trained model, and its destination is fixed to
    # the registry directory rather than any production checkpoint path.
    destination = (args.registry/'continuation.pt').resolve()
    if destination == args.checkpoint.resolve():
        raise ValueError('Diagnostic snapshot cannot overwrite its source')
    continuation = {'format': 'fly-diagnostic-v1', 'model': model.state_dict(),
                    'learner': learner.state_dict(), 'train_cursor': cursor+windows*width,
                    'config': cfg, 'source_sha256': source_hash,
                    'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all(),
                    'execution_ledger': result['execution_ledger']}
    torch.save(continuation, destination)
    result['diagnostic_continuation_snapshot'] = str(destination)
    result['diagnostic_continuation_sha256'] = digest(destination)
    result['model_checkpoints_written'] = 1
    result['source_checkpoint_hash_unchanged'] = digest(args.checkpoint) == source_hash
    if result['source_checkpoint_modified'] or not result['source_checkpoint_hash_unchanged']:
        raise RuntimeError('Source checkpoint changed during diagnostic')
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False,
                                     default=json_scalar), encoding='utf-8')
    print('Completed all registered windows; source lifecycle unchanged.', flush=True)


if __name__ == '__main__':
    main()
