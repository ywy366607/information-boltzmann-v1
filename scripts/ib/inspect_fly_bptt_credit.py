"""Measure BPTT32 credit on a saved, never-reset real OWT trajectory.

This is a differential diagnostic, not an active-evaluation score. The live
learner is untouched. A checkpoint fork keeps weights fixed while physical
state continues. Ring slots get identity clones so each event has distinct
state coordinates when querying adjoints. No credit is pruned.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core import fly_reservoir as fly
from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState


def flatten_state(state):
    return {**{key: getattr(state, key) for key in
               ('h', 'ge', 'gi', 'b', 'x', 'u', 'baseline')},
            **{f'ring{k}': value for k, value in enumerate(state.ring)}}


def concentration(values):
    """Squared-gradient concentration; exact-zero support is kept separate."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    power = values ** 2
    total = float(power.sum(dtype=np.float64))
    result = {'size': len(values), 'norm': total ** .5,
              'rms': (total / max(len(values), 1)) ** .5,
              'exact_zero_fraction': float(np.mean(values == 0))}
    if total == 0:
        return {**result, 'fraction_for_90pct_power': 0.,
                'fraction_for_99pct_power': 0., 'participation_fraction': 0.}
    ordered = np.sort(power)[::-1]
    running, fourth = 0., 0.
    thresholds = {percentage: total*percentage/100 for percentage in (90, 99)}
    for start in range(0, len(ordered), 32768):
        block = ordered[start:start+32768].astype(np.float64)
        cumulative = running+np.cumsum(block)
        fourth += float(block @ block)
        for percentage, threshold in thresholds.items():
            key = f'fraction_for_{percentage}pct_power'
            if key not in result and cumulative[-1] >= threshold:
                result[key] = float((start+np.searchsorted(cumulative, threshold)+1)/len(values))
        running = float(cumulative[-1])
    result['participation_fraction'] = float(total ** 2 / fourth / len(values))
    result['power_share_top_1pct'] = float(ordered[:max(1, len(values)//100)].sum(dtype=np.float64)/total)
    return result


def state_stats(value, gradient, *, classes=None, names=None, firing=None, detailed=True):
    v = value.detach().flatten().cpu().numpy()
    g = np.zeros_like(v) if gradient is None else gradient.detach().flatten().cpu().numpy()
    out = concentration(g) if detailed else {
        'size': len(g), 'norm': float(np.linalg.norm(g.astype(np.float64))),
        'exact_zero_fraction': float(np.mean(g == 0))}
    out['relative_state_perturbation_rms_response'] = float(np.linalg.norm(g*v))
    if classes is not None and len(g) == len(classes):
        power = g.astype(np.float64)**2
        total = float(power.sum())
        class_power = np.bincount(classes, weights=power, minlength=len(names))
        counts = np.bincount(classes, minlength=len(names))
        out['superclasses'] = {name: {'neurons': int(counts[i]),
            'power_share': float(class_power[i]/total) if total else 0.}
            for i, name in enumerate(names)}
        if firing is not None:
            out['credit_power_on_currently_spiking_neurons'] = (
                float(power[firing].sum()/total) if total else 0.)
            out['spiking_fraction'] = float(firing.mean())
    return out


def edge_credit(pre, post, splits, currents, pulses):
    """Full edge gradient from recorded current adjoints, bounded CPU chunks.

    dL/dw_ij = sum_t dL/d(delta_g_j[t]) * transmitted_i[t-delay].
    This includes all downstream BPTT paths through the current adjoint.
    """
    result = np.zeros(len(pre), dtype=np.float32)
    eligible = np.zeros(len(pre), dtype=bool)
    for delay in range(4):
        for start in range(splits[delay], splits[delay+1], 32768):
            stop = min(start+32768, splits[delay+1])
            source = (pulses[:, delay, pre[start:stop]] if isinstance(pulses, np.ndarray)
                      else np.stack([row[delay][pre[start:stop]] for row in pulses]))
            result[start:stop] = np.sum(currents[:, post[start:stop]]*source, axis=0)
            eligible[start:stop] = np.any(source != 0, axis=0)
    return result, eligible


def move_state(state, device, leaf=False):
    return FlyPhysicalState(**{key: tuple(t.detach().to(device).clone().requires_grad_(leaf) for t in value)
        if key == 'ring' else value.detach().to(device).clone().requires_grad_(leaf)
        for key, value in state.state_dict().items()})


def one_event(model, state, token_id, device):
    token = torch.tensor([int(token_id)], device=device)
    drive, baseline = model.topographic_writer.forward_with_state(model.embedding(token), state.h, state.baseline)
    h, _, ring, ge, gi, b, x, u = model.step(state.h, token,
        spike_ring=state.ring, ge=state.ge, gi=state.gi, b=state.b,
        x=state.x, u=state.u, sensory_drive=drive)
    return FlyPhysicalState(h, ring, ge, gi, b, x, u, baseline), drive


def reverse_credit(model, history, inputs, direct, endpoint, device, small_names,
                   describe, window_objective=False):
    """Exact checkpointed BPTT: replay one full event, propagate all adjoints.

    Local VJPs are composed across the entire window, including cross-neuron
    and all physical-state feedback. This is BPTT with activation recomputation,
    not conditional local eligibility or a modified learning rule.
    """
    adjoint = {key: torch.zeros_like(value, device=device)
               for key, value in flatten_state(history[endpoint]).items()}
    small = dict(model.named_parameters())
    parameter_grads = {name: torch.zeros_like(small[name]) for name in small_names}
    rows, drive_rows, current_grads = [], [], [None]*endpoint
    original = fly.execute_delayed_synaptic_transmission
    replay_error = 0.
    for event in reversed(range(endpoint)):
        adjoint['h'][:, model.read_indices] += direct[event].to(device)[None]
        rows.append(describe(event+1, adjoint))
        previous = move_state(history[event], device, leaf=True)
        currents = []
        def record(*params):
            current = original(*params)
            currents.append(current)
            return current
        fly.execute_delayed_synaptic_transmission = record
        try:
            next_state, drive = one_event(model, previous, inputs[event], device)
        finally:
            fly.execute_delayed_synaptic_transmission = original
        previous_flat, next_flat = flatten_state(previous), flatten_state(next_state)
        if event == endpoint-1:
            replay_error = max(float((value.detach().cpu()-flatten_state(history[event+1])[key]).abs().max())
                               for key, value in next_flat.items())
        grads = torch.autograd.grad(tuple(next_flat.values()),
            tuple(previous_flat.values())+tuple(currents)+(drive,)+tuple(small[n] for n in small_names),
            grad_outputs=tuple(adjoint.values()), allow_unused=True)
        adjoint = {key: torch.zeros_like(value) if g is None else g.detach()
                   for (key, value), g in zip(previous_flat.items(), grads[:11])}
        drive_g = grads[13]
        indices = model.topographic_writer.injection_index
        drive_rows.append({'event': event, 'support': 'actual sensory injection indices only',
            **concentration(np.zeros(len(indices), dtype=np.float32)
                if drive_g is None else drive_g.detach()[:, indices].cpu().numpy())})
        if window_objective:
            current_grads[event] = [g.detach().cpu().numpy().reshape(-1) for g in grads[11:13]]
            for name, g in zip(small_names, grads[14:]):
                if g is not None:
                    parameter_grads[name] += g.detach()
        del grads, next_state, drive, currents, previous, previous_flat, next_flat
    rows.append(describe(0, adjoint))
    return rows, drive_rows, current_grads, parameter_grads, replay_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'results/q8_fly_bptt32_continuous_100k')
    parser.add_argument('--output', type=Path, default=ROOT/'results/fly_bptt32_credit_structure')
    parser.add_argument('--windows', type=int, default=8)
    parser.add_argument('--window', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cuda-fraction', type=float, default=.32)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.cuda_fraction)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.run/'last.pt'
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if saved['learner'].get('settle_ticks', 0):
        raise ValueError('This historical credit inspector assumes one physical tick per token. '
                         'Pulse/quiet checkpoints require physical-tick adjoints; use the native '
                         'FlyBPTTLearner.forward_window for complete credit with saved settle_ticks.')
    cfg = saved['config']
    if saved['format'] != 'fly-bptt-v1':
        raise ValueError('Complete continuing BPTT checkpoint required')
    model = fly.FlyReservoirLM(ROOT/cfg['graph'], vocab_size=50257,
        d_model=cfg['d_model'], injection='topographic', read_surface='output',
        synapse_model='coba', use_alif=True, use_stp=True,
        decoder_bias=cfg['decoder_bias'])
    with torch.no_grad():
        weights = {key: value for key, value in saved['model'].items()
                   if key not in ('edge_weight_e', 'edge_weight_i')}
        loaded = model.load_state_dict(weights, strict=False)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise ValueError(str(loaded))
        for name in ('edge_weight_e', 'edge_weight_i'):
            getattr(model, name).copy_(saved['model'][name])
    model.requires_grad_(False)
    # Small physical parameters only; large weight gradients are reconstructed
    # analytically below to avoid a second 25M-edge optimizer allocation.
    small_names = [name for name in saved['learner']['adam_names']
                   if name.startswith('log_') or name.startswith('topographic_writer.gate_linear')]
    for name in small_names:
        dict(model.named_parameters())[name].requires_grad_(True)
    model = model.to(args.device)
    data = np.load(ROOT/cfg['data']/'train.npy', mmap_mode='r')
    cursor = saved['train_cursor']
    previous = saved['learner']['previous_token']
    raw_state = saved['learner']['physical']
    state = FlyPhysicalState(**{key: tuple(t.to(args.device) for t in value) if key == 'ring'
                               else value.to(args.device) for key, value in raw_state.items()})
    metadata = {'scope': 'fixed-weight differential diagnostic on continuous checkpoint fork; no optimizer updates; not primary active evaluation',
        'checkpoint': str(checkpoint), 'checkpoint_bptt_tokens': saved['bptt_train_tokens'],
        'checkpoint_train_cursor': cursor, 'checkpoint_events': saved['learner']['events'],
        'checkpoint_updates': saved['learner']['updates'], 'windows': args.windows,
        'window': args.window, 'dataset': cfg['data'],
        'gradient_definition': 'full within-window BPTT using unchanged ATan threshold backward; unscaled and unclipped',
        'drive_gradient_support': 'actual sensory injection indices only',
        'relative_state_response': 'norm(gradient * state); scale diagnostic, not semantic value or certified pruning bound'}
    classes = model.superclass_id.cpu().numpy()
    names = model.superclass_names
    topology = {kind: (getattr(model, f'edge_pre_{kind}').cpu().numpy(),
                       getattr(model, f'edge_post_{kind}').cpu().numpy(),
                       getattr(model, f'splits_{kind}')) for kind in ('e', 'i')}
    del saved, weights, raw_state
    gc.collect()
    window_files = []
    started = time.perf_counter()
    state = move_state(state, 'cpu')
    for w in range(args.windows):
        print(f'Window {w+1}/{args.windows}, real OWT cursor {cursor}', flush=True)
        targets = np.asarray(data[cursor+1:cursor+args.window+1], dtype=np.int64)
        inputs = np.concatenate(([previous], targets[:-1]))
        history = [state]
        with torch.no_grad():
            for token_id in inputs:
                gpu_state = move_state(state, args.device)
                next_state, _ = one_event(model, gpu_state, token_id, args.device)
                next_cpu = move_state(next_state, 'cpu')
                next_cpu.ring = (next_cpu.ring[0], *state.ring[:3])
                state = next_cpu
                history.append(state)
        del gpu_state, next_state
        motor = torch.cat([item.h[:, model.read_indices.cpu()] for item in history[1:]]).to(args.device).requires_grad_()
        scores = F.cross_entropy(model.decoder(model.read_norm(model.output_read(motor))),
            torch.as_tensor(targets, device=args.device), reduction='none')
        objectives = [args.window//2, args.window, 'window']
        direct = [torch.autograd.grad(scores.mean() if objective == 'window' else scores[objective-1],
                                     motor, retain_graph=i<2)[0].detach().cpu()
                  for i, objective in enumerate(objectives)]
        window_report = {'window_index': w, 'cursor': cursor, 'nll': float(scores.detach().mean()), 'endpoint_credit': []}
        del scores, motor
        for objective, read_grad in zip(objectives, direct):
            endpoint = args.window if objective == 'window' else objective
            def describe(event, adjoint):
                firing = history[event].ring[0].flatten().numpy()>0
                return {'event': event, 'lag': endpoint-event, 'states': {
                    key: state_stats(value, adjoint[key], classes=classes, names=names, firing=firing,
                        detailed=(event in (0, 8, 16, 24, 32) if objective == 'window'
                                  else endpoint-event in (0, 1, 2, 4, 8, 16, 32)))
                    for key, value in flatten_state(history[event]).items()}}
            rows, drive_rows, current_grads, parameter_grads, replay_error = reverse_credit(
                model, history, inputs, read_grad, endpoint, args.device, small_names,
                describe, window_objective=objective=='window')
            if objective != 'window':
                window_report['endpoint_credit'].append({'endpoint': endpoint,
                    'states_by_lag': rows, 'input_drive_by_lag': [dict(item, lag=endpoint-1-item['event']) for item in drive_rows],
                    'replay_state_max_abs_error': replay_error})
            else:
                window_report['window_state_credit'] = rows
                window_report['window_input_drive_credit'] = drive_rows
                window_report['replay_state_max_abs_error'] = replay_error
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                pulse_history = [[slot.numpy().reshape(-1) for slot in item.ring] for item in history[:-1]]
                window_report['edge_gradients'] = {}
                for k, kind in enumerate(('e', 'i')):
                    gradients, eligible = edge_credit(*topology[kind], np.stack([item[k] for item in current_grads]), pulse_history)
                    window_report['edge_gradients'][kind] = {**concentration(gradients),
                        'fraction_with_any_arriving_pulse': float(eligible.mean()),
                        'inactive_edge_max_abs_gradient': float(np.abs(gradients[~eligible]).max(initial=0))}
                    del gradients, eligible
                window_report['physical_and_gate_parameter_gradients'] = {
                    name: concentration(g.detach().cpu().numpy()) for name, g in parameter_grads.items()}
                del pulse_history
            del rows, drive_rows, current_grads, parameter_grads
        previous, cursor = int(targets[-1]), cursor+args.window
        window_report['seconds_elapsed'] = time.perf_counter()-started
        if args.device == 'cuda':
            window_report['diagnostic_peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
            window_report['diagnostic_reserved_mib'] = torch.cuda.memory_reserved()/2**20
        window_file = args.output/f'window_{w:03d}.json'
        window_file.write_text(json.dumps(window_report, indent=2, allow_nan=False), encoding='utf-8')
        window_files.append(window_file)
        # Stream persisted windows; never retain all large spatial reports in RAM.
        with (args.output/'report.json').open('w', encoding='utf-8') as out:
            out.write(json.dumps({**metadata, 'completed_windows': len(window_files)}, allow_nan=False)[:-1])
            out.write(', "measurements": [')
            for i, path in enumerate(window_files):
                if i:
                    out.write(',')
                out.write(path.read_text(encoding='utf-8'))
            out.write(']}')
        print(f'Window done: NLL {window_report["nll"]:.4f}; peak {window_report.get("diagnostic_peak_allocated_mib", 0):.0f} MiB', flush=True)
        del history, direct
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
