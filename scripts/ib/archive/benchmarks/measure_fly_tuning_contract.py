"""Mature-checkpoint material response. Frozen diagnostic, not a learning benchmark.

Explicit quiet ticks preserve all continuing physical fields. Real OWT packets
are replayed through the existing sensory writer, with unchanged hard dynamics.
No optimizer, new weights, speculative targets or cold-start state are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState, advance_fly_input_event


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def clone(s):
    return FlyPhysicalState(**{k: tuple(t.clone() for t in v) if k == 'ring'
                              else v.clone() for k, v in s.state_dict().items()})


def quiet(model, s, token, options):
    """Exact COBA/ALIF/STP quiet substep from advance_fly_input_event.

The input-clock writer baseline stays constant. This implementation is gated
by an all-field identity against the production input+settle interface.
"""
    h, spike, ring, ge, gi, b, x, u = model.step(
        s.h, token, spike_ring=s.ring, ge=s.ge, gi=s.gi, b=s.b,
        x=s.x, u=s.u, sensory_drive=torch.zeros_like(s.h), **options)
    hm = .99 * s.h_mean + .01 * h
    read = (h - hm)[:, model.read_indices] if model.read_centering else h[:, model.read_indices]
    a = model.get_read_gamma_decay()
    z1 = a * s.gamma_z1 + (1 - a) * read
    z2 = a * s.gamma_z2 + (1 - a) * z1
    return FlyPhysicalState(h, ring, ge, gi, b, x, u, s.baseline, hm, s.dan_gate, z1, z2)


def distances(a, b):
    out = {}
    for k, x in a.state_dict().items():
        y = getattr(b, k)
        pairs = zip(x, y) if k == 'ring' else [(x, y)]
        out[k] = max((float((u - v).abs().max()) for u, v in pairs if u.numel()), default=0.)
    return out


def reachability(model):
    """Multi-source minimum delay on actual nonzero directed chemical edges."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    pre, post, delays = [], [], []
    for kind in ('e', 'i'):
        a = getattr(model, 'edge_pre_' + kind).cpu().numpy()
        b = getattr(model, 'edge_post_' + kind).cpu().numpy()
        w = getattr(model, 'edge_weight_' + kind).cpu().numpy()
        splits = getattr(model, 'splits_' + kind)
        dd = np.empty(len(w), dtype=np.float32)
        for k in range(4):
            dd[splits[k]:splits[k + 1]] = k + 1
        valid = w != 0
        pre.append(a[valid]); post.append(b[valid]); delays.append(dd[valid])
    graph = csr_matrix((np.concatenate(delays), (np.concatenate(pre), np.concatenate(post))),
                       shape=(model.n_neurons, model.n_neurons))
    if graph.nnz != sum(len(x) for x in pre):
        raise ValueError('Parallel edges need minimum-delay aggregation, not CSR summation')
    sensory = model.injection_index.cpu().numpy()
    motor = model.read_indices.cpu().numpy()
    result = {'scope': 'Topology and discrete-delay lower bounds; ignores thresholds, signs and task information'}
    for name, source, target in (('sensory_to_motor', sensory, motor), ('motor_to_sensory', motor, sensory)):
        dd = dijkstra(graph, directed=True, indices=source, min_only=True)
        dt = dd[target]; finite = dt[np.isfinite(dt)]
        result[name] = {'source_neurons': len(source), 'target_neurons': len(target),
                        'reachable_targets': len(finite), 'reachable_fraction': float(len(finite) / len(target)),
                        'minimum_delay_ticks': float(finite.min()) if len(finite) else None,
                        'median_delay_lower_bound_ticks': float(np.median(finite)) if len(finite) else None,
                        'max_delay_lower_bound_ticks': float(finite.max()) if len(finite) else None}
    return result


def load(path):
    saved = torch.load(path, mmap=True, map_location='cpu', weights_only=False)
    cfg, life = saved['config'], saved['learner']
    if saved['format'] != 'fly-bptt-v1' or life.get('dan_plastic_lr', 0.) != 0:
        raise ValueError('Requires complete continuing checkpoint without implicit plastic updates')
    if life.get('writer_baseline_clock') != 'input' or life.get('settle_ticks') != 0:
        raise ValueError('Registered checkpoint uses input-clock and one physical tick/event')
    p = life['physical']
    required = ('h', 'ring', 'ge', 'gi', 'b', 'x', 'u', 'baseline', 'h_mean', 'gamma_z1', 'gamma_z2')
    if any(k not in p or (k != 'ring' and not p[k].numel()) for k in required):
        raise ValueError('Complete mature state required; no fallback initialization')
    with torch.device('meta'):
        model = FlyReservoirLM(ROOT / cfg['graph'], vocab_size=50257,
            d_model=cfg['d_model'], injection='topographic', read_surface='output',
            synapse_model='coba', use_alif=True, use_stp=True,
            decoder_bias=cfg['decoder_bias'], read_centering=life['read_centering'],
            use_read_gamma_trace=True)
    model.load_state_dict({k: v for k, v in saved['model'].items()
                           if k not in ('edge_weight_e', 'edge_weight_i')}, strict=True, assign=True)
    for k in ('edge_weight_e', 'edge_weight_i'):
        setattr(model, k, saved['model'][k].detach())
    # Legacy mutable writer buffer is nonpersistent and not used by the
    # functional event interface. Materialize it from this individual's real
    # baseline rather than introducing a zero adaptation state.
    model.topographic_writer.a_adapt = p['baseline'][0].detach().clone()
    missing = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if missing:
        raise ValueError(f'Unrestored meta storage: {missing}')
    model.requires_grad_(False)
    model.dan_plastic_lr = 0.
    model.eval()
    return saved, model


def cosine(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else None


def compare(a, b):
    out = {}
    for key in ('raw', 'centered', 'z1', 'z2', 'projected', 'normalized', 'baseline'):
        delta = a[key].astype(np.float64) - b[key].astype(np.float64)
        metric = 'baseline_magnitude_difference' if key == 'baseline' else key + '_difference_norm'
        out[metric] = np.linalg.norm(delta, axis=1).tolist()
    for key in ('firing', 'sensory_firing', 'motor_firing', 'ring_norm', 'h_norm'):
        out[key + '_a'] = a[key].tolist(); out[key + '_b'] = b[key].tolist()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reg = json.loads(args.registry.read_text(encoding='utf-8'))
    if reg.get('approval', {}).get('status') != 'approved':
        raise ValueError('Independent plan approval required')
    if reg['script_sha256'] != digest(__file__):
        raise ValueError('Script changed after implementation review')
    if reg['execution'] != dict(impulse_ticks=128, distinct_tokens=3, phrase_tokens=8,
                               intervals=[1, 2, 4, 8, 16], cycles=6, tail_ticks=128):
        raise ValueError('Registered bounded design differs')
    torch.set_num_threads(1)
    source_hash = digest(args.checkpoint)
    if source_hash != reg['checkpoint_sha256']:
        raise ValueError('Wrong source individual')
    start = time.perf_counter()
    saved, model = load(args.checkpoint)
    cfg, life = saved['config'], saved['learner']
    data = np.load(ROOT / cfg['data'] / 'train.npy', mmap_mode='r')
    cursor = int(saved['train_cursor'])
    if int(data[cursor]) != life['previous_token']:
        raise ValueError('Stream cursor mismatch')
    selected = list(dict.fromkeys(int(t) for t in data[cursor:cursor + 128]))[:3]
    phrase = np.asarray(data[cursor:cursor + 8], dtype=np.int64).tolist()
    report = {'scope': 'Frozen mature-state mechanism response; no capability or online-learning claim',
              'source_checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': source_hash,
              'script_sha256': digest(__file__), 'registry_sha256': digest(args.registry),
              'cumulative_tokens': int(saved['bptt_train_tokens']), 'stream_cursor': cursor,
              'tokens': selected, 'phrase': phrase, 'optimizer_updates': 0,
              'new_checkpoints': 0, 'weights_frozen': True,
              'physical_units': 'model ticks; no validated biological second conversion'}
    print('Checking directed existing pathways...', flush=True)
    report['reachability'] = reachability(model)
    print(json.dumps(report['reachability']), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    model.cuda()
    initial = FlyPhysicalState(**{k: tuple(t.cuda() for t in v) if k == 'ring'
                                 else v.cuda() for k, v in life['physical'].items()})
    options = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(),
                   conductance_gains=model.get_conductance_gains(),
                   alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
    torch.cuda.reset_peak_memory_stats()
    step_original = model.step
    observation = {}
    def traced_step(*pos, **kw):
        ret = step_original(*pos, **kw)
        observation['spikes'] = ret[1]
        return ret
    model.step = traced_step
    dummy = torch.tensor([phrase[0]], device='cuda')

    def memory_gate():
        torch.cuda.synchronize()
        reserved = torch.cuda.memory_reserved() / 2**20
        if reserved > 3900:
            raise RuntimeError('Diagnostic VRAM exceeds conservative 3900 MiB guard')
        free, total = torch.cuda.mem_get_info()
        if total - free > 4 * 2**30:
            raise RuntimeError('Total device memory usage exceeds 4 GiB')

    with torch.no_grad():
        event = advance_fly_input_event(model, clone(initial), dummy, settle_ticks=0,
                                       writer_baseline_clock='input', **options)
        manual = quiet(model, event, dummy, options)
        native = advance_fly_input_event(model, clone(initial), dummy, settle_ticks=1,
                                        writer_baseline_clock='input', **options)
        # GPU scatter may vary; compare against same-operation replay too.
        native2 = advance_fly_input_event(model, clone(initial), dummy, settle_ticks=1,
                                         writer_baseline_clock='input', **options)
        err, replay = distances(manual, native), distances(native2, native)
        report['interface_all_field_max_error'] = err
        report['interface_same_input_replay_max_error'] = replay
        if any(err[k] > max(1e-5, 8 * replay[k]) for k in err):
            raise ValueError('Quiet interface mismatch exceeds same-input numerical floor')
        memory_gate()
    counter = [0]

    @torch.no_grad()
    def run(schedule, title):
        s = clone(initial)
        collected = {k: [] for k in ('raw', 'centered', 'z1', 'z2', 'projected', 'normalized',
                                    'baseline', 'firing', 'sensory_firing', 'motor_firing',
                                    'ring_norm', 'h_norm', 'drive_squared_norm')}
        for tick, token in enumerate(schedule):
            if token is None:
                drive_sq = 0.
                s = quiet(model, s, dummy, options)
            else:
                ids = torch.tensor([token], device='cuda')
                drive, _ = model.topographic_writer.forward_with_state(model.embedding(ids), s.h, s.baseline)
                drive_sq = float(drive.square().sum())
                s = advance_fly_input_event(model, s, ids, settle_ticks=0,
                                           writer_baseline_clock='input', **options)
            raw = s.h[:, model.read_indices]
            centered = (s.h - s.h_mean)[:, model.read_indices]
            projected = model.output_read(s.gamma_z2)
            normalized = model.read_norm(projected)
            for k, v in (('raw', raw), ('centered', centered), ('z1', s.gamma_z1),
                         ('z2', s.gamma_z2), ('projected', projected), ('normalized', normalized),
                         ('baseline', s.baseline)):
                if not torch.isfinite(v).all():
                    raise ValueError('Non-finite response')
                # Baseline only needs magnitude, not a 30k-vector stored every tick.
                value = v.norm().reshape(1) if k == 'baseline' else v.reshape(-1)
                collected[k].append(value.cpu().numpy())
            sp = observation['spikes']
            for k, v in (('firing', sp.mean()), ('sensory_firing', sp[:, model.injection_index].mean()),
                         ('motor_firing', sp[:, model.read_indices].mean()),
                         ('ring_norm', torch.stack([r.square().sum() for r in s.ring]).sum().sqrt()),
                         ('h_norm', s.h.norm())):
                collected[k].append(float(v))
            collected['drive_squared_norm'].append(drive_sq)
            counter[0] += 1
            if tick % 64 == 0:
                if any(not torch.isfinite(t).all() for k, v in s.state_dict().items()
                       for t in (v if k == 'ring' else (v,))):
                    raise ValueError('Non-finite complete physical state')
                memory_gate()
                print(f'{title}: {tick + 1}/{len(schedule)} ticks; total {counter[0]}', flush=True)
        return {k: np.asarray(v) for k, v in collected.items()}

    def decoder_compare(a, b):
        rows = []
        with torch.no_grad():
            for i in range(len(a['normalized'])):
                qa = torch.as_tensor(a['normalized'][i], device='cuda').unsqueeze(0)
                qb = torch.as_tensor(b['normalized'][i], device='cuda').unsqueeze(0)
                la, lb = model.decoder(qa), model.decoder(qb)
                delta = (la - lb).double()
                delta -= delta.mean(-1, keepdim=True)
                pa, pb = la.double().softmax(-1), lb.double().softmax(-1)
                skl = .5 * ((pa - pb) * (la.double().log_softmax(-1) - lb.double().log_softmax(-1))).sum()
                rows.append((float(delta.norm()), float(skl)))
        return dict(centered_logits_difference_norm=[x[0] for x in rows],
                    symmetric_probability_kl=[x[1] for x in rows])

    base = run([None] * 128, 'impulse quiet control')
    repeat = run([None] * 128, 'identical quiet replay')
    floor = compare(base, repeat)
    floor.update(decoder_compare(base, repeat))
    report['quiet_replay_numerical_floor'] = floor
    report['impulses'] = []
    impulse_arrays = []
    for token in selected:
        a = run([token] + [None] * 127, f'impulse token {token}')
        aa = run([token] + [None] * 127, f'impulse token {token} identical replay')
        row = {'token': token, 'stimulus_tick': 0, 'ticks': 128, 'response': compare(a, base),
               'same_input_replay': compare(a, aa), 'drive_squared_norm': float(a['drive_squared_norm'].sum())}
        row['response'].update(decoder_compare(a, base))
        row['same_input_replay'].update(decoder_compare(a, aa))
        for key in ('raw', 'centered', 'z1', 'z2', 'projected', 'normalized'):
            values = np.array(row['response'][key + '_difference_norm'])
            noise = np.maximum(row['same_input_replay'][key + '_difference_norm'], floor[key + '_difference_norm'])
            resolved = values > np.maximum(1e-5, 8 * noise)
            row.setdefault('summary', {})[key] = {'peak_tick': int(values.argmax()), 'peak_norm': float(values.max()),
                'first_resolved_tick': int(np.flatnonzero(resolved)[0]) if resolved.any() else None,
                'resolved_ticks': int(resolved.sum()), 'contrast_to_replay_at_peak': float(values.max() / max(float(noise[values.argmax()]), 1e-12)),
                'response_energy_centroid_tick': float(np.dot(np.arange(128), values**2) / max(np.dot(values, values), 1e-30)),
                'peak_at_horizon_end': bool(values.argmax() == 127)}
        impulse_arrays.append(a)
        report['impulses'].append(row)
    report['different_token_contrast'] = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        row = {'tokens': [selected[i], selected[j]], 'response': compare(impulse_arrays[i], impulse_arrays[j])}
        row['response'].update(decoder_compare(impulse_arrays[i], impulse_arrays[j]))
        report['different_token_contrast'].append(row)
    del impulse_arrays

    report['periodic'] = []
    for p in (1, 2, 4, 8, 16):
        period, length = 8 * p, 48 * p
        schedule = [phrase[(t // p) % 8] if t % p == 0 else None for t in range(length)] + [None] * 128
        control = run([None] * len(schedule), f'period {period} quiet control')
        control_repeat = run([None] * len(schedule), f'period {period} quiet control identical replay')
        a = run(schedule, f'period {period} drive')
        aa = run(schedule, f'period {period} identical replay')
        row = {'token_interval_ticks': p, 'phrase_period_ticks': period, 'cycles': 6,
               'stimulus_events': 48, 'drive_ticks': length, 'tail_ticks': 128,
               'response': compare(a, control), 'same_input_replay': compare(a, aa),
               'quiet_control_same_input_replay': compare(control, control_repeat),
               'actual_drive_squared_norm': float(a['drive_squared_norm'].sum()),
               'squared_drive_per_stimulus': float(a['drive_squared_norm'].sum() / 48),
               'squared_drive_per_tick_during_drive': float(a['drive_squared_norm'].sum() / length),
               'cycle_comparison': []}
        for key in ('raw', 'centered', 'z2', 'projected', 'normalized'):
            diff = a[key] - control[key]
            # Finite-amplitude response/actual drive. Not an LTI transfer function.
            centered = diff[:length] - diff[:length].mean(0)
            harmonic = (centered.astype(np.float64) * np.exp(-2j * np.pi * np.arange(length) / period)[:, None]).mean(0)
            input_harmonic = np.mean(a['drive_squared_norm'][:length] * np.exp(-2j * np.pi * np.arange(length) / period))
            response_norm = np.linalg.norm(diff[:length].astype(np.float64), axis=1)
            norm_harmonic = np.mean((response_norm - response_norm.mean()) * np.exp(-2j * np.pi * np.arange(length) / period))
            row.setdefault('response_statistics', {})[key] = {
                'rms_difference_per_coordinate_during_drive': float(np.sqrt(np.mean(diff[:length].astype(np.float64)**2))),
                'response_squared_sum_per_drive_squared_sum': float(np.sum(diff[:length].astype(np.float64)**2) / max(a['drive_squared_norm'].sum(), 1e-30)),
                'phrase_fundamental_vector_amplitude': float(np.linalg.norm(harmonic)),
                'input_squared_norm_phrase_fundamental_amplitude': float(abs(input_harmonic)),
                'norm_response_phase_relative_to_squared_drive_radians': float(np.angle(norm_harmonic / input_harmonic)) if abs(input_harmonic) > 1e-12 else None,
                'phase_scope': 'Amplitude-envelope phase, not semantic packet propagation delay',
                'tail_first_norm': float(np.linalg.norm(diff[length])), 'tail_last_norm': float(np.linalg.norm(diff[-1]))}
            for c in range(1, 6):
                before, now = diff[(c-1)*period:c*period], diff[c*period:(c+1)*period]
                row['cycle_comparison'].append({'stage': key, 'cycle': c + 1,
                    'previous_cycle_cosine': cosine(before, now),
                    'relative_cycle_change': float(np.linalg.norm(now - before) / max(np.linalg.norm(before), 1e-12)),
                    'same_input_repeat_relative_error': float(np.linalg.norm(a[key][c*period:(c+1)*period] - aa[key][c*period:(c+1)*period]) / max(np.linalg.norm(now), 1e-12)),
                    'combined_control_and_drive_repeat_relative_error_bound': float((np.linalg.norm(a[key][c*period:(c+1)*period] - aa[key][c*period:(c+1)*period]) + np.linalg.norm(control[key][c*period:(c+1)*period] - control_repeat[key][c*period:(c+1)*period])) / max(np.linalg.norm(now), 1e-12))})
        report['periodic'].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    report['elapsed_seconds'] = time.perf_counter() - start
    report['total_physical_ticks_executed'] = counter[0] + 6
    report['gpu_peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
    report['gpu_peak_reserved_mib'] = torch.cuda.max_memory_reserved() / 2**20
    report['initial_state_summary'] = {k: [list(t.shape) for t in v] if k == 'ring'
                                       else list(v.shape) for k, v in initial.state_dict().items()}
    report['source_checkpoint_unchanged'] = digest(args.checkpoint) == source_hash
    if not report['source_checkpoint_unchanged']:
        raise ValueError('Source checkpoint changed')
    report['interpretation_limits'] = ['Frozen mechanism response, not active-learning performance',
        'Token differences demonstrate sensitivity, not semantic decoding or next-token improvement',
        'Drive cadence changes event rate and STP/baseline adaptation; normalized ratios do not eliminate nonlinearity',
        'Six driven cycles do not certify convergence or endogenous resonance',
        'Same-input replay supplies numerical floor; nonlinear trajectories may amplify atomic summation noise',
        'No prediction-coding mechanism was implemented or trained']
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Done: {args.output}; {report["elapsed_seconds"]:.1f}s; {report["gpu_peak_reserved_mib"]:.1f} MiB reserved', flush=True)


if __name__ == '__main__':
    main()
