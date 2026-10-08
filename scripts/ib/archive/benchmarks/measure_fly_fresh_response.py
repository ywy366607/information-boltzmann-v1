"""Fresh, untrained connectome response identification with real text currents.

No checkpoints, optimizer, random decoder scoring or production state changes.
The fixed graph is an anatomical prior; its physiological constants remain model
assumptions. Diagnostic forks are independent counterfactual initial conditions.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState, advance_fly_input_event


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def clone(state):
    return FlyPhysicalState(**{f.name: tuple(t.clone() for t in state.ring)
        if f.name == 'ring' else getattr(state, f.name).clone() for f in fields(state)})


def state_change(a, b):
    out = {}
    for f in fields(a):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if f.name == 'ring':
            va, vb = torch.cat(va, -1), torch.cat(vb, -1)
        out[f.name] = {'difference_norm': float((va-vb).norm()),
                      'reference_norm': float(vb.norm())}
    return out


def recurrence(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return {'cosine': float(np.sum(a*b)/(na*nb)) if na*nb > 0 else None,
            'relative_change': float(np.linalg.norm(a-b)/nb) if nb > 0 else None,
            'difference_norm': float(np.linalg.norm(a-b)),
            'reference_norm': float(nb)}


def memory_gate():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    if torch.cuda.memory_reserved() > 3900 * 2**20 or total-free > 4 * 2**30:
        raise RuntimeError('Fresh response diagnostic exceeded VRAM budget')


def parameter_digest(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(memoryview(p.detach().cpu().numpy()).cast('B'))
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--registry', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    reg = json.loads(args.registry.read_text(encoding='utf-8'))
    if reg['approval']['status'] != 'approved' or reg['script_sha256'] != digest(__file__):
        raise ValueError('Reviewed registry/source hash gate failed')
    graph, data_path = ROOT/reg['graph'], ROOT/reg['data']
    if digest(graph) != reg['graph_sha256']:
        raise ValueError('Graph prior changed')
    for source, expected in reg['source_hashes'].items():
        if digest(ROOT/source) != expected:
            raise ValueError(f'Registered source changed: {source}')
    data = np.load(data_path, mmap_mode='r')
    tokens = [int(t) for t in data[reg['data_offset']:reg['data_offset']+128]]
    phrase, selected = tokens[:8], list(dict.fromkeys(tokens))[:3]
    if tokens != reg['conditioning_tokens']:
        raise ValueError('Registered real-text stimuli changed')
    torch.set_num_threads(1)
    start = time.perf_counter()
    report = {'scope': 'Untrained material response, no language capability claim',
              'registry': reg, 'phrase': phrase, 'pulse_tokens': selected,
              'optimizer_updates': 0, 'checkpoint_reads': 0, 'checkpoint_writes': 0,
              'units': 'model ticks, not calibrated biological milliseconds', 'seeds': []}
    total_ticks = 0
    peak_reserved = 0
    for seed in reg['seeds']:
        print(f'Constructing fresh model seed {seed}', flush=True)
        torch.manual_seed(seed)
        model = FlyReservoirLM(graph_npz=graph, vocab_size=50257, d_model=768,
            injection='topographic', read_surface='output', read_centering=True,
            use_alif=True, use_stp=True, synapse_model='coba', use_read_gamma_trace=False)
        model.requires_grad_(False).eval()
        # Only the measured physical substrate and writer need device residency.
        # Vocabulary/read/decoder weights remain unchanged on CPU; embedding
        # hooks preserve exact lookup values at the native production interface.
        for name, value in list(model._buffers.items()):
            if value is not None:
                model._buffers[name] = value.cuda()
        for name, value in list(model._parameters.items()):
            if value is not None:
                model._parameters[name] = torch.nn.Parameter(value.cuda(), requires_grad=False)
        model.topographic_writer.cuda()
        model.embedding.register_forward_pre_hook(lambda module, inputs: (inputs[0].cpu(),))
        model.embedding.register_forward_hook(lambda module, inputs, output: output.cuda())
        weights_hash = parameter_digest(model)
        if torch.isin(model.read_indices, model.injection_index).any():
            raise ValueError('Input/read anatomical surfaces overlap')
        options = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(),
            conductance_gains=model.get_conductance_gains(), alif_params=model.get_alif_params(),
            stp_params=model.get_stp_params())
        z = torch.zeros(1, model.n_neurons, device='cuda')
        rest = FlyPhysicalState(z.clone(), tuple(z.clone() for _ in range(4)),
            z.clone(), z.clone(), z.clone(), torch.ones_like(z),
            options['stp_params'][0].clone(),
            torch.zeros(1, model.n_injection, device='cuda'), z.clone())
        dummy = torch.tensor([phrase[0]], device='cuda')
        packet_bank = {}
        with torch.no_grad():
            for tok in set(tokens):
                packet_bank[tok] = model.topographic_writer.forward_with_state(
                    model.embedding(torch.tensor([tok], device='cuda')), rest.h, rest.baseline)[0]
            leak = options['base_rates'][0]
            g_l = -torch.log(leak.clamp(min=1e-5, max=1.-1e-7))
            beta = (1.-torch.exp(-g_l))/g_l
            ratio = (beta*packet_bank[selected[0]]/options['thresholds'])[:, model.injection_index]
            positive = ratio[ratio > 0]
            scale = float(1./positive.median())
        seed_report = {'seed': seed, 'calibrated_amplitude': scale,
            'initial_parameter_sha256': weights_hash,
            'calibration': 'Median positive sensory first-tick v_pre/theta equals one at rest',
            'native_positive_ratio_median': float(positive.median()),
            'neuron_count': model.n_neurons, 'sensory_count': model.n_injection,
            'motor_count': model.n_read, 'impulses': [], 'periodic': []}
        seed_report['initial_physical_priors'] = {
            name: p.detach().cpu().flatten().tolist() for name,p in model.named_parameters()
            if name.startswith(('log_tau', 'log_threshold', 'log_beta', 'log_g_', 'logit_u0'))}
        seed_report['initial_physical_priors'].update(E_E=model.E_E, E_I=model.E_I,
            writer_baseline_decay=model.topographic_writer.lambda_adapt,
            excitatory_edges=int(model.edge_weight_e.numel()), inhibitory_edges=int(model.edge_weight_i.numel()))

        @torch.no_grad()
        def tick(s, token, amplitude=1., fixed_packet=False):
            nonlocal total_ticks
            baseline = s.baseline
            if token is None:
                drive = z
            elif fixed_packet:
                drive = amplitude*packet_bank[token]
            else:
                drive, baseline = model.topographic_writer.forward_with_state(
                    model.embedding(torch.tensor([token], device='cuda')), s.h, s.baseline)
                drive = amplitude*drive
            ret, bio = model.step(s.h, dummy, spike_ring=s.ring, ge=s.ge, gi=s.gi,
                b=s.b, x=s.x, u=s.u, sensory_drive=drive, return_biophysics=True, **options)
            h, spike, ring, ge, gi, b, x, u = ret
            out = FlyPhysicalState(h, ring, ge, gi, b, x, u, baseline,
                .99*s.h_mean+.01*h)
            total_ticks += 1
            return out, spike, bio, drive

        @torch.no_grad()
        def trajectory(initial, schedule, amplitude, fixed_packet=False, period=None):
            s = clone(initial)
            collected = {k: [] for k in ('motor_h', 'motor_pre', 'motor_ge', 'motor_gi',
                'firing', 'sensory_firing', 'motor_firing', 'drive_sq', 'active_count',
                'drive_mean_sensory', 'baseline_norm')}
            previous_boundary = None
            boundaries = []
            for i, token in enumerate(schedule):
                s, spike, bio, drive = tick(s, token, amplitude, fixed_packet)
                for name, value in (('motor_h', s.h), ('motor_pre', bio['v_pre']),
                                    ('motor_ge', s.ge), ('motor_gi', s.gi)):
                    collected[name].append(value[:, model.read_indices].flatten().cpu().numpy())
                for name, value in (('firing', spike.mean()),
                    ('sensory_firing', spike[:, model.injection_index].mean()),
                    ('motor_firing', spike[:, model.read_indices].mean()),
                    ('drive_sq', drive.square().sum()), ('active_count', spike.sum()),
                    ('drive_mean_sensory', drive[:, model.injection_index].mean()),
                    ('baseline_norm', s.baseline.norm())):
                    collected[name].append(float(value))
                if period and (i+1) % period == 0:
                    if previous_boundary is not None:
                        boundaries.append({'cycle': (i+1)//period,
                                           'fields': state_change(s, previous_boundary)})
                    previous_boundary = clone(s)
                if i % 128 == 0:
                    if any(not torch.isfinite(t).all() for f in fields(s)
                           for t in (s.ring if f.name == 'ring' else (getattr(s, f.name),))):
                        raise ValueError('Nonfinite physical state')
                    memory_gate()
            arrays = {k: np.asarray(v) for k,v in collected.items()}
            if any(not np.isfinite(v).all() for v in arrays.values()):
                raise ValueError('Nonfinite collected response')
            if any(not torch.isfinite(t).all() for f in fields(s)
                   for t in (s.ring if f.name == 'ring' else (getattr(s, f.name),))):
                raise ValueError('Nonfinite final complete state')
            return arrays, s, boundaries

        with torch.no_grad():
            # Check custom observation wrapper against the production single-event path.
            manual, _, _, _ = tick(clone(rest), phrase[0])
            native = advance_fly_input_event(model, clone(rest), dummy, settle_ticks=0,
                                            writer_baseline_clock='input', **options)
            total_ticks += 1
            seed_report['interface_error'] = state_change(manual, native)
            if any(v['difference_norm'] > 1e-5 for v in seed_report['interface_error'].values()):
                raise ValueError('Fresh native interface gate failed')
            _, conditioned, _ = trajectory(rest, tokens, 1.)
            for background_name, initial in (('rest', rest), ('text_conditioned', conditioned)):
                quiet, _, _ = trajectory(initial, [None]*128, 1.)
                quiet_replay, _, _ = trajectory(initial, [None]*128, 1.)
                for name, amplitude in (('native', 1.), ('calibrated', scale)):
                    token_responses = []
                    for tok in selected:
                        schedule = [tok]+[None]*127
                        a, _, _ = trajectory(initial, schedule, amplitude)
                        replay, _, _ = trajectory(initial, schedule, amplitude)
                        row = {'background': background_name, 'amplitude_kind': name,
                            'amplitude': amplitude, 'token': tok, 'stages': {},
                            'sensory_first_tick_firing': float(a['sensory_firing'][0]),
                            'motor_peak_firing': float(a['motor_firing'].max()),
                            'sensory_firing_by_tick': a['sensory_firing'].tolist(),
                            'motor_firing_by_tick': a['motor_firing'].tolist(),
                            'whole_brain_firing_by_tick': a['firing'].tolist(),
                            'drive_squared_sum': float(a['drive_sq'].sum())}
                        for stage in ('motor_h','motor_pre','motor_ge','motor_gi'):
                            response = np.linalg.norm(a[stage].astype(np.float64)-quiet[stage], axis=1)
                            floor = np.linalg.norm(a[stage].astype(np.float64)-replay[stage], axis=1)
                            quiet_floor = np.linalg.norm(quiet[stage].astype(np.float64)-quiet_replay[stage], axis=1)
                            combined_floor = floor + quiet_floor
                            resolved = response > np.maximum(1e-5, 8*combined_floor)
                            row['stages'][stage] = {'contrast_norm': response.tolist(),
                                'replay_norm': floor.tolist(),
                                'quiet_replay_norm': quiet_floor.tolist(),
                                'combined_replay_bound_norm': combined_floor.tolist(),
                                'first_resolved_tick': int(np.flatnonzero(resolved)[0]) if resolved.any() else None,
                                'peak_tick': int(response.argmax()), 'peak_norm': float(response.max()),
                                'last_norm': float(response[-1])}
                        seed_report['impulses'].append(row)
                        token_responses.append(a['motor_h'])
                    for i,j in ((0,1),(0,2),(1,2)):
                        seed_report.setdefault('different_token_motor_contrast', []).append({
                            'background': background_name, 'amplitude_kind': name,
                            'tokens': [selected[i], selected[j]],
                            'contrast_norm': np.linalg.norm(token_responses[i].astype(np.float64)
                                                           -token_responses[j], axis=1).tolist()})
                print(f'Seed {seed}: {background_name} pulse responses complete', flush=True)
            for interval in (1,2,4,8,16):
                period = 8*interval
                schedule = [phrase[(t//interval)%8] if t%interval==0 else None
                            for t in range(6*period)]
                for mode in ('native_writer', 'fixed_packet'):
                    a, end, boundary = trajectory(rest, schedule, 1., mode=='fixed_packet', period)
                    replay, replay_end, _ = trajectory(rest, schedule, 1., mode=='fixed_packet', period)
                    row = {'token_interval': interval, 'phrase_period': period,
                        'cycles': 6, 'drive_mode': mode, 'cycle_boundaries': boundary,
                        'actual_drive_squared_norm_by_cycle': a['drive_sq'].reshape(6,period).sum(1).tolist(),
                        'sensory_drive_signed_mean_by_cycle': a['drive_mean_sensory'].reshape(6,period).mean(1).tolist(),
                        'writer_baseline_norm_by_event': a['baseline_norm'][::interval].tolist(),
                        'complete_final_state_replay_floor': state_change(end, replay_end),
                        'boundary_noise_scope': 'Full-state replay floor measured at final boundary only; earlier boundary changes are descriptive',
                        'firing_mean': float(a['firing'].mean()),
                        'motor_firing_mean': float(a['motor_firing'].mean()), 'cycle_waveforms': []}
                    for stage in ('motor_h','motor_pre'):
                        for c in range(1,6):
                            current, before = a[stage][c*period:(c+1)*period], a[stage][(c-1)*period:c*period]
                            row['cycle_waveforms'].append({'stage': stage, 'cycle': c+1,
                                **recurrence(current, before),
                                'same_input_replay': recurrence(current, replay[stage][c*period:(c+1)*period])})
                    tail, _, _ = trajectory(end, [None]*128, 1.)
                    row['motor_h_norm_by_tick'] = np.linalg.norm(a['motor_h'], axis=1).tolist()
                    row['whole_brain_firing_by_tick'] = a['firing'].tolist()
                    row['tail_motor_h_norm'] = np.linalg.norm(tail['motor_h'], axis=1).tolist()
                    row['tail_firing'] = tail['firing'].tolist()
                    seed_report['periodic'].append(row)
                print(f'Seed {seed}: interval {interval} complete; total ticks {total_ticks}', flush=True)
            memory_gate()
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved()/2**20)
            seed_report['parameters_unchanged'] = parameter_digest(model) == weights_hash
            if not seed_report['parameters_unchanged']:
                raise ValueError('Parameters changed during fresh measurement')
        report['seeds'].append(seed_report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        # Release closure-held tensors before constructing another model.
        del model, options, rest, conditioned, z, packet_bank, a, replay, quiet, quiet_replay, end, tail, replay_end
        del manual, native, initial, beta, leak, g_l, ratio, positive
        gc.collect()
        torch.cuda.empty_cache()
    report.update(elapsed_seconds=time.perf_counter()-start, total_physical_ticks=total_ticks,
                  gpu_peak_reserved_mib=peak_reserved, complete=True)
    for source, expected in reg['source_hashes'].items():
        if digest(ROOT/source) != expected:
            raise ValueError(f'Registered source changed during run: {source}')
    if digest(graph) != reg['graph_sha256']:
        raise ValueError('Graph prior changed during run')
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Done: {total_ticks} ticks, {report["elapsed_seconds"]:.1f}s, {peak_reserved:.1f} MiB', flush=True)


if __name__ == '__main__':
    main()
