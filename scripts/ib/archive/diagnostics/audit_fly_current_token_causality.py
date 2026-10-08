"""CPU-only current-token timing audit of a saved continuing OWT individual.

This measures interface causality, not capability. The live learner and its
optimizer remain untouched. Two checkpoint forks differ in one real OWT input;
all subsequent inputs are identical. No cold start or new training is used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState, advance_fly_input_event
from information_boltzmann.core.fly_reservoir import FlyReservoirLM


def validation_reference(run: Path) -> list[dict]:
    """Same train-only add-one unigram prior used to initialize decoder bias."""
    train = np.load(ROOT / 'data/ib_owt_gpt2/train.npy', mmap_mode='r')
    val = np.load(ROOT / 'data/ib_owt_gpt2_31m/validation.npy', mmap_mode='r')
    counts = np.ones(50257, dtype=np.float64)
    for left in range(0, len(train), 1_000_000):
        counts += np.bincount(np.asarray(train[left:left+1_000_000], dtype=np.int64),
                              minlength=len(counts))
    surprisal = -np.log(counts / counts.sum())
    rows = []
    for line in (run / 'lifelong_evaluation.jsonl').read_text().splitlines():
        item = json.loads(line)
        if item['bptt_train_tokens'] < 200000:
            continue
        score = np.asarray(item['B_curve'], dtype=np.float64)
        end = item['fresh_validation_cursor']
        reference = surprisal[val[end-len(score):end]]
        gain = reference-score
        block_gain = gain.reshape(-1, 32).mean(1)
        rows.append({'bptt_train_tokens': item['bptt_train_tokens'],
                     'validation_range': [end-len(score), end],
                     'model_nll': float(score.mean()),
                     'fixed_unigram_nll': float(reference.mean()),
                     'gain_over_fixed_unigram': float(gain.mean()),
                     'gain_per_32_token_block': block_gain.tolist(),
                     'scope': 'different fresh text; unigram adjusts only marginal token difficulty'})
    return rows


def load_mapped_model(saved: dict) -> FlyReservoirLM:
    """Share immutable checkpoint parameter pages; allocate only graph buffers."""
    cfg = saved['config']
    with torch.device('meta'):
        model = FlyReservoirLM(ROOT / cfg['graph'], vocab_size=50257,
            d_model=cfg['d_model'], injection='topographic', read_surface='output',
            synapse_model='coba', use_alif=True, use_stp=True,
            decoder_bias=cfg['decoder_bias'],
            read_centering=saved['learner'].get('read_centering', cfg.get('read_centering', False)))
    model.load_state_dict({name: tensor for name, tensor in saved['model'].items()
                           if name not in ('edge_weight_e', 'edge_weight_i')}, strict=True, assign=True)
    for name in ('edge_weight_e', 'edge_weight_i'):
        setattr(model, name, saved['model'][name].detach())
    model.requires_grad_(False)
    model.dan_plastic_lr = saved['learner'].get('dan_plastic_lr', cfg.get('dan_plastic_lr', 0.0))
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path,
                        default=Path('results/q8_fly_bptt32_adamw_continuous_100k'))
    parser.add_argument('--events', type=int, default=8)
    parser.add_argument('--biophysics', action='store_true',
                        help='Trace sensory spike conversion and motor response before/after reset')
    parser.add_argument('--settle-ticks', type=int, default=None,
                        help='Default uses saved timing; explicit values inspect an alternative timing')
    parser.add_argument('--output', type=Path,
                        default=Path('results/published/fly_current_token_timing_audit.json'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = {'scope': 'CPU checkpoint-fork interface diagnostic; production state and checkpoints untouched',
              'validation_reference': validation_reference(args.run)}
    saved = torch.load(args.run / 'last.pt', map_location='cpu', mmap=True,
                       weights_only=False)
    cfg = saved['config']
    settle_ticks = saved['learner'].get('settle_ticks', 0) if args.settle_ticks is None else args.settle_ticks
    writer_clock = saved['learner'].get('writer_baseline_clock', 'physical')
    # Allocate only real graph buffers. The checkpoint's mapped CPU parameter
    # pages are assigned directly: initializing and copying a second vocabulary
    # matrix needlessly competes with the live learner for system commit space.
    model = load_mapped_model(saved)
    tick_trace = []
    if args.biophysics:
        original_step = model.step

        def traced_step(*positional, **options):
            ret, bio = original_step(*positional, **options, return_biophysics=True)
            motor = model.read_indices
            sensory = model.injection_index
            tick_trace.append({
                'motor_v_pre': bio['v_pre'][:, motor].clone(),
                'motor_h_post': ret[0][:, motor].clone(),
                'motor_spikes': ret[1][:, motor].clone(),
                'motor_transmitted_pulse': bio['transmitted_pulse'][:, motor].clone(),
                'motor_ge': ret[3][:, motor].clone(),
                'motor_gi': ret[4][:, motor].clone(),
                'sensory_spikes': ret[1][:, sensory].clone(),
                'sensory_transmitted_pulse': bio['transmitted_pulse'][:, sensory].clone(),
            })
            return ret

        model.step = traced_step
    raw = saved['learner']['physical']
    state_a = FlyPhysicalState(**{key: tuple(t.detach().clone() for t in value)
        if key == 'ring' else value.detach().clone() for key, value in raw.items()})
    state_b = state_a
    cursor = int(saved['train_cursor'])
    previous = int(saved['learner']['previous_token'])
    train = np.load(ROOT / cfg['data'] / 'train.npy', mmap_mode='r')
    alternate = next(int(token) for token in train[cursor+1:cursor+128]
                     if int(token) != previous)
    overlap = np.intersect1d(model.injection_index.numpy(), model.read_indices.numpy())
    report.update(checkpoint_bptt_tokens=int(saved['bptt_train_tokens']),
                  checkpoint_train_cursor=cursor,
                  checkpoint_events=int(saved['learner']['events']),
                  injected_neurons=int(model.n_injection), read_neurons=int(model.n_read),
                  input_output_overlap=int(len(overlap)),
                  settle_ticks=settle_ticks, physical_ticks_per_input=1+settle_ticks,
                  writer_baseline_clock=writer_clock,
                  parameter_storage='read-only mapped checkpoint tensors; no parameter initialization/copy',
                  changed_real_token_ids=[previous, alternate])
    events = []
    trajectory_motor, trajectory_latent, trajectory_normalized = [], [], []
    with torch.no_grad():
        for event in range(args.events):
            token_a = previous if event == 0 else int(train[cursor+event])
            token_b = alternate if event == 0 else token_a
            ids_a, ids_b = torch.tensor([token_a]), torch.tensor([token_b])
            drive_a, _ = model.topographic_writer.forward_with_state(
                model.embedding(ids_a), state_a.h, state_a.baseline)
            drive_b, _ = model.topographic_writer.forward_with_state(
                model.embedding(ids_b), state_b.h, state_b.baseline)
            state_a = advance_fly_input_event(model, state_a, ids_a, settle_ticks=settle_ticks,
                                            writer_baseline_clock=writer_clock)
            state_b = advance_fly_input_event(model, state_b, ids_b, settle_ticks=settle_ticks,
                                            writer_baseline_clock=writer_clock)
            physical_trace = []
            if args.biophysics:
                ticks = 1 + settle_ticks
                a_ticks, b_ticks = tick_trace[:ticks], tick_trace[ticks:]
                for tick, (trace_a, trace_b) in enumerate(zip(a_ticks, b_ticks)):
                    row = {'physical_tick_after_input': tick,
                           'difference_norms': {key: float((trace_a[key]-trace_b[key]).norm())
                                                for key in trace_a},
                           'motor_spike_fraction_a': float(trace_a['motor_spikes'].mean()),
                           'sensory_spike_fraction_a': float(trace_a['sensory_spikes'].mean()),
                           'motor_pre_reset_energy_fraction_removed_a': float(
                               (trace_a['motor_v_pre'].square()*trace_a['motor_spikes']).sum()
                               /trace_a['motor_v_pre'].square().sum().clamp_min(1e-30))}
                    physical_trace.append(row)
                tick_trace.clear()
            motor_a = state_a.h[:, model.read_indices]
            motor_b = state_b.h[:, model.read_indices]
            latent_a, latent_b = model.output_read(motor_a), model.output_read(motor_b)
            normalized_a, normalized_b = model.read_norm(latent_a), model.read_norm(latent_b)
            logits_a, logits_b = model.decoder(normalized_a), model.decoder(normalized_b)
            trajectory_motor.append(motor_a.clone())
            trajectory_latent.append(latent_a.clone())
            trajectory_normalized.append(normalized_a.clone())
            events.append({'lag_after_changed_input': event,
                'drive_difference_norm': float((drive_a-drive_b).norm()),
                'full_membrane_difference_norm': float((state_a.h-state_b.h).norm()),
                'motor_membrane_difference_norm': float((motor_a-motor_b).norm()),
                'logits_difference_norm': float((logits_a-logits_b).norm()),
                'logits_difference_max': float((logits_a-logits_b).abs().max()),
                'motor_relative_difference': float((motor_a-motor_b).norm()/((motor_a.norm()+motor_b.norm())*.5).clamp_min(1e-30)),
                'latent_relative_difference': float((latent_a-latent_b).norm()/((latent_a.norm()+latent_b.norm())*.5).clamp_min(1e-30)),
                'normalized_latent_relative_difference': float((normalized_a-normalized_b).norm()/((normalized_a.norm()+normalized_b.norm())*.5).clamp_min(1e-30)),
                'motor_spike_fraction_a': float((state_a.ring[0][:,model.read_indices]>0).float().mean())})
            if args.biophysics:
                events[-1]['physical_trace'] = physical_trace
    report['events'] = events
    report['actual_trajectory_variation'] = {}
    for name, rows in (('motor_membrane', trajectory_motor), ('projected_latent', trajectory_latent),
                       ('normalized_latent', trajectory_normalized)):
        values = torch.cat(rows)
        centered = values-values.mean(0, keepdim=True)
        report['actual_trajectory_variation'][name] = {
            'observed_events': len(values),
            'temporal_variation_energy_fraction': float(centered.square().sum()/values.square().sum().clamp_min(1e-30)),
            'total_rms': float(values.square().mean().sqrt()),
            'centered_rms': float(centered.square().mean().sqrt()),
            'scope': 'Short real-token checkpoint trajectory mechanism diagnostic; not intrinsic dimension or capability.'}
    report['first_nonzero_output_lag'] = next(
        (event['lag_after_changed_input'] for event in events
         if event['logits_difference_max'] != 0), None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
