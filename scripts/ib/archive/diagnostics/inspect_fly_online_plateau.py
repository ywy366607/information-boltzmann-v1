"""Inspect a complete saved individual; no capability benchmark or live mutation.

The short real-OWT continuation preserves the production learning rule on an
isolated checkpoint fork. Counterfactuals test the instantaneous causal path.
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
from information_boltzmann.core.fly_online_learning import FlyOnlineLearner
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core import triton_online_credit as fused


def structure(values):
    matrix = torch.stack(values).double()
    mean = matrix.mean(0)
    centered = matrix - mean
    spectrum = torch.linalg.eigvalsh(centered @ centered.T).clamp_min(0)
    probability = spectrum / spectrum.sum().clamp_min(1e-30)
    active = probability > 0
    return {
        'centered_effective_rank': float((-(probability[active] * probability[active].log()).sum()).exp()),
        'mean_energy_fraction': float(mean.square().sum() / matrix.square().sum(-1).mean().clamp_min(1e-30)),
        'centered_variance': float(centered.square().sum(-1).mean()),
        'dimensions_measured': matrix.shape[1],
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'results/q8_fly_infinite_stream_three_factor_100k')
    parser.add_argument('--events', type=int, default=128)
    args = parser.parse_args()
    saved = torch.load(args.run / 'last.pt', map_location='cpu', weights_only=False)
    cfg = saved['config']
    model = FlyReservoirLM(ROOT / cfg['graph'], vocab_size=50257, d_model=cfg['d_model'],
                           injection='topographic', read_surface='output', synapse_model='coba',
                           use_alif=True, use_stp=True, decoder_bias=cfg['decoder_bias']).cuda()
    initial_edges = {name: getattr(model, name)[::128].clone()
                     for name in ('edge_weight_e', 'edge_weight_i')}
    weights = dict(saved['model'])
    for name in ('edge_weight_e', 'edge_weight_i'):
        getattr(model, name).copy_(weights.pop(name))
    model.load_state_dict(weights, strict=False)
    learner = FlyOnlineLearner(model, lr=cfg['lr'], lr_synapse=cfg['lr_synapse'],
        lr_sensory=cfg['lr_sensory'], grad_accum_tokens=cfg['grad_accum_tokens'],
        synapse_update_interval=cfg['synapse_update_interval'],
        dopamine_k_on=cfg['dopamine_k_on'], dopamine_k_off=cfg['dopamine_k_off'], dopamine_q0=cfg['dopamine_q0'])
    learner.load_state_dict(saved['learner'])
    cursor = saved['train_cursor']
    print(f'Inspecting complete checkpoint at {cursor}; isolated real-stream fork.', flush=True)
    del saved, weights
    train = np.load(ROOT / cfg['data'] / 'train.npy', mmap_mode='r')
    report = {'checkpoint_train_tokens': cursor, 'continuation_events': args.events,
              'scope': 'mechanism diagnostics only; live individual remains stopped and unmodified'}
    report['edge_change_since_graph_initialization'] = {
        name: {'relative_change': float((getattr(model, name)[::128] - old).norm() / old.norm()),
               'fraction_exactly_changed': float((getattr(model, name)[::128] != old).float().mean())}
        for name, old in initial_edges.items()}
    writer = model.topographic_writer
    injection = writer.injection_index
    output = model.read_indices
    report['write_read_overlap_neurons'] = int(model.read_mask[injection].sum())
    baseline = writer.a_adapt.clone()
    arguments = dict(spike_ring=learner.ring, **learner.syn, base_rates=learner.rates,
        thresholds=learner.threshold, conductance_gains=learner.gains,
        alif_params=learner.alif, stp_params=learner.stp)
    first = model.step(learner.h, torch.tensor([learner.previous_token], device='cuda'), **arguments)
    writer.a_adapt.copy_(baseline)
    alternative = int(train[cursor + 1])
    if alternative == learner.previous_token:
        alternative = int(train[cursor + 2])
    second = model.step(learner.h, torch.tensor([alternative], device='cuda'), **arguments)
    writer.a_adapt.copy_(baseline)
    report['same_event_token_intervention'] = {
        'observed_token': learner.previous_token, 'alternative_real_token': alternative,
        'sensory_voltage_difference_norm': float((first[0][:, injection] - second[0][:, injection]).norm()),
        'output_voltage_max_abs_difference': float((first[0][:, output] - second[0][:, output]).abs().max()),
        'logits_max_abs_difference': float((model.decoder(model.read_norm(model.output_read(first[0][:, output]))) -
            model.decoder(model.read_norm(model.output_read(second[0][:, output])))).abs().max()),
    }
    # Conditional physical impulse response on two isolated copies of this
    # mature state. Identical subsequent real tokens; no operator or live reset.
    responses = []
    branch_states = [first, second]
    branch_baselines = []
    for token in (learner.previous_token, alternative):
        writer.a_adapt.copy_(baseline)
        model.step(learner.h, torch.tensor([token], device='cuda'), **arguments)
        branch_baselines.append(writer.a_adapt.clone())
    for lag in range(32):
        difference = branch_states[0][0] - branch_states[1][0]
        responses.append({'lag_events': lag,
            'output_relative_voltage_difference': float(difference[:, output].norm() /
                branch_states[0][0][:, output].norm().clamp_min(1e-30)),
            'brain_relative_voltage_difference': float(difference.norm() /
                branch_states[0][0].norm().clamp_min(1e-30))})
        for branch in range(2):
            writer.a_adapt.copy_(branch_baselines[branch])
            state = branch_states[branch]
            branch_states[branch] = model.step(state[0], torch.tensor([int(train[cursor + lag + 1])], device='cuda'),
                spike_ring=state[2], **dict(zip(('ge', 'gi', 'b', 'x', 'u'), state[3:])),
                base_rates=learner.rates, thresholds=learner.threshold,
                conductance_gains=learner.gains, alif_params=learner.alif, stp_params=learner.stp)
            branch_baselines[branch] = writer.a_adapt.clone()
    writer.a_adapt.copy_(baseline)
    report['conditional_input_response'] = responses
    groups = {'all_brain_sample': torch.arange(0, model.n_neurons, 128, device='cuda'),
              'sensory_sample': injection[::16], 'output_surface': output}
    fields = {name: [] for name in groups}
    fields.update(pre_norm_latent=[], decoder_latent=[])
    activity, feedback, gate_values = [], [], []
    resolution = []
    projection_update = fused.update_projection_weights
    edge_update = fused.update_edge_traces
    def capture_projection(weight, eligibility, factor, lr, decay):
        if len(activity) == 16:
            indices = torch.arange(0, weight.numel(), 256, device='cuda')
            old = weight.flatten()[indices]
            change = -lr * (factor[indices // weight.shape[1]] * eligibility.flatten()[indices] + decay * old)
            ulp = (torch.nextafter(old, torch.full_like(old, float('inf'))) - old).abs()
            resolution.append({'kind': 'sensory_projection',
                'proposed_change_rms': float(change.square().mean().sqrt()),
                'median_float32_ulp': float(ulp.median()),
                'fraction_below_half_ulp': float((change.abs() < 0.5 * ulp).float().mean())})
        return projection_update(weight, eligibility, factor, lr, decay)

    def capture_edges(weight, pending, pre, post, L, phi, q, z, splits, lr, apply):
        if len(activity) in (16, 17) and apply:
            indices = torch.arange(0, weight.numel(), 256, device='cuda')
            source, destination = pre[indices].long(), post[indices].long()
            tier = sum((indices >= cutoff).long() for cutoff in splits[1:4])
            change = pending[indices] - lr * (L.reshape(-1) * phi.reshape(-1) * q)[destination] * z[0, tier, source]
            old = weight[indices]
            ulp = (torch.nextafter(old, torch.full_like(old, float('inf'))) - old).abs()
            resolution.append({'kind': 'recurrent_synapse',
                'proposed_change_rms': float(change.square().mean().sqrt()),
                'median_float32_ulp': float(ulp.median()),
                'fraction_below_half_ulp': float((change.abs() < 0.5 * ulp).float().mean())})
        return edge_update(weight, pending, pre, post, L, phi, q, z, splits, lr, apply)
    fused.update_projection_weights = capture_projection
    fused.update_edge_traces = capture_edges
    physical_step = model.step
    signal = learner.eprop.compute_learning_signal_whole_brain
    snapshots = {}
    for name, value in model.named_parameters():
        if name == 'embedding.weight':
            continue
        stride = max(1, value.numel() // 65536)
        snapshots[name] = (stride, value.flatten()[::stride].clone())
    for name in ('edge_weight_e', 'edge_weight_i'):
        value = getattr(model, name)
        stride = max(1, value.numel() // 65536)
        snapshots[name] = (stride, value[::stride].clone())

    def capture_physics(*positional, **keywords):
        result, bio = physical_step(*positional, **keywords)
        for name, indices in groups.items():
            fields[name].append(result[0][0, indices].cpu())
        z = model.output_read(result[0][:, output])
        fields['pre_norm_latent'].append(z[0].cpu())
        fields['decoder_latent'].append(model.read_norm(z)[0].cpu())
        indices = model.dan_indices
        activity.append({
            'whole_brain_spike_rate': float(result[1].mean()),
            'sensory_spike_rate': float(result[1][0, injection].mean()),
            'output_spike_rate': float(result[1][0, output].mean()),
            'dan_spike_rate': float(result[1][0, indices].mean()),
            'dan_threshold_margin_mean': float((bio['v_pre'] - bio['eff_threshold'])[0, indices].mean()),
            'dan_threshold_margin_max': float((bio['v_pre'] - bio['eff_threshold'])[0, indices].max()),
            'dan_voltage_mean': float(bio['v_pre'][0, indices].mean()),
            'dan_threshold_mean': float(bio['eff_threshold'][0, indices].mean()),
        })
        gate_values.append(writer.last_gates[0].cpu())
        return result, bio

    def capture_feedback(*positional, **keywords):
        result = signal(*positional, **keywords)
        L = result[0][0]
        feedback.append({'sensory_feedback_rms': float(L[injection].square().mean().sqrt()),
                         'output_feedback_rms': float(L[output].square().mean().sqrt())})
        return result

    model.step = capture_physics
    learner.eprop.compute_learning_signal_whole_brain = capture_feedback
    for i in range(args.events):
        target = int(train[cursor + i + 1])
        learner.step(torch.tensor([learner.previous_token], device='cuda'), torch.tensor([target], device='cuda'))
        learner.previous_token = target
    report['representation'] = {name: structure(values) for name, values in fields.items()}
    report['mean_activity'] = {key: sum(row[key] for row in activity) / len(activity) for key in activity[0]}
    report['mean_feedback'] = {key: sum(row[key] for row in feedback) / len(feedback) for key in feedback[0]}
    report['float32_update_resolution'] = resolution
    report['routing_gates_mean'] = torch.stack(gate_values).mean(0).tolist()
    report['routing_gates_std'] = torch.stack(gate_values).std(0).tolist()
    report['dopamine_max_occupancy'] = float(learner.dopamine.q.max())
    report['dopamine_max_release'] = float(learner.dopamine.dopamine_buffer.max())
    report['relative_parameter_changes_sampled'] = {}
    current = dict(model.named_parameters())
    for name in ('edge_weight_e', 'edge_weight_i'):
        current[name] = getattr(model, name)
    for name, (stride, old) in snapshots.items():
        new = current[name].flatten()[::stride]
        report['relative_parameter_changes_sampled'][name] = float((new - old).norm() / old.norm().clamp_min(1e-30))
    report['physical_parameter_ranges'] = {
        name: {'minimum': float(getattr(model, name).exp().min()),
               'maximum': float(getattr(model, name).exp().max()),
               'fraction_active_derivatives': float(learner.physical_masks[:, i].mean())}
        for i, name in enumerate(learner.physical_names)}
    report['gpu_peak_mib'] = torch.cuda.max_memory_allocated() / 2**20
    (args.run / 'plateau_mechanism_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
