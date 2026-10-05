"""CPU-only clock audit of the retained fly checkpoint and writer equations.

No model evolution, learning update, capability test or production-state change
is performed. Stationary packet calculations are analytic counterfactuals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_reservoir import BiologicalTopographicWriter


def writer_clock_coefficients(decay: float, quiet_ticks: int, clock: str) -> dict:
    """Exact scalar coefficients of one complete input event's baseline map."""
    if not 0 < decay < 1 or quiet_ticks < 0 or clock not in ('physical', 'input'):
        raise ValueError('Decay in (0,1), nonnegative quiet count and explicit clock required')
    quiet_decay = decay ** quiet_ticks if clock == 'physical' else 1.0
    carry = quiet_decay * decay
    packet = quiet_decay * (1.0 - decay)
    dc_baseline = packet / (1.0 - carry)
    return {'baseline_carry_per_input': carry,
            'baseline_packet_coefficient_per_input': packet,
            'stationary_baseline_fraction_of_constant_packet': dc_baseline,
            'stationary_innovation_fraction_of_constant_packet': 1.0 - dc_baseline,
            'baseline_efold_in_input_events': float(-1.0 / np.log(carry))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'results/q8_fly_bptt32_adamw_continuous_100k/last.pt'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/published/fly_input_physical_clock_audit.json'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    parameters = saved['model']
    learner = saved['learner']
    quiet_ticks = learner['settle_ticks']
    ticks = 1 + quiet_ticks
    # Obtain the implementation's actual writer decay rather than duplicating
    # its literal initializer in the audit.
    writer = BiologicalTopographicWriter(saved['config']['d_model'])
    decay = writer.lambda_adapt
    del writer
    clock = learner.get('writer_baseline_clock', 'physical')
    current = writer_clock_coefficients(decay, quiet_ticks, clock)
    event = writer_clock_coefficients(decay, quiet_ticks, 'input')
    tokens = torch.tensor(saved['recent_a'], dtype=torch.long)
    with torch.no_grad():
        embeddings = F.embedding(tokens, parameters['embedding.weight'])
        gates = 3.0 * F.linear(embeddings, parameters['topographic_writer.gate_linear.weight'],
                              parameters['topographic_writer.gate_linear.bias']).softmax(-1)
        packet = torch.cat([
            gates[:, i:i+1] * F.linear(embeddings, parameters[f'topographic_writer.proj_{name}.weight'])
            for i, name in enumerate(('vis', 'chemo', 'mech'))], -1)
        packet_mean = packet.mean(0)
    ranges = {'log_tau_m': (1., 250.), 'log_tau_s_e': (.5, 100.),
              'log_tau_s_i': (.5, 100.), 'log_tau_a': (2., 500.)}
    time_constants = {}
    for name, bounds in ranges.items():
        tau = parameters[name].detach().exp().clamp(*bounds)
        time_constants[name] = {
            'physical_tick_tau_min': float(tau.min()),
            'physical_tick_tau_max': float(tau.max()),
            'physical_tick_tau_mean_over_classes': float(tau.mean()),
            'source_free_efold_input_events_mean_over_classes': float(tau.mean()) / ticks,
            'input_event_time_ratio_to_one_tick_at_same_parameters': 1.0 / ticks,
            'scope': 'free component only; recurrent input, reset and conductance gating also affect memory'}
    report = {
        'checkpoint_train_targets': saved['bptt_train_tokens'],
        'physical_ticks_per_input': ticks,
        'writer_baseline_clock': clock,
        'writer_decay_from_implementation': decay,
        'current_clock_baseline': current,
        'input_event_clock_baseline': event,
        'archived_physical_clock_baseline': writer_clock_coefficients(decay, quiet_ticks, 'physical'),
        'recent_already_observed_packet_mean': {
            'tokens': len(tokens), 'packet_mean_norm': float(packet_mean.norm()),
            'packet_mean_rms': float(packet_mean.square().mean().sqrt()),
            'stationary_pulse_innovation_mean_norm_current_clock': float(packet_mean.norm())
                * current['stationary_innovation_fraction_of_constant_packet'],
            'scope': 'mean of actual retained training-token packets at saved weights; stationary filter value is analytic, not measured deployed NLL'},
        'time_constants': time_constants,
        'finding': ('Input-clock baseline is held during quiet propagation, retaining zero stationary DC innovation.'
                    if clock == 'input' else
                    'Archived quiet ticks decay the baseline toward a fictitious zero packet, losing zero DC response at pulse times.'),
        'scope': 'Implementation and analytic clock audit only. No learning, model evolution, GPU allocation or state modification; NLL causation is unproven.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
