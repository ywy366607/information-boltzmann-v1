"""Export small read-only CPU material maps when a training checkpoint changes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch


def export_maps(checkpoint):
    """Read only small coefficient tensors; never initialize a CUDA context."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=False)
    config = saved['config']
    constructor = config['constructor']
    shape = constructor['shape']
    # Clone only the small material maps before releasing the mapped file.
    wanted = ('medium.material.', 'medium.structural_posterior.',
              'medium.transport_shear.', 'medium.conductance_response.log_parameters.',
              'medium.short_term_plasticity.parameters_map.',
              'medium.conduction_plasticity.log_rate.')
    weights = {name: value.detach().numpy().copy() for name, value in saved['model'].items()
               if name.startswith(wanted)}
    step, fresh = saved['step'], saved['cursor'] - 1
    del saved
    axes = [np.arange(n) / n for n in shape]
    coordinates = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    key = 'medium.material.'
    basis = np.concatenate((np.ones((len(coordinates), 1)),
                            np.cos(2 * np.pi * coordinates @ weights[key + 'modes'].T),
                            np.sin(2 * np.pi * coordinates @ weights[key + 'sine_modes'].T)), -1)
    material = basis @ weights[key + 'coefficients']

    def linear(prefix):
        return material @ weights[prefix + '.weight'].T + weights.get(prefix + '.bias', 0.)

    prefix = 'medium.structural_posterior.'
    logits = basis @ weights[prefix + 'mean']
    logits = np.concatenate((logits, np.zeros((len(coordinates), 1))), -1)
    probabilities = np.exp(logits - logits.max(-1, keepdims=True))
    allocation = weights[prefix + 'resource_density'] * probabilities / probabilities.sum(-1, keepdims=True)
    shear = linear('medium.transport_shear')
    rows = np.broadcast_to(np.eye(3), (len(coordinates), 3, 3)).copy()
    rows[:, 1, 0], rows[:, 2, 0], rows[:, 2, 1] = shear.T
    rows /= np.linalg.norm(rows, axis=-1, keepdims=True)
    installed = float(weights[prefix + 'speed_reference']) * allocation[:, :3, None] * rows

    channels = constructor['channels']
    raw = linear('medium.conductance_response.log_parameters').reshape(-1, 14, channels)
    t0 = constructor.get('response_time_reference', 1.)
    c0 = constructor.get('capacitance_reference', 1.)
    # Per-site geometric channel summaries, explicitly labeled in the UI.
    fields = {'capacity': allocation[:, :3].sum(-1),
              'capacity_x': allocation[:, 0], 'capacity_y': allocation[:, 1],
              'capacity_z': allocation[:, 2],
              'idle': allocation[:, 3],
              'capacitance': c0 * np.exp(raw[:, 0].mean(-1)),
              'leak': c0 / t0 * np.exp(raw[:, 1].mean(-1)),
              'excitation': c0 / t0 * np.exp(raw[:, 2].mean(-1)),
              'inhibition': c0 / t0 * np.exp(raw[:, 3].mean(-1)),
              'membrane_time': t0 * np.exp((raw[:, 0] - raw[:, 1]).mean(-1)),
              'receptor_time': t0 * np.exp(-raw[:, 6:8].mean((1, 2))),
              'flux_time': t0 * np.exp((raw[:, 8:11] - raw[:, 11:14]).mean((1, 2))),
              'conduction_time': constructor.get('plasticity_time_reference', 1.) *
                  np.exp(-linear('medium.conduction_plasticity.log_rate').mean(-1))}
    stp = linear('medium.short_term_plasticity.parameters_map').reshape(-1, 3, 4)
    fields['stp_recovery_time'] = constructor.get('plasticity_time_reference', 1.) * np.exp(-stp[..., 0].mean(-1))
    return {'run': checkpoint.parent.name, 'checkpoint_step': step,
            'fresh_training_tokens': fresh, 'shape': shape,
            'wall_time': time.time(), 'checkpoint_mtime': checkpoint.stat().st_mtime,
            'scope': 'checkpoint posterior-mean installed structure; electrical channel geometric means',
            'fields': {name: value.tolist() for name, value in fields.items()},
            'installed_factor': installed.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=15.)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = args.run / 'last.pt'
    destination = args.run / 'material_view.json'
    previous = None
    while True:
        try:
            stat = checkpoint.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp != previous:
                data = export_maps(checkpoint)
                if (checkpoint.stat().st_mtime_ns, checkpoint.stat().st_size) == stamp:
                    temporary = destination.with_suffix(f'.{os.getpid()}.tmp')
                    temporary.write_text(json.dumps(data, allow_nan=False), encoding='utf-8')
                    os.replace(temporary, destination)
                    previous = stamp
                    print(f"Material view: step {data['checkpoint_step']}, fresh {data['fresh_training_tokens']}", flush=True)
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            if not args.watch:
                raise
            print(f'Waiting for complete checkpoint: {error}', flush=True)
        if not args.watch:
            return
        time.sleep(max(1., args.interval))


if __name__ == '__main__':
    main()
