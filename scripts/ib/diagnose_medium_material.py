"""Read-only CPU audit of learned material, installed capacity and spatial order."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D


def distribution(values):
    values = np.asarray(values, dtype=float)
    return dict(zip(('min', 'p10', 'median', 'p90', 'max'),
                    np.quantile(values, [0, .1, .5, .9, 1]).tolist()),
                mean=float(values.mean()), std=float(values.std()),
                cv=float(values.std() / max(abs(values.mean()), 1e-30)))


def neighbor_correlation(values, shape):
    values = np.asarray(values).reshape(*shape, -1)
    centered = values - values.mean((0, 1, 2), keepdims=True)
    denominator = float((centered ** 2).sum())
    if denominator < 1e-25:
        return None
    return float(sum((centered * np.roll(centered, 1, a)).sum()
                     for a in range(3)) / (3 * denominator))


def components(mask):
    """Six-neighbor periodic site components, independent of view interpolation."""
    shape = mask.shape
    seen, sizes = set(), []
    for index in zip(*np.where(mask)):
        if index in seen:
            continue
        todo, count = [index], 0
        seen.add(index)
        while todo:
            node = todo.pop()
            count += 1
            for axis in range(3):
                for step in (-1, 1):
                    neighbor = list(node)
                    neighbor[axis] = (neighbor[axis] + step) % shape[axis]
                    neighbor = tuple(neighbor)
                    if mask[neighbor] and neighbor not in seen:
                        seen.add(neighbor)
                        todo.append(neighbor)
        sizes.append(count)
    return sorted(sizes, reverse=True)


def spatial_order(values, shape, rng, trials=499):
    values = np.asarray(values).reshape(-1)
    order = neighbor_correlation(values, shape)
    threshold = float(np.quantile(values, .8))
    mask = values > threshold
    sizes = components(mask.reshape(shape))
    null_order, null_largest = [], []
    for _ in range(trials):
        permutation = rng.permutation(len(values))
        null_order.append(neighbor_correlation(values[permutation], shape))
        shuffled = components(mask[permutation].reshape(shape))
        null_largest.append(shuffled[0] if shuffled else 0)
    largest = sizes[0] if sizes else 0
    return {'neighbor_correlation': order, 'top20_percent_threshold': threshold,
            'high_sites': int(mask.sum()), 'component_sizes': sizes,
            'largest_component': largest,
            'shuffle_largest_median': float(np.median(null_largest)),
            'shuffle_largest_p95': float(np.quantile(null_largest, .95)),
            'cluster_shuffle_p': (1 + sum(x >= largest for x in null_largest)) / (trials + 1),
            'correlation_shuffle_p': None if order is None else
                (1 + sum(x is not None and x >= order for x in null_order)) / (trials + 1)}


def summarize(model, state=None):
    medium = model.medium
    with torch.no_grad():
        prepared = medium.prepare_evolution()
        material = prepared.material
        installed = prepared.structural_factor
        factor = installed if state is None else medium.current_transport_factor(state, prepared)[0]
        electrical = prepared.response
        raw = medium.conductance_response.log_parameters(material).reshape(-1, 14, medium.channels).exp()
        coefficients = {}
        names = ['C', 'gL', 'gE_max', 'gI_max', 'E_E_magnitude', 'E_I_magnitude',
                 'closing_E', 'closing_I', 'L_x', 'L_y', 'L_z', 'R_x', 'R_y', 'R_z']
        for index, name in enumerate(names):
            field = raw[:, index].numpy()
            coefficients[name] = {'median_channel_spatial_cv': float(np.median(
                field.std(0) / field.mean(0))),
                'site_geomean': distribution(np.exp(np.log(field).mean(1)))}
        times = {'membrane_passive_C_over_gL': (electrical.capacitance / electrical.leak),
                 'receptor_E': 1 / electrical.closing[..., 0, :],
                 'receptor_I': 1 / electrical.closing[..., 1, :],
                 'flux_x_L_over_R': electrical.inductance[..., 0, :] / electrical.resistance[..., 0, :],
                 'flux_y_L_over_R': electrical.inductance[..., 1, :] / electrical.resistance[..., 1, :],
                 'flux_z_L_over_R': electrical.inductance[..., 2, :] / electrical.resistance[..., 2, :]}
        matrix = factor @ factor.transpose(-1, -2)
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        speeds = eigenvalues.clamp_min(0).sqrt().reshape(-1, 3).numpy()
        installed_speeds = torch.linalg.eigvalsh(
            installed @ installed.transpose(-1, -2)).clamp_min(0).sqrt().reshape(-1, 3).numpy()
        allocation = prepared.structural_allocation.reshape(-1, 4).numpy()
        shear = medium.transport_shear(material)
        tilts = torch.stack((torch.zeros_like(shear[..., 0]),
                             shear[..., 0].abs().atan(),
                             shear[..., 1:].norm(dim=-1).atan()), -1) * (180 / np.pi)
        conduction_gain, conduction_rate, _ = medium.conduction_plasticity.coefficients(material)
        stp_rates, stp_baseline = prepared.short_term
        result = {'material_channels': material.shape[-1],
                  'material_spatial_std': material.reshape(-1, material.shape[-1]).std(0, unbiased=False).tolist(),
                  'material_neighbor_correlation': neighbor_correlation(material.numpy(), medium.shape),
                  'electrical': coefficients,
                  'times': {k: distribution(v.numpy()) for k, v in times.items()},
                  'allocation': {k: distribution(allocation[:, i]) for i, k in
                                 enumerate(('active_x', 'active_y', 'active_z', 'idle'))},
                  'installed_row_speed': distribution(installed.norm(dim=-1).numpy()),
                  'actual_row_speed': distribution(factor.norm(dim=-1).numpy()),
                  'principal_speed': distribution(speeds),
                  'fastest_principal_speed': distribution(speeds[:, -1]),
                  'anisotropy_speed_ratio': distribution(speeds[:, -1] / np.maximum(speeds[:, 0], 1e-30)),
                  'installed_anisotropy_speed_ratio': distribution(installed_speeds[:, -1] / installed_speeds[:, 0]),
                  'learned_direction_tilt_degrees': {name: distribution(tilts[..., axis].numpy())
                                                   for axis, name in enumerate(('x', 'y', 'z'))},
                  'conduction': {'gain': distribution(conduction_gain.numpy()),
                                 'relaxation_time': distribution((1 / conduction_rate).numpy())},
                  'short_term_plasticity': {'recovery_time': distribution((1 / stp_rates[..., 0]).numpy()),
                                           'facilitation_time': distribution((1 / stp_rates[..., 1]).numpy()),
                                           'activity_rate': distribution(stp_rates[..., 2].numpy()),
                                           'baseline_utilization': distribution(stp_baseline.numpy())},
                  'activity_adaptation_gain': distribution(electrical.adaptation_gain.numpy()),
                  'shear': distribution(shear.numpy())}
        fields = {'installed_total_capacity': allocation[:, :3].sum(-1),
                  'fastest_principal_speed': speeds[:, -1]}
        return result, fields, material.detach().clone(), installed.detach().clone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(args.run / 'last.pt', map_location='cpu', weights_only=False, mmap=True)
    config = saved['config']
    root = Path(__file__).resolve().parents[2]
    source_matches = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
                      for name, digest in config['source_hashes'].items() if '/core/' in name}
    # Reconstruct declared seed initialization, without reading datasets, decoding
    # tokens, executing dynamics, touching CUDA, or modifying the live individual.
    torch.manual_seed(config['seed'])
    model = PlasticMediumPorts3D(**config['constructor'])
    if config['material_initialization'] == 'spectral-xavier':
        model.medium.material.initialize_spectral_xavier(field_std=config['material_field_std'])
    birth, _, birth_material, birth_factor = summarize(model)
    model.load_state_dict(saved['model'])
    from scripts.ib.train_plastic_conductance import unpack_belief
    belief = unpack_belief(saved['belief'], 'cpu')
    current, fields, material, factor = summarize(model, belief.medium)
    rng = np.random.default_rng(20261009)
    clusters = {k: spatial_order(v, model.medium.shape, rng) for k, v in fields.items()}
    report = {'run': args.run.name, 'checkpoint_step': saved['step'],
              'fresh_tokens': saved['cursor'] - 1,
              'optimizer_updates': saved['learner']['optimizer_updates'],
              'grid': list(model.medium.shape),
              'birth_reference': 'Declared seed reconstructed with current constructor and source; not an archived birth checkpoint',
              'birth_core_source_matches_training_config': source_matches,
              'birth': birth, 'current': current, 'spatial_order': clusters,
              'change_from_reconstructed_birth': {
                  'material_relative_rms': float((material - birth_material).norm() / birth_material.norm()),
                  'installed_factor_relative_rms': float((factor - birth_factor).norm() / birth_factor.norm())},
              'scope': 'Read-only CPU coefficient/state audit; clustering is geometric, not task causality or axon/dendrite identification'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('checkpoint_step', 'fresh_tokens',
                                          'change_from_reconstructed_birth', 'spatial_order')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
