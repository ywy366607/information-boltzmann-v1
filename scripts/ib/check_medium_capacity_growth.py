"""CPU numerical/cost acceptance on real checkpoint material geometry.

The credit is an explicit constructed coefficient sensitivity, not language
evidence. This script neither runs the language model nor changes its checkpoint.
Only small material tensors are cloned from a memory-mapped checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from information_boltzmann.core.plastic_medium import ContinuousMaterial
from information_boltzmann.core.structural_posterior import StructuralPosterior


def check(checkpoint, *, step_size, max_capacity_kl, repeats=3):
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    config, step = saved['config'], saved['step']
    source_prefix = 'medium.structural_posterior.'
    weights = {k.removeprefix(source_prefix): v.clone()
               for k, v in saved['model'].items() if k.startswith(source_prefix)}
    material_prefix = 'medium.material.'
    material_weights = {k.removeprefix(material_prefix): v.clone()
                        for k, v in saved['model'].items() if k.startswith(material_prefix)}
    del saved
    shape = config['constructor']['shape']
    options = dict(config['constructor']['structure_options'])
    options.pop('capacity_growth', None)
    growth_options = dict(step_size=step_size, max_capacity_kl=max_capacity_kl)
    posterior = StructuralPosterior(weights['mean'].shape[0], **options,
                                    capacity_growth=growth_options)
    missing, unexpected = posterior.load_state_dict(weights, strict=False)
    if unexpected or any(not key.startswith('capacity_growth.') for key in missing):
        raise ValueError('Source structural state does not match the candidate')
    if bool(posterior.window_active):
        raise ValueError('Read a completed-window checkpoint for numerical calibration')
    material = ContinuousMaterial(width=material_weights['coefficients'].shape[-1],
                                  modes=material_weights['modes'].tolist(),
                                  sine_modes=material_weights['sine_modes'].tolist())
    material.load_state_dict(material_weights)
    axes = [torch.arange(n, dtype=posterior.mean.dtype) / n for n in shape]
    coordinates = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1)
    basis = material.basis(coordinates).detach()
    posterior.begin_window(torch.zeros_like(posterior.mean))
    count = config['tokens_per_update']
    shares_before = posterior.allocation(basis).detach()
    maintenance = posterior.maintenance(shares_before, 1 / coordinates[..., 0].numel())
    posterior.record_window_evidence(count, config['event_duration'] * count, maintenance)
    # Exercise every coefficient direction with bounded signed sensitivity.
    posterior.mean.grad = torch.linspace(-1, 1, posterior.mean.numel()).reshape_as(posterior.mean)
    timings, proposal = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        proposal = posterior.capacity_growth.propose(posterior, basis, 1 / coordinates[..., 0].numel())
        timings.append(time.perf_counter() - start)
    posterior.validate_window_commit(posterior_mean=proposal['mean'],
                                     dual_learning_rate=config['structural_learning']['dual_learning_rate'])
    posterior.capacity_growth.apply(posterior, proposal)
    shares_after = posterior.allocation(basis).detach()
    torch.testing.assert_close(shares_after.sum(-1), shares_before.sum(-1), rtol=1e-6, atol=1e-6)
    report = {
        'scope': 'numerical geometry and cost only; constructed credit; no language run or capability claim',
        'source_checkpoint': str(checkpoint), 'source_step': step,
        'shape': shape, 'coefficient_count': posterior.coefficient_count,
        'metric_dimension': posterior.mean.numel(),
        'one_float64_metric_mib': posterior.mean.numel() ** 2 * 8 / 2 ** 20,
        'cpu_threads': torch.get_num_threads(), 'cuda_initialized': torch.cuda.is_initialized(),
        'timings_seconds': timings, 'median_seconds_per_update': statistics.median(timings),
        'gradient_scope': 'constructed numerical sensitivity, not observed task value',
        'new_learnable_parameters': 0,
        'resource_sum_max_error': float((shares_after.sum(-1) - posterior.resource_density).abs().max()),
        'acceptance': {key: value for key, value in proposal.items() if key not in ('mean', 'old_version')},
        'configuration': growth_options}
    if report['cuda_initialized']:
        raise RuntimeError('Acceptance unexpectedly initialized CUDA')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--step-size', type=float, required=True)
    parser.add_argument('--max-capacity-kl', type=float, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('Positive repeats required')
    torch.set_num_threads(1)
    report = check(args.checkpoint, step_size=args.step_size,
                   max_capacity_kl=args.max_capacity_kl, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
