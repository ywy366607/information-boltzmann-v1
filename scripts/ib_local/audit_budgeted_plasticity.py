"""Real-layout optimizer arithmetic/overhead audit, no capability training.

No live checkpoint tensors, world calls, gradients or optimizer state are read
or modified. Timing uses fixed tensors of the current GDN layer shapes only.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from information_boltzmann.budgeted_adamw import BudgetedAdamW
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--repetitions', type=int, default=12)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    spec = json.loads((root / 'configs/information_boltzmann/mengli_gdn_l32_d256_20261114.json').read_text())
    with torch.device('meta'):
        body = GatedDeltaNetLM(**spec['backbone'])
    shapes = {n: list(p.shape) for n, p in body.named_parameters()}
    sample = {n: s for n, s in shapes.items() if n.startswith('layers.0.')}
    total = sum(p.numel() for p in body.parameters())
    allocated = sum(p.numel() for p in body.parameters()
                    if p.ndim == 2 and p.shape[0] % 64 == 0 and p.shape[1] % 64 == 0)
    units = sum((p.shape[0] // 64 * p.shape[1] // 64)
                if p.ndim == 2 and p.shape[0] % 64 == 0 and p.shape[1] % 64 == 0 else 1
                for p in body.parameters())
    # Four per-tile tensors: energy, phase, gate, inherited moment RMS.
    report = dict(scope='tensor_update_admission_not_language_capability', device=args.device,
                  body_parameters=total, budget_eligible_parameters=allocated,
                  observer_units=units, extra_state_bytes=16 * units,
                  live_body_modified=False, world_calls=0,
                  cooling_migration='old_moments_preserved_with_triangle_bound_and_new_observation_clock', arms={},
                  one_layer_shapes=sample,
                  caveat='live learner may contend for device; one-layer timing is not whole-loop timing')
    live = {}
    for name, allocation, cooling in [('adamw', False, False), ('budget', True, False),
                                      ('budget_cooling', True, True)]:
        torch.manual_seed(1115)
        params = [torch.nn.Parameter(torch.randn(s, device=args.device) * .02) for s in sample.values()]
        for p in params:
            p.grad = torch.randn_like(p) * .01
        opt = (torch.optim.AdamW(params, foreach=True) if name == 'adamw' else
               BudgetedAdamW(params, allocation=allocation, cooling=cooling))
        # Compile and warm each arm before interleaving; do not compare arms
        # that ran in different periods of the continuing learner's activity.
        for _ in range(5):
            opt.step()
        live[name] = (params, opt, [])
    if args.device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    names = list(live)
    for i in range(args.repetitions):
        for name in names[i % 3:] + names[:i % 3]:
            _, opt, durations = live[name]
            if args.device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            opt.step()
            if args.device == 'cuda':
                torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)
    for name, (params, opt, durations) in live.items():
        state_bytes = sum(v.numel() * v.element_size() for s in opt.state.values()
                          for v in s.values() if torch.is_tensor(v))
        report['arms'][name] = dict(median_step_ms=statistics.median(durations) * 1000,
            p90_step_ms=sorted(durations)[int(.9 * (len(durations) - 1))] * 1000,
            state_bytes=state_bytes,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0,
            finite_parameters=all(bool(torch.isfinite(p).all()) for p in params))
    report['interleaved'] = True
    report['source_hashes'] = {s: hashlib.sha256((root / s).read_bytes()).hexdigest() for s in [
        'information_boltzmann/budgeted_adamw.py',
        'information_boltzmann/budgeted_adamw_fused.py',
        'lean/CGBAxioms/CGBAxioms/GenerationBudgetedPlasticity.lean']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
