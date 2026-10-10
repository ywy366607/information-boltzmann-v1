"""Autonomous 3D Self-Gravitational Morphogenesis Simulation (Generation 4).

Simulates pure self-gravitational collapse, Zel'dovich caustics, and cosmic web
morphogenesis (soma hubs, axonal filaments, sheets, and voids) on a periodic 3D torus.
Continuously exports live telemetry and 3D spatial fields to the ETHER dashboard.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from information_boltzmann.core.self_gravity import GravitationalCosmos3D


def atomic_json(target: Path, payload: dict):
    tmp = target.with_suffix(f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    for _ in range(50):
        try:
            os.replace(tmp, target)
            return
        except (PermissionError, OSError):
            time.sleep(0.02)
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape', type=int, nargs=3, default=[32, 32, 32], help='3D grid resolution')
    parser.add_argument('--steps', type=int, default=1000, help='Total physical evolution steps')
    parser.add_argument('--dt', type=float, default=0.04, help='Time step size')
    parser.add_argument('--G', type=float, default=1.0, help='Gravitational constant')
    parser.add_argument('--damping', type=float, default=0.02, help='Viscous damping for virialization')
    parser.add_argument('--pressure-cs', type=float, default=0.05, help='Barotropic sound speed for core stabilization')
    parser.add_argument('--power-index', type=float, default=-1.5, help='Gaussian perturbation power spectrum index')
    parser.add_argument('--seed', type=int, default=449, help='Random seed for initial Zel\'dovich perturbation')
    parser.add_argument('--output', type=Path, default=Path('results/medium_gravitational_cosmos_gen4'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-every', type=int, default=10, help='Publish spatial snapshot every N steps')
    parser.add_argument('--substeps', type=int, default=1, help='Substeps per reported cycle')
    parser.add_argument('--sleep-interval', type=float, default=0.05, help='Artificial sleep between steps for smooth browser observation')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"============================================================")
    print(f" ETHER · Generation 4: Self-Gravitational Morphogenesis")
    print(f" Grid: {args.shape[0]}x{args.shape[1]}x{args.shape[2]} | Steps: {args.steps} | Device: {args.device}")
    print(f" Output: {args.output}")
    print(f"============================================================")

    cosmos = GravitationalCosmos3D(
        shape=tuple(args.shape),
        G=args.G,
        dt=args.dt,
        damping=args.damping,
        pressure_cs=args.pressure_cs,
        power_index=args.power_index,
        seed=args.seed,
        device=args.device
    )

    config = {
        'protocol': 'self_gravitational_cosmos_v4',
        'shape': args.shape,
        'steps': args.steps,
        'dt': args.dt,
        'G': args.G,
        'damping': args.damping,
        'pressure_cs': args.pressure_cs,
        'power_index': args.power_index,
        'seed': args.seed,
        'device': args.device,
        'particles': math.prod(args.shape)
    }
    atomic_json(args.output / 'config.json', config)

    history_file = args.output / 'history.jsonl'
    # Start fresh or append
    history_handle = history_file.open('a', encoding='utf-8')

    start_wall = time.time()
    last_publish_wall = 0.0

    # Initial export at t=0
    initial_spatial = cosmos.export_ether_spatial_snapshot()
    initial_spatial['wall_time'] = time.time()
    atomic_json(args.output / 'spatial.json', initial_spatial)

    progress = {
        'status': 'training',
        'pid': os.getpid(),
        'step': 0,
        'fresh_training_tokens': 0,
        'physical_time': 0.0,
        'tokens_per_second': 0.0,
        'density_contrast': 0.0,
        'cosmic_web': initial_spatial['cosmic_web']
    }
    atomic_json(args.output / 'progress.json', progress)

    try:
        for current_step in range(1, args.steps + 1):
            t0 = time.time()
            diag = cosmos.advance(substeps=args.substeps)
            step_elapsed = time.time() - t0

            if args.sleep_interval > 0:
                time.sleep(args.sleep_interval)

            total_elapsed = time.time() - start_wall
            fps = current_step / max(1e-6, total_elapsed)

            # Record history entry
            history_entry = {
                'step': current_step,
                'physical_time': diag['time'],
                'density_contrast': diag['density_contrast'],
                'kinetic_energy': diag['kinetic_energy'],
                'potential_energy': diag['potential_energy'],
                'total_energy': diag['total_energy'],
                'virial_ratio': diag['virial_ratio'],
                'max_density': diag['max_density'],
                'train_prequential_nll': -math.log(max(1e-4, diag['max_density'] / 200.0)), # Virtual loss for dashboard chart
                'fixed_unigram_nll': 5.0,
                'spatial_share': min(1.0, diag['density_contrast'] / 5.0),
                'tokens_per_second': fps * 32.0
            }
            history_handle.write(json.dumps(history_entry) + '\n')
            history_handle.flush()

            # Periodic full spatial snapshot export
            if current_step % args.save_every == 0 or current_step == args.steps or (time.time() - last_publish_wall > 2.0):
                spatial = cosmos.export_ether_spatial_snapshot()
                spatial['step'] = current_step
                spatial['wall_time'] = time.time()
                atomic_json(args.output / 'spatial.json', spatial)

                web = spatial['cosmic_web']
                progress.update({
                    'status': 'training',
                    'step': current_step,
                    'fresh_training_tokens': current_step * 32,
                    'optimizer_updates': current_step,
                    'physical_time': diag['time'],
                    'tokens_per_second': fps * 32.0,
                    'density_contrast': diag['density_contrast'],
                    'virial_ratio': diag['virial_ratio'],
                    'max_density': diag['max_density'],
                    'cosmic_web': web,
                    'health': {
                        'structure': {
                            'field_spatial_fraction': min(1.0, diag['density_contrast'] / 5.0)
                        }
                    }
                })
                atomic_json(args.output / 'progress.json', progress)
                last_publish_wall = time.time()

                print(f"[Step {current_step:4d}/{args.steps}] t={diag['time']:.2f} | Contrast={diag['density_contrast']:.2f} | "
                      f"MaxRho={diag['max_density']:.1f} | Filaments={web['filaments_fraction']*100:.1f}% | "
                      f"Knots={web['knots_fraction']*100:.2f}% | Voids={web['voids_fraction']*100:.1f}% | {fps:.1f} steps/s")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user. Saving final state...")
    finally:
        history_handle.close()
        progress['status'] = 'completed'
        atomic_json(args.output / 'progress.json', progress)
        print("Final Gen 4 state saved cleanly.")


if __name__ == '__main__':
    main()
