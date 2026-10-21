"""Preregistered numerical balance and real V3d-to-V2 interface acceptance."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.birth import BirthParams, seed_from_contacts
from information_boltzmann.cgb.checkpoint import load_medium, save_medium
from information_boltzmann.cgb.medium import CellClass, Medium, SynapseParams, TrophicParams
from information_boltzmann.cgb.structural_budget import MaterialParams, StructuralBudget
from information_boltzmann.cgb.type_generator import TypeStrengthGenerator


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def numerical_balance(dt, supply):
    """Integrate the specified mass-transfer ODE, not a learning task."""
    p = MaterialParams(supply=supply, cost=2, maintenance=.002, recycle=.5)
    b = StructuralBudget(1, [0], np.zeros(1), p)
    g = np.zeros(1)
    lam, synthesis, capacity = .01, .1, 1.
    expected = supply*dt/(p.maintenance*dt + p.cost*(1-p.recycle)*(-np.expm1(-lam*dt)))
    continuous = supply/(p.maintenance+p.cost*(1-p.recycle)*lam)
    max_balance = 0.
    trajectory = []

    def advance(duration, phase):
        nonlocal g, max_balance
        for step in range(round(duration/dt)):
            b.advance(g, dt)
            g = b.transfer(g, g*np.exp(-dt*lam))
            request = capacity-(capacity-g)*np.exp(-synthesis*dt/capacity)
            g = b.transfer(g, request)
            max_balance = max(max_balance, float(np.max(np.abs(b.balance(g)))))
            if step % max(1, round(100/dt)) == 0:
                trajectory.append({'phase': phase, 'ms': step*dt, 'g': float(g[0])})

    advance(4000, 'supply')
    plateau = float(g[0])
    b.params = replace(p, supply=0)
    advance(4000, 'withdrawal')
    withdrawn = float(g[0])
    b.params = p
    advance(4000, 'restoration')
    restored = float(g[0])
    return {'dt': dt, 'supply': supply, 'expected_discrete_plateau': expected,
            'continuous_limit': continuous, 'measured_plateau': plateau,
            'relative_plateau_error': abs(plateau-expected)/expected,
            'continuous_limit_error': abs(plateau-continuous)/continuous,
            'withdrawn_structure': withdrawn, 'restored_structure': restored,
            'max_local_balance_error': max_balance, 'trajectory': trajectory,
            'passed': bool(abs(plateau-expected)/expected < .01 and
                           withdrawn < .01*plateau and abs(restored-plateau)/plateau < .01
                           and max_balance < 1e-10)}


def bytes_of_medium(m):
    # Deduplicate shared objects; sparse immutable delivery geometry included.
    seen, size = set(), 0
    values = list(m.__dict__.values()) + [m.to_post.data, m.to_post.indices, m.to_post.indptr]
    values += [getattr(m.material, name) for name in m.material._fields]
    values += [m.material.post]
    for value in values:
        if isinstance(value, np.ndarray) and id(value) not in seen:
            seen.add(id(value))
            size += value.nbytes
    return size


def real_interface(args):
    # This preparation reuses the already locked real connected-type-pair
    # population and its frozen train-only feature standardization. No fit.
    from scripts.ib.generative.v3d_stable_matching import prepare
    data = prepare(args.work)
    generator = TypeStrengthGenerator.load(args.generator)
    if not np.array_equal(generator.types, np.asarray(data['names']).astype(str)):
        raise ValueError('Deployment type ordering differs from geometry data')
    if not (np.array_equal(generator.values['feature_mean'], data['feature_mean']) and
            np.array_equal(generator.values['feature_scale'], data['feature_scale'])):
        raise ValueError('Deployment feature normalization differs from locked data')
    rng = np.random.default_rng(21)
    # The interface uses one node per type, so within-type pairs would be
    # self-loops here even though they can join distinct biological cells.
    # Exclude them rather than silently representing a many-cell type as one cell.
    eligible = np.flatnonzero(data['pre'] != data['post'])
    take = rng.choice(eligible, size=args.pairs, replace=False)
    raw = data['features'][take]*data['feature_scale'] + data['feature_mean']
    provenance = {'study': 'V2f numerical/interface acceptance', 'generator': str(args.generator),
                  'generator_sha256': digest(args.generator), 'real_data_sha256': data['files'],
                  'panel_seed': 21, 'panel_pairs': args.pairs,
                  'excluded_within_type_pairs': int(len(data['pre'])-len(eligible)),
                  'scope': 'Observed connected type-pair panel; not independent morphological birth'}
    medium, birth = seed_from_contacts(generator, np.arange(data['n']), [CellClass()],
        np.zeros(data['n'], int), data['pre'][take], data['post'][take], raw,
        np.ones(args.pairs), params=BirthParams(10, seed=21, dt=.25),
        syn=SynapseParams(gmax=.04, eta=4e-5, m0=.02, slide_A0=5,
                          aK=.2, thP=.7, nP=4, tau_bap=10, bap_amp=2),
        material=MaterialParams(supply=1e-5, clearance=.001,
                                maintenance=1e-4, recycle=.5), initial_free=.006)
    medium.trophic = TrophicParams(rule='constitutive', capture='tag',
                                    q0=.001, T_star=1, lambda0=1/20000)
    medium.trophic_edges[:] = True
    max_balance = 0.
    delivered = 0
    started = time.perf_counter()
    fork = None
    roundtrip_path = args.out/'real_interface_halfway.npz'
    for k in range(4000):
        if k == 2000:
            save_medium(roundtrip_path, medium, provenance)
            fork, restored_provenance = load_medium(roundtrip_path)
            if restored_provenance != provenance:
                raise RuntimeError('Checkpoint provenance did not round-trip')
        for obj in (medium,) if fork is None else (medium, fork):
            # A prescribed diagnostic spike drive, not a language curriculum or
            # proof of meaningful neural selectivity. RNG lives in the archive.
            forced = obj.rng.random(obj.N) < 4/1000*obj.dt
            obj.step(forced=forced)
        delivered += int(medium.spike.sum())
        max_balance = max(max_balance, float(np.max(np.abs(medium.material.balance(medium.g)))))
        if fork is not None:
            for name in Medium._state_arrays + Medium._physiology_arrays + ('g',):
                if not np.array_equal(getattr(medium, name), getattr(fork, name)):
                    raise RuntimeError('Resume mismatch: '+name)
            for name in medium.material._fields:
                if not np.array_equal(getattr(medium.material, name), getattr(fork.material, name)):
                    raise RuntimeError('Material resume mismatch: '+name)
    elapsed = time.perf_counter()-started
    saved = args.out/'real_interface_complete.npz'
    save_medium(saved, medium, provenance)
    rng_equal = medium.rng.bit_generator.state == fork.rng.bit_generator.state
    report = {'birth': birth, 'provenance': provenance, 'cells_as_type_interface_nodes': medium.N,
              'edges': medium.E, 'simulated_ms': medium.t, 'paired_resume_steps': 2000,
              'all_dynamic_arrays_bitwise_equal': True, 'rng_bitwise_equal': rng_equal,
              'max_local_balance_error': max_balance,
              'active_calcium_pools': int(np.count_nonzero(medium.c)),
              'nonzero_tags': int(np.count_nonzero(medium.z)),
              'diagnostic_spikes': delivered, 'structure_sum': float(medium.g.sum()),
              'material_imported': float(medium.material.imported.sum()),
              'material_maintenance_paid': float(medium.material.spent.sum()),
              'material_maintenance_unpaid': float(medium.material.unpaid.sum()),
              'medium_array_bytes': bytes_of_medium(medium),
              'complete_archive_bytes': saved.stat().st_size,
              'seconds_including_paired_restore_and_verification': elapsed,
              'complete_archive': str(saved),
              'passed': bool(rng_equal and max_balance < 1e-10 and
                             np.count_nonzero(medium.c) > 0 and np.count_nonzero(medium.z) > 0)}
    return report


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    models = [numerical_balance(1, .0003), numerical_balance(1, .00015),
              numerical_balance(.5, .0003)]
    ratio = models[1]['measured_plateau']/models[0]['measured_plateau']
    print(json.dumps({'analytic_checks_passed': all(m['passed'] for m in models),
                      'half_supply_ratio': ratio}), flush=True)
    interface = real_interface(args)
    source = [Path(__file__), ROOT/'information_boltzmann/cgb/medium.py',
              ROOT/'information_boltzmann/cgb/structural_budget.py',
              ROOT/'information_boltzmann/cgb/birth.py', ROOT/'information_boltzmann/cgb/checkpoint.py',
              ROOT/'docs/information_boltzmann/CGB_V2F_PREREG_20261021.md']
    report = {'status': 'completed', 'scope': 'Numerical and interface checks only',
              'numerical': models, 'half_supply_ratio': ratio, 'real_interface': interface,
              'source_sha256': {str(p): digest(p) for p in source},
              'seconds': time.perf_counter()-started,
              'passed': bool(all(m['passed'] for m in models) and .49 <= ratio <= .51 and
                             models[2]['continuous_limit_error'] < models[0]['continuous_limit_error']
                             and interface['passed'])}
    (args.out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'real_interface': interface,
                      'report': str(args.out/'report.json')}, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--generator', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=4096)
    main(parser.parse_args())
