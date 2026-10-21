"""Finite-end 3D material audit; no target tree or neural capability claim."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.end_exchange import EndExchange
from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.spatial_exchange import SpatialExchangeParams, cubic_mesh
from scripts.ib.generative.compare_support_transport import digest, same
from scripts.ib.generative.driven_ports_audit import input_voltage


def construct(params, end_supply=1.6, contacts=True, seed=True):
    xyz, a, b = cubic_mesh(5)
    coordinates = np.repeat(xyz, 3, axis=0)
    # Three independent material domains share each voxel coordinate. Spatial
    # arms stay within their orientation; local bridges can join the domains.
    axis = np.argmax(np.abs(xyz[b]-xyz[a]), axis=1)
    pre, post = [3*a+axis], [3*b+axis]
    cells = np.arange(len(xyz))
    for u, v in ((0, 1), (0, 2), (1, 2)):
        pre.append(3*cells+u)
        post.append(3*cells+v)
    pre, post = np.concatenate(pre), np.concatenate(post)
    local = np.linalg.norm(coordinates[pre]-coordinates[post], axis=1) == 0.
    center = int(np.flatnonzero(np.all(xyz == [2., 2., 2.], axis=1))[0])
    initial = (~local)&((pre == 3*center)|(post == 3*center))
    installed = np.zeros(len(pre))
    installed[initial] = .4 if seed else 0.
    material = np.full(len(coordinates), .8)
    if not seed:
        material += np.bincount(post[initial], weights=np.full(initial.sum(), .4),
                                minlength=len(coordinates))
    ports = np.array([3*center, 3*center+1, 3*center+2, 0, 3*4+1,
                     3*20+2, 3*100, 3*124+2])
    scale = np.ones(len(pre))
    if not contacts:
        scale[local] = 0.
    m = EndExchange(pre, post, installed, np.full(len(coordinates), .1),
        np.zeros(len(pre)), np.zeros(len(coordinates)), np.full(len(coordinates), .8),
        np.full(len(coordinates), .2), np.ones(len(pre)), material,
        np.zeros(len(coordinates)), params=params, coordinates=coordinates,
        end_capacity=end_supply, reaction_scale=scale, trace=0., port_nodes=ports,
        world_capacitance=2., port_conductance=.05, source_conductance=.1)
    return m, local, initial


def measures(m, local, initial):
    incident = m._sum(m.pre, m.capacity)+m._sum(m.post, m.capacity)
    weights = m.capacity
    mid = .5*(m.coordinates[m.pre]+m.coordinates[m.post])
    centroid = np.average(mid, axis=0, weights=weights)
    return dict(installed=float(weights.sum()), bridge_occupancy=float(weights[local].sum()),
        spatial_occupancy=float(weights[~local].sum()), new_position_occupancy=float(weights[~initial].sum()),
        free_ends=float(m.remaining_ends().sum()), minimum_free_ends=float(m.remaining_ends().min()),
        maximum_end_utilization=float(np.max(incident/m.end_capacity)),
        bound=float(m.bound.sum()), trace=float(m.trace.sum()),
        radius_of_gyration=float(np.sqrt(np.average(np.sum((mid-centroid)**2, axis=1), weights=weights))),
        level_counts={str(cut): dict(bridges=int(np.sum(local&(weights >= cut))),
            spatial=int(np.sum(~local&(weights >= cut)))) for cut in (.1, .3, .5)})


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    shelf_path = ROOT/'data/harness/reading/shelf.json'
    story = json.loads(shelf_path.read_text(encoding='utf-8'))['entries'][0]
    data = story['text'].encode('utf-8')[:64]
    (args.out/'reading_shelf_snapshot.json').write_bytes(shelf_path.read_bytes())
    sources = [Path(__file__)] + [ROOT/'information_boltzmann/cgb'/name for name in (
        'end_exchange.py', 'spatial_exchange.py', 'trace_exchange.py', 'driven_exchange.py',
        'flow_exchange.py', 'structural_exchange.py', 'finite_exchange.py',
        'material_reactions.py', 'exchange_checkpoint.py')]
    sources += [ROOT/'docs/information_boltzmann/CGB_END_PAIRING_PREREG_20261022.md']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    snapshot = args.out/'source_snapshot'
    snapshot.mkdir()
    for p in sources:
        (snapshot/p.name).write_bytes(p.read_bytes())
    params = SpatialExchangeParams(assembly_rate=.05, conversion_rate=0., active_rate=0.,
        write_rate=.0005, material_diffusion=.005, support_diffusion=.0025,
        drive_diffusion=.0025, cohesion_energy=.6, bulk_attempt=.001, surface_attempt=2.)
    arms, frames, finals = {}, [], {}
    for label in ('coupled', 'no_contacts', 'no_capture', 'fewer_ends', 'no_seed'):
        p = replace(params, capture_coupling=0.) if label == 'no_capture' else params
        m, local, initial = construct(p, end_supply=1.0 if label == 'fewer_ends' else 1.6,
            contacts=label != 'no_contacts', seed=label != 'no_seed')
        before = measures(m, local, initial) if label != 'no_seed' else dict(installed=0.)
        initial_u, initial_g = m.energy(), m.free_energy()
        errors = {k: 0. for k in m.balances()}
        lowest, fork, scene = 0., None, []
        def record():
            scene.append(dict(ms=m.t, installed=float(m.capacity.sum()),
                material=float(m.material.sum()), source_work=m.source_work,
                density=np.round(m.capacity, 7).tolist(), trace=np.round(m.trace, 7).tolist()))
        record()
        started = time.perf_counter()
        for step in range(128):
            volts = input_voltage(data, step, 2.)
            change = m.step(2., world_voltage=volts)
            lowest = min(lowest, change['free_energy_drop'])
            for k, v in m.balances().items():
                errors[k] = max(errors[k], float(np.max(np.abs(v), initial=0.)))
            if fork is not None:
                fork.step(2., world_voltage=volts)
            if (step+1)%8 == 0:
                record()
            if step == 63:
                save_exchange(args.out/(label+'_half.npz'), m,
                    dict(next_step=64, dt_ms=2., byte_interval_ms=4., byte_stream_hex=data.hex()))
                if label == 'coupled':
                    fork, _ = load_exchange(args.out/(label+'_half.npz'))
        if fork is not None:
            same(m.state_dict(), fork.state_dict())
        assert max(errors.values()) < 1e-8 and lowest >= -1e-9
        archive = args.out/(label+'_complete.npz')
        save_exchange(archive, m, dict(next_step=128, dt_ms=2., byte_interval_ms=4.,
            byte_stream_hex=data.hex(), kind='finite_end_material_not_generated_neuron'))
        restored, _ = load_exchange(archive)
        same(m.state_dict(), restored.state_dict())
        arms[label] = dict(params=asdict(p), initial=before, final=measures(m, local, initial),
            elapsed_ms=m.t, wall_seconds=time.perf_counter()-started,
            initial_energy=initial_u, initial_free_energy=initial_g,
            material_total=float(m.material.sum()+m.capacity.sum()),
            free_material=float(m.material.sum()), charged=float(m.charged.sum()),
            source_work=m.source_work, trace_written=float(m.trace_written.sum()),
            maximum_balance_errors=errors, minimum_irreversible_g_drop=lowest,
            full_restore_loaded=True, future_bitwise_tested=fork is not None,
            archive=str(archive), sha256=digest(archive),
            physical_state_bytes=sum(getattr(m, k).nbytes for k in m._arrays)
                +sum(v.nbytes for v in (m.pre, m.post, m.capacity, m.port_nodes)))
        frames.append(dict(id=label, label=label, frames=scene))
        finals[label] = m
        print(json.dumps(dict(arm=label, final=arms[label]['final'],
            max_error=max(errors.values())), ensure_ascii=False), flush=True)
    control = finals['coupled']
    comparisons = {label: {name: float(np.max(np.abs(getattr(control, name)-getattr(m, name))))
        for name in ('capacity', 'bound', 'trace', 'charge')}
        for label, m in finals.items() if label != 'coupled'}
    report = dict(date='2026-10-22', scope='finite end composition, independent crossing domains and new bridges',
        baseline_git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        preregistered='docs/information_boltzmann/CGB_END_PAIRING_PREREG_20261022.md',
        nodes=control.n, candidate_positions=len(control.pre),
        colocated_candidate_contacts=int(local.sum()), initial_seed_bonds=int(initial.sum()),
        story=story['title'], story_source=story['source'], bytes_presented=len(data),
        byte_stream_hex=data.hex(), byte_ms=4., dt_ms=2., code_sha256=hashes,
        arms=arms, maximum_state_differences=comparisons,
        lineage_generation=False, directed_identity_matching_invoked=False,
        spatial_geometry='fixed precursor-port lattice; no continuous moving material ends yet',
        shape_or_learning_capability='unmeasured', passed=True)
    assert hashes == {str(p.relative_to(ROOT)): digest(p) for p in sources}
    (args.out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                allow_nan=False), encoding='utf-8')
    (args.out/'material_scene.json').write_text(json.dumps(dict(
        scope=report['scope'], coordinates=control.coordinates.tolist(), pre=control.pre.tolist(),
        post=control.post.tolist(), arms=frames, default='coupled',
        display='Independent colocated domains; density thresholds are display-only.'),
        ensure_ascii=False, separators=(',', ':')), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
