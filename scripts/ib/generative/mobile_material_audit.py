"""Actual finite material movement, encounters and same-network reactions."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.mobile_exchange import MobileExchange, MobileExchangeParams
from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from scripts.ib.generative.compare_support_transport import digest, same
from scripts.ib.generative.driven_ports_audit import input_voltage


def preparation(params, fewer_ends=False):
    rng = np.random.default_rng(61521)
    xyz = [np.array(v) for v in ([1.4, 2.5, 2.5], [2.2, 2.5, 2.5], [3., 2.5, 2.5])]
    while len(xyz) < 24:
        proposal = rng.uniform(.2, 4.8, 3)
        if min(np.linalg.norm(proposal-v) for v in xyz) >= params.core_diameter:
            xyz.append(proposal)
    return MobileExchange(xyz, np.full(24, 4), np.full(24, 2), np.full(24, 8),
        np.full(24, 2), end_capacity=2 if fewer_ends else 3, pre=[0, 1], post=[1, 2],
        params=params, port_nodes=np.arange(8), unit_conductance=.03, seed=7521)


def morphology(m):
    degree = m.degrees()
    adjacency = coo_matrix((np.r_[m.lengths(), m.lengths()],
        (np.r_[m.pre, m.post], np.r_[m.post, m.pre])), shape=(m.n, m.n)).tocsr()
    distance = dijkstra(adjacency, directed=False, indices=0)
    component = np.isfinite(distance)
    angles = []
    for node in range(m.n):
        neighbors = np.r_[m.post[m.pre == node], m.pre[m.post == node]]
        if len(neighbors) < 2: continue
        vectors = m.coordinates[neighbors]-m.coordinates[node]
        vectors /= np.linalg.norm(vectors, axis=1)[:, None]
        for a in range(len(vectors)):
            for b in range(a+1, len(vectors)):
                angles.append(float(np.arccos(np.clip(np.dot(vectors[a], vectors[b]), -1., 1.))))
    return dict(bridges=len(m.pre), used_material_units=int(m.capacity.sum()),
        free_ends=int(m.remaining_ends().sum()), branch_nodes=int(np.sum(degree >= 3)),
        isolated_cores=int(np.sum(degree == 0)), tips=int(np.sum(degree == 1)),
        seed_component_nodes=int(component.sum()),
        seed_component_path_extent=float(distance[component].max()),
        mean_pair_angle_radians=float(np.mean(angles)) if angles else None)


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__)] + [ROOT/'information_boltzmann/cgb'/name for name in (
        'mobile_exchange.py', 'spatial_exchange.py', 'trace_exchange.py', 'driven_exchange.py',
        'flow_exchange.py', 'structural_exchange.py', 'finite_exchange.py',
        'material_reactions.py', 'exchange_checkpoint.py')]
    source_paths += [ROOT/'docs/information_boltzmann/CGB_MOBILE_MATERIAL_PREREG_20261022.md']
    snapshot = args.out/'source_snapshot'
    snapshot.mkdir()
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    for p in source_paths:
        (snapshot/p.name).write_bytes(p.read_bytes())
    shelf = ROOT/'data/harness/reading/shelf.json'
    story = json.loads(shelf.read_text(encoding='utf-8'))['entries'][0]
    (args.out/'reading_shelf_snapshot.json').write_bytes(shelf.read_bytes())
    data = story['text'].encode('utf-8')[:64]
    base = MobileExchangeParams(assembly_rate=.05, bulk_attempt=.2, surface_attempt=1.,
        binding_rate=.05, transport_rate=.01, release_rate=.01,
        write_rate=.02, recovery_rate=.02, conversion_rate=.01, active_rate=.005,
        material_diffusion=0., support_diffusion=0., drive_diffusion=0.,
        mobility=.002, elastic_stiffness=4., cohesion_energy=.6)
    reports, scenes = {}, []
    for label in ('coupled', 'no_motion', 'no_capture', 'no_write', 'fewer_ends'):
        p = replace(base, **({'mobility': 0.} if label == 'no_motion' else
            {'capture_coupling': 0.} if label == 'no_capture' else
            {'write_rate': 0.} if label == 'no_write' else {}))
        m = preparation(p, fewer_ends=label == 'fewer_ends')
        xyz0 = m.coordinates.copy()
        initial_pairs = {tuple(pair) for pair in m.encounters().tolist()}
        before = morphology(m)
        initial_u, initial_g = m.energy(), m.free_energy()
        stats = dict(formations=0, new_spatial_pair_formations=0, free_core_attachments=0,
                     branch_formations=0, nucleations=0, joins_or_closures=0, removals=0)
        log_path = args.out/(label+'_events.jsonl')
        log = log_path.open('w', encoding='utf-8')
        def observer(event):
            if event['kind'] != 'assembly': return
            if event['sign'] < 0:
                stats['removals'] += 1
            else:
                stats['formations'] += 1
                pair = tuple(sorted((event['i'], event['j'])))
                if pair not in initial_pairs:
                    stats['new_spatial_pair_formations'] += 1
                a, b = event['degrees_before']
                if min(a, b) == 0 and max(a, b) >= 1: stats['free_core_attachments'] += 1
                if max(a, b) >= 2: stats['branch_formations'] += 1
                if a == b == 0: stats['nucleations'] += 1
                if min(a, b) >= 1: stats['joins_or_closures'] += 1
            log.write(json.dumps(event, separators=(',', ':'))+'\n')
        m.event_observer = observer
        errors = {k: 0. for k in m.balances()}
        frames, fork = [], None
        def record():
            frames.append(dict(ms=m.t, coordinates=np.round(m.coordinates, 7).tolist(),
                pre=m.pre.tolist(), post=m.post.tolist(), trace=m.trace.tolist(), bound=m.bound.tolist(),
                material=int(m.material.sum()), source_work=m.work_in, heat_out=m.heat_out,
                morphology=morphology(m)))
        record()
        started = time.perf_counter()
        for step in range(128):
            volts = input_voltage(data, step, 2.)
            m.step(2., world_voltage=volts)
            if fork is not None:
                fork.step(2., world_voltage=volts)
            for k, v in m.balances().items():
                errors[k] = max(errors[k], abs(float(v)))
            if (step+1)%8 == 0: record()
            if step == 63:
                path = args.out/(label+'_half.npz')
                save_exchange(path, m, dict(next_step=64, dt_ms=2., byte_interval_ms=4.,
                    byte_stream_hex=data.hex(), geometry='actual moving finite material'))
                if label == 'coupled': fork, _ = load_exchange(path)
        log.close()
        if fork is not None: same(m.state_dict(), fork.state_dict())
        path = args.out/(label+'_complete.npz')
        save_exchange(path, m, dict(next_step=128, dt_ms=2., byte_interval_ms=4.,
            byte_stream_hex=data.hex(), geometry='actual moving finite material'))
        restored, _ = load_exchange(path)
        same(m.state_dict(), restored.state_dict())
        assert max(errors.values()) < 1e-8
        reports[label] = dict(params=asdict(p), initial=before, final=morphology(m), events=stats,
            initial_energy=initial_u, initial_free_energy=initial_g, elapsed_ms=m.t,
            seconds=time.perf_counter()-started, source_work=m.work_in, heat_out=m.heat_out,
            motion_heat=m.motion_heat, reaction_heat=m.reaction_heat,
            source_heat=m.source_heat, conduction_heat=m.conduction_heat,
            free_material=int(m.material.sum()), bound_support=int(m.bound.sum()),
            occupied_trace=int(m.trace.sum()), reaction_counts=m.reaction_counts,
            motion_attempts=m.motion_attempts, motion_accepted=m.motion_accepted,
            rms_displacement=float(np.sqrt(np.mean(np.sum((m.coordinates-xyz0)**2, axis=1)))),
            maximum_balance_errors=errors, archive=str(path), sha256=digest(path),
            full_restore_loaded=True, future_bitwise_tested=fork is not None,
            state_bytes=sum(getattr(m, name).nbytes for name in m._arrays)
                +m.pre.nbytes+m.post.nbytes+m.capacity.nbytes+m.port_nodes.nbytes,
            current_encounter_pairs=len(m.encounters()))
        scenes.append(dict(id=label, frames=frames))
        print(json.dumps(dict(arm=label, final=reports[label]['final'], events=stats,
            rms=reports[label]['rms_displacement'], error=max(errors.values())), ensure_ascii=False), flush=True)
    assert hashes == {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    result = dict(date='2026-10-22', baseline_git=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        preregistered='docs/information_boltzmann/CGB_MOBILE_MATERIAL_PREREG_20261022.md',
        scope='mobile finite material with same reaction graph; no prescribed branch/turn command',
        nodes=24, real_text_source=story['source'], story=story['title'], bytes_presented=len(data),
        byte_stream_hex=data.hex(), dt_ms=2., byte_interval_ms=4., code_sha256=hashes, arms=reports,
        directed_matching_invoked=False, signed_predictive_learning=False,
        physical_assumptions='finite-valence cores; hard exclusion/container/extensibility; elastic bridges; isothermal bath',
        continuum_or_language_capability='not established by this finite numerical/interface audit', passed=True)
    (args.out/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (args.out/'mobile_scene.json').write_text(json.dumps(dict(box_size=base.box_size, arms=scenes,
        default='coupled', scope=result['scope']), ensure_ascii=False, separators=(',', ':')), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
