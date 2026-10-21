"""Preregistered remote-source transport interruption on a saved candidate.

Retained real text drives the unmodified medium; a passive support label keeps
new boundary input distinct from inherited inventory. No learning claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.checkpoint import load_medium, save_medium
from information_boltzmann.cgb.medium import Medium
from information_boltzmann.cgb.support_analysis import (
    choose_support_boundary, frozen_support_flow, reverse_distances)
from information_boltzmann.cgb.support_transport import SupportTransport, SupportTransportParams
from scripts.ib.generative.compare_support_transport import digest, same


def carrier(m):
    return np.maximum(m.z, 0.) if m.trophic.capture == 'tag' else m.x_pre.copy()


def label_inventory(m, fresh, f, t):
    return f + fresh.cargo + np.bincount(m.pre, weights=t, minlength=m.N)


def snapshot(m, fresh, f, t, distance):
    inv = label_inventory(m, fresh, f, t)
    owned_t = np.bincount(m.pre, weights=t, minlength=m.N)
    return dict(clock_ms=float(m.t), positive_capture_edges=int(np.count_nonzero(carrier(m)*m.trophic_edges)),
                source_imported=float(fresh.imported.sum()), fresh_inventory=float(inv.sum()),
                fresh_by_anatomical_distance={str(k): float(inv[distance == k].sum())
                                              for k in np.unique(distance)},
                fresh_at_least_two_hops=float(inv[distance >= 2].sum()),
                fresh_terminal_at_least_two_hops=float(owned_t[distance >= 2].sum()),
                fresh_terminal=float(t.sum()), fresh_rereleased=float(fresh.rereleased.sum()),
                total_structure=float(m.g.sum()),
                upstream_structure=float(m.g[distance[m.pre] >= 2].sum()))


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), ROOT/'information_boltzmann/cgb/support_analysis.py',
                    ROOT/'information_boltzmann/cgb/support_transport.py',
                    ROOT/'information_boltzmann/cgb/medium.py']
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    snap = args.out/'source_snapshot'
    snap.mkdir()
    for path in source_paths:
        (snap/path.name).write_bytes(path.read_bytes())
    m0, inherited = load_medium(args.archive)
    original_hash = digest(args.archive)
    if (m0.support_transport is not None or m0.trophic is None
            or m0.trophic.rule != 'constitutive' or m0.trophic.phi != 0.):
        raise ValueError('This preregistered source-label assay requires pre-transport constitutive phi=0 state')
    frozen_carrier = carrier(m0) * m0.trophic_edges
    target = choose_support_boundary(m0.N, m0.pre, m0.post, frozen_carrier > 0)
    distance = reverse_distances(m0.N, m0.pre, m0.post, target, m0.g > 0)
    capture_distance = reverse_distances(m0.N, m0.pre, m0.post, target, frozen_carrier > 0)
    q = np.zeros(m0.N)
    q[target] = m0.N * m0.trophic.q0
    params = SupportTransportParams(.001, .001, .00005)
    frozen = frozen_support_flow(m0.pre, m0.post, frozen_carrier, q,
                                kappa=m0.trophic.kappa, free_clearance=1/m0.trophic.tau_N,
                                terminal_clearance=1/m0.trophic.tau_T, params=params)
    report = dict(scope='Remote support delivery and structural dependence; not learning capability',
                  preregistered='docs/information_boltzmann/CGB_REMOTE_SUPPORT_PREREG_20261021.md',
                  source_archive=str(args.archive), source_sha256=original_hash,
                  inherited_provenance=inherited, cells=m0.N, edges=m0.E,
                  boundary_node=target, fixed_total_source_per_ms=float(q.sum()),
                  anatomical_distances=distance.tolist(), frozen_capture_distances=capture_distance.tolist(),
                  frozen=dict(positive_capture_edges=int(np.count_nonzero(frozen_carrier)),
                              two_hop_nodes=int((capture_distance >= 2).sum()),
                              max_one_hop_fraction=frozen['max_one_hop_fraction'],
                              residence_survival=frozen['residence_survival'],
                              direct_return=float(frozen['direct_return'].sum()),
                              two_hop_return=float(frozen['two_hop_return'].sum()),
                              predicted_total_loss=frozen['total_clearance'],
                              flow_residual=frozen['flow_residual']), arms={},
                  geometry_and_boundary_semantics='Historical empirical geometry; imposed support boundary for assay only',
                  world_consequence_learning=False, language_readout=False)
    shelf_path = ROOT/'data/harness/reading/shelf.json'
    shelf = json.loads(shelf_path.read_text(encoding='utf-8'))
    story = shelf['entries'][0]
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,!?;:]", story['text'])[:24]
    report.update(story=story['title'], words=words, shelf_sha256=digest(shelf_path),
                  elapsed_ms=4800., dt_ms=m0.dt, source_label='zero at start, linear phi=0 passive tag')
    drives = {}
    for word in set(words):
        seed = int.from_bytes(hashlib.blake2b(word.lower().encode(), digest_size=8).digest(), 'little')
        value = np.zeros(m0.N)
        value[np.random.default_rng(seed).choice(m0.N, 6, replace=False)] = .18
        drives[word] = value
    steps = round(4800. / m0.dt)
    if not np.isclose(steps*m0.dt, 4800.) or steps % 3:
        raise ValueError('Clock does not exactly divide the registered three phases')
    phase = steps // 3
    endpoints = {}
    for arm in ('local', 'continuous', 'sham', 'interruption'):
        started = time.perf_counter()
        m = Medium.from_state_dict(m0.state_dict())
        initial_params = replace(params, transport_rate=0.) if arm == 'local' else params
        m.enable_support_transport(initial_params, source=q)
        f, t = np.zeros(m.N), np.zeros(m.E)
        fresh = SupportTransport(m.N, m.pre, m.post, f, t, q, initial_params)
        max_balance = max_material = max_label = max_excess = 0.
        ever_positive = np.zeros(m.E, bool)
        windows = []
        for k in range(steps):
            if arm in ('sham', 'interruption') and k in (phase, 2*phase):
                par = replace(params, transport_rate=0.) if arm == 'interruption' and k == phase else params
                m.support_transport.params = par
                fresh.params = par
            current = drives[words[min(23, int(k*m.dt/200.))]]
            m.step(I_ext=current)
            actual_carrier = carrier(m)
            fresh.advance(f, t, actual_carrier, m.g, m.trophic_edges, m.dt, m.trophic)
            ever_positive |= (actual_carrier > 0) & m.trophic_edges
            if k % 100 == 0 or (k+1) % phase == 0:
                max_balance = max(max_balance, float(np.abs(m.support_transport.balance(m.Nf, m.T)).max()))
                max_material = max(max_material, float(np.abs(m.material.balance(m.g)).max()))
                max_label = max(max_label, float(np.abs(fresh.balance(f, t)).max()))
                max_excess = max(max_excess, float(np.max(f-m.Nf)), float(np.max(t-m.T)),
                                 float(np.max(fresh.cargo-m.support_transport.cargo)))
            if (k+1) % phase == 0:
                windows.append(snapshot(m, fresh, f, t, distance))
        metadata = dict(arm=arm, source_sha256=original_hash, scope=report['scope'], boundary=target)
        path = args.out/(arm+'_complete.npz')
        save_medium(path, m, metadata)
        restored, provenance = load_medium(path)
        same(m.state_dict(), restored.state_dict())
        assert provenance == metadata
        label_path = args.out/(arm+'_source_label.npz')
        label_state = fresh.state_dict()
        np.savez_compressed(label_path, free=f, bound=t, pre=m.pre, post=m.post,
                            metadata=np.array(json.dumps(dict(params=asdict(fresh.params), n=m.N,
                                                             schema=1, elapsed_ms=4800.))),
                            **label_state['arrays'])
        with np.load(label_path, allow_pickle=False) as z:
            state = dict(schema=1, n=m.N, pre=z['pre'], post=z['post'],
                         params=json.loads(str(z['metadata']))['params'],
                         arrays={name:z[name] for name in fresh._fields})
            restored_fresh = SupportTransport.from_state_dict(state, z['free'], z['bound'])
            same(fresh.state_dict(), restored_fresh.state_dict())
        report['arms'][arm] = dict(windows=windows, final=windows[-1],
            seconds=time.perf_counter()-started, positive_edges_ever=int(ever_positive.sum()),
            max_support_balance=max_balance, max_material_balance=max_material,
            max_label_balance=max_label, max_label_excess_over_actual=max_excess,
            complete_restore_bitwise=True, label_restore_bitwise=True,
            archive=str(path), archive_sha256=digest(path), label_archive=str(label_path),
            label_archive_sha256=digest(label_path))
        endpoints[arm] = (m, fresh, f, t)
        print(json.dumps({'arm':arm, **report['arms'][arm]}), flush=True)
    same(endpoints['continuous'][0].state_dict(), endpoints['sham'][0].state_dict())
    same(endpoints['continuous'][1].state_dict(), endpoints['sham'][1].state_dict())
    for i in (2, 3):
        np.testing.assert_array_equal(endpoints['continuous'][i], endpoints['sham'][i])
    assert report['arms']['local']['final']['fresh_at_least_two_hops'] == 0.
    assert all(a['max_support_balance'] < 1e-8 and a['max_material_balance'] < 1e-8
               and a['max_label_balance'] < 1e-8 and a['max_label_excess_over_actual'] < 1e-9
               for a in report['arms'].values())
    assert len({a['final']['source_imported'] for a in report['arms'].values()}) == 1
    base, interrupted = endpoints['continuous'][0], endpoints['interruption'][0]
    report['comparisons'] = dict(sham_bitwise=True,
        interrupted_structure_l1=float(np.abs(base.g-interrupted.g).sum()),
        interrupted_upstream_structure_l1=float(np.abs(base.g-interrupted.g)[distance[base.pre] >= 2].sum()),
        continuous_minus_interrupted_fresh_two_hop=float(report['arms']['continuous']['final']['fresh_at_least_two_hops']
                                                        - report['arms']['interruption']['final']['fresh_at_least_two_hops']))
    assert digest(args.archive) == original_hash
    assert source_hashes == {str(p.relative_to(ROOT)):digest(p) for p in source_paths}
    report.update(code_sha256=source_hashes, source_unchanged=True, passed=True,
                  capabilities_proven=False)
    (args.out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
