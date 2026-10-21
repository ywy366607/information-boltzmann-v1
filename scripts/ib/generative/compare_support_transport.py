"""Paired transport toggle on an existing candidate, using retained real text.

This measures delivery, turnover, numerical cost and resume, NOT language
learning or functional module selection. No new birth or main-state migration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
from information_boltzmann.cgb.support_transport import SupportTransportParams


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            same(left, right)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


def support_bytes(obj):
    arrays = [obj.pre, obj.post] + [getattr(obj, k) for k in obj._fields]
    return sum(a.nbytes for a in arrays)


def main(args):
    if args.words <= 0 or args.word_ms <= 0:
        raise ValueError('Positive text exposure required')
    args.out.mkdir(parents=True, exist_ok=False)
    sources = (Path(__file__), ROOT/'information_boltzmann/cgb/medium.py',
               ROOT/'information_boltzmann/cgb/support_transport.py')
    source_digests = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    snapshot = args.out/'source_snapshot'
    snapshot.mkdir()
    for path in sources:
        (snapshot/path.name).write_bytes(path.read_bytes())
    original_digest = digest(args.archive)
    reference, inherited = load_medium(args.archive)
    if reference.support_transport is not None:
        raise ValueError('Use a pre-transport candidate so both arms share one ledger origin')
    if reference.N < 6:
        raise ValueError('Inherited fixed sensory transduction requires at least six cells')
    shelf_path = ROOT / 'data/harness/reading/shelf.json'
    shelf = json.loads(shelf_path.read_text(encoding='utf-8'))
    story = shelf['entries'][0]
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,!?;:]", story['text'])[:args.words]
    drive = {}
    for word in set(words):
        seed = int.from_bytes(hashlib.blake2b(word.lower().encode(), digest_size=8).digest(), 'little')
        value = np.zeros(reference.N)
        value[np.random.default_rng(seed).choice(reference.N, 6, replace=False)] = .18
        drive[word] = value
    params = SupportTransportParams(args.transport, args.release, args.cargo_clearance)
    steps = round(len(words) * args.word_ms / reference.dt)
    start_state = reference.state_dict()
    initial_g = reference.g.copy()
    report = {
        'scope': 'Same saved candidate and actual text; transport delivery/interface audit only',
        'pre_registered': 'docs/information_boltzmann/CGB_SUPPORT_TRANSPORT_PREREG_20261021.md',
        'source_archive': str(args.archive), 'source_sha256': original_digest,
        'inherited_provenance': inherited, 'cells': reference.N, 'edges': reference.E,
        'story': story['title'], 'story_source': story['source'],
        'shelf_sha256': digest(shelf_path), 'words_presented': words,
        'elapsed_ms': steps * reference.dt, 'dt_ms': reference.dt,
        'initial_bound_support': float(reference.T.sum()),
        'initial_free_support': float(reference.Nf.sum()),
        'initial_structure': float(initial_g.sum()),
        'fixed_sensory_transduction': 'Inherited six-cell hash drive .18, 200ms/token by default',
        'transport_candidate_params': asdict(params),
        'external_support': 'Inherited q0 at every original cell, identical in both arms; no per-layer source',
        'arms': {}, 'language_readout': False, 'world_consequence_learning': False,
        'hierarchical_function_or_self_proven': False,
    }
    for name, rate in (('local_conserved', 0.), ('reverse_transport', args.transport)):
        m = type(reference).from_state_dict(start_state)
        m.enable_support_transport(SupportTransportParams(rate, args.release, args.cargo_clearance))
        archive_meta = dict(scope=report['scope'], arm=name, source_sha256=original_digest,
                            story=story['title'], params=asdict(m.support_transport.params))
        fork = None
        max_balance, max_material = 0., 0.
        spikes = 0
        started = time.perf_counter()
        for k in range(steps):
            current = drive[words[min(len(words)-1, int(k * m.dt / args.word_ms))]]
            m.step(I_ext=current)
            spikes += int(m.spike.sum())
            if fork is not None:
                fork.step(I_ext=current)
            if k == steps // 2:
                path = args.out / (name + '_half.npz')
                save_medium(path, m, archive_meta)
                fork, restored_meta = load_medium(path)
                assert restored_meta == archive_meta
            if k % 100 == 0 or k == steps - 1:
                max_balance = max(max_balance, float(np.abs(m.support_transport.balance(m.Nf, m.T)).max()))
                if m.material is not None:
                    max_material = max(max_material, float(np.abs(m.material.balance(m.g)).max()))
        same(m.state_dict(), fork.state_dict())
        path = args.out / (name + '_complete.npz')
        save_medium(path, m, archive_meta)
        loaded, _ = load_medium(path)
        same(m.state_dict(), loaded.state_dict())
        chain = m.support_transport
        item = dict(
            simulated_ms=steps*m.dt, seconds_with_halfway_replay=time.perf_counter()-started,
            full_restore_future_bitwise=True, support_array_bytes=support_bytes(chain),
            archive_bytes=path.stat().st_size, support_balance_max=max_balance,
            material_balance_max=max_material, external_imported=float(chain.imported.sum()),
            captured=float(chain.captured.sum()), transported=float(chain.transported.sum()),
            internally_rereleased=float(chain.rereleased.sum()),
            cargo_positive_cells=int(np.count_nonzero(chain.cargo)),
            bound_support=float(m.T.sum()), free_support=float(m.Nf.sum()),
            cargo=float(chain.cargo.sum()), cleared=float(chain.cleared.sum()),
            maintenance_paid=float(chain.spent.sum()), spikes=spikes,
            structure_sum=float(m.g.sum()), structure_l1_from_start=float(np.abs(m.g-initial_g).sum()),
            complete_archive=str(path), complete_sha256=digest(path))
        report['arms'][name] = item
        print(json.dumps({'arm': name, **item}), flush=True)
    assert report['arms']['local_conserved']['transported'] == 0.
    assert report['arms']['local_conserved']['internally_rereleased'] == 0.
    assert report['arms']['reverse_transport']['transported'] > 0.
    assert report['arms']['reverse_transport']['internally_rereleased'] > 0.
    assert report['arms']['local_conserved']['external_imported'] == report['arms']['reverse_transport']['external_imported']
    assert all(v['support_balance_max'] < 1e-8 and v['material_balance_max'] < 1e-8 for v in report['arms'].values())
    assert digest(args.archive) == original_digest
    report['source_archive_unchanged'] = True
    assert source_digests == {str(p.relative_to(ROOT)): digest(p) for p in sources}
    report['code_sha256'] = source_digests
    report['passed'] = True
    (args.out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--words', type=int, default=24)
    parser.add_argument('--word-ms', type=float, default=200.)
    parser.add_argument('--transport', type=float, default=.001)
    parser.add_argument('--release', type=float, default=.001)
    parser.add_argument('--cargo-clearance', type=float, default=.00005)
    main(parser.parse_args())
