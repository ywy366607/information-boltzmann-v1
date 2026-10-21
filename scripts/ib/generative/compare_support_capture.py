"""Post-registered activity-vs-tag capture audit at the identical boundary.

The tag arm is the already completed continuous branch. The activity arm
reuses Claude's existing x_pre carrier, no new reinforcement or tag floor.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.checkpoint import load_medium, save_medium
from information_boltzmann.cgb.support_transport import SupportTransport, SupportTransportParams
from scripts.ib.generative.compare_support_transport import digest, same
from scripts.ib.generative.remote_support_dependency import snapshot


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    ref = json.loads(args.reference.read_text(encoding='utf-8'))
    if not ref['passed'] or not ref['comparisons']['sham_bitwise']:
        raise ValueError('Require completed controlled tag reference')
    m, provenance = load_medium(ref['source_archive'])
    assert digest(ref['source_archive']) == ref['source_sha256']
    assert m.trophic.phi == 0 and m.trophic.capture == 'tag'
    paths = [Path(__file__), ROOT/'information_boltzmann/cgb/medium.py',
             ROOT/'information_boltzmann/cgb/support_transport.py',
             ROOT/'scripts/ib/generative/remote_support_dependency.py']
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in paths}
    source = args.out/'source_snapshot'
    source.mkdir()
    for path in paths:
        (source/path.name).write_bytes(path.read_bytes())
    m.trophic.capture = 'activity'
    params = SupportTransportParams(.001, .001, .00005)
    q = np.zeros(m.N)
    q[ref['boundary_node']] = ref['fixed_total_source_per_ms']
    m.enable_support_transport(params, source=q)
    f, t = np.zeros(m.N), np.zeros(m.E)
    fresh = SupportTransport(m.N, m.pre, m.post, f, t, q, params)
    distance = np.array(ref['anatomical_distances'])
    words = ref['words']
    import hashlib
    drives = {}
    for word in set(words):
        seed = int.from_bytes(hashlib.blake2b(word.lower().encode(), digest_size=8).digest(), 'little')
        value = np.zeros(m.N)
        value[np.random.default_rng(seed).choice(m.N, 6, replace=False)] = .18
        drives[word] = value
    steps = round(ref['elapsed_ms']/m.dt)
    windows = []
    error = material_error = label_error = excess = 0.
    started = time.perf_counter()
    for k in range(steps):
        m.step(I_ext=drives[words[min(23, int(k*m.dt/200.))]])
        fresh.advance(f, t, m.x_pre, m.g, m.trophic_edges, m.dt, m.trophic)
        if k % 100 == 0 or k == steps-1:
            error = max(error, float(np.abs(m.support_transport.balance(m.Nf, m.T)).max()))
            material_error = max(material_error, float(np.abs(m.material.balance(m.g)).max()))
            label_error = max(label_error, float(np.abs(fresh.balance(f, t)).max()))
            excess = max(excess, float(np.max(f-m.Nf)), float(np.max(t-m.T)),
                         float(np.max(fresh.cargo-m.support_transport.cargo)))
        if (k+1) % (steps//3) == 0:
            windows.append(snapshot(m, fresh, f, t, distance))
    path = args.out/'activity_complete.npz'
    save_medium(path, m, dict(scope='Existing activity capture candidate; delivery, not usefulness',
                             original_provenance=provenance, reference=str(args.reference)))
    restored, _ = load_medium(path)
    same(m.state_dict(), restored.state_dict())
    np.savez_compressed(args.out/'activity_source_label.npz', free=f, bound=t,
                        pre=m.pre, post=m.post,
                        metadata=np.array(json.dumps(dict(params=asdict(params), n=m.N,
                                                         schema=1, elapsed_ms=ref['elapsed_ms']))),
                        **fresh.state_dict()['arrays'])
    assert error < 1e-8 and material_error < 1e-8 and label_error < 1e-8 and excess < 1e-9
    assert windows[-1]['source_imported'] == ref['arms']['continuous']['final']['source_imported']
    assert hashes == {str(p.relative_to(ROOT)):digest(p) for p in paths}
    assert digest(ref['source_archive']) == ref['source_sha256']
    report = dict(scope='Post-registered existing-carrier comparison; no capability claim',
                  reference_report=str(args.reference), reference_sha256=digest(args.reference),
                  source_sha256=ref['source_sha256'], boundary=ref['boundary_node'],
                  code_sha256=hashes, windows=windows,
                  tag_final=ref['arms']['continuous']['final'], activity_final=windows[-1],
                  seconds=time.perf_counter()-started, full_restore_bitwise=True,
                  support_balance_max=error, material_balance_max=material_error,
                  label_balance_max=label_error, label_excess_max=excess,
                  archive=str(path), archive_sha256=digest(path),
                  usefulness_selection_proven=False, source_unchanged=True, passed=True)
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
