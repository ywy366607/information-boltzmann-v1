"""Measure loop phases invariant to a free rotation of each native S/P head."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

TRIANGLES = np.asarray(list(itertools.combinations(range(8), 3)))


def cycles(c):
    i, j, k = TRIANGLES.T
    a, b, d = c[..., i, j], c[..., j, k], c[..., i, k]
    return a*b*d.conj(), np.minimum(np.minimum(abs(a), abs(b)), abs(d))


def stats(values):
    a = np.asarray(values)
    if not len(a):
        return dict(count=0)
    return dict(count=len(a), min=float(a.min()), median=float(np.median(a)),
                mean=float(a.mean()), max=float(a.max()))


def audit(work):
    changes = {k: [] for k in ('quiet', 'update', 'external')}
    strength = {k: [] for k in changes}
    pairs = dict.fromkeys(changes, 0)
    phase, magnitude, rotation_error, duration = [], [], 0., 0.
    for path in sorted(work.glob('physical_*.npz')):
        with np.load(path, allow_pickle=False) as z:
            c = z['complex_overlap_real'].astype(float)+1j*z['complex_overlap_imag'].astype(float)
            loop, mag = cycles(c)
            for left, right in itertools.pairwise(range(len(c))):
                if z['event'][left] != z['event'][right]:
                    continue
                kinds = tuple(z['kind'][[left,right]])
                if kinds == ('quiet_before', 'quiet_after'):
                    kind = 'quiet'
                    dt = float(z['quiet_time'][right]-z['quiet_time'][left])
                    duration += dt
                    r = np.exp(-1j*z['omega'][left].astype(float)*dt)
                    free, _ = cycles(c[left]*r[..., :, None]*r[..., None, :].conj())
                    rotation_error = max(rotation_error, float(abs(free-loop[left]).max()))
                    phase.extend(abs(np.angle(loop[left])).ravel())
                    magnitude.extend(mag[left].ravel())
                elif kinds == ('update_before', 'update_after'):
                    kind = 'update'
                elif kinds == ('quiet_after','event_after'):
                    kind = 'external'
                else:
                    continue
                pairs[kind] += 1
                changes[kind].extend(abs(np.angle(loop[right]*loop[left].conj())).ravel())
                strength[kind].extend(np.minimum(mag[left],mag[right]).ravel())
    m, p = np.asarray(magnitude), np.asarray(phase)
    return dict(pairs=pairs, actual_quiet_time=duration, free_rotation_loop_error=rotation_error,
        initial_loop_angle={str(t):stats(p[m>=t]) for t in (0., .03, .1)},
        observed_loop_phase_change={k:{str(t):stats(np.asarray(v)[np.asarray(strength[k])>=t])
            for t in (0., .03, .1)} for k,v in changes.items()},
        units='radians; counts are layer/loop samples, not independent experiments')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('work', type=Path)
    p.add_argument('report', type=Path)
    args = p.parse_args()
    result = audit(args.work)
    with args.report.open('x', encoding='utf8') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps(result))
