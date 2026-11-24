"""Native S/P overlap phases on committed quiet samples, no synthetic phase."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('work', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    overlap, omega, dt, segment = [], [], [], []
    for n, d in enumerate(sorted(a.work.glob('collect_observations*'))):
        for path in sorted(d.glob('physical_*.npz')):
            with np.load(path, allow_pickle=False) as z:
                for i in np.flatnonzero(z['kind'] == 'quiet_after'):
                    overlap.append(z['complex_overlap_real'][i] + 1j*z['complex_overlap_imag'][i])
                    omega.append(z['omega'][i]); dt.append(float(z['dt'][i])); segment.append(n)
    c = np.asarray(overlap); w = np.asarray(omega); dt = np.asarray(dt)
    ii,jj = np.triu_indices(c.shape[-1], 1)
    c = c[:,:,ii,jj]; mag = abs(c)
    same = np.asarray(segment[1:]) == np.asarray(segment[:-1])
    products = c[1:]*c[:-1].conj()
    phase_move = abs(np.angle(products))
    out = {'committed_samples': len(c), 'layers': c.shape[1], 'pairs_per_layer': c.shape[2],
           'overlap_magnitude_quantiles': np.quantile(mag,[0,.25,.5,.75,.9,.99,1]).tolist(),
           'intrinsic_cycles_min_max': [float((w*dt[:,None,None]).sum(0).min()/(2*np.pi)),
                                        float((w*dt[:,None,None]).sum(0).max()/(2*np.pi))],
           'same_segment_adjacent_pairs': int(same.sum()), 'thresholds': {},
           'note':'Adjacent committed samples may have external input and updates between them; phase concentration is not proof of dynamical locking or Dirac synchrony.'}
    for threshold in (.1,.25,.5):
        valid = mag >= threshold
        valid_adj = (valid[1:] & valid[:-1]) & same[:,None,None]
        angles = phase_move[valid_adj]
        unit = np.divide(c, mag, out=np.zeros_like(c), where=mag>0)
        n = valid.sum(0)
        plv = np.divide(abs(np.where(valid,unit,0).sum(0)),n,out=np.zeros_like(n,dtype=float),where=n>0)
        eligible = n>=10
        out['thresholds'][str(threshold)] = dict(sample_fraction=float(valid.mean()),
            pairs_with_at_least_10_samples=int(eligible.sum()),
            eligible_phase_concentration_median=float(np.median(plv[eligible])) if eligible.any() else None,
            adjacent_qualified=int(valid_adj.sum()),
            adjacent_abs_phase_change_median_radians=float(np.median(angles)) if len(angles) else None)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(out,ensure_ascii=False))


if __name__ == '__main__':
    main()
