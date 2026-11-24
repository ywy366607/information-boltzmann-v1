"""Native within-quiet S/P phase increments, excluding preview/world gaps."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    pairs=[]
    for segment,d in enumerate(sorted(a.work.glob('collect_observations*'))):
        pending=None
        for f in sorted(d.glob('physical_*.npz')):
            with np.load(f,allow_pickle=False) as z:
                for i,kind in enumerate(z['kind']):
                    if kind=='quiet_before':
                        pending={'event':int(z['event'][i]),'complex':z['complex_overlap_real'][i]+1j*z['complex_overlap_imag'][i]}
                    elif kind=='quiet_after':
                        if pending is None or pending['event']!=int(z['event'][i]):raise ValueError('unmatched committed quiet boundary')
                        pairs.append((pending['complex'],z['complex_overlap_real'][i]+1j*z['complex_overlap_imag'][i],
                                      z['phase_delta'][i],z['temporal_overlap'][i],z['omega'][i],float(z['dt'][i])))
                        pending=None
        if pending is not None:raise ValueError('unmatched segment end')
    if not pairs:raise ValueError('no committed quiet pairs')
    old,new,phase,temporal,omega,dt=map(np.asarray,zip(*pairs))
    expected=-omega*dt[:,None,None]
    err=np.angle(np.exp(1j*(phase-expected)))
    valid=np.isfinite(phase)&(temporal>=.9)
    ii,jj=np.triu_indices(phase.shape[-1],1)
    aold=old[:,:,ii,jj];anew=new[:,:,ii,jj]
    relative=np.angle(anew*np.conj(aold))
    free=-(omega[:,:,ii]-omega[:,:,jj])*dt[:,None,None]
    residual=np.angle(np.exp(1j*(relative-free)))
    both=(abs(aold)>=.1)&(abs(anew)>=.1)&valid[:,:,ii]&valid[:,:,jj]
    rv=relative[both];fv=free[both];ev=residual[both]
    out={'committed_quiet_pairs':len(pairs),'actual_quiet_time':float(dt.sum()),
         'head_phase_valid_fraction':float(valid.mean()),
         'head_abs_phase_error_vs_free_median':float(np.median(abs(err[valid]))) if valid.any() else None,
         'eligible_pair_samples_overlap_0_1':int(both.sum()),
         'relative_phase_change_abs_median':float(np.median(abs(rv))) if len(rv) else None,
         'uncoupled_relative_phase_change_abs_median':float(np.median(abs(fv))) if len(fv) else None,
         'relative_phase_residual_abs_median':float(np.median(abs(ev))) if len(ev) else None,
         'relative_phase_residual_abs_p95':float(np.quantile(abs(ev),.95)) if len(ev) else None,
         'relative_phase_residual_abs_max':float(np.max(abs(ev))) if len(ev) else None,
         'uncoupled_relative_phase_change_abs_p95':float(np.quantile(abs(fv),.95)) if len(fv) else None,
         'note':'A_i inner products define phase only where overlap is adequate. A single-quiet-step phase difference is not synchronization; full trajectory may include intervening world and learning events.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__':main()
