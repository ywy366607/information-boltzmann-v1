"""Decompose recorded native weight change where full final S was anchored.

Shapley order symmetry is possible only with both endpoint matrices. With a
final-S-only anchor report the exact final-S counterfactual, not a Shapley
allocation or a causal intervention. Rebuild in float64 and disclose residual.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def weights(s, theta):
    h=s.shape[1]
    i,j=np.triu_indices(h,1)
    s=np.asarray(s,dtype=np.float64); theta=np.asarray(theta,dtype=np.float64)
    norm=np.square(s).sum((-2,-1))
    corr=(s[:,i]*s[:,j]).sum((-2,-1))/np.sqrt((1+norm[:,i])*(1+norm[:,j]))
    logit=theta[:,h+2:h+3]*corr+theta[:,h+3:h+4]*(np.log1p(norm[:,i])+np.log1p(norm[:,j]))
    return 2*np.exp(-np.logaddexp(0,-logit))


def main():
    p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    cases=[]
    for seg,d in enumerate(sorted(a.work.glob('collect_observations*'))):
        endpoints={}
        for path in sorted(d.glob('physical_*.npz')):
            with np.load(path,allow_pickle=False) as z:
                for k,kind in enumerate(z['kind']):
                    if kind in ('event_before','event_after'):
                        endpoints.setdefault(int(z['event'][k]),{})[str(kind)]=(z['edge_weight'][k].copy(),z['theta'][k].copy())
        for path in sorted(d.glob('anchor_*.npz')):
            with np.load(path,allow_pickle=False) as z:
                event=int(z['event']); pair=endpoints.get(event,{})
                if set(pair)!={'event_before','event_after'}:continue
                before,t0=pair['event_before']; after,t1=pair['event_after']
                if np.array_equal(t0[:,10:12],t1[:,10:12]):continue
                s=z['S']; reconstructed=weights(s,t1); baseline=weights(s,t0)
                raw=after.astype(float)-before.astype(float)
                parameter=reconstructed-baseline
                residual=reconstructed-after
                cases.append(dict(segment=seg,event=event,anchor=str(path),
                    observed_weight_change_l2=float(np.linalg.norm(raw)),
                    fixed_final_s_theta_effect_l2=float(np.linalg.norm(parameter)),
                    fixed_old_theta_state_remainder_l2=float(np.linalg.norm(baseline-before)),
                    rebuild_error_l2=float(np.linalg.norm(residual)),
                    rebuild_max_abs_error=float(abs(residual).max()),
                    parameter_max_abs=float(abs(parameter).max()),
                    layers_theta_changed=int(np.any(t1[:,10:12]!=t0[:,10:12],axis=-1).sum())))
    out=dict(cases=cases,count=len(cases),note='one-endpoint algebraic decomposition, not counterfactual world rollout; anchor captures final state only')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf8');print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__':main()
