"""First-passage projection of existing rates against observed R7 runs."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from information_boltzmann.cgb.mobile_exchange import MobileExchange,MaterialEvent
from scripts.ib.generative.native_learning_stream import atomic_json
from scripts.ib.generative.r7_lifetime_constraints import observed_intervals


def retention_sector(params,support):
    if params.polarized_storage:
        raise ValueError('A polarized carrier must retain charge in its first-passage state')
    # Conditional rate assay only: no evolved organism or capability claim.
    p=MobileExchange([[1.,1.,1.],[1.+params.rest_length,1.,1.]],
        material=0,free=[0,support],charged=0,depleted=0,
        pre=[0],post=[1],trace=1,bound=0,params=params)
    b=p.event_rate(MaterialEvent('binding',0,1,0,1))
    d0=p.event_rate(MaterialEvent('thermal',0,1,0,-1))
    p.bound[0]=1;p.free[1]-=1
    u=p.event_rate(MaterialEvent('binding',0,1,0,-1))
    d1=p.event_rate(MaterialEvent('thermal',0,1,0,-1))
    transient=np.array([[-b-d0,b],[u,-u-d1]])
    return transient,dict(binding=b,unbinding=u,disappearance_unbound=d0,disappearance_bound=d1)


def interval_mass(transient,clock,unbound):
    step=expm(transient*clock)
    row=np.array([unbound,1-unbound]);exit_probability=1-step.sum(axis=1)
    survival=[];event=[]
    for _ in range(60):
        survival.append(float(row.sum()));event.append(float(row@exit_probability))
        row=row@step
    return np.asarray(survival),np.asarray(event)


def objective(theta,transient,duration,ended):
    survival,event=interval_mass(transient,np.exp(theta[0]),theta[1])
    probs=np.where(ended,event[duration],survival[duration])
    if np.any(probs<=0) or not np.isfinite(probs).all():return float('inf')
    return -float(np.log(probs).mean())


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    parent_hash=hashlib.sha256(args.archive.read_bytes()).hexdigest()
    parent,_=load_exchange(args.archive)
    before=parent.state_dict()
    observations=json.loads((args.audit/'observations.json').read_text())
    control=sorted([w for w in observations['workbooks']
        if w['kind']=='filopodium' and w['genotype']=='Control'],key=lambda w:w['file'])
    data={key:(d.astype(int),ended.astype(bool)) for key,group in [('train',control[:2]),('test',control[2:])]
          for d,ended in [observed_intervals(group)]}
    rows=[]
    for support in [1,2,4,8,16]:
        transient,rates=retention_sector(parent.params,support)
        fits=[]
        for clock in [1.,10.,100.]:
            for unbound in [0.,.5,1.]:
                result=minimize(objective,[np.log(clock),unbound],
                    args=(transient,*data['train']),method='L-BFGS-B',
                    bounds=[(np.log(.001),np.log(1000.)),(0.,1.)],
                    options=dict(maxiter=1000,ftol=1e-13,gtol=1e-8,maxls=40))
                fits.append(dict(parameters=result.x.tolist(),train_nll=float(result.fun),
                    success=bool(result.success),iterations=int(result.nit),message=str(result.message)))
        best=min(fits,key=lambda f:f['train_nll']);theta=np.asarray(best['parameters'])
        row=dict(total_support=support,actual_native_rates_per_reference_ms=rates,
            transient_generator=transient.tolist(),clock_reference_ms_per_minute=float(np.exp(theta[0])),
            initially_unbound_probability=float(theta[1]),optimizer_success=best['success'],
            train_nll=best['train_nll'],multistart=fits,
            entry_distribution_on_boundary=bool(theta[1]<1e-5 or theta[1]>1-1e-5))
        rows.append(row)
    selected=min(rows,key=lambda r:r['train_nll'])
    atomic_json(args.out/'training_locked.json',dict(rows=rows,selected_support=selected['total_support'],
        selection='Training objective only; no law or native parameter fitted',parent_sha256=parent_hash))
    for row in rows:
        transient=np.asarray(row['transient_generator'])
        theta=np.array([np.log(row['clock_reference_ms_per_minute']),row['initially_unbound_probability']])
        row['test_nll']=objective(theta,transient,*data['test'])
        row['test_tracks']=len(data['test'][0])
    from scripts.ib.generative.compare_support_transport import same
    same(before,parent.state_dict())
    assert parent_hash==hashlib.sha256(args.archive.read_bytes()).hexdigest()
    report=dict(rows=rows,selected_support=selected['total_support'],
        native_parameter_source=str(args.archive),parent_sha256=parent_hash,parent_unchanged=True,
        native_source_sha256=hashlib.sha256(Path(inspect.getfile(MobileExchange)).read_bytes()).hexdigest(),
        selected_native_parameters={k:getattr(parent.params,k) for k in
            ['trace_energy','capture_coupling','bound_energy','binding_rate','recovery_rate']},
        evidence_level='Exploratory conditional native-generator observation bridge on real R7 data; not a full biological validation')
    atomic_json(args.out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    print(json.dumps([{k:v for k,v in row.items() if k not in ['multistart','transient_generator','actual_native_rates_per_reference_ms']} for row in rows],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
