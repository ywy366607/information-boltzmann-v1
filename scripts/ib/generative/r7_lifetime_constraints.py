"""Real-data, grouped/censored R7 constraints; no physical model update."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.ib.generative.native_learning_stream import atomic_json

RATE_BOUNDS=(np.log(1e-6), np.log(20.))


def observed_intervals(workbooks):
    records=[r for w in workbooks for r in w['records'] if r['first_frame']>0]
    duration=np.array([r['last_frame']-r['first_frame'] for r in records],float)
    ended=np.array([r['last_frame']<59 for r in records],float)
    return duration,ended


def mixture_objective(theta,duration,ended):
    rates=np.exp(theta[:2]);p=expit(theta[2])
    log_weights=np.array([-np.logaddexp(0,-theta[2]),-np.logaddexp(0,theta[2])])
    component=log_weights[:,None]-rates[:,None]*duration
    component+=ended[None,:]*np.log(-np.expm1(-rates[:,None]))
    log_likelihood=logsumexp(component,axis=0)
    posterior=np.exp(component-log_likelihood)
    grad_rates=-duration[None,:]+ended[None,:]/np.expm1(rates[:,None])
    gradient=np.r_[-np.mean(posterior*grad_rates*rates[:,None],axis=1),
                    -np.mean(posterior[0]-p)]
    return -float(log_likelihood.mean()),gradient


def score(model,duration,ended):
    rates=np.asarray(model['rates_per_minute'])
    weights=np.asarray(model['weights_at_first_observation'])
    terms=np.log(weights)[:,None]-rates[:,None]*duration
    terms+=ended[None,:]*np.log(-np.expm1(-rates[:,None]))
    values=-logsumexp(terms,axis=0)
    return dict(incident_tracks=len(duration),observed_endings=int(ended.sum()),
        right_censored=int(len(ended)-ended.sum()),
        total_nll=float(values.sum()),mean_nll=float(values.mean()),
        per_track_nll=values.tolist())


def fit(train):
    duration,ended=observed_intervals(train)
    assert len(duration)>0 and duration.sum()>0 and ended.sum()>0
    exact=float(np.log1p(ended.sum()/duration.sum()))
    single=dict(rates_per_minute=[exact],weights_at_first_observation=[1.],
        analytic_discrete_mle=True,identified_within_family=True)
    fits=[]
    for slow,fast in ((.001,.1),(.01,1.),(.1,2.)):
        for p in (.1,.5,.9):
            initial=np.array([np.log(slow),np.log(fast),np.log(p/(1-p))])
            trace=[]
            def objective(theta):
                value,gradient=mixture_objective(theta,duration,ended)
                trace.append(value)
                return value,gradient
            result=minimize(objective,initial,jac=True,method='L-BFGS-B',
                bounds=[RATE_BOUNDS,RATE_BOUNDS,(-8,8)],
                options=dict(maxiter=1500,ftol=1e-13,gtol=1e-8,maxls=40))
            fits.append(dict(initial=initial.tolist(),parameters=result.x.tolist(),
                objective=float(result.fun),gradient_inf=float(np.max(np.abs(result.jac))),
                success=bool(result.success),message=str(result.message),
                iterations=int(result.nit),function_evaluations=int(result.nfev),curve=trace))
    chosen=min(fits,key=lambda x:x['objective'])
    theta=np.asarray(chosen['parameters'])
    extra=minimize(lambda x:mixture_objective(x,duration,ended),theta,jac=True,method='L-BFGS-B',
        bounds=[RATE_BOUNDS,RATE_BOUNDS,(-8,8)],
        options=dict(maxiter=1500,ftol=1e-14,gtol=1e-9,maxls=50))
    if extra.fun<chosen['objective']:
        theta=extra.x
    rates=np.exp(theta[:2]);weights=np.array([expit(theta[2]),expit(-theta[2])])
    order=np.argsort(rates)
    bounds=np.array([RATE_BOUNDS,RATE_BOUNDS,(-8,8)])
    hits=[i for i in range(3) if min(abs(theta[i]-bounds[i,0]),abs(theta[i]-bounds[i,1]))<1e-4]
    mixture=dict(rates_per_minute=rates[order].tolist(),
        weights_at_first_observation=weights[order].tolist(),
        optimizer_parameters=theta.tolist(),parameter_boundary_hits=hits,
        optimization_success=bool(extra.success),gradient_inf=float(np.max(np.abs(extra.jac))),
        extra_objective_improvement=float(chosen['objective']-extra.fun),
        multistart_objective_range=[min(f['objective'] for f in fits),max(f['objective'] for f in fits)],
        microscopic_parameters_identified=False)
    return dict(single=single,mixture=mixture),fits


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    source=json.loads((args.audit/'observations.json').read_text())
    candidates=[w for w in source['workbooks'] if w['kind']=='filopodium' and w['genotype']=='Control']
    candidates=sorted(candidates,key=lambda w:w['file'])
    assert [Path(w['file']).name for w in candidates]==[
        'Control_P60_GC1.xlsx','Control_P60_GC2.xlsx','Control_P60_GC3.xlsx']
    assert all(w['fit_admitted'] for w in candidates)
    train,test=candidates[:2],candidates[2:]
    models,fits=fit(train)
    manifest=dict(source_commit=source['source']['commit'],
        observations_sha256=hashlib.sha256((args.audit/'observations.json').read_bytes()).hexdigest(),
        train=[dict(file=w['file'],sha256=w['sha256']) for w in train],
        test=[dict(file=w['file'],sha256=w['sha256']) for w in test],
        sampling_interval_minutes=1.,parameters=model_dump(models),
        model_selection='None between families; all fits use CG1/CG2 only',
        prereg_sha256=hashlib.sha256((ROOT/'docs/information_boltzmann/CGB_R7_DYNAMIC_AMENDMENT_20261022.md').read_bytes()).hexdigest())
    atomic_json(args.out/'parameters_locked.json',manifest)
    atomic_json(args.out/'optimization.json',fits)
    assessed={}
    arrival=float(sum(w['incident_records'] for w in train)/(59*len(train)))
    for family,model in models.items():
        splits={key:score(model,*observed_intervals(group)) for key,group in [('train',train),('test',test)]}
        rates=np.asarray(model['rates_per_minute']);weights=np.asarray(model['weights_at_first_observation'])
        assessed[family]=dict(parameters=model,scores=splits,
            observed_time_survival={str(t):float(np.sum(weights*np.exp(-rates*t))) for t in [1,8,40,59,120]},
            stationary_visible_lifetime_frames=float(np.sum(weights/(-np.expm1(-rates)))),
            conditional_stationary_mean_count=float(arrival*np.sum(weights/(-np.expm1(-rates)))))
    data=[{k:w[k] for k in ['file','record_count','initial_preexisting','incident_records',
          'incident_observed_endings','incident_right_censored','counts','mean_count','fano','zero_frame_fraction']}
          for w in candidates]
    report=dict(source=manifest,models=assessed,observed_control_cones=data,
        train_observed_arrivals_per_minute=arrival,
        observed_train_mean_count=float(np.mean([w['mean_count'] for w in train])),
        observed_test_mean_count=float(test[0]['mean_count']),
        evidence_level='Grouped real-data observation constraints; neither microscopic law identification nor biological validation of the full CGB kernel',
        censored_scope='First-observed remaining runs; initially present tracks excluded from lifetime fitting but retained in counts',
        model_scope='Mixture is a declared slow-exchange observational limit; no new physical state or update rule')
    atomic_json(args.out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['source','observed_control_cones']},indent=2))


def model_dump(models):
    return models


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
