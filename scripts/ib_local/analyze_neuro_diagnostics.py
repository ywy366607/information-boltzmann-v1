"""Analyze the fixed real-text battery without additional body execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.ib_local.closed_loop_information import (
    partitions,gaussian_transition,geometric_cut,gaussian_nll,mean_interval)
from scripts.ib_local.neuro_diagnostic_metrics import (
    factorial_interactions,paired_effect,topology_report,avalanche_report,
    effective_dimension,phi_star_gaussian)
from scripts.ib_local.neuro_diagnostic_metrics import gaussian_mmi_pid


def byte_conditioned(hidden,ids,ntrain):
    flat=hidden.reshape(*hidden.shape[:2],-1).astype(np.float64)
    means=np.zeros((256,flat.shape[-1]));counts=np.zeros(256)
    np.add.at(means,ids[:ntrain].reshape(-1),flat[:ntrain].reshape(-1,flat.shape[-1]))
    np.add.at(counts,ids[:ntrain].reshape(-1),1)
    means/=np.maximum(counts[:,None],1)
    return (flat-means[ids]).reshape(hidden.shape)


def project_layers(hidden,ntrain):
    """Two PCs per fixed eight-channel layer view; fit only earlier segments."""
    z=[];inactive=[]
    for layer in range(hidden.shape[2]):
        train=hidden[:ntrain,:,layer,:].reshape(-1,hidden.shape[-1])
        center=train.mean(0)
        _,vec=np.linalg.eigh(np.cov(train,rowvar=False))
        basis=vec[:,-2:]
        scale=((train-center)@basis).std(0)
        inactive.extend((np.where(scale<1e-6)[0]+2*layer).tolist())
        z.append((hidden[:,:,layer,:]-center)@basis/np.maximum(scale,1e-6))
    return np.concatenate(z,-1),inactive


def fit_information(z,ntrain,heldout=True,known=None):
    # No artificial transition across disjoint text segments.
    x=z[:ntrain,:-1].reshape(-1,z.shape[-1]);y=z[:ntrain,1:].reshape(-1,z.shape[-1])
    mx,my=x.mean(0),y.mean(0)
    A,sx,noise,mi=gaussian_transition(x-mx,y-my)
    names=list(range(4));rows=[];cuts=[]
    for left,right in partitions(names):
        blocks=[[2*i+j for i in side for j in (0,1)] for side in (left,right)]
        g,cut=geometric_cut(A,sx,noise,blocks)
        star=phi_star_gaussian(A,sx,noise,blocks)
        rows.append(dict(left=left,right=right,**g,**star));cuts.append(cut)
    best=int(np.argmin([r['phi_g_nats'] for r in rows]))
    result=dict(mi_nats=float(mi),minimum_phi_g=rows[best],
                minimum_phi_star=min(rows,key=lambda r:r['phi_star_nats']),partitions=rows,
                all_gaussian_bounds_ok=all(r['bounds_ok'] for r in rows),
                all_optimizers_converged=all(r['converged'] for r in rows))
    if heldout:
        tx=z[ntrain:,:-1].reshape(-1,z.shape[-1]);ty=z[ntrain:,1:].reshape(-1,z.shape[-1])
        full=gaussian_nll(ty,(tx-mx)@A.T+my,noise)
        cut=gaussian_nll(ty,(tx-mx)@cuts[best].T+my,noise)
        # Sampling units remain independent declared text segments.
        per_pair=(cut-full).reshape(len(z)-ntrain,-1)
        valid=np.ones_like(per_pair,dtype=bool) if known is None else known[ntrain:,:-1]&known[ntrain:,1:]
        per_segment=np.array([row[mask].mean() for row,mask in zip(per_pair,valid)])
        result['heldout_cut_penalty']=paired_effect(per_segment)
        result['heldout_valid_transition_pairs']=int(valid.sum())
        sy=noise+A@sx@A.T;cyx=A@sx
        joint=np.block([[sx,cyx.T],[cyx,sy]])
        result['gaussian_mmi_pid']=gaussian_mmi_pid(joint,list(range(4)),list(range(4,8)),list(range(8,16)))
        predictive=[]
        for source in range(4):
            keep=[i for i in range(8) if i//2!=source]
            for target in range(4):
                if source==target:continue
                out=[2*target,2*target+1]
                B=np.linalg.solve(sx[np.ix_(keep,keep)],cyx[np.ix_(out,keep)].T).T
                reduced=sy[np.ix_(out,out)]-B@sx[np.ix_(keep,keep)]@B.T
                log_ratio=.5*(np.linalg.slogdet(reduced)[1]-np.linalg.slogdet(noise[np.ix_(out,out)])[1])
                predictive.append(dict(source=source,target=target,conditional_predictive_information_nats=float(log_ratio)))
        ff=sum(a['conditional_predictive_information_nats'] for a in predictive if a['source']<2<=a['target'])
        fb=sum(a['conditional_predictive_information_nats'] for a in predictive if a['target']<2<=a['source'])
        result['conditional_predictive_links']=predictive
        result['predictive_feedback_feedforward_ratio']=float(fb/ff) if ff>1e-12 else None
        result['predictive_direction_note']='Prediction from incomplete observed hidden projections; actual causal direction is checked separately by state intervention.'
    return result


def information_report(hidden,ntrain,surrogates=39,known=None):
    z,inactive=project_layers(hidden,ntrain)
    if len(inactive)==z.shape[-1]:
        return dict(status='all projected input-conditioned channels below numerical activity threshold',
                    inactive_coordinates=inactive)
    observed=fit_information(z,ntrain,known=known)
    rng=np.random.default_rng(77);null_g=[];null_star=[]
    for _ in range(surrogates):
        shuffled=z.copy()
        for segment in range(len(z)):
            for layer in range(4):
                shift=int(rng.integers(128,z.shape[1]-128))
                shuffled[segment,:,2*layer:2*layer+2]=np.roll(z[segment,:,2*layer:2*layer+2],shift,axis=0)
        measured=fit_information(shuffled,ntrain,False)
        null_g.append(measured['minimum_phi_g']['phi_g_nats'])
        null_star.append(measured['minimum_phi_star']['phi_star_nats'])
    observed.update(inactive_coordinates=inactive,null_count=surrogates,
        phi_g_null_median=float(np.median(null_g)),phi_star_null_median=float(np.median(null_star)),
        phi_g_null_upper_tail=float((1+sum(v>=observed['minimum_phi_g']['phi_g_nats'] for v in null_g))/(surrogates+1)),
        phi_star_null_upper_tail=float((1+sum(v>=observed['minimum_phi_star']['phi_star_nats'] for v in null_star))/(surrogates+1)),
        scope='8-dimensional frozen driven-body projection; residualized on current byte; not whole physical IIT4')
    return observed


def calibration(data,name):
    p=data[name+'_confidence'].flatten();y=data[name+'_correctness'].flatten()
    bins=[]
    for ids in np.array_split(np.argsort(p),10):
        bins.append(dict(count=len(ids),confidence=float(p[ids].mean()),accuracy=float(y[ids].mean())))
    return dict(top1_accuracy=float(y.mean()),mean_confidence=float(p.mean()),
                ece=float(sum(b['count']*abs(b['confidence']-b['accuracy']) for b in bins)/len(p)),
                brier=float(data[name+'_brier'].mean()),bins=bins)


def dynamics_summary(path):
    r=json.loads(path.read_text(encoding='utf-8'))
    global_values=np.array([a['mean_log_gain_per_byte'] for a in r['global_conditional_lyapunov']]).reshape(-1,2).mean(1)
    local=np.array([a['mean_log_gain_per_byte'] for a in r['local_head_conditional_lyapunov']]).reshape(-1,4,4)
    heads={f'L{i}H{j}':dict(mean=float(local[:,i,j].mean()),
                          observed_range=[float(local[:,i,j].min()),float(local[:,i,j].max())],
                          e_folding_bytes=float(-1/local[:,i,j].mean()) if local[:,i,j].mean()<0 else None)
           for i in range(4) for j in range(4)}
    upper=max(max(a['hidden_response_gain'][:a['source_layer']]) for a in r['directed_state_interventions'] if a['source_layer']>0)
    lower=max(max(a['hidden_response_gain'][a['source_layer']+1:],default=0) for a in r['directed_state_interventions'])
    curves=[]
    for context in range(4):
        rows=[a for a in r['amplitude_response'] if a['context']==context]
        x=np.log([a['relative_amplitude'] for a in rows])
        y=np.array([a['late_response'][-1] for a in rows])
        if np.max(y)<1e-6:
            curves.append(dict(context=context,status='late response at numerical floor'))
        else:
            slope,intercept=np.polyfit(x,np.log(y),1)
            curves.append(dict(context=context,loglog_amplitude_slope=float(slope),
                linear_fit_r2=float(1-np.sum((np.log(y)-(slope*x+intercept))**2)/np.sum((np.log(y)-np.log(y).mean())**2))))
    return dict(global_finite_time_log_gain_per_byte=paired_effect(global_values),
                global_context_values=global_values.tolist(),
                e_folding_bytes=float(-1/global_values.mean()) if global_values.mean()<0 else None,
                local_heads=heads,all_sampled_local_head_rates_negative=bool(np.all(local<0)),
                direct_upper_to_lower_gain_max=upper,direct_lower_to_upper_gain_max=lower,
                late_amplitude_curves=curves,
                broadcast_example=r['amplitude_response'][2].get('existing_readout_changes'),
                scope=r['scope'])


def prior_wood_calibration(path):
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    used=[];base=[];g_used=[];g_base=[]
    for row in rows:
        option=('silence','generate','reuse').index(row['choice'])
        actual=float(row['world']['match']>0)
        used.append((row['match_probability']-actual)**2)
        base.append((row['wood_all_bases'][option][2]-actual)**2)
        g_used.append((row['wood_all_predictions'][option][3]/10-row['metal_g'])**2)
        g_base.append((row['wood_all_bases'][option][3]/10-row['metal_g'])**2)
    def grouped(values):
        values=np.asarray(values);n=len(values)//8
        return values[:8*n].reshape(n,8).mean(1)
    return dict(source=str(path),rounds=len(rows),match_brier=float(np.mean(used)),action_base_brier=float(np.mean(base)),
                brier_improvement=paired_effect(grouped(np.array(base)-np.array(used))),
                gain_prediction_mse=float(np.mean(g_used)),gain_base_mse=float(np.mean(g_base)),
                gain_mse_improvement=paired_effect(grouped(np.array(g_base)-np.array(g_used))),
                note='Reused actual prequential forecasts; no new learning and no hindsight action means.')


def paired_information_increment(data,ntrain,draws=99):
    """Paired segment bootstrap, fixed predeclared fitted projection charts.

    Supplementary effect-size check, conditional on the estimated two-PC charts;
    not a new primary hypothesis and not an IIT4 causal identification claim.
    """
    states=[]
    for name in ('M1T1R1','M0T1R1'):
        adjusted=byte_conditioned(data[name+'_hidden'],data['input_bytes'],ntrain)
        z,_=project_layers(adjusted,ntrain);states.append(z)
    rng=np.random.default_rng(512);deltas=[]
    for _ in range(draws):
        pick=rng.integers(0,ntrain,ntrain)
        estimates=[fit_information(z[pick],ntrain,False) for z in states]
        deltas.append([estimates[0]['minimum_phi_g']['phi_g_nats']-estimates[1]['minimum_phi_g']['phi_g_nats'],
                       estimates[0]['minimum_phi_star']['phi_star_nats']-estimates[1]['minimum_phi_star']['phi_star_nats']])
    values=np.asarray(deltas)
    return dict(draws=draws,paired_train_segments=ntrain,
                phi_g_increment_ci95=np.quantile(values[:,0],[.025,.975]).tolist(),
                phi_star_increment_ci95=np.quantile(values[:,1],[.025,.975]).tolist(),
                scope='Memory-on minus same-weight GDN+table, conditional fixed two-PC charts, supplementary segment bootstrap')


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--quick',action='store_true');args=p.parse_args()
    data=np.load(args.input/'observations.npz')
    scores=json.loads((args.input/'scores.json').read_text())
    losses={k:np.array(v['segment_nll']) for k,v in scores.items()}
    result=dict(factorial=factorial_interactions(losses),
                shuffled_memory_loss_increase=paired_effect(losses['shuffled_memory']-losses['M1T1R1']),
                heldout_nll_means={k:v['mean_nll'] for k,v in scores.items()},
                source_integrity=json.loads((args.input/'completion.json').read_text()),
                macro_downward=json.loads((args.input/'macro_downward.json').read_text()))
    result['dynamics']={name:dynamics_summary(args.input/(name+'_dynamics.json'))
                        for name in ('cgb','same_weight_gdn','original_delta')}
    names=['M1T1R1','M1T1R0','M0T1R1','M0T0R1','original_delta_R1','original_delta_R0']
    ntrain=len(data['input_bytes'])//2
    byte_counts=np.bincount(data['input_bytes'][:ntrain].flatten(),minlength=256)
    known=byte_counts[data['input_bytes']]>0
    result['input_control_unseen_test_bytes']=int((~known[ntrain:]).sum())
    result['views']={}
    for name in names:
        hidden=data[name+'_hidden']
        adjusted=byte_conditioned(hidden,data['input_bytes'],ntrain)
        flat=hidden[ntrain:].reshape(-1,32); residual=adjusted[ntrain:].reshape(-1,32)[known[ntrain:].flatten()]
        entry=dict(calibration=calibration(data,name),effective_dimension=effective_dimension(flat),
                   topology_raw=topology_report(flat),topology_byte_conditioned=topology_report(residual))
        entry['avalanches']=[avalanche_report(hidden[ntrain:].reshape(len(hidden)-ntrain,hidden.shape[1],32),threshold,bin_size)
                             for threshold in (2.,2.5,3.) for bin_size in (1,2,4)]
        print('NETWORK/CASCADES',name,flush=True)
        if not args.quick:
            entry['information_byte_conditioned']=information_report(adjusted,ntrain,known=known)
            print('INFORMATION',name,flush=True)
        result['views'][name]=entry
        (args.input/'analysis_partial.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    result['existing_wood_calibration']=prior_wood_calibration(Path('E:/cgb_closed_loop_information_20261105/main/full.jsonl'))
    if not args.quick:
        result['memory_information_increment']=paired_information_increment(data,ntrain)
    result['existing_macro_phi']=json.loads(Path('results/published/cgb_closed_loop_information_20261105.json').read_text(encoding='utf-8'))['sampling_control']
    result['interpretation']='Fixed preregistered diagnostic battery. Text segments are sampling units. Conditional body stability and input-conditioned projections are not complete hybrid-system IIT4/criticality.'
    (args.input/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    for script in (Path(__file__),Path('scripts/ib_local/neuro_diagnostic_metrics.py')):
        h=hashlib.sha256(script.read_bytes()).hexdigest()
        shutil.copyfile(script,args.input/'executed_sources'/(script.stem+'_'+h[:12]+script.suffix))
    print('COMPLETE ANALYSIS',str(args.input/'analysis.json'),flush=True)


if __name__=='__main__':main()
