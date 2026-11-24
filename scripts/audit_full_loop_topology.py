"""Read-only higher-order analysis of the executed five-phase language loop.

No model, Torch import, artificial training data, GPU or world calls. The
activity complex is statistical; it is not a claimed causal simplicial model.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy import signal


PHASES = ('water', 'wood', 'fire', 'earth', 'metal')
NAMES = (
    'water.workspace_rms', 'water.context_norm', 'water.log_max_recall_age', 'water.response_credit',
    'wood.future_mean', 'wood.future_range', 'wood.information_mean', 'wood.momentum_prediction_delta',
    'fire.choice_entropy', 'fire.generate_silence_log_odds', 'fire.actual_self_bytes', 'fire.internal_steps',
    'earth.response_nll', 'earth.short_loss', 'earth.actual_boundary_transition_loss', 'earth.all_32_layer_P_norm',
    'metal.real_g', 'metal.critic_rms', 'metal.actor_norm', 'metal.delayed_advantage',
)


def extract(row):
    future = np.asarray(row['causal_program']['expected_after_value'], float)
    prob = np.asarray(row['choice_probability'], float).clip(1e-20)
    return np.asarray([
        row['past_present']['workspace_rms'], row['question_context']['context_norm'],
        np.log1p(row['episodic_max_bank_age']), row['read_credit']['response']['credit_sum'],
        future.mean(), np.ptp(future), np.mean(row['causal_program']['conditional_information']),
        row['momentum_wood']['prediction_delta'], -(prob*np.log(prob)).sum(),
        np.log(prob[1]/prob[0]), row['self_bytes'],
        row['generative_relation']['autonomous']['internal_steps'], row['response_nll'],
        row['short_loss'], row['internal_transition_loss'], np.linalg.norm(row['bfss_dynamics']['momentum_norms']),
        row['metal_g'], row['critic_rms'], row['actor_norm'], row['td_previous']['advantage'],
    ], float)


def activity(values, block=64):
    """Locally detrended coactivity; apply identical construction to controls."""
    result = np.zeros_like(values, dtype=bool)
    for start in range(0, len(values), block):
        part = values[start:start+block]
        time = np.linspace(-1, 1, len(part))[:, None]
        residual = part-part.mean(0)-time*((time*part).sum(0)/(time*time).sum())
        result[start:start+len(part)] = residual > np.median(residual, axis=0)
    return result


def phase_shift(values, rng, block=64):
    """Preserve each phase's four channels and 64-event local marginals."""
    shifted = values.copy()
    for start in range(0, len(values), block):
        for phase in range(5):
            columns = slice(4*phase, 4*phase+4)
            shifted[start:start+block, columns] = np.roll(
                values[start:start+block, columns], int(rng.integers(8, block-7)), axis=0)
    return shifted


def cells(active, max_order=3):
    bitsets = [int.from_bytes(np.packbits(c, bitorder='little').tobytes(), 'little') for c in active.T]
    result = []
    for size in range(1, max_order+2):
        for simplex in combinations(range(active.shape[1]), size):
            bits = bitsets[simplex[0]]
            for vertex in simplex[1:]:
                bits &= bitsets[vertex]
            count = bits.bit_count()
            if count:
                result.append((count, simplex))
    return sorted(result, key=lambda item: (-item[0], len(item[1]), item[1]))


def persistence(active):
    """Exact GF(2) boundary reduction through tetrahedra: H0, H1 and H2."""
    ordered = cells(active)
    index = {s:i for i, (_, s) in enumerate(ordered)}
    pivots, birth, paired, bars = {}, set(), set(), [[], [], []]
    for col, (count, simplex) in enumerate(ordered):
        boundary = 0
        if len(simplex) > 1:
            for face in combinations(simplex, len(simplex)-1):
                boundary ^= 1 << index[face]
        while boundary:
            row = boundary.bit_length()-1
            if row not in pivots:
                pivots[row] = boundary
                paired.add(row)
                dimension = len(ordered[row][1])-1
                if dimension <= 2:
                    bars[dimension].append((ordered[row][0]/len(active), count/len(active), False))
                break
            boundary ^= pivots[row]
        if not boundary:
            birth.add(col)
    for row in birth-paired:
        dimension = len(ordered[row][1])-1
        if dimension <= 2:
            bars[dimension].append((ordered[row][0]/len(active), 0., True))
    return dict(cells_by_order=[sum(len(s)==k+1 for _,s in ordered) for k in range(4)],
        dimensions=[dict(dimension=d, positive_bars=sum(b-a>0 for b,a,_ in records),
            maximum_lifetime=max((b-a for b,a,_ in records), default=0.),
            total_lifetime=sum(b-a for b,a,_ in records),
            essential_at_zero_support=sum(e for _,_,e in records),
            bars=[list(item) for item in records if item[0]>item[1]]) for d,records in enumerate(bars)])


def operators(active, cutoff=.16):
    groups = [[s for count,s in cells(active) if len(s)==k+1 and count/len(active)>=cutoff]
              for k in range(4)]
    boundaries = []
    for order in range(1, 4):
        lookup = {s:i for i,s in enumerate(groups[order-1])}
        ri, ci, val = [], [], []
        for col, simplex in enumerate(groups[order]):
            for omit in range(len(simplex)):
                face = simplex[:omit]+simplex[omit+1:]
                ri.append(lookup[face]); ci.append(col); val.append((-1.)**omit)
        boundaries.append(sparse.csr_matrix((val,(ri,ci)), shape=(len(lookup),len(groups[order]))))
    blocks = [[None]*4 for _ in range(4)]
    for order in range(4):
        blocks[order][order] = sparse.csr_matrix((len(groups[order]),len(groups[order])))
    for order, boundary in enumerate(boundaries):
        blocks[order][order+1], blocks[order+1][order] = boundary, boundary.T
    d = sparse.bmat(blocks, format='csr')
    chain_error = max((abs((a@b)).max() if (a@b).nnz else 0.
                       for a,b in zip(boundaries, boundaries[1:])), default=0.)
    return groups, boundaries, d, chain_error


def solve_regression(train, target, validation, val_target, test, alpha_grid=(.001,.1,1.,10.,100.)):
    """Time-held selection; diagonal intercept is not penalized."""
    mean, std = train.mean(0), train.std(0).clip(1e-8)
    x = (train-mean)/std; v=(validation-mean)/std; q=(test-mean)/std
    ym = target.mean(0); y=target-ym
    best = None
    for alpha in alpha_grid:
        w = np.linalg.solve(x.T@x+alpha*np.eye(x.shape[1]), x.T@y)
        error = float(np.mean((v@w+ym-val_target)**2))
        if best is None or error<best[0]:best=(error,alpha,w)
    return q@best[2]+ym, best[1]


def dirac_dynamics(active, cutoff=.16, seed=20261116):
    """Cross-order dynamics of empirical coactivity cochains, not native phases."""
    groups, bs, d, chain_error = operators(active[:256], cutoff)
    raw = np.stack([active[:,s].all(axis=1).astype(float) for group in groups for s in group],axis=1)
    report=dirac_signal_dynamics(raw,d,[len(g) for g in groups],seed)
    return dict(cutoff=cutoff,chain_composition_max_error=chain_error,**report,
        meaning='time-held Dirac-neighbour predictive increment on statistical coactivity signals; not Dirac synchronization or native Dirac equation proof')


def dirac_signal_dynamics(raw,d,counts,seed=20261116):
    slices=np.cumsum([0]+counts)
    mean=raw[:256].mean(0); scale=raw[:256].std(0).clip(1e-8)
    y=(raw-mean)/scale
    # A single global scale preserves boundary composition and D^2 blocks.
    d=d/max(float(abs(d).sum(1).max()),1.)
    l=d@d
    dy=(d@y.T).T; ly=(l@y.T).T
    rng=np.random.default_rng(seed)
    permutations=[]
    for _ in range(15):
        # Block-orthogonal cell relabelling is exactly isospectral for D.
        permutation=np.concatenate([rng.permutation(np.arange(a,b)) for a,b in zip(slices[:-1],slices[1:])])
        inverse=np.argsort(permutation)
        permutations.append((d@y[:,inverse].T).T[:,permutation])
    report=[]
    for order,(a,b) in enumerate(zip(slices[:-1],slices[1:])):
        if a==b:continue
        target=y[2:,a:b]
        basic=np.stack((y[1:-1,a:b], y[:-2,a:b], ly[1:-1,a:b]),axis=-1)
        tr=slice(0,254);va=slice(262,382);te=slice(390,510)
        def predict(features):
            # Per-cell intercepts make reorienting a cochain and its operator
            # a change of coordinates, not a change in the fitted result.
            fm=features[tr].mean(0);tm=target[tr].mean(0)
            features=features-fm;centered=target-tm
            pred,alpha=solve_regression(features[tr].reshape(-1,features.shape[-1]),centered[tr].reshape(-1),
                features[va].reshape(-1,features.shape[-1]),centered[va].reshape(-1),
                features[te].reshape(-1,features.shape[-1]))
            return float(np.mean((pred-centered[te].reshape(-1))**2)),alpha
        base,ab=predict(basic)
        actual,aa=predict(np.concatenate((basic,dy[1:-1,a:b,None]),-1))
        null=[]
        for values in permutations:
            value,_=predict(np.concatenate((basic,values[1:-1,a:b,None]),-1));null.append(value)
        report.append(dict(order=order,cells=int(b-a),hodge_only_test_mse=base,dirac_test_mse=actual,
            relative_gain=(base-actual)/max(base,1e-15),null_test_mse=null,
            isospectral_tail_rank=(1+sum(v<=actual for v in null))/(1+len(null)),
            hodge_alpha=ab,dirac_alpha=aa))
    return dict(cells_by_order=np.diff(slices).tolist(),orders=report)


def executed_port_dirac(rows):
    """Independent logged port signals, rather than activity monomials.

    Faces designate supports of executed joint operations, not certified
    equivalences between causal paths. Fixed orientation is the phase order.
    """
    edges=list(combinations(range(5),2))
    faces=[(0,1,2),(0,3,4),(1,2,4),(1,2,3)]
    b1=np.zeros((5,len(edges)));b2=np.zeros((len(edges),len(faces)))
    for k,(i,j) in enumerate(edges):b1[i,k]=-1;b1[j,k]=1
    for k,face in enumerate(faces):
        for omit in range(3):b2[edges.index(face[:omit]+face[omit+1:]),k]=(-1.)**omit
    d=sparse.bmat([[sparse.csr_matrix((5,5)),sparse.csr_matrix(b1),None],
                  [sparse.csr_matrix(b1.T),sparse.csr_matrix((10,10)),sparse.csr_matrix(b2)],
                  [None,sparse.csr_matrix(b2.T),sparse.csr_matrix((4,4))]],format='csr')
    signals=[]
    for row in rows:
        x=extract(row);credit=row['read_credit'];q=row['question_context']
        settled=row['internal_rollout']['settled_two_turn']
        edge_values=[row['relation_wood']['prediction_delta'], q['decision_byte_probability_tv'],
            row['past_present']['workspace_rms'],credit['heldout']['credit_sum'],
            row['internal_rollout']['choice_probability_tv'],
            row['causal_program']['mother_prequential']['mse'],
            settled['prediction']-settled['baseline'] if settled else 0.,
            row['self_bytes'],row['actor_norm'],row['progress']]
        face_values=[row['relation_wood']['choice_probability_tv_without_relation'],
            credit['response']['credit_sum'],row['wood_to_fire']['policy_logit_F_effect_max'],
            row['causal_program']['mother_prequential']['log_score']]
        signals.append(np.r_[x[[2,4,9,12]],row['value_now'],edge_values,face_values])
    raw=np.asarray(signals,dtype=float)
    if not np.isfinite(raw).all():raise ValueError('missing executed port: do not fabricate a zero signal')
    report=dirac_signal_dynamics(raw,d,[5,10,4])
    return dict(**report,edge_roles=[[PHASES[i],PHASES[j]] for i,j in edges],
        face_roles=[[PHASES[i] for i in face] for face in faces],
        node_signals=[NAMES[i] for i in (2,4,9,12)]+['metal.value_now'],
        edge_signals=['relation_prediction_delta','question_content_to_expression_TV','memory_workspace_rms',
            'heldout_read_credit','imagination_to_choice_TV','world_response_forecast_MSE',
            'settled_prediction_minus_baseline','actual_self_bytes','actual_actor_gradient_norm','actual_heldout_learning_progress'],
        face_signals=['relation_to_choice_counterfactual_TV','actual_response_read_credit',
                     'wood_to_fire_calibrated_logit_effect','selected_action_response_log_score'],
        chain_composition_max_error=float(abs(b1@b2).max()),
        phase_probe=dirac_phase_probe(raw[:,:15],b1),
        meaning='observed node/edge/joint-operation port dynamics over the entire executed loop; measurement complex, not an installed Dirac mechanism or a synchronization proof')


def dirac_phase_probe(raw,b1):
    """Test Eq.19/34/37/38 of Calmon et al. on observed port phases.

    Band phases are descriptive coordinates. This is a test of the specified
    Dirac signature, not a claim that a logged error scalar is a native rotor.
    Each 128-event block is filtered independently and margins discarded.
    """
    d=np.block([[np.zeros((5,5)),b1],[b1.T,np.zeros((10,10))]])
    lap=d@d;gamma=np.r_[np.ones(5),-np.ones(10)];degree=np.r_[np.full(5,4.),np.full(10,2.)]
    filt=signal.butter(2,[1/64,1/8],btype='bandpass',fs=1.,output='sos')
    phases=[]
    for block in np.array_split(raw,4):
        filtered=signal.sosfiltfilt(filt,signal.detrend(block,axis=0,type='linear'),axis=0)
        phases.append(np.unwrap(np.angle(signal.hilbert(filtered,axis=0)),axis=0))
    def score(values):
        states=[];derivatives=[];blocks=[]
        for index,phase in enumerate(values):
            states.append(phase[16:-17]);derivatives.append(np.diff(phase,axis=0)[16:-16])
            blocks.extend([index]*len(states[-1]))
        phase=np.concatenate(states);speed=np.concatenate(derivatives);blocks=np.asarray(blocks)
        train=blocks<2;test=blocks==3
        unc=(d@np.sin((d@phase.T))).T
        coupled=(d@np.sin((d@phase.T)-(gamma/degree)[:,None]*(lap@phase.T))).T
        errors=[];couplings=[]
        for force in (unc,coupled):
            ym=speed[train].mean(0);fm=force[train].mean(0)
            f=force[train]-fm;y=speed[train]-ym
            k=max(0.,-float((f*y).mean())/max(float((f*f).mean()),1e-15))
            error=float(np.mean((ym-k*(force[test]-fm)-speed[test])**2))
            errors.append(error);couplings.append(k)
        theta=phase[:,:5];psi=(b1@phase[:,5:].T).T
        alpha=theta+psi/2;beta=5/4*(theta-theta.mean(1,keepdims=True))-psi
        return dict(uncoupled_mse=errors[0],eq19_mse=errors[1],
            relative_gain=(errors[0]-errors[1])/max(errors[0],1e-15),
            fitted_nonnegative_couplings=couplings,
            alpha_order_mean=float(abs(np.exp(1j*alpha).mean(1)).mean()),
            beta_order_mean=float(abs(np.exp(1j*beta).mean(1)).mean()))
    actual=score(phases);rng=np.random.default_rng(20261116);null=[]
    for _ in range(31):
        shifted=[]
        for phase in phases:
            copy=phase.copy()
            for j in range(5,15):
                rotated=np.roll(phase[:,j],int(rng.integers(16,113)))
                copy[:,j]=np.unwrap(np.angle(np.exp(1j*rotated)))
            shifted.append(copy)
        null.append(score(shifted))
    return dict(actual=actual,edge_phase_shift_controls=null,
        order_tail_ranks={name:(1+sum(q[name]>=actual[name] for q in null))/32
                          for name in ('alpha_order_mean','beta_order_mean','relative_gain')},
        band_period_external_cycles=[8,64],filter_block=128,discard_margin=16,
        meaning='specified Dirac phase-lag signature on independently logged whole-loop ports; descriptive band-phase test, not evidence of native rotor identity, hysteresis or spontaneous no-input synchronization')


def triadic_dynamics(values):
    # One specified scalar per phase, fixed before looking at results.
    x=values[:,[0,4,9,12,16]]
    xm=x[:256].mean(0);xs=x[:256].std(0).clip(1e-8);x=(x-xm)/xs
    tr=slice(0,254);va=slice(262,382);te=slice(390,510)
    result=[]
    for j in range(5):
        basic=np.column_stack((x[1:-1],x[1:-1]**2,x[:-2,j]))
        pairs=list(combinations([i for i in range(5) if i!=j],2))
        cross=np.stack([x[1:-1,i]*x[1:-1,k] for i,k in pairs],-1)
        target=x[2:,j]
        errors=[]
        for features in (basic,np.column_stack((basic,cross))):
            pred,alpha=solve_regression(features[tr],target[tr],features[va],target[va],features[te])
            errors.append((float(np.mean((pred-target[te])**2)),alpha))
        result.append(dict(target_phase=PHASES[j],baseline_mse=errors[0][0],
            regulated_mse=errors[1][0],relative_gain=(errors[0][0]-errors[1][0])/max(errors[0][0],1e-15),
            distinct_source_phase_pairs=[[PHASES[i],PHASES[k]] for i,k in pairs],
            alpha=[e[1] for e in errors]))
    return result


def usage_topology(rows, window=32):
    history,counts=deque(),Counter();sequence=[]
    previous=set()
    for row in rows:
        roles=row['causal_program']['roles']
        relation=f"r:{roles['identity']}:{roles['generation']}"
        edges={(relation,u['identity']) for u in row['joint_relation_update']['variable_uses']}
        history.append(edges);counts.update(edges)
        if len(history)>window:
            for edge in history.popleft():
                counts[edge]-=1
                if not counts[edge]:del counts[edge]
        current=set(counts);parent={}
        def find(a):
            parent.setdefault(a,a)
            while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
            return a
        for a,b in current:
            a,b=find(a),find(b)
            if a!=b:parent[a]=b
        sizes=Counter(find(a) for a in parent)
        sequence.append(dict(cycle=row['cycle'],nodes=len(parent),edges=len(current),
            components=len(sizes),beta1=len(current)-len(parent)+len(sizes),
            giant_fraction=max(sizes.values(),default=0)/max(len(parent),1),
            opened=len(current-previous),closed=len(previous-current)))
        previous=current
    r=np.asarray([s['giant_fraction'] for s in sequence[window:]])
    return dict(window=window,sequence=sequence,giant_fraction_range=[float(r.min()),float(r.max())],
        giant_fraction_std=float(r.std()),giant_fraction_mean=float(r.mean()),
        mean_beta1=float(np.mean([s['beta1'] for s in sequence[window:]])),
        lag1_correlation=float(np.corrcoef(r[:-1],r[1:])[0,1]),
        lag2_correlation=float(np.corrcoef(r[:-2],r[2:])[0,1]),
        meaning='actually executed relation-unknown usage graph with a fixed sliding observation window; not proof of the triadic-percolation transition')


def usage_order_controls(rows, count=31, seed=20261116):
    """Same whole event edges and global degrees; rearrange eight-event blocks.

    Sliding windows themselves induce long autocorrelation. Recompute that
    artifact on all surrogates rather than treating lag1 as biological rhythm.
    """
    rng=np.random.default_rng(seed)
    actual=usage_topology(rows)
    null=[];local_null=[]
    for _ in range(count):
        indices=np.arange(len(rows)).reshape(-1,8)[rng.permutation(len(rows)//8)].reshape(-1)
        summary=usage_topology([rows[i] for i in indices])
        null.append({k:summary[k] for k in ('giant_fraction_std','giant_fraction_mean',
                                           'mean_beta1','lag1_correlation','lag2_correlation')})
        indices=np.concatenate([np.arange(start,start+128).reshape(-1,8)[rng.permutation(16)].reshape(-1)
                                for start in range(0,len(rows),128)])
        summary=usage_topology([rows[i] for i in indices])
        local_null.append({k:summary[k] for k in ('giant_fraction_std','giant_fraction_mean',
                                                 'mean_beta1','lag1_correlation','lag2_correlation')})
    keys=list(null[0])
    return dict(block=8,surrogates=count,statistics={k:dict(actual=actual[k],
        null_values=[v[k] for v in null],
        upper_tail_rank=(1+sum(v[k]>=actual[k] for v in null))/(1+count),
        lower_tail_rank=(1+sum(v[k]<=actual[k] for v in null))/(1+count)) for k in keys},
        local_128_statistics={k:dict(actual=actual[k],null_values=[v[k] for v in local_null],
            upper_tail_rank=(1+sum(v[k]>=actual[k] for v in local_null))/(1+count),
            lower_tail_rank=(1+sum(v[k]<=actual[k] for v in local_null))/(1+count)) for k in keys},
        preserved='exact event edge sets, global edge multiplicity, global degree and within-eight-event order; local control additionally preserves every 128-event learning stage',
        meaning='temporal organization of the actually used relation graph relative to block reorder; not spontaneous triadic-percolation phase-transition evidence')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=int,default=1802);p.add_argument('--cycles',type=int,default=512)
    p.add_argument('--surrogates',type=int,default=31);a=p.parse_args()
    if a.cycles!=512:raise ValueError('preregistered 512-event split required')
    rows=[];hashes=[]
    for line in a.metrics.open('rb'):
        try:row=json.loads(line)
        except ValueError:continue
        if a.start<=row.get('cycle',-1)<a.start+a.cycles:
            rows.append(row);hashes.append(hashlib.sha256(line).hexdigest())
    if [r['cycle'] for r in rows]!=list(range(a.start,a.start+a.cycles)):
        raise ValueError('source must contain exactly one complete record for each frozen cycle')
    x=np.stack([extract(r) for r in rows]);assert np.isfinite(x).all()
    if not all(r['world']['source']=='text' for r in rows):raise ValueError('unexpected non-text world')
    act=activity(x)
    report=dict(schema=1,source=dict(path=str(a.metrics),start=a.start,end=rows[-1]['cycle'],
        rows=len(rows),line_sha256=hashes),names=NAMES,observations=x.tolist(),
        semantics='five-phase statistical activity topology, actual usage network and held dynamics; no capability/synchronization/chaos claim',
        activity_encoding='64-event independent linear detrending, residual above local median',
        persistence=persistence(act),dirac=[dirac_dynamics(act,c) for c in (.08,.16)],
        triadic=triadic_dynamics(x),usage=usage_topology(rows),
        executed_port_dirac=executed_port_dirac(rows),usage_controls=usage_order_controls(rows),surrogates=[])
    for i in range(a.surrogates):
        shifted=phase_shift(x,np.random.default_rng(20261116+i))
        # Shift the already encoded activity so its complete within-phase
        # event distribution is preserved; re-detrending a wrapped trend is
        # a different null, not just loss of cross-phase coordination.
        surrogate_ph=persistence(phase_shift(act,np.random.default_rng(20261116+i)))
        for dimension in surrogate_ph['dimensions']:dimension.pop('bars',None)
        report['surrogates'].append(dict(persistence=surrogate_ph,
            triadic=triadic_dynamics(shifted)))
        if (i+1)%4==0:print(f'surrogates {i+1}/{a.surrogates}',flush=True)
    for d in (1,2):
        original=report['persistence']['dimensions'][d]['maximum_lifetime']
        null=[s['persistence']['dimensions'][d]['maximum_lifetime'] for s in report['surrogates']]
        report['persistence']['dimensions'][d]['phase_shift_tail_rank']=(1+sum(v>=original for v in null))/(1+len(null))
    for j,item in enumerate(report['triadic']):
        null=[s['triadic'][j]['relative_gain'] for s in report['surrogates']]
        item['phase_shift_gain_tail_rank']=(1+sum(v>=item['relative_gain'] for v in null))/(1+len(null))
        max_null=[max(q['relative_gain'] for q in s['triadic']) for s in report['surrogates']]
        item['five_targets_max_statistic_tail_rank']=(1+sum(v>=item['relative_gain'] for v in max_null))/(1+len(max_null))
    # Bind results to exact implementation and data; no checksum depends on
    # the authoritative live checkpoint's subsequently rotating file name.
    report['analyzer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['schema']=2
    report['measurement_repairs']=['replace inactive legacy long_loss with actual internal_transition_loss',
        'encode activity before phase shifts to preserve all within-phase joint frequencies',
        'use unit-equivariant per-cell standardization instead of an absolute .02 scale floor',
        'control sliding-window graph autocorrelation with block-reordered actual event edges']
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(a.output),rows=len(rows),
        h1_lifetime=report['persistence']['dimensions'][1]['maximum_lifetime'],
        h2_lifetime=report['persistence']['dimensions'][2]['maximum_lifetime'],
        executed_port_dirac=report['executed_port_dirac']['orders'],
        usage_range=report['usage']['giant_fraction_range']),ensure_ascii=False))


if __name__=='__main__':main()
