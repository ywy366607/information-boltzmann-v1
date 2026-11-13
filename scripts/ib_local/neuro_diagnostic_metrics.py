"""Defined information/network diagnostics; no biological thresholds implied."""
from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.optimize import minimize_scalar
from scipy.special import zeta

from scripts.ib_local.closed_loop_information import logdet, block_covariance, mean_interval


def mismatched_information(beta, sx, sy, cyx, B, noise):
    """Gaussian I*(beta), Oizumi et al. 2016 mismatched decoding objective.

    q(y|x)=N(Bx,noise), p(x,y) is the joint Gaussian supplied by the caller.
    This is not conditional mutual information or a sum of transfer entropies.
    """
    if beta == 0:
        return 0.
    riB = np.linalg.solve(noise, B)
    residual = sy-B@cyx.T-cyx@B.T+B@sx@B.T
    K = np.linalg.inv(sx)+beta*B.T@riB
    value = (logdet(sx)+logdet(K) + beta*np.trace(np.linalg.solve(noise, sy-residual))
             - beta**2*np.trace(np.linalg.solve(K,riB.T@sy@riB)))/2
    return float(value)


def phi_star_gaussian(A, sx, innovation, blocks):
    sy=innovation+A@sx@A.T
    cyx=A@sx
    mi=.5*(logdet(sy)-logdet(innovation))
    B, noise=block_covariance(A,sx,innovation,blocks)
    # beta=0 guarantees nonnegative decodable information; extend the search
    # interval if its largest endpoint is still preferred.
    upper=4.
    while True:
        opt=minimize_scalar(lambda b:-mismatched_information(b,sx,sy,cyx,B,noise),
                            bounds=(0,upper),method='bounded',options={'xatol':1e-9})
        if opt.x < .95*upper or upper >= 256:
            break
        upper*=2
    decoded=max(0.,-float(opt.fun))
    return dict(phi_star_nats=float(mi-decoded),mutual_information_nats=float(mi),
                mismatched_information_nats=decoded,beta=float(opt.x),
                search_interior=bool(opt.x < .95*upper),
                bounds_ok=bool(-1e-7 <= mi-decoded <= mi+1e-7))


def paired_effect(values, seed=51):
    values=np.asarray(values,float)
    rng=np.random.default_rng(seed)
    # Text segments, not bytes, are sampling units.
    boots=values[rng.integers(0,len(values),(5000,len(values)))].mean(1)
    signs=rng.choice([-1.,1.],(9999,len(values)))
    null=(signs*values).mean(1)
    return dict(mean=float(values.mean()),ci95=np.quantile(boots,[.025,.975]).tolist(),
                paired_text_segments=len(values),
                sign_flip_two_sided_p=float((1+(np.abs(null)>=abs(values.mean())).sum())/10000))


def factorial_interactions(losses):
    """Keys M{0/1}T{0/1}R{0/1}; vectors are matched segment NLLs."""
    results={}
    for r in (0,1):
        l00,l01,l10,l11=[np.asarray(losses[f'M{m}T{t}R{r}']) for m,t in ((0,0),(0,1),(1,0),(1,1))]
        benefit=l10+l01-l11-l00
        results[f'R{r}']=dict(joint_use_superadditive_benefit=paired_effect(benefit),
            joint_lesion_superadditive_damage=paired_effect(-benefit),
            memory_benefit_with_table=paired_effect(l01-l11),
            memory_benefit_without_table=paired_effect(l00-l10),
            table_benefit_with_memory=paired_effect(l10-l11))
    return results


def graph_summary(graph):
    connected=nx.is_connected(graph) if len(graph) else False
    components=sorted((len(c) for c in nx.connected_components(graph)),reverse=True)
    return dict(nodes=len(graph),edges=graph.number_of_edges(),connected=connected,
                components=components,largest_component_fraction=components[0]/len(graph),
                transitivity=float(nx.transitivity(graph)),
                global_efficiency=float(nx.global_efficiency(graph)),
                path_length=float(nx.average_shortest_path_length(graph)) if connected else None)


def topology_report(channels, surrogates=39, seed=91):
    """Degree-matched graph null; disconnected distances never discarded."""
    var=channels.var(0)
    active=var>1e-12
    if active.sum()<3:
        return dict(active_channels=int(active.sum()),status='insufficient varying channels')
    corr=np.corrcoef(channels[:,active].T)
    upper=np.triu_indices(len(corr),1)
    order=np.argsort(np.abs(corr[upper]))
    nedge=max(1,int(.2*len(order)))
    adjacency=np.zeros_like(corr,dtype=bool)
    selected=order[-nedge:]
    adjacency[upper[0][selected],upper[1][selected]]=True
    adjacency |= adjacency.T
    graph=nx.from_numpy_array(adjacency)
    observed=graph_summary(graph)
    null=[]
    for i in range(surrogates):
        other=graph.copy()
        try:
            nx.double_edge_swap(other,nswap=10*nedge,max_tries=300*nedge,seed=seed+i)
            row=graph_summary(other); row['swaps_completed']=True
        except nx.NetworkXAlgorithmError:
            row=graph_summary(other); row['swaps_completed']=False
        null.append(row)
    c=np.array([r['transitivity'] for r in null])
    e=np.array([r['global_efficiency'] for r in null])
    paths=[r['path_length'] for r in null if r['path_length'] is not None]
    sigma=None
    if observed['connected'] and len(paths)==surrogates and c.mean()>0:
        sigma=(observed['transitivity']/c.mean())/(observed['path_length']/np.mean(paths))
    return dict(observed=observed,degree_matched_null=null,classic_small_world_sigma=sigma,
                clustering_upper_tail_p=float((1+(c>=observed['transitivity']).sum())/(surrogates+1)),
                efficiency_lower_tail_p=float((1+(e<=observed['global_efficiency']).sum())/(surrogates+1)),
                null_mean_clustering=float(c.mean()),null_mean_efficiency=float(e.mean()),
                note='Correlation topology, not structural connectivity; no unreachable pairs discarded.')


def sample_powerlaw_tail(alpha,xmin,count,rng):
    """Discrete conditional Zipf draws by Hurwitz-zeta survival inversion.

    Avoid rejection sampling: it becomes effectively impossible for large
    xmin/alpha, exactly where short measured avalanche tails often lie.
    """
    target=rng.random(count)*zeta(alpha,xmin)
    lower=np.full(count,xmin-1,dtype=np.int64)
    upper=np.full(count,xmin,dtype=np.int64)
    while True:
        grow=zeta(alpha,upper.astype(float)+1)>target
        if not np.any(grow):break
        if np.any(upper[grow]>np.iinfo(np.int64).max//2):
            raise OverflowError('fitted tail exceeds representable event counts')
        upper[grow]*=2
    while np.any(upper-lower>1):
        middle=lower+(upper-lower)//2
        lower_half=zeta(alpha,middle.astype(float)+1)<=target
        upper=np.where(lower_half,middle,upper)
        lower=np.where(lower_half,lower,middle)
    return upper


def avalanche_report(channels, threshold=2.5, bin_size=1):
    """Threshold-crossing event cascades; censored terminal cascades excluded.

    This is an event-based diagnostic on continuous hidden activations, not
    measured biological spikes. Power-law comparison is descriptive, with a
    minimum tail count; it cannot establish self-organized criticality alone.
    """
    if channels.ndim==2:channels=channels[None]
    z=(channels-channels.mean((0,1)))/np.maximum(channels.std((0,1)),1e-10)
    sizes=[];durations=[];censored=0
    for segment in z:
        # Never join the last event of one text segment to the next segment.
        over=np.abs(segment)>threshold
        events=over[1:] & ~over[:-1]
        n=len(events)//bin_size
        population=events[:n*bin_size].reshape(n,bin_size,-1).sum((1,2))
        size=duration=0
        left_censored=bool(len(population) and population[0]>0)
        for count in population:
            if left_censored:
                if count==0:left_censored=False
                continue
            if count:
                size+=int(count);duration+=1
            elif duration:
                sizes.append(size);durations.append(duration);size=duration=0
        censored+=bool(duration)
    result=dict(threshold=threshold,bin_bytes=bin_size,completed=len(sizes),
                censored_segments=int(censored),sizes=sizes,durations=durations)
    if len(sizes)<100:
        result['tail_fit_status']='fewer than 100 completed cascades'
        return result
    values=np.array(sizes)
    xmin=max(1,int(np.median(values)))
    tail=values[values>=xmin]
    opt=minimize_scalar(lambda a:len(tail)*np.log(zeta(a,xmin))+a*np.log(tail).sum(),
                        bounds=(1.01,8),method='bounded')
    alpha=float(opt.x)
    lp=-alpha*np.log(tail)-np.log(zeta(alpha,xmin))
    geometric_p=1/(1+np.mean(tail-xmin))
    if geometric_p==1:
        le=np.zeros(len(tail))
    else:
        le=np.log(geometric_p)+(tail-xmin)*np.log1p(-geometric_p)
    result.update(xmin=xmin,tail_count=len(tail),powerlaw_exponent=alpha,
                  tail_loglik_powerlaw_minus_geometric=float((lp-le).sum()),
                  tail_fit_status='conditional fixed-xmin bootstrap; not a finite-size scaling test')
    def ks(values,a):
        u=np.unique(values)
        ordered=np.sort(values)
        # A discrete CDF may differ most just before an observed jump.
        # Check both sides, including support points absent from the sample.
        empirical_right=np.searchsorted(ordered,u,side='right')/len(values)
        empirical_left=np.searchsorted(ordered,u,side='left')/len(values)
        theory_right=1-zeta(a,u+1)/zeta(a,xmin)
        theory_left=1-zeta(a,u)/zeta(a,xmin)
        return float(max(np.max(np.abs(empirical_right-theory_right)),
                         np.max(np.abs(empirical_left-theory_left))))
    observed=ks(tail,alpha)
    rng=np.random.default_rng(51);null=[]
    for _ in range(99):
        synthetic=sample_powerlaw_tail(alpha,xmin,len(tail),rng)
        fit=minimize_scalar(lambda a:len(synthetic)*np.log(zeta(a,xmin))+a*np.log(synthetic).sum(),
                            bounds=(1.01,8),method='bounded')
        null.append(ks(synthetic,float(fit.x)))
    result.update(powerlaw_ks=observed,powerlaw_bootstrap_p=(1+sum(v>=observed for v in null))/100,
                  powerlaw_bootstrap_draws=99)
    return result


def calibration_report(probabilities, targets, bins=10):
    p=np.asarray(probabilities); y=np.asarray(targets)
    conf=p.max(1); correct=p.argmax(1)==y
    rows=[]
    for ids in np.array_split(np.argsort(conf),bins):
        if len(ids):
            rows.append(dict(count=len(ids),confidence=float(conf[ids].mean()),accuracy=float(correct[ids].mean())))
    return dict(top1_accuracy=float(correct.mean()),mean_confidence=float(conf.mean()),
                ece=float(sum(r['count']*abs(r['confidence']-r['accuracy']) for r in rows)/len(y)),
                multiclass_brier=float(np.mean((p*p).sum(1)-2*p[np.arange(len(y)),y]+1)),bins=rows)


def effective_dimension(channels):
    values=np.linalg.eigvalsh(np.cov(channels,rowvar=False)).clip(0)
    w=values/values.sum() if values.sum()>0 else np.zeros_like(values)
    return dict(participation_ratio=float(1/(w*w).sum()) if np.any(w) else 0.,
                spectral_entropy=float(-(w*np.log(w+1e-30)).sum()),channels=channels.shape[1])


def gaussian_mmi_pid(covariance,source_a,source_b,target):
    """Declared minimum-mutual-information PID, not Williams--Beer I_min."""
    def mi(left,right):
        both=list(left)+list(right)
        return .5*(logdet(covariance[np.ix_(left,left)])+
                   logdet(covariance[np.ix_(right,right)])-
                   logdet(covariance[np.ix_(both,both)]))
    a,b=mi(source_a,target),mi(source_b,target)
    joint=mi(list(source_a)+list(source_b),target)
    redundant=min(a,b)
    return dict(definition='Gaussian MMI PID, redundancy=min(I(A;Y),I(B;Y))',
                mi_a=a,mi_b=b,mi_joint=joint,redundancy=redundant,
                unique_a=a-redundant,unique_b=b-redundant,synergy=joint-max(a,b))
