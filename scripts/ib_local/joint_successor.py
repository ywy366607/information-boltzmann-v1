"""Coherent finite-evidence successor uncertainty and native-cell integration.

Schema5 keeps the established mean reconciliation and all coordinate scales.
It restores the residual scatter's cross-output entries, retaining correlated
program disagreement. Learning scores this same multivariate Student-t law.
Cheap posterior draws are partitioned by the ORIGINAL address cells; bounded
representatives then execute the original expensive decision program.
"""
from functools import lru_cache

import torch


def coordinate(distributions,weights):
    from scripts.ib_local.joint_relation_update import overlap_consensus
    from scripts.ib_local.successor_joint_reference import residual_scatter
    result=overlap_consensus(distributions,weights)
    w=weights/weights.sum();mean=result['mean']
    scale=mean.new_zeros(len(mean),mean.shape[-1],mean.shape[-1])
    for weight,d in zip(w,distributions):
        # Original posterior treats residual noise evidence as a frozen scale
        # statistic; its learned prior scale and predictive means keep gradients.
        with torch.no_grad():
            from scripts.ib_local.preview_runtime import _scope, tensor_key
            active = _scope.get()
            if active is not None:
                r_id = getattr(d['ridge'], '_cached_float', None) if torch.is_tensor(d['ridge']) else d['ridge']
                sc_key = ('joint_residual_scatter', tensor_key(d['x']), tensor_key(d['y']), r_id)
                sc_entry = active[1].get(sc_key)
                if sc_entry is not None:
                    scatter = sc_entry[0]
                else:
                    scatter = residual_scatter(d['x'], d['y'], d['ridge']).to(mean)
                    active[1][sc_key] = (scatter, (d['x'], d['y']))
            else:
                scatter = residual_scatter(d['x'], d['y'], d['ridge']).to(mean)
        off=scatter-torch.diag(scatter.diag())
        full=torch.diag_embed(d['scale2'])+(1+d['leverage'][:,None,None])*off/float(d['df'])
        delta=d['mean']-mean
        scale=scale+weight*(full+delta[:,:,None]*delta[:,None,:])
    result['scale_matrix']=(scale+scale.transpose(-1,-2))*.5
    return result


def score(distribution,target):
    scale=distribution['scale_matrix']
    factor=torch.linalg.cholesky(scale)
    delta=(target-distribution['mean']).unsqueeze(-1)
    whitened=torch.linalg.solve_triangular(factor,delta,upper=False).squeeze(-1)
    df=torch.as_tensor(distribution['df'],device=target.device,dtype=target.dtype)
    width=target.shape[-1]
    logdet=factor.diagonal(dim1=-2,dim2=-1).log().sum(-1)
    return (torch.lgamma((df+width)/2)-torch.lgamma(df/2)-width/2*torch.log(df*torch.pi)-
        logdet-(df+width)/2*torch.log1p(whitened.square().sum(-1)/df))/width


@lru_cache(maxsize=16)
def standardized(df,width,power=12):
    import numpy as np
    from scipy.special import ndtri,gammaincinv
    from scipy.stats import qmc
    u=qmc.Sobol(d=width+1,scramble=True,seed=191119).random_base2(power)
    normal=ndtri(u[:,:width])
    radial=np.sqrt(2*gammaincinv(df/2,u[:,-1])/df)
    return torch.from_numpy(normal/radial[:,None]).float()


def reduce_cells(points,whitened,identities,budget):
    """Preserve empirical mass; each representative is an actual native point.

Highest-mass cells get distinct representatives. The final stratum retains
all remaining mass and uses one of its own points. For a cell-constant read,
the grouped estimator is exact when every occupied cell fits the budget.
"""
    unique,count=identities.unique(return_counts=True)
    ordered=sorted(zip(unique.tolist(),count.tolist()),key=lambda p:(-p[1],p[0]))
    if budget<1:raise ValueError('positive native execution budget required')
    chosen=ordered if len(ordered)<=budget else ordered[:budget-1]
    groups=[identities==identity for identity,_ in chosen]
    labels=[identity for identity,_ in chosen]
    if len(ordered)>budget:
        remainder=torch.ones_like(identities,dtype=torch.bool)
        for mask in groups:remainder &= ~mask
        groups.append(remainder);labels.append('remaining_mass')
    selected=[];weights=[];trace=[]
    for label,mask in zip(labels,groups):
        indices=mask.nonzero().flatten()
        x=whitened[indices]
        nearest=(x-x.mean(0)).square().sum(-1).argmin()
        index=indices[nearest]
        selected.append(points[index]);weights.append(len(indices)/len(points))
        trace.append(dict(stratum=label,representative_cell=int(identities[index]),
                          count=len(indices)))
    return torch.stack(selected),points.new_tensor(weights),trace


@torch.no_grad()
def quadrature(engine,state,source,action,mother,distribution,*,project=None):
    from scripts.ib_local.prospective_decision import prepare_boundary
    from scripts.ib_local import successor_address_geometry as geometry
    if project is None:
        project=prepare_boundary(engine,state,source,action,mother)
    raw=project(distribution['mean'])
    packet=geometry.snapshot(engine.model,raw,distribution['mean'])
    factor=torch.linalg.cholesky(distribution['scale_matrix'][0])
    white=standardized(float(distribution['df']),distribution['mean'].shape[-1]).to(source)
    points=distribution['mean']+white@factor.T
    identities=geometry.addresses(packet,points)['identity']
    selected,weights,strata=reduce_cells(points,white,identities,engine.joint_proposal_budget)
    return selected[None],weights,dict(samples=len(points),strata=strata,
        fresh_mass=float((identities==-1).float().mean()),
        represented_cells=sorted(set(identities.tolist())),
        semantics='finite posterior mass through original native address cells')
