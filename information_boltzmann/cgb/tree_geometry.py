"""Sampling-invariant tree-chain geometry for real and generated cable trees."""
from __future__ import annotations

import numpy as np


def chain_geometry(xyz, parent, root_start=None):
    """Contract nodes with exactly one child; keep actual cable arc lengths.

    xyz is the endpoint of each incoming cable; parent=-1 denotes a root.
    Generated root segments can carry a nonzero root_start, unlike skeleton
    root nodes. Root ordering need not be topological. Cycles are rejected.
    """
    x=np.asarray(xyz,float); p=np.asarray(parent,int); n=len(p)
    if x.shape!=(n,3) or np.any(p>=n) or np.any(p < -1):
        raise ValueError('Invalid tree coordinates or parent indices')
    valid=p>=0
    reach=np.r_[np.where(valid,p,n),n]
    for _ in range(max(1,n.bit_length())):
        reach=reach[reach]
    if np.any(reach[:n]!=n): raise ValueError('Parent graph contains a cycle')
    degree=np.bincount(p[valid],minlength=n)
    boundary=(degree!=1)|~valid
    length=np.zeros(n)
    length[valid]=np.linalg.norm(x[valid]-x[p[valid]],axis=1)
    if root_start is not None:
        length[~valid]=np.linalg.norm(x[~valid]-np.asarray(root_start)[~valid],axis=1)
    # Pointer doubling finds the upstream boundary with O(N) bounded arrays.
    ancestor=np.r_[np.where(valid,p,n),n]
    distance=np.r_[length,0.]
    boundary_ext=np.r_[boundary,True]
    for _ in range(max(1,n.bit_length()+1)):
        active=np.flatnonzero(~boundary_ext[ancestor[:n]])
        if not len(active): break
        old=ancestor[active].copy()
        distance[active]+=distance[old]
        ancestor[active]=ancestor[old]
    else: raise ValueError('Parent graph contains a cycle')
    ends=np.flatnonzero(boundary&valid)
    chain_lengths=distance[ends]
    upstream=ancestor[ends]
    chord=np.linalg.norm(x[ends]-x[upstream],axis=1)
    root_lengths=length[~valid].copy()
    root_children=np.flatnonzero((upstream<n)&~valid[np.minimum(upstream,n-1)])
    primary=chain_lengths[root_children]
    return {'length':float(length.sum()),'branches':int(np.count_nonzero(degree>=2)),
            'tips':int(np.count_nonzero(degree==0)),'roots':int(np.count_nonzero(~valid)),
            'chains':chain_lengths,'primary':primary,'root_lengths':root_lengths,
            'chord':chord,'degree':degree}


def forest_branch_probability(branches, frontier):
    """Critical binary forest conditioned on B branch sites and m open tips.

    Count completions T(B,m)=m/(2B+m)*choose(2B+m,B). The ratio after
    splitting one tip is B*(m+1)/(m*(2B+m-1)). No global history or gradient.
    """
    b,m=np.asarray(branches),np.asarray(frontier)
    if np.any(b<0) or np.any(m<1): raise ValueError('Nonnegative branch stock and live frontier required')
    return np.divide(b*(m+1),m*(2*b+m-1),out=np.zeros(np.broadcast_shapes(b.shape,m.shape)),where=b>0)
