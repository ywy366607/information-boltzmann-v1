"""Actual KC release sites with local receptor signals and compartment eligibility.

All supplied morphology volumes are retained. Positive reaction/transport laws
describe effective intracellular receptor signals; they do not identify a
molecular receptor subtype or reconstruct the axonal action-potential cable.
KC spikes retain the existing native cell dynamics. Every physical outgoing
path reads delayed site packets, never the KC-wide compatibility readout.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch import nn
try:
    from numba import prange
except ImportError:
    prange = range

from .fly_apl_cable import CableGeometry, diffuse, cable_state
from .fly_kc_site_stp import KC_SITE_STP_STATE_FIELDS, site_state, release_site_resources

KC_TERMINAL_VERSION = 1
KC_TERMINAL_PARAMETER_NAMES = ('kc_terminal_log_diffusion',)
KC_TERMINAL_ID_NAMES = ('kc_terminal_site_ids','kc_terminal_digest')
KC_TERMINAL_STATE_FIELDS = ('kc_terminal_gaba','kc_terminal_5ht','kc_terminal_receptor',
    'kc_terminal_packets','kc_terminal_eligibility','kc_terminal_eligibility_credit') + KC_SITE_STP_STATE_FIELDS


def _forest_order(parent, children, ptr):
    order=np.empty(len(parent),np.int64);end=0
    for i in range(len(parent)):
        if parent[i]<0:order[end]=i;end+=1
    cursor=0
    while cursor<end:
        node=order[cursor]
        for j in range(ptr[node],ptr[node+1]):order[end]=children[j];end+=1
        cursor+=1
    if end!=len(parent):raise ValueError('Native KC morphology is not a rooted forest')
    return order


def _forest_factor(parent,order,diagonal,weight):
    for j in range(len(order)-1,-1,-1):
        i=order[j];p=parent[i]
        if diagonal[i]<=0:raise ValueError('Nonpositive native tree pivot')
        if p>=0:
            diagonal[p]-=(weight[i]/diagonal[i])*weight[i]
    return diagonal


def _forest_solve(parent,order,diagonal,weight,rhs):
    out=rhs.copy()
    for j in range(len(order)-1,-1,-1):
        i=order[j];p=parent[i]
        if p>=0:
            ratio=weight[i]/diagonal[i]
            for k in range(rhs.shape[1]):out[p,k]+=ratio*out[i,k]
    for i in order:
        p=parent[i]
        ratio=weight[i]/diagonal[i] if p>=0 else 0.
        for k in range(rhs.shape[1]):
            out[i,k]=out[i,k]/diagonal[i]+(ratio*out[p,k] if p>=0 else 0.)
    return out


def _forest_product(area,parent,weight,value):
    out=area[:,None]*value
    for i in range(len(parent)):
        p=parent[i]
        if p>=0:
            for k in range(value.shape[1]):
                flow=weight[i]*(value[i,k]-value[p,k])
                out[i,k]+=flow;out[p,k]-=flow
    return out


def _forest_coefficients(area, first, second, conductance, owner, scales):
    diagonal = area.copy()
    parent_weight = np.zeros_like(area)
    for j in range(len(first)):
        w = conductance[j] * scales[owner[first[j]]]
        parent_weight[first[j]] = w
        diagonal[first[j]] += w
    for j in range(len(second)):
        diagonal[second[j]] += parent_weight[first[j]]
    return diagonal, parent_weight


def _forest_owned_residual(area, parent, weight, value, rhs):
    # rhs is private to this solve. Reuse it instead of allocating A@value.
    for i in range(len(parent)):
        for k in range(value.shape[1]):
            rhs[i,k] = area[i]*value[i,k] - rhs[i,k]
    for i in range(len(parent)):
        p=parent[i]
        if p>=0:
            for k in range(value.shape[1]):
                flow=weight[i]*(value[i,k]-value[p,k])
                rhs[i,k]+=flow;rhs[p,k]-=flow
    error=0.
    for i in range(len(parent)):
        for k in range(value.shape[1]):
            if not np.isfinite(value[i,k]) or not np.isfinite(rhs[i,k]):
                raise FloatingPointError('Nonfinite native tree solution/residual')
            error=max(error,abs(rhs[i,k]))
    return error


def _forest_trees(parent, order):
    """Root of every node's tree (execution metadata: trees are independent systems)."""
    tree=np.empty(len(parent),np.int64)
    for i in order:
        p=parent[i]
        tree[i]=i if p<0 else tree[p]
    return tree


def _forest_solve_trees(parent,seg_order,seg_ptr,diagonal,weight,rhs):
    """_forest_solve, one independent tree per thread.

    seg_order lists each tree's nodes contiguously in the global elimination
    order (a stable regrouping of it), so every node sees exactly the same
    sequence of floating-point operations: the result is bitwise identical.
    """
    out=np.empty_like(rhs)
    for t in prange(len(seg_ptr)-1):
        a,b=seg_ptr[t],seg_ptr[t+1]
        for j in range(a,b):
            i=seg_order[j]
            for k in range(rhs.shape[1]):out[i,k]=rhs[i,k]
        for j in range(b-1,a-1,-1):
            i=seg_order[j];p=parent[i]
            if p>=0:
                ratio=weight[i]/diagonal[i]
                for k in range(rhs.shape[1]):out[p,k]+=ratio*out[i,k]
        for j in range(a,b):
            i=seg_order[j];p=parent[i]
            ratio=weight[i]/diagonal[i] if p>=0 else 0.
            for k in range(rhs.shape[1]):
                out[i,k]=out[i,k]/diagonal[i]+(ratio*out[p,k] if p>=0 else 0.)
    return out


def _forest_owned_residual_trees(area,parent,seg_order,seg_ptr,weight,value,rhs):
    """_forest_owned_residual per independent tree (the contract check, not a result)."""
    n_trees=len(seg_ptr)-1
    errors=np.zeros(n_trees)
    bad=np.zeros(n_trees,np.bool_)
    for t in prange(n_trees):
        a,b=seg_ptr[t],seg_ptr[t+1]
        for j in range(a,b):
            i=seg_order[j]
            for k in range(value.shape[1]):
                rhs[i,k]=area[i]*value[i,k]-rhs[i,k]
        for j in range(a,b):
            i=seg_order[j];p=parent[i]
            if p>=0:
                for k in range(value.shape[1]):
                    flow=weight[i]*(value[i,k]-value[p,k])
                    rhs[i,k]+=flow;rhs[p,k]-=flow
        e=0.
        for j in range(a,b):
            i=seg_order[j]
            for k in range(value.shape[1]):
                if not np.isfinite(value[i,k]) or not np.isfinite(rhs[i,k]):bad[t]=True
                e=max(e,abs(rhs[i,k]))
        errors[t]=e
    if bad.any():raise FloatingPointError('Nonfinite native tree solution/residual')
    return errors.max() if n_trees else 0.


def _forest_diffuse_rows_trees(parent,seg_order,seg_ptr,diagonal,weight,area,rows):
    """Fused rows -> tree solve -> residual check -> rows, one tree per thread, in the geometry's precision.

    Per node the same operations as rhs=v*area, _forest_solve and _forest_owned_residual.
    Returns (out [R, n] float32, error [R], scale [R], finite [R]).
    """
    n_rows=rows.shape[0];n=rows.shape[1];n_trees=len(seg_ptr)-1
    out=np.empty((n_rows,n),np.float32)
    x=np.empty(n,area.dtype);res=np.empty(n,area.dtype)
    errors=np.zeros((n_rows,n_trees));scales=np.zeros((n_rows,n_trees));bad=np.zeros((n_rows,n_trees),np.bool_)
    for r in range(n_rows):
        for t in prange(n_trees):
            a,b=seg_ptr[t],seg_ptr[t+1]
            for j in range(a,b):
                i=seg_order[j];x[i]=rows[r,i]*area[i]
            for j in range(b-1,a-1,-1):
                i=seg_order[j];p=parent[i]
                if p>=0:x[p]+=(weight[i]/diagonal[i])*x[i]
            for j in range(a,b):
                i=seg_order[j];p=parent[i]
                ratio=weight[i]/diagonal[i] if p>=0 else 0.
                x[i]=x[i]/diagonal[i]+(ratio*x[p] if p>=0 else 0.)
            sc=0.
            for j in range(a,b):
                i=seg_order[j];rhs=rows[r,i]*area[i]
                sc=max(sc,abs(rhs));res[i]=area[i]*x[i]-rhs
            for j in range(a,b):
                i=seg_order[j];p=parent[i]
                if p>=0:
                    flow=weight[i]*(x[i]-x[p]);res[i]+=flow;res[p]-=flow
            e=0.
            for j in range(a,b):
                i=seg_order[j]
                if not np.isfinite(x[i]) or not np.isfinite(res[i]):bad[r,t]=True
                e=max(e,abs(res[i]));out[r,i]=np.float32(x[i])
            errors[r,t]=e;scales[r,t]=sc
    error=np.zeros(n_rows);scale=np.zeros(n_rows);finite=np.ones(n_rows,np.bool_)
    for r in range(n_rows):
        for t in range(n_trees):
            error[r]=max(error[r],errors[r,t]);scale[r]=max(scale[r],scales[r,t])
            if bad[r,t]:finite[r]=False
    return out,error,scale,finite


_TREE_KERNELS=None

def _tree_kernels():
    global _TREE_KERNELS
    if _TREE_KERNELS is None:
        from numba import njit
        _TREE_KERNELS=tuple(njit(cache=True)(f) for f in
            (_forest_order,_forest_factor,_forest_solve,_forest_product,_forest_coefficients,
             _forest_owned_residual,_forest_trees))+tuple(
            njit(cache=True,parallel=True)(f) for f in (_forest_solve_trees,_forest_owned_residual_trees,
                                                        _forest_diffuse_rows_trees))
    return _TREE_KERNELS


class _TreeMatrix:
    def __init__(self,area,parent,weight):self.area,self.parent,self.weight=area,parent,weight
    def __matmul__(self,value):return _tree_kernels()[3](self.area,self.parent,self.weight,value)


class _TreeFactor:
    def __init__(self,parent,order,diagonal,weight,segments=None):
        self.parent,self.order,self.diagonal,self.weight=parent,order,diagonal,weight
        self.segments=segments
    def solve(self,rhs):
        if self.segments is not None:
            return _tree_kernels()[7](self.parent,*self.segments,self.diagonal,self.weight,rhs)
        return _tree_kernels()[2](self.parent,self.order,self.diagonal,self.weight,rhs)


class _TreeEdgeWeights:
    """View actual edge weights through their unique child; no duplicate array."""
    def __init__(self, weight, first):
        self.weight, self.first = weight, first

    def __getitem__(self, index):
        return self.weight[self.first[index]]


class CellwiseGeometry(CableGeometry):
    """Exact tree elimination on every supplied node; no general-LU workspace.

    Root/leaf order is execution metadata, not a new connection or reduced
    compartment. Branch currents and the original full-volume residual remain.
    """
    # Precision of the native diffusion (2026-10-19): float32. The brain is not an equation to be solved to FP64;
    # the individual's own noise (intrinsic jitter, stochastic release) is orders of magnitude above float32
    # rounding. The residual contract is relative to float32: a tree solve accumulates ~depth * eps.
    DTYPE = np.float32
    RESIDUAL_TOLERANCE = 1e-3

    def __init__(self,*args):
        super().__init__(*args)
        self.area=self.area.astype(self.DTYPE)
        self.conductance=self.conductance.astype(self.DTYPE)
        # Exact indices into this forest, not anatomical IDs. Use the narrow
        # lossless representation and retain int64 for exceptionally large trees.
        index_dtype = (np.int32 if max(len(self.area), int(self.owner.max(initial=0)))
                       <= np.iinfo(np.int32).max else np.int64)
        self.first = self.first.astype(index_dtype, copy=False)
        self.second = self.second.astype(index_dtype, copy=False)
        self.owner = self.owner.astype(index_dtype, copy=False)
        self._owner_count = int(self.owner.max(initial=-1)) + 1
        if len(np.unique(self.first))!=len(self.first):raise ValueError('KC child has multiple parents')
        self.parent=np.full(len(self.area),-1,dtype=index_dtype);self.parent[self.first]=self.second
        sequence=np.argsort(self.second,kind='stable')
        children=self.first[sequence]
        ptr=np.r_[0,np.cumsum(np.bincount(self.second,minlength=len(self.area)))]
        self.order=_tree_kernels()[0](self.parent,children,ptr).astype(index_dtype, copy=False)
        # Independent trees, each contiguous in the (stably regrouped) elimination order.
        tree=_tree_kernels()[6](self.parent,self.order)
        regroup=np.argsort(tree[self.order],kind='stable')
        self.seg_order=self.order[regroup]
        roots=np.flatnonzero(self.parent<0)
        sizes=np.bincount(np.searchsorted(roots,tree),minlength=len(roots))
        self.seg_ptr=np.r_[0,np.cumsum(sizes)].astype(np.int64)
        self._cell_cache={}

    def factor(self,log_axial):
        values=tuple(log_axial.detach().double().cpu().tolist())
        if not all(np.isfinite(values)):raise ValueError('Nonfinite KC transport coefficient')
        if len(values) < self._owner_count:raise ValueError('Missing KC transport coefficient')
        if values in self._cell_cache:return self._cell_cache[values]
        # Discard an unused execution-cache reference before allocating its
        # replacement; autograd contexts still retain any factor they need.
        if len(self._cell_cache)==2:self._cell_cache.pop(next(iter(self._cell_cache)))
        diagonal,parent_weight=_tree_kernels()[4](
            self.area,self.first,self.second,self.conductance,self.owner,np.exp(values).astype(self.DTYPE))
        diagonal=_tree_kernels()[1](self.parent,self.order,diagonal,parent_weight)
        result=(_TreeMatrix(self.area,self.parent,parent_weight),
                _TreeFactor(self.parent,self.order,diagonal,parent_weight,(self.seg_order,self.seg_ptr)),
                _TreeEdgeWeights(parent_weight,self.first))
        self._cell_cache[values]=result
        return result

    def diffuse_rows(self, matrix, factor, rows):
        """rows: float32 [R, n] voltages -> float32 [R, n] solutions of (A) x = area * v (fused, see kernel)."""
        out,error,scale,finite=_tree_kernels()[9](self.parent,self.seg_order,self.seg_ptr,factor.diagonal,
                                                  factor.weight,self.area,np.ascontiguousarray(rows,np.float32))
        if not finite.all():raise FloatingPointError('Nonfinite native tree solution/residual')
        if np.any(error>self.RESIDUAL_TOLERANCE*np.maximum(1.,scale)):
            raise FloatingPointError('Native tree solve did not meet residual contract')
        return out

    def solve_owned_rhs(self, matrix, factor, rhs):
        result=factor.solve(rhs)
        scale=max(1.,float(rhs.max(initial=0.)),-float(rhs.min(initial=0.)))
        error=_tree_kernels()[8](self.area,self.parent,self.seg_order,self.seg_ptr,matrix.weight,result,rhs)
        if error>self.RESIDUAL_TOLERANCE*scale:
            raise FloatingPointError('Native tree solve did not meet residual contract')
        return result


def attach_kc_terminals(model, path, *, saved=None, metadata_only=False):
    path=Path(path)
    with np.load(path,allow_pickle=False) as archive:
        data={k:archive[k] for k in archive.files}
    device=model.edge_pre_e.device
    def buf(name,value,dtype=torch.long,persistent=False):
        model.register_buffer(name,torch.as_tensor(value,dtype=dtype,device=device),persistent=persistent)
    buf('kc_terminal_site_ids',data['site_ids'],persistent=True)
    buf('kc_terminal_digest',np.frombuffer(hashlib.sha256(path.read_bytes()).digest(),dtype=np.uint8).copy(),torch.uint8,True)
    if saved is not None and 'kc_terminal_log_diffusion' in saved:
        for name in KC_TERMINAL_ID_NAMES:
            if name not in saved or not torch.equal(saved[name].cpu(),getattr(model,name).cpu()):
                raise ValueError('KC terminal anatomical identity mismatch: '+name)
    model.kc_terminal_log_diffusion=nn.Parameter(torch.zeros(2,len(data['kc_index']),device=device))
    if saved is not None and 'kc_terminal_log_diffusion' in saved:
        model.load_state_dict({n:saved[n] for n in KC_TERMINAL_PARAMETER_NAMES},strict=False)
    model.kc_terminal_manifest=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if metadata_only:return dict(model.kc_terminal_manifest)
    if not hasattr(model,'apl_cable_index') or not hasattr(model,'mb_terminal_kc_index'):
        raise ValueError('Native KC site routing requires APL local release and MB axonal chemistry')
    if model.use_stp:
        raise ValueError('KC site-specific vesicle/STP state is not implemented; refuse global-STP substitution')
    if not np.array_equal(model.mb_terminal_kc_index.cpu().numpy(),data['kc_index']):
        raise ValueError('KC terminal body order mismatch')
    if not np.array_equal(model.comp_pre.cpu().numpy(),data['comp_pre']) or not np.array_equal(model.comp_post.cpu().numpy(),data['comp_post']):
        raise ValueError('KC compartment identity/order mismatch')
    for name,key in [('kct_index','kc_index'),('kct_owner','owner'),('kct_site_owner','site_owner'),('kct_site_node','site_node')]:buf(name,data[key])
    totals=np.bincount(data['owner'],weights=data['volume'],minlength=len(data['kc_index']))
    buf('kct_density',totals[data['owner']]/data['volume'],torch.float32)
    # Two species use the same actual geometry with separate learned material
    # coefficients. Cache their distinct sparse factorizations independently.
    geometry=CellwiseGeometry(data['volume'],data['u'],data['v'],data['g'],data['owner'])
    model.kct_geometries=(geometry,geometry)
    def segments(prefix,target):
        order=np.argsort(target,kind='stable');unique,count=np.unique(np.asarray(target)[order],return_counts=True)
        for key,v in [('order',order),('target',unique),('count',count)]:buf(prefix+'_'+key,v)
    def routing(prefix,arrays,target):
        for k,v in arrays.items():buf(prefix+'_'+k,v,torch.float32 if k in ('fraction','gaba','serotonin') else torch.long)
        segments(prefix,target)
    n=model.n_neurons
    def pair_indices(pre,post):
        if not len(pre):return np.full(len(data['fast_pre']),-1,dtype=np.int64)
        keys=pre.astype(np.int64)*n+post;order=np.argsort(keys);keys=keys[order]
        query=data['fast_pre']*n+data['fast_post'];pos=np.searchsorted(keys,query)
        valid=(pos<len(keys));valid &= keys[np.minimum(pos,max(0,len(keys)-1))]==query
        out=np.full(len(query),-1,dtype=np.int64);out[valid]=order[pos[valid]]
        return out
    comp_index=pair_indices(data['comp_pre'],data['comp_post'])
    kcd_index=(pair_indices(model.kcd_pre.cpu().numpy(),model.kcd_post.cpu().numpy())
               if hasattr(model,'kcd_pre') else np.full_like(comp_index,-1))
    for lag in range(1,5):
        sel=data['fast_delay']==lag
        routing(f'kct_fast_{lag}',{k:data['fast_'+k][sel] for k in ('edge','post','site','fraction')},data['fast_post'][sel])
        for label,index in [('comp',comp_index),('kcd',kcd_index)]:
            choose=sel&(index>=0)
            routing(f'kct_{label}_{lag}',{'site':data['fast_site'][choose],'edge':index[choose],
                'fraction':data['fast_fraction'][choose]},index[choose])
        choose=data['apl_delay']==lag
        routing(f'kct_apl_{lag}',{k:data['apl_'+k][choose] for k in ('edge','site','node','fraction')},data['apl_node'][choose])
    # Current small axonal chemistry weights already contain measured axonal
    # shares and serotonin normalization. Split them over actual local sites.
    gaba={};serotonin={}
    for lag in range(1,5):
        for p,g,s in zip(getattr(model,f'mbt_post_{lag}').cpu().tolist(),getattr(model,f'mbt_gaba_weight_{lag}').cpu().tolist(),getattr(model,f'mbt_5ht_weight_{lag}').cpu().tolist()):
            gaba[p]=g;serotonin[p]=s
    nk=len(data['kc_index']);nnodes=len(data['owner']);ns=len(data['source_index'])
    for lag in range(1,5):
        choose=data['receptor_delay']==lag
        source=data['receptor_source'][choose];node=data['receptor_node'][choose]
        pairs=source*nk+data['receptor_kc'][choose]
        arrays={k:data['receptor_'+k][choose] for k in ('source','node','pre_node','fraction')}
        arrays['gaba']=np.asarray([gaba.get(int(k),0.) for k in pairs])
        arrays['serotonin']=np.asarray([serotonin.get(int(k),0.) for k in pairs])
        arrays['pre']=data['source_index'][source]
        routing(f'kct_receptor_{lag}',arrays,source*nnodes+node)
    # Every live outgoing edge/compartment must have physical output locations.
    covered=set(map(int,data['fast_edge']))
    pre=model.edge_pre_e.cpu().numpy();weights=model.edge_weight_e.detach().cpu().numpy()
    live=np.flatnonzero(np.isin(pre,data['kc_index'])&(weights!=0))
    if any(int(i) not in covered for i in live):raise ValueError('Nonzero KC output edge has no actual site')
    if len(np.unique(comp_index[comp_index>=0]))!=len(data['comp_pre']):raise ValueError('Unlocated KC-MBON compartment')
    if hasattr(model,'kcd_pre') and len(np.unique(kcd_index[kcd_index>=0]))!=model.kcd_pre.numel():raise ValueError('Unlocated KC-DAN terminal feedback')
    model.kc_terminal_enabled=model.kc_terminal_gating_enabled=model.kc_terminal_transport_enabled=True
    return dict(model.kc_terminal_manifest)


def _sum(model,prefix,value,size):
    order=getattr(model,prefix+'_order')
    if not order.numel():return value.new_zeros(size)
    sums=torch.segment_reduce(value.index_select(0,order),'sum',lengths=getattr(model,prefix+'_count'))
    return value.new_zeros(size).index_copy_(0,getattr(model,prefix+'_target'),sums)


def _add_receptor_arrival(model, prefix, release, arrivals, amines):
    """Add one real delay's sparse contact sums into private dense workspaces."""
    order=getattr(model,prefix+'_order')
    count=getattr(model,prefix+'_count')
    target=getattr(model,prefix+'_target')
    gaba_sum=torch.segment_reduce((release*getattr(model,prefix+'_gaba')).index_select(0,order),
                                  'sum',lengths=count)
    amine_sum=torch.segment_reduce((release*getattr(model,prefix+'_serotonin')).index_select(0,order),
                                   'sum',lengths=count)
    arrivals.index_add_(0,target,gaba_sum)
    kept=target < amines.numel()
    amines.index_add_(0,target[kept],amine_sum[kept])


def terminal_state(model,current):
    owner=model.kct_owner;ns=model.mb_terminal_source_index.numel();nd=model.mb_terminal_n_dpm
    shapes=[(ns,owner.numel()),(nd,owner.numel()),(1,owner.numel()),(4,model.kct_site_owner.numel()),
        (1,model.comp_pre.numel()),(1,model.comp_pre.numel())]
    old=[current.mb_terminal_gaba,current.mb_dpm_concentration,current.mb_dpm_receptor]
    values=[]
    for i,(name,shape) in enumerate(zip(KC_TERMINAL_STATE_FIELDS[:6],shapes)):
        value=getattr(current,name,current.h.new_empty(0))
        if not value.numel():
            if i<3:
                value=old[i].index_select(1,owner) if old[i].numel() else current.h.new_zeros(shape)
            elif i==3:value=torch.cat([p.index_select(1,model.kct_index).index_select(1,model.kct_site_owner) for p in current.ring],0)
            elif i==4:value=(current.comp_trace.index_select(1,model.comp_edge_kc)
                if current.comp_trace.numel() else current.h.new_zeros(shape))
            else:value=(current.credit_comp_trace.index_select(1,model.comp_edge_kc) if current.credit_comp_trace.numel() else current.h.new_zeros(shape))
        if value.shape!=shape:raise ValueError('Nonempty KC terminal state has incompatible shape: '+name)
        values.append(value)
    return values + site_state(model,current)


def advance_kc_terminals(model,current,raw_packet,*,base_rates=None,conductance_gains=None,spikes=None,
                         local_inputs=None):
    """OLD source packets update local chemistry; NEW KC events emit site packets."""
    if not getattr(model,'kc_terminal_enabled',False):return None
    gaba,serotonin,receptor,packets,eligibility,credit,*_=terminal_state(model,current)
    lateral = ()
    if getattr(model, 'kc_lateral_enabled', False):
        from .fly_kc_lateral import advance_kc_lateral
        lateral = advance_kc_lateral(model, current, packets)
    ns,nnodes=gaba.shape;nd=serotonin.shape[0]
    arrivals=gaba.new_zeros(ns*nnodes);amines=gaba.new_zeros(nd*nnodes)
    apl_packets=cable_state(model,current)[-1]
    for lag in range(1,5):
        prefix=f'kct_receptor_{lag}';pre=getattr(model,prefix+'_pre')
        if not pre.numel():continue
        release=current.ring[lag-1].index_select(1,pre)[0]
        nodes=getattr(model,prefix+'_pre_node');local=nodes>=0
        # Node locations are fixed, not an arbitrary spatial cutoff.
        if bool(local.any()):
            slots=local.nonzero().flatten()
            local_release=apl_packets[lag-1].index_select(0,nodes[local])
            if not model.apl_output_enabled:local_release=local_release*0.
            release=release.index_copy(0,slots,local_release)
        release=release*getattr(model,prefix+'_fraction')
        # Reduce actual contacts first. Targets are unique within each delay;
        # sequential additions retain the original delay summation order without
        # constructing two dense source-by-node zero fields on every lag.
        _add_receptor_arrival(model,prefix,release,arrivals,amines)
    arrivals=arrivals.reshape(ns,nnodes)*model.kct_density
    amines=amines.reshape(nd,nnodes)*model.kct_density
    rates=base_rates if base_rates is not None else model.get_decay_rates()
    def spread(value):return value.expand_as(current.h).index_select(1,model.kct_index).index_select(1,model.kct_owner)
    rho=spread(rates[2]);gaba=rho*gaba+(1-rho)*arrivals
    a=-torch.expm1(-torch.exp(-model.mb_dpm_log_tau)).unsqueeze(1)
    serotonin=serotonin+a*(amines-serotonin)
    if model.kc_terminal_transport_enabled:
        gaba=diffuse(gaba,model.kc_terminal_log_diffusion[0],model.kct_geometries[0])
        serotonin=diffuse(serotonin,model.kc_terminal_log_diffusion[1],model.kct_geometries[1])
    total=serotonin.sum(0,keepdim=True);affinity=torch.exp(model.mb_dpm_log_affinity)
    from .fly_biophysical_operators import ligand_binding
    receptor=ligand_binding(receptor,total,affinity,torch.exp(-model.mb_dpm_log_off),clamp=True)
    site=model.kct_site_node;site_owner=model.kct_site_owner
    gains=conductance_gains if conductance_gains is not None else model.get_conductance_gains()
    gi=spread(gains[1]);leak=-torch.log(spread(rates[0]).clamp(1e-5,1-1e-7))
    fraction=gaba.new_ones(1,nnodes)
    if model.kc_terminal_gating_enabled and model.mb_terminal_enabled:
        if model.mb_terminal_gaba_enabled:
            gain=torch.exp(model.mb_terminal_log_gaba_gain).index_select(0,model.kct_owner)
            fraction=leak/(leak+gi*gain*gaba.sum(0,keepdim=True))
        if model.mb_dpm_5ht_enabled:
            gain=torch.exp(model.mb_terminal_log_5ht_gain).index_select(0,model.kct_owner)
            fraction=fraction*torch.exp(-gain*receptor)
    raw=raw_packet.index_select(1,model.kct_index).index_select(1,site_owner)
    fraction = fraction.index_select(1, site)
    if local_inputs and 'kc_calcium_gate' in local_inputs:
        from .fly_native_runtime import local_input
        fraction = fraction * local_input(local_inputs, 'kc_calcium_gate', fraction)
    if lateral:
        from .fly_kc_lateral import effector_gate
        fraction = fraction * effector_gate(model, lateral[1],
            torch.arange(site.numel(), device=site.device), 'calcium')
    emitted,resources,utilization=release_site_resources(
        model,current,raw,fraction,spikes)
    packets=torch.cat((emitted,packets[:3]),0)
    return [gaba,serotonin,receptor,packets,eligibility,credit,resources,utilization, *lateral]


def kc_arrival(model,current,kind,*,new_values=None,weights=None,instant=False):
    packets=(new_values[3] if new_values is not None else terminal_state(model,current)[3])
    if kind=='fast':size=model.n_neurons
    elif kind=='apl':size=model.aplc_owner.numel()
    elif kind=='comp':size=model.comp_pre.numel()
    elif kind=='kcd':size=model.kcd_pre.numel()
    else:raise ValueError('Unknown native terminal route')
    out=packets.new_zeros(size)
    for lag in range(1,5):
        prefix=f'kct_{kind}_{lag}';site=getattr(model,prefix+'_site')
        value=packets[0 if instant else lag-1].index_select(0,site)*getattr(model,prefix+'_fraction')
        if kind in ('fast','apl'):value=value*model.edge_weight_e.index_select(0,getattr(model,prefix+'_edge'))
        elif weights is not None:value=value*weights.index_select(0,getattr(model,prefix+'_edge'))
        out=out+_sum(model,prefix,value,size)
    return out.unsqueeze(0)
