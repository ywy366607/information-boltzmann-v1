"""Actual reciprocal KC contacts control local DAN calcium-related release.

Full supplied morphology is retained. The receptor/calcium kinetics are
learnable effective laws, not identified VGCC/Na/K channel reconstruction.
Somatic action-potential drive remains the existing native-cell spike packet;
the former cell-wide KC feedback broadcast is replaced by local chemistry.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch import nn

from .fly_apl_cable import diffuse
from .fly_kc_terminals import CellwiseGeometry, _sum

DAN_RELEASE_VERSION=1
DAN_RELEASE_PARAMETER_NAMES=('dan_local_log_transport','dan_local_log_affinity',
    'dan_local_log_off','dan_local_logit_calcium_decay',
    'dan_local_log_release_rate','dan_local_logit_resource_recovery')
DAN_RELEASE_ID_NAMES=('dan_release_digest',)
DAN_RELEASE_STATE_FIELDS=('dan_local_receptor','dan_local_calcium','dan_local_resources','dan_local_packets')


def initial_dan_parameters(model,index):
    zero=torch.zeros(index.numel(),device=index.device)
    fast=model.dopamine_logit_fast.detach().expand_as(zero).clone()
    slow=model.dopamine_logit_slow.detach().expand_as(zero).clone()
    values=(zero,zero,torch.log(-torch.log(torch.sigmoid(fast))),fast,zero,slow)
    return dict(zip(DAN_RELEASE_PARAMETER_NAMES,values))


def attach_dan_release(model,path,*,saved=None):
    if not getattr(model,'kc_dopamine_enabled',False) or not hasattr(model,'kcd_pre'):
        raise ValueError('Local DAN source needs actual KC release, reciprocal feedback and spatial KC teacher')
    path=Path(path)
    with np.load(path,allow_pickle=False) as archive:data={k:archive[k] for k in archive.files}
    device=model.kct_index.device
    def buf(name,value,dtype=torch.long,persistent=False):
        model.register_buffer(name,torch.as_tensor(value,device=device,dtype=dtype),persistent=persistent)
    digest=np.frombuffer(hashlib.sha256(path.read_bytes()).digest(),dtype=np.uint8).copy()
    if saved is not None and ('dan_release_digest' not in saved or not torch.equal(
        saved['dan_release_digest'].cpu(),torch.from_numpy(digest))):
        raise ValueError('DAN release geometry identity mismatch')
    manifest=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if manifest['kc_geometry_sha256']!=bytes(model.kc_terminal_digest.cpu().tolist()).hex():
        raise ValueError('DAN reciprocal contacts belong to a different KC geometry')
    expected=np.unique(model.dan_edge_pre.cpu().numpy())
    if not np.array_equal(np.sort(data['dan_index']),expected):
        raise ValueError('DAN source cell coverage mismatch')
    if not np.array_equal(model.dan_edge_pre.cpu().numpy()[data['out_edge']],data['out_pre']) or not np.array_equal(
            model.dan_edge_post.cpu().numpy()[data['out_edge']],data['out_post']):
        raise ValueError('DAN output edge identity mismatch')
    if not np.array_equal(data['dan_index'][data['site_owner'][data['out_site']]],data['out_pre']):
        raise ValueError('Release site belongs to a different DAN')
    if not np.array_equal(data['dan_index'][data['owner'][data['in_node']]],data['in_post']):
        raise ValueError('Reciprocal input crosses a DAN cell boundary')
    if not np.array_equal(model.kct_index.cpu().numpy()[model.kct_site_owner.cpu().numpy()[data['in_site']]],data['in_pre']):
        raise ValueError('Reciprocal input uses another KC release site')
    n=model.n_neurons
    keys=model.kcd_pre.cpu().numpy().astype(np.int64)*n+model.kcd_post.cpu().numpy()
    order=np.argsort(keys); query=data['in_pre']*n+data['in_post']; positions=np.searchsorted(keys[order],query)
    if np.any(positions>=len(keys)) or not np.array_equal(keys[order][positions],query):
        raise ValueError('Reciprocal KC-DAN connection missing from existing feedback')
    input_edge=order[positions]
    if len(np.unique(input_edge))!=len(keys):raise ValueError('Unlocated live KC-DAN feedback connection')
    for prefix,edge,count in [('in',input_edge,len(keys)),('out',data['out_edge'],model.dan_edge_pre.numel())]:
        fraction=data[prefix+'_fraction'];total=np.bincount(edge,weights=fraction,minlength=count)
        if not np.isfinite(fraction).all() or np.any(fraction<=0) or not np.allclose(total,1.,rtol=0,atol=1e-7):
            raise ValueError('DAN native contact shares do not conserve original edge totals')
    if not np.array_equal(np.unique(data['outkc_edge']),np.flatnonzero(np.isin(model.dan_edge_post.cpu().numpy(),model.kct_index.cpu().numpy()))):
        raise ValueError('Missing DAN-KC recipient routes')
    post=model.dan_edge_post.cpu().numpy()[data['outkc_edge']]
    if not np.array_equal(model.kct_index.cpu().numpy()[model.kct_owner.cpu().numpy()[data['outkc_node']]],post):
        raise ValueError('DAN-KC recipient node ownership mismatch')
    if not np.array_equal(data['dan_index'][data['site_owner'][data['outkc_site']]],
            model.dan_edge_pre.cpu().numpy()[data['outkc_edge']]):
        raise ValueError('DAN-KC release site ownership mismatch')
    expected=np.flatnonzero(np.isin(model.dan_edge_post.cpu().numpy(),model.kct_index.cpu().numpy()))
    total=np.bincount(data['outkc_edge'],weights=data['outkc_fraction'],minlength=model.dan_edge_pre.numel())
    if not np.allclose(total[expected],1.,rtol=0,atol=1e-7):
        raise ValueError('DAN-KC shares do not conserve original edge totals')
    delays=np.repeat(np.arange(1,5),np.diff([int(v) for v in model.dan_delay_splits]))
    if not np.array_equal(delays[data['out_edge']],data['out_delay']) or not np.array_equal(
            delays[data['outkc_edge']],data['outkc_delay']):
        raise ValueError('DAN local release transmission delay mismatch')
    buf('dan_release_digest',digest,torch.uint8,True)
    for name,key in [('danl_index','dan_index'),('danl_owner','owner'),('danl_site_owner','site_owner'),('danl_site_node','site_node')]:buf(name,data[key])
    total=np.bincount(data['owner'],weights=data['volume'],minlength=len(data['dan_index']))
    buf('danl_density',total[data['owner']]/data['volume'],torch.float32)
    model.danl_geometry=CellwiseGeometry(data['volume'],data['u'],data['v'],data['g'],data['owner'])
    for prefix,target in [('in','node'),('out','post'),('outkc','node')]:
        for lag in range(1,5):
            select=data[prefix+'_delay']==lag; route=f'danl_{prefix}_{lag}'
            names=('site','fraction',target,'edge')
            for name in dict.fromkeys(names):
                value=input_edge[select] if prefix=='in' and name=='edge' else data[prefix+'_'+name][select]
                buf(route+'_'+name,value,torch.float32 if name=='fraction' else torch.long)
            destinations=data[prefix+'_'+target][select]
            order=np.argsort(destinations,kind='stable');unique,count=np.unique(destinations[order],return_counts=True)
            for name,value in [('order',order),('target',unique),('count',count)]:buf(route+'_'+name,value)
    for name,value in initial_dan_parameters(model,model.danl_index).items():
        if saved is not None:
            if name not in saved or saved[name].shape!=value.shape or not torch.isfinite(saved[name]).all():
                raise ValueError('Invalid DAN local physiological parameter: '+name)
            value=saved[name].to(value).clone()
        model.register_parameter(name,nn.Parameter(value.clone()))
    model.dan_local_enabled=model.dan_local_feedback_enabled=model.dan_local_source_enabled=True
    model.dan_release_manifest=manifest


def dan_state(model,current):
    values=[getattr(current,n) for n in DAN_RELEASE_STATE_FIELDS]
    nodes=model.danl_owner.numel();sites=model.danl_site_owner.numel()
    if not values[0].numel():
        if any(v.numel() for v in values):raise ValueError('Incomplete local DAN history')
        index=model.danl_index[model.danl_site_owner]
        values=[current.h.new_zeros(1,nodes),current.h.new_zeros(1,nodes),
                current.h.new_ones(1,sites),torch.cat([p.index_select(1,index) for p in current.ring],0)]
    expected=((1,nodes),(1,nodes),(1,sites),(4,sites))
    if any(v.shape!=s for v,s in zip(values,expected)):raise ValueError('Malformed local DAN history')
    return values


def reciprocal_arrival(model,kc_values):
    packets=kc_values[3];out=packets.new_zeros(model.danl_owner.numel())
    from .fly_native_runtime import owns_route
    if owns_route(model, 'kc_dan'):
        return out.unsqueeze(0)
    for lag in range(1,5):
        prefix=f'danl_in_{lag}';site=getattr(model,prefix+'_site')
        value=packets[lag-1].index_select(0,site)*model.kcd_weight[getattr(model,prefix+'_edge')]*getattr(model,prefix+'_fraction')
        out=out+_sum(model,prefix,value,out.numel())
    if not model.dan_local_feedback_enabled:out=out*0.
    return out.unsqueeze(0)*model.danl_density


def advance_dan_release(model,current,ring,kc_values,*,local_inputs=None):
    receptor,calcium,resources,packets=dan_state(model,current)
    signal=reciprocal_arrival(model,kc_values)
    if local_inputs and 'dan_receptor_input' in local_inputs:
        from .fly_native_runtime import local_input
        signal=signal+local_input(local_inputs,'dan_receptor_input',signal)
    owner=model.danl_owner;site_owner=model.danl_site_owner
    affinity=torch.exp(model.dan_local_log_affinity)[owner].unsqueeze(0)
    off=torch.exp(model.dan_local_log_off)[owner].unsqueeze(0)
    from .fly_biophysical_operators import ligand_binding, resource_recovery
    receptor=ligand_binding(receptor,signal,affinity,off)
    rho=torch.sigmoid(model.dan_local_logit_calcium_decay)[owner].unsqueeze(0)
    calcium=diffuse(rho*calcium+(1-rho)*receptor,model.dan_local_log_transport,model.danl_geometry)
    # Calcium is a normalized positive effective release-activation state, not
    # molar calcium or identified VGCC kinetics. Gain and all clocks learn.
    gain=torch.exp(model.log_kc_release)[model.superclass_id[model.danl_index]]
    local=gain[site_owner].unsqueeze(0)*calcium.index_select(1,model.danl_site_node)
    ap=ring[0].index_select(1,model.danl_index).index_select(1,site_owner)
    rate=torch.exp(model.dan_local_log_release_rate)[site_owner].unsqueeze(0)
    probability=-torch.expm1(-rate*(ap+local))
    released=resources*probability
    recovery=torch.sigmoid(model.dan_local_logit_resource_recovery)[site_owner].unsqueeze(0)
    resources=resource_recovery(resources,released,recovery)
    normalization=(-torch.expm1(-rate)).clamp_min(torch.finfo(rate.dtype).tiny)
    emitted=released/normalization
    return [receptor,calcium,resources,torch.cat((emitted,packets[:3]),0)]


def dan_arrival(model,values,kind='global'):
    packets=values[3]
    if kind=='global':prefix='out';size=model.n_neurons
    elif kind=='kc':prefix='outkc';size=model.kct_owner.numel()
    else:raise ValueError('Unknown local DAN destination')
    result=packets.new_zeros(size)
    for lag in range(1,5):
        route=f'danl_{prefix}_{lag}';site=getattr(model,route+'_site')
        weight=model.dan_edge_weight[getattr(model,route+'_edge')]/model.dan_scale
        value=packets[lag-1].index_select(0,site)*weight*getattr(model,route+'_fraction')
        result=result+_sum(model,route,value,size)
    return result.unsqueeze(0)
