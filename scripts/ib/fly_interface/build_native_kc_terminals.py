"""Retain actual KC trees, receptor contacts and individual presynaptic sites."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def tree(path):
    swc = np.loadtxt(path)
    identity = swc[:,0].astype(np.int64)
    if len(np.unique(identity)) != len(identity):
        raise ValueError('Duplicate native morphology identity')
    lookup = {int(n):i for i,n in enumerate(identity)}
    child = np.flatnonzero(swc[:,6]>=0)
    parent = np.asarray([lookup[int(n)] for n in swc[child,6]])
    xyz, r = swc[:,2:5]*.008, swc[:,5]*.008
    length = np.linalg.norm(xyz[child]-xyz[parent],axis=1)
    if np.any(length<=0) or np.any(r<=0):
        raise ValueError('Native tree has non-positive segment/radius')
    volume = np.pi*length*(r[child]**2+r[child]*r[parent]+r[parent]**2)/3.
    capacity = np.bincount(np.r_[child,parent],weights=np.r_[volume/2,volume/2],minlength=len(swc))
    empty = capacity==0
    capacity[empty] = 4*np.pi*r[empty]**3/3
    return identity, xyz, capacity, child, parent, np.pi*r[child]*r[parent]/length


def build(graph_path, raw, apl_path, output):
    graph = np.load(graph_path,allow_pickle=False)
    ids = graph['neuron_body_ids']; lookup={int(b):i for i,b in enumerate(ids)}
    ann = pd.read_feather(graph_path.parent/'body-annotations.feather',columns=['bodyId','class','type']).set_index('bodyId').reindex(ids)
    kc = np.flatnonzero(ann['class'].fillna('').to_numpy()=='Kenyon_Cell')
    kc_slot = {int(ids[p]):i for i,p in enumerate(kc)}
    sources = np.r_[np.flatnonzero(ann['type'].fillna('').to_numpy()=='DPM'),np.flatnonzero(ann['type'].fillna('').to_numpy()=='APL')]
    dpm_count = int((ann.iloc[sources]['type']=='DPM').sum())
    if len(sources)!=4 or dpm_count!=2:
        raise ValueError('Require actual native DPM/APL identities')
    ei = graph['edge_pre_e'].astype(np.int64); ej=graph['edge_post_e'].astype(np.int64)
    e_delay=graph['edge_delay_e']
    selected = np.flatnonzero(np.isin(ei,kc))
    comp_edges=selected[(ann['class'].fillna('').to_numpy()=='MBON')[ej[selected]]]
    comp_pre=ei[comp_edges].copy();comp_post=ej[comp_edges].copy()
    pair = {(int(ids[ei[e]]),int(ids[ej[e]])):(int(e),int(ei[e]),int(ej[e]),int(e_delay[e])) for e in selected}
    del ei,ej
    ii=graph['edge_pre_i'].astype(np.int64); ij=graph['edge_post_i'].astype(np.int64)
    i_delay=graph['edge_delay_i']
    selected_i=np.flatnonzero(np.isin(ii,sources)&np.isin(ij,kc))
    ipair={(int(ids[ii[e]]),int(ids[ij[e]])):(int(e),int(ii[e]),int(ij[e]),int(i_delay[e])) for e in selected_i}
    del ii,ij
    batches = json.loads((raw/'kc_bodies.json').read_text())
    batch_by_body = {int(b):i//64 for i,b in enumerate(batches)}
    incoming = pd.concat([pd.read_feather((apl_path if j>=dpm_count else raw)/f'{ids[p]}_outgoing.feather') for j,p in enumerate(sources)],ignore_index=True)
    incoming = incoming[incoming.body_post.isin(kc_slot) & ~incoming.post_in_calyx]
    source_slot = {int(ids[p]):j for j,p in enumerate(sources)}
    apl_trees={int(ids[p]):cKDTree(tree(apl_path/f'{ids[p]}.swc')[1]) for p in sources[dpm_count:]}
    apl_offsets={}
    with np.load(apl_path/'apl_cable_native_v1.npz',allow_pickle=False) as packed:
        apl_node_ids=packed['node_ids']
    for p in sources[dpm_count:]:
        positions=np.flatnonzero(apl_node_ids[:,0]==int(ids[p]))
        if len(positions)!=apl_trees[int(ids[p])].n:
            raise ValueError('APL target-node geometry differs from active cable')
        apl_offsets[int(ids[p])]=int(positions[0])
    parts={k:[] for k in ['node_ids','owner','volume','u','v','g','site_ids','site_node','site_owner']}
    fast=[]; apl=[]; receptor=[]; counts=[]; distances=[]
    node_offset=site_offset=0; current_batch=None; table=None
    for slot,p in enumerate(kc):
        body=int(ids[p]); batch=batch_by_body[body]
        if batch!=current_batch:
            table=pd.read_feather(raw/f'kc_outgoing_{batch:03d}.feather');current_batch=batch
        rows=table[table.body_pre==body].copy()
        rows=rows[rows.body_post.map(lambda b:(body,int(b)) in pair)]
        ident,xyz,volume,u,v,g=tree(raw/f'{body}.swc'); spatial=cKDTree(xyz)
        parts['node_ids'].append(np.column_stack((np.full(len(ident),body),ident)))
        parts['owner'].append(np.full(len(ident),slot));parts['volume'].append(volume)
        parts['u'].append(u+node_offset);parts['v'].append(v+node_offset);parts['g'].append(g)
        sites,inverse=np.unique(rows[['x_pre','y_pre','z_pre']].to_numpy(dtype=np.int64),axis=0,return_inverse=True)
        if not len(sites):
            raise ValueError(f'Native KC has no localized output sites: {body}')
        distance,node=spatial.query(sites*.008);distances.extend(distance.tolist())
        parts['site_ids'].append(np.column_stack((np.full(len(sites),body),sites)))
        parts['site_node'].append(node+node_offset);parts['site_owner'].append(np.full(len(sites),slot))
        rows['site']=inverse+site_offset
        for post,group in rows.groupby('body_post',sort=False):
            edge,pre,target,delay=pair[body,int(post)]
            denom=len(group)
            for site,n in group.groupby('site').size().items():
                fast.append((edge,pre,target,delay,int(site),n/denom))
            if int(post) in apl_trees:
                distance,nodes=apl_trees[int(post)].query(group[['x_post','y_post','z_post']].to_numpy()*.008)
                group=group.copy();group['target_node']=nodes+apl_offsets[int(post)]
                for (site,n),cnt in group.groupby(['site','target_node']).size().items():
                    apl.append((edge,delay,int(site),int(n),cnt/denom))
        receptors=incoming[incoming.body_post==body].copy()
        if len(receptors):
            distance,nodes=spatial.query(receptors[['x_post','y_post','z_post']].to_numpy()*.008)
            distances.extend(distance.tolist());receptors['node']=nodes+node_offset;receptors['pre_node']=-1
            for other in apl_trees:
                mask=receptors.body_pre==other
                if mask.any():
                    _,n=apl_trees[other].query(receptors.loc[mask,['x_pre','y_pre','z_pre']].to_numpy()*.008)
                    receptors.loc[mask,'pre_node']=n+apl_offsets[other]
            for source,group in receptors.groupby('body_pre',sort=False):
                spec=ipair.get((int(source),body))
                if spec is None:continue
                edge,pre,target,delay=spec
                for (node,pre_node),cnt in group.groupby(['node','pre_node']).size().items():
                    receptor.append((source_slot[int(source)],slot,delay,int(node),int(pre_node),cnt/len(group)))
        counts.append({'body':body,'nodes':len(ident),'sites':len(sites),'receptor_contacts':len(receptors),'roots':len(ident)-len(u)})
        node_offset+=len(ident);site_offset+=len(sites)
        if (slot+1)%512==0: print(json.dumps({'built':slot+1,'nodes':node_offset,'sites':site_offset}),flush=True)
    data={k:np.concatenate(v) for k,v in parts.items()}
    data.update(kc_index=kc,source_index=sources,source_body_ids=ids[sources],kc_body_ids=ids[kc],dpm_count=np.array(dpm_count))
    data.update(comp_pre=comp_pre,comp_post=comp_post,comp_edge_kc=np.searchsorted(kc,comp_pre))
    for name,rows,columns in [('fast',fast,('edge','pre','post','delay','site','fraction')),('apl',apl,('edge','delay','site','node','fraction')),('receptor',receptor,('source','kc','delay','node','pre_node','fraction'))]:
        for i,col in enumerate(columns):data[f'{name}_{col}']=np.asarray([r[i] for r in rows],dtype=np.float64 if col=='fraction' else np.int64)
    np.savez_compressed(output,**data)
    manifest={'version':1,'dataset':'male-cns:v1.0','neurons':len(kc),'nodes':node_offset,'sites':site_offset,'segments':len(data['u']),
        'routes':{'fast':len(fast),'apl':len(apl),'receptor':len(receptor)},'tree_counts':counts,
        'projection_um':{'median':float(np.median(distances)),'p99':float(np.quantile(distances,.99)),'max':float(np.max(distances))},
        'npz_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'scope':'every supplied KC tree node; actual presynaptic site and receptor contact positions. Transported states are effective intracellular receptor signals, not a reconstructed axonal action-potential cable or identified molecular receptor map.'}
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='tree_counts'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--graph',type=Path,default=Path('data/malecns_v1/fly_reservoir_coba_r2.npz'))
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--apl',type=Path,default=Path('data/malecns_v1/spatial'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.graph,a.raw,a.apl,a.output)
