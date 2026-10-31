"""Cache full DAN trees, real release positions, and reciprocal KC contacts."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from fetch_native_apl_geometry import fetch_tree
from fetch_native_kc_geometry import outgoing, retry
from build_native_kc_terminals import tree


def build(graph_path,kc_geometry,kc_raw,output,pack_name='dan_release_native_v1.npz'):
    output.mkdir(parents=True,exist_ok=True)
    # NpzFile indexing inflates an entire array. Materialize once before the
    # per-contact loops; never re-inflate a 25M-edge array for one edge lookup.
    with np.load(graph_path,allow_pickle=False) as archive:
        graph={k:archive[k] for k in ('neuron_body_ids','dan_indices','dan_edge_pre',
            'dan_edge_post','dan_edge_delay','edge_pre_e','edge_post_e','edge_delay_e')}
    ids=graph['neuron_body_ids']; dan=graph['dan_indices'].astype(np.int64)
    bodies=list(map(int,ids[dan])); records=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs=[pool.submit(retry,fetch_tree,b,output) for b in bodies]
        for f in as_completed(jobs):
            records.append(f.result())
    print(json.dumps({'trees':len(records)}),flush=True)
    # Reuse the same measured SynapsesTo query as KC geometry, without a new
    # arbitrary ROI/cell-type cutoff. Distinct output directory keeps identities.
    batches=[bodies[i:i+16] for i in range(0,len(bodies),16)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(retry,outgoing,b,output,j) for j,b in enumerate(batches)]
        for j,f in enumerate(as_completed(jobs),1):
            records.append(f.result());print(json.dumps({'output_batches':j,'total':len(jobs)}),flush=True)
    outgoing_table=pd.concat([pd.read_feather(output/f'kc_outgoing_{i:03d}.feather') for i in range(len(batches))],ignore_index=True)
    with np.load(kc_geometry,allow_pickle=False) as data:
        kc=data['kc_index']; kc_sites=data['site_ids'];kc_nodes=data['node_ids'];kc_owner=data['owner']
    incoming=[]
    for p in sorted(kc_raw.glob('kc_outgoing_*.feather')):
        t=pd.read_feather(p); t=t[t.body_post.isin(bodies)]
        if len(t):incoming.append(t)
    incoming=pd.concat(incoming,ignore_index=True)
    live={(int(ids[a]),int(ids[b])):i for i,(a,b) in enumerate(zip(graph['dan_edge_pre'],graph['dan_edge_post']))}
    outgoing_table=outgoing_table[[ (int(a),int(b)) in live for a,b in zip(outgoing_table.body_pre,outgoing_table.body_post)]].copy()
    edge=graph['edge_pre_e'].astype(np.int64);post=graph['edge_post_e'].astype(np.int64)
    selected=np.flatnonzero(np.isin(edge,kc)&np.isin(post,dan))
    kc_pairs={(int(ids[edge[e]]),int(ids[post[e]])):int(e) for e in selected}
    incoming=incoming[[(int(a),int(b)) in kc_pairs for a,b in zip(incoming.body_pre,incoming.body_post)]].copy()
    if set(zip(incoming.body_pre,incoming.body_post)) != set(kc_pairs):
        raise ValueError('Missing existing KC-DAN contact coordinates')
    if set(zip(outgoing_table.body_pre,outgoing_table.body_post)) != set(live):
        raise ValueError('Missing existing DAN output coordinates')
    release_groups={b:t for b,t in outgoing_table.groupby('body_pre',sort=False)}
    input_groups={b:t for b,t in incoming.groupby('body_post',sort=False)}
    site_lookup={tuple(map(int,row)):j for j,row in enumerate(kc_sites)}
    arrays={n:[] for n in ('node_ids','owner','volume','u','v','g','site_ids','site_node','site_owner')}
    outputs=[]; inputs=[]; distances=[];node_offset=site_offset=0
    for slot,index in enumerate(dan):
        body=int(ids[index]);identity,xyz,volume,u,v,g=tree(output/f'{body}.swc')
        spatial=cKDTree(xyz)
        arrays['node_ids'].append(np.column_stack((np.full(len(identity),body),identity)))
        arrays['owner'].append(np.full(len(identity),slot));arrays['volume'].append(volume)
        arrays['u'].append(u+node_offset);arrays['v'].append(v+node_offset);arrays['g'].append(g)
        out=release_groups.get(body)
        if out is not None:
            coordinates,inverse=np.unique(out[['x_pre','y_pre','z_pre']].to_numpy(dtype=np.int64),axis=0,return_inverse=True)
            distance,nodes=spatial.query(coordinates*.008);distances.extend(distance.tolist())
            arrays['site_ids'].append(np.column_stack((np.full(len(coordinates),body),coordinates)))
            arrays['site_node'].append(nodes+node_offset);arrays['site_owner'].append(np.full(len(nodes),slot))
            out=out.copy();out['site']=inverse+site_offset
            out['edge']=[live[body,int(target)] for target in out.body_post]
            totals=out.groupby('edge').size()
            grouped=out.groupby(['edge','site']).size().reset_index(name='count')
            for e,site,count in grouped.itertuples(index=False,name=None):
                outputs.append((int(e),int(index),int(graph['dan_edge_post'][e]),int(graph['dan_edge_delay'][e]),int(site),count/totals[e]))
            site_offset+=len(nodes)
        inc=input_groups.get(body)
        if inc is not None:
            inc=inc.copy();distance,nodes=spatial.query(inc[['x_post','y_post','z_post']].to_numpy()*.008)
            distances.extend(distance.tolist());inc['node']=nodes+node_offset
            inc['site']=[site_lookup[(int(b),int(x),int(y),int(z))] for b,x,y,z in inc[['body_pre','x_pre','y_pre','z_pre']].itertuples(index=False,name=None)]
            inc['edge']=[kc_pairs[int(source),body] for source in inc.body_pre]
            totals=inc.groupby('edge').size()
            grouped=inc.groupby(['edge','site','node']).size().reset_index(name='count')
            for e,site,node,count in grouped.itertuples(index=False,name=None):
                inputs.append((int(e),int(edge[e]),int(index),int(graph['edge_delay_e'][e]),int(site),int(node),count/totals[e]))
        node_offset+=len(identity)
        if (slot+1)%64==0:print(json.dumps({'mapped_DANs':slot+1,'nodes':node_offset}),flush=True)
    data={k:np.concatenate(v) for k,v in arrays.items()}
    data['dan_index']=dan
    for prefix,rows,names in [('out',outputs,('edge','pre','post','delay','site','fraction')),
        ('in',inputs,('edge','pre','post','delay','site','node','fraction'))]:
        for column,name in enumerate(names):
            data[prefix+'_'+name]=np.asarray([row[column] for row in rows],dtype=np.float64 if name=='fraction' else np.int64)
    # Preserve BOTH ends of each physical DAN->KC contact. The previous target
    # cache grouped by edge/KC node and therefore cannot recover source sites.
    dan_sites={tuple(map(int,row)):j for j,row in enumerate(data['site_ids'])}
    kc_targets=set(map(int,ids[kc]));receiver=outgoing_table[outgoing_table.body_post.isin(kc_targets)].copy()
    offsets=np.r_[0,np.cumsum(np.bincount(kc_owner,minlength=len(kc)))];groups=receiver.groupby('body_post').indices
    receiver_nodes=np.empty(len(receiver),dtype=np.int64)
    for slot,index in enumerate(kc):
        positions=groups.get(int(ids[index]))
        if positions is None:continue
        identity,xyz,*_=tree(kc_raw/f'{int(ids[index])}.swc')
        loc=np.arange(offsets[slot],offsets[slot+1])
        if not np.array_equal(identity,kc_nodes[loc,1]):raise ValueError('KC recipient tree identity mismatch')
        distance,nodes=cKDTree(xyz).query(receiver.iloc[positions][['x_post','y_post','z_post']].to_numpy()*.008)
        receiver_nodes[positions]=loc[nodes]
    receiver['node']=receiver_nodes
    receiver['site']=[dan_sites[(int(b),int(x),int(y),int(z))] for b,x,y,z in receiver[['body_pre','x_pre','y_pre','z_pre']].itertuples(index=False,name=None)]
    receiver['edge']=[live[int(b),int(t)] for b,t in zip(receiver.body_pre,receiver.body_post)]
    totals=receiver.groupby('edge').size().to_dict()
    groups=receiver.groupby(['edge','site','node'],sort=True).size().reset_index(name='count')
    for name in ('edge','site','node'):data['outkc_'+name]=groups[name].to_numpy(dtype=np.int64)
    data['outkc_delay']=graph['dan_edge_delay'][data['outkc_edge']].astype(np.int64)
    data['outkc_fraction']=np.asarray([count/totals[e] for e,count in zip(groups.edge,groups['count'])])
    path=output/pack_name;np.savez_compressed(path,**data)
    report={'dataset':'male-cns:v1.0','confidence':.5,'neurons':len(dan),'nodes':node_offset,
        'sites':site_offset,'segments':len(data['u']),'KC_DAN_edges':len(kc_pairs),
        'DAN_output_edges':len(live),'input_routes':len(inputs),'output_routes':len(outputs),
        'KC_receiver_routes':len(groups),
        'contact_projection_um_quantiles':np.quantile(distances,[0,.5,.95,1]).tolist(),
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'kc_geometry_sha256':hashlib.sha256(kc_geometry.read_bytes()).hexdigest(),
        'scope':'All supplied DAN morphology nodes; actual KC reciprocal contacts and DAN release positions, original graph edge fractions/delays. No ROI pooling, invented root joins, or identified channel-kinetics claim.',
        'raw_files':[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(output.iterdir()) if p.suffix in ('.swc','.feather')]}
    path.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='raw_files'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--graph',type=Path,default=Path('data/malecns_v1/fly_reservoir_coba_r2.npz'))
    p.add_argument('--kc-geometry',type=Path,required=True)
    p.add_argument('--kc-raw',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pack-name',default='dan_release_native_v1.npz')
    a=p.parse_args();build(a.graph,a.kc_geometry,a.kc_raw,a.output,a.pack_name)
