"""Stream real morphology moments and a cell-pair occurrence panel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import scipy.sparse as sp

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def main(args):
    started=time.perf_counter()
    args.out.mkdir(parents=True,exist_ok=True)
    with np.load(args.types,allow_pickle=True) as z:
        names=z['names'].astype(str)
    index={name:i for i,name in enumerate(names)}
    ann=feather.read_table(args.annotations,columns=['bodyId','type','somaSide','somaLocation']).to_pandas().drop_duplicates('bodyId')
    ann=ann[ann.type.isin(index)]
    ids=np.load(args.graph)['neuron_body_ids']
    ann=ann[ann.bodyId.isin(ids)].copy()
    ann['tc']=[index[t] for t in ann.type]
    counts=ann.groupby('tc').size()
    eligible=counts[counts>=2].index.to_numpy()
    rng=np.random.default_rng(20261021)
    chosen=rng.choice(eligible,size=min(args.panel_types,len(eligible)),replace=False)
    panel=[]
    for tc in chosen:
        cells=np.sort(ann.loc[ann.tc==tc,'bodyId'].to_numpy())
        panel.extend(rng.choice(cells,2,replace=False).tolist())
    panel=np.sort(panel)
    panel_index={int(b):i for i,b in enumerate(panel)}
    meta={int(row.bodyId):(int(row.tc),str(row.somaSide),row.somaLocation) for row in ann.itertuples()}
    profiles={scale:[[],[],[]] for scale in (4,2)}
    records=[]
    source=[]
    for si,path in enumerate(sorted(args.skeletons.glob('skeletons_*.npz'))):
        with zipfile.ZipFile(path) as archive:
            source.append({'name':path.name,'bytes':path.stat().st_size,
                           'members':{f.filename:[f.CRC,f.file_size] for f in archive.infolist()}})
        with np.load(path) as z:
            bodies,counts=z['body_ids'],z['node_count']
            xyz,parent=z['xyz_nm'],z['parent']
        offset=0
        for body,count in zip(bodies,counts):
            count=int(count);body=int(body)
            info=meta.get(body)
            if info is None:
                offset+=count
                continue
            x=xyz[offset:offset+count]
            p=parent[offset:offset+count]
            offset+=count
            child=np.flatnonzero((p>=0)&(p<count))
            if not len(child): continue
            a,b=x[child].astype(float)*.008,x[p[child]].astype(float)*.008
            lengths=np.linalg.norm(a-b,axis=1)
            mid=(a+b)*.5
            total=float(lengths.sum())
            if total<=0: continue
            center=np.sum(mid*lengths[:,None],axis=0)/total
            centered=mid-center
            covariance=(centered.T*lengths)@centered/total
            degree=np.bincount(p[child],minlength=count)
            branch=int(np.count_nonzero(degree>=2))
            soma=np.asarray(info[2],dtype=float)*.008 if info[2] is not None else np.full(3,np.nan)
            if soma.shape!=(3,): soma=np.full(3,np.nan)
            records.append((body,info[0],info[1],soma,center,covariance,total,branch))
            row=panel_index.get(body)
            if row is not None:
                for scale in (4,2):
                    v=np.floor(mid/scale).astype(np.int64)
                    keys=v[:,0]+4096*(v[:,1]+4096*v[:,2])
                    uq,inv=np.unique(keys,return_inverse=True)
                    weight=np.bincount(inv,weights=lengths)
                    profiles[scale][0].append(np.full(len(uq),row,dtype=np.int32))
                    profiles[scale][1].append(uq)
                    profiles[scale][2].append(weight)
        del xyz,parent
        if si%12==0:
            status={'stage':'skeletons','shards':si+1,'cells':len(records),'seconds':time.perf_counter()-started}
            print(json.dumps(status),flush=True)
            (args.out/'progress.json').write_text(json.dumps(status),encoding='utf-8')
    body=np.array([r[0] for r in records],np.int64)
    tc=np.array([r[1] for r in records],np.int32)
    side=np.array([r[2] for r in records])
    np.savez_compressed(args.out/'morphology.npz',body=body,cell_type=tc,side=side,
        soma=np.stack([r[3] for r in records]),centroid=np.stack([r[4] for r in records]),
        covariance=np.stack([r[5] for r in records]),length=np.array([r[6] for r in records]),
        branches=np.array([r[7] for r in records]),types=names)
    present=np.isin(panel,body)
    if not present.all():
        raise RuntimeError('Selected panel skeleton missing; do not silently change the panel')
    for scale,(rows,keys,weights) in profiles.items():
        rows=np.concatenate(rows);keys=np.concatenate(keys);weights=np.concatenate(weights)
        voxels,col=np.unique(keys,return_inverse=True)
        density=sp.coo_matrix((weights,(rows,col)),shape=(len(panel),len(voxels))).tocsr()
        overlap=(density@density.T).tocsr()
        sp.save_npz(args.out/f'cell_overlap_{scale}um.npz',overlap)
        print(json.dumps({'stage':'cell_overlap','scale_um':scale,'voxels':len(voxels),
                          'density_entries':density.nnz,'pair_entries':overlap.nnz}),flush=True)
        del rows,keys,weights,density,overlap,voxels,col
    weights=np.zeros((len(panel),len(panel)),np.int64)
    reader=ipc.open_file(args.weights)
    rows_scanned=0
    for bi in range(reader.num_record_batches):
        batch=reader.get_batch(bi)
        pre=batch.column(0).to_numpy();post=batch.column(1).to_numpy()
        p=np.searchsorted(panel,pre);q=np.searchsorted(panel,post)
        valid=(p<len(panel))&(q<len(panel))
        keep=np.flatnonzero(valid)
        keep=keep[(panel[p[keep]]==pre[keep])&(panel[q[keep]]==post[keep])]
        np.add.at(weights,(p[keep],q[keep]),batch.column(2).to_numpy()[keep])
        rows_scanned+=len(batch)
        if bi%400==0:
            print(json.dumps({'stage':'weights','batch':bi,'rows':rows_scanned}),flush=True)
    types_panel=np.array([meta[int(b)][0] for b in panel],np.int32)
    np.savez_compressed(args.out/'occurrence_panel.npz',body=panel,cell_type=types_panel,weights=weights,
                         side=np.array([meta[int(b)][1] for b in panel]),types=names)
    provenance={'skeleton_manifest_sha256':sha(args.skeletons/'manifest.json'),
        'annotation_sha256':sha(args.annotations),'types_sha256':sha(args.types),
        'weights_sha256':sha(args.weights),'source_shards':source,
        'extraction_source_sha256':sha(Path(__file__))}
    result={'status':'completed','cells':len(records),'types':int(len(np.unique(tc))),
            'valid_soma_cells':int(np.isfinite(np.stack([r[3] for r in records])).all(1).sum()),
            'panel_cells':len(panel),'panel_types':len(chosen),
            'panel_positive_pairs':int(np.count_nonzero(weights)),
            'rows_scanned':rows_scanned,'seconds':time.perf_counter()-started,'provenance':provenance}
    for filename in ('morphology.npz','occurrence_panel.npz','cell_overlap_4um.npz','cell_overlap_2um.npz'):
        result.setdefault('output_sha256',{})[filename]=sha(args.out/filename)
    (args.out/'extraction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (args.out/'progress.json').write_text(json.dumps({k:v for k,v in result.items() if k!='provenance'},indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='provenance'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--skeletons',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--types',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--panel-types',type=int,default=768)
    main(p.parse_args())
