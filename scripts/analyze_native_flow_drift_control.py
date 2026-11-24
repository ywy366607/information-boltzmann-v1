"""Test earlier native flow modes on later changes after removing drift.

No artificial faces: each vector is the real 28-head-pair edge flux in one
layer, observed at actual committed quiet states in the same recording segment.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def median(values):
    a=np.asarray(values,dtype=float)
    return float(np.median(a)) if a.size else None


def main():
    p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    samples=[]
    for segment,directory in enumerate(sorted(a.work.glob('collect_observations*'))):
        for path in sorted(directory.glob('physical_*.npz')):
            with np.load(path,allow_pickle=False) as z:
                for i in np.flatnonzero(z['kind']=='quiet_after'):
                    samples.append((segment,int(z['event'][i]),z['edge_flux'][i].astype(float)))
    # Chronological split, no differences across a segment or the split.
    split=len(samples)//2; cut=samples[split][0:2]
    increments=[]
    for i in range(1,len(samples)):
        if samples[i-1][0] == samples[i][0] and i != split:
            increments.append((i,samples[i][2]-samples[i-1][2]))
    train=np.stack([v for i,v in increments if i<split]); test=np.stack([v for i,v in increments if i>split])
    # Remove each half's own mean increment: compare fluctuations in direction,
    # not a shared linear drift. Test mean is used only to remove that drift.
    rows=[]; rng=np.random.default_rng(241117); rng_relabel=np.random.default_rng(241118)
    edge_pairs=list(zip(*np.triu_indices(8,1)))
    edge_index={pair:i for i,pair in enumerate(edge_pairs)}
    for layer in range(train.shape[1]):
        tr=train[:,layer,:];te=test[:,layer,:]
        tr=tr-tr.mean(0);te=te-te.mean(0)
        _,singular,vh=np.linalg.svd(tr,full_matrices=False)
        energy=np.square(te).sum()
        if energy==0:continue
        score=float(np.square(te@vh[0]).sum()/energy)
        # Independently circularly shift each observed edge's increments in
        # each half. This preserves each edge's empirical values and circular
        # autocorrelation while breaking its alignment with other edges. It
        # does not preserve the physical graph-flow constraints.
        surrogate=[]
        for _ in range(199):
            st=np.column_stack([np.roll(tr[:,j],int(rng.integers(len(tr)))) for j in range(tr.shape[1])])
            se=np.column_stack([np.roll(te[:,j],int(rng.integers(len(te)))) for j in range(te.shape[1])])
            _,_,sv=np.linalg.svd(st,full_matrices=False)
            surrogate.append(float(np.square(se@sv[0]).sum()/np.square(se).sum()))
        # A single head relabeling induces a signed edge permutation in the
        # held-out trajectory. It preserves topology, all edge amplitudes,
        # graph incidence and every within-test temporal relationship.
        relabel=[]
        for _ in range(499):
            permutation=rng_relabel.permutation(8)
            indices=[];signs=[]
            for i,j in edge_pairs:
                u,v=int(permutation[i]),int(permutation[j])
                indices.append(edge_index[(min(u,v),max(u,v))]);signs.append(1 if u<v else -1)
            mapped=te[:,indices]*np.asarray(signs)
            relabel.append(float(np.square(mapped@vh[0]).sum()/np.square(mapped).sum()))
        pvalue=(1+sum(x>=score for x in surrogate))/(len(surrogate)+1)
        p_relabel=(1+sum(x>=score for x in relabel))/(len(relabel)+1)
        rows.append(dict(layer=layer,heldout_fraction=score,isotropic_reference=1/te.shape[-1],
                         train_top_fraction=float(singular[0]**2/np.square(singular).sum()) if singular.any() else None,
                         shifted_edge_null_median=median(surrogate),
                         shifted_edge_null_p95=float(np.quantile(surrogate,.95)),
                         shifted_edge_null_rank_p=float(pvalue),
                         relabel_null_median=median(relabel),relabel_null_p95=float(np.quantile(relabel,.95)),
                         relabel_null_rank_p=float(p_relabel)))
    # Benjamini-Hochberg adjusted values across 32 exploratory layer tests.
    order=sorted(range(len(rows)),key=lambda i: rows[i]['shifted_edge_null_rank_p'])
    running=1.0
    for rank in range(len(order)-1,-1,-1):
        i=order[rank];running=min(running,rows[i]['shifted_edge_null_rank_p']*len(order)/(rank+1))
        rows[i]['shifted_edge_null_bh_q']=running
    order=sorted(range(len(rows)),key=lambda i: rows[i]['relabel_null_rank_p'])
    running=1.0
    for rank in range(len(order)-1,-1,-1):
        i=order[rank];running=min(running,rows[i]['relabel_null_rank_p']*len(order)/(rank+1))
        rows[i]['relabel_null_bh_q']=running
    out=dict(samples=len(samples),split_at=cut,train_increments=len(train),test_increments=len(test),
        layers=rows,median_heldout_fraction=median([r['heldout_fraction'] for r in rows]),
        surrogate_method='199 independent per-edge circular shifts within each train/test half, RNG seed 241117; destroys simultaneous edge relationships, does not preserve native flow constraints',
        relabel_method='499 randomly sampled signed head permutations per layer applied consistently to held-out trajectory; preserves incidence, simultaneous flow and time ordering but permutes learned head identities and frequencies',
        note='Within-segment consecutive quiet calls can contain real external input or learning. Isotropic reference is geometric; samples are temporally dependent and circular shifts also wrap across segment boundaries. BH results are exploratory and not causal.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out))


if __name__=='__main__':main()
