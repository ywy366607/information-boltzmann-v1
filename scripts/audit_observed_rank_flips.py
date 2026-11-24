"""Check actual before/after evidence-induced order changes on saved legal menus."""
import gzip
from itertools import combinations
import json
from pathlib import Path
import shutil
import tempfile
import torch
from scripts.audit_revisit_exchange import alternate, natural, observe, rank_probe, score_port


def measure(folder):
    body=folder/'body'
    meta=json.loads((body/'last.metadata.json').read_text(encoding='utf8'))
    events=folder/'events.jsonl'
    if not events.exists(): events=body/'metrics.jsonl'
    with events.open(encoding='utf8') as f:
        root=next(r for line in f if (r:=json.loads(line)).get('cycle')==meta['cycle'])
    with tempfile.TemporaryDirectory(dir='scratch') as tmp:
        z=Path(tmp)/'body.pt'
        with gzip.open(body/'last.pt.gz','rb') as src,z.open('wb') as dst:shutil.copyfileobj(src,dst,2**20)
        s=torch.load(z,weights_only=False,mmap=True,map_location='cpu')
        result=rank_probe(s,score_port(s),root,folder); del s
    q=set(result['queries']); assert q==set(root['causal_use']['finite_U']['fresh_use']['benefits'])
    result.update(folder=str(folder),source_sha256=meta['checkpoint_sha256'],pairs=[])
    for i,j in combinations(range(len(result['menu'])),2):
        gap=lambda phase,path:result[phase][path][i]-result[phase][path][j]
        raw=[gap(p,'raw') for p in ('pre','post')]; encoded=[gap(p,'encoded') for p in ('pre','post')]
        result['pairs'].append(dict(actions=[result['menu'][i],result['menu'][j]],raw=raw,encoded=encoded,
            resolved=min(map(abs,raw+encoded))>6e-6,raw_flip=raw[0]*raw[1]<0,encoded_flip=encoded[0]*encoded[1]<0))
    return result


if __name__=='__main__':
    torch.set_num_threads(1)
    folders=[alternate,natural,observe,observe.parent/'apply',Path('G:/cgb_mother_return_2609_20261120')]
    for run,arms in [('cgb_competing_action_teachers_2521_20261120',('control','competing')),
                     ('cgb_persistent_rhythm_2537_20261120',('control','coupled','port')),
                     ('cgb_root_native_correction_2553_20261120',('observe','apply')),
                     ('cgb_retention_scope_prior_2569_20261120',('off','relation'))]:
        folders.extend(Path('scratch')/run/arm for arm in arms)
    rows=[]
    for folder in folders:
        r=measure(folder);rows.append(r);print(r['chart'],r['pairs'],flush=True)
        if any(p['resolved'] and p['raw_flip'] for p in r['pairs']): break
    Path('results/published/cgb_observed_rank_flips_20261121.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')
