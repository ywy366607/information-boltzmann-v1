"""Equal-budget release exchanges on 2594→2605 and a scoped U commuting-square probe."""
from dataclasses import replace
import gzip
from itertools import combinations
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from scripts.ib_local import epistemic_value_reference as ev, finite_use_value as finite
from scripts.ib_local.causal_relation_program import Evidence, Port, mechanism, log_score, revise
from scripts.ib_local.action_value_cross_score import fact_digest

natural = Path('scratch/cgb_wood_policy_use_16rounds_20261120/natural/run')
observe = Path('G:/cgb_arrived_actor_2601_20261120/observe')
alternate = Path('G:/cgb_active_evidence_2594_20261120')


def row(folder, cycle):
    with (folder/'events.jsonl').open(encoding='utf8') as f:
        return next(r for line in f if (r:=json.loads(line)).get('cycle') == cycle)


def fact(folder, key):
    meta = json.loads((folder/'body/last.metadata.json').read_text(encoding='utf8'))
    path = Path(meta['config']['causal_archive']['snapshot'])
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True) as db:
        schema = json.loads(db.execute("select value from meta where key='schema'").fetchone()[0])
        blob, = db.execute('select payload from episodes where id=?',(int(key.split(':')[1]),)).fetchone()
    offset, fields = 0, {}
    for k in sorted(schema):
        spec = schema[k]; n = int(np.prod(spec['shape'])); dtype = np.dtype(spec['dtype'])
        fields[k] = np.frombuffer(blob, dtype, n, offset); offset += n*dtype.itemsize
    assert fields['count'].item() == 1 and fields['mother_valid'].item()
    return Evidence(*(torch.from_numpy(fields[k].copy())[None] for k in ('source','action','mother_next')),
                    Port.MOTHER, True, stamp=int(fields['last_write'].item()), fact_id=key)


@torch.no_grad()
def score_port(saved):
    capsule = torch.load('scratch/cgb_action_cross_score_2587_20261120_silence/second_settlement.pt',weights_only=False,map_location='cpu')
    m = capsule['model']; w = saved['model']; eng = saved['engine']; ridge = eng['relation_ridge']
    m.load_state_dict({k:w[k] for k in m.state_dict()}); m.causal_relation_ridge = ridge
    m.causal_relations = SimpleNamespace(**{k:w['causal_relations.'+k] for k in ('directions','observation_directions')})
    schema = saved['config']['epistemic_value']['schema']
    e = SimpleNamespace(model=m,relation_ridge=ridge,epistemic_value_schema=schema)
    m._epistemic_value_engine=e
    m._preview_batch_enabled=False  # independent serial scoring on both paths
    weight=w['momentum_to_boundary.weight']; bias=w.get('momentum_to_boundary.bias')
    m.momentum_to_boundary=torch.nn.Linear(weight.shape[1],weight.shape[0],bias=bias is not None)
    m.momentum_to_boundary.load_state_dict({'weight':weight,**({'bias':bias} if bias is not None else {})})
    return e


@torch.no_grad()
def audit(saved):
    e=score_port(saved); m=e.model; ridge=e.relation_ridge; eng=saved['engine']
    src, alt, later = row(natural,2594), row(alternate,2594), row(observe,2605)
    assert src['shared_unknown']['signature'] == alt['shared_unknown']['signature'] == later['shared_unknown']['signature']
    ids = list(src['causal_use']['finite_U']['fresh_use']['benefits']); assert len(ids)==8
    old = [fact(natural,k) for k in ids]; target = fact(observe,later['causal_program']['committed_episode_identity'])
    acquisitions = {a:fact(folder,r['causal_program']['committed_episode_identity']) for a,folder,r in [('reuse',natural,src),('generate',alternate,alt)]}
    assert fact_digest(acquisitions['reuse'])!=fact_digest(acquisitions['generate'])
    chosen = set(ids)-set(src['causal_use']['finite_U']['supports']); assert len(chosen)==1
    state = finite.score_state(eng['internal_pending']['causal_program']['state'])
    def support(fs): return replace(state, constraints=tuple(fs),extra=(),evidence_bound=True,organization=None,variable_uses=())
    def value(fs):
        d=mechanism(m,support(fs),target.source,target.action,ridge=ridge)
        return float(log_score(d,m.encoder(target.outcome)).mean())
    ranks = {a:{k:value([f for f in old if f.fact_id!=k]+[incoming]) for k in ids} for a,incoming in acquisitions.items()}
    flips = [(i,j) for i,j in combinations(ids,2) if (ranks['reuse'][i]-ranks['reuse'][j])*(ranks['generate'][i]-ranks['generate'][j]) < 0]
    # Independently recorded pre-world chart at the actual revisit, not the same weights as 2601.
    actual = later['causal_use']['finite_U']['fresh_use']; fs=[f for f in old if f.fact_id not in chosen]+[acquisitions['reuse']]
    assert set(actual['benefits'])=={f.fact_id for f in fs}
    full=value(fs); marg={f.fact_id:full-value([x for x in fs if x is not f]) for f in fs}
    exchange=dict(chart=2601,acquire=2594,revisit=2605,original_release=next(iter(chosen)),scores=ranks,
        flips=flips,actual_revisit_benefits=actual['benefits'],early_chart_benefits=marg)
    # Same legal 2601 menu, common frozen settlement chart, before/after one actual observation.
    return dict(exchange=exchange,rank_probe=rank_probe(saved,e,row(natural,2601),natural),updates=0,world_calls=0)


@torch.no_grad()
def rank_probe(saved,e,root,folder):
    eng=saved['engine']; m=e.model; ridge=e.relation_ridge
    state=finite.score_state(eng['internal_pending']['causal_program']['state'])
    legal=[i for i,p in enumerate(root['choice_probability']) if p>0]
    actions=eng['_internal_action'][legal]; source=eng['_internal_source'].reshape(1,-1).expand_as(actions)
    before=state; observed=fact(folder,root['causal_program']['committed_episode_identity'])
    from scripts.ib_local.relation_evidence import mechanism_evidence
    mom=eng['internal_pending']['momentum_read'].reshape(1,-1).expand(len(actions),-1)
    qs=tuple(mechanism_evidence(m,before,Port.MOTHER)); ctx=ev.from_body(e,before,qs,momentum=mom)
    assert observed.fact_id not in {q.fact_id for q in qs} and len(qs)==8
    after=revise(before,observed,model=m); after=replace(after,constraints=after.constraints+after.extra,extra=())
    next_ctx=ev.revise(m,ctx,ev.encode_fact(m,observed),ridge=ridge)
    out={}
    for name,st,k in [('pre',before,ctx),('post',after,next_ctx)]:
        raw=finite.expected(e,st,source,actions,mechanism(m,st,source,actions,ridge=ridge,momentum=mom),qs)
        encoded=ev.expected(m,k,m.encoder(source),m.encoder(actions),ridge=ridge)
        out[name]=dict(raw=raw.tolist(),encoded=encoded.tolist(),same_support={f.identity for f in k.facts}=={q.fact_id for q in mechanism_evidence(m,st,Port.MOTHER)})
    return dict(chart=saved['cycle'],menu=legal,queries=[q.fact_id for q in qs],evidence=observed.fact_id,**out)


if __name__=='__main__':
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(dir='G:/') as tmp:
        z=Path(tmp)/'body.pt'
        with gzip.open(natural/'body/last.pt.gz','rb') as src,z.open('wb') as dst: shutil.copyfileobj(src,dst,2**20)
        s=torch.load(z,weights_only=False,mmap=True,map_location='cpu'); assert s['cycle']==2601
        r=audit(s);del s
    Path('results/published/cgb_revisit_exchange_2605_20261121.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf8')
    print(json.dumps(r))
