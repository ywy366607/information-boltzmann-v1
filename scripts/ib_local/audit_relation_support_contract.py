"""Inspect the actual prediction/organization evidence contract, read-only."""
from __future__ import annotations

import argparse
import gc
import gzip
import json
from pathlib import Path
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts.ib_local.audit_relation_nonlinearity import digest
from scripts.ib_local.causal_relation_program import mechanism,Port,real_evidence
from scripts.ib_local.generative_relations import dataset
from scripts.ib_local.shared_relation_basis import SharedRelationBasis


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    source_hash=digest(args.checkpoint)
    with gzip.open(args.checkpoint,'rb') as stream:
        saved=torch.load(stream,map_location='cpu',weights_only=False)
    sd,engine=saved['model'],saved['engine']
    state=engine['_program_question_state'];cycle=saved['cycle']
    manifest=saved['config']['causal_archive']
    memory=SimpleNamespace(**{k.removeprefix('memory.'):v for k,v in sd.items()
                             if k.startswith('memory.') and '.' not in k.removeprefix('memory.')})
    encoder=nn.Linear(sd['encoder.weight'].shape[1],sd['encoder.weight'].shape[0],bias=False)
    encoder.load_state_dict({'weight':sd['encoder.weight']})
    linear=lambda name:lambda x:F.linear(x,sd[name+'.weight'],sd.get(name+'.bias'))
    def prior(x):
        return linear('long_predictor.2')(F.silu(linear('long_predictor.0')(x)))
    basis=SharedRelationBasis(encoder.out_features)
    basis.load_state_dict({k.removeprefix('relation_basis.'):v for k,v in sd.items() if k.startswith('relation_basis.')})
    model=SimpleNamespace(memory=memory,encoder=encoder,long_predictor=prior,
        mother_from_boundary=linear('mother_from_boundary'),relation_basis=basis,
        joint_relation_update_schema=1,causal_relation_ridge=engine['relation_ridge'],
        joint_relation_update_revision=saved['config']['joint_relation_update'].get('implementation_revision',1))
    # Preserve the small current support tensors, never construct another body,
    # fast table, optimizer, world or archive working database.
    s,a=engine['_internal_source'][None],engine['_internal_action']
    with torch.no_grad():
        prediction=mechanism(model,state,s.expand_as(a),a,ridge=model.causal_relation_ridge)
        ox,oy,_,refs=dataset(model,state.organization)
        px,py=prediction['x'],prediction['y']
        pc=sorted([tuple(v) for v in px.double().tolist()])
        oc=sorted([tuple(v) for v in ox.double().tolist()])
        identical=len(pc)==len(oc) and bool(torch.allclose(torch.tensor(pc),torch.tensor(oc),atol=1e-6,rtol=1e-6))
    path=Path(manifest['snapshot']).resolve();archive_hash=digest(path)
    db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    hot=json.loads(db.execute("SELECT value FROM meta WHERE key='hot'").fetchone()[0])
    support_ids=sorted({int(e.fact_id.split(':')[1]) for e in state.constraints if e.kind==Port.MOTHER and e.fact_id})
    organization_ids=sorted({int(ref.split(':')[1]) if isinstance(ref,str) else hot[str(ref[0])]['id']
        for ref in refs if ref is not None and (isinstance(ref,str) or str(ref[0]) in hot)})
    old_bound_y=py.detach().clone()
    orgslots=real_evidence(model,state.organization,Port.MOTHER)
    # An intervention only on this detached read-only model: demonstrate
    # whether organization consumes outcomes the predictor does not consume.
    chosen=None
    for slot in orgslots.tolist():
        if hot[str(slot)]['id'] not in support_ids:
            chosen=slot;break
    effect=None
    if chosen is not None:
        old=memory.mother_next[chosen].clone()
        memory.mother_next[chosen].add_(.1)
        with torch.no_grad():
            altered=mechanism(model,state,s.expand_as(a),a,ridge=model.causal_relation_ridge)
            _,changed_y,_,_=dataset(model,state.organization)
        memory.mother_next[chosen].copy_(old)
        effect=dict(intervened_episode=hot[str(chosen)]['id'],
            prediction_residual_max_change=float((altered['y']-old_bound_y).abs().max()),
            organization_residual_max_change=float((changed_y-oy).abs().max()),
            prediction_mean_max_change=float((altered['mean']-prediction['mean']).abs().max()))
    shared_effect=None
    if model.joint_relation_update_revision>=3:
        fact=next(e for e in state.constraints if e.kind==Port.MOTHER)
        change=lambda fs:tuple(replace(e,outcome=e.outcome+.1) if e.kind==Port.MOTHER and
                              e.fact_id==fact.fact_id else e for e in fs)
        altered_state=replace(state,constraints=change(state.constraints),
            organization=replace(state.organization,constraints=change(state.organization.constraints)))
        with torch.no_grad():
            altered=mechanism(model,altered_state,s.expand_as(a),a,ridge=model.causal_relation_ridge)
            _,changed_y,*_=dataset(model,altered_state.organization)
        dy=altered['y']-py;doy=changed_y-oy
        shared_effect=dict(fact_id=fact.fact_id,prediction_residual_max_change=float(dy.abs().max()),
            organization_residual_max_change=float(doy.abs().max()),
            intervention_difference_max=float((dy-doy).abs().max()))
    db.close()
    report=dict(cycle=cycle,checkpoint_sha256=source_hash,
        prediction_facts=support_ids,organization_facts=organization_ids,
        intersection=len(set(support_ids)&set(organization_ids)),
        prediction_x_shape=list(px.shape),organization_x_shape=list(ox.shape),
        same_feature_evidence=identical,read_only_support_intervention=effect,
        shared_fact_intervention=shared_effect,
        source_unchanged=digest(args.checkpoint)==source_hash,
        archive_unchanged=digest(path)==archive_hash,world_calls=0,optimizer_updates=0)
    assert report['source_unchanged'] and report['archive_unchanged']
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
