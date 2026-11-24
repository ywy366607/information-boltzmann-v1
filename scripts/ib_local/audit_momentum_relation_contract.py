"""Frozen-body audit of additive momentum and residual fit consistency."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts.ib_local.audit_relation_nonlinearity import digest
from scripts.ib_local.causal_relation_program import mechanism
from scripts.ib_local.shared_relation_basis import SharedRelationBasis


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);source_hash=digest(args.checkpoint)
    with gzip.open(args.checkpoint,'rb') as stream:
        saved=torch.load(stream,map_location='cpu',weights_only=False)
    sd,e=saved['model'],saved['engine']
    enc=nn.Linear(sd['encoder.weight'].shape[1],sd['encoder.weight'].shape[0],bias=False)
    enc.load_state_dict({'weight':sd['encoder.weight']})
    linear=lambda name:lambda x:F.linear(x,sd[name+'.weight'],sd.get(name+'.bias'))
    def prior(x):return linear('long_predictor.2')(F.silu(linear('long_predictor.0')(x)))
    memory=SimpleNamespace(**{k.removeprefix('memory.'):v for k,v in sd.items()
                             if k.startswith('memory.') and '.' not in k.removeprefix('memory.')})
    basis=SharedRelationBasis(enc.out_features)
    basis.load_state_dict({k.removeprefix('relation_basis.'):v for k,v in sd.items() if k.startswith('relation_basis.')})
    model=SimpleNamespace(memory=memory,encoder=enc,long_predictor=prior,
        mother_from_boundary=linear('mother_from_boundary'),momentum_to_boundary=linear('momentum_to_boundary'),
        relation_basis=basis,joint_relation_update_schema=1,
        joint_relation_update_revision=saved['config']['joint_relation_update']['implementation_revision'])
    state=e['_program_question_state'];actions=e['_internal_action'];source=e['_internal_source'][None].expand_as(actions)
    momentum=e['_momentum_decision_read'][None].expand(len(actions),-1)
    chosen=int(e['_pending_record']['choice']);target=enc(e['_mother_boundary_target'][None])
    with torch.no_grad():
        actual=mechanism(model,state,source,actions,ridge=e['relation_ridge'],momentum=momentum)
        quiet=mechanism(model,state,source,actions,ridge=e['relation_ridge'])
        delta=model.momentum_to_boundary(momentum)
        delta=delta+model.mother_from_boundary(delta)
        reproduction=(actual['mean']-quiet['mean']-delta).abs().max()
        # Same support/query coordinates with a zero residual target: a
        # separate additive branch still shifts even perfectly known support.
        result=dict(cycle=saved['cycle'],checkpoint_sha256=source_hash,
            chosen=chosen,controlled_current_weight_recomputation=True,
            factual_mse_with_momentum=float((actual['mean'][chosen]-target[0]).square().mean()),
            factual_mse_without_momentum=float((quiet['mean'][chosen]-target[0]).square().mean()),
            all_candidate_mse_with_momentum=(actual['mean']-target).square().mean(-1).tolist(),
            all_candidate_mse_without_momentum=(quiet['mean']-target).square().mean(-1).tolist(),
            additive_shift_energy=float(delta.square().mean()),
            exact_additive_difference_error=float(reproduction),
            support_x_max_difference=float((actual['x']-quiet['x']).abs().max()),
            support_y_max_difference=float((actual['y']-quiet['y']).abs().max()),
            scope='same frozen current weights and actual observed response, not original prequential scores',
            world_calls=0,optimizer_updates=0,source_unchanged=digest(args.checkpoint)==source_hash)
    assert result['source_unchanged'] and reproduction<1e-5
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
