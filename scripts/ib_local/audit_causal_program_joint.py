"""Check the composed causal program on a real continuing body, without training."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from scripts.ib_local.causal_relation_program import Evidence, Port, compose, log_score, mechanism, revise
from scripts.ib_local.language_action import peek
from scripts.ib_local.navigation_causal_program import migrate_causal_program
from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
from scripts.ib_local.run_causal_water_reply import build, sha256_file


def fingerprint(model):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if torch.is_tensor(tensor):
            digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def audit(checkpoint, device='cpu'):
    torch.set_num_threads(1)
    before_sha=sha256_file(checkpoint)
    engine,*_=build(checkpoint,device=device)
    migrate_causal_program(engine)
    model,m=engine.model,engine.model.memory
    with torch.no_grad():
        model._question_context=None
        engine.decision.start(engine.recurrent)
        _,h=engine.decision.step(engine.pending,peek(engine.table,engine.pending))
        source=h.reshape(1,-1)
        state=model.causal_relations.bind(model,source)
        available=torch.nonzero(state.valid).flatten()
        if len(available)<2: raise ValueError('real relation needs two actual facts for mismatch control')
        j=int(available[0]); slot=int(state.slots[j]); other=int(state.slots[int(available[-1])])
        mask=state.valid.clone(); mask[j]=False
        without=replace(state,valid=mask)
        action=source+m.action[slot:slot+1]-m.source[slot:slot+1]
        actual=Evidence(m.source[slot:slot+1],m.action[slot:slot+1],m.mother_next[slot:slot+1],
                        Port.MOTHER,True,slot=slot,stamp=int(m.last_write[slot]))
        wrong=replace(actual,outcome=m.mother_next[other:other+1])
        restored=revise(without,actual); mismatched=revise(without,wrong)
        if bool(m.internal_valid[slot]) and int(m.internal_stamp[slot])==int(m.last_write[slot]):
            decision_fact=Evidence(m.source[slot:slot+1],m.action[slot:slot+1],m.internal_next[slot:slot+1],
                Port.DECISION,True,slot=slot,stamp=int(m.last_write[slot]),mother=m.mother_next[slot:slot+1])
            restored=revise(restored,decision_fact)
            mismatched=revise(mismatched,decision_fact)
        p=decision_momentum_read(model,engine.bfss_momenta,source)
        kwargs=dict(ridge=engine.relation_ridge,reward_valid=engine.reward_label_valid(wood=True),momentum=p)
        before_model=fingerprint(model); rng=torch.get_rng_state().clone()
        original=compose(model,state,source,action,**kwargs)
        removed=compose(model,without,source,action,**kwargs)
        recovered=compose(model,restored,source,action,**kwargs)
        corrupt=compose(model,mismatched,source,action,**kwargs)
        no_p=compose(model,state,source,action,**dict(kwargs,momentum=torch.zeros_like(p)))
        gap=lambda a,b: float((a-b).norm())
        errors={port:gap(original[port]['mean'],recovered[port]['mean']) for port in ('mother','decision')}
        if max(errors.values())>2e-5: raise AssertionError(('real/shadow U restoration failed',errors))
        if before_model!=fingerprint(model) or not torch.equal(rng,torch.get_rng_state()):
            raise AssertionError('counterfactual altered live model or RNG')
        effects=dict(
            evidence_mother=gap(original['mother']['mean'],removed['mother']['mean']),
            evidence_decision=gap(original['decision']['mean'],removed['decision']['mean']),
            evidence_future=gap(original['value'],removed['value']),
            mismatch_mother=gap(original['mother']['mean'],corrupt['mother']['mean']),
            mismatch_decision=gap(original['decision']['mean'],corrupt['decision']['mean']),
            mismatch_future=gap(original['value'],corrupt['value']),
            momentum_mother=gap(original['mother']['mean'],no_p['mother']['mean']),
            momentum_future=gap(original['value'],no_p['value']))
    # The old fact is re-encoded NOW and its current prediction derivative
    # reaches shared mechanisms; no historical computation graph is restored.
    d=mechanism(model,without,m.source[slot:slot+1],m.action[slot:slot+1],momentum=p,ridge=engine.relation_ridge)
    loss=-log_score(d,model.encoder(m.mother_next[slot:slot+1])).mean()
    names=('encoder.weight','long_predictor.0.weight','mother_from_boundary.weight','momentum_to_boundary.weight')
    params=dict(model.named_parameters())
    grads=torch.autograd.grad(loss,[params[name] for name in names],allow_unused=True)
    gradients={name:0. if g is None else float(g.norm()) for name,g in zip(names,grads)}
    if not all(torch.isfinite(g).all() for g in grads if g is not None):
        raise AssertionError('nonfinite shared mechanism derivatives')
    learned={id(p) for group in engine.optimizer.param_groups for p in group['params']}
    missing=[name for name,p in model.named_parameters() if p.requires_grad and id(p) not in learned]
    if missing: raise AssertionError(('unregistered trainable parameters',missing))
    assert sha256_file(checkpoint)==before_sha
    return dict(checkpoint=str(checkpoint),source_sha256=before_sha,source_unchanged=True,
        relation_identity=state.identity,evidence_slot=slot,evidence_stamp=int(m.last_write[slot]),
        evidence_age_bytes=int(m.clock)-int(m.last_write[slot]),
        restore_errors=errors,effects=effects,gradients=gradients,
        bank_bytes=sum(b.numel()*b.element_size() for b in model.causal_relations.buffers()),
        parameter_count=sum(p.numel() for p in model.parameters()),
        trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        missing_optimizer_parameters=missing,
        scope='same current query; versioned real-evidence removal/recovery/mismatch through composed body',
        claim='joint interface and current derivatives; independent benefit requires real continuation')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    result=audit(args.checkpoint,args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__': main()
