"""Compare actual captured decoding with replay on one real restored draft."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
from scripts.ib_local.scoped_decision_graph import ScopedDecisionGraph
from scripts.ib_local.language_navigation_turns_v3 import DecisionGraphV3
from scripts.ib_local.variable_expression import budget_scope


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gradient-check',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    opts=['--checkpoint',str(args.body/'last.pt.gz'),'--output',str(args.body),
          '--cycles','1','--resume-output','--world-source','text',
          '--table-storage',str(args.output/'table'),'--causal-storage',str(args.output/'causal')]
    initialize=DecisionGraphV3.__init__
    def instrument(self,model,*a,**kw):
        predict=model.predict_observed;captured={};memory=model.memory
        flags=dict(read_credit_enabled=memory.read_credit_enabled,
            consolidation_enabled=memory.consolidation_enabled,
            consolidation_rule=memory.consolidation_rule)
        def capture(*args,**kwargs):
            result=predict(*args,**kwargs);captured['read']=result[-1];return result
        model.predict_observed=capture
        try:initialize(self,model,*a,**kw)
        finally:del model.predict_observed
        self.audit_captured_read=captured.get('read');self.audit_capture_flags=flags
    DecisionGraphV3.__init__=instrument
    try:loaded=prepare_continuation(continuation_parser().parse_args(opts),resume_output=True)
    finally:DecisionGraphV3.__init__=initialize
    engine=loaded.engine;model=engine.model;predict=model.predict_observed
    actual_read=engine.decision.audit_captured_read;series=[]
    start,step=engine.decision.start,engine.decision.step
    def observed_start(state):
        series.append(dict(initial=state.detach().clone(),steps=[]))
        return start(state)
    def observed_step(byte,probability):
        result=step(byte,probability)
        series[-1]['steps'].append(dict(byte=int(byte),logits=result[0].detach().cpu(),
            feature=result[1].detach().cpu(),slots=actual_read.slots.detach().cpu(),
            weights=actual_read.weights.detach().cpu()))
        return result
    engine.decision.start=observed_start;engine.decision.step=observed_step
    # Run every real outer boundary operation (including workset refresh and
    # S/P motion). Stop after actual drafting, before observing the world.
    # Calling _utter directly would omit those operations and test another path.
    utter=engine._utter
    class DraftCaptured(Exception):pass
    def stop_after_draft():
        utter()
        raise DraftCaptured()
    engine._utter=stop_after_draft
    try:engine.step(torch.randn(model.encoder.out_features,32,device=engine.device))
    except DraftCaptured:pass
    finally:engine._utter=utter
    proposal=engine._joint_proposals[0];words=proposal['words']
    inputs=[engine.pending,*words]
    actual=next(s for s in series if [x['byte'] for x in s['steps']]==inputs)
    if args.gradient_check:
        from scripts.ib_local.variable_expression import proposal_score
        parameters=list(model.parameters())
        terms,error=proposal_score(engine,proposal)
        fast_terms=terms.detach().cpu()
        grads=torch.autograd.grad(terms.sum(),parameters,allow_unused=True)
        reference=[None if g is None else g.detach().cpu() for g in grads]
        del grads,terms
        def force_exact(ids,state,lp,**kw):
            result=predict(ids,state,lp,**kw)
            if ids.shape[1]>1:
                logits=result[0].clone();logits[0,0,words[0]]-=100.
                result=(logits,*result[1:])
            return result
        model.predict_observed=force_exact
        try:terms,exact_error=proposal_score(engine,proposal)
        finally:del model.predict_observed
        grads=torch.autograd.grad(terms.sum(),parameters,allow_unused=True)
        maximum=squared=norm_fast=norm_exact=dot=0.;presence=True
        for a,b in zip(reference,grads):
            presence &= (a is None)==(b is None)
            if a is not None and b is not None:
                b=b.detach().cpu()
                maximum=max(maximum,float((a-b).abs().max()))
                squared+=float((a.double()-b.double()).square().sum())
                norm_fast+=float(a.double().square().sum())
                norm_exact+=float(b.double().square().sum())
                dot+=float((a.double()*b.double()).sum())
        result=dict(source_cycle=loaded.cycle,source_sha=loaded.source_sha,
            proposal_bytes=len(words),read_identity_bytes=sum(proposal['handover'][k].numel()*8
                for k in ('read_slots','read_versions')),gradient_presence_equal=presence,
            gradient_max_error=maximum,gradient_relative_l2=(squared/norm_exact)**.5,
            gradient_cosine=dot/(norm_fast*norm_exact)**.5,
            terms_max_error=float((terms.detach().cpu()-fast_terms).abs().max()),
            fast_actual_score_error=error,exact_actual_score_error=exact_error,
            optimizer_updates=0,world_calls=0,main_written=False,
            scope='same real full-boundary draft: recorded-route batch derivative versus exact one-byte derivative')
        (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result),flush=True)
        model._causal_archive.db.close()
        return
    probabilities=[*proposal['probs'],proposal['handover']['final_probability']]
    logps=torch.as_tensor(np.log(np.maximum(np.stack(probabilities),1e-6)),device=engine.device)
    expected=proposal['logp'];results={}
    print(json.dumps(dict(stage='captured',source_cycle=loaded.cycle,bytes=len(words))),flush=True)
    for name in ('grad_path','grad_chain_path','decoded_eager_path'):
        state=actual['initial'].detach().clone();details=[]
        engine.decision._refresh_projections()
        with (torch.enable_grad() if name!='decoded_eager_path' else torch.no_grad()):
            context=(engine.decision._decode_context() if name=='decoded_eager_path' else budget_scope(model,None))
            with context:
                for index,byte in enumerate(inputs):
                    with budget_scope(model,1-index/engine.max_utterance):
                        out=predict(torch.tensor([[byte]],device=engine.device),state,logps[index][None,None])
                    state=out[1] if name=='grad_chain_path' else out[1].detach()
                    reference=actual['steps'][index]
                    logits=out[0][0,0].detach().cpu();feature=out[3][0].detach().cpu()
                    slots=out[-1].slots.detach().cpu();weights=out[-1].weights.detach().cpu()
                    byte_error=(abs(float(logits.log_softmax(-1)[words[index]])-expected[index])
                                if index<len(words) else None)
                    details.append(dict(position=index,byte_error=byte_error,
                        logits_max_error=float((logits-reference['logits']).abs().max()),
                        feature_max_error=float((feature-reference['feature']).abs().max()),
                        same_slots=torch.equal(slots,reference['slots']),
                        weight_error=float((weights-reference['weights']).abs().max()),
                        actual_slots=reference['slots'].tolist(),replay_slots=slots.tolist()))
                    del out
        results[name]=dict(max_byte_error=max(x['byte_error'] or 0 for x in details),
            max_feature_error=max(x['feature_max_error'] for x in details),
            differing_slot_positions=sum(not x['same_slots'] for x in details),
            first_slot_difference=next((x for x in details if not x['same_slots']),None),
            largest_byte_error=max(details,key=lambda x:x['byte_error'] or 0))
        print(name,json.dumps(results[name]),flush=True)
        del state
        torch.cuda.empty_cache()
    result=dict(source_cycle=loaded.cycle,source_sha=loaded.source_sha,proposal_bytes=len(words),
        optimizer_updates=0,world_calls=0,main_written=False,paths=results,
        capture_flags=engine.decision.audit_capture_flags,
        current_flags=dict(read_credit_enabled=model.memory.read_credit_enabled,
            consolidation_enabled=model.memory.consolidation_enabled,
            consolidation_rule=model.memory.consolidation_rule))
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    model._causal_archive.db.close()


if __name__=='__main__':main()
