"""Audit all old-read lineages and finite utility versus retention credit.

Real heldout text, frozen body, exact record-version identities. Shadow credit
settlement uses the production routine on an isolated memory copy only.
"""
import argparse
import copy
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path

import torch

from scripts.ib_local.audit_pretrained_gdn_null_effect import sha256
from scripts.ib_local.experience_consolidation import bounded_preference
from scripts.ib_local.language_navigation_kanli_v4l import live_sigreg
from scripts.ib_local.navigation_read_credit import ReadParticipation
from scripts.ib_local.pretrained_gdn import load_pure_gdn, enable_streaming_convolution, clone_cache
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner, language_loss
from scripts.ib_local.probe_pretrained_gdn_retained_values import retained_value_probe
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same, write_json


def lineages(pair):
    result = {}
    for arm, data in pair['arms'].items():
        previous, rows = {}, []
        for step in data['steps']:
            for event in step['admissions']:
                for read in event['reads']:
                    if read['age_tokens'] <= 128 or read['signed_read_parameter_credit'] <= 1e-7:
                        continue
                    identity = read['identity']
                    old = previous.get(identity)
                    row = dict(step=step['step'], **read)
                    if old is not None:
                        row.update(previous_positive_step=old,
                                   gap_tokens=(step['step']-old)*128)
                    rows.append(row)
                    previous[identity] = step['step']
        repeated = [r for r in rows if 'gap_tokens' in r]
        result[arm] = dict(positive_old_reads=len(rows), distinct_identities=len(previous),
            repeated_positive_reads=len(repeated),
            repeated_beyond_window=sum(r['gap_tokens'] > 128 for r in repeated),
            maximum_positive_reuse_gap_tokens=max((r['gap_tokens'] for r in repeated), default=0),
            maximum_positive_read_age_tokens=max((r['age_tokens'] for r in rows), default=0),
            records=rows)
    return result


@contextmanager
def receipt_batch(learner):
    saved = {}
    try:
        for layer, wrapper in learner.memories.items():
            m = wrapper.memory
            saved[layer] = (m._read_participation, m._read_credit_active, m._read_credit_entries)
            if m._read_credit_active or m._read_participation:
                raise RuntimeError('finish live credit before probing')
            m.begin_read_credit()
        yield
    finally:
        for layer, values in saved.items():
            m = learner.memories[layer].memory
            m._read_participation, m._read_credit_active, m._read_credit_entries = values


def forward_objective(learner, x, target, cache):
    with learner.observation(), ExitStack() as stack:
        for layer, wrapper in learner.memories.items():
            stack.enter_context(wrapper.attach(learner.model, layer=layer))
        output = learner.model(x, past_key_values=clone_cache(cache), use_cache=True)
        _, ce = language_loss(output.logits, target)
        regs, reads = [], {}
        for layer, w in learner.memories.items():
            m = w.memory
            regs.append(live_sigreg(w.encoder(w.current['source'][0].float().cpu()),
                w.encoder(m.history), m.history_valid, learner.directions))
            reads[layer] = w.last_read
        reg = torch.stack(regs).mean()
        return output.logits, 10*ce+.001*reg.to(ce.device), ce, output.past_key_values, reads


def shadow_settle(memory, receipt, gradient):
    # Copy buffers, not the live autograd receipt; no persistent history added.
    saved = memory._read_participation
    memory._read_participation = []
    try:
        shadow = copy.deepcopy(memory)
    finally:
        memory._read_participation = saved
    bias = receipt.bias.detach().clone().requires_grad_()
    bias.grad = gradient.detach().clone()
    shadow._read_participation = [ReadParticipation(bias, receipt.slots.clone(),
        receipt.versions.clone(), receipt.valid.clone())]
    before = bounded_preference(shadow).clone()
    shadow.settle_read_credit()
    return before, bounded_preference(shadow)


def summary(rows):
    useful = [r for r in rows if r['drop_minus_full_objective'] > 1e-5]
    harmful = [r for r in rows if r['drop_minus_full_objective'] < -1e-5]
    return dict(records=len(rows), old_records=sum(r['age_tokens'] > 128 for r in rows),
        beneficial=len(useful), harmful=len(harmful),
        beneficial_negative_logit_credit=sum(r['logit_credit'] < -1e-7 for r in useful),
        harmful_positive_logit_credit=sum(r['logit_credit'] > 1e-7 for r in harmful),
        beneficial_positive_value_credit=sum(r['value_credit'] > 1e-7 for r in useful),
        beneficial_protection_increases=sum(r['preference_after'] > r['preference_before']+1e-7 for r in useful),
        beneficial_and_logit_positive_and_protection_increases=sum(
            r['logit_credit'] > 1e-7 and r['preference_after'] > r['preference_before']+1e-7 for r in useful),
        objective_sign_agreement_logit=sum(r['drop_minus_full_objective']*r['logit_credit'] > 0 for r in useful+harmful),
        objective_sign_agreement_value=sum(r['drop_minus_full_objective']*r['value_credit'] > 0 for r in useful+harmful))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model','checkpoint','curriculum','pair','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--published',type=Path)
    p.add_argument('--offset',type=int,default=512)
    p.add_argument('--tokens',type=int,default=512)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prior evidence')
    if args.tokens < 128 or args.tokens % 128 or args.offset < 0:
        p.error('positive 128-aligned budget and nonnegative offset required')
    torch.set_num_threads(1)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    state=payload['learner']
    curriculum=torch.load(args.curriculum,map_location='cpu',weights_only=False)
    if any(args.offset+args.tokens+1 > len(v) for v in curriculum['heldouts'].values()):
        raise ValueError('insufficient real heldout continuation')
    pair=json.loads(args.pair.read_text(encoding='utf-8'))
    report=dict(status='running',checkpoint=str(args.checkpoint),checkpoint_sha256=sha256(args.checkpoint),
        source_updates=state['updates'],dataset_revision=curriculum['dataset_revision'],
        protocol=dict(offset=args.offset,tokens_per_source=args.tokens,window=128,
            loss='actual 10*CE+.001*SIGReg; CE also reported separately',
            ablation='remove one actual weighted contribution; no renormalization, same complete incoming body/cache',
            value_credit='-weight*dLoss/dWeight with other mixture coefficients held fixed; diagnostic only',
            logit_credit='-dLoss/dAdditiveReadLogit, the actual production receipt',
            settlement='isolated memory-copy production settlement; real body never updated',
            cache='advance frozen unmodified trajectory only; each source starts from saved cache'),
        optimizer_updates=0,episode_writes=0,lineages=lineages(pair),records=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,report)
    model,_=load_pure_gdn(args.model)
    enable_streaming_convolution(model)
    learner=PretrainedExperienceLearner(model,layers=state['layers'],capacity=state['capacity'])
    learner.load_state_dict(state)
    before=learner.state_dict()
    try:
        for source,ids in curriculum['heldouts'].items():
            cache=clone_cache(learner.cache)
            for start in range(args.offset,args.offset+args.tokens,128):
                x,y=ids[start:start+128][None].cuda(),ids[start+1:start+129][None].cuda()
                with torch.no_grad():
                    natural,_=learner.predict(x,cache)
                with receipt_batch(learner):
                    logits,loss,ce,next_cache,reads=forward_objective(learner,x,y,cache)
                    if not torch.equal(logits,natural):
                        raise AssertionError('read-credit observation changed forward')
                    layers=list(learner.memories)
                    receipts=[learner.memories[i].memory._read_participation[0] for i in layers]
                    if any(len(learner.memories[i].memory._read_participation)!=1 for i in layers):
                        raise AssertionError('expected one actual read per layer')
                    tensors=[r.bias for r in receipts]+[reads[i].weights for i in layers]
                    ce_grad=torch.autograd.grad(ce,tensors,retain_graph=True)
                    gradients=torch.autograd.grad(loss,tensors)
                    full_loss,full_ce=float(loss.detach()),float(ce.detach())
                    for j,layer in enumerate(layers):
                        w=learner.memories[layer]; m=w.memory; read=reads[layer]
                        prior,after=shadow_settle(m,receipts[j],gradients[j])
                        for k,slot in enumerate(read.slots[0].tolist()):
                            version=int(m.read_credit_version[slot])
                            with torch.no_grad(), retained_value_probe(w,slot,version,'drop'):
                                _,drop_loss,drop_ce,_,_=forward_objective(learner,x,y,cache)
                            weight=float(read.weights[0,k].detach())
                            report['records'].append(dict(source=source,start=start,layer=layer,
                                slot=slot,version=version,identity=f'{layer}:{slot}:{version}',
                                age_tokens=int(m.clock-m.birth[slot]),weight=weight,
                                full_ce=full_ce,drop_minus_full_ce=float(drop_ce)-full_ce,
                                drop_minus_full_objective=float(drop_loss)-full_loss,
                                logit_credit=-float(gradients[j][0,k]),
                                value_credit=-weight*float(gradients[len(layers)+j][0,k]),
                                ce_logit_credit=-float(ce_grad[j][0,k]),
                                ce_value_credit=-weight*float(ce_grad[len(layers)+j][0,k]),
                                preference_before=float(prior[slot]),preference_after=float(after[slot])))
                cache=clone_cache(next_cache)
                del logits,loss,ce,next_cache,reads,gradients,ce_grad,tensors,receipts,natural
                assert_same(before,learner.state_dict(),'frozen_complete_body')
                write_json(args.output,report)
                print(source,start,summary(report['records']),flush=True)
        report.update(status='complete',summary=summary(report['records']),
            persistent_state_unchanged=True,source_checkpoint_unchanged=sha256(args.checkpoint)==report['checkpoint_sha256'])
        if not report['source_checkpoint_unchanged']:
            raise AssertionError('source checkpoint changed')
        write_json(args.output,report)
        if args.published:
            write_json(args.published,report)
        print('COMPLETE',report['summary'],flush=True)
    except Exception as error:
        report.update(status='failed',error=repr(error)); write_json(args.output,report)
        raise


if __name__=='__main__':
    main()
