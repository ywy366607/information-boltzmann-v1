"""Frozen real-text gradient/actual-Adam decomposition of old local credit.

Full and held have identical predictions. Their gradient difference isolates
the operational contribution of reexecuted native computations, including FP16
accumulation. Shadow Adam steps use the actual optimizer, clipping and moments;
no observation, experience, native weight or update counter is committed.
"""
import argparse
from contextlib import ExitStack
from pathlib import Path

import torch

from scripts.ib_local.audit_pretrained_gdn_null_effect import sha256
from scripts.ib_local.pretrained_gdn import load_pure_gdn, enable_streaming_convolution, clone_cache
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner, language_loss
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same, write_json


def norm(values):
    return sum(float(x.double().square().sum()) for x in values)**.5


def difference(left,right):
    return [a-b for a,b in zip(left,right)]


def relation(current, old):
    """Magnitude, alignment, cancellation and orthogonal correction."""
    now, past = norm(current), norm(old)
    dot = sum(float(a.double().flatten().dot(b.double().flatten())) for a,b in zip(current,old))
    combined = norm([a+b for a,b in zip(current,old)])
    cosine = dot/(now*past) if now and past else None
    return dict(current_norm=now,old_norm=past,
        old_to_current=past/now if now else None,cosine=cosine,
        combined_norm=combined,
        combined_to_sum=combined/(now+past) if now+past else None,
        old_orthogonal_norm=(max(0.,past*past-dot*dot/(now*now))**.5 if now else past))


def group_indices(names,layer):
    base = [i for i,name in enumerate(names) if name.startswith(f'layer{layer}.')]
    groups = dict(all=base)
    groups['forget'] = [i for i in base if names[i].split('.')[1] in ('a_proj','A_log','dt_bias')]
    groups['write'] = [i for i in base if names[i].split('.')[1] == 'b_proj']
    for key in ('q_proj','k_proj','v_proj'):
        groups[key] = [i for i in base if names[i].split('.')[1] == key]
    return groups


def parameter_summary(names,indices,full,held,full_update,held_update):
    take = lambda values:[values[i] for i in indices]
    old = difference(full['total'],held['total'])
    old_ce = difference(full['ce'],held['ce'])
    old_reg = difference(full['reg'],held['reg'])
    update_change = difference(full_update['master'],held_update['master'])
    visible_change = difference(full_update['native'],held_update['native'])
    gradients = relation(take(held['total']),take(old))
    gradients.update(ce=relation(take(held['ce']),take(old_ce)),
        weighted_sigreg_full_norm=norm(take(full['reg'])),
        weighted_sigreg_held_norm=norm(take(held['reg'])),
        weighted_sigreg_old_norm=norm(take(old_reg)),
        sigreg_to_ce_full=(norm(take(full['reg']))/norm(take(full['ce']))
                           if norm(take(full['ce'])) else None))
    updates = relation(take(held_update['master']),take(update_change))
    updates.update(full_master_step_norm=norm(take(full_update['master'])),
        full_fp16_visible_step_norm=norm(take(full_update['native'])),
        held_fp16_visible_step_norm=norm(take(held_update['native'])),
        fp16_full_minus_held_norm=norm(take(visible_change)),
        fp16_different_coordinates=sum(int(torch.count_nonzero(x)) for x in take(visible_change)),
        coordinates=sum(full['total'][i].numel() for i in indices))
    return dict(parameters=[names[i] for i in indices],gradient=gradients,adam_update=updates)


def collect(learner,state,x,target,interval,held):
    learner.load_state_dict(state,detach_replay=held)
    learner.set_read_interval(interval)
    learner.optimizer.zero_grad(set_to_none=True)
    for _,parameter in learner.native:
        parameter.grad = None
    before = learner.state_dict()
    with learner.observation(), ExitStack() as stack:
        for layer,wrapper in learner.memories.items():
            stack.enter_context(wrapper.attach(learner.model,layer=layer))
        output = learner.model(x,past_key_values=clone_cache(learner.cache),use_cache=True)
        _,ce = language_loss(output.logits,target)
        reg = learner._regularizer()
        parameters = [parameter for _,parameter in learner.native]
        gradients = {}
        for name,objective,retain in [('ce',10*ce,True),('reg',.001*reg.to(ce.device),True),
                                       ('total',10*ce+.001*reg.to(ce.device),False)]:
            values = torch.autograd.grad(objective,parameters,retain_graph=retain,allow_unused=True)
            gradients[name] = [(torch.zeros_like(p,device='cpu',dtype=torch.float32) if g is None else
                                g.detach().float().cpu()) for p,g in zip(parameters,values)]
        metadata = dict(ce=float(ce.detach()),sigreg=float(reg.detach()),
            by_layer={str(layer):dict(query_count=w.last_read.slots.shape[0],
                unique_replays=len(w.replayed_by_slot),
                oldest_age_tokens=int((w.memory.clock-w.memory.birth[w.last_read.slots]).max()))
                for layer,w in learner.memories.items()},
            fp16_gradient_additivity_relative_error=norm(difference(gradients['total'],
                [a+b for a,b in zip(gradients['ce'],gradients['reg'])]))/
                max(norm(gradients['total']),1e-30))
        logits = output.logits.detach().cpu()
    assert_same(before,learner.state_dict(),'read_only_gradient_audit')
    return gradients,metadata,logits


def shadow_step(learner,state,gradients):
    """Use production AdamW and global native clipping, then discard the step."""
    learner.load_state_dict(state)
    learner.optimizer.zero_grad(set_to_none=True)
    for master,gradient in zip(learner.masters,gradients):
        master.grad = gradient.to(master.device).clone()
    before_norm = torch.nn.utils.clip_grad_norm_(learner.masters,1.)
    learner.optimizer.step()
    with torch.no_grad():
        master_delta = [p.detach().cpu()-old for p,old in zip(learner.masters,state['masters'])]
        visible_delta = [(master.detach().to(native.dtype).cpu().float()-
                          state['native'][name].float())
                         for master,(name,native) in zip(learner.masters,learner.native)]
    return dict(master=master_delta,native=visible_delta,preclip_norm=float(before_norm),
                clip_coefficient=min(1.,1./(float(before_norm)+1e-6)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model','checkpoint','curriculum','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--published',type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prior measurements')
    torch.set_num_threads(1)
    payload = torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    state = payload.get('learner',payload)
    curriculum = torch.load(args.curriculum,map_location='cpu',weights_only=False)
    report = dict(status='running',checkpoint=str(args.checkpoint),checkpoint_sha256=sha256(args.checkpoint),
        source_updates=state['updates'],dataset_revision=curriculum['dataset_revision'],window=128,
        persistent_updates=0,experience_commits=0,shadow_optimizer_steps=0,cases=[],
        protocol='three real windows at the same complete incoming state; legacy and positionwise reads; '
                 'same forward full/held; CE10 SIGReg.001; actual Adam/native clipping with restored moments; '
                 'old=current total gradient minus old-response-detached gradient; frozen diagnostic, not convergence')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,report)
    try:
        model,_ = load_pure_gdn(args.model)
        enable_streaming_convolution(model)
        learner = PretrainedExperienceLearner(model,layers=state['layers'],capacity=state['capacity'])
        learner.load_state_dict(state)
        report['trainable_native_layers'] = [layer+1 for layer in learner.layers]
        names = [name for name,_ in learner.native]
        samples = [('stream_41472',curriculum['stream'][41472:41601]),
                   ('validation_0',curriculum['heldouts']['validation'][:129]),
                   ('a_unseen_0',curriculum['heldouts']['a_unseen'][:129])]
        for label,ids in samples:
            if len(ids) != 129:
                raise ValueError('real window incomplete')
            x,target = ids[:-1][None].cuda(),ids[1:][None].cuda()
            for cadence,interval in [('single',None),('positionwise',1)]:
                full,full_meta,full_logits = collect(learner,state,x,target,interval,False)
                held,held_meta,held_logits = collect(learner,state,x,target,interval,True)
                delta = float((full_logits-held_logits).abs().max())
                if delta != 0 or full_meta != held_meta:
                    # Additivity rounding depends on graph accumulation and
                    # belongs to each gradient report, not forward equality.
                    a,b = dict(full_meta),dict(held_meta)
                    a.pop('fp16_gradient_additivity_relative_error'); b.pop('fp16_gradient_additivity_relative_error')
                    if delta != 0 or a != b:
                        raise AssertionError('credit cut changed observations')
                del full_logits,held_logits
                fu = shadow_step(learner,state,full['total']); report['shadow_optimizer_steps'] += 1
                hu = shadow_step(learner,state,held['total']); report['shadow_optimizer_steps'] += 1
                row = dict(sample=label,cadence=cadence,forward_max_difference=delta,
                    full=full_meta,held=held_meta,
                    clip=dict(full_norm=fu['preclip_norm'],held_norm=hu['preclip_norm'],
                              full_coefficient=fu['clip_coefficient'],held_coefficient=hu['clip_coefficient']),
                    by_layer={})
                for layer in learner.layers:
                    row['by_layer'][str(layer+1)] = {
                        key:parameter_summary(names,indices,full,held,fu,hu)
                        for key,indices in group_indices(names,layer).items()}
                report['cases'].append(row)
                write_json(args.output,report)
                print('CASE',label,cadence,{k:dict(
                    old_to_current=v['all']['gradient']['old_to_current'],
                    cosine=v['all']['gradient']['cosine'],
                    adam_change=v['all']['adam_update']['old_to_current'])
                    for k,v in row['by_layer'].items()},flush=True)
                del full,held,fu,hu
        learner.load_state_dict(state)
        restored = learner.state_dict()
        # Source is the new-format migrated individual: no legacy metadata normalization.
        assert_same(state,restored,'complete_individual_unchanged')
        report.update(status='complete',persistent_state_unchanged=True,
            source_checkpoint_unchanged=sha256(args.checkpoint)==report['checkpoint_sha256'],
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        if not report['source_checkpoint_unchanged']:
            raise AssertionError('source checkpoint changed')
        write_json(args.output,report)
        if args.published:
            write_json(args.published,report)
        print('COMPLETE',len(report['cases']),'cases; zero retained updates',flush=True)
    except Exception as error:
        report.update(status='failed',error=repr(error)); write_json(args.output,report)
        raise


if __name__ == '__main__':
    main()
