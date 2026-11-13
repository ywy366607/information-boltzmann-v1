"""Real-update admission for distinct read-dose and retention-use evidence."""
import argparse
import hashlib
from pathlib import Path
import time

import torch

from scripts.ib_local.audit_pretrained_gdn_null_effect import sha256
from scripts.ib_local.pretrained_gdn import load_pure_gdn, enable_streaming_convolution
from scripts.ib_local.pretrained_gdn_experience import copy_tree
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','checkpoint','curriculum','output','save-candidate'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--published',type=Path)
    p.add_argument('--offset',type=int,default=20992)
    p.add_argument('--max-updates',type=int,default=8)
    args=p.parse_args()
    if args.output.exists() or args.save_candidate.exists():
        raise FileExistsError('preserve previous evidence')
    if args.offset<0 or args.offset%128 or not 1<=args.max_updates<=8:
        p.error('aligned real offset and one to eight wiring updates required')
    torch.set_num_threads(1)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    state=payload['learner']
    curriculum=torch.load(args.curriculum,map_location='cpu',weights_only=False)
    stream=curriculum['stream']
    if len(stream)<args.offset+128*args.max_updates+1:
        raise ValueError('insufficient actual continuation')
    report=dict(status='running',checkpoint=str(args.checkpoint),checkpoint_sha256=sha256(args.checkpoint),
        source_updates=state['updates'],dataset_revision=curriculum['dataset_revision'],
        source_stream_sha256=curriculum['stream_sha256'],
        protocol=dict(offset=args.offset,window=128,max_retained_updates=args.max_updates,
            scope='mechanism/complete-restore admission only, not language convergence or validation gain',
            stopping='stop at first positive old-record removal benefit with increased retention, at most eight actual updates',
            comparisons='first update against original dose-based retention; all weights/Adam/cache must be identical before admission histories diverge'),
        executed_optimizer_updates=0,records=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,report)
    start=time.perf_counter()
    model,_=load_pure_gdn(args.model)
    enable_streaming_convolution(model)
    learner=PretrainedExperienceLearner(model,layers=state['layers'],capacity=state['capacity'])
    learner.load_state_dict(state)
    baseline=learner.state_dict()
    try:
        x=stream[args.offset:args.offset+128][None].cuda()
        y=stream[args.offset+1:args.offset+129][None].cuda()
        old_logits,_=learner.predict(x)
        original=learner.step(x,y)
        report['executed_optimizer_updates']+=1
        original_after=learner.state_dict()
        learner.load_state_dict(baseline)
        learner.enable_utility_retention()
        before=learner.state_dict()
        expected=copy_tree(baseline)
        for layer in learner.layers:
            expected['memories'][layer]['memory._extra_state']['consolidation_rule']='utility'
        assert_same(expected,before,'only_explicit_policy_flag_changes')
        del expected,baseline
        new_logits,_=learner.predict(x)
        report['migration_forward_max_difference']=float((new_logits-old_logits).abs().max())
        if not torch.equal(old_logits,new_logits):
            raise AssertionError('utility-only retention migration changed read')
        del old_logits,new_logits
        row=learner.step(x,y)
        report['executed_optimizer_updates']+=1
        after=learner.state_dict()
        for key in ('native','masters','optimizer','cache','rng','cuda_rng'):
            assert_same(original_after[key],after[key],'same_first_step_'+key)
        for key in ('ce','sigreg','reads'):
            assert_same(original[key],row[key],'same_first_forward_'+key)
        report['first_update_weights_adam_and_cache_exact']=True
        del original_after
        learner.load_state_dict(before)
        repeated=learner.step(x,y)
        report['executed_optimizer_updates']+=1
        assert_same(row,repeated,'next_real_update_record')
        assert_same(after,learner.state_dict(),'next_real_update_complete_state')
        report['next_update_restore_exact']=True
        del before,after,repeated
        for index in range(args.max_updates):
            if index:
                pos=args.offset+128*index
                row=learner.step(stream[pos:pos+128][None].cuda(),stream[pos+1:pos+129][None].cuda())
                report['executed_optimizer_updates']+=1
            observations=[]
            for layer,w in learner.memories.items():
                evidence=row['credit'][str(layer)]['retention_utility']
                for item in evidence['observations']:
                    # The actual pre-update age, preserved in the step's read.
                    read=row['reads'][str(layer)]
                    age=read['ages_tokens'][read['slots'].index(item['slot'])]
                    survived = (bool(w.memory.occupied[item['slot']]) and
                                int(w.memory.read_credit_version[item['slot']]) == item['version'])
                    observations.append(dict(layer=layer,age_tokens=age,
                                             survives_current_admission=survived,**item))
            positive=[r for r in observations if r['age_tokens']>128 and r['benefit']>1e-5
                      and r['preference_after']>r['preference_before']+1e-7]
            row.update(utility_observations=observations,positive_old_retention_witnesses=positive)
            report['records'].append(row)
            write_json(args.output,report)
            print('UTILITY',index+1,'CE',row['ce'],'positive old witnesses',len(positive),flush=True)
            if positive:
                break
        candidate=learner.state_dict()
        args.save_candidate.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(learner=candidate,step=len(report['records']),
            parent_checkpoint=str(args.checkpoint),protocol=report['protocol']),args.save_candidate)
        disk=torch.load(args.save_candidate,map_location='cpu',weights_only=False,mmap=True)
        learner.load_state_dict(disk['learner'])
        assert_same(candidate,learner.state_dict(),'complete_disk_roundtrip')
        report.update(status='passed',retained_new_updates=len(report['records']),
            training_tokens=128*len(report['records']),extra_buffer_bytes=16*state['capacity'],
            executed_stream_sha256=hashlib.sha256(stream[
                args.offset:args.offset+128*len(report['records'])+1].numpy().tobytes()).hexdigest(),
            candidate=str(args.save_candidate),complete_disk_restore_exact=True,
            source_checkpoint_unchanged=sha256(args.checkpoint)==report['checkpoint_sha256'],
            positive_old_retention_witnesses=sum(len(r['positive_old_retention_witnesses']) for r in report['records']),
            seconds=time.perf_counter()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        if not report['source_checkpoint_unchanged']:
            raise AssertionError('source checkpoint changed')
        write_json(args.output,report)
        if args.published:
            write_json(args.published,report)
        print('COMPLETE',report['retained_new_updates'],report['positive_old_retention_witnesses'],flush=True)
    except Exception as error:
        report.update(status='failed',error=repr(error)); write_json(args.output,report)
        raise


if __name__=='__main__':
    main()
