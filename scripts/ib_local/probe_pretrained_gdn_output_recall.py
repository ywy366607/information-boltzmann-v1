"""Real language admission for recall at the layer's outgoing boundary."""
import argparse
from pathlib import Path

import torch

from scripts.ib_local.audit_pretrained_gdn_null_effect import sha256
from scripts.ib_local.pretrained_gdn import load_pure_gdn, enable_streaming_convolution
from scripts.ib_local.pretrained_gdn_experience import copy_tree
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner, language_loss
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','checkpoint','curriculum','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--published',type=Path)
    p.add_argument('--offset',type=int,default=29184)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prior evidence')
    torch.set_num_threads(1)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    state=payload['learner']
    curriculum=torch.load(args.curriculum,map_location='cpu',weights_only=False)
    ids=curriculum['stream'][args.offset:args.offset+129]
    if len(ids)!=129:
        raise ValueError('one actual 128-token observation required')
    report=dict(status='running',checkpoint=str(args.checkpoint),checkpoint_sha256=sha256(args.checkpoint),
        source_updates=state['updates'],dataset_revision=curriculum['dataset_revision'],
        stream_offset=args.offset,real_tokens=128,executed_optimizer_updates=0,
        retained_new_updates=0,scope='next-update and causal-path admission; no retained training progress')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,report)
    try:
        model,_=load_pure_gdn(args.model)
        enable_streaming_convolution(model)
        learner=PretrainedExperienceLearner(model,layers=state['layers'],capacity=state['capacity'])
        learner.load_state_dict(state)
        baseline=learner.state_dict()
        x,y=ids[:-1][None].cuda(),ids[1:][None].cuda()
        original,_=learner.predict(x)
        learner.set_recall_placement('output')
        enabled=learner.state_dict()
        expected=copy_tree(baseline)
        for layer in learner.layers:
            expected['memories'][layer]['_extra_state']['recall_placement']='output'
        assert_same(expected,enabled,'only_explicit_placement_flag_changes')
        changed,_=learner.predict(x)
        _,old_ce=language_loss(original,y); _,new_ce=language_loss(changed,y)
        report.update(migration_preserves_existing_state=True,input_ce=float(old_ce),
            output_ce=float(new_ce),initial_max_logit_difference=float((original-changed).abs().max()))
        del baseline,expected,original,changed
        row=learner.step(x,y,diagnose=True,diagnose_retention=True)
        report['executed_optimizer_updates']+=1
        after=learner.state_dict()
        learner.load_state_dict(enabled)
        repeat=learner.step(x,y,diagnose=True,diagnose_retention=True)
        report['executed_optimizer_updates']+=1
        assert_same(row,repeat,'same_next_update_report')
        assert_same(after,learner.state_dict(),'same_next_update_complete_state')
        report.update(update=row,next_update_restore_exact=True,
            per_layer_old_native_credit={str(layer):row['diagnostics'][f'layer{layer}_old_native_grad_norms']
                for layer in learner.layers},
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        if any(not any(v is not None and v>0 for v in values.values())
               for values in report['per_layer_old_native_credit'].values()):
            raise AssertionError('old local native credit failed to reach an attached layer')
        learner.load_state_dict(state)
        legacy_expected=copy_tree(state)
        for layer in learner.layers:
            legacy_expected['memories'][layer]['_extra_state'].setdefault('recall_placement','input')
        assert_same(legacy_expected,learner.state_dict(),'restore_unchanged_original_individual')
        report['legacy_default_metadata_normalized']=True
        report.update(status='passed',source_checkpoint_unchanged=sha256(args.checkpoint)==report['checkpoint_sha256'])
        if not report['source_checkpoint_unchanged']:
            raise AssertionError('source checkpoint changed')
        write_json(args.output,report)
        if args.published:
            write_json(args.published,report)
        print('PASSED',report['input_ce'],report['output_ce'],report['per_layer_old_native_credit'],flush=True)
    except Exception as error:
        report.update(status='failed',error=repr(error)); write_json(args.output,report)
        raise


if __name__=='__main__':
    main()
