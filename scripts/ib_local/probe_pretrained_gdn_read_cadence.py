"""Real-model admission for reading cadence independent of the BPTT window.

Two temporary next updates test exact continuation; neither is retained. The
saved candidate changes only cadence at the original source update count.
"""
import argparse
from contextlib import ExitStack
from pathlib import Path
import time

import torch

from scripts.ib_local.audit_pretrained_gdn_null_effect import sha256
from scripts.ib_local.pretrained_gdn import load_pure_gdn, enable_streaming_convolution, clone_cache
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner, language_loss
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same, write_json


@torch.no_grad()
def inspect(learner,x):
    with learner.observation(), ExitStack() as stack:
        for layer,w in learner.memories.items():
            stack.enter_context(w.attach(learner.model,layer=layer))
        output = learner.model(x,past_key_values=clone_cache(learner.cache),use_cache=True)
        rows = {}
        for layer,w in learner.memories.items():
            read = w.last_read
            rows[str(layer)] = dict(layer_number=layer+1,query_count=read.slots.shape[0],
                positions=w.current['query_positions'],unique_replays=len(w.replayed_by_slot),
                capacity=w.memory.capacity,experience_cpu_bytes=w.storage_nbytes,
                adjacent_weight_changes=int((read.weights[1:] != read.weights[:-1]).any(-1).sum()))
        return output.logits,rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model','checkpoint','curriculum','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--published',type=Path)
    parser.add_argument('--offset',type=int,default=41472)
    parser.add_argument('--read-interval',type=int,default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous evidence')
    torch.set_num_threads(1)
    payload = torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    source = payload.get('learner',payload)
    curriculum = torch.load(args.curriculum,map_location='cpu',weights_only=False)
    ids = curriculum['stream'][args.offset:args.offset+129]
    if len(ids) != 129:
        raise ValueError('one real 128-token observation required')
    report = dict(status='running',checkpoint=str(args.checkpoint),
        checkpoint_sha256=sha256(args.checkpoint),source_updates=source['updates'],
        stream_offset=args.offset,dataset_revision=curriculum['dataset_revision'],
        window=128,read_interval=args.read_interval,retained_new_updates=0,
        executed_optimizer_updates=0,scope='real-model wiring and restore, not language capability comparison')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,report)
    try:
        model,_ = load_pure_gdn(args.model)
        enable_streaming_convolution(model)
        learner = PretrainedExperienceLearner(model,layers=source['layers'],capacity=source['capacity'])
        learner.load_state_dict(source)
        x,y = ids[:-1][None].cuda(),ids[1:][None].cuda()
        old,old_rows = inspect(learner,x)
        expected = learner.state_dict()
        learner.set_read_interval(args.read_interval)
        for layer in learner.layers:
            expected['memories'][layer]['_extra_state']['read_interval'] = args.read_interval
        enabled = learner.state_dict()
        assert_same(expected,enabled,'only_cadence_changes')
        del expected
        candidate = args.output.parent/'migrated.pt'
        torch.save(dict(learner=enabled,source_checkpoint=str(args.checkpoint)),candidate)
        del enabled
        started = time.perf_counter()
        new,new_rows = inspect(learner,x)
        report['read_seconds'] = time.perf_counter()-started
        _,old_ce = language_loss(old,y); _,new_ce = language_loss(new,y)
        report.update(migration_preserves_other_state=True,legacy=old_rows,per_layer=new_rows,
            old_ce=float(old_ce),new_ce=float(new_ce),initial_logit_max=float((old-new).abs().max()))
        changed = x.clone(); changed[:,64:] = x[:,64:].flip(1)
        future,_ = inspect(learner,changed)
        report['future_to_prefix_logit_max'] = float((future[:,:64]-new[:,:64]).abs().max())
        if report['future_to_prefix_logit_max'] != 0:
            raise AssertionError('later queries changed earlier predictions')
        del old,new,future
        print('READS',report['per_layer'],flush=True)
        started = time.perf_counter()
        row = learner.step(x,y,diagnose=True)
        report['executed_optimizer_updates'] += 1
        report['update_seconds'] = time.perf_counter()-started
        after = learner.state_dict()
        # Actual disk continuation, not just resetting tensors in memory.
        disk = torch.load(candidate,map_location='cpu',weights_only=False,mmap=True)
        learner.load_state_dict(disk['learner'])
        repeat = learner.step(x,y,diagnose=True)
        report['executed_optimizer_updates'] += 1
        assert_same(row,repeat,'same_next_update_report')
        assert_same(after,learner.state_dict(),'same_next_update_complete_state')
        del after
        report.update(next_update_disk_restore_exact=True,update=row,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),candidate=str(candidate),
            candidate_updates=source['updates'])
        for layer in learner.layers:
            gradients = row['diagnostics'].get(f'layer{layer}_old_native_grad_norms',{})
            if not any(value is not None and value > 0 for value in gradients.values()):
                raise AssertionError('old native credit did not reach a selected layer')
        learner.load_state_dict(source)
        report.update(status='passed',source_checkpoint_unchanged=sha256(args.checkpoint)==report['checkpoint_sha256'])
        if not report['source_checkpoint_unchanged']:
            raise AssertionError('source checkpoint changed')
        write_json(args.output,report)
        if args.published:
            write_json(args.published,report)
        print('PASSED',report['old_ce'],report['new_ce'],report['peak_cuda_allocated_bytes'],flush=True)
    except Exception as error:
        report.update(status='failed',error=repr(error))
        write_json(args.output,report)
        raise


if __name__ == '__main__':
    main()
