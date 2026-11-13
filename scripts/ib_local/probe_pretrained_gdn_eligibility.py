"""Same-forward real-stream audit of delayed teaching/participation matching.

This is a bounded learning-route experiment, not an instruction benchmark or
an assertion that dense next-token text requires the added credit route.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.ib_local.audit_eligibility_routing import eligibility_routing, receipt_summary
from scripts.ib_local.pretrained_gdn import load_pure_gdn, enable_streaming_convolution
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.run_pretrained_gdn_long_sft import write_json
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same


def group(name):
    if any(x in name for x in ('A_log','dt_bias','a_proj')):
        return 'retention'
    if 'b_proj' in name:
        return 'erase_write_gain'
    if any(x in name for x in ('q_proj','k_proj','q_conv','k_conv')):
        return 'address'
    if any(x in name for x in ('v_proj','v_conv')):
        return 'content'
    return 'readout'


def credit_difference(full, exchanged, held):
    rows = {}
    for category in sorted({group(n) for n in held}):
        names = [n for n in held if group(n) == category]
        matched = torch.cat([(full[n]-held[n]).flatten().double() for n in names])
        swapped = torch.cat([(exchanged[n]-held[n]).flatten().double() for n in names])
        rows[category] = dict(matched_old_gradient_norm=float(matched.norm()),
            control_old_gradient_norm=float(swapped.norm()),
            difference_norm=float((matched-swapped).norm()),
            cosine=float(torch.dot(matched,swapped)/(matched.norm()*swapped.norm()).clamp_min(1e-30)))
    return rows


def forward_equal(rows):
    fields = ('ce','nll_sum','sigreg','reads','supervised_tokens','supervised_nll_sum')
    for name, row in rows.items():
        for key in fields:
            if row[key] != rows['matched'][key]:
                raise AssertionError(f'{name} changed forward field {key}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--curriculum',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--published',type=Path)
    parser.add_argument('--pairs',type=int,default=16)
    parser.add_argument('--offset',type=int,default=8192)
    parser.add_argument('--mark-tokens',type=int,default=4096)
    parser.add_argument('--future-tokens',type=int,default=512)
    parser.add_argument('--wrong-kind',choices=('exchange','other'),default='exchange')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prior evidence')
    if args.pairs < 1 or args.mark_tokens % 128 or args.future_tokens % 128:
        parser.error('positive pairs and128-aligned token budgets required')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    source = torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    curriculum = torch.load(args.curriculum,map_location='cpu',weights_only=False)
    stream = curriculum['stream']
    if args.offset+args.mark_tokens+128*args.pairs+args.future_tokens+1 > len(stream):
        raise ValueError('insufficient actual stream')
    report = dict(kind='real_stream_eligibility_correspondence',status='running',
        checkpoint=str(args.checkpoint),source_updates=source['updates'],
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        dataset_revision=curriculum['dataset_revision'],stream_sha256=curriculum['stream_sha256'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol=dict(window=128,offset=args.offset,mark_tokens=args.mark_tokens,
            pairs=args.pairs,future_tokens=args.future_tokens,
            wrong_kind=args.wrong_kind,
            common_trajectory='matched post-update state retained, all sidecar observations rolled back',
            exchange=('send teaching to distinct unselected old participation in the same compartment, preserving selected forward values'
                      if args.wrong_kind == 'other' else
                      'permute teaching only among selected responses older than128tokens; forward values and live encoders unchanged'),
            scope='correlated one-update real-OWT route probes, not converged language capability or proof that the extra route is necessary'),
        records=[])
    write_json(args.output,report)
    model, _ = load_pure_gdn(args.model)
    enable_streaming_convolution(model)
    learner = PretrainedExperienceLearner(model,layers=source['layers'],capacity=source['capacity'])
    learner.load_state_dict(source,detach_replay=False)
    if any(w.memory.consolidation_enabled for w in learner.memories.values()):
        raise ValueError('use the saved consolidation-off source')
    try:
        # Real prior participation is observed, not assigned a synthetic label.
        for start in range(args.offset,args.offset+args.mark_tokens,128):
            learner.observe(stream[start:start+128][None].cuda())
        for index in range(args.pairs):
            start = args.offset+args.mark_tokens+128*index
            x = stream[start:start+128][None].cuda()
            target = stream[start+1:start+129][None].cuda()
            future = stream[start+128:start+128+args.future_tokens+1]
            before = learner.state_dict()
            rows,signals,gradients,forecasts,masters = {},{},{},{},{}
            matched_after = None
            for arm in ('matched','exchanged','held'):
                learner.load_state_dict(before,detach_replay=arm == 'held')
                with eligibility_routing(learner,
                        exchange=arm == 'exchanged' and args.wrong_kind == 'exchange',
                        other_participation=arm == 'exchanged' and args.wrong_kind == 'other') as receipts:
                    rows[arm] = learner.step(x,target)
                signals[arm] = receipt_summary(receipts)
                if arm != 'held':
                    for r in signals[arm]:
                        if 'sent_norm' in r and not r['correspondence_exact']:
                            raise AssertionError('delivered teaching does not match registered permutation')
                        if 'sent_norm' in r and abs(r['sent_norm']-r['received_norm']) > 1e-9:
                            raise AssertionError('exchange changed the teaching norm')
                gradients[arm] = {n:p.grad.detach().float().cpu().clone() for n,p in learner.native}
                masters[arm] = [m.detach().cpu().clone() for m in learner.masters]
                if arm == 'matched':
                    matched_after = learner.state_dict()
                else:
                    # Same forward current transition implies exactly the same
                    # incoming state for the later forecast, before weight effects.
                    assert_same(matched_after['cache'],dict(states=learner.cache.states,
                                seen_tokens=learner.cache._seen_tokens))
                    control_after = learner.state_dict()
                    for layer in learner.layers:
                        control_after['memories'][layer]['_extra_state']['detach_replay'] = False
                    assert_same(matched_after['memories'],control_after['memories'])
                    del control_after
                forecasts[arm] = learner.evaluate(future)['nats_per_token']
            forward_equal(rows)
            # Current encoder/inlet learning and stored observations must not
            # be changed by the native-old-gradient intervention.
            held_after = learner.state_dict()
            for layer in learner.layers:
                held_after['memories'][layer]['_extra_state']['detach_replay'] = False
            assert_same(matched_after['memories'],held_after['memories'])
            changed = {arm:max(float((a-b).abs().max()) for a,b in zip(
                masters[arm],masters['held'])) for arm in ('matched','exchanged')}
            active = [r for r in signals['exchanged'] if
                (r['exchanged_entries'] > 0 or r['wrong_receiver_count'] > 0) and 'sent_norm' in r]
            record = dict(pair=index+1,stream_start=start,forward_exact=True,
                current_ce=rows['matched']['ce'],signals=signals,
                exchanged_compartments=len(active),
                wrong_receiver_entries=sum(r['wrong_receiver_count'] for r in signals['exchanged']),
                nonzero_changed_teaching=sum(r['changed_teaching_norm'] > 0 for r in active),
                old_gradient_by_group=credit_difference(gradients['matched'],gradients['exchanged'],gradients['held']),
                fp32_update_max_difference_from_held=changed,
                next_continuation_nll=forecasts,
                matched_minus_exchanged=forecasts['matched']-forecasts['exchanged'],
                matched_minus_held=forecasts['matched']-forecasts['held'])
            report['records'].append(record)
            learner.load_state_dict(matched_after)
            write_json(args.output,report)
            print('PAIR',index+1,'exchanges',len(active),'future',forecasts,flush=True)
            del before,matched_after,held_after,gradients,masters
        records = report['records']
        report.update(status='complete',
            active_exchanged_compartments=sum(r['exchanged_compartments'] for r in records),
            mean_matched_minus_exchanged=sum(r['matched_minus_exchanged'] for r in records)/len(records),
            mean_matched_minus_held=sum(r['matched_minus_held'] for r in records)/len(records),
            matched_better_than_exchanged=sum(r['matched_minus_exchanged'] < 0 for r in records),
            matched_better_than_held=sum(r['matched_minus_held'] < 0 for r in records),
            executed_optimizer_updates=3*args.pairs,retained_optimizer_updates=args.pairs,
            source_checkpoint_unchanged=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==report['checkpoint_sha256'])
        saved = args.output.parent/'matched_last.pt'
        torch.save(learner.state_dict(),saved.with_suffix('.tmp'))
        saved.with_suffix('.tmp').replace(saved)
        report['checkpoint_after_probe'] = str(saved)
        write_json(args.output,report)
        if args.published:
            args.published.parent.mkdir(parents=True,exist_ok=True)
            write_json(args.published,report)
        print('COMPLETE',report['mean_matched_minus_exchanged'],report['mean_matched_minus_held'],flush=True)
    except Exception as error:
        report.update(status='failed',error=repr(error))
        write_json(args.output,report)
        raise


if __name__ == '__main__':
    main()
