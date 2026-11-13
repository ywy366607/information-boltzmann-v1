"""Continuous real-text retention-only/FIFO pair with record-level lineage.

Both arms keep S, read learning and old-native replay credit. Only admission
victims differ. Probes are frozen sidecars; saved production state has no hooks.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import types

import torch

from scripts.ib_local.experience_consolidation import bounded_preference, replacement_slot
from scripts.ib_local.pretrained_gdn import load_pure_gdn, enable_streaming_convolution
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.pretrained_gdn_experience import copy_tree
from scripts.ib_local.probe_pretrained_gdn_consolidation import copy_tree_for_null_migration
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same, write_json


def identity(layer, slot, version):
    return f'{layer}:{slot}:{version}'


def real_schedule(curriculum, offset, b_steps):
    """Continue remaining real A, new B, then revisit exactly that A interval."""
    window = curriculum['window']
    a_end = curriculum['warm_tokens'] + curriculum['phase_steps']*window
    b_end = a_end + curriculum['phase_steps']*window
    if offset % window or not curriculum['warm_tokens'] <= offset < a_end:
        raise ValueError('offset must be a genuine aligned continuation within A')
    if not 0 < b_steps <= curriculum['phase_steps']:
        raise ValueError('B budget exceeds the existing genuine B interval')
    stream = curriculum['stream']
    a, b = stream[offset:a_end], stream[a_end:a_end+b_steps*window]
    if a_end+b.numel() > b_end:
        raise ValueError('B interval crossed the source revisit')
    a_steps = a.numel()//window
    sequence = torch.cat((a, b, a, stream[a_end:a_end+1]))
    phases = ['A_continue']*a_steps + ['B_fresh']*b_steps + ['A_revisit']*a_steps
    return sequence, phases, [a_steps, a_steps+b_steps, len(phases)]


def continuation_schedule(curriculum, offset, segment_steps):
    """Two actual adjacent chunks, then revisit the first; count real bridges."""
    window=curriculum['window']
    length=segment_steps*window
    stream=curriculum['stream']
    if offset<0 or offset%window or segment_steps<1 or offset+2*length>=len(stream):
        raise ValueError('aligned actual continuation and two available segments required')
    first,second=stream[offset:offset+length],stream[offset+length:offset+2*length]
    sequence=torch.cat((first,second,first,second[:1]))
    phases=['X_continue']*segment_steps+['Y_continue']*segment_steps+['X_revisit']*segment_steps
    return sequence,phases,[segment_steps,2*segment_steps,3*segment_steps]


def enrich_reads(events, row):
    """Join finite utility to the same slot-version receipt before admission."""
    for event in events:
        measured=row['credit'][str(event['layer'])].get('retention_utility',{})
        by_identity={identity(event['layer'],v['slot'],v['version']):v
                     for v in measured.get('observations',[])}
        for read in event['reads']:
            if read['identity'] in by_identity:
                observed=by_identity[read['identity']]
                read.update(removal_benefit=observed['benefit'],
                    removal_benefit_scope='whole observation, all reads of this identity',
                    preference_before=observed['preference_before'],
                    preference_after=observed['preference_after'])


def summarize_reuse(rows):
    positive,repeats,old,rescued=[],[],[],[]
    last={}
    for row in rows:
        seen = set()
        for event in row['admissions']:
            for read in event['reads']:
                if read['age_tokens']<=128 or 'removal_benefit' not in read:
                    continue
                # A whole-record removal is one effect, even when 128 queries
                # read it. Keep the historical record/window counting unit.
                if read['identity'] in seen:
                    continue
                seen.add(read['identity'])
                observed=dict(step=row['step'],phase=row['phase'],**read)
                old.append(observed)
                if read['removal_benefit']>1e-5:
                    positive.append(observed)
                    if read['identity'] in last:
                        repeats.append(dict(observed,
                            previous_beneficial_step=last[read['identity']],
                            gap_tokens=(row['step']-last[read['identity']])*128))
                    last[read['identity']]=row['step']
        rescued.extend(dict(step=row['step'],**read) for read in row['later_rescued_reads']
            if read['age_tokens']>128 and read.get('removal_benefit',0)>1e-5
            and read['original_evidence']>0)
    return dict(old_read_events=len(old),beneficial_old_reads=len(positive),
        harmful_old_reads=sum(r['removal_benefit'] < -1e-5 for r in old),
        distinct_beneficial_old_identities=len(last),repeated_beneficial_old_reads=len(repeats),
        repeated_beneficial_gaps_over_window=sum(r['gap_tokens']>128 for r in repeats),
        max_beneficial_reuse_gap_tokens=max((r['gap_tokens'] for r in repeats),default=0),
        max_beneficial_read_age_tokens=max((r['age_tokens'] for r in positive),default=0),
        beneficial_old_protection_increases=sum(r['preference_after']>r['preference_before']+1e-7 for r in positive),
        confirmed_beneficial_rescued_reuses=len(rescued),beneficial_reuse_records=repeats,
        confirmed_rescued_records=rescued)


def snapshots(learner):
    return {layer: dict(version=w.memory.read_credit_version.clone(),
                        first=w.memory.read_credit_first.clone())
            for layer, w in learner.memories.items()}


@contextmanager
def admission_trace(learner, previous):
    """Observe actual post-credit victim choice, before slot identity is reset."""
    events, originals = [], []
    for layer, wrapper in learner.memories.items():
        memory = wrapper.memory
        if 'commit_episode' in memory.__dict__:
            raise RuntimeError('do not nest admission instrumentation')
        original = memory.commit_episode
        originals.append(memory)

        def commit(*args, _layer=layer, _m=memory, _original=original, **kwargs):
            if not bool(_m.occupied.all()):
                raise ValueError('lineage study starts from a full observed table')
            oldest = int(_m.last_write.argmin())
            chosen = replacement_slot(_m) if _m.consolidation_enabled else int(_m.cursor)
            evidence = bounded_preference(_m)
            wrapper = learner.memories[_layer]
            read = wrapper.last_read
            positions = (wrapper.current['query_positions'] if wrapper.current is not None
                         else [0]*read.slots.shape[0])
            observed = (len(wrapper.current['source'][0]) if wrapper.current is not None else 128)
            b1 = _m.read_credit_betas[0]
            reads = []
            for position, slots, weights in zip(positions,read.slots.tolist(),read.weights.detach().tolist()):
                for slot, weight in zip(slots,weights):
                    version = int(_m.read_credit_version[slot])
                    if version != int(previous[_layer]['version'][slot]):
                        raise AssertionError('record changed before the update was settled')
                    gradient = (_m.read_credit_first[slot]-b1*previous[_layer]['first'][slot])/(1-b1)
                    reads.append(dict(identity=identity(_layer,slot,version), slot=slot,
                        age_tokens=int(_m.clock-_m.birth[slot])-observed+position, weight=weight,
                        query_position=position, credit_scope='aggregate across observation queries',
                        signed_read_parameter_credit=float(gradient)))
            event = dict(layer=_layer,
                victim=identity(_layer,chosen,int(_m.read_credit_version[chosen])),
                victim_slot=chosen, victim_age_tokens=int(_m.clock-_m.birth[chosen]),
                victim_evidence=float(evidence[chosen]),
                oldest=identity(_layer,oldest,int(_m.read_credit_version[oldest])),
                oldest_slot=oldest, oldest_evidence=float(evidence[oldest]),
                oldest_age_tokens=int(_m.clock-_m.birth[oldest]),
                differs_from_oldest=chosen != oldest, reads=reads)
            slot, hit = _original(*args, **kwargs)
            if hit or slot != chosen:
                raise AssertionError('actual admission disagreed with the recorded policy')
            event['new_identity'] = identity(_layer,slot,int(_m.read_credit_version[slot]))
            events.append(event)
            return slot, hit

        memory.commit_episode = commit
    try:
        yield events
    finally:
        for memory in originals:
            del memory.commit_episode


@contextmanager
def suppress_records(learner, identifiers):
    """Cut only named surviving values; keep selected weights unrenormalized."""
    changed = []
    for layer, wrapper in learner.memories.items():
        if 'retrieve' in wrapper.__dict__:
            raise RuntimeError('do not nest retrieval instrumentation')
        original = wrapper.retrieve

        def retrieve(self, query, replay, _layer=layer, _original=original):
            value = _original(query, replay)
            read, m = self.last_read, self.memory
            if not bool(m.occupied[read.slots].all()):
                raise ValueError('value probe requires occupied selected records')
            for slot in dict.fromkeys(read.slots.flatten().tolist()):
                key = identity(_layer,slot,int(m.read_credit_version[slot]))
                if key in identifiers:
                    encoded = self.encoder(self.replayed_by_slot[slot].float().cpu())
                    weight = (read.weights*(read.slots == slot)).sum(-1,keepdim=True)
                    value = value-weight*encoded
            return value

        wrapper.retrieve = types.MethodType(retrieve, wrapper)
        changed.append(wrapper)
    try:
        yield
    finally:
        for wrapper in changed:
            del wrapper.retrieve


def evaluate(learner, curriculum, protected, tokens):
    before = learner.state_dict()
    values = {name:learner.evaluate(ids[:tokens+1]) for name,ids in curriculum['heldouts'].items()}
    # Only genuinely positive-credit rescued records are called protected.
    live = {key for key, value in protected.items() if value['evidence'] > 0}
    with suppress_records(learner, live):
        cut = {name:learner.evaluate(ids[:tokens+1]) for name,ids in curriculum['heldouts'].items()}
    assert_same(before, learner.state_dict(), 'frozen_sidecar_does_not_change_individual')
    return dict(metrics=values, live_positive_rescued=len(live),
        suppressed_positive_rescued=cut,
        cut_minus_intact={name:cut[name]['nats_per_token']-values[name]['nats_per_token'] for name in values},
        persistent_state_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--curriculum', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--published', type=Path)
    parser.add_argument('--offset', type=int, default=14336)
    parser.add_argument('--b-steps', type=int, default=32)
    parser.add_argument('--evaluation-tokens', type=int, default=512)
    parser.add_argument('--comparison',choices=('retention','null','utility','placement','credit'),default='retention')
    parser.add_argument('--read-interval',type=int,
                        help='explicit token cadence migration; 1 reads every position; omitted restores checkpoint cadence')
    parser.add_argument('--segment-steps',type=int,default=32,
                        help='utility/placement/credit comparison: actual windows per X/Y/X phase')
    args = parser.parse_args()
    if args.read_interval is not None and args.read_interval < 1:
        parser.error('read interval must be positive')
    if args.output.exists():
        raise FileExistsError('preserve prior results')
    if args.evaluation_tokens <= 0 or args.evaluation_tokens % 128:
        parser.error('evaluation tokens must be a positive multiple of128')
    torch.set_num_threads(1)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    source = payload.get('learner',payload)
    curriculum = torch.load(args.curriculum, map_location='cpu', weights_only=False)
    sequence, phases, endpoints = (continuation_schedule(curriculum,args.offset,args.segment_steps)
        if args.comparison in ('utility','placement','credit') else real_schedule(curriculum,args.offset,args.b_steps))
    if any(len(ids) <= args.evaluation_tokens for ids in curriculum['heldouts'].values()):
        raise ValueError('independent sidecar budget exceeds the saved real data')
    if any(not m['memory._extra_state']['consolidation_enabled'] or
           m['memory._extra_state']['consolidation_rule'] != 'retention' for m in source['memories'].values()):
        raise ValueError('use the explicit migrated current individual, not a silent FIFO restart')
    if args.comparison in ('utility','placement','credit') and any(not m['_extra_state'].get('null_read_enabled',False)
                                         for m in source['memories'].values()):
        raise ValueError('utility pair requires the same already enabled null body')
    if args.comparison=='placement' and any(m['_extra_state'].get('recall_placement','input')!='input'
                                         for m in source['memories'].values()):
        raise ValueError('placement comparison requires the saved input-recall common start')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', checkpoint=str(args.checkpoint),
        curriculum=str(args.curriculum),source_locations=curriculum.get('source_locations'),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        source_updates=source['updates'], dataset_revision=curriculum['dataset_revision'],
        source_stream_sha256=curriculum['stream_sha256'],
        executed_stream_sha256=hashlib.sha256(sequence.numpy().tobytes()).hexdigest(),
        protocol=dict(offset=args.offset, window=128, read_interval_override=args.read_interval,
            comparison_kind=args.comparison,
            updates_per_arm=len(phases),
            train_tokens_per_arm=128*len(phases), endpoints=endpoints,
            b_steps=args.b_steps if args.comparison not in ('utility','placement','credit') else None,
            evaluation_tokens_per_source=args.evaluation_tokens,
            comparison=dict(retention='retention-only versus explicit historical FIFO; S, recall, native replay credit and read learning on in both',
                null='both keep credit retention; add a learned zero-valued null choice only in the null arm',
                utility='same enabled null/old-native learning/read dose; only retention evidence changes from dose gradient to finite removal benefit',
                placement='same enabled null/dose-based retention/old-native learning; only recall placement input versus output',
                credit='both retain S/recall/current encoder/read-dose learning/null/retention; only held old-native response gradient versus full')[args.comparison],
            segment_steps=args.segment_steps if args.comparison in ('utility','placement','credit') else None,
            finite_removal_probes_both_arms=args.comparison in ('utility','placement'),
            skipped_source_indices='not counted as time or observed experience; saved full cache is preserved',
            scope='single-source finite continuous A/B/A mechanism study; not convergence or a general architecture ranking'),
        arms={})
    code_paths=[Path(__file__).parent/name for name in (
        'run_pretrained_gdn_retention.py','pretrained_gdn_learning.py',
        'pretrained_gdn_experience.py','experience_consolidation.py',
        'latent_navigation_v3.py','navigation_read_credit.py','pretrained_gdn.py',
        'language_navigation_kanli_v4l.py')]
    report['code_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths}
    write_json(args.output,report)
    model, _ = load_pure_gdn(args.model)
    enable_streaming_convolution(model)
    learner = PretrainedExperienceLearner(model,layers=source['layers'],capacity=source['capacity'])

    def save_checkpoint(arm, protected, step):
        destination = args.output.parent/f'{arm}_last.pt'
        previous = args.output.parent/f'{arm}_previous.pt'
        tmp = destination.with_suffix('.tmp')
        torch.save(dict(learner=learner.state_dict(), step=step,
                        protected=protected, report=str(args.output)),tmp)
        if destination.exists():
            destination.replace(previous)
        tmp.replace(destination)
        return str(destination)

    try:
        arms = dict(retention=('fifo','retention'),null=('no_null','null'),
                    utility=('retention','utility'),placement=('input','output'),
                    credit=('held','full'))[args.comparison]
        for arm in arms:
            learner.load_state_dict(source,detach_replay=args.comparison=='credit' and arm=='held')
            if args.read_interval is not None:
                learner.set_read_interval(args.read_interval)
            if arm == 'fifo':
                for wrapper in learner.memories.values():
                    wrapper.memory.consolidation_enabled = False
            if args.comparison == 'null':
                if any(w.null_read_enabled for w in learner.memories.values()):
                    raise ValueError('null comparison needs a common legacy-read starting state')
                if arm == 'null':
                    old = learner.state_dict()
                    learner.enable_null_read()
                    enabled = learner.state_dict()
                    assert_same(copy_tree_for_null_migration(old,enabled),enabled,
                                'only_new_null_parameters_and_flag_change')
                    report['null_initialization_preserves_existing_state'] = True
                    report['null_extra_parameters'] = sum(p.numel() for p in learner.null_parameters)
                    del old,enabled
            if arm=='utility':
                old=learner.state_dict()
                learner.enable_utility_retention()
                enabled=learner.state_dict()
                expected=copy_tree(old)
                for layer in learner.layers:
                    expected['memories'][layer]['memory._extra_state']['consolidation_rule']='utility'
                assert_same(expected,enabled,'only_retention_evidence_policy_flag_changes')
                report['utility_migration_preserves_existing_state']=True
                del old,enabled,expected
            if args.comparison=='placement':
                old=learner.state_dict()
                learner.set_recall_placement(arm)
                enabled=learner.state_dict()
                expected=copy_tree(old)
                for layer in learner.layers:
                    expected['memories'][layer]['_extra_state']['recall_placement']=arm
                assert_same(expected,enabled,'only_recall_placement_flag_changes')
                report['placement_migration_preserves_existing_state']=True
                del old,enabled,expected
            protected = {}
            results = dict(status='running', steps=[], evaluations=[],
                rescued_admissions=0, positive_rescued_admissions=0,
                later_rescued_reads=0, later_positive_rescued_reads=0)
            report['arms'][arm] = results
            initial = evaluate(learner,curriculum,protected,args.evaluation_tokens)
            results['evaluations'].append(dict(step=0,phase='initial',**initial))
            if args.comparison=='placement' and arm=='output':
                report['initial_output_minus_input']={name:
                    initial['metrics'][name]['nats_per_token']-
                    report['arms']['input']['evaluations'][0]['metrics'][name]['nats_per_token']
                    for name in curriculum['heldouts']}
            if ((args.comparison=='retention' and arm=='retention') or arm=='utility' or
                    (args.comparison=='credit' and arm=='full')):
                assert_same(report['arms'][arms[0]]['evaluations'][0],results['evaluations'][0],
                            'same_complete_start_predictions')
            write_json(args.output,report)
            for index, phase in enumerate(phases):
                if (args.output.parent/'STOP').exists():
                    results['checkpoint'] = save_checkpoint(arm,protected,index)
                    results['status'] = report['status'] = 'stopped'
                    write_json(args.output,report)
                    return
                x,y = sequence[index*128:(index+1)*128][None].cuda(),sequence[index*128+1:(index+1)*128+1][None].cuda()
                previous = snapshots(learner)
                with admission_trace(learner,previous) as events:
                    row = (learner.step(x,y,diagnose_retention=True) if args.comparison in ('utility','placement')
                           else learner.step(x,y,diagnose=args.comparison=='credit' and index%32==0))
                if args.comparison in ('utility','placement'):
                    enrich_reads(events,row)
                later = []
                for event in events:
                    for read in event['reads']:
                        saved = protected.get(read['identity'])
                        if saved is not None and saved['step'] < index+1:
                            later.append(dict(**read,first_rescue_step=saved['step'],
                                              original_evidence=saved['evidence']))
                    protected.pop(event['victim'],None)
                    if event['differs_from_oldest']:
                        results['rescued_admissions'] += 1
                        positive = event['oldest_evidence'] > 0
                        results['positive_rescued_admissions'] += int(positive)
                        record = protected.setdefault(event['oldest'],dict(step=index+1,evidence=event['oldest_evidence']))
                        record['evidence'] = max(record['evidence'],event['oldest_evidence'])
                results['later_rescued_reads'] += len(later)
                results['later_positive_rescued_reads'] += sum(r['original_evidence'] > 0 for r in later)
                row.update(step=index+1,phase=phase,admissions=events,later_rescued_reads=later)
                results['steps'].append(row)
                if (((args.comparison=='retention' and arm=='retention') or arm=='utility' or
                     (args.comparison=='credit' and arm=='full')) and index == 0):
                    first = report['arms'][arms[0]]['steps'][0]
                    for key in ('ce','sigreg','reads'):
                        assert_same(first[key],row[key],f'first_update_same_{key}')
                    report['first_training_forward_exact'] = True
                if (index+1) % 8 == 0:
                    print(arm,index+1,phase,'CE',round(row['ce'],6),'rescues',results['rescued_admissions'],
                          'later_positive_reads',results['later_positive_rescued_reads'],flush=True)
                if index+1 in endpoints:
                    metrics = evaluate(learner,curriculum,protected,args.evaluation_tokens)
                    results['evaluations'].append(dict(step=index+1,phase=phase,**metrics))
                    results['checkpoint'] = save_checkpoint(arm,protected,index+1)
                    if args.comparison in ('utility','placement'):
                        results['reuse']=summarize_reuse(results['steps'])
                    print('EVAL',arm,index+1,{n:r['nats_per_token'] for n,r in metrics['metrics'].items()},flush=True)
                write_json(args.output,report)
            results.update(status='complete', new_updates=len(phases), train_tokens=128*len(phases),
                phases={phase:sum(r['nll_sum'] for r in results['steps'] if r['phase']==phase)/
                    sum(r['target_tokens'] for r in results['steps'] if r['phase']==phase) for phase in set(phases)})
            write_json(args.output,report)
        report.update(status='complete',executed_optimizer_updates=2*len(phases),
            source_checkpoint_unchanged=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==report['checkpoint_sha256'])
        report['executed_code_unchanged']=all(hashlib.sha256(path.read_bytes()).hexdigest()==
            report['code_sha256'][str(path)] for path in code_paths)
        if not report['source_checkpoint_unchanged'] or not report['executed_code_unchanged']:
            raise AssertionError('source checkpoint or executed code changed during the run')
        difference_key = dict(retention='final_retention_minus_fifo',null='final_null_minus_no_null',
                              utility='final_utility_minus_retention',
                              placement='final_output_minus_input',credit='final_full_minus_held')[args.comparison]
        report[difference_key] = {name:
            report['arms'][arms[1]]['evaluations'][-1]['metrics'][name]['nats_per_token']-
            report['arms'][arms[0]]['evaluations'][-1]['metrics'][name]['nats_per_token']
            for name in curriculum['heldouts']}
        write_json(args.output,report)
        if args.published:
            args.published.parent.mkdir(parents=True,exist_ok=True)
            write_json(args.published,report)
        print('COMPLETE',report[difference_key],flush=True)
    except Exception as error:
        report.update(status='failed',error=repr(error))
        write_json(args.output,report)
        raise


if __name__ == '__main__':
    main()
