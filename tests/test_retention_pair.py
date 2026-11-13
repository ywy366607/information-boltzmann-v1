"""Real-stream schedule and causal-lineage accounting invariants."""
import torch

from scripts.ib_local.run_pretrained_gdn_retention import (
    continuation_schedule, enrich_reads, summarize_reuse,
)


def test_continuation_schedule_counts_every_scored_bridge_and_exact_revisit():
    stream=torch.arange(2048)
    sequence,phases,endpoints=continuation_schedule(dict(stream=stream,window=128),256,2)
    assert phases==['X_continue']*2+['Y_continue']*2+['X_revisit']*2
    assert endpoints==[2,4,6] and len(sequence)==769
    assert torch.equal(sequence[:512],stream[256:768])
    assert torch.equal(sequence[512:768],stream[256:512])
    assert sequence[256]==stream[512]  # X->Y target
    assert sequence[512]==stream[256]  # actual Y->X revisit target
    assert sequence[-1]==stream[512]  # actual final bridge target


def test_utility_receipt_is_joined_by_slot_and_version_not_just_slot():
    events=[dict(layer=1,reads=[dict(identity='1:0:2'),dict(identity='1:0:3')])]
    row=dict(credit={'1':dict(retention_utility=dict(observations=[dict(slot=0,version=3,
        benefit=.1,preference_before=0.,preference_after=.5)]))})
    enrich_reads(events,row)
    assert 'removal_benefit' not in events[0]['reads'][0]
    assert events[0]['reads'][1]['removal_benefit']==.1


def test_reuse_counts_finite_benefit_separately_from_gradient_and_rescue():
    def read(identity,benefit):
        return dict(identity=identity,age_tokens=512,removal_benefit=benefit,
                    preference_before=0.,preference_after=.5,signed_read_parameter_credit=-.1)
    def row(step,records,later=()):
        return dict(step=step,phase='X_continue',admissions=[dict(reads=records)],
                    later_rescued_reads=list(later))
    records=[row(1,[read('1:0:2',.1)]),row(2,[read('1:0:3',.2)]),
             row(4,[read('1:0:2',.3)], [dict(read('1:0:2',.3),original_evidence=.5)])]
    result=summarize_reuse(records)
    assert result['beneficial_old_reads']==3
    assert result['distinct_beneficial_old_identities']==2
    assert result['repeated_beneficial_old_reads']==1
    assert result['max_beneficial_reuse_gap_tokens']==384
    assert result['beneficial_old_protection_increases']==3
    assert result['confirmed_beneficial_rescued_reuses']==1


def test_first_segment_reuse_requires_the_original_identity_and_actual_gap():
    from scripts.ib_local.analyze_pretrained_gdn_utility_pair import trace_written_reuse
    def event(victim,new,reads=()):
        return dict(victim=victim,new_identity=new,reads=list(reads))
    steps=[dict(step=1,phase='X_continue',admissions=[event('seed0','X0')]),
           dict(step=2,phase='Y_continue',admissions=[event('seed1','Y0')]),
           dict(step=3,phase='X_revisit',admissions=[event('X0','X1',[
               dict(identity='X0',age_tokens=192,removal_benefit=.1),
               dict(identity='Y0',age_tokens=64,removal_benefit=.1)])])]
    result=trace_written_reuse(steps,[1,2,3])
    assert result['first_X_records_still_alive']=={'1':1,'2':1,'3':0}
    assert result['first_X_reads_during_revisit']==1
    assert result['beneficial_first_X_reads_during_revisit']==1
    assert result['maximum_beneficial_source_age_tokens']==192
    assert result['maximum_beneficial_elapsed_since_write_tokens']==128
