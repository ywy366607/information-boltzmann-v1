"""Real-data preparation indexing checks; no synthetic capability study."""
import pytest
import torch

from scripts.ib_local.prepare_pretrained_gdn_credit_stream import pack_stream
from scripts.ib_local.run_pretrained_gdn_retention import continuation_schedule


def test_fresh_stream_revisits_first_segment_and_keeps_continuations_untrained():
    x,y,v = torch.arange(1000),torch.arange(2000,3000),torch.arange(4000,5000)
    data = pack_stream(x,y,v,phase_tokens=256,validation_tokens=128)
    seq,phases,ends = continuation_schedule(data,0,2)
    assert ends == [2,4,6]
    assert len(seq) == 769
    assert torch.equal(seq[:256],seq[512:768])
    assert seq[256] == 2000 and seq[512] == 0 and seq[-1] == 2000
    assert len(data['heldouts']['a_unseen']) == 129
    assert not bool(torch.isin(data['heldouts']['a_unseen'],seq).any())
    assert not bool(torch.isin(data['heldouts']['b_unseen'],seq).any())


def test_missing_real_continuation_is_rejected():
    with pytest.raises(ValueError,match='real-text'):
        pack_stream(torch.arange(256),torch.arange(1000),torch.arange(1000),
                    phase_tokens=256,validation_tokens=128)


def credit_report():
    report = dict(status='complete',checkpoint='source.pt',checkpoint_sha256='source',
        source_updates=491,dataset_revision='r',source_stream_sha256='prepared',
        executed_stream_sha256='executed',protocol=dict(comparison_kind='credit',updates_per_arm=6),
        source_checkpoint_unchanged=True,executed_code_unchanged=True,first_training_forward_exact=True,
        executed_optimizer_updates=12,arms={})
    for arm,ce in [('held',2.),('full',1.9)]:
        report['arms'][arm] = dict(status='complete',new_updates=6,train_tokens=768,checkpoint=arm+'.pt',
            phases={'X':ce},evaluations=[dict(step=6,phase='X',persistent_state_unchanged=True,
                metrics={'validation':{'nats_per_token':ce}})],
            steps=[dict(step=i+1,phase='X',ce=ce,sigreg=.1,diagnostics={},updates=492+i,
                nll_sum=ce*128,target_tokens=128,reads={'1':dict(query_count=128,unique_replays=2,
                    ages_tokens=[300,301])}) for i in range(6)])
    return report


def test_credit_summary_keeps_learning_and_query_counting_units_distinct():
    from scripts.ib_local.analyze_pretrained_gdn_credit_pair import summarize
    result = summarize(credit_report())
    assert result['full_minus_held']['stream_nll'] == pytest.approx(-.1)
    assert result['paired_16_window_blocks'][0]['last_step'] == 6
    assert result['paired_16_window_blocks'][0]['full_minus_held'] == pytest.approx(-.1)
    assert result['arms']['full']['layers']['2']['query_count'] == 768


def test_credit_summary_rejects_incomplete_or_changed_source():
    from scripts.ib_local.analyze_pretrained_gdn_credit_pair import summarize
    report = credit_report(); report['source_checkpoint_unchanged'] = False
    with pytest.raises(ValueError,match='integrity'):
        summarize(report)
