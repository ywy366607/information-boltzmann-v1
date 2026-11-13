"""Data/interface integrity only; not an invented language capability test."""
import pytest

from scripts.ib_local.run_pretrained_gdn_long_sft import select_papers, assert_first_forward_equal


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


def payload():
    def qa(qid, question, span):
        return dict(question_id=qid,question=question,answers=[dict(answer=dict(
            unanswerable=False,extractive_spans=[span]))])
    return {'paper': dict(title='A title',abstract='Alpha and beta appear here.',
        full_text=[dict(section_name=None,paragraphs=['Real text.'])],
        qas=[qa('one','First?', 'Alpha'),qa('two','Second?', 'beta'),
             qa('missing','Absent?', 'Not in the context')])}


def test_real_annotation_filter_handles_missing_heading_and_preserves_spans():
    rows = select_papers(payload(),CharacterTokenizer(),1,minimum=8,maximum=100)
    assert len(rows) == 1
    assert [q['id'] for q in rows[0]['qa']] == ['one','two']
    for qa in rows[0]['qa']:
        assert rows[0]['context'][qa['answer_start']:].startswith(qa['answer'])
    assert 'None' not in rows[0]['context']


def test_context_length_is_not_filled_with_unobserved_or_repeated_material():
    with pytest.raises(ValueError,match='only 0 eligible'):
        select_papers(payload(),CharacterTokenizer(),1,minimum=101,maximum=200)


def test_answer_outside_retained_prefix_cannot_be_used_as_a_label():
    data = payload()
    data['paper']['qas'][1]['answers'][0]['answer']['extractive_spans'] = ['Real text.']
    with pytest.raises(ValueError,match='only 0 eligible'):
        select_papers(data,CharacterTokenizer(),1,minimum=8,maximum=35)


def test_held_credit_gate_rejects_forward_or_reference_drift():
    import copy
    row=dict(ce=2.,nll_sum=256.,sigreg=.1,reads={'1':dict(slots=[1,2],weights=[.3,.7])},
             supervised_tokens=3,supervised_nll_sum=6.)
    assert_first_forward_equal(row,copy.deepcopy(row),copy.deepcopy(row))
    changed=copy.deepcopy(row)
    changed['reads']['1']['weights']=[.4,.6]
    with pytest.raises(AssertionError,match='held intervention'):
        assert_first_forward_equal(row,changed,row)
    changed=copy.deepcopy(row)
    changed['ce']+=1e-7
    with pytest.raises(AssertionError,match='reference'):
        assert_first_forward_equal(row,row,changed)
