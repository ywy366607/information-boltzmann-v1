import copy
import pytest

from scripts.ib_local.delayed_squad_protocol import answer_scores, make_episodes, validate_recovery


def paragraphs():
    return [dict(title=f'article{i}', context=f'Actual annotated context{i}',
                 qa=[dict(id=f'{i}_{j}', question=f'Question{i}_{j}?', answer=f'value{i}_{j}',
                          answers=[f'value{i}_{j}']) for j in range(4)]) for i in range(4)]


def test_new_questions_and_return_do_not_reintroduce_evidence():
    episodes = make_episodes(paragraphs())
    acquired = {e['qa_id'] for e in episodes if e['stage'].endswith(('acquire','interfere'))}
    for e in episodes:
        assert e['answer'] not in e['prompt']
        assert e['question'] not in e['feedback_prefix'] if 'question' in e else True
        if e['stage'].endswith('transfer'):
            assert e['qa_id'] not in acquired and e['new_question']
        if e['stage'] not in ('A_acquire','B_interfere'):
            assert 'Passage:' not in e['prompt'] and not e['present_context']
    assert [e['stage'] for e in episodes][::2] == [
        'A_acquire','A_before_B','B_interfere','B_before_A','A_retention','A_relearn1','A_relearn2',
        'A_transfer','B_retention','B_transfer']
    assert all(not e['provide_correction'] for e in episodes
               if e['stage'].endswith(('retention','transfer')))


def test_balanced_delays_on_both_groups_and_each_stage():
    episodes = make_episodes(paragraphs())
    for stage in set(e['stage'] for e in episodes):
        assert {e['feedback_delay_tokens'] for e in episodes if e['stage']==stage} == {256,512}


def test_qa_score_checks_content_not_just_answer_shaped_output():
    assert answer_scores('The red fox\nextra generated text', ['red fox']) == dict(exact_match=1.,token_f1=1.)
    assert answer_scores('7470251',['red fox']) == dict(exact_match=0.,token_f1=0.)
    assert answer_scores('red',['red fox'])['token_f1'] == 2/3


def test_recovery_does_not_splice_unsaved_later_cases_or_feedback():
    cases=[dict(index=i,gap_token_cursor=[i*384,i*384+256],end_event=500+i,updates=10+i,seen=100+i)
           for i in range(9)]
    prior=dict(arm='mismatch',task_sha256='task',requested_episode_count=64,cases=cases,windows=[1,2,3])
    state=dict(delayed_squad_driver=dict(arm='mismatch',task_sha256='task',episodes_completed=8,
               gap_cursor=cases[7]['gap_token_cursor'][1]+128,history_chain='held',windows_completed=2),
               stream=dict(events=507,chain='held',targets=[1],delayed_feedback_positions=[True]),
               learner=dict(updates=17,cache=dict(seen_tokens=107),native_s_sidecar=dict(mode='mismatch')))
    result=validate_recovery(state,prior,task_sha256='task',arm='mismatch',episodes=64)
    assert len(result['cases'])==8 and len(result['windows'])==2
    changed=copy.deepcopy(state);changed['learner']['native_s_sidecar']['mode']='raw'
    with pytest.raises(ValueError,match='pending feedback'):
        validate_recovery(changed,prior,task_sha256='task',arm='mismatch',episodes=64)
    changed=copy.deepcopy(state);changed['stream']['delayed_feedback_positions']=[]
    with pytest.raises(ValueError,match='pending feedback'):
        validate_recovery(changed,prior,task_sha256='task',arm='mismatch',episodes=64)
