import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('analysis',Path(__file__).parents[1]/'scripts/ib_local/analyze_delayed_squad.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture(root):
    stages=['A_acquire','A_before_B','B_interfere','B_before_A','A_retention','A_relearn1',
            'A_relearn2','A_transfer','B_retention','B_transfer']
    for arm in ('raw','cut','mismatch'):
        cases=[]
        for stage in stages:
            for i in range(8):
                origin=len(cases)*1000
                transfer=stage.endswith('transfer')
                group=stage[0]
                cases.append(dict(stage=stage,qa_id=f'{group}{i}_{int(transfer)}',
                    context_reintroduced=stage in ('A_acquire','B_interfere'),
                    correction_provided=stage in ('A_acquire','B_interfere','A_relearn1','A_relearn2'),
                    first_corrective_target_distance=270 if stage in ('A_acquire','B_interfere','A_relearn1','A_relearn2') else None,
                    exact_match=.25,token_f1=.5,first_answer_token_nll=2.,feedback_delay_tokens=256 if i%2==0 else 512,
                    origin_events=origin,end_event=origin+1000,emission_end_event=origin+100,
                    feedback_start_event=origin+100+(256 if i%2==0 else 512),
                    study_qa_feedbacks_before=2 if stage=='A_relearn2' else 1,updates=len(cases)))
        report=dict(status='complete',episodes_completed=80,cases=cases,end_events=80000,
                    source_events=0,source_seen=100,end_seen=80100)
        for key in ('task_sha256','source_sha256','historical_runtime_files','execution_sources','numeric_runtime',
                    'frozen_mlp_precision','model_source_sha256','ordinary_teaching'):report[key]='same'
        (root/arm).mkdir();(root/arm/'report.json').write_text(json.dumps(report),encoding='utf-8')


def test_matched_questions_and_preinterference_format_are_explicit(tmp_path):
    fixture(tmp_path)
    result=module.analyze(tmp_path)
    trajectory=result['matched_question_trajectories']['raw']['A']
    assert len(trajectory['cases'])==8
    assert trajectory['mean_nll_forgetting']==0
    assert trajectory['cases'][0]['relearning'][1]['feedbacks_before']==2


def test_reset_or_reintroduced_evidence_invalidates_retention(tmp_path):
    fixture(tmp_path)
    path=tmp_path/'raw/report.json'
    value=json.loads(path.read_text());value['cases'][8]['context_reintroduced']=True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='leaked'):module.analyze(tmp_path)
    value['cases'][8]['context_reintroduced']=False;value['cases'][8]['origin_events']=0
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='reset'):module.analyze(tmp_path)
