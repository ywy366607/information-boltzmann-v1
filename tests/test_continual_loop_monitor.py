"""JSONL interface checks; no model, learning, or synthetic capability task."""
import json
import pytest
from scripts.monitor_continual_loop import EventTail, read_rows
from scripts.analyze_complete_optimizer_loop import unknown_lineage


def test_live_tail_defers_partial_line_and_bounds_memory(tmp_path):
    path=tmp_path/'events.jsonl'
    rows=[dict(cycle=i,external_bytes=960,updates=4) for i in range(4)]
    rows[2]['fixed_validation']={'owt_nll':1.5}
    path.write_text(json.dumps(rows[0])+'\n'+json.dumps(rows[1])[:10],encoding='utf-8')
    reader=EventTail(path,limit=2)
    try:
        assert reader.poll()==rows[:1]
        assert reader.events==1
        with path.open('a',encoding='utf-8') as f:
            f.write(json.dumps(rows[1])[10:]+'\n')
            f.write('\n'.join(json.dumps(x) for x in rows[2:])+'\n')
        assert reader.poll()==rows[-2:]
        assert reader.events==4 and reader.external_bytes==3840 and reader.updates==16
        assert reader.last_validation_cycle==2
        assert reader.poll()==rows[-2:]
        assert reader.events==4
    finally:reader.close()


def test_monitor_rejects_corruption_inside_history(tmp_path):
    path=tmp_path/'events.jsonl'
    path.write_text('{}\nwrong\n{}\n',encoding='utf-8')
    with pytest.raises(json.JSONDecodeError):read_rows(path)


def test_unknown_lineage_checks_current_reads_before_same_event_revision():
    # Receipt-schema check, not a model/learning capability experiment.
    rows = [dict(cycle=10, since_birth_external_bytes=960,
        shared_unknown=dict(identity='q', version=0, relation_identity=1,
                            revision=dict(committed=True, version_after=1))),
        dict(cycle=12, since_birth_external_bytes=2880,
        shared_unknown=dict(identity='q', version=1, relation_identity=2,
                            revision=dict(committed=True, version_after=2)),
        joint_relation_update=dict(variable_uses=[dict(identity='q',version=1,seen_version=0)]))]
    result=unknown_lineage(rows)
    assert result['version_regressions']==[]
    assert result['observed_selected_reactivations']==1
    assert result['longest_selected_gaps'][0]['selected_gap_external_bytes']==1920
    assert result['longest_selected_gaps'][0]['prior_revision']==1
    # A dependent really using the old version is different from an old
    # registration stamp or a protected historical pending snapshot.
    rows.append(dict(cycle=13,joint_relation_update=dict(
        variable_uses=[dict(identity='q',version=1,seen_version=0)])))
    failures=unknown_lineage(rows)['version_regressions']
    assert failures==[dict(cycle=13,identity='q',port='dependent',
                           actual_version=1,prior_revision=2)]


def test_unknown_lineage_does_not_infer_unobserved_history():
    result=unknown_lineage([dict(cycle=400,shared_unknown=dict(
        identity='old',version=7,relation_identity=2))])
    assert result['version_regressions']==[]
    assert result['observed_selected_reactivations']==0
    assert not result['capability_claim']
