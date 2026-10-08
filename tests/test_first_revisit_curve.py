import copy
from information_boltzmann.runtime.forgetting_curve import FirstRevisitCurve


def test_independent_episodes_rotated_assignment_real_gap_and_bridge():
    curve=FirstRevisitCurve([4,12],episode_tokens=4,cohort_every=8)
    for cursor in range(1,17):
        curve.fresh_observation(cursor,float(cursor),event=cursor,train_cursor=cursor,updates=cursor//4)
    assert len(curve.pending)==2
    episode=curve.pop_due(16)
    assert episode['tokens']==[9,10,11,12]
    assert episode['target_lag']==4
    row=curve.record_revisit(episode,[99.,9.,10.,11.],event_start=16,event_end=20,updates_start=4,updates_end=5)
    assert row['actual_intervening_events']==4
    assert row['initial_nll']==11
    assert row['first_revisit_nll']==10
    assert row['revisit_bridge_nll']==99
    assert curve.pop_due(20) is None
    assert curve.curve()['points'][0]['episodes']==1
    assert curve.curve()['points'][0]['forgetting_nll_gap']==-1


def test_measurement_resume_keeps_partial_capture_and_pending_episode():
    original=FirstRevisitCurve([3,10],episode_tokens=4,cohort_every=8)
    for cursor in range(1,15): original.fresh_observation(cursor,cursor*.1,event=cursor,train_cursor=cursor,updates=cursor//4)
    resumed=FirstRevisitCurve([3,10],episode_tokens=4,cohort_every=8)
    resumed.load_state_dict(copy.deepcopy(original.state_dict()))
    for cursor in range(15,20):
        for curve in (original,resumed): curve.fresh_observation(cursor,cursor*.1,event=cursor,train_cursor=cursor,updates=cursor//4)
    assert original.state_dict()==resumed.state_dict()
    first=original.pop_due(30); second=resumed.pop_due(30)
    assert first==second
    assert original.record_revisit(first,[1.,2.,3.,4.],event_start=30,event_end=34,updates_start=7,updates_end=8)==resumed.record_revisit(second,[1.,2.,3.,4.],event_start=30,event_end=34,updates_start=7,updates_end=8)


def test_interrupted_initial_exposure_is_preserved_but_excluded():
    curve=FirstRevisitCurve([4],episode_tokens=4,cohort_every=8)
    for cursor,event in zip(range(9,13),[9,10,15,16]):
        curve.fresh_observation(cursor,1.,event=event,train_cursor=cursor,updates=event//4)
    episode=curve.pop_due(20)
    row=curve.record_revisit(episode,[1.,2.,3.,4.],event_start=20,event_end=24,updates_start=5,updates_end=6)
    assert row['source_contiguous'] is False
    assert len(row['initial_curve'])==4
    assert curve.curve()['points'][0]['episodes']==0
    assert curve.excluded_interrupted_sources==1
    curve.rebuild_summary([row],through_event=24)
    assert curve.excluded_interrupted_sources==1
    curve.rebuild_summary([row],through_event=20)
    assert curve.excluded_interrupted_sources==0


def test_source_start_is_the_actual_first_event_after_a_boundary_revisit():
    curve=FirstRevisitCurve([4,12],episode_tokens=4,cohort_every=8)
    for cursor in range(9,13):
        curve.fresh_observation(cursor,1.,event=cursor,train_cursor=cursor,updates=cursor//4)
    for cursor in range(13,17):
        curve.fresh_observation(cursor,1.,event=cursor+8,train_cursor=cursor,updates=(cursor+8)//4)
    second=curve.pending[1]
    assert second['source_start_event']==21
    assert second['source_end_event']==24
    row=curve.record_revisit(second,[1.,2.,3.,4.],event_start=36,event_end=40,updates_start=9,updates_end=10)
    assert row['source_contiguous'] is True


def test_legacy_boundary_start_is_recovered_from_executed_revisit():
    curve=FirstRevisitCurve([4,12],episode_tokens=4,cohort_every=8)
    def record(start,end,revisit_start,revisit_end,lag):
        return dict(source_event_interval=[start,end],revisit_event_interval=[revisit_start,revisit_end],
            initial_curve=[1.]*4,target_lag_events=lag,forgetting_nll_gap=.5,
            opening_16_forgetting_gap=.5,initial_nll=1.,first_revisit_nll=1.5,
            actual_intervening_events=revisit_start-end-1)
    rows=[record(1,4,9,12,4),record(9,16,29,32,12)]
    curve.rebuild_summary(rows,through_event=32)
    assert curve.excluded_interrupted_sources==0
    assert curve.corrected_boundary_starts==1
    assert curve.curve()['points'][1]['episodes']==1
