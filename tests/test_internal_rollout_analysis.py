from scripts.ib_local.analyze_internal_rollout import summarize


def test_report_uses_saved_prequential_forecasts_and_keeps_empty_support_explicit():
    p=dict(next_boundary_prequential_mse=2.,unchanged_state_prequential_mse=3.,
           other_action_prequential_mse=4.,settled_two_turn=dict(
               target=2.,baseline=0.,prediction=1.,uncalibrated=3.,eligible=True),
           delivered_offset=[.1,0.,-.1],choice_probability_tv=.02,valid_boundary_pairs=5)
    row=dict(cycle=2310,choice='generate',external_bytes=960,metal_g=.1,
             world=dict(mother_bytes=100),internal_rollout=p,updates=4,response_nll=1.5)
    result=summarize([row])
    assert result['settled_mse']==dict(baseline=4.,prediction=1.,uncalibrated=1.)
    assert result['optimizer_updates']==4
    assert result['paired_boundary_mse']==2.
    assert result['unchanged_state_mse']==3.
    assert result['delivered_events']==1
    empty=summarize([])
    assert empty['settled_mse']['baseline'] is None
    assert empty['paired_boundary_count']==0
