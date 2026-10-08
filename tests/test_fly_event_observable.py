"""CPU operator/telemetry identities only; no capability or training claim."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_fly_bptt_learning import make_model, physical

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('fly_event_observable',ROOT/'scripts/ib/diagnose_fly_event_observable.py')
event=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(event)


def measured(model,learner,initial,tokens,folder,label,tape=None):
    trace=event.Trajectory(folder,label,model.n_read,model.n_neurons,
        model.output_read.out_features,model.decoder.out_features)
    learner.state=event.old.clone_state(initial)
    with torch.no_grad(),event.observe_arrays(model,trace,fixed=tape):
        scores,state,_=learner.forward_window(tokens[None],tokens[None])
    trace.validate(scores.numpy(),state)
    return trace


def test_native_batched_telemetry_and_all64_fixed_tape_shapes(tmp_path):
    torch.manual_seed(18)
    model=make_model(tmp_path,read_centering=False)
    learner=event.pipeline.FlyPipelineLearner(model,physical(model))
    initial=event.old.clone_state(learner.state)
    initial.h.copy_(torch.linspace(-.03,.06,model.n_neurons)[None])
    initial.ring=tuple(torch.ones_like(initial.h)*.2 for _ in range(4))
    tokens=torch.arange(32)%9
    learner.state=event.old.clone_state(initial)
    with torch.no_grad():
        plain,state,_=learner.forward_window(tokens[None],tokens[None])
    a=measured(model,learner,initial,tokens,tmp_path,'A')
    c=None
    try:
        np.testing.assert_array_equal(a.scores,plain.numpy())
        assert all(row['passed'] for rows in event.state_repeat(state,a.state).values() for row in rows)
        assert len(a.tape())==64
        assert [row.shape for row in a.tape()]==[(1,model.n_read),(1,model.n_neurons)]*32
        c=measured(model,learner,initial,tokens,tmp_path,'C',a.tape())
        assert event.repeat_gate(a,c)['passed']
        assert a.calls==c.calls==64 and a.commits==c.commits==32
        assert a.norm_calls==a.decoder_calls==1
        assert all(row['passed'] for tick in a.pulse_identity for row in tick.values())
        with torch.no_grad():
            native_z=model.read_norm(torch.from_numpy(a.arrays['r'].copy()))
            native_logits=model.decoder(native_z)
        np.testing.assert_array_equal(a.arrays['z'],native_z.numpy())
        np.testing.assert_array_equal(a.arrays['logits'],native_logits.numpy())
    finally:
        a.close()
        if c is not None: c.close()


def test_sealed_motor_observation_is_independent_of_current_token(tmp_path):
    torch.manual_seed(7)
    model=make_model(tmp_path,read_centering=False)
    learner=event.pipeline.FlyPipelineLearner(model,physical(model))
    initial=event.old.clone_state(learner.state)
    first=torch.arange(32)%9; second=first.clone(); second[0]=(second[0]+1)%9
    a=measured(model,learner,initial,first,tmp_path,'first')
    b=measured(model,learner,initial,second,tmp_path,'changed_current')
    try:
        for field in ('v','threshold','s','p','h','r','z','logits'):
            np.testing.assert_array_equal(a.arrays[field][0],b.arrays[field][0])
    finally:
        a.close(); b.close()


def test_repeat_gate_detects_intermediate_array_and_ring_instability(tmp_path):
    model=make_model(tmp_path,read_centering=False)
    learner=event.pipeline.FlyPipelineLearner(model,physical(model))
    initial=event.old.clone_state(learner.state); tokens=torch.arange(32)%9
    a=measured(model,learner,initial,tokens,tmp_path,'reference')
    b=measured(model,learner,initial,tokens,tmp_path,'repeat')
    try:
        assert event.repeat_gate(a,b)['passed']
        b.arrays['r'][7,0]+=1e-3
        assert not event.repeat_gate(a,b)['passed']
        b.arrays['r'][:]=a.arrays['r'][:]
        b.state.ring[3][0,0]+=1e-3
        assert not event.repeat_gate(a,b)['passed']
    finally:
        a.close(); b.close()


def test_fp64_exact_kl_alignment_and_both_centering_conventions():
    rng=np.random.default_rng(4)
    reference=rng.normal(size=(4,19))*30
    changed=reference+rng.normal(size=reference.shape)*.3
    targets=[0,2,7,11]
    result=event.probability_contrast(reference,changed,targets)
    shifted=event.probability_contrast(reference+71,changed+71+np.arange(4)[:,None]*4,targets)
    assert result['mean_risk_change']==pytest.approx(shifted['mean_risk_change'],abs=2e-13)
    for row in result['per_target']:
        assert row['risk_change']==pytest.approx(row['kl']-row['alignment'],abs=row['arithmetic_floor'])
        assert row['kl']>=-row['arithmetic_floor']
    assert result['common_linear_cost']+result['covariance_linear_cost']==pytest.approx(-result['mean_alignment'],abs=1e-14)
    assert result['shared_only_retrospective_risk_change']+result['ordered_residual_with_interactions']==pytest.approx(result['mean_risk_change'])
    shared=event.probability_contrast(np.zeros((3,4)),np.tile([.3,0,0,0],(3,1)),[0,0,0])
    assert shared['mean_delta_norm']>0
    assert shared['centered_time_delta_energy']<1e-30
    assert shared['mean_risk_change']<0


def test_rms_finite_endpoint_and_local_radial_prediction_are_separate():
    a=np.array([[1.,2.,3.],[0.,0.,0.]])
    b=np.array([[2.,4.,6.],[.3,-.2,.1]])
    metadata={'gain':[.1,.2,.3],'epsilon':1e-5}
    result=event.rms_geometry(a,b,metadata)
    gain=np.asarray(metadata['gain'])
    expected=gain*(b/np.sqrt((b*b).mean(1,keepdims=True)+1e-5)-a/np.sqrt((a*a).mean(1,keepdims=True)+1e-5))
    np.testing.assert_allclose(result['finite'],expected,rtol=0,atol=0)
    assert result['per_target']['radial_fraction'][0]==pytest.approx(1.)
    assert result['per_target']['local_J_finite_remainder_norm'][1]>0


def test_small_field_floors_and_reconstruction_are_relative():
    assert event.precision_floor(1e-6)==pytest.approx(64*event.EPS*1e-6)
    assert event.precision_floor(1e-6,1e-9)==pytest.approx(1e-8)
    passed,floor=event.comparison_floor(1.001e-5,1e-5)
    assert not passed and floor<1e-9
    passed,_=event.comparison_floor(454916,454915)
    assert not passed
    error=event.array_error(np.array([1e-6],np.float32),np.array([1.01e-6],np.float32))
    assert not error['passed']


def test_exact416_budget_no_second_update_or_role_expansion():
    ledger=event.Budget()
    ledger.charge('reconstruct_observe',observe=True)
    for role in ('A','B','C'):
        for _ in range(4): ledger.charge(role)
    ledger.finish()
    assert ledger.value['physical_ticks']==416
    assert ledger.value['backwards']==ledger.value['disposable_optimizer_updates']==1
    with pytest.raises(ValueError): ledger.charge('B')
    with pytest.raises(ValueError): ledger.charge('reconstruct_observe',observe=True)
    incomplete=event.Budget()
    with pytest.raises(ValueError): incomplete.finish()


def test_failure_json_handles_numpy_and_nonfinite_without_retry(tmp_path):
    path=tmp_path/'partial.json'
    event.write_json(path,{'complete':False,'auto_retry':False,'count':np.int64(32),
                          'bad':np.float64(np.nan),'passed':np.bool_(False)})
    result=json.loads(path.read_text(encoding='utf8'))
    assert result['bad']=={'invalid_numeric':'nan'} and result['count']==32
    assert result['complete'] is False and result['auto_retry'] is False


def test_observer_restores_globals_and_hooks_on_failure(tmp_path):
    model=make_model(tmp_path,read_centering=False)
    trace=event.Trajectory(tmp_path,'abort',model.n_read,model.n_neurons,3,9)
    spike=event.pipeline.SpikeFn; begin=event.pipeline.begin_fly_prediction
    try:
        with pytest.raises(RuntimeError,match='intentional'):
            with event.observe_arrays(model,trace):
                raise RuntimeError('intentional')
        assert event.pipeline.SpikeFn is spike
        assert event.pipeline.begin_fly_prediction is begin
        assert len(model.decoder._forward_hooks)==len(model.read_norm._forward_hooks)==0
    finally: trace.close()
