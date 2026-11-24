import torch
import pytest

from scripts.ib_local.self_action_observer import FrozenActionForecast, attach_self_action_observer


def test_passive_snapshot_survives_later_learning_and_preserves_rng():
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.encoder.weight.copy_(torch.eye(2))
    dist = dict(mean=torch.tensor([[1., 0.], [0., 1.]]),
                base=torch.tensor([[.5, .5], [.5, .5]]),
                scale2=torch.ones(2, 2), df=4.)
    actions = torch.eye(2)
    rng = torch.get_rng_state().clone()
    observer = FrozenActionForecast.capture(model, dist, actions)
    assert torch.equal(rng, torch.get_rng_state())
    with torch.no_grad():
        model.encoder.weight.zero_()
        dist['mean'].zero_()
    result = observer.score(torch.tensor([1., 0.]), 0)
    assert result['relation_mse'] == [0., 1.]
    assert result['original_margin'] == 0.
    assert result['relation_margin'] == 1.
    assert result['relation_log_margin'] > 0
    assert torch.equal(rng, torch.get_rng_state())


def test_duplicate_actions_are_not_counted_as_independent_alternatives():
    observer = FrozenActionForecast(torch.zeros(3, 2), torch.zeros(3, 2),
        torch.ones(3, 2), torch.eye(2), torch.tensor([[1., 0.], [1., 0.], [0., 1.]]), 4.)
    result = observer.score(torch.zeros(2), 0)
    assert result['different_alternatives'] == [2]
    with pytest.raises(ValueError, match='outside'):
        observer.score(torch.zeros(2), 3)


def test_observer_keeps_forward_return_sampling_and_execution_unchanged(monkeypatch):
    from types import SimpleNamespace
    from scripts.ib_local import navigation_joint_cycle as joint
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.encoder.weight.copy_(torch.eye(2))
    make = lambda n: dict(mean=torch.arange(n*2, dtype=torch.float32).reshape(n, 2),
                          base=torch.zeros(n, 2), scale2=torch.ones(n, 2), df=4.)
    options, drafts = make(3), make(4)
    forward_result = object()
    engine = SimpleNamespace(model=model, _program_mother_forecast=options,
        _mother_pre_base=options['base'], _internal_action=options['mean'],
        _mother_boundary_target=torch.tensor([2., 3.]), _pending_record={'choice':1},
        _joint_selection={'selected':2},
        _joint_proposals=[{'words':bytes([i])} for i in range(4)])
    calls = []
    def imagine(*args, **kwargs):
        calls.append('imagine')
        return {'path':{'mother':drafts}}
    monkeypatch.setattr(joint, 'imagine', imagine)
    def value():
        calls.append('value')
        return forward_result
    def step():
        assert engine._value() is forward_result
        joint.imagine(model, None, None, drafts['mean'])
        return dict(choice='generate',utterance_hex=b'\x02'.hex(),
                    causal_program={'mother_prequential':{'mse':0.}})
    engine._value, engine.step = value, step
    rng, weight = torch.get_rng_state().clone(), model.encoder.weight.detach().clone()
    attach_self_action_observer(engine)
    row = engine.step()
    assert calls == ['value','imagine']
    assert row['self_action_observer']['actual_score_reproduction_error'] == 0.
    assert row['self_action_observer']['generated_drafts']['chosen'] == 2
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(weight, model.encoder.weight)


def test_passive_callbacks_are_excluded_from_real_streaming_checkpoint(tmp_path, monkeypatch):
    import gzip
    from types import SimpleNamespace
    import numpy as np
    from scripts.ib_local import navigation_joint_cycle as joint
    from scripts.ib_local import run_navigation_residual_wood as runtime
    from scripts.ib_local.mother_checkpoint import save
    monkeypatch.setattr(runtime, '_DERIVED', set(runtime._DERIVED))
    monkeypatch.setattr(joint, 'imagine', joint.imagine)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(2, 2, bias=False)
    table = SimpleNamespace(**{name:np.zeros((1,256) if name=='rows' else (1,),np.float32)
        for name in ('rows','count','tag','err','epoch','last_write')}, history=0,events=0,
        orders=(0,),heads=1,slots_log2=1,gain_floor=.01,err_floor=.02,prior_weight=.05)
    engine = SimpleNamespace(model=model, optimizer=torch.optim.AdamW(model.parameters()),
        table=table,world=SimpleNamespace(state_dict=lambda:{}),
        _value=lambda:None,step=lambda:None,pending=7)
    attach_self_action_observer(engine)
    save(engine,tmp_path,123,{})
    with gzip.open(tmp_path/'last.pt.gz','rb') as stream:
        state=torch.load(stream,map_location='cpu',weights_only=False)
    assert state['cycle']==123
    assert state['engine']=={'pending':7}
