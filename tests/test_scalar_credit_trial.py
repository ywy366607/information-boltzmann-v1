import numpy as np
import pytest
import torch

from scripts.ib_local.scalar_credit_trial import select_scalar_credit, shadow_adam_comparison, upgrade_legacy_value_map_optimizer


def test_legacy_optimizer_inserts_value_map_without_moving_old_moments():
    class Memory(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=torch.nn.Linear(3,2)
            self.value_map=torch.nn.Linear(3,2)
            self.inlet=torch.nn.Linear(2,3,bias=False)
            self.null_key=torch.nn.Parameter(torch.ones(2))
            self.null_bias=torch.nn.Parameter(torch.zeros(()))
    old={'param_groups':[{'params':[0,1]},{'params':[2,3,4,5,6,7]},
                         {'params':[8,9,10,11]}],
         'state':{i:{'step':torch.tensor(float(i))} for i in range(12)}}
    result=upgrade_legacy_value_map_optimizer(old,{1:Memory(),12:Memory()})
    assert result['param_groups'][1]['params']==[2,3,12,13,4,5,6,14,15,7]
    assert set(result['state'])==set(old['state'])
    assert result['param_groups'][0]==old['param_groups'][0]
    assert old['param_groups'][1]['params']==[2,3,4,5,6,7]


def test_controls_change_only_strict_old_and_preserve_mismatch_norm():
    old={'layer1.A_log':np.array([1.,2.,3.,4.],dtype=np.float32),
         'layer12.A_log':np.array([5.,6.,7.,8.],dtype=np.float32)}
    raw=select_scalar_credit(old,'raw')
    cut=select_scalar_credit(old,'cut_old')
    wrong=select_scalar_credit(old,'mismatch_old')
    for name,value in old.items():
        np.testing.assert_array_equal(raw[name],value)
        assert not cut[name].any()
        assert np.linalg.norm(wrong[name])==np.linalg.norm(value)
        assert not np.array_equal(wrong[name],value)
        assert raw[name] is not value
    with pytest.raises(ValueError):
        select_scalar_credit(old,'unknown')


def test_shadow_adam_keeps_original_parameter_and_moments():
    a=torch.nn.Parameter(torch.tensor([.2,.4,.1,.8]))
    b=torch.nn.Parameter(torch.tensor([.3,.1,.5,.7]))
    opt=torch.optim.AdamW([a,b],lr=.01,weight_decay=0.,foreach=False)
    a.grad=torch.tensor([.2,.1,.3,.4]);b.grad=torch.ones(4)
    opt.step();opt.zero_grad(set_to_none=True)
    before_a=a.detach().clone();before_avg=opt.state[a]['exp_avg'].clone()
    selected={'layer1.A_log':(a,a),'layer12.A_log':(b,b)}
    old={'layer1.A_log':np.array([.02,0,0,0],np.float32),
         'layer12.A_log':np.zeros(4,np.float32)}
    result=shadow_adam_comparison(opt,selected,[torch.ones(4),torch.ones(4)],old)
    assert result['raw']['layer1.A_log']!=result['cut_old']['layer1.A_log']
    assert torch.equal(a,before_a)
    assert torch.equal(opt.state[a]['exp_avg'],before_avg)
