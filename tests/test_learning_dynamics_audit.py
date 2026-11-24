"""Observer invariance and exact sampled Adam displacement; no training study."""
import copy
import json
from types import SimpleNamespace

import torch
from torch import nn
import pytest

from scripts.ib_local.learning_dynamics_audit import LearningDynamicsAudit


class TupleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4,4)

    def forward(self,x):
        return x+self.proj(x).tanh(),None,{}


class AuditModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Module()
        self.body.base = nn.Module()
        self.body.base.layers = nn.ModuleList([TupleLayer(),TupleLayer()])

    def objective(self,x,y):
        for layer in self.body.base.layers:
            x=layer(x)[0]
        return (x-y).square().mean()


def test_observer_preserves_actual_adam_updates_gradients_and_rng(tmp_path):
    torch.manual_seed(47)
    model=AuditModel()
    reference=copy.deepcopy(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    reference_optimizer=torch.optim.AdamW(reference.parameters(),lr=.01)
    x,y=torch.randn(7,4),torch.randn(7,4)
    original_objective=model.objective
    rng=torch.get_rng_state().clone()
    report_path=tmp_path/'audit.json'
    audit=LearningDynamicsAudit(model,optimizer,report_path,2)
    assert torch.equal(torch.get_rng_state(),rng)
    for cycle in range(3):
        before=model.body.base.layers[0].proj.weight.detach().clone()
        for m,o in ((model,optimizer),(reference,reference_optimizer)):
            o.zero_grad(set_to_none=True)
            m.objective(x,y).backward()
            o.step()
        if audit.active:
            audit.finish_event(cycle,dict(external_bytes=0))
        for a,b in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
        if cycle==0:
            expected=(model.body.base.layers[0].proj.weight-before).square().mean().sqrt().item()
            report=json.loads(report_path.read_text())
            measured=report['layer_summary'][0]['families']['proj.weight']['sampled_delta_rms']
            assert measured==expected
    assert not audit.active
    assert model.objective==original_objective
    assert torch.equal(torch.get_rng_state(),rng)
    report=json.loads(report_path.read_text())
    assert report['status']=='complete' and report['observed_events']==2
    assert len(report['events'][0]['traffic'])==2
    assert report['events'][0]['traffic'][0]['output_gradient_rms']


def test_unused_parameters_are_not_reported_as_updated(tmp_path):
    model=AuditModel()
    model.unused=nn.Parameter(torch.ones(9))
    optimizer=torch.optim.AdamW(model.parameters())
    audit=LearningDynamicsAudit(model,optimizer,tmp_path/'audit.json',1)
    model.objective(torch.ones(1,4),torch.zeros(1,4)).backward()
    optimizer.step()
    audit.finish_event(1,{})
    names={r['name'] for rows in audit.events[0]['updates'] for r in rows}
    assert 'unused' not in names
    assert not audit.active


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA Adam state admission')
def test_cuda_adam_cpu_step_counter_is_observed_on_the_right_device(tmp_path):
    model=AuditModel().cuda()
    optimizer=torch.optim.AdamW(model.parameters())
    x,y=torch.ones(1,4,device='cuda'),torch.zeros(1,4,device='cuda')
    model.objective(x,y).backward();optimizer.step();optimizer.zero_grad()
    assert next(iter(optimizer.state.values()))['step'].device.type=='cpu'
    audit=LearningDynamicsAudit(model,optimizer,tmp_path/'audit.json',1)
    model.objective(x,y).backward();optimizer.step()
    audit.finish_event(1,{})
    assert not audit.active
