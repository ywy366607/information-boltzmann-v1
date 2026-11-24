"""Actual update/eligibility interfaces; numerical checks, no capability tasks."""
import copy
import pytest
import torch
from torch.nn import functional as F

from scripts.ib_local.contact_momentum import ContactMomentum, transition, configure


@pytest.mark.parametrize('budget', [0., 2.])
@pytest.mark.parametrize('dimension', [8, 32])
def test_fused_scan_matches_reference_with_complete_local_jvp(budget, dimension):
    if not torch.cuda.is_available():
        pytest.skip('CUDA kernel contract')
    torch.manual_seed(31)
    h,d,t=2,dimension,7
    k=F.normalize(torch.randn(1,t,h,d),dim=-1);v=torch.randn_like(k)
    a=torch.rand(1,t,h);b=torch.rand_like(a)
    s=torch.randn(1,h,d,d);p=torch.randn_like(s);z=torch.rand(1,h)+.3
    cpu=ContactMomentum(1,h,d,d)
    with torch.no_grad():
        cpu.phi.copy_(torch.tensor([[.2,-.4]]))
        cpu.sensitivity_s.normal_(0,.1);cpu.sensitivity_p.normal_(0,.1)
    gpu=copy.deepcopy(cpu).cuda()
    expected=cpu.observe(0,k,v,a,b,s,p,z,budget)
    actual=gpu.observe(0,*[x.cuda() for x in (k,v,a,b,s,p,z)],budget)
    for x,y in zip(expected[:2],actual[:2]):
        torch.testing.assert_close(x,y.cpu(),atol=3e-5,rtol=3e-5)
    for name in ['sensitivity_s','sensitivity_p']:
        torch.testing.assert_close(getattr(cpu,name),getattr(gpu,name).cpu(),atol=4e-5,rtol=4e-5)
    torch.testing.assert_close(expected[2],actual[2].cpu(),atol=4e-5,rtol=4e-5)


def test_local_eligibility_equals_unrolled_gate_derivative():
    torch.manual_seed(41)
    module=ContactMomentum(1,2,3,3).double()
    k=F.normalize(torch.randn(1,4,2,3,dtype=torch.float64),dim=-1);v=torch.randn_like(k)
    a=torch.rand(1,4,2,dtype=k.dtype);b=torch.rand_like(a)
    s=torch.randn(1,2,3,3,dtype=k.dtype);p=torch.randn_like(s);z=torch.ones(1,2,dtype=k.dtype)
    def unroll(phi):
        sn,pn=s,p
        for t in range(4):
            sn,pn=transition(sn,pn,k[:,t],v[:,t],a[:,t],b[:,t],z,phi,0.)
        return pn
    derivative=torch.func.jacfwd(unroll)(module.phi[0].detach()).movedim(-1,0)
    module.observe(0,k,v,a,b,s,p,z,0.)
    torch.testing.assert_close(derivative,module.sensitivity_p[:,0])


def test_complete_loop_delivers_contact_credit_and_restores(tmp_path):
    from test_joint_relation_update import body
    from scripts.ib_local.adaptive_hamiltonian import configure as adaptive
    engine,archive=body(tmp_path/'first',revision=4)
    try:
        adaptive(engine);configure(engine)
        module=engine.model.head_coupling.contact
        before=module.phi.detach().clone()
        for _ in range(3):
            row=engine.step(torch.randn(engine.model.encoder.out_features,4))
        assert module.observations.min()>0
        assert module.momentum_credit_count>0
        assert not torch.equal(before,module.phi)
        assert row['causal_program']['evidence_committed']
        pending=engine.internal_pending['momentum_read_eligibility']
        assert pending['schema']==1
        saved=copy.deepcopy(module.state_dict())
        # Live disk archive snapshots use the existing archive checkpoint contract;
        # module tensor persistence is checked without copying the whole organism.
        clone=ContactMomentum(module.phi.shape[0],module.sensitivity_s.shape[3],
                              module.sensitivity_s.shape[4],module.sensitivity_s.shape[5])
        clone.load_state_dict(saved)
        for k,v in saved.items():
            torch.testing.assert_close(v,clone.state_dict()[k],rtol=0,atol=0)
        assert not torch.equal(pending['contact'],torch.zeros_like(pending['contact']))
    finally:
        archive.close()


def test_complete_state_restore_repeats_next_learning_event():
    from test_generative_cycle import configured
    from test_joint_world_cycle import assert_same_saved_values
    from scripts.ib_local.adaptive_hamiltonian import configure as adaptive
    from scripts.ib_local.navigation_checkpoint import snapshot,restore
    torch.set_num_threads(1)
    engine=configured();adaptive(engine);configure(engine)
    for _ in range(2):
        engine.step(torch.randn(8,4))
    saved=snapshot(engine)
    expected=engine.step(torch.randn(8,4))
    weights=copy.deepcopy(engine.model.state_dict())
    optimizer=copy.deepcopy(engine.optimizer.state_dict())
    fresh=configured();adaptive(fresh);configure(fresh)
    restore(fresh,saved)
    actual=fresh.step(torch.randn(8,4))
    assert expected['metal_g']==actual['metal_g']
    assert expected['utterance_hex']==actual['utterance_hex']
    assert_same_saved_values(weights,fresh.model.state_dict())
    assert_same_saved_values(optimizer,fresh.optimizer.state_dict())
    from dataclasses import asdict, is_dataclass
    def plain(value):
        if is_dataclass(value):
            return plain(asdict(value))
        if isinstance(value,dict):
            return {k:plain(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):
            return type(value)(plain(v) for v in value)
        return value
    assert_same_saved_values(plain(engine.internal_pending),plain(fresh.internal_pending))
