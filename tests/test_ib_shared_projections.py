import pytest
import copy
import torch
from fine_grain.information_boltzmann import InformationBoltzmann
from fine_grain.information_boltzmann.streaming import StreamRunner
from scripts.ib_shared_projections import install_shared_projections


def test_shared_projections_values_and_nonzero_gradients():
    torch.manual_seed(13);torch.set_num_threads(1)
    a=InformationBoltzmann(vocab_size=12,phase_dim=2,particles=8,hidden_dim=16,flow_layers=2,steps=2).double()
    b=install_shared_projections(copy.deepcopy(a))
    x=torch.randn(8,2,dtype=torch.double)
    left=a.force.drive(x,3,.25);right=b.force.drive(x,3,.25)
    torch.testing.assert_close(left,right,rtol=1e-10,atol=1e-12)
    left.square().sum().backward();right.square().sum().backward()
    assert a.force.net[0].weight.grad.norm()>1e-6
    for pa,pb in zip(a.parameters(),b.parameters()):
        if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad,rtol=1e-9,atol=1e-11)
    a.zero_grad();b.zero_grad()
    v=torch.randn(8,2,dtype=torch.double);normal=torch.tensor([.6,.8],dtype=torch.double)
    args=(x[0],v[0],v[1],normal,x,v)
    left=a.collision.rate(*args);right=b.collision.rate(*args)
    torch.testing.assert_close(left,right,rtol=1e-10,atol=1e-12)
    left.backward();right.backward()
    for pa,pb in zip(a.parameters(),b.parameters()):
        if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad,rtol=1e-9,atol=1e-11)


@pytest.mark.parametrize('checkpoint_drive',[False,True])
def test_shared_projection_online_state_updates_and_rng(checkpoint_drive):
    torch.manual_seed(13);torch.set_num_threads(1)
    a=InformationBoltzmann(vocab_size=12,phase_dim=2,particles=8,hidden_dim=16,flow_layers=2,steps=2,temperature=.1).double()
    b=install_shared_projections(copy.deepcopy(a),checkpoint_drive=checkpoint_drive)
    ra=StreamRunner(a,1,seed=17,optimizer=torch.optim.Adam(a.parameters()),update_every=2)
    rb=StreamRunner(b,1,seed=17,optimizer=torch.optim.Adam(b.parameters()),update_every=2)
    for token in [2,3,4,5]:
        torch.testing.assert_close(ra.predict(),rb.predict(),rtol=1e-8,atol=1e-10)
        ra.observe(token);rb.observe(token)
        torch.testing.assert_close(ra.state.x,rb.state.x,rtol=1e-8,atol=1e-10)
        torch.testing.assert_close(ra.state.v,rb.state.v,rtol=1e-8,atol=1e-10)
        assert torch.equal(ra.generator.get_state(),rb.generator.get_state())
        assert b.force._shared_token_projection is None and b.collision._shared_context is None
    for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa,pb,rtol=1e-8,atol=1e-10)


def test_shared_projection_deepcopy_isolation():
    torch.manual_seed(8)
    original=install_shared_projections(InformationBoltzmann(vocab_size=12,phase_dim=2,particles=8,hidden_dim=16,flow_layers=2,steps=2).double())
    replica=copy.deepcopy(original)
    with torch.no_grad():replica.force.net[0].bias.add_(.5)
    reference=InformationBoltzmann(vocab_size=12,phase_dim=2,particles=8,hidden_dim=16,flow_layers=2,steps=2).double()
    reference.load_state_dict(replica.state_dict())
    ra=StreamRunner(replica,1,seed=23)
    rb=StreamRunner(reference,1,seed=23)
    torch.testing.assert_close(ra.predict(),rb.predict(),rtol=1e-9,atol=1e-11)
    assert original.force._shared_token_projection is None
    assert original.collision._shared_context is None
    assert all(p.grad is None for p in original.parameters())
