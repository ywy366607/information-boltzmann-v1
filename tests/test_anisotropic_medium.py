"""Constitutive/numerical contracts, not capability experiments."""
from dataclasses import replace
import math
import copy
import torch
from information_boltzmann.core.plastic_medium import PlasticMedium3D
from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.continuous import ContinuousStream


def medium(shape=(4, 4, 2), tensor=True):
    torch.manual_seed(19)
    net = PlasticMedium3D(shape, channels=4, material_width=3, hidden=6,
                          anisotropic_transport=tensor).double()
    with torch.no_grad():
        net.material.coefficients.normal_(std=.1)
        if tensor:
            net.transport_shear.weight.normal_(std=.2)
            net.transport_shear.bias.copy_(torch.tensor([.3, -.2, .4]))
    return net


def state(net):
    s = net.initial_state(2)
    return replace(s, field=torch.randn_like(s.field),
                   flux=tuple(torch.randn_like(x) for x in s.flux))


def test_zero_shear_recovers_old_heterogeneous_transport():
    old, new = medium(tensor=False), medium()
    new.load_state_dict(old.state_dict(), strict=False)
    with torch.no_grad():
        new.transport_shear.weight.zero_()
        new.transport_shear.bias.zero_()
    s = state(old)
    for dt in (0., .023):
        a, b = old.transport(s, dt), new.transport(s, dt)
        for x, y in zip((a.field, *a.flux), (b.field, *b.flux)):
            torch.testing.assert_close(x, y, atol=2e-14, rtol=2e-14)


def test_spd_non_axis_principal_direction_and_energy_mass():
    net = medium()
    material = net.material_field()
    speed = net.log_speed(material).exp()[None]
    b = net.transport_factor(material, speed)
    a = b @ b.transpose(-1, -2)
    assert torch.linalg.eigvalsh(a).min() > 0
    assert a[..., 0, 1].abs().max() > .1
    s = state(net)
    for dt in (.001, .19):
        out = net.transport(s, dt)
        torch.testing.assert_close(net.energy(out), net.energy(s), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(out.field.sum((1,2,3)), s.field.sum((1,2,3)), atol=1e-12, rtol=1e-12)


def test_tensor_transport_linear_and_shear_gradient_matches_finite_difference():
    net = medium()
    s, t = state(net), state(net)
    combo = replace(s, field=.3*s.field+.7*t.field,
                    flux=tuple(.3*x+.7*y for x,y in zip(s.flux,t.flux)))
    a,b,c = net.transport(s,.011),net.transport(t,.011),net.transport(combo,.011)
    torch.testing.assert_close(c.field,.3*a.field+.7*b.field)
    probe=torch.randn_like(s.field)
    loss=(a.field*probe).sum()
    loss.backward()
    analytic=net.transport_shear.bias.grad[0].item()
    assert net.transport_shear.weight.grad.norm()>0
    assert net.material.coefficients.grad.norm()>0
    values=[]
    with torch.no_grad():
        for change in (1e-6,-2e-6):
            net.transport_shear.bias[0].add_(change)
            values.append((net.transport(s,.011).field*probe).sum().item())
        net.transport_shear.bias[0].add_(1e-6)
    assert math.isclose(analytic,(values[0]-values[1])/2e-6,rel_tol=1e-6,abs_tol=1e-7)


def test_dynamic_read_rhs_agrees_with_tensor_flow():
    net=medium()
    s=state(net)
    expected=net.field_rhs(s)
    out,_=net.advance(s,1e-8)
    torch.testing.assert_close((out.field-s.field)/1e-8,expected,atol=1e-4,rtol=1e-5)


def test_resolution_refinement_matches_same_continuum_plane_wave():
    errors=[]
    # Constant B and k; exact solution f=cos(k.x)cos(|B^T k|t).
    for n in (4,8,16):
        net=medium((n,n,n))
        with torch.no_grad():
            net.material.coefficients.zero_()
            net.log_speed.weight.zero_()
        s=net.initial_state()
        k=torch.tensor([2*math.pi,2*math.pi,0.],dtype=torch.float64)
        phase=net.coordinates@k
        s=replace(s,field=phase.cos()[None,...,None].expand_as(s.field).clone())
        duration=.02
        steps=4*n
        with torch.no_grad():
            for _ in range(steps):
                s=net.transport(s,duration/steps)
            b=net.transport_factor(net.material_field(),torch.ones(1,n,n,n,3,dtype=torch.float64))
            omega=(b[0,0,0,0].T@k).norm()
            exact=phase.cos()*torch.cos(omega*duration)
            errors.append((s.field[0,...,0]-exact).square().mean().sqrt().item())
    assert errors[1]<errors[0] and errors[2]<errors[1], errors


def test_runtime_checkpoint_preserves_tensor_and_temporal_history(tmp_path):
    torch.manual_seed(91)
    model=PlasticMediumPorts3D(vocab_size=17,shape=(2,2,2),channels=8,
        hidden=6,heads=2,queries=2,anisotropic_transport=True,read_mode='temporal',
        temporal_rates=[1.,4.],temporal_frequencies=[0.,2.]).double()
    stream=ContinuousStream(model,max_step=.005)
    with torch.no_grad():
        model.medium.transport_shear.bias.copy_(torch.tensor([.2,-.3,.1]))
        stream.observe(0.,torch.tensor([2]))
        stream.advance_to(.01)
        payload=stream.state_dict()
    restored=ContinuousStream.from_state_dict(copy.deepcopy(model),payload)
    assert restored.model.medium.anisotropic_transport
    with torch.no_grad():
        stream.advance_to(.02)
        restored.advance_to(.02)
    torch.testing.assert_close(restored.belief.medium.field,stream.belief.medium.field)
    torch.testing.assert_close(restored.belief.temporal.value,stream.belief.temporal.value)


def test_closed_transmission_row_is_finite_and_identity():
    net=medium()
    material=net.material_field()
    s=state(net)
    factor=net.transport_factor(material,torch.zeros(2,4,4,2,3,dtype=torch.float64))
    out=net._tensor_transport(s,net._duration(.1,s.field),factor)
    torch.testing.assert_close(out.field,s.field,atol=1e-14,rtol=1e-14)
    for x,y in zip(out.flux,s.flux):
        torch.testing.assert_close(x,y,atol=0,rtol=0)


def test_tensor_runtime_rejects_mismatched_continuation():
    import pytest
    model=PlasticMediumPorts3D(vocab_size=17,shape=(2,2,2),channels=8,
        hidden=6,heads=2,queries=2,anisotropic_transport=True).double()
    payload=ContinuousStream(model,max_step=.005).state_dict()
    payload['anisotropic_transport']=False
    with pytest.raises(ValueError,match='propagation tensor'):
        ContinuousStream.from_state_dict(model,payload)


def test_cuda_graph_tensor_backward_matches_eager():
    import os
    import pytest
    from information_boltzmann.runtime.training import CapturedPlasticChunk, quiet_training_chunk
    if os.environ.get('IB_TEST_CUDA') != '1' or not torch.cuda.is_available():
        pytest.skip('Explicit CUDA numerical check')
    torch.manual_seed(47)
    net=PlasticMediumPorts3D(vocab_size=17,shape=(2,2,2),channels=8,
        hidden=6,heads=2,queries=2,anisotropic_transport=True).cuda()
    ids=torch.tensor([[1,2]],device='cuda')
    targets=torch.tensor([[2,3]],device='cuda')
    belief=net.initial_belief().detach()
    capture=CapturedPlasticChunk(net,ids,targets,belief,event_duration=.005)
    capture.zero_grad()
    loss,out,nll=capture.backward(ids,targets,belief)
    torch.cuda.synchronize()
    expected=net.medium.transport_shear.bias.grad.clone()
    expected_field=out.medium.field.clone()
    net.zero_grad(set_to_none=True)
    eager,output,_=quiet_training_chunk(net,ids,targets,belief,event_duration=.005)
    eager.backward()
    torch.testing.assert_close(output.medium.field,expected_field,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(net.medium.transport_shear.bias.grad,expected,atol=1e-5,rtol=1e-5)


def test_joint_likelihood_reaches_new_tensor_and_writer_with_temporal_read():
    from information_boltzmann.runtime.training import quiet_training_chunk
    torch.manual_seed(47)
    model=PlasticMediumPorts3D(vocab_size=17,shape=(2,2,2),channels=8,
        hidden=6,heads=2,queries=2,anisotropic_transport=True,read_mode='temporal',
        temporal_rates=[1.,4.],temporal_frequencies=[0.,2.],bath_type='conductance',
        short_term_plasticity=True,activity_adaptation=True).double()
    ids=torch.tensor([[1,2,3]])
    objective,belief,nll=quiet_training_chunk(model,ids,torch.tensor([[2,3,4]]),
        model.initial_belief(),event_duration=.005)
    nll.backward()
    assert torch.isfinite(objective)
    assert model.medium.transport_shear.bias.grad.norm()>0
    assert model.source.embedding.weight.grad.norm()>0
    assert model.decoder.weight.grad.norm()>0
    torch.testing.assert_close(belief.temporal.elapsed,belief.medium.elapsed)
