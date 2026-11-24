"""Production algebra/credit contracts, no synthetic capability experiment."""
import copy

import torch

from scripts.ib_local.adaptive_hamiltonian import (
    AdaptiveHamiltonian, coefficients, configure, invariants, potential, quiet)
from scripts.ib_local.bfss_gdn_reference import quiet_step, rotation_charge
from test_generative_cycle import configured
from test_autonomous_cycle import force_choices
from scripts.ib_local.autonomous_cycle import configure as configure_autonomous, dispatch
from scripts.ib_local.adaptive_spectrum import admit, energy


def test_uniform_initialization_contains_original_force_and_rotation():
    torch.manual_seed(6)
    module = AdaptiveHamiltonian(2, 3, 2, 2).double()
    s, p = torch.randn(2, 1, 3, 2, 2, dtype=torch.float64), torch.randn(2, 1, 3, 2, 2, dtype=torch.float64)
    # FP32 constructor values converted to double: compare actual decoded units.
    omega, strength = coefficients(module.theta, module.context, 3)
    a, b = quiet(s, p, module.theta, module.context, .01)
    x, y = quiet_step(s, p, dt=.01, frequency=omega[0,0,0,0,0], coupling=strength[0])
    torch.testing.assert_close(a, x, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(b, y, atol=1e-12, rtol=1e-12)


def test_content_attention_has_complete_force_and_preserves_basis_symmetry_charge():
    torch.manual_seed(7)
    module = AdaptiveHamiltonian(2, 3, 2, 2).double()
    with torch.no_grad():
        module.theta[:, -2:] = torch.tensor([.4, -.2], dtype=torch.float64)
        module.theta[:, :3] += torch.tensor([-.1, 0., .1], dtype=torch.float64)
    s = torch.randn(2, 1, 3, 2, 2, dtype=torch.float64)
    p = torch.randn_like(s)
    u, _ = torch.linalg.qr(torch.randn(2, 2, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(2, 2, dtype=torch.float64))
    transformed = u@s@v.T
    torch.testing.assert_close(potential(s, module.theta, 3), potential(transformed, module.theta, 3))
    a, b = quiet(s, p, module.theta, module.context, .01)
    for old, new in zip(rotation_charge(s, p), rotation_charge(a, b)):
        torch.testing.assert_close(old, new, atol=1e-12, rtol=1e-12)
    direction = torch.randn_like(s)
    force = torch.func.grad(lambda x: potential(x, module.theta, 3).sum())(s)
    finite = (potential(s+1e-5*direction,module.theta,3).sum()-
              potential(s-1e-5*direction,module.theta,3).sum())/2e-5
    torch.testing.assert_close((force*direction).sum(), finite, atol=1e-9, rtol=1e-8)


def test_carried_local_sensitivities_match_full_quiet_unroll_and_endpoint_gradient():
    torch.set_num_threads(1)
    e = configured(); e.step(torch.randn(8,4)); configure(e)
    module = e.model.head_coupling.double()
    base = e.model.body.base
    state = e.recurrent.double()
    momentum = [p.double() for p in e.bfss_momenta]
    theta = module.theta.detach().clone().requires_grad_(True)
    s = state[..., :base.layers[0].matrix_size].reshape(module.layers, 1, module.heads,module.dk,module.dv)
    p = torch.stack(momentum)
    with torch.no_grad():
        module.context.copy_(invariants(s)[0].mean(-1)[...,None,None])
        module.context_valid.fill_(True)
    context = module.context.detach().clone()
    expected_s, expected_p = s, p
    for _ in range(3):
        expected_s, expected_p = quiet(expected_s, expected_p, theta, context, .01)
        state, momentum = module.quiet(base,state,momentum,dt=.01,capture=True)
    cotangent = torch.randn_like(expected_s)
    reference, = torch.autograd.grad((expected_s*cotangent).sum(), theta)
    bridged = module.bridge(state)
    actual, = torch.autograd.grad((bridged[..., :base.layers[0].matrix_size].reshape_as(expected_s)*cotangent).sum(), module.theta)
    torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(state[..., :base.layers[0].matrix_size].reshape_as(expected_s),expected_s)
    assert int(module.trace_steps) == 3


def test_full_production_event_learns_every_layer_and_restores_quiet_credit(monkeypatch):
    from scripts.ib_local.navigation_checkpoint import snapshot, restore
    torch.set_num_threads(1)
    e = configured(); e.step(torch.randn(8,4)); configure(e); configure_autonomous(e)
    force_choices(monkeypatch, [1, 1, 0])
    dispatch(e); dispatch(e); dispatch(e)
    saved = snapshot(e)
    before = e.model.head_coupling.theta.detach().clone()
    row = dispatch(e)
    assert row['updates'] > 0 and row['external_bytes'] == 128
    diag = row['adaptive_dynamics']
    assert diag['last_consumed_steps'] == 3  # two chosen + original boundary quiet
    assert all(v>0 for v in diag['per_layer_gradient_norm'])
    assert all(v>0 for v in diag['per_layer_parameter_step'])
    assert not torch.equal(before,e.model.head_coupling.theta)
    expected = copy.deepcopy(e.model.state_dict())
    fresh = configured();configure(fresh);configure_autonomous(fresh);restore(fresh,saved)
    repeated = dispatch(fresh)
    assert row['adaptive_dynamics'] == repeated['adaptive_dynamics']
    assert row['utterance_hex'] == repeated['utterance_hex'] and row['metal_g'] == repeated['metal_g']
    for key, value in expected.items():
        if torch.is_tensor(value):
            assert torch.equal(value,fresh.model.state_dict()[key]),key


def test_fixed_impedance_changes_speed_without_inverse_frequency_amplitude_and_preserves_charge():
    torch.manual_seed(19)
    module = AdaptiveHamiltonian(2,3,2,2).double()
    with torch.no_grad():
        module.theta[:,4] = -1000.  # exact uncoupled numerical subcase
    s,p = torch.randn(2,1,3,2,2,dtype=torch.float64),torch.randn(2,1,3,2,2,dtype=torch.float64)
    scale = torch.tensor([.1,1.,10.],dtype=torch.float64)[None,None,:,None,None].expand(2,-1,-1,-1,-1)
    z = torch.full_like(scale,.8)
    a,b = quiet(s,p,module.theta,module.context,.13,scale,z)
    torch.testing.assert_close(z.square()*a.square()+b.square(),
        z.square()*s.square()+p.square(), atol=1e-12,rtol=1e-12)
    for old,new in zip(rotation_charge(s,p),rotation_charge(a,b)):
        torch.testing.assert_close(old,new,atol=1e-12,rtol=1e-12)
    x,y=quiet(a,b,module.theta,module.context,-.13,scale,z)
    torch.testing.assert_close(x,s,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(y,p,atol=1e-12,rtol=1e-12)


def test_commutator_potential_does_not_equate_equal_frequency_with_equal_phase():
    module = AdaptiveHamiltonian(1,3,2,2).double()
    template = torch.tensor([[1.,2.],[-.3,.7]],dtype=torch.float64)
    phase = torch.tensor([0.,torch.pi,torch.pi/3],dtype=torch.float64)[None,None,:,None,None]
    z = torch.full_like(phase,.8)
    s,p = phase.cos()*template,-z*phase.sin()*template
    force = torch.func.grad(lambda x:potential(x,module.theta,3).sum())(s)
    torch.testing.assert_close(force,torch.zeros_like(force),atol=1e-25,rtol=0)
    a,b = quiet(s,p,module.theta,module.context,.2,torch.ones_like(z),z)
    omega,_ = coefficients(module.theta,module.context,3)
    # All heads have the same angular speed, so both the antipodal pair and
    # the nonzero phase offset survive. Matrix compatibility is not phase locking.
    torch.testing.assert_close(a,(phase+omega*.2).cos()*template,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(b,-z*(phase+omega*.2).sin()*template,atol=1e-12,rtol=1e-12)


def test_spectrum_admission_retains_state_rng_and_optimizer_and_restores_buffers():
    torch.set_num_threads(1)
    e=configured();e.step(torch.randn(8,4));configure(e)
    module=e.model.head_coupling
    state=e.recurrent.clone();p=[x.clone() for x in e.bfss_momenta]
    theta=module.theta.detach().clone();rng=torch.get_rng_state().clone()
    optimizer=copy.deepcopy(e.optimizer.state_dict())
    receipt=admit(e,probe_steps=3)
    omega,_=coefficients(module.theta,module.context,module.heads,module.frequency_scale)
    assert receipt['schema']==1 and module.spectrum_enabled
    assert float((omega.max()/omega.min()).detach())>1
    assert torch.equal(e.recurrent,state) and torch.equal(module.theta,theta)
    assert all(torch.equal(a,b) for a,b in zip(p,e.bfss_momenta))
    assert torch.equal(torch.get_rng_state(),rng)
    assert e.optimizer.state_dict()['param_groups']==optimizer['param_groups']
    fresh=AdaptiveHamiltonian(module.layers,module.heads,module.dk,module.dv)
    fresh.load_state_dict(module.state_dict())
    assert fresh.spectrum_enabled and fresh.phase_limit==float(module.solver_phase_limit)
    torch.testing.assert_close(fresh.frequency_scale,module.frequency_scale)


def test_historical_P_read_credit_survives_trace_consumption_and_matches_partial_derivative():
    torch.set_num_threads(1)
    e=configured();e.step(torch.randn(8,4));configure(e);admit(e,probe_steps=2)
    module=e.model.head_coupling.double();base=e.model.body.base
    state=e.recurrent.double();momenta=[p.double() for p in e.bfss_momenta]
    s=state[...,:base.layers[0].matrix_size].reshape(module.layers,1,module.heads,module.dk,module.dv)
    p=torch.stack(momenta);theta=module.theta.detach().clone().requires_grad_(True)
    context=invariants(s)[0].mean(-1)[...,None,None]
    with torch.no_grad():module.context.copy_(context);module.context_valid.fill_(True)
    queries=[torch.randn(1,module.heads,module.dk,dtype=torch.float64) for _ in range(module.layers)]
    for _ in range(2):
        s,p=quiet(s,p,theta,context,.01,module.frequency_scale,module.impedance)
        state,momenta=module.quiet(base,state,momenta,dt=.01,capture=True)
    q=torch.stack(queries)
    reference_read=torch.einsum('lbhkv,lbhk->lbhv',p,q).flatten(2).flatten()
    direction=torch.randn_like(reference_read)
    reference,=torch.autograd.grad((reference_read*direction).sum(),theta)
    eligibility=module.capture_momentum_read(queries)
    saved=eligibility.clone()
    module.consume()  # later real evidence must use the old read, not new E_P
    assert torch.equal(saved,eligibility)
    bridged=module.bridge_momentum_read(reference_read.detach()[None],eligibility)
    actual,=torch.autograd.grad((bridged*direction).sum(),module.theta)
    torch.testing.assert_close(actual,reference,atol=1e-10,rtol=1e-9)
    assert int(module.momentum_credit_count)==1


def test_full_spectrum_cycle_pays_later_P_credit_and_keeps_time_under_subdivision(monkeypatch):
    torch.set_num_threads(1)
    e=configured();e.step(torch.randn(8,4));configure(e);admit(e,probe_steps=2);configure_autonomous(e)
    force_choices(monkeypatch,[1,0])
    dispatch(e);dispatch(e);first=dispatch(e)
    receipt=e.internal_pending['momentum_read_eligibility']
    assert receipt is not None and receipt.numel() < e.recurrent.numel()*20
    force_choices(monkeypatch,[0])
    dispatch(e);second=dispatch(e)
    assert first['momentum_current_credit_samples']==1
    assert second['momentum_delayed_credit_samples']==1
    assert second['adaptive_dynamics']['momentum_credit_count']>0
    assert any(x>0 for x in second['adaptive_dynamics']['per_layer_momentum_credit_norm'])
    module=e.model.head_coupling
    with torch.no_grad():module.frequency_scale.mul_(4)
    before=int(module.trace_steps)
    e.recurrent,e.bfss_momenta=module.quiet(e.model.body.base,e.recurrent,e.bfss_momenta,dt=e.bfss_dt,capture=True)
    assert int(module.last_solver_substeps)>=2
    assert int(module.trace_steps)==before+1  # solver resolution does not invent more thought events
