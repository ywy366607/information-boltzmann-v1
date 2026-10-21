"""Finite-volume physical checks, not a synthetic learning benchmark."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.spatial_exchange import SpatialExchange, SpatialExchangeParams, cubic_mesh
from information_boltzmann.cgb.trace_exchange import TraceExchange


def panel(params=None):
    xyz,pre,post = cubic_mesh(2)
    e,n = len(pre),len(xyz)
    return SpatialExchange(pre,post,np.full(e,.4),np.full(n,.3),np.full(e,.1),np.full(n,.1),
        np.full(n,.8),np.full(n,.2),np.ones(e),np.full(n,.8),np.linspace(0.,.3,n),
        params=params,coordinates=xyz,trace=.2)


def test_local_cohesive_affinity_is_derivative_of_the_full_joint_energy():
    m = panel()
    plus,minus = m.fluxes()['assembly']
    a = np.log(plus[0]/minus[0])
    g,h = m.free_energy(),1e-7
    post = m.post[0]
    m.capacity[0] += h
    m.material[post] -= h
    m.charged[post] -= h
    m.depleted[post] += h
    assert abs((m.free_energy()-g)/h+a) < 3e-6


@pytest.mark.parametrize('dt',[.01,1.,100.])
def test_spatial_reaction_transport_conserve_and_have_closed_g_descent(dt):
    m = panel()
    for _ in range(10):
        update = m.step(dt)
        assert update['free_energy_drop'] >= -1e-10
        for v in m.balances().values():
            assert np.max(np.abs(v),initial=0.) < 1e-10
        assert np.all(m.capacity >= m.bound) and np.all(m.capacity <= m.positions)


def test_diffusion_moves_material_without_creating_mass_or_a_new_supply():
    m = panel()
    m.material[:] = 0.
    m.material[0] = 1.
    m.initial_material = m.material+m._sum(m.post,m.capacity)
    m.initial_energy,m.initial_free_energy = m.energy(),m.free_energy()
    before = m.free_energy()
    m._diffuse(1.)
    assert m.material[0] < 1. and np.count_nonzero(m.material) > 1
    assert abs(m.material.sum()-1.) < 1e-14
    assert m.free_energy() <= before
    for v in m.balances().values():
        assert np.max(np.abs(v),initial=0.) < 1e-11


def test_cohesion_is_one_pair_energy_not_a_duplicate_surface_charge():
    m = panel()
    manual = 0.
    for e in range(len(m.pre)):
        for f in range(e+1,len(m.pre)):
            shared = len({int(m.pre[e]),int(m.post[e])}&{int(m.pre[f]),int(m.post[f])})
            manual -= m.params.cohesion_energy*shared*m.capacity[e]*m.capacity[f]
    assert abs((m.energy()-super(SpatialExchange,m).energy())-manual) < 1e-13


def test_spatial_extensions_can_be_disabled_to_recover_parent_bitwise():
    p = SpatialExchangeParams(cohesion_energy=0.,surface_attempt=0.,bulk_attempt=1.,
                             material_diffusion=0.,support_diffusion=0.,drive_diffusion=0.)
    a = panel(p)
    s = a.state_dict()
    s['kind'] = 'trace_exchange'
    for k in ('cohesion_energy','surface_attempt','bulk_attempt','material_diffusion',
              'support_diffusion','drive_diffusion','spacing'):
        s['params'].pop(k)
    b = TraceExchange.from_state_dict(s)
    for _ in range(4):
        a.step(.3)
        b.step(.3)
    for name in TraceExchange._arrays:
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    for name in TraceExchange._ledgers:
        assert getattr(a,name)==getattr(b,name)


def test_seed_changes_where_material_is_deposited_without_a_direction_or_branch_count():
    p = SpatialExchangeParams(binding_rate=0.,transport_rate=0.,release_rate=0.,conversion_rate=0.,
        active_rate=0.,write_rate=0.,recovery_rate=0.,assembly_rate=.1)
    a = panel(p)
    a.capacity[:] = 0.
    a.bound[:] = 0.
    a.capacity[0] = .8
    old = a.capacity.copy()
    neighbor = a.neighbor_structure()
    a._assembly(1.)
    delta = a.capacity-old
    assert delta[(neighbor>0)&(old==0)].mean() > delta[(neighbor==0)&(old==0)].mean()


def test_spatial_checkpoint_preserves_diffusion_coupled_future(tmp_path):
    a = panel()
    for _ in range(3): a.step(.4)
    save_exchange(tmp_path/'spatial.npz',a,{'mesh':'finite-volume material'})
    b,meta = load_exchange(tmp_path/'spatial.npz')
    assert meta['mesh']=='finite-volume material'
    for _ in range(4):
        a.step(.4)
        b.step(.4)
    for name in a._arrays:
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    for name in a._ledgers:
        assert getattr(a,name)==getattr(b,name)


def test_diffusion_refinement_matches_independent_graph_heat_equation():
    # A 2x2x2 cube has commuting per-axis matchings and is already exact to
    # roundoff. Three sites per axis expose the noncommuting split error.
    xyz,pre,post = cubic_mesh(3)
    n,e = len(xyz),len(pre)
    p = SpatialExchangeParams(binding_rate=0.,transport_rate=0.,release_rate=0.,assembly_rate=0.,
        conversion_rate=0.,active_rate=0.,write_rate=0.,recovery_rate=0.,
        material_diffusion=.4,support_diffusion=0.,drive_diffusion=0.)
    initial = np.zeros(n)
    initial[0] = 1.
    def rhs(_,y):
        j = .4*(y[pre]-y[post])
        return -np.bincount(pre,weights=j,minlength=n)+np.bincount(post,weights=j,minlength=n)
    exact = solve_ivp(rhs,[0.,2.],initial,rtol=1e-12,atol=1e-14).y[:,-1]
    errors = []
    for dt in (.2,.1,.05):
        m = SpatialExchange(pre,post,np.zeros(e),np.zeros(n),np.zeros(e),np.zeros(n),
            np.zeros(n),np.zeros(n),np.ones(e),initial,np.zeros(n),params=p,coordinates=xyz)
        for _ in range(round(2/dt)): m.step(dt)
        errors.append(np.linalg.norm(m.material-exact))
    assert 1.7 < errors[0]/errors[1] < 2.3
    assert 1.7 < errors[1]/errors[2] < 2.3


def test_invalid_physical_mesh_or_material_parameters_rejected():
    with pytest.raises(ValueError): cubic_mesh(1)
    with pytest.raises(ValueError): SpatialExchangeParams(cohesion_energy=-1.)
    s = panel().state_dict()
    s['arrays']['coordinates'][0] += 1.
    with pytest.raises(ValueError): SpatialExchange.from_state_dict(s)


def test_cohesive_deposition_refines_toward_its_independent_continuous_ode():
    p = SpatialExchangeParams(binding_rate=0.,transport_rate=0.,release_rate=0.,conversion_rate=0.,
        active_rate=0.,write_rate=0.,recovery_rate=0.,assembly_rate=.7,structure_energy=.5,
        charged_energy=1.,cohesion_energy=.4,bulk_attempt=.1,surface_attempt=2.,
        material_diffusion=0.,support_diffusion=0.,drive_diffusion=0.)
    pre,post = np.array([0,1]),np.array([1,2])
    start = np.r_[[.3,.4],np.full(3,.8),np.full(3,.8),np.full(3,.2)]
    def rhs(_,y):
        n,b,fuel,waste = y[:2],y[2:5],y[5:8],y[8:11]
        neighbor = n[::-1]
        rate = .7*(.1+2*neighbor)
        j = rate*(b[post]*(1.-n)*fuel[post]-np.exp(.5-.4*neighbor-1.)*(n-.1)*waste[post])
        debit = np.bincount(post,weights=j,minlength=3)
        return np.r_[j,-debit,-debit,debit]
    exact = solve_ivp(rhs,[0.,2.],start,rtol=1e-12,atol=1e-14).y[:,-1]
    errors=[]
    for dt in (.1,.05,.025):
        m = SpatialExchange(pre,post,start[:2],np.zeros(3),np.full(2,.1),np.zeros(3),
            start[5:8],start[8:11],np.ones(2),start[2:5],np.zeros(3),params=p,
            coordinates=np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]))
        for _ in range(round(2/dt)):m.step(dt)
        errors.append(np.linalg.norm(np.r_[m.capacity,m.material,m.charged,m.depleted]-exact))
    assert 1.7 < errors[0]/errors[1] < 2.3
    assert 1.7 < errors[1]/errors[2] < 2.3


def test_spatial_diffusion_operator_has_cell_centered_continuum_limit():
    # This checks the actual geometric stencil, not nonlinear shape convergence.
    errors=[]
    for side in (5,9,17):
        a = 4./side
        xyz,pre,post = cubic_mesh(side,a)
        mode = np.cos(np.pi*(xyz[:,0]+a/2)/4.)
        current = (mode[pre]-mode[post])/a**2
        lap = np.bincount(pre,weights=current,minlength=len(xyz))-np.bincount(post,weights=current,minlength=len(xyz))
        numerical = np.dot(mode,lap)/np.dot(mode,mode)
        errors.append(abs(numerical-(np.pi/4.)**2))
    assert errors[1] < .4*errors[0] and errors[2] < .4*errors[1]
