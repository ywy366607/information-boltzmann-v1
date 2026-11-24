"""Exact numeric, one-use and full-learning contracts for shared force jets."""
import copy
from types import SimpleNamespace

import pytest
import torch

from scripts.quiet_force_jet import SplitJet, prepare, independent_pair_potential
from scripts.ib_local.adaptive_hamiltonian import AdaptiveHamiltonian, quiet, potential
from scripts.ib_local.contact_momentum import ContactMomentum


@pytest.mark.parametrize('impedance', [False, True])
@pytest.mark.parametrize('dt', [.02, -1e-7, 0.])
def test_shared_force_jet_matches_full_jvp(impedance, dt):
    torch.set_num_threads(1); torch.manual_seed(7)
    s = torch.randn(2, 1, 3, 2, 2, dtype=torch.float64)
    p = torch.randn_like(s)
    theta = torch.randn(2, 7, dtype=s.dtype)*.2
    context = torch.randn(2, 1, 3, 1, 1, dtype=s.dtype)
    scale = torch.exp(torch.randn_like(context))
    z = torch.exp(torch.randn_like(context)) if impedance else None
    ds = torch.randn(5, *s.shape, dtype=s.dtype)
    dp = torch.randn_like(ds); dtp = torch.randn(5, *theta.shape, dtype=s.dtype)
    jet = SplitJet(s, p, theta, context, dt, scale, z)
    actual = jet.apply(ds, dp, dtp)
    fn = lambda x, y, t: quiet(x, y, t, context, dt, scale, z)
    values, expected = torch.vmap(lambda x, y, t: torch.func.jvp(
        fn, (s, p, theta), (x, y, t)))(ds, dp, dtp)
    torch.testing.assert_close(jet.next_s, values[0][0], atol=0, rtol=0)
    torch.testing.assert_close(jet.next_p, values[1][0], atol=0, rtol=0)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)


def fixture():
    torch.manual_seed(8)
    module = AdaptiveHamiltonian(2, 3, 2, 2).double()
    module.contact = ContactMomentum(2, 3, 2, 2).double()
    with torch.no_grad():
        module.theta[:, -2:] = torch.tensor([.4, -.2], dtype=torch.float64)
        for t in (module.sensitivity_s, module.sensitivity_p,
                  module.contact.sensitivity_s, module.contact.sensitivity_p):
            t.normal_()
    layout = SimpleNamespace(layers=[SimpleNamespace(matrix_size=12, heads=3, dk=2, dv=2)]*2)
    recurrent = torch.randn(2, 1, 15, dtype=torch.float64)
    momentum = list(torch.randn(2, 1, 3, 2, 2, dtype=torch.float64).unbind())
    return module, layout, recurrent, momentum


def test_prepare_commit_preserves_state_all_eligibility_and_real_read_gradient():
    module, layout, state, momentum = fixture()
    reference = copy.deepcopy(module)
    with torch.no_grad():
        reference.quiet(layout, state, momentum, dt=.02, capture=False)
        expected_state, expected_p = reference.quiet(layout, state, momentum, dt=.02, capture=True)
    proposed = prepare(module, layout, state, momentum, dt=.02)
    torch.testing.assert_close(proposed.result, expected_state, atol=0, rtol=0)
    actual_state, actual_p = proposed.commit(state, momentum)
    torch.testing.assert_close(actual_state, expected_state, atol=0, rtol=0)
    torch.testing.assert_close(torch.stack(actual_p), torch.stack(expected_p), atol=0, rtol=0)
    for k, v in reference.state_dict().items():
        torch.testing.assert_close(module.state_dict()[k], v, atol=2e-12, rtol=2e-12)
    read = torch.randn_like(state)
    actual_grad = torch.autograd.grad((module.bridge(actual_state)*read).sum(),
                                      (module.theta, module.contact.phi))
    expected_grad = torch.autograd.grad((reference.bridge(expected_state)*read).sum(),
                                        (reference.theta, reference.contact.phi))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)
    with pytest.raises(RuntimeError):
        proposed.commit(state, momentum)


def test_discard_matches_reference_trial_and_keeps_carried_eligibility():
    module, layout, state, momentum = fixture()
    reference = copy.deepcopy(module)
    with torch.no_grad(): reference.quiet(layout, state, momentum, dt=.02, capture=False)
    proposed = prepare(module, layout, state, momentum, dt=.02)
    assert proposed.jet.first is None and proposed.jet.second is None
    proposed.close()
    for k, v in reference.state_dict().items():
        torch.testing.assert_close(module.state_dict()[k], v, atol=0, rtol=0)
    with pytest.raises(RuntimeError): proposed.commit(state, momentum)


@pytest.mark.parametrize('mutation', ['state', 'theta_fused', 'eligibility', 'replace_theta'])
def test_stale_proposal_rejects_changed_scope(mutation):
    module, layout, state, momentum = fixture()
    proposed = prepare(module, layout, state, momentum, dt=.02)
    with torch.no_grad():
        if mutation == 'state': state.add_(.1)
        elif mutation == 'theta_fused': module.theta.data.add_(.1)
        elif mutation == 'replace_theta': module.theta = torch.nn.Parameter(module.theta.clone())
        else: module.contact.sensitivity_s.add_(.1)
    with pytest.raises(RuntimeError): proposed.commit(state, momentum)
    proposed.close()


@pytest.mark.parametrize('heads', [1, 2, 8])
def test_independent_pairs_preserve_energy_force_and_mixed_derivative(heads):
    torch.set_num_threads(1); torch.manual_seed(17)
    s = torch.randn(2, 1, heads, 2, 3, dtype=torch.float64)
    theta = torch.randn(2, heads+4, dtype=s.dtype)*.2
    ds, dt = torch.randn_like(s), torch.randn_like(theta)
    f = lambda x, t: potential(x, t, heads).sum()
    g = lambda x, t: independent_pair_potential(x, t, heads).sum()
    torch.testing.assert_close(f(s, theta), g(s, theta), atol=1e-11, rtol=1e-12)
    gf, gg = torch.func.grad(f, argnums=(0, 1)), torch.func.grad(g, argnums=(0, 1))
    for x, y in zip(gf(s, theta), gg(s, theta)):
        torch.testing.assert_close(x, y, atol=1e-11, rtol=1e-12)
    _, a = torch.func.jvp(gf, (s, theta), (ds, dt))
    _, b = torch.func.jvp(gg, (s, theta), (ds, dt))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=5e-11, rtol=1e-12)


def test_full_learning_restore_and_next_actor_match_reference(monkeypatch):
    from test_generative_cycle import configured
    from scripts.ib_local.adaptive_hamiltonian import configure
    from scripts.ib_local.navigation_checkpoint import snapshot, restore
    from scripts.ib_local.autonomous_cycle import configure as autonomous, dispatch
    from test_autonomous_cycle import force_choices
    torch.set_num_threads(1)
    engine = configured(); engine.step(torch.randn(8, 4)); configure(engine); autonomous(engine)
    engine.model.head_coupling._prepared_quiet_enabled = False
    force_choices(monkeypatch, [1, 1, 0])
    initial = snapshot(engine)
    dispatch(engine); dispatch(engine); dispatch(engine)
    expected = dispatch(engine)
    reference = snapshot(engine)
    restore(engine, initial)
    engine.model.head_coupling._prepared_quiet_enabled = True
    force_choices(monkeypatch, [1, 1, 0])
    dispatch(engine); dispatch(engine); dispatch(engine)
    actual = dispatch(engine)
    assert actual['utterance_hex'] == expected['utterance_hex']
    assert actual['choice'] == expected['choice']
    assert actual['updates'] == expected['updates']
    for k, v in reference['model'].items():
        if torch.is_tensor(v):
            torch.testing.assert_close(engine.model.state_dict()[k], v, atol=2e-6, rtol=2e-5)
    assert '_quiet_proposal' not in snapshot(engine)['engine']
