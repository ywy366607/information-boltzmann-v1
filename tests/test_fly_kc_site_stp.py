"""Resource-law, native route and persistence contracts, not capability tests."""
import copy
from dataclasses import replace

import pytest
import torch

from test_fly_kc_terminals import setup
from information_boltzmann.core.fly_kc_site_stp import (
    attach_kc_site_stp, site_state, site_parameters, release_site_resources,
    KC_SITE_STP_PARAMETER_NAMES,
)
from information_boltzmann.core.fly_kc_terminals import (
    advance_kc_terminals, kc_arrival, KC_TERMINAL_STATE_FIELDS,
)
from information_boltzmann.core.fly_native_policy_credit import native_parameter_names


def test_site_resource_conservation_local_gate_and_recovery(tmp_path):
    model, state, _ = setup(tmp_path)
    attach_kc_site_stp(model)
    resources, utilization = site_state(model, state)
    raw = torch.ones(1, 3)
    spikes = state.h.new_tensor([[0, 0, 0, 1, 1, 0]])
    gate = raw.new_tensor([[1., 0., .5]])
    packet, x, u = release_site_resources(model, state, raw, gate, spikes)
    torch.testing.assert_close(packet, gate)
    u0, fac, rec = site_parameters(model)
    active = utilization + u0 * (1 - utilization)
    used = active * resources * gate
    torch.testing.assert_close((x - 1) / rec + 1, resources - used)
    assert x[0, 0] < x[0, 2] < x[0, 1] == 1
    assert torch.all(u > utilization)
    local = replace(state, kc_site_resources=x, kc_site_utilization=u)
    _, recovered, relaxed = release_site_resources(
        model, local, raw * 0, gate, spikes * 0)
    assert torch.all(recovered >= x) and torch.all(relaxed < u)
    torch.testing.assert_close(recovered, 1 + (x - 1) * rec)


def test_repeated_release_preserves_bounds_and_parameter_gradients(tmp_path):
    model, state, _ = setup(tmp_path)
    attach_kc_site_stp(model)
    spikes = state.h.new_tensor([[0, 0, 0, 1, 1, 0]])
    raw = torch.ones(1, 3)
    packets = []
    for _ in range(6):
        packet, x, u = release_site_resources(model, state, raw, raw, spikes)
        assert torch.all((x >= 0) & (x <= 1))
        assert torch.all((u >= 0) & (u <= 1))
        state = replace(state, kc_site_resources=x, kc_site_utilization=u)
        packets.append(packet)
    assert torch.all(packets[-1] < packets[0])
    parameters = [getattr(model, n) for n in KC_SITE_STP_PARAMETER_NAMES]
    grads = torch.autograd.grad(sum(p.sum() for p in packets), parameters)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    assert set(KC_SITE_STP_PARAMETER_NAMES) <= set(native_parameter_names(model))


def test_native_delayed_routes_read_resource_limited_site_packets(tmp_path):
    model, state, _ = setup(tmp_path)
    attach_kc_site_stp(model)
    model.kc_terminal_transport_enabled = False
    raw = state.h.new_tensor([[0, 0, 0, 1, 1, 0]])
    state = replace(state, kc_site_resources=raw.new_tensor([[.1, .8, .4]]),
                    kc_site_utilization=site_parameters(model)[0].detach())
    values = advance_kc_terminals(model, state, raw, spikes=raw)
    torch.testing.assert_close(values[3][0], state.kc_site_resources[0])
    expected = raw.new_tensor([[.45, .4]])
    for route in ('comp', 'kcd'):
        torch.testing.assert_close(kc_arrival(model, state, route, new_values=values, instant=True), expected)
    model.kc_site_stp_enabled = False
    bypass = advance_kc_terminals(model, state, raw, spikes=raw)
    torch.testing.assert_close(bypass[3][0], torch.ones(3))
    # A forward bypass preserves prior resource history for matched controls.
    torch.testing.assert_close(bypass[6], state.kc_site_resources)


def test_full_local_state_roundtrip_detach_and_missing_history_rejection(tmp_path):
    model, state, _ = setup(tmp_path)
    attach_kc_site_stp(model)
    raw = state.h.new_tensor([[0, 0, 0, 1, 1, 0]])
    first = advance_kc_terminals(model, state, raw, spikes=raw)
    state = replace(state, **dict(zip(KC_TERMINAL_STATE_FIELDS, first))).detached()
    expected = advance_kc_terminals(model, state, raw, spikes=raw)
    resumed = type(state)(**copy.deepcopy(state.state_dict()))
    actual = advance_kc_terminals(model, resumed, raw, spikes=raw)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match='Incomplete'):
        site_state(model, replace(state, kc_site_utilization=torch.empty(0)))
    with pytest.raises(ValueError, match='Incompatible'):
        site_state(model, replace(state, kc_site_utilization=torch.zeros(1, 2)))
    with pytest.raises(ValueError, match='spike events'):
        advance_kc_terminals(model, state, raw)


def test_refuse_double_global_stp_and_incomplete_parameter_restore(tmp_path):
    model, _, _ = setup(tmp_path)
    model.use_stp = True
    with pytest.raises(ValueError, match='simultaneous'):
        attach_kc_site_stp(model)
    model.use_stp = False
    with pytest.raises(ValueError, match='Incomplete'):
        attach_kc_site_stp(model, saved={})


def test_learned_parameters_roundtrip_and_nonfinite_restore_rejected(tmp_path):
    model, _, _ = setup(tmp_path)
    attach_kc_site_stp(model)
    with torch.no_grad():
        model.kc_site_log_tau_rec.add_(.7)
    saved = {n: getattr(model, n).detach().clone() for n in KC_SITE_STP_PARAMETER_NAMES}
    restored = copy.deepcopy(model)
    attach_kc_site_stp(restored, saved=saved)
    for name in KC_SITE_STP_PARAMETER_NAMES:
        torch.testing.assert_close(getattr(restored, name), saved[name], rtol=0, atol=0)
    saved['kc_site_log_tau_rec'][0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        attach_kc_site_stp(restored, saved=saved)
