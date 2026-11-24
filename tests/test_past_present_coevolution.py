"""Numerical/causal interfaces, not a synthetic capability experiment."""
import copy
import importlib.util

import pytest
import torch

from scripts.ib_local.causal_workspace_scan import causal_scan, reference_scan
from scripts.ib_local.past_present_coevolution import configure_coevolution, refresh_workset


def engine():
    # Reuse the complete native graph's existing bounded tensor fixture.
    spec = importlib.util.spec_from_file_location('native_fixture', 'tests/test_native_navigation_body.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    cfg, e, _ = fixture.construct()
    return cfg, e


SPEC = dict(schema=1, layers='all', slots=8, rank=8, seed=151132)


def populated(e):
    memory = e.model.memory
    with torch.no_grad():
        memory.occupied[:8] = True
        for name in ('source', 'action', 'successor', 'long_successor'):
            getattr(memory, name)[:8].normal_()
        memory.pending_count = 1
        memory.pending[-1].copy_(memory.source[3])
    return refresh_workset(e)


def test_no_field_has_exact_old_forward_and_preserves_physical_state():
    _, e = engine()
    ids = torch.tensor([[9, 7, 3, 2, 8, 6, 5, 4]])
    lp = torch.full((1, 8, 256), -5.54517744448)
    before = e.recurrent.clone()
    with torch.no_grad():
        original = e.model.predict(ids, before, lp)
    old_parameters = list(e.params)
    e.prev = dict(actor=tuple(torch.ones_like(p) for p in old_parameters))
    configure_coevolution(e, SPEC)
    torch.testing.assert_close(e.recurrent[..., :before.shape[-1]], before, rtol=0, atol=0)
    for p, g in zip(e.params, e.prev['actor']):
        assert (g is not None) == any(p is old for old in old_parameters)
    with torch.no_grad():
        result = e.model.predict(ids, e.recurrent, lp)
    torch.testing.assert_close(result[0], original[0], rtol=0, atol=0)
    torch.testing.assert_close(result[1][..., :before.shape[-1]], original[1], rtol=0, atol=0)


def test_causal_future_exclusion_chunk_and_single_token_equivalence():
    _, e = engine()
    configure_coevolution(e, SPEC)
    populated(e)
    ids = torch.tensor([[9, 7, 3, 2, 8, 6, 5, 4]])
    lp = torch.full((1, 8, 256), -5.54517744448)
    with torch.no_grad():
        whole = e.model.predict(ids, e.recurrent, lp)
        changed = ids.clone(); changed[:, 4:] = 80
        alternate = e.model.predict(changed, e.recurrent, lp)
        torch.testing.assert_close(whole[0][:, :4], alternate[0][:, :4], rtol=0, atol=0)
        state = e.recurrent
        pieces = []
        for t in range(8):
            item = e.model.predict(ids[:, t:t+1], state, lp[:, t:t+1])
            pieces.append(item[0]); state = item[1]
    torch.testing.assert_close(torch.cat(pieces, 1), whole[0], atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(state, whole[1], atol=3e-6, rtol=3e-5)


def test_both_directions_and_every_layer_receive_real_loss_gradient():
    _, e = engine()
    configure_coevolution(e, SPEC)
    populated(e)
    m = e.model
    memory_before = m.memory.source.clone()
    m.memory.begin_read_credit()
    ids = torch.tensor([[11, 12, 13, 14, 15, 16]])
    lp = torch.full((1, 6, 256), -5.54517744448)
    logits, state, diagnostics = m.predict(ids, e.recurrent, lp)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), (ids+1).flatten())
    loss.backward()
    field = m.body.coevolution
    assert field.bind.weight.grad.norm() > 0
    assert field.memory_ffn[-1].weight.grad.norm() > 0
    assert field.memory_gate.weight.grad.norm() > 0
    for i, layer in enumerate(m.base.layers):
        assert layer.q_proj.weight.grad.norm() > 0
        assert diagnostics[f'l{i}_workspace_residual_rms'] > 0
        assert state[i, :, -field.state_size:].norm() > 0
    torch.testing.assert_close(m.memory.source, memory_before, rtol=0, atol=0)
    receipt = m.memory.settle_read_credit()
    assert receipt['records'] > 0


def test_versioned_transport_cannot_inherit_recycled_address_state():
    _, e = engine()
    configure_coevolution(e, SPEC)
    populated(e)
    f = e.model.body.coevolution
    e.recurrent[..., -f.state_size:] = 1
    refresh_workset(e)
    assert e.recurrent[..., -f.state_size:].min() == 1
    e.model.memory.read_credit_version[:8] += 1
    refresh_workset(e)
    assert e.recurrent[..., -f.state_size:].count_nonzero() == 0


def test_restore_reconstructs_same_full_graph_and_optimizer_group_order():
    cfg, e = engine()
    configure_coevolution(e, SPEC)
    populated(e)
    cfg['past_present_coevolution'] = copy.deepcopy(SPEC)
    from scripts.ib_local.native_navigation_body import construct_engine
    from scripts.ib_local.run_causal_water_reply import configure_components
    r = construct_engine(cfg, e.world, restoring=True)
    configure_components(r, cfg)
    r.model.load_state_dict(e.model.state_dict(), strict=True)
    r.optimizer.load_state_dict(e.optimizer.state_dict())
    assert [len(g['params']) for g in e.optimizer.param_groups] == [
        len(g['params']) for g in r.optimizer.param_groups]
    for a, b in zip(e.model.parameters(), r.model.parameters()):
        torch.testing.assert_close(a, b)
    ids = torch.tensor([[3, 5, 7]])
    lp = torch.full((1, 3, 256), -5.54517744448)
    with torch.no_grad():
        expected = e.model.predict(ids, e.recurrent, lp)
        actual = r.model.predict(ids, e.recurrent, lp)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_affine_scan_matches_reference_including_all_derivatives(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(15)
    a = torch.rand(1, 11, 3, 8, device=device, requires_grad=True)
    u = torch.randn_like(a, requires_grad=True)
    s = torch.randn(1, 3, 8, device=device, requires_grad=True)
    derivative = torch.randn_like(a)
    actual = causal_scan(a, u, s)
    expected = reference_scan(a, u, s)
    ga = torch.autograd.grad((actual*derivative).sum(), (a, u, s))
    ge = torch.autograd.grad((expected*derivative).sum(), (a, u, s))
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
    for x, y in zip(ga, ge):
        torch.testing.assert_close(x, y, atol=3e-6, rtol=2e-5)
