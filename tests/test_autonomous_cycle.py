"""Production scheduling/eligibility/persistence contracts, not ability tasks."""
import copy

import pytest
import torch

from test_generative_cycle import configured
from scripts.ib_local.autonomous_cycle import configure, poll, dispatch


def force_choices(monkeypatch, values):
    original = torch.multinomial
    choices = iter(values)
    def draw(weights, n, *a, **kw):
        try:
            choice = next(choices)
        except StopIteration:
            return original(weights, n, *a, **kw)
        assert weights.shape == (2,)
        return torch.tensor([choice], device=weights.device)
    monkeypatch.setattr(torch, 'multinomial', draw)


def test_internal_events_run_without_world_and_accumulate_exact_bounded_score(monkeypatch):
    torch.set_num_threads(1)
    e = configured()
    configure(e)
    before = copy.deepcopy(e.world.state_dict())
    clock = int(e.model.memory.clock)
    versions = e.model.memory.last_write.clone()
    params = [p.detach().clone() for p in e.model.parameters()]
    expected = torch.zeros_like(e.model.binding_choice.weight)
    original = e._sample_internal_choice
    def observed():
        nonlocal expected
        selected, choice, receipt = original()
        with torch.enable_grad():
            logits = e.model.binding_choice(choice['features']).flatten()+choice['prior']
            expected += torch.autograd.grad(logits.log_softmax(0)[selected], e.model.binding_choice.weight)[0].detach()
        return selected, choice, receipt
    monkeypatch.setattr(e, '_sample_internal_choice', observed)
    # Longer than the former per-world proposal budget, with no forced release.
    force_choices(monkeypatch, [1]*7+[0])
    for i in range(7):
        row = dispatch(e)
        assert row['autonomous_event']['kind'] == 'internal'
        assert row['external_bytes'] == row['updates'] == 0
        assert e.autonomous_choices == i+1 and not e.autonomous_ready
        assert len(e.autonomous_eligibility) == 1
        assert e.autonomous_eligibility[0].numel() == e.model.binding_choice.weight.numel()
    row = dispatch(e)
    assert row['autonomous_event']['kind'] == 'release' and e.autonomous_ready
    torch.testing.assert_close(e.autonomous_eligibility[0], expected)
    assert int(e.model.memory.clock) == clock
    assert torch.equal(versions, e.model.memory.last_write)
    assert e.world.state_dict() == before
    assert all(torch.equal(a, b) for a, b in zip(params, e.model.parameters()))
    with pytest.raises(RuntimeError, match='released'):
        poll(e)


def test_release_actual_world_and_same_delayed_actor_include_internal_eligibility(monkeypatch):
    torch.set_num_threads(1)
    e = configured(); configure(e)
    force_choices(monkeypatch, [1, 0])
    dispatch(e); dispatch(e)
    eligible = e.autonomous_eligibility[0].clone()
    row = dispatch(e)
    assert row['autonomous_event']['kind'] == 'external'
    assert row['external_bytes'] == 128
    assert row['generative_relation']['autonomous']['internal_steps'] == 1
    assert not e.autonomous_ready and e.autonomous_choices == 0
    assert torch.equal(e.autonomous_action_eligibility[0], eligible)
    index = next(i for i, p in enumerate(e.params) if p is e.model.binding_choice.weight)
    assert e.prev['actor'][index] is not None
    assert bool(e.prev['actor'][index].abs().sum())


def test_quiet_snapshot_restore_preserves_next_event_and_full_transition(monkeypatch):
    from scripts.ib_local.navigation_checkpoint import snapshot, restore
    torch.set_num_threads(1)
    e = configured(); configure(e)
    force_choices(monkeypatch, [1, 1])
    dispatch(e); dispatch(e)
    saved = snapshot(e)
    # Normal sampled selector and world event: same RNG and full state.
    first = dispatch(e)
    while not e.autonomous_ready:
        dispatch(e)
    actual = dispatch(e)
    weights = copy.deepcopy(e.model.state_dict())
    fresh = configured(); configure(fresh); restore(fresh, saved)
    second = dispatch(fresh)
    assert first == second
    while not fresh.autonomous_ready:
        dispatch(fresh)
    repeated = dispatch(fresh)
    for key in ('choice', 'utterance_hex', 'metal_g', 'generative_relation', 'autonomous_event'):
        assert actual[key] == repeated[key]
    for key, value in weights.items():
        if torch.is_tensor(value):
            assert torch.equal(value, fresh.model.state_dict()[key]), key


def test_direct_world_call_is_rejected_without_self_release():
    e = configured(); configure(e)
    with pytest.raises(RuntimeError, match='release itself'):
        e.step(torch.randn(8, 4))
