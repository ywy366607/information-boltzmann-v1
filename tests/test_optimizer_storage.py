"""Host staging changes storage residency, never Adam's learning state."""
import copy

import pytest
import torch

from information_boltzmann.runtime.optimizer_storage import optimizer_state_on_host


def assert_state_equal(left, right):
    assert left.keys() == right.keys()
    for key, value in left.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, right[key], atol=0, rtol=0)
        else:
            assert value == right[key]


def test_host_copies_preserve_adam_updates_none_zero_and_exception_restore():
    a = torch.nn.Parameter(torch.arange(6., dtype=torch.float64).reshape(2, 3).T)
    b = torch.nn.Parameter(a.detach().clone())
    oa = torch.optim.AdamW([a], lr=.0002, weight_decay=.02)
    ob = torch.optim.AdamW([b], lr=.0002, weight_decay=.02)
    for gradient in (torch.ones_like(a), None, torch.zeros_like(a), torch.full_like(a, .7)):
        a.grad = None if gradient is None else gradient.clone()
        b.grad = None if gradient is None else gradient.clone()
        snapshots = copy.deepcopy(ob.state[b])
        old_ids = {key: id(value) for key, value in ob.state[b].items()}
        with optimizer_state_on_host(ob, enabled=True, device_types=('cpu',)):
            assert_state_equal(snapshots, ob.state[b])
            for key, value in ob.state[b].items():
                assert (id(value) != old_ids[key]) == (value.numel() > 1)
        oa.step()
        ob.step()
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert_state_equal(oa.state[a], ob.state[b])
    snapshots = copy.deepcopy(ob.state[b])
    with pytest.raises(RuntimeError, match='injected'):
        with optimizer_state_on_host(ob, enabled=True, device_types=('cpu',)):
            raise RuntimeError('injected')
    assert_state_equal(snapshots, ob.state[b])


def test_disabled_and_default_cpu_scope_preserve_storage():
    p = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.AdamW([p])
    p.grad = torch.ones_like(p)
    optimizer.step()
    values = dict(optimizer.state[p])
    for enabled in (False, True):
        with optimizer_state_on_host(optimizer, enabled=enabled):
            assert all(value is optimizer.state[p][key] for key, value in values.items())
