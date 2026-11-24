"""Storage placement preserves layouts, gradients and saved-value identities."""
import gc

import pytest
import torch

from scripts.native_policy_activation_storage import activation_storage, MIB


@pytest.mark.parametrize('mode,gpu,host', [('device', 0, 0), ('disk', 0, 0),
                                         ('tiered', 4*MIB, 0), ('tiered', 0, 4*MIB),
                                         ('tiered', 1024, 16384)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA storage paths')
def test_storage_gradient_layout_and_budgets(tmp_path, mode, gpu, host):
    torch.manual_seed(11)
    x = torch.randn(80, 64, device='cuda', requires_grad=True)
    w = torch.nn.Parameter(torch.randn(40, 64, device='cuda'))

    def forward():
        # Nonzero offset, transposed stride, parameter alias, repeated save.
        a = (x * 1.25)[3:75:2].t()
        return ((w @ a).sin().square()).sum()

    expected = torch.autograd.grad(forward(), (x, w))
    with activation_storage([w], tmp_path/'spill', threshold_bytes=1, mode=mode,
                            gpu_budget_bytes=gpu, host_budget_bytes=host) as (r, hooks):
        with hooks():
            loss = forward()
        actual = torch.autograd.grad(loss, (x, w))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    gc.collect()
    assert r['gpu_live_bytes'] == r['host_live_bytes'] == 0
    assert r['gpu_peak_bytes'] <= gpu
    assert r['host_peak_bytes'] <= host
    assert not list((tmp_path/'spill').glob('*'))
    if mode == 'disk':
        assert r['disk_written_bytes'] > 0
    if mode == 'tiered' and host == 4*MIB:
        assert r['host_packed_bytes'] > 0 and r['disk_written_bytes'] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA storage paths')
def test_saved_values_survive_mutation_and_repeated_backward(tmp_path):
    x = torch.arange(4096, device='cuda', dtype=torch.float32)
    with activation_storage([], tmp_path, threshold_bytes=1, mode='disk') as (r, hooks):
        h = hooks()
        with h:
            first = h.pack_hook(x)
            assert h.pack_hook(x) is first
            x.add_(3)
            second = h.pack_hook(x)
            torch.testing.assert_close(h.unpack_hook(first), x-3, atol=0, rtol=0)
            torch.testing.assert_close(h.unpack_hook(second), x, atol=0, rtol=0)
            torch.testing.assert_close(h.unpack_hook(first), x-3, atol=0, rtol=0)
            assert r['deduplicated_saves'] == 1
        del first, second
    assert not list(tmp_path.glob('*'))


def test_invalid_storage_options(tmp_path):
    with pytest.raises(ValueError):
        with activation_storage([], tmp_path, host_budget_bytes=-1):
            pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA admission timing')
def test_tiered_residency_is_refreshed_at_first_save_after_planning(tmp_path, monkeypatch):
    calls, reads = [], []
    w = torch.nn.Parameter(torch.ones(64, device='cuda'))

    def free(device):
        reads.append(1)
        return (768*MIB if len(reads) == 1 else 1024*MIB, 4096*MIB)

    monkeypatch.setattr(torch.cuda, 'mem_get_info', free)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: calls.append('release unused blocks'))
    with activation_storage([w], tmp_path, threshold_bytes=1,
                            gpu_budget_bytes=4*MIB, host_budget_bytes=0) as (receipt, hooks):
        assert receipt['gpu_budget_bytes'] == 0
        with hooks():
            loss = (w*2).sin().square().sum()
        loss.backward()
    assert receipt['admission_refreshed'] and receipt['gpu_budget_bytes'] == 4*MIB
    assert receipt['gpu_peak_bytes'] > 0 and receipt['disk_written_bytes'] == 0
    assert len(calls) == 1 and len(reads) == 2
