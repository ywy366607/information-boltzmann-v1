"""Topology packing equivalence; no behavioral/capability experiment."""
import numpy as np
import pytest
import torch

from information_boltzmann.core.triton_synapse import _pulse_run_arrays, _outgoing_tier_arrays, PULSE_BLOCK


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_large_strided_score_finiteness_is_checked_with_bounded_temporaries(monkeypatch, device):
    from information_boltzmann.core.fly_native_policy_credit import _finite
    value = torch.zeros((1 << 20) + 17, 2, device=device).T
    assert not value.is_contiguous()
    original = torch.isfinite
    sizes = []
    def bounded(block):
        sizes.append(block.numel())
        assert block.numel() <= 1 << 20
        return original(block)
    monkeypatch.setattr(torch, 'isfinite', bounded)
    _finite(value, 'strided score')
    assert len(sizes) > 1
    value[-1, -1] = float('nan')
    with pytest.raises(ValueError, match='Non-finite'):
        _finite(value, 'strided score')


def reference(pre, splits, neurons):
    key = pre.long().clone()
    for tier, (a, b) in enumerate(zip(splits, splits[1:])):
        key[a:b] += tier * neurons
    position = torch.arange(key.numel())
    new = torch.ones(key.numel(), dtype=torch.bool)
    new[1:] = (key[1:] != key[:-1]) | (position[1:] % PULSE_BLOCK == 0)
    piece = torch.cumsum(new.long(), 0) - 1
    first = piece[::PULSE_BLOCK]
    local = (piece - first[position // PULSE_BLOCK]).to(torch.uint8)
    slot_ptr = torch.zeros(4 * neurons + 1, dtype=torch.int32)
    edge_ptr = slot_ptr.clone()
    slot_ptr[1:] = torch.cumsum(torch.bincount(key[new], minlength=4 * neurons), 0).int()
    edge_ptr[1:] = torch.cumsum(torch.bincount(key, minlength=4 * neurons), 0).int()
    return local, first.int(), slot_ptr, key[new].int(), edge_ptr


@pytest.mark.parametrize('chunk_edges', [257, 1024, 65536])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_chunked_layout_is_bitwise_original_across_tiers_blocks_and_chunks(chunk_edges, device):
    rng = np.random.default_rng(7)
    splits, neurons = (0, 257, 66000, 66000, 140123), 67
    values = np.concatenate([np.sort(rng.integers(0, neurons, b-a, dtype=np.int32))
                             for a, b in zip(splits, splits[1:])])
    pre = torch.from_numpy(values)
    expected = reference(pre, splits, neurons)
    actual = _pulse_run_arrays(pre.to(device), splits, neurons, chunk_edges=chunk_edges)
    for got, want in zip(actual, expected):
        assert np.array_equal(got, want.numpy())
        assert got.dtype == want.numpy().dtype


def test_chunked_layout_rejects_disorder_across_chunk_boundary():
    pre = torch.cat((torch.ones(256, dtype=torch.int32), torch.zeros(3, dtype=torch.int32)))
    assert _pulse_run_arrays(pre, (0, 259, 259, 259, 259), 2, chunk_edges=256) is None
    assert _pulse_run_arrays(pre[:0], (0, 0, 0, 0, 0), 2) is None


@pytest.mark.parametrize('chunk_edges', [257, 65536])
def test_chunked_outgoing_layout_matches_whole_stable_sort(chunk_edges):
    rng = np.random.default_rng(8)
    pre = torch.from_numpy(rng.integers(0, 31, 140123, dtype=np.int32))
    for start, end in ((0, 257), (257, 66000), (66000, 66000), (66000, pre.numel())):
        order, ptr = _outgoing_tier_arrays(pre, start, end, 31, chunk_edges=chunk_edges)
        expected = (torch.argsort(pre[start:end].long(), stable=True) + start).int()
        assert np.array_equal(order, expected.numpy())
        counts = torch.bincount(pre[start:end].long(), minlength=31)
        assert np.array_equal(ptr, torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0))).numpy())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_model_device_transfer_preserves_host_fallback_and_full_transmission(tmp_path):
    from test_fly_reservoir import make_tiny
    from information_boltzmann.core.fly_reservoir import FlyReservoirLM
    from information_boltzmann.core.triton_synapse import PyTorchDelayedSynapticTransmission
    graph, _ = make_tiny(tmp_path)
    model = FlyReservoirLM(graph, vocab_size=23, d_model=16,
                           synapse_model='coba', transmission_mode='incoming').cuda()
    names = [name for tiers in model._incoming_layout_names.values() for tier in tiers for name in tier]
    assert names and all(getattr(model, name).device.type == 'cpu' for name in names)
    assert model.edge_pre_e.is_cuda
    pulses = tuple(torch.ones(1, model.n_neurons, device='cuda', requires_grad=True) for _ in range(4))
    actual = model.transmit(pulses, 'e')
    expected = PyTorchDelayedSynapticTransmission.apply(*pulses, model.edge_pre_e,
        model.edge_post_e, model.edge_weight_e, model.splits_e)
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual.sum(), pulses)
    gb = torch.autograd.grad(expected.sum(), pulses)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b)
    model.cpu()
    assert all(getattr(model, name).device.type == 'cpu' for name in names)
