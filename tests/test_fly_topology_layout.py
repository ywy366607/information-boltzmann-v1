"""Exact topology order and memory-bound construction contracts."""
import numpy as np
import pytest

from information_boltzmann.core.fly_topology_layout import incoming_arrays


@pytest.mark.parametrize('chunk', [1, 3, 31, 4096])
@pytest.mark.parametrize('dtype', [np.int32, np.int64])
def test_chunked_layout_preserves_full_stable_order_across_tiers(chunk, dtype):
    rng = np.random.default_rng(14)
    pre = rng.integers(0, 19, 257, dtype=dtype)
    post = rng.integers(0, 19, 257, dtype=dtype)
    splits = (0, 120, 120, 251, 257)
    actual = incoming_arrays(pre, post, splits, 21, chunk_edges=chunk)
    for (start, end), (order, source, offsets) in zip(zip(splits, splits[1:]), actual):
        expected_order = np.argsort(post[start:end], kind='stable') + start
        np.testing.assert_array_equal(order, expected_order)
        np.testing.assert_array_equal(source, pre[expected_order])
        np.testing.assert_array_equal(offsets, np.r_[0, np.cumsum(np.bincount(post[start:end], minlength=21))])
        assert order.dtype == source.dtype == offsets.dtype == np.int32


def test_cached_layout_is_mapped_and_invalidates_when_topology_changes(tmp_path, monkeypatch):
    pre = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    post = np.array([2, 0, 2, 0, 1], dtype=np.int64)
    args = (pre, post, (0, 0, 5), 3)
    cold = incoming_arrays(*args, chunk_edges=2, cache_dir=tmp_path)
    expected = tuple(tuple(a.copy() for a in tier) for tier in cold)
    def no_sort(*args, **kwargs):
        raise AssertionError('A warm topology cache must avoid sorting')
    with monkeypatch.context() as patch:
        patch.setattr(np, 'argsort', no_sort)
        warm = incoming_arrays(*args, chunk_edges=2, cache_dir=tmp_path)
        for tier, reference in zip(warm, expected):
            for array, wanted in zip(tier, reference):
                assert isinstance(array, np.memmap)
                np.testing.assert_array_equal(array, wanted)
    changed = pre.copy()
    changed[0] = 2
    updated = incoming_arrays(changed, post, (0, 0, 5), 3, cache_dir=tmp_path)
    assert len(list(tmp_path.glob('*/complete.json'))) == 2
    assert not np.array_equal(updated[1][1], warm[1][1])


def test_sort_and_gather_never_exceed_the_requested_block(monkeypatch):
    rng = np.random.default_rng(3)
    pre = rng.integers(0, 25, 11003)
    post = rng.integers(0, 25, 11003)
    sort = np.argsort
    sizes = []
    def check_sort(value, **kwargs):
        sizes.append(len(value))
        assert len(value) <= 127
        return sort(value, **kwargs)
    monkeypatch.setattr(np, 'argsort', check_sort)
    incoming_arrays(pre, post, (0, len(pre)), 25, chunk_edges=127)
    assert len(sizes) > 1 and max(sizes) == 127
