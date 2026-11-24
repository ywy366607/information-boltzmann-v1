"""Bounded decoding must preserve the exact inherited world byte sequence."""
import numpy as np
import pytest

from scripts.ib_local.world_source import decoded_world_bytes


@pytest.mark.parametrize('chunk', [1, 3, 16, 100])
def test_chunked_source_matches_original_decoding(chunk):
    raw = np.array([-3, 0, 2, 3, 4, 258, 259, 511, 700], dtype=np.int64)
    expected = (raw[raw >= 3] - 3).astype(np.uint8)
    assert np.array_equal(decoded_world_bytes(raw, 3, chunk), expected)
    assert np.array_equal(raw, [-3, 0, 2, 3, 4, 258, 259, 511, 700])


def test_source_handles_empty_selection_and_invalid_shape():
    assert decoded_world_bytes(np.arange(5), 10, 2).size == 0
    with pytest.raises(ValueError):
        decoded_world_bytes(np.zeros((2, 2)), 0)


def test_private_checkpoint_table_reuses_storage_without_changing_default_copy(tmp_path):
    import torch
    from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable

    original = RetrievedCreditTable(orders=(0, 1), heads=1, slots_log2=2)
    original.run_credit(np.frombuffer(b'abcabc', dtype=np.uint8))
    state = original.state_dict()
    saved_rows = state['arrays']['rows'].copy()
    # Checkpoint writer persists table arrays as CPU tensors.
    state['arrays'] = {k: torch.from_numpy(v) for k, v in state['arrays'].items()}
    copied = RetrievedCreditTable.from_state_dict(state)
    owned = RetrievedCreditTable.from_state_dict(state, take_ownership=True)
    disk = RetrievedCreditTable.from_state_dict(state, take_ownership=True,
                                                storage_directory=tmp_path/'workspace')
    assert isinstance(disk.rows, np.memmap)
    assert np.shares_memory(owned.rows, state['arrays']['rows'].numpy())
    assert not np.shares_memory(copied.rows, state['arrays']['rows'].numpy())
    assert np.array_equal(owned.rows, saved_rows)
    assert owned.events == copied.events == original.events
    stream = np.frombuffer(b'cabd', dtype=np.uint8)
    owned_result = owned.run_credit(stream)
    copied_result = copied.run_credit(stream)
    disk_result = disk.run_credit(stream)
    assert np.array_equal(owned_result[0], copied_result[0])
    assert np.array_equal(owned_result[1], copied_result[1])
    assert np.array_equal(disk_result[0], copied_result[0])
    assert np.array_equal(disk_result[1], copied_result[1])
    assert np.array_equal(owned.rows, copied.rows)
    assert np.array_equal(disk.rows, copied.rows)


def test_disk_table_birth_and_credit_match_ram_and_preserve_existing_files(tmp_path):
    from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
    config = dict(orders=(0, 1, 3), heads=2, slots_log2=4)
    ram = RetrievedCreditTable(**config)
    disk = RetrievedCreditTable(**config, storage=tmp_path/'table')
    rng = np.random.default_rng(27)
    for _ in range(3):
        stream = rng.integers(0, 256, 65, dtype=np.uint8)
        a, b = ram.run_credit(stream), disk.run_credit(stream)
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])
        factor = rng.normal(size=(64, 256))
        assert ram.settle(a[2], factor) == disk.settle(b[2], factor)
    disk.flush()
    for name in disk._array_specs(len(disk.count)):
        assert isinstance(getattr(disk, name), np.memmap)
        assert np.array_equal(getattr(disk, name), getattr(ram, name))
        assert np.array_equal(np.load(tmp_path/'table'/f'{name}.npy'), getattr(ram, name))
    before = (tmp_path/'table'/'rows.npy').read_bytes()
    with pytest.raises(FileExistsError):
        RetrievedCreditTable(**config, storage=tmp_path/'table')
    assert (tmp_path/'table'/'rows.npy').read_bytes() == before
