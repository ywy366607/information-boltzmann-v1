"""Exact reversible writes and sidecar equivalence, not capability studies."""
import copy
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.table_branch import FIELDS, table_branch


def assert_same(table, state):
    assert int(table.history) == state['history'] and table.events == state['events']
    for name in FIELDS:
        assert np.array_equal(getattr(table, name), state['arrays'][name]), name


def test_branch_matches_full_copy_with_collisions_and_restores_every_field():
    live = RetrievedCreditTable(orders=(0,1,2,6), heads=2, slots_log2=2)
    live.run_credit(np.arange(25, dtype=np.uint8))
    before = live.state_dict()
    expected = copy.deepcopy(live)
    with table_branch(live) as branch:
        for data in (b'context abc', b'other bytes', b'context abc'):
            raw = np.frombuffer(data, np.uint8)
            a = type(branch).run_credit(branch, raw)
            b = type(expected).run_credit(expected, raw)
            for actual, reference in zip(a[:2], b[:2]):
                assert np.array_equal(actual, reference)
            for name in ('slots','epochs','ages','eligibility'):
                assert np.array_equal(getattr(a[2],name), getattr(b[2],name))
        assert_same(branch, expected.state_dict())
    assert_same(live, before)
    with pytest.raises(RuntimeError, match='closed'):
        branch.run_credit(np.array([1,2], np.uint8))


def test_branch_rolls_back_on_exception_and_copies_only_addressed_rows():
    live = RetrievedCreditTable(orders=(0,1,2,3), heads=2, slots_log2=7)
    before = live.state_dict()
    with pytest.raises(ValueError, match='intentional'):
        with table_branch(live) as branch:
            branch.run_credit(np.array([1,2], np.uint8))
            assert branch.journal_bytes < live.nbytes/10
            raise ValueError('intentional')
    assert_same(live, before)


def test_branch_rejects_unjournaled_occupancy_transfer():
    live = RetrievedCreditTable(orders=(0,1), heads=1, slots_log2=2)
    before = live.state_dict()
    with table_branch(live) as branch:
        branch.run_credit(np.array([1,2,3], np.uint8))
        with pytest.raises(RuntimeError, match='occupancy reinvestment'):
            branch.settle(None, None)
    assert_same(live, before)


def test_sidecar_journal_is_identical_to_old_full_copy_evaluation(monkeypatch):
    from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
    from scripts.ib_local.train_gdn_fast_table import TableGDN
    from scripts.ib_local.latent_navigation_v3 import LatentNavigationV3
    from scripts.ib_local import run_language_navigation_v2 as runner
    torch.manual_seed(7)
    model = LatentNavigationV3(TableGDN(GatedDeltaNetLM(256,16,2,2,'decay_then_delta',32,4,32.),True),
        latent_dim=8, capacity=8, short=1, long=4, topk=2, history_size=4, max_chunk=4)
    table = RetrievedCreditTable(orders=(0,1,2), heads=2, slots_log2=3)
    valid = np.arange(17, dtype=np.uint8)
    original = table.state_dict()
    actual = runner.validate(model, table, valid, chunk=4)
    assert_same(table, original)
    @contextmanager
    def former_full_copy(parent):
        yield copy.deepcopy(parent)
    monkeypatch.setattr(runner, 'table_branch', former_full_copy)
    expected = runner.validate(model, table, valid, chunk=4)
    assert actual == expected
    assert_same(table, original)
