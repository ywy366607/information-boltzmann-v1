"""The sealed participation waits; it is not a second copy of this window's BPTT."""
import math

import pytest
import torch

from scripts.ib_local.frozen_participation import FrozenParticipation


def test_later_consequence_scales_the_sealed_record_and_alpha_does_not():
    table = FrozenParticipation(4)
    birth = torch.tensor([1., 0., -2.])
    table.commit(1, {'gate': birth}, window=0)
    sealed = table.participation(1)['gate']

    for _ in range(256):
        assert torch.equal(table.participation(1)['gate'], sealed)
    decayed = sealed * math.exp(-10.2)
    assert not torch.allclose(table.participation(1)['gate'], decayed)

    with pytest.raises(ValueError, match='birth window'):
        table.contribute(1, consequence=0.5, window=0)

    fresh = torch.tensor([0., 4., 0.])
    delayed = table.contribute(1, consequence=0.5, window=300)
    assert torch.equal(delayed['gate'], 0.5 * sealed)
    assert torch.equal(fresh + delayed['gate'], torch.tensor([0.5, 4., -1.]))
    assert torch.equal(table.participation(1)['gate'], sealed)


def test_replacing_the_slot_drops_the_old_participation():
    table = FrozenParticipation(2)
    table.commit(0, {'gate': torch.tensor([3., 0.])}, window=1)
    table.commit(0, {'gate': torch.ones(2)}, window=4)
    assert torch.equal(table.contribute(0, consequence=1., window=5)['gate'], torch.ones(2))


def test_a_consequence_names_one_slot():
    table = FrozenParticipation(2)
    table.commit(0, {'gate': torch.tensor([1.])}, window=0)
    table.commit(1, {'gate': torch.tensor([8.])}, window=2)
    assert torch.equal(table.contribute(1, consequence=-1., window=3)['gate'], torch.tensor([-8.]))
    assert torch.equal(table.participation(0)['gate'], torch.tensor([1.]))
