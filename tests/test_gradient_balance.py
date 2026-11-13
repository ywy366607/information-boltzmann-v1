"""Algebra checks for reporting actual old-credit contributions."""
import pytest
import torch

from scripts.ib_local.audit_pretrained_gdn_gradient_balance import relation,group_indices


def test_relation_distinguishes_small_correction_alignment_and_cancellation():
    current = [torch.tensor([1.,0.])]
    aligned = relation(current,[torch.tensor([.1,0.])])
    orthogonal = relation(current,[torch.tensor([0.,.1])])
    opposite = relation(current,[torch.tensor([-1.,0.])])
    assert aligned['old_to_current'] == pytest.approx(.1)
    assert aligned['cosine'] == pytest.approx(1.)
    assert orthogonal['cosine'] == 0
    assert orthogonal['old_orthogonal_norm'] == pytest.approx(.1)
    assert opposite['combined_norm'] == 0
    assert opposite['combined_to_sum'] == 0


def test_zero_gradient_does_not_produce_invented_cosine_or_ratio():
    result = relation([torch.zeros(3)],[torch.ones(3)])
    assert result['old_to_current'] is None
    assert result['cosine'] is None
    assert result['old_orthogonal_norm'] == pytest.approx(3**.5)


def test_groups_preserve_layer_and_forget_write_roles():
    names = ['layer1.a_proj.weight','layer1.A_log','layer1.b_proj.weight',
             'layer12.a_proj.weight','layer1.q_proj.weight']
    groups = group_indices(names,1)
    assert groups['all'] == [0,1,2,4]
    assert groups['forget'] == [0,1]
    assert groups['write'] == [2]
    assert groups['q_proj'] == [4]
