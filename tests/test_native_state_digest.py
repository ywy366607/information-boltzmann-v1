"""State identity includes empty shape, without copying a full tensor payload."""
import numpy as np
import torch

from scripts.ib.fly_interface.action_loop_v2_audit import state_digest


def test_empty_multidimensional_tensor_and_array_digest_retains_shape():
    assert state_digest(torch.empty(0, 3)) == state_digest(torch.empty(0, 3))
    assert state_digest(torch.empty(0, 3)) != state_digest(torch.empty(0, 4))
    assert state_digest(np.empty((2, 0))) == state_digest(np.empty((2, 0)))
    assert state_digest(np.empty((2, 0))) != state_digest(np.empty((3, 0)))


def test_digest_matches_contiguous_value_copy_and_detects_changes():
    value = torch.arange(8).reshape(2, 4).T
    assert state_digest(value) == state_digest(value.contiguous())
    assert state_digest(value) != state_digest(value + 1)
    array = np.arange(8).reshape(2, 4).T
    assert state_digest(array) == state_digest(array.copy())
