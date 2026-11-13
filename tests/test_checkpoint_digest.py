import copy

import numpy as np
import torch

from scripts.ib_local.checkpoint_digest import state_digest


def test_digest_covers_values_rng_optimizer_history_and_integer_layer_keys():
    state = {'S': {1: torch.arange(8).reshape(2, 4), 12: np.ones((2, 3), np.float32)},
             'optimizer': {'step': 5, 'm': torch.ones(4)}, 'rng': torch.get_rng_state(),
             'pending': [3, 4], 'eligibility': {'owner': np.eye(2)}}
    reference = state_digest(state)['sha256']
    reordered = dict(reversed(list(state.items())))
    assert state_digest(reordered)['sha256'] == reference
    for field in ('S', 'optimizer', 'rng', 'pending', 'eligibility'):
        changed = copy.deepcopy(state)
        if field == 'S': changed[field][1][0, 0] += 1
        elif field == 'optimizer': changed[field]['step'] += 1
        elif field == 'rng': changed[field][0] ^= 1
        elif field == 'pending': changed[field].append(5)
        else: changed[field]['owner'][0, 0] += 1
        assert state_digest(changed)['sha256'] != reference


def test_alias_cache_does_not_change_value_identity_and_stride_is_checked():
    x = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6)
    shared = state_digest([x, x, x[:, ::2]])
    copied = state_digest([x.clone(), x.clone(), x[:, ::2]])
    assert shared['sha256'] == copied['sha256']
    assert shared['tensor_occurrences'] == 3 and shared['unique_tensor_views'] == 2
    assert shared['largest_tensor_buffer_bytes'] == 48
    assert state_digest(x.T)['sha256'] != state_digest(x.T.contiguous())['sha256']


def test_parameter_layout_slices_and_typed_metadata_are_not_ignored():
    value = {'owners': {'gate': slice(0, 4096)}, 'dtype': torch.float32,
             'device': torch.device('cpu'), 'flags': {1, 12}, 'n': np.int64(8)}
    digest = state_digest(value)['sha256']
    assert state_digest(copy.deepcopy(value))['sha256'] == digest
    value['owners']['gate'] = slice(4096, 8192)
    assert state_digest(value)['sha256'] != digest
