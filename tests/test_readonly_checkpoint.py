import gc
import hashlib

import torch

from scripts.ib_local.readonly_checkpoint import load_readonly_checkpoint


def test_values_alias_stride_lifetime_and_clone_updates_leave_file_unchanged(tmp_path):
    path = tmp_path/'state.pt'
    original = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    torch.save(dict(weights=original, view=original[:, ::2], integer=torch.tensor([5, 7]),
                    history=('pending', 14), optimizer={'moment': original + 1}), path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    value = load_readonly_checkpoint(path)
    gc.collect()
    assert torch.equal(value['weights'], original)
    assert value['view'].stride() == original[:, ::2].stride()
    assert value['weights'].untyped_storage().data_ptr() == value['view'].untyped_storage().data_ptr()
    assert value['history'] == ('pending', 14)
    live = value['weights'].clone().requires_grad_()
    live.square().mean().backward()
    with torch.no_grad():
        live.add_(live.grad, alpha=-.1)
    assert torch.equal(value['weights'], original)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_repeated_loads_have_independent_metadata_and_same_tensor_values(tmp_path):
    path = tmp_path/'state.pt'
    torch.save({'capsules': {1: torch.arange(16)}, 'count': 5}, path)
    a, b = load_readonly_checkpoint(path), load_readonly_checkpoint(path)
    a['capsules'][1] = torch.zeros(2)
    assert torch.equal(b['capsules'][1], torch.arange(16))
