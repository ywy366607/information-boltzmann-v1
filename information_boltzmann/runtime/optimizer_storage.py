"""Exact host staging for optimizer tensors unused by forward and backward."""
from contextlib import contextmanager

import torch


@contextmanager
def optimizer_state_on_host(optimizer, *, enabled=False, device_types=('cuda',),
                            release_cached_memory=False):
    """Temporarily move nonscalar state to CPU, restoring before the update.

    Standard PyTorch Adam/AdamW states are flat dictionaries of tensor moments
    and scalar counters. Parameters, gradients and counters are untouched. This
    scope must end before ``step`` or serialization; it cannot be used around
    a captured graph. Synchronous, same-dtype copies preserve every moment bit.
    ``device_types=('cpu',)`` exercises the copy/restore contract in CPU tests.
    """
    staged = []
    try:
        if enabled:
            for state in optimizer.state.values():
                for key, value in tuple(state.items()):
                    if (isinstance(value, torch.Tensor) and value.numel() > 1
                            and value.device.type in device_types):
                        host = value.detach().to(device='cpu', copy=True)
                        staged.append((state, key, value.device))
                        state[key] = host
                        # Do not retain the old CUDA storage in this frame.
                        del value, host
        if release_cached_memory and any(device.type == 'cuda' for _, _, device in staged):
            # Only unused allocator segments are released. The live physical
            # state, parameters and pending gradients retain their addresses.
            torch.cuda.empty_cache()
        yield
    finally:
        if release_cached_memory and any(device.type == 'cuda' for _, _, device in staged):
            torch.cuda.empty_cache()
        for state, key, device in staged:
            state[key] = state[key].to(device=device, copy=True)
