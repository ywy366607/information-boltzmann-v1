"""BPTT participation kept as an experience record.

The birth window has already applied these gradients. The record then waits.
A later consequence that names the slot scales that same tensor. The record
is not transported by a forget gate, and it is not recomputed from saved
inputs. Replacing the slot drops the participation with the experience.
"""
from __future__ import annotations

import math

import torch


class FrozenParticipation:
    def __init__(self, capacity: int):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('capacity must be a positive integer')
        self.capacity = capacity
        self._grads: list[dict[str, torch.Tensor] | None] = [None] * capacity
        self._window: list[int | None] = [None] * capacity

    def _slot(self, slot: int) -> None:
        if type(slot) is not int or not 0 <= slot < self.capacity:
            raise ValueError('slot is outside the experience table')

    def commit(self, slot: int, grads: dict[str, torch.Tensor], window: int) -> None:
        """Seal one window's participation into a slot, replacing any occupant."""
        self._slot(slot)
        if type(window) is not int or window < 0:
            raise ValueError('window must be a nonnegative integer')
        if not grads:
            raise ValueError('empty participation')
        frozen = {}
        for name, grad in grads.items():
            if not isinstance(grad, torch.Tensor) or grad.ndim == 0:
                raise ValueError(f'{name}: participation must be a parameter gradient')
            if not bool(torch.isfinite(grad).all()):
                raise ValueError(f'{name}: nonfinite participation')
            frozen[name] = grad.detach().to(dtype=torch.float32).clone()
        self._grads[slot] = frozen
        self._window[slot] = window

    def participation(self, slot: int) -> dict[str, torch.Tensor]:
        self._slot(slot)
        record = self._grads[slot]
        if record is None:
            raise ValueError('empty slot')
        return {name: grad.clone() for name, grad in record.items()}

    def contribute(self, slot: int, consequence: float, window: int) -> dict[str, torch.Tensor]:
        """Scale the sealed record by a consequence from a later window."""
        self._slot(slot)
        record, born = self._grads[slot], self._window[slot]
        if record is None or born is None:
            raise ValueError('empty slot')
        if type(window) is not int or window <= born:
            raise ValueError('a consequence cannot settle inside its birth window')
        scale = float(consequence)
        if not math.isfinite(scale):
            raise ValueError('nonfinite consequence')
        return {name: scale * grad for name, grad in record.items()}
