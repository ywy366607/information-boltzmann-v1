"""D3Q8 geometry used by the periodic 3D kinetic field."""
from __future__ import annotations

import math
import torch


def cube_velocities(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return normalized D3Q8 corner velocities."""
    velocity = torch.tensor(
        [(x, y, z) for x in (-1.0, 1.0)
         for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        dtype=dtype,
    )
    return velocity / math.sqrt(3.0)
