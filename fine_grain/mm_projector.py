"""Vision→language connector (LLaVA-1.5 style).

Default is a 2-layer MLP with GELU — the de-facto connector in modern VLMs
(LLaVA-1.5+, most open multimodal stacks). Linear is kept for ablation.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MMProjector(nn.Module):
    """Map vision token features into the LLM embedding space.

    Args:
        d_in: vision-side channel dim (frontend mixer / stem dim).
        d_out: LLM hidden size (default = d_in).
        hidden: MLP intermediate width (default = d_out; LLaVA-1.5 style).
        kind: ``"mlp"`` (Linear→GELU→Linear) or ``"linear"`` (single Linear).
    """

    def __init__(
        self,
        d_in: int,
        d_out: Optional[int] = None,
        hidden: Optional[int] = None,
        kind: str = "mlp",
    ):
        super().__init__()
        d_out = int(d_out if d_out is not None else d_in)
        kind = (kind or "mlp").lower()
        self.kind = kind
        self.d_in = int(d_in)
        self.d_out = d_out
        if kind == "linear":
            self.net = nn.Linear(d_in, d_out)
            self.hidden = None
        elif kind in ("mlp", "mlp2", "gelu_mlp"):
            # LLaVA-1.5: two linear layers with GELU; intermediate often = d_out
            h = int(hidden if hidden is not None else d_out)
            self.hidden = h
            self.net = nn.Sequential(
                nn.Linear(d_in, h),
                nn.GELU(),
                nn.Linear(h, d_out),
            )
        else:
            raise ValueError(
                f"unknown projector kind={kind!r}; use 'mlp' or 'linear'"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d_in] → [..., d_out] (token sequence preserved)."""
        return self.net(x)

    def extra_repr(self) -> str:
        return f"kind={self.kind}, d_in={self.d_in}, d_out={self.d_out}, hidden={self.hidden}"
