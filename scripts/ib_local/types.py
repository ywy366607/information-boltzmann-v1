"""Data structures for local learning collision operator."""
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class CandidateTable:
    i: Tensor  # int64[M]
    j: Tensor  # int64[M]
    normal: Tensor  # float[M, d], unit vectors
    uniform: Tensor  # float[M] in [0, 1)

    def __post_init__(self):
        if not (self.i.ndim == 1 and self.j.ndim == 1 and self.normal.ndim == 2 and self.uniform.ndim == 1):
            raise ValueError("Invalid candidate table tensor dimensions")
        m = len(self.i)
        if not (len(self.j) == m and len(self.normal) == m and len(self.uniform) == m):
            raise ValueError(
                f"Candidate table length mismatch: i={m}, j={len(self.j)}, "
                f"normal={len(self.normal)}, uniform={len(self.uniform)}"
            )
        if self.i.dtype != torch.int64 or self.j.dtype != torch.int64:
            raise TypeError("Candidate indices i and j must be torch.int64")
        if self.normal.shape[-1] < 1:
            raise ValueError("Candidate normal must have dimension >= 1")
        if m > 0:
            if not (self.normal.device == self.uniform.device == self.i.device == self.j.device):
                raise ValueError("All candidate table tensors must be on the same device")
            if self.normal.dtype != self.uniform.dtype or not torch.is_floating_point(self.normal):
                raise TypeError("Candidate normal and uniform must have the same floating point dtype")
            if not (torch.isfinite(self.normal).all() and torch.isfinite(self.uniform).all()):
                raise ValueError("Candidate normal and uniform must be finite")
            if (self.i < 0).any() or (self.j < 0).any():
                raise ValueError("Candidate indices i and j must be non-negative")
            if (self.i == self.j).any():
                raise ValueError("Candidate table contains self-collisions (i == j)")
            if (self.uniform < 0.0).any() or (self.uniform >= 1.0).any():
                raise ValueError("Candidate uniforms must be in half-open interval [0, 1)")
            # Strict unit normal validation (reject non-unit normals, do NOT silently renormalize)
            norms = self.normal.norm(dim=-1)
            tol = 1e-5 if self.normal.dtype in (torch.float32, torch.bfloat16) else 1e-7
            if (norms - 1.0).abs().max() > tol:
                raise ValueError("Candidate normals must be strictly unit vectors with norm 1")

    def to(self, device=None, dtype=None) -> "CandidateTable":
        return CandidateTable(
            i=self.i.to(device=device),
            j=self.j.to(device=device),
            normal=self.normal.to(device=device, dtype=dtype),
            uniform=self.uniform.to(device=device, dtype=dtype),
        )

    def __len__(self) -> int:
        return len(self.i)


@dataclass
class FrozenContext:
    x: Tensor  # float[N, d]
    v: Tensor  # float[N, d]

    def __post_init__(self):
        if not (self.x.ndim == 2 and self.v.ndim == 2):
            raise ValueError("FrozenContext x and v must be 2D tensors [N, d]")
        if self.x.shape != self.v.shape:
            raise ValueError(f"FrozenContext shape mismatch: x={self.x.shape}, v={self.v.shape}")
        if self.x.device != self.v.device:
            raise ValueError(f"FrozenContext x and v must be on the same device, got {self.x.device} vs {self.v.device}")
        if self.x.dtype != self.v.dtype or not torch.is_floating_point(self.x):
            raise TypeError(f"FrozenContext x and v must have the same floating point dtype, got {self.x.dtype} vs {self.v.dtype}")

    def to(self, device=None, dtype=None) -> "FrozenContext":
        return FrozenContext(
            x=self.x.to(device=device, dtype=dtype),
            v=self.v.to(device=device, dtype=dtype),
        )


@dataclass
class CollisionResult:
    v: Tensor  # float[N, d]
    log_prob: Tensor  # scalar float
    accepted: Tensor  # bool[M]
    stats: dict

    def __post_init__(self):
        if self.v.ndim != 2:
            raise ValueError("CollisionResult v must be a 2D tensor [N, d]")
        if self.log_prob.numel() != 1:
            raise ValueError("CollisionResult log_prob must be a scalar")
        if self.accepted.ndim != 1 or self.accepted.dtype != torch.bool:
            raise ValueError("CollisionResult accepted must be a 1D boolean tensor [M]")
