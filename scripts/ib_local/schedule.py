"""Dependency DAG construction and layer scheduling for local collision events."""
import torch
from torch import Tensor


def schedule_dependencies(
    active_i: Tensor,
    active_j: Tensor,
    n_particles: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Construct dependency levels for active collision pairs.

    Each event e depends on the most recent event involving particle i[e] or j[e]:
        level[e] = 1 + max(last_level[i[e]], last_level[j[e]])
        last_level[i[e]] = level[e]
        last_level[j[e]] = level[e]

    Returns:
        levels: int64[K] level assigned to each active event.
        permutation: int64[K] stable ordering grouping events by layer.
        offsets: int64[num_levels + 1] boundary offsets for each layer slice:
                 layer l contains events permutation[offsets[l]:offsets[l+1]].
    """
    if active_i.ndim != 1 or active_j.ndim != 1:
        raise ValueError("active_i and active_j must be 1D tensors")
    if len(active_i) != len(active_j):
        raise ValueError(f"active_i and active_j length mismatch: {len(active_i)} vs {len(active_j)}")
    if active_i.dtype != torch.int64 or active_j.dtype != torch.int64:
        raise TypeError("active_i and active_j must be int64 tensors")

    k = len(active_i)
    device = active_i.device

    if k == 0:
        return (
            torch.zeros(0, dtype=torch.int64, device=device),
            torch.zeros(0, dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int64, device=device),
        )

    if (active_i < 0).any() or (active_j < 0).any():
        raise ValueError("Particle indices must be non-negative")
    if (active_i == active_j).any():
        raise ValueError("Self-collision detected in active pairs")

    if n_particles is None:
        n = int(max(active_i.max().item(), active_j.max().item())) + 1
    else:
        n = n_particles
        if (active_i >= n).any() or (active_j >= n).any():
            raise IndexError(f"Particle index exceeds n_particles={n}")

    # Build levels in original sequence order
    last_level = [-1] * n
    levels_list = []
    i_list = active_i.cpu().tolist()
    j_list = active_j.cpu().tolist()

    for e in range(k):
        pi = i_list[e]
        pj = j_list[e]
        lvl = 1 + max(last_level[pi], last_level[pj])
        levels_list.append(lvl)
        last_level[pi] = lvl
        last_level[pj] = lvl

    levels = torch.tensor(levels_list, dtype=torch.int64, device=device)
    num_levels = int(levels.max().item()) + 1

    # Stable sort by level: events in the same layer preserve their original relative order
    permutation = torch.argsort(levels, stable=True)
    counts = torch.bincount(levels, minlength=num_levels)
    offsets = torch.zeros(num_levels + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(counts, dim=0)

    return levels, permutation, offsets
