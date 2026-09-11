"""Batched and dependency-scheduled implementation of local collision operator."""
import math
import torch
from torch import Tensor, nn

from .reference import reflect, spatial_kernel
from .schedule import schedule_dependencies
from .types import CandidateTable, CollisionResult, FrozenContext


def collision_batched(
    x: Tensor,
    v: Tensor,
    context: FrozenContext,
    table: CandidateTable,
    kernel: nn.Module,
    mode: str = "strict_local_v1",
    width: float = 1.0,
    max_rate: float = 1.0,
) -> CollisionResult:
    """Dependency-scheduled layer-batched execution of candidate collisions.

    1. Partitions active candidate collisions into independent levels where no two pairs
       in the same level share particles (mutually disjoint pairs).
    2. Batches query orbit construction and executes independent collisions in parallel per layer.
    3. Guarantees exact topological order and unique functional scatter writeback per layer.
    """
    if mode not in ("legacy_eps", "strict_local_v1"):
        raise ValueError(f"Unknown mode '{mode}'; must be 'legacy_eps' or 'strict_local_v1'")
    if not isinstance(width, (int, float)) or not math.isfinite(width) or width <= 0.0:
        raise ValueError(f"width must be positive and finite, got {width}")
    if not isinstance(max_rate, (int, float)) or not math.isfinite(max_rate) or max_rate < 0.0:
        raise ValueError(f"max_rate must be non-negative and finite, got {max_rate}")

    # x and v shape, dtype, device checks
    if x.ndim != 2 or v.ndim != 2:
        raise ValueError(f"x and v must be 2D tensors [N, dim], got x.ndim={x.ndim}, v.ndim={v.ndim}")
    if x.shape != v.shape:
        raise ValueError(f"x and v must have identical shapes, got x.shape={x.shape}, v.shape={v.shape}")
    if x.device != v.device:
        raise ValueError(f"x and v must be on the same device, got {x.device} vs {v.device}")
    if x.dtype != v.dtype:
        raise TypeError(f"x and v must have identical dtype, got {x.dtype} vs {v.dtype}")
    if not torch.is_floating_point(x) or not torch.is_floating_point(v):
        raise TypeError(f"x and v must be floating point tensors, got x={x.dtype}, v={v.dtype}")

    n, dim = v.shape

    # context checks
    if context.x.ndim != 2 or context.v.ndim != 2:
        raise ValueError(f"context.x and context.v must be 2D tensors [N_ctx, dim], got {context.x.ndim}, {context.v.ndim}")
    if context.x.shape != context.v.shape:
        raise ValueError(f"context.x and context.v must have identical shapes, got {context.x.shape} vs {context.v.shape}")
    if context.x.shape[-1] != dim:
        raise ValueError(f"Context dimension {context.x.shape[-1]} does not match state dimension {dim}")
    if context.x.device != v.device or context.v.device != v.device:
        raise ValueError(f"Context must be on the same device as state ({v.device}), got x={context.x.device}, v={context.v.device}")
    if context.x.dtype != v.dtype or context.v.dtype != v.dtype:
        raise TypeError(f"Context dtype must match state dtype ({v.dtype}), got x={context.x.dtype}, v={context.v.dtype}")

    # table checks
    if table.normal.shape[-1] != dim:
        raise ValueError(f"Candidate normal dimension {table.normal.shape[-1]} does not match state dimension {dim}")
    if table.normal.device != v.device or table.uniform.device != v.device or table.i.device != v.device or table.j.device != v.device:
        raise ValueError("Candidate table tensors must be on the same device as state")
    if table.normal.dtype != v.dtype or table.uniform.dtype != v.dtype:
        raise TypeError(f"Candidate normal and uniform dtype must match state dtype ({v.dtype}), got {table.normal.dtype}")

    m_count = len(table)
    if m_count > 0:
        if (table.i >= n).any() or (table.j >= n).any():
            raise IndexError(
                f"Candidate table indices out of bounds for state with N={n} particles: "
                f"max i={table.i.max().item()}, max j={table.j.max().item()}"
            )

    curr_v = v.clone()
    zero = curr_v.sum() * 0.0

    if m_count == 0 or max_rate == 0.0:
        return CollisionResult(
            v=curr_v,
            log_prob=zero,
            accepted=torch.zeros(m_count, dtype=torch.bool, device=curr_v.device),
            stats={"candidates": 0, "accepted": 0, "cross_moment_change": 0.0, "pairs": [], "num_layers": 0},
        )

    # Filter active candidate events based on spatial kernel support
    deltas_all = x[table.i] - x[table.j]
    is_active = deltas_all.abs().lt(width).all(dim=-1)
    active_m = torch.nonzero(is_active, as_tuple=True)[0]
    num_active = len(active_m)

    if num_active == 0:
        return CollisionResult(
            v=curr_v,
            log_prob=zero,
            accepted=torch.zeros(m_count, dtype=torch.bool, device=curr_v.device),
            stats={"candidates": m_count, "accepted": 0, "cross_moment_change": 0.0, "pairs": [], "num_layers": 0},
        )

    # Schedule active events into non-interfering parallel layers
    levels, permutation, offsets = schedule_dependencies(table.i[active_m], table.j[active_m], n_particles=n)
    num_layers = len(offsets) - 1

    log_prob = zero
    accepted = torch.zeros(m_count, dtype=torch.bool, device=curr_v.device)
    pairs = []
    kernel_bound = width ** (-dim)
    log2 = math.log(2.0)

    for l in range(num_layers):
        layer_perm = permutation[offsets[l]:offsets[l + 1]]
        layer_m = active_m[layer_perm]
        b_size = len(layer_m)

        idx_i = table.i[layer_m]
        idx_j = table.j[layer_m]
        normals = table.normal[layer_m]
        uniforms = table.uniform[layer_m]

        # Enforce unique scatter writeback indices in parallel layer
        all_indices = torch.cat([idx_i, idx_j])
        assert torch.unique(all_indices).numel() == all_indices.numel(), (
            f"Layer {l} contains duplicate particle index: parallel scatter indices must be strictly unique"
        )

        vi = curr_v[idx_i]
        vj = curr_v[idx_j]
        xi = x[idx_i]
        xj = x[idx_j]
        centers = (xi + xj) / 2.0
        deltas = xi - xj

        # Reflected velocities for layer
        vp, wp = reflect(vi, vj, normals)

        # Construct 8-element orbit tensor for all B pairs in parallel: [B, 8, 3*dim]
        pairs_v = torch.stack([
            torch.cat([vi, vj], dim=-1),
            torch.cat([vj, vi], dim=-1),
            torch.cat([vp, wp], dim=-1),
            torch.cat([wp, vp], dim=-1),
        ], dim=1)  # [B, 4, 2*dim]
        pairs_v_exp = pairs_v.unsqueeze(2).expand(b_size, 4, 2, 2 * dim)
        n_stack = torch.stack([normals, -normals], dim=1)  # [B, 2, dim]
        n_exp = n_stack.unsqueeze(1).expand(b_size, 4, 2, dim)
        orbit = torch.cat([pairs_v_exp, n_exp], dim=-1).reshape(b_size, 8, 3 * dim)

        q = kernel.query(orbit)  # [B, 8, Hc]

        choose_list = []
        for b in range(b_size):
            q_b = q[b]
            delta_x_b = context.x - centers[b]
            weights_b = spatial_kernel(delta_x_b, width)

            if mode == "strict_local_v1":
                mask_b = weights_b > 0.0
                if not mask_b.any():
                    raw_score_b = kernel.output(torch.tanh(q_b)).mean()
                else:
                    act_idx = torch.nonzero(mask_b, as_tuple=True)[0]
                    act_dx = delta_x_b[act_idx]
                    act_cv = context.v[act_idx]
                    act_feat = torch.cat((act_dx, act_cv), dim=-1)
                    k_b = kernel.key(act_feat)
                    val_b = kernel.value(act_feat)
                    scores_b = q_b @ k_b.T / math.sqrt(q.shape[-1]) + weights_b[act_idx].log()[None, :]
                    att_b = scores_b.softmax(-1) @ val_b
                    raw_score_b = kernel.output(torch.tanh(att_b + q_b)).mean()

                # Exact log-space probability (SPEC §7)
                log_geom_b = torch.log1p(-deltas[b].abs() / width).sum(-1)
                log_score_b = torch.nn.functional.logsigmoid(raw_score_b.clamp(-12.0, 12.0))
                log_p_b = log_geom_b + log_score_b

                u_b = uniforms[b]
                log_u_b = u_b.log() if u_b > 0.0 else u_b.new_tensor(float("-inf"))
                choose_b = bool((log_u_b < log_p_b).item())

                if log_p_b < -log2:
                    log_reject_b = torch.log1p(-torch.exp(log_p_b))
                else:
                    log_reject_b = torch.log(-torch.expm1(log_p_b))

                event_lp_b = log_p_b if choose_b else log_reject_b
                log_prob = log_prob + event_lp_b
            else:
                # legacy_eps mode
                features_b = torch.cat((delta_x_b, context.v), dim=-1)
                k_b = kernel.key(features_b)
                val_b = kernel.value(features_b)
                scores_b = q_b @ k_b.T / math.sqrt(q.shape[-1]) + weights_b.clamp_min(1e-12).log()[None, :]
                att_b = scores_b.softmax(-1) @ val_b
                raw_score_b = kernel.output(torch.tanh(att_b + q_b)).mean()
                rate_b = max_rate * torch.sigmoid(raw_score_b.clamp(-12.0, 12.0))

                local_b = spatial_kernel(deltas[b], width) / kernel_bound
                prob_b = local_b * rate_b / max_rate
                choose_b = bool((uniforms[b] < prob_b).item())
                event_lp_b = prob_b.clamp_min(1e-30).log() if choose_b else torch.log1p(-prob_b)
                log_prob = log_prob + event_lp_b

            accepted[layer_m[b]] = choose_b
            choose_list.append(choose_b)
            if choose_b:
                pairs.append([int(idx_i[b].item()), int(idx_j[b].item())])

        # Functional non-interfering parallel scatter writeback for the entire layer
        acc_indices = [b for b, ch in enumerate(choose_list) if ch]
        if acc_indices:
            acc_t = torch.tensor(acc_indices, device=v.device, dtype=torch.int64)
            scatter_indices = torch.cat([idx_i[acc_t], idx_j[acc_t]])
            scatter_values = torch.cat([vp[acc_t], wp[acc_t]], dim=0)
            curr_v = curr_v.index_copy(0, scatter_indices, scatter_values)

    cross = ((curr_v - v) * x).sum(-1).mean()

    return CollisionResult(
        v=curr_v,
        log_prob=log_prob,
        accepted=accepted,
        stats={
            "candidates": m_count,
            "accepted": int(accepted.sum().item()),
            "cross_moment_change": float(cross.detach().item()),
            "pairs": pairs,
            "num_layers": num_layers,
        },
    )
