"""Serial reference implementation of local learning collision operator."""
import math
import torch
from torch import Tensor, nn

from .types import CandidateTable, CollisionResult, FrozenContext


def reflect(v: Tensor, w: Tensor, normal: Tensor) -> tuple[Tensor, Tensor]:
    """Microreversible elastic reflection between two particles along unit normal n."""
    change = ((v - w) * normal).sum(-1, keepdim=True) * normal
    return v - change, w + change


def spatial_kernel(delta: Tensor, width: float) -> Tensor:
    """Normalized product triangular kernel on R^d with compact L_inf support [-width, width]^d."""
    return (1.0 - (delta / width).abs()).clamp_min(0.0).prod(-1) / (width ** delta.shape[-1])


def sample_candidates(
    n: int,
    dim: int,
    duration: float,
    width: float,
    max_rate: float,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> CandidateTable:
    """Sample candidate collision events from dominating Poisson process. No capping."""
    if not isinstance(n, int) or n < 2:
        raise ValueError(f"n must be an integer >= 2, got {n}")
    if not isinstance(dim, int) or dim < 1:
        raise ValueError(f"dim must be an integer >= 1, got {dim}")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0.0:
        raise ValueError(f"duration must be non-negative and finite, got {duration}")
    if not isinstance(width, (int, float)) or not math.isfinite(width) or width <= 0.0:
        raise ValueError(f"width must be positive and finite, got {width}")
    if not isinstance(max_rate, (int, float)) or not math.isfinite(max_rate) or max_rate < 0.0:
        raise ValueError(f"max_rate must be non-negative and finite, got {max_rate}")

    kernel_bound = width ** (-dim)
    total_rate = (n - 1) * max_rate * kernel_bound / 2.0
    rate_tensor = torch.tensor(total_rate * duration, device=device, dtype=dtype)
    count = int(torch.poisson(rate_tensor, generator=generator).item())

    if count == 0:
        return CandidateTable(
            i=torch.zeros(0, dtype=torch.int64, device=device),
            j=torch.zeros(0, dtype=torch.int64, device=device),
            normal=torch.zeros((0, dim), dtype=dtype, device=device),
            uniform=torch.zeros(0, dtype=dtype, device=device),
        )

    indices_i = torch.randint(n, (count,), device=device, generator=generator, dtype=torch.int64)
    indices_j = torch.randint(n - 1, (count,), device=device, generator=generator, dtype=torch.int64)
    indices_j = indices_j + (indices_j >= indices_i).long()

    normals = torch.randn(count, dim, device=device, dtype=dtype, generator=generator)
    normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(normals.dtype).tiny)
    uniforms = torch.rand(count, device=device, dtype=dtype, generator=generator)

    return CandidateTable(indices_i, indices_j, normals, uniforms)


def compute_rate(
    center: Tensor,
    v: Tensor,
    w: Tensor,
    normal: Tensor,
    context: FrozenContext,
    kernel: nn.Module,
    mode: str = "strict_local_v1",
    width: float = 1.0,
    max_rate: float = 1.0,
    return_raw: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Compute microreversible pair collision rate B_phi from attention over frozen background context."""
    if mode not in ("legacy_eps", "strict_local_v1"):
        raise ValueError(f"Unknown mode '{mode}'; must be 'legacy_eps' or 'strict_local_v1'")
    if not isinstance(width, (int, float)) or not math.isfinite(width) or width <= 0.0:
        raise ValueError(f"width must be positive and finite, got {width}")
    if not isinstance(max_rate, (int, float)) or not math.isfinite(max_rate) or max_rate < 0.0:
        raise ValueError(f"max_rate must be non-negative and finite, got {max_rate}")

    dim = normal.shape[-1]
    if center.shape[-1] != dim or v.shape[-1] != dim or w.shape[-1] != dim or context.x.shape[-1] != dim:
        raise ValueError(
            f"Dimension mismatch in compute_rate: normal={dim}, center={center.shape[-1]}, "
            f"v={v.shape[-1]}, w={w.shape[-1]}, context={context.x.shape[-1]}"
        )
    if not (center.device == v.device == w.device == normal.device == context.x.device == context.v.device):
        raise ValueError("All compute_rate tensor inputs must be on the same device")
    if not (center.dtype == v.dtype == w.dtype == normal.dtype == context.x.dtype == context.v.dtype):
        raise TypeError("All compute_rate tensor inputs must have the same dtype")

    vp, wp = reflect(v, w, normal)
    # 8-element orbit: exchange pairs, reverse collision, normal sign
    orbit = torch.stack([
        torch.cat((a, b, sign * normal))
        for a, b in ((v, w), (w, v), (vp, wp), (wp, vp))
        for sign in (1, -1)
    ])  # [8, 3*dim]

    delta_x = context.x - center  # [N, dim]
    q = kernel.query(orbit)       # [8, Hc]

    if mode == "legacy_eps":
        features = torch.cat((delta_x, context.v), dim=-1)  # [N, 2*dim]
        k = kernel.key(features)     # [N, Hc]
        val = kernel.value(features) # [N, Hc]
        weights = spatial_kernel(delta_x, width)  # [N]
        scores = q @ k.T / math.sqrt(q.shape[-1]) + weights.clamp_min(1e-12).log()
        attended = scores.softmax(-1) @ val
        score = kernel.output(torch.tanh(attended + q)).mean()
        rate = max_rate * torch.sigmoid(score.clamp(-12.0, 12.0))
        if return_raw:
            return rate, score
        return rate

    # strict_local_v1: filter active context within compact L_inf support [-width, width]^d
    weights = spatial_kernel(delta_x, width)  # [N]
    mask = weights > 0.0  # [N]

    if not mask.any():
        # Query-only branch when background support is empty (SPEC §3)
        raw_score = kernel.output(torch.tanh(q)).mean()
    else:
        # Active context slicing: completely isolates distant particles from the autograd graph,
        # preventing log(0) and avoiding NaN gradients for single-axis out-of-bound particles.
        active_indices = torch.nonzero(mask, as_tuple=True)[0]
        active_delta_x = delta_x[active_indices]
        active_context_v = context.v[active_indices]
        active_features = torch.cat((active_delta_x, active_context_v), dim=-1)
        active_weights = weights[active_indices]

        k = kernel.key(active_features)
        val = kernel.value(active_features)
        log_weights = active_weights.log()  # active_weights > 0 strictly, never log(0)

        scores = q @ k.T / math.sqrt(q.shape[-1]) + log_weights[None, :]
        attended = scores.softmax(-1) @ val
        raw_score = kernel.output(torch.tanh(attended + q)).mean()

    rate = max_rate * torch.sigmoid(raw_score.clamp(-12.0, 12.0))
    if return_raw:
        return rate, raw_score
    return rate


def collision_serial(
    x: Tensor,
    v: Tensor,
    context: FrozenContext,
    table: CandidateTable,
    kernel: nn.Module,
    mode: str = "strict_local_v1",
    width: float = 1.0,
    max_rate: float = 1.0,
) -> CollisionResult:
    """Sequential reference execution of candidate collisions."""
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
            stats={"candidates": 0, "accepted": 0, "cross_moment_change": 0.0, "pairs": []},
        )

    log_prob = zero
    accepted = torch.zeros(m_count, dtype=torch.bool, device=curr_v.device)
    pairs = []
    kernel_bound = width ** (-dim)
    log2 = math.log(2.0)

    for m in range(m_count):
        i = int(table.i[m].item())
        j = int(table.j[m].item())
        normal = table.normal[m]

        delta = x[i] - x[j]
        # Active in L_inf support if every coordinate satisfies |r_a| < width
        is_active = bool((delta.abs() < width).all().item())

        if mode == "strict_local_v1":
            if is_active:
                center = (x[i] + x[j]) / 2.0
                rate, raw_score = compute_rate(
                    center, curr_v[i], curr_v[j], normal, context, kernel,
                    mode=mode, width=width, max_rate=max_rate, return_raw=True,
                )
                # Exact log-space probability from SPEC §7:
                # logp = sum_a log1p(-|r_a|/h) + logsigmoid(clamp(raw, -12, 12))
                log_geom = torch.log1p(-delta.abs() / width).sum(-1)
                log_score = torch.nn.functional.logsigmoid(raw_score.clamp(-12.0, 12.0))
                log_p = log_geom + log_score

                # Acceptance decision in log space: log(uniform) < logp (uniform=0 -> -inf)
                u = table.uniform[m]
                log_u = u.log() if u > 0.0 else u.new_tensor(float("-inf"))
                choose = bool((log_u < log_p).item())

                # Stable rejection log probability: log1p(-exp(logp)) for logp < -log2, else log(-expm1(logp))
                if log_p < -log2:
                    log_reject = torch.log1p(-torch.exp(log_p))
                else:
                    log_reject = torch.log(-torch.expm1(log_p))

                # Select log_p or log_reject by boolean branch (SPEC §7: 不做 A 乘 logp)
                event_lp = log_p if choose else log_reject
                log_prob = log_prob + event_lp
            else:
                # p=0 几何外事件贡献 0 (SPEC §7)
                choose = False
        else:
            # legacy_eps mode
            local = spatial_kernel(delta, width) / kernel_bound
            if local > 0.0:
                center = (x[i] + x[j]) / 2.0
                rate = compute_rate(
                    center, curr_v[i], curr_v[j], normal, context, kernel,
                    mode=mode, width=width, max_rate=max_rate, return_raw=False,
                )
                prob = local * rate / max_rate
            else:
                prob = zero

            choose = bool((table.uniform[m] < prob).item())
            log_prob = log_prob + (prob.clamp_min(1e-30).log() if choose else torch.log1p(-prob))

        if choose:
            vp, wp = reflect(curr_v[i], curr_v[j], normal)
            indices = torch.tensor([i, j], device=curr_v.device)
            curr_v = curr_v.index_copy(0, indices, torch.stack((vp, wp)))
            accepted[m] = True
            pairs.append([i, j])

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
        },
    )
