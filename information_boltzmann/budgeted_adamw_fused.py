"""One Triton program per real parameter tile; same arithmetic as CPU reference."""
from torch.autograd.graph import increment_version
import triton
import triton.language as tl


@triton.jit
def _step(P, G, M, V, E, H, K, PRIOR, C1, C2, OBS_POWER, NOISE, LR, WD, EPS,
          B1: tl.constexpr, B2: tl.constexpr, ROWS: tl.constexpr,
          COLS: tl.constexpr, N: tl.constexpr, TILE: tl.constexpr,
          MATRIX: tl.constexpr, ALLOCATE: tl.constexpr,
          COOL: tl.constexpr, WIENER: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    x = tl.arange(0, BLOCK)
    if MATRIX:
        cols = COLS // TILE
        rr = (tile // cols) * TILE + x // TILE
        cc = (tile % cols) * TILE + x % TILE
        index = rr * COLS + cc
        valid = x < TILE * TILE
        n = TILE * TILE
    else:
        index = x
        valid = x < N
        n = N
    g = tl.load(G + index, valid, 0)
    m = B1 * tl.load(M + index, valid, 0) + (1 - B1) * g
    v = B2 * tl.load(V + index, valid, 0) + (1 - B2) * g * g
    d = (m / C1) / (tl.sqrt(v / C2) + EPS)
    q = B1 * tl.load(E + tile) + (1 - B1) * tl.sum(tl.where(valid, g * g, 0), 0) / n
    energy = q / (1 - OBS_POWER)
    moment_rms = tl.sqrt(tl.sum(tl.where(valid, m * m, 0), 0) / n)
    recent_rms = (moment_rms + OBS_POWER * tl.load(PRIOR + tile)) / (1 - OBS_POWER)
    signal = recent_rms * recent_rms
    gate = 1.0
    if COOL:
        if WIENER:
            gate = tl.where(NOISE >= 0.999999,
                            1.0, tl.where(signal > 0,
                            tl.minimum(1.0, tl.maximum(0.0, (signal - NOISE * energy) /
                            tl.maximum((1 - NOISE) * signal, 1.1754943508222875e-38))), 0.0))
        else:
            gate = tl.where(NOISE >= 0.999999,
                            1.0, tl.where(energy > 0,
                            tl.minimum(1.0, tl.maximum(0.0, (signal - NOISE * energy) /
                            tl.maximum((1 - NOISE) * energy, 1.1754943508222875e-38))), 0.0))
    phase = tl.load(H + tile)
    factor = 1.0
    if MATRIX:
        diagonal = ((x // TILE) < (TILE // 2)) == ((x % TILE) < (TILE // 2))
        power = tl.where(valid, d * d, 0)
        contrast = tl.sum(tl.where(diagonal, power, -power), 0)
        cue = tl.minimum(1.0, tl.maximum(-1.0, contrast / tl.maximum(tl.sum(power, 0), 1.1754943508222875e-38)))
        phase = tl.minimum(1.0, tl.maximum(-1.0, B1 * phase + (1 - B1) * cue))
        if ALLOCATE:
            factor = tl.where(diagonal, 1 + phase, 1 - phase)
    p = tl.load(P + index, valid, 0)
    tl.store(P + index, p - LR * gate * (factor * d + WD * p), valid)
    tl.store(M + index, m, valid)
    tl.store(V + index, v, valid)
    tl.store(E + tile, q)
    tl.store(H + tile, phase)
    tl.store(K + tile, gate)


def _launch(p, g, state, *, count, beta1, beta2, lr, decay, eps,
               tile_size, matrix, allocation, cooling, observer_count, noise,
               wiener=False):
    n = p.numel()
    count_tiles = (p.shape[0] // tile_size * p.shape[1] // tile_size) if matrix else 1
    _step[(count_tiles,)](
        p, g, state['exp_avg'], state['exp_avg_sq'], state['budget_energy'],
        state['budget_phase'], state['budget_gate'], state['budget_prior_rms'],
        1 - beta1 ** count, 1 - beta2 ** count, beta1 ** observer_count, noise, lr, decay, eps,
        beta1, beta2, p.shape[0] if matrix else 1, p.shape[1] if matrix else n,
        n, tile_size, matrix, allocation, cooling, bool(wiener),
        triton.next_power_of_2(tile_size * tile_size) if matrix else triton.next_power_of_2(n),
        num_warps=4, enable_fp_fusion=False)


def fused_step(p, g, state, **kwargs):
    _launch(p, g, state, **kwargs)
    # Triton stores bypass PyTorch's in-place dispatcher (and its version bumps).
    increment_version((p, state['exp_avg'], state['exp_avg_sq'], state['budget_energy'],
                       state['budget_phase'], state['budget_gate']))
