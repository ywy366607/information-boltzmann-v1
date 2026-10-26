"""Fused Triton leap for the nested world (pellet = 1): the same event classes and Lean rates as
`cgb_nested_world.World.rates`, sampled and applied in two kernels per leap.

Phase 1 (per cell): rates of the 12 classes (supply in/out of food at sources and of waste at
sinks; one quantum of food or waste to each neighbour), Poisson counts by inversion (rate * dt is
small; the tail beyond K = 12 is below 1e-9 for rate * dt <= 1.2), clipped so that no pool leaves
0..V: supply in <= (V - n) // 2, sends to a neighbour <= (V - n_neighbour) // 8, total sends <= the
pool after supply. Writes the pool after its own events and the four send counts.
Phase 2 (per cell): adds what the four neighbours sent.
Optionally (`rates_out`) phase 1 writes the 12 rates, so the gate can compare them with Lean.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _up(n, V):
    nf = n.to(tl.float32)
    return tl.where(n < V, tl.sqrt((V - nf) / (nf + 1.0)), 0.0)


@triton.jit
def _dn(n, V):
    nf = n.to(tl.float32)
    return tl.where(n > 0, tl.sqrt(nf / (V - nf + 1.0)), 0.0)


@triton.jit
def _pois(lam, u):
    p = tl.exp(-lam)
    c = p
    k = tl.zeros_like(lam)
    for i in tl.static_range(1, 13):
        k += tl.where(u > c, 1.0, 0.0)
        p = p * lam / i
        c = c + p
    return k.to(tl.int32)


@triton.jit
def _phase1(f_ptr, w_ptr, mu_ptr, kind_ptr, seed_ptr, fmid_ptr, wmid_ptr, send_ptr, rates_ptr,
            N, G, V, kin, kdiff, dt, WRITE_RATES: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    GG = G * G
    b = offs // GG
    c = offs % GG
    x = c // G
    y = c % G
    seed = tl.load(seed_ptr)
    f = tl.load(f_ptr + offs, mask=m, other=0)
    w = tl.load(w_ptr + offs, mask=m, other=0)
    mu = tl.load(mu_ptr + offs, mask=m, other=0.0)
    kd = tl.load(kind_ptr + c, mask=m, other=0)
    em = tl.exp(mu / 2.0)
    ep = tl.exp(-mu / 2.0)
    src = kd == 1
    snk = kd == 2
    r0 = tl.where(src, kin * _up(f, V) * em, 0.0)
    r1 = tl.where(src, kin * _dn(f, V) * ep, 0.0)
    r2 = tl.where(snk, kin * _up(w, V) * em, 0.0)
    r3 = tl.where(snk, kin * _dn(w, V) * ep, 0.0)
    base = offs * 12
    k0 = _pois(r0 * dt, tl.rand(seed, base + 0))
    k1 = _pois(r1 * dt, tl.rand(seed, base + 1))
    k2 = _pois(r2 * dt, tl.rand(seed, base + 2))
    k3 = _pois(r3 * dt, tl.rand(seed, base + 3))
    fa = f + tl.minimum(k0, (V - f) // 2) - tl.minimum(k1, f)
    wa = w + tl.minimum(k2, (V - w) // 2) - tl.minimum(k3, w)
    if WRITE_RATES:
        tl.store(rates_ptr + base + 0, r0, mask=m)
        tl.store(rates_ptr + base + 1, r1, mask=m)
        tl.store(rates_ptr + base + 2, r2, mask=m)
        tl.store(rates_ptr + base + 3, r3, mask=m)
    dnf = _dn(f, V)
    dnw = _dn(w, V)
    for d in tl.static_range(4):
        if d == 0:
            nc = ((x + 1) % G) * G + y
        elif d == 1:
            nc = x * G + (y + 1) % G
        elif d == 2:
            nc = ((x + G - 1) % G) * G + y
        else:
            nc = x * G + (y + G - 1) % G
        no = b * GG + nc
        fb = tl.load(f_ptr + no, mask=m, other=0)
        wb = tl.load(w_ptr + no, mask=m, other=0)
        rf = 2.0 * kdiff * dnf * _up(fb, V)
        rw = 2.0 * kdiff * dnw * _up(wb, V)
        if WRITE_RATES:
            tl.store(rates_ptr + base + 4 + d, rf, mask=m)
            tl.store(rates_ptr + base + 8 + d, rw, mask=m)
        sf = tl.minimum(tl.minimum(_pois(rf * dt, tl.rand(seed, base + 4 + d)), (V - fb) // 8), fa)
        sw = tl.minimum(tl.minimum(_pois(rw * dt, tl.rand(seed, base + 8 + d)), (V - wb) // 8), wa)
        fa -= sf
        wa -= sw
        tl.store(send_ptr + offs * 8 + d, sf, mask=m)
        tl.store(send_ptr + offs * 8 + 4 + d, sw, mask=m)
    tl.store(fmid_ptr + offs, fa, mask=m)
    tl.store(wmid_ptr + offs, wa, mask=m)


@triton.jit
def _phase2(f_ptr, w_ptr, fmid_ptr, wmid_ptr, send_ptr, seed_ptr, N, G, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    GG = G * G
    b = offs // GG
    c = offs % GG
    x = c // G
    y = c % G
    fa = tl.load(fmid_ptr + offs, mask=m, other=0)
    wa = tl.load(wmid_ptr + offs, mask=m, other=0)
    # the neighbour at -x sent in direction 0 (+x), the one at -y in direction 1, ...
    s0 = b * GG + ((x + G - 1) % G) * G + y
    s1 = b * GG + x * G + (y + G - 1) % G
    s2 = b * GG + ((x + 1) % G) * G + y
    s3 = b * GG + x * G + (y + 1) % G
    fa += tl.load(send_ptr + s0 * 8 + 0, mask=m, other=0) + tl.load(send_ptr + s1 * 8 + 1, mask=m, other=0) + \
        tl.load(send_ptr + s2 * 8 + 2, mask=m, other=0) + tl.load(send_ptr + s3 * 8 + 3, mask=m, other=0)
    wa += tl.load(send_ptr + s0 * 8 + 4, mask=m, other=0) + tl.load(send_ptr + s1 * 8 + 5, mask=m, other=0) + \
        tl.load(send_ptr + s2 * 8 + 6, mask=m, other=0) + tl.load(send_ptr + s3 * 8 + 7, mask=m, other=0)
    tl.store(f_ptr + offs, fa, mask=m)
    tl.store(w_ptr + offs, wa, mask=m)
    if pid == 0:
        tl.store(seed_ptr, tl.load(seed_ptr) + 1)


class FusedWorld:
    """B worlds as int32 pools on CUDA; `step(mu, T)` advances T by leaps of dt. Same table and
    stimulus conventions as `cgb_nested_world.World` (pellet must be 1)."""

    BLOCK = 256

    def __init__(self, tbl, B=1, device="cuda", seed=0, dt=0.05):
        import cgb_nested_world as NW
        t = tbl["table"]
        assert t["pellet"] == 1
        self.V, self.kin, self.kdiff, self.dt = t["V"], t["kappa_in"], t["kappa_diff"], dt
        self.G = G = tbl["geometry"]["G"]
        self.B = B
        self.N = B * G * G
        self.kind = torch.tensor(NW.kinds(G).reshape(-1), dtype=torch.int32, device=device)
        self.f = torch.zeros(B, G, G, dtype=torch.int32, device=device)
        self.w = torch.zeros_like(self.f)
        self.fmid = torch.empty_like(self.f)
        self.wmid = torch.empty_like(self.f)
        self.send = torch.empty(self.N * 8, dtype=torch.int32, device=device)
        self.seed = torch.tensor([seed * 1000003 + 17], dtype=torch.int32, device=device)
        self.mu32 = torch.zeros(B, G, G, dtype=torch.float32, device=device)

    def leap(self, rates_out=None):
        grid = (triton.cdiv(self.N, self.BLOCK),)
        _phase1[grid](self.f, self.w, self.mu32, self.kind, self.seed, self.fmid, self.wmid, self.send,
                      rates_out if rates_out is not None else self.send, self.N, self.G, self.V,
                      self.kin, self.kdiff, self.dt, WRITE_RATES=rates_out is not None, BLOCK=self.BLOCK)
        _phase2[grid](self.f, self.w, self.fmid, self.wmid, self.send, self.seed, self.N, self.G, BLOCK=self.BLOCK)

    def rates(self, mu):
        """(B, 12, G, G) rates of the current state (runs one leap on a copy of the state)."""
        f0, w0, s0 = self.f.clone(), self.w.clone(), self.seed.clone()
        self.mu32.copy_(mu)
        out = torch.empty(self.N * 12, dtype=torch.float32, device=self.f.device)
        self.leap(out)
        self.f.copy_(f0)
        self.w.copy_(w0)
        self.seed.copy_(s0)
        return out.view(self.B, self.G, self.G, 12).permute(0, 3, 1, 2)

    def step(self, mu, T=1.0):
        self.mu32.copy_(mu)
        for _ in range(max(1, int(round(T / self.dt)))):
            self.leap()

    def captured(self, T=1.0):
        """CUDA Graph of `step` on the static stimulus buffer `mu32`: returns (mu32, replay)."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        f0, w0 = self.f.clone(), self.w.clone()
        with torch.cuda.stream(s):
            for _ in range(max(1, int(round(T / self.dt)))):
                self.leap()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(max(1, int(round(T / self.dt)))):
                self.leap()
        self.f.copy_(f0)
        self.w.copy_(w0)
        return self.mu32, g.replay
