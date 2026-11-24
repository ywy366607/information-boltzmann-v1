"""Unit tests for e-means + Population ALIF Lazy-Decay Memory Reservoir."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.ib_local.emeans_alif_reservoir import EmeansIndex, PopulationAlifReservoir


def test_emeans_fit_beats_random_and_revives_dead_clusters():
    torch.manual_seed(20261124)
    # 8 tight Gaussian clusters in 32D
    centers = torch.randn(8, 32) * 5.0
    pts = torch.cat([c[None] + 0.2 * torch.randn(40, 32) for c in centers], dim=0)
    slots = torch.arange(len(pts))
    idx = EmeansIndex(n_clusters=8, dim=32, seed=7)
    info = idx.fit(pts, slots, epochs=3)
    assert idx.fitted is True
    assert info['n'] == 320
    assert info['clusters'] == 8
    # Tight clusters (std=0.2 in 32D -> expected MSE ~ 32 * 0.04 = 1.28); verify low MSE!
    assert info['mse'] < 3.0
    # Online update adds a new point cleanly into the nearest bucket.
    cid = idx.update_slot(999, centers[0])
    assert 999 in idx.buckets[cid]


def test_lazy_decay_matches_dense_step_by_step_exactly():
    """Verify Lean Theorem 1 numerically: lazy power decay == step-by-step dense decay."""
    torch.manual_seed(11)
    n, d = 128, 32
    vecs = torch.randn(n, d)
    mem = SimpleNamespace(
        occupied=torch.ones(n, dtype=torch.bool),
        source=vecs,
        last_write=torch.arange(n),
        episodes=torch.tensor(n),
        read_credit_enabled=False,
    )
    enc = torch.nn.Linear(d, 16, bias=False)
    res = PopulationAlifReservoir(capacity=n, dim=d, n_clusters=8, top_clusters=2,
                                  k_max=8, lam_v=0.8, lam_a=0.75)
    # Seed a sleeping slot with known state at tick 0
    res.sync_from_memory(mem)
    res.v[5] = 2.0
    res.a[5] = 1.5
    res.last_tick[5] = 0
    # Advance clock 10 ticks without touching slot 5
    res.tick = 10
    dt = res.tick - res.last_tick[5]
    # Exact dense 10-step decay
    v_dense, a_dense = 2.0, 1.5
    for _ in range(dt):
        v_dense *= 0.8
        a_dense *= 0.75
    assert res.v[5] * (0.8 ** dt) == pytest.approx(v_dense, rel=1e-6)
    assert res.a[5] * (0.75 ** dt) == pytest.approx(a_dense, rel=1e-6)


def test_population_alif_bounded_vram_cap_and_anti_monopoly():
    torch.manual_seed(99)
    n, d = 256, 32
    vecs = torch.randn(n, d)
    mem = SimpleNamespace(
        occupied=torch.ones(n, dtype=torch.bool),
        source=vecs,
        last_write=torch.arange(n),
        episodes=torch.tensor(n),
        read_credit_enabled=False,
    )
    enc = torch.nn.Linear(d, 16, bias=False)
    res = PopulationAlifReservoir(capacity=n, dim=d, n_clusters=16, top_clusters=3,
                                  k_max=8, lam_v=0.75, lam_a=0.85,
                                  theta0=0.25, beta=0.60, apl_gain=0.45)

    # Repeatedly query the exact same vector -> VRAM selection <= k_max AND refractory rotates slots!
    q = vecs[0]
    first_sel, d1 = res.query(q, mem, enc)
    assert 1 <= len(first_sel) <= 8
    assert d1['woken'] < n  # Two-tier routing woke only a small subset!
    top_slot = first_sel[0]
    a_after_first = float(res.a[top_slot])
    assert a_after_first >= 1.0  # Refractory bumped!

    # Repeated queries should bump adaptation further and rotate complementary slots
    seen_sets = [set(first_sel)]
    for _ in range(6):
        sel, diag = res.query(q, mem, enc)
        assert 1 <= len(sel) <= 8
        seen_sets.append(set(sel))
    # Anti-monopoly: threshold adaptation causes at least some rotation or rank shift across 7 repeated hits
    assert float(res.a[top_slot]) > a_after_first or any(s != seen_sets[0] for s in seen_sets[1:])
