"""Spike semantics of the validated whole-brain LIF (Shiu et al. 2024): a spike consumes the cell's synaptic drive
(g = 0) and the cell is refractory; defaults keep the old engine (v reset only)."""
import numpy as np
import torch

from information_boltzmann.core.fly_reservoir import FlyReservoirLM


def tiny_coba(tmp_path, n=30, seed=42):
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n, 100).astype(np.int32)
    post = rng.integers(0, n, 100).astype(np.int32)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    path = tmp_path / "tiny_coba.npz"
    np.savez_compressed(path, edge_pre=pre, edge_post=post, edge_weight=(rng.standard_normal(pre.size) * 0.1).astype(np.float32),
                        edge_delay=np.ones_like(pre, dtype=np.int32), neuron_body_ids=np.arange(n, dtype=np.int64),
                        nt_sign=np.array([1] * 15 + [-1] * 15, dtype=np.int8), superclass_id=np.zeros(n, dtype=np.int8),
                        superclass_names=np.array(["test"]), tau_m=np.full(n, 20.0, np.float32), meta=np.array("{}"))
    return path


def build(path, **kw):
    torch.manual_seed(0)
    return FlyReservoirLM(path, vocab_size=23, d_model=16, synapse_model="coba", use_alif=False, use_stp=False, **kw).eval()


def tick(model, h, ring, ge, gi, drive):
    out = model.step(h, torch.zeros(1, dtype=torch.long), spike_ring=ring, ge=ge, gi=gi, sensory_drive=drive)
    return out[0], out[1], out[2], out[3], out[4]


def test_spike_consumes_drive_and_refractory(tmp_path):
    path = tiny_coba(tmp_path)
    n = 30
    h = torch.zeros(1, n); ring = tuple(torch.zeros(1, n) for _ in range(4))
    ge = torch.full((1, n), 0.01); gi = torch.full((1, n), 0.002)
    drive = torch.zeros(1, n); drive[0, :10] = 10.0                      # cells 0..9 forced over threshold
    with torch.no_grad():
        old = build(path)
        h1, s1, r1, ge1, gi1 = tick(old, h, ring, ge, gi, drive)
        assert s1[0, :10].min() > 0
        assert ge1[0, :10].min() > 0                                       # old engine: conductance survives the spike
        _, s2, *_ = tick(old, h1, r1, ge1, gi1, drive)
        assert s2[0, :10].min() > 0                                        # and the cell refires next tick

        new = build(path, spike_consumes_drive=True, refractory_ticks=1)
        h1, s1, r1, ge1, gi1 = tick(new, h, ring, ge, gi, drive)
        assert s1[0, :10].min() > 0
        assert ge1[0, :10].abs().max() == 0 and gi1[0, :10].abs().max() == 0 and h1[0, :10].abs().max() == 0
        silent = s1[0] == 0
        assert silent.any() and ge1[0, silent].min() > 0                   # silent cells keep their conductance
        h2, s2, r2, ge2, gi2 = tick(new, h1, r1, ge1, gi1, drive)
        assert s2[0, :10].abs().max() == 0                                 # refractory: no spike one tick later
        assert h2[0, :10].abs().max() == 0                                 # held at reset
        _, s3, *_ = tick(new, h2, r2, ge2, gi2, drive)
        assert s3[0, :10].min() > 0                                        # free again after the refractory tick


def test_refractory_ticks_bounds(tmp_path):
    path = tiny_coba(tmp_path)
    for bad in (-1, 5):
        try:
            build(path, refractory_ticks=bad)
        except ValueError:
            continue
        raise AssertionError(f'refractory_ticks={bad} accepted')
