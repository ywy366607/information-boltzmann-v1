"""Benchmark Conductance-Based (COBA) vs Current-Based (CUBA) LIF on GTX 1650 (4 GB).

Empirical test of VRAM usage, CUDA graph capturability, backward pass feasibility,
and throughput on the 25.32-million-edge biological Drosophila connectome.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch
from torch import nn
import torch.nn.functional as Fn

from information_boltzmann.core.fly_reservoir import (
    SpikeFn,
    BiologicalTopographicWriter,
    FlyReservoirLM,
)
from information_boltzmann.core.triton_synapse import execute_delayed_synaptic_transmission
from scripts.ib.train_fly_reservoir_delayed import GraphChunkRunnerDelayed


class COBAReservoirChunk(nn.Module):
    """Conductance-Based (COBA) LIF reservoir chunk.

    Synapses drive independent excitatory and inhibitory conductances:
      Delta g_E = sum_{j in Exc} W_ij s_j
      Delta g_I = sum_{j in Inh} |W_ij| s_j
      g_E(t) = lambda_{s,E} g_E(t-1) + (1 - lambda_{s,E}) Delta g_E(t)
      g_I(t) = lambda_{s,I} g_I(t-1) + (1 - lambda_{s,I}) Delta g_I(t)
      I_syn(t) = g_E(t) * (E_E - V(t-1)) + g_I(t) * (E_I - V(t-1))
    """

    def __init__(self, packed_npz: str, d_model: int = 128):
        super().__init__()
        packed = np.load(packed_npz)
        self.n_neurons = int(packed["neuron_body_ids"].shape[0])
        pre = packed["edge_pre"].astype(np.int64)
        post = packed["edge_post"].astype(np.int64)
        weight = packed["edge_weight"].astype(np.float32)
        delay = packed["edge_delay"].astype(np.int32) if "edge_delay" in packed else np.ones_like(pre, dtype=np.int32)
        sign = packed["nt_sign"]
        pre_sign = sign[pre]

        # Partition edges into Excitatory (ACh) and Inhibitory (GABA, Glu, His)
        exc_mask = (pre_sign > 0)
        inh_mask = (pre_sign < 0)

        # Sort each partition by delay (1..4) to build delay_splits
        def build_partition(mask, abs_w: bool = False):
            p_pre = pre[mask]
            p_post = post[mask]
            p_w = np.abs(weight[mask]) if abs_w else weight[mask]
            p_delay = delay[mask]
            order = np.argsort(p_delay, kind="stable")
            p_pre, p_post, p_w, p_delay = p_pre[order], p_post[order], p_w[order], p_delay[order]
            splits = [0]
            for d in (1, 2, 3, 4):
                splits.append(int(np.searchsorted(p_delay, d, side="right")))
            return (
                torch.from_numpy(p_pre.astype(np.int32)),
                torch.from_numpy(p_post.astype(np.int32)),
                torch.from_numpy(p_w.astype(np.float32)),
                tuple(splits),
            )

        pre_e, post_e, w_e, splits_e = build_partition(exc_mask, abs_w=False)
        pre_i, post_i, w_i, splits_i = build_partition(inh_mask, abs_w=True)

        self.register_buffer("edge_pre_e", pre_e, persistent=False)
        self.register_buffer("edge_post_e", post_e, persistent=False)
        self.register_buffer("edge_weight_e", w_e, persistent=False)
        self.splits_e = splits_e

        self.register_buffer("edge_pre_i", pre_i, persistent=False)
        self.register_buffer("edge_post_i", post_i, persistent=False)
        self.register_buffer("edge_weight_i", w_i, persistent=False)
        self.splits_i = splits_i

        superclass_id = packed["superclass_id"]
        superclass_names = [str(s) for s in packed["superclass_names"]]
        self.n_superclasses = len(superclass_names)
        self.register_buffer("superclass_id", torch.from_numpy(superclass_id.astype(np.int64)), persistent=False)

        # Learned time constants
        self.log_tau_m = nn.Parameter(torch.zeros(self.n_superclasses))
        self.log_tau_s_e = nn.Parameter(torch.zeros(self.n_superclasses))
        self.log_tau_s_i = nn.Parameter(torch.zeros(self.n_superclasses))
        init_tau_m = np.zeros(self.n_superclasses, dtype=np.float32)
        tau_m_arr = packed["tau_m"]
        for c in range(self.n_superclasses):
            mask = superclass_id == c
            init_tau_m[c] = float(np.mean(tau_m_arr[mask])) if mask.any() else 20.0
        self.log_tau_m.data.copy_(torch.log(torch.tensor(init_tau_m)))
        self.log_tau_s_e.data.copy_(torch.log(torch.tensor(init_tau_m / 4.0)))
        self.log_tau_s_i.data.copy_(torch.log(torch.tensor(init_tau_m / 2.0)))

        # Physical reversal potentials (normalized: V_rest = 0, V_th = 0.1)
        # E_E = +1.0 (ACh channel reversal ~ +60 mV relative to rest)
        # E_I = -0.2 (GABA/GluCl reversal ~ -12 mV relative to rest)
        self.E_E = 1.0
        self.E_I = -0.2
        self.threshold = 0.1

        self.writer = BiologicalTopographicWriter(d_model)
        self.embedding = nn.Embedding(50257, d_model)
        self.output_read = nn.Linear(self.n_neurons, d_model, bias=False)
        self.decoder = nn.Linear(d_model, 50257)
        self.decoder.weight = self.embedding.weight

    def forward_chunk(self, input_ids: torch.Tensor, targets: torch.Tensor,
                      h: torch.Tensor, spike_ring: tuple[torch.Tensor, ...],
                      g_e: torch.Tensor, g_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        tau_m = torch.clamp(torch.exp(self.log_tau_m), 1.0, 250.0)
        tau_s_e = torch.clamp(torch.exp(self.log_tau_s_e), 0.5, 100.0)
        tau_s_i = torch.clamp(torch.exp(self.log_tau_s_i), 0.5, 100.0)

        leak_m = torch.exp(-1.0 / tau_m)[self.superclass_id].unsqueeze(0)
        leak_s_e = torch.exp(-1.0 / tau_s_e)[self.superclass_id].unsqueeze(0)
        leak_s_i = torch.exp(-1.0 / tau_s_i)[self.superclass_id].unsqueeze(0)

        total_loss = input_ids.new_zeros((), dtype=torch.float32)
        T = input_ids.shape[1]
        for t in range(T):
            delta_g_e = execute_delayed_synaptic_transmission(
                spike_ring, self.edge_pre_e, self.edge_post_e, self.edge_weight_e, self.splits_e)
            delta_g_i = execute_delayed_synaptic_transmission(
                spike_ring, self.edge_pre_i, self.edge_post_i, self.edge_weight_i, self.splits_i)

            g_e = leak_s_e * g_e + (1.0 - leak_s_e) * delta_g_e
            g_i = leak_s_i * g_i + (1.0 - leak_s_i) * delta_g_i

            # Non-linear conductance-based current: I = g_E * (E_E - V) + g_I * (E_I - V)
            i_syn = g_e * (self.E_E - h) + g_i * (self.E_I - h)
            drive = self.writer(self.embedding(input_ids[:, t]), h)

            h_next = leak_m * h + i_syn + drive
            spike_next = SpikeFn.apply(h_next - self.threshold)
            h = h_next * (1.0 - spike_next)
            spike_ring = (spike_next, spike_ring[0], spike_ring[1], spike_ring[2])

            logits = self.decoder(self.output_read(h))
            total_loss = total_loss + Fn.cross_entropy(logits, targets[:, t], reduction="sum")

        return total_loss / input_ids.numel(), h, spike_ring, g_e, g_i


class GraphChunkRunnerCOBA:
    """Captures one COBA training chunk (forward + backward) in a CUDA Graph."""

    def __init__(self, model: COBAReservoirChunk, chunk_tokens: int,
                 optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.chunk = chunk_tokens
        self.ids = torch.zeros(1, chunk_tokens, dtype=torch.long, device="cuda")
        self.targets = torch.zeros_like(self.ids)
        self.h_buf = torch.zeros(1, model.n_neurons, device="cuda")
        self.ring_bufs = [torch.zeros(1, model.n_neurons, device="cuda") for _ in range(4)]
        self.ge_buf = torch.zeros(1, model.n_neurons, device="cuda")
        self.gi_buf = torch.zeros(1, model.n_neurons, device="cuda")

        original_parameters = [p.detach().clone() for p in model.parameters()]

        def chunk_fn():
            loss, h_next, next_ring, next_ge, next_gi = model.forward_chunk(
                self.ids, self.targets, self.h_buf, tuple(self.ring_bufs), self.ge_buf, self.gi_buf)
            loss.backward()
            return loss, h_next, next_ring, next_ge, next_gi

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.optimizer.zero_grad(set_to_none=False)
                chunk_fn()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.empty_cache()
            with torch.no_grad():
                for parameter, original in zip(model.parameters(), original_parameters):
                    parameter.copy_(original)
            self.optimizer.zero_grad(set_to_none=False)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss, self.h_next, self.next_ring, self.next_ge, self.next_gi = chunk_fn()

    def replay(self, ids: torch.Tensor, targets: torch.Tensor, h: torch.Tensor,
               ring: tuple[torch.Tensor, ...], ge: torch.Tensor, gi: torch.Tensor):
        self.ids.copy_(ids)
        self.targets.copy_(targets)
        self.h_buf.copy_(h.detach())
        for buf, r in zip(self.ring_bufs, ring):
            buf.copy_(r.detach())
        self.ge_buf.copy_(ge.detach())
        self.gi_buf.copy_(gi.detach())
        self.graph.replay()
        return (self.loss.detach(), self.h_next.detach(),
                tuple(r.detach() for r in self.next_ring),
                self.next_ge.detach(), self.next_gi.detach())


def benchmark_cuba(graph_npz: str, chunk_tokens: int = 32):
    print(f"\n=======================================================")
    print(f"1. BENCHMARKING CUBA (Current Production Baseline)")
    print(f"=======================================================")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    m_init = torch.cuda.memory_allocated() / 1e6
    model = FlyReservoirLM(graph_npz, d_model=128, injection="topographic", read_surface="output").cuda()
    m_model = torch.cuda.memory_allocated() / 1e6
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # 1. Native step
    ids = torch.randint(0, 1000, (1, chunk_tokens), device="cuda")
    targets = torch.randint(0, 1000, (1, chunk_tokens), device="cuda")
    h0 = torch.zeros(1, model.n_neurons, device="cuda")
    ring0 = tuple(torch.zeros(1, model.n_neurons, device="cuda") for _ in range(4))
    isyn0 = torch.zeros(1, model.n_neurons, device="cuda")

    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    loss, h1, diag, ring1, isyn1 = model.forward_chunk(ids, targets, h0, ring0, isyn0)
    torch.cuda.synchronize()
    t_fwd = (time.time() - t0) * 1000

    t1 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    t_bwd = (time.time() - t1) * 1000
    optimizer.step()

    native_peak_alloc = torch.cuda.max_memory_allocated() / 1e6
    native_peak_res = torch.cuda.max_memory_reserved() / 1e6

    print(f"  Model Buffer VRAM:  {m_model - m_init:.1f} MB")
    print(f"  Native Forward:     {t_fwd:.1f} ms ({t_fwd / chunk_tokens:.1f} ms/tok)")
    print(f"  Native Backward:    {t_bwd:.1f} ms ({t_bwd / chunk_tokens:.1f} ms/tok)")
    print(f"  Native Step Total:  {t_fwd + t_bwd:.1f} ms ({(t_fwd + t_bwd) / chunk_tokens:.1f} ms/tok)")
    print(f"  Native Peak Alloc:  {native_peak_alloc:.1f} MB")
    print(f"  Native Peak Res:    {native_peak_res:.1f} MB")

    # 2. CUDA Graph
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

    model_g = FlyReservoirLM(graph_npz, d_model=128, injection="topographic", read_surface="output").cuda()
    opt_g = torch.optim.AdamW(model_g.parameters(), lr=3e-4)

    torch.cuda.reset_peak_memory_stats()
    runner = GraphChunkRunnerDelayed(model_g, chunk_tokens, opt_g)

    # Replay 5 times
    t_rep_start = time.time()
    n_reps = 5
    for _ in range(n_reps):
        loss, h0, diag, ring0, isyn0 = runner.replay(ids, targets, h0, ring0, isyn0)
        opt_g.step()
    torch.cuda.synchronize()
    t_rep = ((time.time() - t_rep_start) / n_reps) * 1000

    graph_peak_alloc = torch.cuda.max_memory_allocated() / 1e6
    graph_peak_res = torch.cuda.max_memory_reserved() / 1e6

    print(f"  CUDA Graph Capture: SUCCESS")
    print(f"  Graph Replay Time:  {t_rep:.1f} ms ({t_rep / chunk_tokens:.1f} ms/tok)")
    print(f"  Graph Peak Alloc:   {graph_peak_alloc:.1f} MB")
    print(f"  Graph Peak Res:     {graph_peak_res:.1f} MB")
    print(f"  GTX 1650 Headroom:  {4096.0 - graph_peak_res:.1f} MB free ({((4096.0 - graph_peak_res)/4096.0)*100:.1f}%)")

    del model_g, opt_g, runner
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "model_mb": m_model - m_init,
        "native_step_ms": t_fwd + t_bwd,
        "native_alloc_mb": native_peak_alloc,
        "native_res_mb": native_peak_res,
        "graph_step_ms": t_rep,
        "graph_alloc_mb": graph_peak_alloc,
        "graph_res_mb": graph_peak_res,
        "headroom_mb": 4096.0 - graph_peak_res,
    }


def benchmark_coba(graph_npz: str, chunk_tokens: int = 32):
    print(f"\n=======================================================")
    print(f"2. BENCHMARKING COBA (Conductance-Based Physical LIF)")
    print(f"=======================================================")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    m_init = torch.cuda.memory_allocated() / 1e6
    model = COBAReservoirChunk(graph_npz, d_model=128).cuda()
    m_model = torch.cuda.memory_allocated() / 1e6
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # 1. Native step
    ids = torch.randint(0, 1000, (1, chunk_tokens), device="cuda")
    targets = torch.randint(0, 1000, (1, chunk_tokens), device="cuda")
    h0 = torch.zeros(1, model.n_neurons, device="cuda")
    ring0 = tuple(torch.zeros(1, model.n_neurons, device="cuda") for _ in range(4))
    ge0 = torch.zeros(1, model.n_neurons, device="cuda")
    gi0 = torch.zeros(1, model.n_neurons, device="cuda")

    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    loss, h1, ring1, ge1, gi1 = model.forward_chunk(ids, targets, h0, ring0, ge0, gi0)
    torch.cuda.synchronize()
    t_fwd = (time.time() - t0) * 1000

    t1 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    t_bwd = (time.time() - t1) * 1000
    optimizer.step()

    native_peak_alloc = torch.cuda.max_memory_allocated() / 1e6
    native_peak_res = torch.cuda.max_memory_reserved() / 1e6

    print(f"  Model Buffer VRAM:  {m_model - m_init:.1f} MB")
    print(f"  Native Forward:     {t_fwd:.1f} ms ({t_fwd / chunk_tokens:.1f} ms/tok)")
    print(f"  Native Backward:    {t_bwd:.1f} ms ({t_bwd / chunk_tokens:.1f} ms/tok)")
    print(f"  Native Step Total:  {t_fwd + t_bwd:.1f} ms ({(t_fwd + t_bwd) / chunk_tokens:.1f} ms/tok)")
    print(f"  Native Peak Alloc:  {native_peak_alloc:.1f} MB")
    print(f"  Native Peak Res:    {native_peak_res:.1f} MB")

    # 2. CUDA Graph
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

    model_g = COBAReservoirChunk(graph_npz, d_model=128).cuda()
    opt_g = torch.optim.AdamW(model_g.parameters(), lr=3e-4)

    torch.cuda.reset_peak_memory_stats()
    runner = GraphChunkRunnerCOBA(model_g, chunk_tokens, opt_g)

    # Replay 5 times
    t_rep_start = time.time()
    n_reps = 5
    for _ in range(n_reps):
        loss, h0, ring0, ge0, gi0 = runner.replay(ids, targets, h0, ring0, ge0, gi0)
        opt_g.step()
    torch.cuda.synchronize()
    t_rep = ((time.time() - t_rep_start) / n_reps) * 1000

    graph_peak_alloc = torch.cuda.max_memory_allocated() / 1e6
    graph_peak_res = torch.cuda.max_memory_reserved() / 1e6

    print(f"  CUDA Graph Capture: SUCCESS")
    print(f"  Graph Replay Time:  {t_rep:.1f} ms ({t_rep / chunk_tokens:.1f} ms/tok)")
    print(f"  Graph Peak Alloc:   {graph_peak_alloc:.1f} MB")
    print(f"  Graph Peak Res:     {graph_peak_res:.1f} MB")
    print(f"  GTX 1650 Headroom:  {4096.0 - graph_peak_res:.1f} MB free ({((4096.0 - graph_peak_res)/4096.0)*100:.1f}%)")

    del model_g, opt_g, runner
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "model_mb": m_model - m_init,
        "native_step_ms": t_fwd + t_bwd,
        "native_alloc_mb": native_peak_alloc,
        "native_res_mb": native_peak_res,
        "graph_step_ms": t_rep,
        "graph_alloc_mb": graph_peak_alloc,
        "graph_res_mb": graph_peak_res,
        "headroom_mb": 4096.0 - graph_peak_res,
    }


def main():
    graph_npz = "data/malecns_v1/fly_reservoir_biological.npz"

    print("=================================================================")
    print("PHYSICAL FEASIBILITY STUDY: COBA vs CUBA on NVIDIA GTX 1650 (4GB)")
    print("=================================================================")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\n")

    res_cuba = benchmark_cuba(graph_npz, chunk_tokens=32)
    res_coba = benchmark_coba(graph_npz, chunk_tokens=32)

    print("\n\n=======================================================")
    print("FINAL COMPARISON TABLE")
    print("=======================================================")
    print(f"{'Metric':<30} | {'CUBA (Current)':<15} | {'COBA (Conductance)':<18} | {'Delta (COBA - CUBA)':<20}")
    print("-" * 90)
    print(f"{'Model Buffer Footprint':<30} | {res_cuba['model_mb']:>12.1f} MB | {res_coba['model_mb']:>15.1f} MB | {res_coba['model_mb'] - res_cuba['model_mb']:>+17.1f} MB")
    print(f"{'Native Step Total (32 tok)':<30} | {res_cuba['native_step_ms']:>12.1f} ms | {res_coba['native_step_ms']:>15.1f} ms | {res_coba['native_step_ms'] - res_cuba['native_step_ms']:>+17.1f} ms")
    print(f"{'Native Per-Token Throughput':<30} | {res_cuba['native_step_ms']/32:>12.1f} ms | {res_coba['native_step_ms']/32:>15.1f} ms | {(res_coba['native_step_ms'] - res_cuba['native_step_ms'])/32:>+17.1f} ms")
    print(f"{'Native Peak Allocated VRAM':<30} | {res_cuba['native_alloc_mb']:>12.1f} MB | {res_coba['native_alloc_mb']:>15.1f} MB | {res_coba['native_alloc_mb'] - res_cuba['native_alloc_mb']:>+17.1f} MB")
    print(f"{'Native Peak Reserved VRAM':<30} | {res_cuba['native_res_mb']:>12.1f} MB | {res_coba['native_res_mb']:>15.1f} MB | {res_coba['native_res_mb'] - res_cuba['native_res_mb']:>+17.1f} MB")
    print(f"{'CUDA Graph Replay (32 tok)':<30} | {res_cuba['graph_step_ms']:>12.1f} ms | {res_coba['graph_step_ms']:>15.1f} ms | {res_coba['graph_step_ms'] - res_cuba['graph_step_ms']:>+17.1f} ms")
    print(f"{'CUDA Graph Per-Token':<30} | {res_cuba['graph_step_ms']/32:>12.1f} ms | {res_coba['graph_step_ms']/32:>15.1f} ms | {(res_coba['graph_step_ms'] - res_cuba['graph_step_ms'])/32:>+17.1f} ms")
    print(f"{'CUDA Graph Peak Alloc VRAM':<30} | {res_cuba['graph_alloc_mb']:>12.1f} MB | {res_coba['graph_alloc_mb']:>15.1f} MB | {res_coba['graph_alloc_mb'] - res_cuba['graph_alloc_mb']:>+17.1f} MB")
    print(f"{'CUDA Graph Peak Reserved':<30} | {res_cuba['graph_res_mb']:>12.1f} MB | {res_coba['graph_res_mb']:>15.1f} MB | {res_coba['graph_res_mb'] - res_cuba['graph_res_mb']:>+17.1f} MB")
    print(f"{'Remaining Headroom (4GB)':<30} | {res_cuba['headroom_mb']:>12.1f} MB | {res_coba['headroom_mb']:>15.1f} MB | {res_coba['headroom_mb'] - res_cuba['headroom_mb']:>+17.1f} MB")
    print("-" * 90)


if __name__ == "__main__":
    main()
