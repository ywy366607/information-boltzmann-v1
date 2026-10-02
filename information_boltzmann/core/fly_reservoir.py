"""Level-0 fly reservoir: a frozen, signed, full-connectome LIF substrate.

The MaleCNS v1.0 wiring (165k neurons, ~23M signed edges) is loaded frozen
and never trained.  Language enters by a learned broadcast projection into
every neuron; a learned weighted readout over all neurons feeds a decoder
whose weight is tied to the token embedding.  The only dynamics are the
biology's own: leak, synaptic transmission along real wiring, threshold.

This is the fidelity anchor of the fly ladder: no imposed IB dynamics.  Any
capability above an identically-trained random-wiring control is attributable
to the evolved topology.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as Fn


class SpikeFn(torch.autograd.Function):
    """Heaviside forward with the standard sigmoid surrogate backward."""

    @staticmethod
    def forward(ctx, voltage_minus_threshold: torch.Tensor) -> torch.Tensor:
        spike = (voltage_minus_threshold >= 0).to(voltage_minus_threshold.dtype)
        ctx.save_for_backward(voltage_minus_threshold)
        return spike

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (voltage,) = ctx.saved_tensors
        surrogate = torch.sigmoid(4.0 * voltage)
        return grad_output * surrogate * 4.0 * (1.0 - surrogate)


class SynapticTransmission(torch.autograd.Function):
    """current_i = sum_j W_ij * spikes_j along the frozen wiring.

    Memory-critical: the naive index_add implementation retains one
    [E, B] gathered tensor PER STEP for its backward (3.3 GB across a
    32-step chunk, spilling into Windows shared memory).  This function
    keeps only the edge buffers (references) in the autograd graph and
    recomputes the gather symmetrically in the backward: the backward
    of a scatter is a gather and vice versa.
    """

    @staticmethod
    def forward(ctx, spikes, edge_pre, edge_post, edge_weight):
        contribution = spikes.t()[edge_pre] * edge_weight[:, None]
        current = torch.zeros_like(spikes).t()
        current = current.index_add(0, edge_post, contribution).t()
        ctx.save_for_backward(edge_pre, edge_post, edge_weight)
        ctx.spikes_shape = spikes.shape
        return current

    @staticmethod
    def backward(ctx, grad_current):
        edge_pre, edge_post, edge_weight = ctx.saved_tensors
        grad_spikes = torch.zeros(ctx.spikes_shape, dtype=grad_current.dtype,
                                  device=grad_current.device)
        grad_spikes = grad_spikes.t().index_add(
            0, edge_pre,
            grad_current.t()[edge_post] * edge_weight[:, None]).t()
        return grad_spikes, None, None, None


def participation_ratio(weights: torch.Tensor) -> torch.Tensor:
    """Effective number of neurons carrying the weight mass (vital sign).

    Returns a TENSOR: this runs inside CUDA-graph-captured regions, where a
    host sync (float()) would invalidate the capture.  Convert with .item()
    only outside the captured region.
    """
    squared = weights.detach().float().square().flatten()
    total = squared.sum()
    # Branchless: a data-dependent python branch on a GPU tensor would
    # synchronize the stream and invalidate CUDA-graph capture.
    p = squared / total.clamp_min(1e-12)
    return (1.0 / p.square().sum().clamp_min(1e-12)).clamp_max(1e9)


class FlyReservoirLM(nn.Module):
    """Frozen MaleCNS wiring + trained broadcast input / weighted readout."""

    SENSORY_CLASSES = ("cb_sensory", "ol_sensory", "vnc_sensory",
                       "sensory_ascending", "sensory_descending",
                       "cb_sensory_tbc", "vnc_sensory_tbc",
                       "sensory_ascending_tbc")
    OUTPUT_CLASSES = ("cb_motor", "vnc_motor", "descending_neuron",
                      "cb_efferent", "vnc_efferent", "efferent_ascending",
                      "efferent_descending", "cb_endocrine", "vnc_endocrine")

    def __init__(self, graph_npz: str | Path, vocab_size: int = 50257,
                 d_model: int = 128, leak: float = 0.9, threshold: float = 0.1,
                 injection: str = "broadcast", read_surface: str = "all"):
        super().__init__()
        packed = np.load(graph_npz, allow_pickle=False)
        edge_pre = torch.from_numpy(packed["edge_pre"].astype(np.int64))
        edge_post = torch.from_numpy(packed["edge_post"].astype(np.int64))
        edge_weight = torch.from_numpy(packed["edge_weight"].astype(np.float32))
        self.n_neurons = int(packed["neuron_body_ids"].shape[0])
        # int32 halves the index memory (410 -> 205 MiB) and is accepted by
        # index_add and the custom synaptic function.
        edge_pre = edge_pre.to(torch.int32)
        edge_post = edge_post.to(torch.int32)
        # Non-persistent: rebuilt from the npz on every init, keeping
        # checkpoints free of the ~300 MB graph structure.
        self.register_buffer("edge_pre", edge_pre, persistent=False)
        self.register_buffer("edge_post", edge_post, persistent=False)
        self.register_buffer("edge_weight", edge_weight, persistent=False)
        # Frozen signed wiring: never trained, never collapsed by an objective.
        # Transmission is gather + index_add along real edges (see propagate):
        # a CSR sparse.mm backward is not CUDA-graph capturable on this
        # platform, while index_add's gather/scatter backward is.
        # Injection surface: broadcast (every neuron) or the annotated
        # sensory neurons only.  Sensory injection removes the broadcast
        # shortcut: information must flow through the wiring to be read.
        superclass_names = [str(s) for s in packed["superclass_names"]]
        superclass_id = packed["superclass_id"]
        if injection == "sensory":
            sensory = np.zeros(self.n_neurons, dtype=bool)
            for name in self.SENSORY_CLASSES:
                if name in superclass_names:
                    sensory |= superclass_id == superclass_names.index(name)
            injection_index = np.flatnonzero(sensory).astype(np.int64)
            self.injection_mode = "sensory"
        elif injection == "broadcast":
            injection_index = np.arange(self.n_neurons, dtype=np.int64)
            self.injection_mode = "broadcast"
        else:
            raise ValueError(f"Unknown injection mode: {injection}")
        self.n_injection = int(injection_index.size)
        self.register_buffer("injection_index",
                             torch.from_numpy(injection_index), persistent=False)
        # Read surface: "all" neurons, or only non-injected (interneuron/output)
        # neurons - with separated write and read surfaces, information must
        # physically traverse the wiring between them.
        if read_surface == "interneuron":
            read_mask = np.ones(self.n_neurons, dtype=np.float32)
            read_mask[injection_index] = 0.0
        elif read_surface == "output":
            # The biological action surface: motor, descending and endocrine
            # output neurons only (~2.3k of 165k) - the sharpest read
            # bottleneck in the ladder.
            read_mask = np.zeros(self.n_neurons, dtype=np.float32)
            for name in self.OUTPUT_CLASSES:
                if name in superclass_names:
                    read_mask[superclass_id == superclass_names.index(name)] = 1.0
        elif read_surface == "all":
            read_mask = np.ones(self.n_neurons, dtype=np.float32)
        else:
            raise ValueError(f"Unknown read surface: {read_surface}")
        self.read_surface = read_surface
        self.register_buffer("read_mask",
                             torch.from_numpy(read_mask), persistent=False)

        self.leak = leak
        self.threshold = threshold
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.input_proj = nn.Linear(d_model, self.n_injection, bias=False)
        self.output_read = nn.Linear(self.n_neurons, d_model, bias=False)
        self.decoder = nn.Linear(d_model, vocab_size)
        self.decoder.weight = self.embedding.weight

        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.input_proj.weight, std=0.05)
        nn.init.normal_(self.output_read.weight, std=1e-3)

    def step(self, h: torch.Tensor, token: torch.Tensor):
        """One LIF event.  h: [B, N]; token: [B]."""
        spikes = SpikeFn.apply(h - self.threshold)
        current = SynapticTransmission.apply(
            spikes, self.edge_pre, self.edge_post, self.edge_weight)
        drive = torch.zeros_like(h)
        drive.index_copy_(1, self.injection_index,
                          self.input_proj(self.embedding(token)))
        h_next = self.leak * h + current + drive
        spike_next = SpikeFn.apply(h_next - self.threshold)
        h_next = h_next - self.threshold * spike_next
        return h_next, spike_next

    def read(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.output_read(h * self.read_mask[None]))

    def forward_chunk(self, input_ids: torch.Tensor, targets: torch.Tensor,
                      h: torch.Tensor):
        """Process a chunk; read out after every step.  Returns
        (loss, h_next, diagnostics)."""
        total = input_ids.new_zeros((), dtype=torch.float32)
        spikes_all = []
        for index in range(input_ids.shape[1]):
            h, spikes = self.step(h, input_ids[:, index])
            spikes_all.append(spikes.detach())
            logits = self.read(h)
            total = total + Fn.cross_entropy(
                logits, targets[:, index], reduction="sum")
        loss = total / input_ids.numel()
        spikes_all = torch.stack(spikes_all)
        diag = {
            "firing_rate": spikes_all.mean().detach(),
            "energy": h.detach().square().mean(),
            "neuron_activity_pr": participation_ratio(
                h.detach().square().mean(0).flatten()),
            "input_weight_pr": participation_ratio(self.input_proj.weight),
            "output_weight_pr": participation_ratio(self.output_read.weight),
        }
        return loss, h, diag


def load_fly_reservoir_checkpoint(
        path: str | Path) -> tuple["FlyReservoirLM", dict]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    meta = json.loads(str(saved["graph_meta"])) if "graph_meta" in saved else {}
    model = FlyReservoirLM(saved.get("graph_npz",
                                     "data/malecns_v1/fly_reservoir_full.npz"),
                           vocab_size=saved.get("vocab_size", 50257),
                           d_model=saved.get("d_model", 128),
                           leak=saved.get("leak", 0.9),
                           threshold=saved.get("threshold", 1.0))
    model.load_state_dict(saved["model"])
    return model, {"state": saved.get("state"), "config": saved.get("config"),
                   "meta": meta}
