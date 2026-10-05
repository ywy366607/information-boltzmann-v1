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
import math
from torch import nn
import torch.nn.functional as Fn

from information_boltzmann.core.triton_synapse import (
    execute_synaptic_transmission,
    execute_delayed_synaptic_transmission,
)


class SpikeFn(torch.autograd.Function):
    """Heaviside forward with peak-normalized ATan surrogate backward.

    Surrogate: S_tilde(x) = (1 / pi) * arctan(pi * x) + 0.5
    Derivative: d/dx = 1 / (1 + (pi * x)^2)

    Properties:
    - Unit peak gain at x = 0: d/dx = 1.0 (prevents gradient vanishing/explosion).
    - Heavy-tailed Cauchy/polynomial decay: prevents subthreshold dead neurons.
    - Zero exponentials: fast, numerically stable, no exp() underflow/overflow.
    """

    @staticmethod
    def forward(ctx, voltage_minus_threshold: torch.Tensor) -> torch.Tensor:
        spike = (voltage_minus_threshold >= 0).to(voltage_minus_threshold.dtype)
        ctx.save_for_backward(voltage_minus_threshold)
        return spike

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (v_diff,) = ctx.saved_tensors
        pi_x = math.pi * v_diff
        grad = 1.0 / (1.0 + pi_x * pi_x)
        return grad_output * grad


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


class BiologicalTopographicWriter(nn.Module):
    """Biological Topographic Selective Writer (W_bio).

    Partitions the 15,912 sensory neurons into their authentic anatomical pathways:
    1. Visual (Optic Lobe: 4,107 neurons) -> Lamina/Medulla (2D/3D retinotopic spatial modes)
    2. Chemo/Olfactory (AN + gustatory: 4,987 neurons) -> Antennal Lobe -> Kenyon Cells (12.5D associative manifold)
    3. Mechanosensory/Proprioceptive (6,818 neurons) -> VNC motor coordination & AMMC steering

    Implements:
    1. Learned Sensory Gating (Selective routing, inspired by W4 chart_gate)
    2. Biological Temporal Adaptation (Innovation / Differential coding: delta_u = u - a_adapt)
    """

    def __init__(self, d_model: int, partitions_npz: Path | str = "data/malecns_v1/sensory_partitions.npz"):
        super().__init__()
        p_path = Path(partitions_npz)
        if not p_path.is_absolute():
            p_path = Path(__file__).resolve().parents[2] / p_path
        parts = np.load(p_path)
        idx_vis = torch.from_numpy(parts["visual_idx"].astype(np.int64))
        idx_chemo = torch.from_numpy(parts["chemo_idx"].astype(np.int64))
        idx_mech = torch.from_numpy(parts["mechano_idx"].astype(np.int64))

        self.n_vis = len(idx_vis)
        self.n_chemo = len(idx_chemo)
        self.n_mech = len(idx_mech)
        self.n_total = self.n_vis + self.n_chemo + self.n_mech

        combined = np.concatenate([
            parts["visual_idx"],
            parts["chemo_idx"],
            parts["mechano_idx"],
        ]).astype(np.int64)
        self.register_buffer("injection_index", torch.from_numpy(combined), persistent=False)

        # 1. Modality Gating: maps token embedding to 3 modality routing weights
        self.gate_linear = nn.Linear(d_model, 3)

        # 2. Dedicated per-modality projections
        self.proj_vis = nn.Linear(d_model, self.n_vis, bias=False)
        self.proj_chemo = nn.Linear(d_model, self.n_chemo, bias=False)
        self.proj_mech = nn.Linear(d_model, self.n_mech, bias=False)

        # Biological adaptation rate (tau_adapt ~ 20 ms -> lambda_adapt ~ 0.9512)
        # Slow baseline subtraction removes DC semantic drift, passing innovations
        self.lambda_adapt = 0.9512
        self.register_buffer("a_adapt", torch.zeros(1, self.n_total), persistent=False)

        # Initialization: calibrated for 3.5% biological spiking threshold
        nn.init.normal_(self.proj_vis.weight, std=0.25)
        nn.init.normal_(self.proj_chemo.weight, std=0.25)
        nn.init.normal_(self.proj_mech.weight, std=0.25)
        nn.init.zeros_(self.gate_linear.bias)
        nn.init.normal_(self.gate_linear.weight, std=0.02)

    @property
    def idx_vis(self) -> torch.Tensor:
        return self.injection_index[:self.n_vis]

    @property
    def idx_chemo(self) -> torch.Tensor:
        return self.injection_index[self.n_vis:self.n_vis + self.n_chemo]

    @property
    def idx_mech(self) -> torch.Tensor:
        return self.injection_index[self.n_vis + self.n_chemo:]

    def forward(self, token_emb: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """token_emb: [B, d_model]; h: [B, N] (full brain state) -> returns drive: [B, N]"""
        B = token_emb.shape[0]
        # 3.0 * softmax preserves unit mean modality scale across the 3 pathways
        gates = 3.0 * torch.softmax(self.gate_linear(token_emb), dim=-1)

        p_vis = self.proj_vis(token_emb)
        p_chemo = self.proj_chemo(token_emb)
        p_mech = self.proj_mech(token_emb)

        u_vis = gates[:, 0:1] * p_vis
        u_chemo = gates[:, 1:2] * p_chemo
        u_mech = gates[:, 2:3] * p_mech
        u_all = torch.cat([u_vis, u_chemo, u_mech], dim=-1)

        self.last_token_emb = token_emb
        self.last_gates = gates
        self.last_p_vis = p_vis
        self.last_p_chemo = p_chemo
        self.last_p_mech = p_mech

        if B == 1:
            innovation = u_all - self.a_adapt
            self.a_adapt.copy_(self.lambda_adapt * self.a_adapt + (1.0 - self.lambda_adapt) * u_all.detach())
        else:
            if not hasattr(self, "_a_adapt_eval") or self._a_adapt_eval.shape[0] != B or self._a_adapt_eval.device != token_emb.device:
                self._a_adapt_eval = torch.zeros(B, self.n_total, device=token_emb.device, dtype=token_emb.dtype)
            innovation = u_all - self._a_adapt_eval
            self._a_adapt_eval.copy_(self.lambda_adapt * self._a_adapt_eval + (1.0 - self.lambda_adapt) * u_all.detach())

        drive = torch.zeros_like(h)
        drive.index_copy_(1, self.injection_index, innovation)
        return drive

    def reset_eval(self) -> None:
        if hasattr(self, "_a_adapt_eval"):
            self._a_adapt_eval.zero_()

    def forward_with_state(self, token_emb: torch.Tensor, h: torch.Tensor,
                           baseline: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Functional writer for BPTT; baseline is a continuing physical state.

        Unlike the legacy buffer API, this retains the adaptation dependency
        within a training window. The caller detaches only at window boundaries.
        """
        gates = 3.0 * torch.softmax(self.gate_linear(token_emb), dim=-1)
        packet = torch.cat((gates[:, :1] * self.proj_vis(token_emb),
                            gates[:, 1:2] * self.proj_chemo(token_emb),
                            gates[:, 2:3] * self.proj_mech(token_emb)), dim=-1)
        drive = torch.zeros_like(h).index_copy(1, self.injection_index, packet - baseline)
        next_baseline = self.lambda_adapt * baseline + (1.0 - self.lambda_adapt) * packet
        return drive, next_baseline


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
                 injection: str = "broadcast", read_surface: str = "all",
                 read_centering: bool = False,
                 learnable_time_constants: bool = True,
                 learnable_thresholds: bool = True,
                 learnable_conductance_gains: bool = True,
                 use_alif: bool = False,
                 use_stp: bool = False,
                 synapse_model: str = "auto",
                 E_E: float = 1.0, E_I: float = -0.2,
                 decoder_bias: bool = True):
        super().__init__()
        packed = np.load(graph_npz, allow_pickle=False)
        self.n_neurons = int(packed["neuron_body_ids"].shape[0])

        if synapse_model == "auto":
            if "edge_pre_e" in packed or ("nt_sign" in packed and "coba" in str(graph_npz).lower()):
                synapse_model = "coba"
            else:
                synapse_model = "cuba"
        self.synapse_model = synapse_model
        self.E_E = float(E_E)
        self.E_I = float(E_I)

        if self.synapse_model == "coba":
            if "edge_pre_e" in packed:
                self.register_buffer("edge_pre_e", torch.from_numpy(packed["edge_pre_e"].astype(np.int32)), persistent=False)
                self.register_buffer("edge_post_e", torch.from_numpy(packed["edge_post_e"].astype(np.int32)), persistent=False)
                self.register_buffer("edge_weight_e", torch.from_numpy(packed["edge_weight_e"].astype(np.float32)), persistent=False)
                self.splits_e = tuple(int(x) for x in packed["delay_splits_e"])

                self.register_buffer("edge_pre_i", torch.from_numpy(packed["edge_pre_i"].astype(np.int32)), persistent=False)
                self.register_buffer("edge_post_i", torch.from_numpy(packed["edge_post_i"].astype(np.int32)), persistent=False)
                self.register_buffer("edge_weight_i", torch.from_numpy(packed["edge_weight_i"].astype(np.float32)), persistent=False)
                self.splits_i = tuple(int(x) for x in packed["delay_splits_i"])
            else:
                pre = packed["edge_pre"].astype(np.int64)
                post = packed["edge_post"].astype(np.int64)
                weight = packed["edge_weight"].astype(np.float32)
                delay = packed["edge_delay"].astype(np.int32) if "edge_delay" in packed else np.ones_like(pre, dtype=np.int32)
                sign = packed["nt_sign"]
                pre_sign = sign[pre]

                exc_mask = (pre_sign > 0)
                inh_mask = (pre_sign < 0)

                def _build_part(mask, is_abs: bool):
                    p_pre, p_post, p_w, p_delay = pre[mask], post[mask], (np.abs(weight[mask]) if is_abs else weight[mask]), delay[mask]
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

                pre_e, post_e, w_e, splits_e = _build_part(exc_mask, is_abs=False)
                pre_i, post_i, w_i, splits_i = _build_part(inh_mask, is_abs=True)
                self.register_buffer("edge_pre_e", pre_e, persistent=False)
                self.register_buffer("edge_post_e", post_e, persistent=False)
                self.register_buffer("edge_weight_e", w_e, persistent=False)
                self.splits_e = splits_e
                self.register_buffer("edge_pre_i", pre_i, persistent=False)
                self.register_buffer("edge_post_i", post_i, persistent=False)
                self.register_buffer("edge_weight_i", w_i, persistent=False)
                self.splits_i = splits_i
            self.has_delays = True
        else:
            edge_pre = torch.from_numpy(packed["edge_pre"].astype(np.int32))
            edge_post = torch.from_numpy(packed["edge_post"].astype(np.int32))
            edge_weight = torch.from_numpy(packed["edge_weight"].astype(np.float32))
            self.register_buffer("edge_pre", edge_pre, persistent=False)
            self.register_buffer("edge_post", edge_post, persistent=False)
            self.register_buffer("edge_weight", edge_weight, persistent=False)
            if "delay_splits" in packed:
                self.has_delays = True
                self.delay_splits = tuple(int(x) for x in packed["delay_splits"])
            else:
                self.has_delays = False
                self.delay_splits = None

        # Scheme B biological topological baseline leak (zero magic numbers)
        if "lambda_0" in packed:
            self.has_biological_leak = True
            self.register_buffer("lambda_0", torch.from_numpy(packed["lambda_0"].astype(np.float32)), persistent=False)
        else:
            self.has_biological_leak = False

        # Superclass metadata and learned cell-type specific time constants (Flyvis paradigm)
        if "superclass_names" in packed and "superclass_id" in packed:
            superclass_names = [str(s) for s in packed["superclass_names"]]
            superclass_id = packed["superclass_id"]
            self.n_superclasses = len(superclass_names)
            self.superclass_names = superclass_names
            self.register_buffer("superclass_id", torch.from_numpy(superclass_id.astype(np.int64)), persistent=False)
        else:
            superclass_names = []
            superclass_id = None
            self.n_superclasses = 0
            self.superclass_names = []
            self.superclass_id = None

        self.learnable_time_constants = learnable_time_constants and (self.n_superclasses > 0)
        if self.learnable_time_constants:
            init_tau_m = np.zeros(self.n_superclasses, dtype=np.float32)
            tau_m_arr = packed["tau_m"] if "tau_m" in packed else None
            for c in range(self.n_superclasses):
                if tau_m_arr is not None:
                    mask_c = (superclass_id == c)
                    c_mean = float(np.mean(tau_m_arr[mask_c])) if mask_c.any() else 20.0
                else:
                    c_mean = 20.0
                init_tau_m[c] = max(c_mean, 1.0)
            self.log_tau_m = nn.Parameter(torch.log(torch.tensor(init_tau_m, dtype=torch.float32)))
            if self.synapse_model == "coba":
                self.log_tau_s_e = nn.Parameter(torch.log(torch.tensor(init_tau_m / 4.0, dtype=torch.float32)))
                self.log_tau_s_i = nn.Parameter(torch.log(torch.tensor(init_tau_m / 2.0, dtype=torch.float32)))
            else:
                self.log_tau_s = nn.Parameter(torch.log(torch.tensor(init_tau_m / 4.0, dtype=torch.float32)))

        self.learnable_thresholds = learnable_thresholds and (self.n_superclasses > 0)
        if self.learnable_thresholds:
            init_thresh = max(float(threshold), 0.01)
            self.log_threshold = nn.Parameter(
                torch.full((self.n_superclasses,), math.log(init_thresh), dtype=torch.float32)
            )
        else:
            self.threshold = float(threshold)

        self.learnable_conductance_gains = bool(learnable_conductance_gains) and (self.synapse_model == "coba")
        if self.learnable_conductance_gains:
            self.log_g_e = nn.Parameter(torch.zeros((), dtype=torch.float32))
            self.log_g_i = nn.Parameter(torch.zeros((), dtype=torch.float32))

        self.use_alif = bool(use_alif) and (self.n_superclasses > 0)
        if self.use_alif:
            # ALIF activity-dependent adaptation (Bellec et al. 2018):
            # tau_a: slow adaptation timescale per superclass (~50 steps initial)
            self.log_tau_a = nn.Parameter(
                torch.full((self.n_superclasses,), math.log(50.0), dtype=torch.float32)
            )
            # beta: adaptation coupling strength per superclass (~0.05 initial)
            self.log_beta = nn.Parameter(
                torch.full((self.n_superclasses,), math.log(0.05), dtype=torch.float32)
            )

        self.use_stp = bool(use_stp) and (self.n_superclasses > 0)
        if self.use_stp:
            # STP: Tsodyks-Markram (1998) Dynamic Synapses (Depression + Facilitation)
            # logit_u0: baseline release probability U0 per superclass (init ~0.25 -> logit(0.25) ~ -1.0986)
            # log_tau_fac: facilitation timescale per superclass (init ~100 ms -> log(100) ~ 4.6052)
            # log_tau_rec: depression recovery timescale per superclass (init ~200 ms -> log(200) ~ 5.2983)
            init_logit_u0 = math.log(0.25 / 0.75)
            self.logit_u0 = nn.Parameter(
                torch.full((self.n_superclasses,), init_logit_u0, dtype=torch.float32)
            )
            self.log_tau_fac = nn.Parameter(
                torch.full((self.n_superclasses,), math.log(100.0), dtype=torch.float32)
            )
            self.log_tau_rec = nn.Parameter(
                torch.full((self.n_superclasses,), math.log(200.0), dtype=torch.float32)
            )


        # Biological dopamine neuromodulatory circuit (241k delayed synapses)
        if "dan_edge_pre" in packed:
            self.has_dopamine = True
            self.register_buffer("dan_edge_pre", torch.from_numpy(packed["dan_edge_pre"].astype(np.int64)), persistent=False)
            self.register_buffer("dan_edge_post", torch.from_numpy(packed["dan_edge_post"].astype(np.int64)), persistent=False)
            self.register_buffer("dan_edge_weight", torch.from_numpy(packed["dan_edge_weight"].astype(np.float32)), persistent=False)
            self.dan_delay_splits = tuple(int(x) for x in packed["dan_delay_splits"])
            self.dan_scale = float(packed["dan_scale"]) if "dan_scale" in packed else 0.005044
            if "dan_indices" in packed:
                self.register_buffer("dan_indices", torch.from_numpy(packed["dan_indices"].astype(np.int64)), persistent=False)
            else:
                self.register_buffer("dan_indices", torch.unique(self.dan_edge_pre), persistent=False)
        else:
            self.has_dopamine = False
            self.dan_delay_splits = None
            self.dan_scale = 1.0
        # Frozen signed wiring: never trained, never collapsed by an objective.
        # Transmission is gather + index_add along real edges (see propagate):
        # a CSR sparse.mm backward is not CUDA-graph capturable on this
        # platform, while index_add's gather/scatter backward is.
        # Injection surface: broadcast (every neuron) or the annotated
        # sensory neurons only.  Sensory injection removes the broadcast
        # shortcut: information must flow through the wiring to be read.
        if injection == "topographic":
            self.topographic_writer = BiologicalTopographicWriter(d_model)
            self.injection_mode = "topographic"
            injection_index = self.topographic_writer.injection_index.cpu().numpy()
        elif injection == "sensory":
            sensory = np.zeros(self.n_neurons, dtype=bool)
            for name in self.SENSORY_CLASSES:
                if name in superclass_names:
                    sensory |= superclass_id == superclass_names.index(name)
            injection_index = np.flatnonzero(sensory).astype(np.int64)
            self.injection_mode = "sensory"
            self.topographic_writer = None
        elif injection == "broadcast":
            injection_index = np.arange(self.n_neurons, dtype=np.int64)
            self.injection_mode = "broadcast"
            self.topographic_writer = None
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
        self.read_centering = bool(read_centering)
        self.register_buffer("read_mask",
                             torch.from_numpy(read_mask), persistent=False)
        read_indices = np.flatnonzero(read_mask).astype(np.int64)
        self.n_read = int(read_indices.size)
        self.register_buffer("read_indices",
                             torch.from_numpy(read_indices), persistent=False)

        self.leak = leak
        self.threshold = threshold
        self.embedding = nn.Embedding(vocab_size, d_model)
        if self.injection_mode == "topographic":
            self.input_proj = None
        else:
            self.input_proj = nn.Linear(d_model, self.n_injection, bias=False)
            nn.init.normal_(self.input_proj.weight, std=0.05)

        if self.read_surface == "all":
            self.output_read = nn.Linear(self.n_neurons, d_model, bias=False)
        else:
            self.output_read = nn.Linear(self.n_read, d_model, bias=False)

        self.read_norm = nn.RMSNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size, bias=decoder_bias)
        self.decoder.weight = nn.Parameter(self.embedding.weight.clone())
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)

        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.output_read.weight, std=1e-3)

    def get_decay_rates(self) -> tuple[torch.Tensor, ...]:
        """Computes per-neuron [1, N] decay rates."""
        if self.synapse_model == "coba":
            if self.learnable_time_constants:
                tau_m = torch.clamp(torch.exp(self.log_tau_m), min=1.0, max=250.0)
                tau_s_e = torch.clamp(torch.exp(self.log_tau_s_e), min=0.5, max=100.0)
                tau_s_i = torch.clamp(torch.exp(self.log_tau_s_i), min=0.5, max=100.0)
                leak_m = torch.exp(-1.0 / tau_m)[self.superclass_id].unsqueeze(0)
                leak_s_e = torch.exp(-1.0 / tau_s_e)[self.superclass_id].unsqueeze(0)
                leak_s_i = torch.exp(-1.0 / tau_s_i)[self.superclass_id].unsqueeze(0)
                return leak_m, leak_s_e, leak_s_i
            elif self.has_biological_leak:
                leak_m = self.lambda_0.unsqueeze(0)
                return leak_m, torch.pow(leak_m, 4.0), torch.pow(leak_m, 2.0)
            else:
                device = self.edge_pre_e.device
                leak_m = torch.full((1, self.n_neurons), self.leak, device=device)
                return leak_m, torch.zeros_like(leak_m), torch.zeros_like(leak_m)
        else:
            if self.learnable_time_constants:
                tau_m = torch.clamp(torch.exp(self.log_tau_m), min=1.0, max=250.0)
                tau_s = torch.clamp(torch.exp(self.log_tau_s), min=0.5, max=100.0)
                leak_m = torch.exp(-1.0 / tau_m)[self.superclass_id].unsqueeze(0)
                leak_s = torch.exp(-1.0 / tau_s)[self.superclass_id].unsqueeze(0)
                return leak_m, leak_s
            elif self.has_biological_leak:
                leak_t = self.lambda_0.unsqueeze(0)
                return leak_t, torch.pow(leak_t, 4.0)
            else:
                device = self.edge_pre.device
                leak_t = torch.full((1, self.n_neurons), self.leak, device=device)
                return leak_t, torch.zeros_like(leak_t)

    def get_thresholds(self) -> torch.Tensor:
        """Computes per-neuron [1, N] firing thresholds."""
        if getattr(self, "learnable_thresholds", False):
            theta_c = torch.clamp(torch.exp(self.log_threshold), min=0.01, max=2.0)
            return theta_c[self.superclass_id].unsqueeze(0)
        else:
            return torch.tensor(self.threshold)

    def get_conductance_gains(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes global scalar conductance gains (g_e, g_i)."""
        if getattr(self, "learnable_conductance_gains", False):
            g_e = torch.clamp(torch.exp(self.log_g_e), min=0.01, max=100.0)
            g_i = torch.clamp(torch.exp(self.log_g_i), min=0.01, max=100.0)
            return g_e, g_i
        else:
            return torch.tensor(1.0), torch.tensor(1.0)

    def get_alif_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes per-neuron [1, N] adaptation decay rate rho and strength beta."""
        if getattr(self, "use_alif", False):
            tau_a = torch.clamp(torch.exp(self.log_tau_a), min=2.0, max=500.0)
            rho = torch.exp(-1.0 / tau_a)[self.superclass_id].unsqueeze(0)
            beta = torch.clamp(torch.exp(self.log_beta), min=1e-4, max=2.0)[self.superclass_id].unsqueeze(0)
            return rho, beta
        else:
            dev = self.edge_pre_e.device if hasattr(self, "edge_pre_e") else self.edge_pre.device
            return torch.ones((1, self.n_neurons), device=dev), torch.zeros((1, self.n_neurons), device=dev)

    def get_stp_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes per-neuron [1, N] STP parameters: U_0, rho_fac, rho_rec, norm."""
        if getattr(self, "use_stp", False) and self.n_superclasses > 0:
            u0_c = torch.clamp(torch.sigmoid(self.logit_u0), min=0.05, max=0.95)
            tau_fac_c = torch.clamp(torch.exp(self.log_tau_fac), min=5.0, max=1000.0)
            tau_rec_c = torch.clamp(torch.exp(self.log_tau_rec), min=5.0, max=2000.0)

            u0 = u0_c[self.superclass_id].unsqueeze(0)
            rho_fac = torch.exp(-1.0 / tau_fac_c)[self.superclass_id].unsqueeze(0)
            rho_rec = torch.exp(-1.0 / tau_rec_c)[self.superclass_id].unsqueeze(0)
            norm = (u0 * (2.0 - u0)).clamp_min(1e-4)
            return u0, rho_fac, rho_rec, norm
        else:
            dev = self.edge_pre_e.device if hasattr(self, "edge_pre_e") else self.edge_pre.device
            ones = torch.ones((1, self.n_neurons), device=dev)
            return ones, ones, ones, ones

    def step(self, h: torch.Tensor, token: torch.Tensor,
             spike_ring: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | list[torch.Tensor] | None = None,
             i_syn: torch.Tensor | None = None,
             ge: torch.Tensor | None = None,
             gi: torch.Tensor | None = None,
             b: torch.Tensor | None = None,
             x: torch.Tensor | None = None,
             u: torch.Tensor | None = None,
             base_rates: tuple[torch.Tensor, ...] | None = None,
             thresholds: torch.Tensor | float | None = None,
             conductance_gains: tuple[torch.Tensor, torch.Tensor] | None = None,
             alif_params: tuple[torch.Tensor, torch.Tensor] | None = None,
             stp_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
             return_biophysics: bool = False,
             sensory_drive: torch.Tensor | None = None):
        """One LIF event with continuous physical synaptic conductance (COBA) or current (CUBA).
        h: [B, N]; token: [B]"""
        if sensory_drive is not None:
            if sensory_drive.shape != h.shape:
                raise ValueError("sensory_drive must have the full physical-state shape")
            drive = sensory_drive
        elif self.injection_mode == "topographic":
            drive = self.topographic_writer(self.embedding(token), h)
        else:
            drive = torch.zeros_like(h)
            drive.index_copy_(1, self.injection_index,
                              self.input_proj(self.embedding(token)))

        if base_rates is None:
            base_rates = self.get_decay_rates()
        if thresholds is None:
            thresholds = self.get_thresholds()
        if conductance_gains is None and self.synapse_model == "coba":
            conductance_gains = self.get_conductance_gains()
        if alif_params is None and getattr(self, "use_alif", False):
            alif_params = self.get_alif_params()
        if stp_params is None and getattr(self, "use_stp", False):
            stp_params = self.get_stp_params()

        if self.synapse_model == "coba":
            if spike_ring is None:
                spike_ring = tuple(torch.zeros_like(h) for _ in range(4))
            delta_ge = execute_delayed_synaptic_transmission(
                spike_ring, self.edge_pre_e, self.edge_post_e, self.edge_weight_e, self.splits_e)
            delta_gi = execute_delayed_synaptic_transmission(
                spike_ring, self.edge_pre_i, self.edge_post_i, self.edge_weight_i, self.splits_i)

            leak_m, leak_se, leak_si = base_rates
            if ge is None:
                ge = torch.zeros_like(h)
            if gi is None:
                gi = torch.zeros_like(h)

            ge_next = leak_se * ge + (1.0 - leak_se) * delta_ge
            gi_next = leak_si * gi + (1.0 - leak_si) * delta_gi
            g_e, g_i = conductance_gains
            G_E = g_e * ge_next
            G_I = g_i * gi_next

            # Exact continuous-time exponential integrator (GDN physical forget gate):
            # Biological membrane passive leak conductance: g_L = -log(leak_m)
            g_L = -torch.log(leak_m.clamp(min=1e-5, max=1.0 - 1e-7))
            g_total = g_L + G_E + G_I

            # alpha(t) = exp(-Delta_t / tau_eff) = exp(-g_total) = leak_m * exp(-(G_E + G_I))
            # Guaranteed alpha in (0, 1] strictly, eliminating negative coefficient ringing!
            alpha = torch.exp(-g_total.clamp(min=1e-5, max=20.0))

            # Steady-state drive: i_drive = G_E * E_E + G_I * E_I + drive
            i_drive = G_E * self.E_E + G_I * self.E_I + drive

            # Integration multiplier: beta = (1 - alpha) / g_total
            beta_int = (1.0 - alpha) / g_total.clamp_min(1e-5)

            # Pre-spike membrane potential (before reset): V_pre = alpha * h + beta * i_drive
            v_pre = alpha * h + beta_int * i_drive

            # ALIF activity-dependent threshold: theta_t = theta_0 + beta * b_t
            if getattr(self, "use_alif", False):
                if b is None:
                    b = torch.zeros_like(h)
                rho_a, beta_a = alif_params
                eff_threshold = thresholds + beta_a * b
            else:
                eff_threshold = thresholds

            spike_next = SpikeFn.apply(v_pre - eff_threshold)
            h_next = v_pre * (1.0 - spike_next)

            # ALIF slow adaptation state update: b_{t+1} = rho * b_t + (1 - rho) * s_t
            if getattr(self, "use_alif", False):
                b_next = rho_a * b + (1.0 - rho_a) * spike_next

            # STP dynamic synapse update:
            if getattr(self, "use_stp", False):
                u0, rho_fac, rho_rec, norm = stp_params
                if x is None:
                    x = torch.ones_like(h)
                if u is None:
                    u = u0.expand_as(h).clone()

                u_active = u + u0 * (1.0 - u) * spike_next
                transmitted_pulse = torch.clamp((u_active * x / norm) * spike_next, max=3.0)
                x_post = x - u_active * x * spike_next

                u_next = u0 + (u_active - u0) * rho_fac
                x_next = 1.0 + (x_post - 1.0) * rho_rec
            else:
                transmitted_pulse = spike_next

            next_spike_ring = (transmitted_pulse, spike_ring[0], spike_ring[1], spike_ring[2])

            ret = [h_next, spike_next, next_spike_ring, ge_next, gi_next]
            if getattr(self, "use_alif", False):
                ret.append(b_next)
            if getattr(self, "use_stp", False):
                ret.extend([x_next, u_next])

            if return_biophysics:
                biophysics = {
                    "v_pre": v_pre,
                    "alpha_eff": alpha,
                    "beta_int": beta_int,
                    "eff_threshold": eff_threshold,
                    "g_total": g_total,
                    "delayed_pulses": (spike_ring[0], spike_ring[1], spike_ring[2], spike_ring[3]),
                    "transmitted_pulse": transmitted_pulse,
                    "leak_se": leak_se,
                    "leak_si": leak_si,
                    "g_e": g_e,
                    "g_i": g_i,
                }
                return tuple(ret), biophysics
            return tuple(ret)

        else:
            user_passed_i_syn = (i_syn is not None)
            if self.has_delays:
                if spike_ring is None:
                    spike_ring = tuple(torch.zeros_like(h) for _ in range(4))
                current = execute_delayed_synaptic_transmission(
                    spike_ring, self.edge_pre, self.edge_post, self.edge_weight, self.delay_splits)
            else:
                spikes_now = SpikeFn.apply(h - self.threshold)
                current = execute_synaptic_transmission(
                    spikes_now, self.edge_pre, self.edge_post, self.edge_weight)

            leak_m, leak_s = base_rates
            # Biological continuous leak + delayed dopamine modulation
            if self.has_dopamine and self.has_delays:
                M = torch.zeros_like(h)
                for d in range(1, 5):
                    s_idx = self.dan_delay_splits[d - 1]
                    e_idx = self.dan_delay_splits[d]
                    if e_idx > s_idx:
                        pre_d = self.dan_edge_pre[s_idx:e_idx]
                        post_d = self.dan_edge_post[s_idx:e_idx]
                        w_d = self.dan_edge_weight[s_idx:e_idx]
                        spk_d = spike_ring[d - 1][:, pre_d]
                        M.index_add_(1, post_d, spk_d * w_d[None, :])
                norm_M = M / max(self.dan_scale, 1e-8)
                mod = torch.exp(-norm_M)
                leak_t = torch.pow(leak_m, mod)
                leak_syn_t = torch.pow(leak_s, mod)
            else:
                leak_t = leak_m
                leak_syn_t = leak_s

            if i_syn is None:
                i_syn_next = current
            else:
                i_syn_next = leak_syn_t * i_syn + (1.0 - leak_syn_t) * current

            v_pre = leak_t * h + i_syn_next + drive
            if getattr(self, "use_alif", False):
                if b is None:
                    b = torch.zeros_like(h)
                rho_a, beta_a = alif_params
                eff_threshold = thresholds + beta_a * b
            else:
                eff_threshold = thresholds

            spike_next = SpikeFn.apply(v_pre - eff_threshold)
            h_next = v_pre * (1.0 - spike_next)

            if getattr(self, "use_alif", False):
                b_next = rho_a * b + (1.0 - rho_a) * spike_next

            if getattr(self, "use_stp", False):
                u0, rho_fac, rho_rec, norm = stp_params
                if x is None:
                    x = torch.ones_like(h)
                if u is None:
                    u = u0.expand_as(h).clone()

                u_active = u + u0 * (1.0 - u) * spike_next
                transmitted_pulse = torch.clamp((u_active * x / norm) * spike_next, max=3.0)
                x_post = x - u_active * x * spike_next

                u_next = u0 + (u_active - u0) * rho_fac
                x_next = 1.0 + (x_post - 1.0) * rho_rec
            else:
                transmitted_pulse = spike_next

            ret = [h_next, spike_next]
            if self.has_delays:
                next_spike_ring = (transmitted_pulse, spike_ring[0], spike_ring[1], spike_ring[2])
                ret.append(next_spike_ring)
            if user_passed_i_syn:
                ret.append(i_syn_next)
            if getattr(self, "use_alif", False):
                ret.append(b_next)
            if getattr(self, "use_stp", False):
                ret.extend([x_next, u_next])

            if return_biophysics:
                delayed_pulses = (spike_ring[0], spike_ring[1], spike_ring[2], spike_ring[3]) if self.has_delays else None
                biophysics = {
                    "v_pre": v_pre,
                    "alpha_eff": leak_t,
                    "beta_int": torch.ones_like(v_pre),
                    "eff_threshold": eff_threshold,
                    "g_total": torch.ones_like(v_pre),
                    "delayed_pulses": delayed_pulses,
                    "transmitted_pulse": transmitted_pulse,
                }
                return tuple(ret), biophysics
            return tuple(ret)


    def read(self, h: torch.Tensor) -> torch.Tensor:
        if self.read_surface == "all":
            z = self.output_read(h * self.read_mask[None])
        elif h.shape[-1] == self.n_read:
            z = self.output_read(h)
        else:
            z = self.output_read(h[:, self.read_indices])
        if hasattr(self, "read_norm"):
            z = self.read_norm(z)
        return self.decoder(z)

    def forward_chunk(self, input_ids: torch.Tensor, targets: torch.Tensor,
                      h: torch.Tensor,
                      spike_ring: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | list[torch.Tensor] | None = None,
                      i_syn: torch.Tensor | None = None,
                      ge: torch.Tensor | None = None,
                      gi: torch.Tensor | None = None,
                      b: torch.Tensor | None = None,
                      x: torch.Tensor | None = None,
                      u: torch.Tensor | None = None):
        """Process a chunk; read out after every step."""
        base_rates = self.get_decay_rates()
        thresholds = self.get_thresholds()
        alif_params = self.get_alif_params() if getattr(self, "use_alif", False) else None
        stp_params = self.get_stp_params() if getattr(self, "use_stp", False) else None
        total = input_ids.new_zeros((), dtype=torch.float32)
        spikes_all = []
        pulses_all = []

        if self.synapse_model == "coba":
            conductance_gains = self.get_conductance_gains()
            g_e_val, g_i_val = conductance_gains
            if ge is None:
                ge = torch.zeros_like(h)
            if gi is None:
                gi = torch.zeros_like(h)
            if getattr(self, "use_alif", False) and b is None:
                b = torch.zeros_like(h)
            if getattr(self, "use_stp", False):
                if x is None:
                    x = torch.ones_like(h)
                if u is None:
                    u = stp_params[0].detach().expand_as(h).clone()

            for index in range(input_ids.shape[1]):
                step_kwargs = {
                    "ge": ge, "gi": gi,
                    "base_rates": base_rates,
                    "thresholds": thresholds,
                    "conductance_gains": conductance_gains,
                }
                if getattr(self, "use_alif", False):
                    step_kwargs["b"] = b
                    step_kwargs["alif_params"] = alif_params
                if getattr(self, "use_stp", False):
                    step_kwargs["x"] = x
                    step_kwargs["u"] = u
                    step_kwargs["stp_params"] = stp_params

                step_res = self.step(h, input_ids[:, index], spike_ring, **step_kwargs)
                h, spikes, spike_ring, ge, gi = step_res[0], step_res[1], step_res[2], step_res[3], step_res[4]
                idx = 5
                if getattr(self, "use_alif", False):
                    b = step_res[idx]
                    idx += 1
                if getattr(self, "use_stp", False):
                    x = step_res[idx]
                    u = step_res[idx + 1]
                    idx += 2

                spikes_all.append(spikes.detach())
                pulses_all.append(spike_ring[0].detach())
                logits = self.read(h)
                total = total + Fn.cross_entropy(logits, targets[:, index], reduction="sum")
            loss = total / input_ids.numel()
            spikes_all = torch.stack(spikes_all)
            pulses_all = torch.stack(pulses_all)
            input_pr = ((participation_ratio(self.topographic_writer.proj_vis.weight) +
                         participation_ratio(self.topographic_writer.proj_chemo.weight) +
                         participation_ratio(self.topographic_writer.proj_mech.weight)) / 3.0
                        if self.injection_mode == "topographic"
                        else participation_ratio(self.input_proj.weight))
            diag = {
                "firing_rate": spikes_all.mean().detach(),
                "energy": h.detach().square().mean(),
                "ge_energy": ge.detach().square().mean(),
                "gi_energy": gi.detach().square().mean(),
                "neuron_activity_pr": participation_ratio(
                    h.detach().square().mean(0).flatten()),
                "input_weight_pr": input_pr,
                "output_weight_pr": participation_ratio(self.output_read.weight),
                "tau_m_mean": torch.clamp(torch.exp(self.log_tau_m), min=1.0, max=250.0).mean().detach() if self.learnable_time_constants else torch.tensor(0.0),
                "tau_s_e_mean": torch.clamp(torch.exp(self.log_tau_s_e), min=0.5, max=100.0).mean().detach() if self.learnable_time_constants else torch.tensor(0.0),
                "tau_s_i_mean": torch.clamp(torch.exp(self.log_tau_s_i), min=0.5, max=100.0).mean().detach() if self.learnable_time_constants else torch.tensor(0.0),
                "threshold_mean": torch.clamp(torch.exp(self.log_threshold), min=0.01, max=2.0).mean().detach() if getattr(self, "learnable_thresholds", False) else torch.tensor(float(self.threshold)),
                "threshold_min": torch.clamp(torch.exp(self.log_threshold), min=0.01, max=2.0).min().detach() if getattr(self, "learnable_thresholds", False) else torch.tensor(float(self.threshold)),
                "threshold_max": torch.clamp(torch.exp(self.log_threshold), min=0.01, max=2.0).max().detach() if getattr(self, "learnable_thresholds", False) else torch.tensor(float(self.threshold)),
                "g_e": g_e_val.detach() if getattr(self, "learnable_conductance_gains", False) else torch.tensor(1.0),
                "g_i": g_i_val.detach() if getattr(self, "learnable_conductance_gains", False) else torch.tensor(1.0),
            }
            if getattr(self, "use_alif", False):
                tau_a_val = torch.clamp(torch.exp(self.log_tau_a), min=2.0, max=500.0)
                beta_val = torch.clamp(torch.exp(self.log_beta), min=1e-4, max=2.0)
                diag["b_mean"] = b.mean().detach()
                diag["b_max"] = b.max().detach()
                diag["tau_a_mean"] = tau_a_val.mean().detach()
                diag["beta_mean"] = beta_val.mean().detach()

            if getattr(self, "use_stp", False):
                u0_c = torch.clamp(torch.sigmoid(self.logit_u0), min=0.05, max=0.95)
                tau_fac_c = torch.clamp(torch.exp(self.log_tau_fac), min=5.0, max=1000.0)
                tau_rec_c = torch.clamp(torch.exp(self.log_tau_rec), min=5.0, max=2000.0)
                diag["x_mean"] = x.mean().detach()
                diag["u_mean"] = u.mean().detach()
                diag["u0_mean"] = u0_c.mean().detach()
                diag["tau_fac_mean"] = tau_fac_c.mean().detach()
                diag["tau_rec_mean"] = tau_rec_c.mean().detach()
                diag["pulse_mean"] = pulses_all.mean().detach()

            ret = [loss, h, diag, spike_ring, ge, gi]
            if getattr(self, "use_alif", False):
                ret.append(b)
            if getattr(self, "use_stp", False):
                ret.extend([x, u])
            return tuple(ret)

        else:
            user_passed_i_syn = (i_syn is not None)
            if i_syn is None:
                i_syn = torch.zeros_like(h)
            if getattr(self, "use_alif", False) and b is None:
                b = torch.zeros_like(h)
            if getattr(self, "use_stp", False):
                if x is None:
                    x = torch.ones_like(h)
                if u is None:
                    u = stp_params[0].detach().expand_as(h).clone()

            for index in range(input_ids.shape[1]):
                step_kwargs = {
                    "i_syn": i_syn,
                    "base_rates": base_rates,
                    "thresholds": thresholds
                }
                if getattr(self, "use_alif", False):
                    step_kwargs["b"] = b
                    step_kwargs["alif_params"] = alif_params
                if getattr(self, "use_stp", False):
                    step_kwargs["x"] = x
                    step_kwargs["u"] = u
                    step_kwargs["stp_params"] = stp_params

                step_res = self.step(h, input_ids[:, index], spike_ring, **step_kwargs)
                h, spikes = step_res[0], step_res[1]
                idx = 2
                if self.has_delays:
                    spike_ring = step_res[idx]; idx += 1
                i_syn = step_res[idx]; idx += 1
                if getattr(self, "use_alif", False):
                    b = step_res[idx]; idx += 1
                if getattr(self, "use_stp", False):
                    x = step_res[idx]; idx += 1
                    u = step_res[idx]; idx += 1

                spikes_all.append(spikes.detach())
                if self.has_delays:
                    pulses_all.append(spike_ring[0].detach())
                logits = self.read(h)
                total = total + Fn.cross_entropy(logits, targets[:, index], reduction="sum")
            loss = total / input_ids.numel()
            spikes_all = torch.stack(spikes_all)
            input_pr = ((participation_ratio(self.topographic_writer.proj_vis.weight) +
                         participation_ratio(self.topographic_writer.proj_chemo.weight) +
                         participation_ratio(self.topographic_writer.proj_mech.weight)) / 3.0
                        if self.injection_mode == "topographic"
                        else participation_ratio(self.input_proj.weight))
            diag = {
                "firing_rate": spikes_all.mean().detach(),
                "energy": h.detach().square().mean(),
                "syn_energy": i_syn.detach().square().mean(),
                "neuron_activity_pr": participation_ratio(
                    h.detach().square().mean(0).flatten()),
                "input_weight_pr": input_pr,
                "output_weight_pr": participation_ratio(self.output_read.weight),
                "tau_m_mean": torch.clamp(torch.exp(self.log_tau_m), min=1.0, max=250.0).mean().detach() if self.learnable_time_constants else torch.tensor(0.0),
                "tau_s_mean": torch.clamp(torch.exp(self.log_tau_s), min=0.5, max=100.0).mean().detach() if self.learnable_time_constants else torch.tensor(0.0),
                "threshold_mean": torch.clamp(torch.exp(self.log_threshold), min=0.01, max=2.0).mean().detach() if getattr(self, "learnable_thresholds", False) else torch.tensor(float(self.threshold)),
                "threshold_min": torch.clamp(torch.exp(self.log_threshold), min=0.01, max=2.0).min().detach() if getattr(self, "learnable_thresholds", False) else torch.tensor(float(self.threshold)),
                "threshold_max": torch.clamp(torch.exp(self.log_threshold), min=0.01, max=2.0).max().detach() if getattr(self, "learnable_thresholds", False) else torch.tensor(float(self.threshold)),
            }
            if getattr(self, "use_alif", False):
                tau_a_val = torch.clamp(torch.exp(self.log_tau_a), min=2.0, max=500.0)
                beta_val = torch.clamp(torch.exp(self.log_beta), min=1e-4, max=2.0)
                diag["b_mean"] = b.mean().detach()
                diag["b_max"] = b.max().detach()
                diag["tau_a_mean"] = tau_a_val.mean().detach()
                diag["beta_mean"] = beta_val.mean().detach()

            if getattr(self, "use_stp", False):
                u0_c = torch.clamp(torch.sigmoid(self.logit_u0), min=0.05, max=0.95)
                tau_fac_c = torch.clamp(torch.exp(self.log_tau_fac), min=5.0, max=1000.0)
                tau_rec_c = torch.clamp(torch.exp(self.log_tau_rec), min=5.0, max=2000.0)
                diag["x_mean"] = x.mean().detach()
                diag["u_mean"] = u.mean().detach()
                diag["u0_mean"] = u0_c.mean().detach()
                diag["tau_fac_mean"] = tau_fac_c.mean().detach()
                diag["tau_rec_mean"] = tau_rec_c.mean().detach()
                if len(pulses_all) > 0:
                    diag["pulse_mean"] = torch.stack(pulses_all).mean().detach()

            ret = [loss, h, diag]
            if self.has_delays:
                ret.append(spike_ring)
            if user_passed_i_syn:
                ret.append(i_syn)
            if getattr(self, "use_alif", False):
                ret.append(b)
            if getattr(self, "use_stp", False):
                ret.extend([x, u])
            return tuple(ret)


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
