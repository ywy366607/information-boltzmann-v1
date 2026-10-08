"""Graph-Constrained Distributed State Observer with Predict-Arrive-Correct Closed Loop.

Implements the HX-1 architecture:
1. 4-Node Functional Connectome Macro-Graph derived from MaleCNS_v1:
   - Node 0: Sensory Port (15.9k neurons)
   - Node 1: Central Brain Association & Navigation Hub (133.2k neurons)
   - Node 2: Premotor VNC / CPG Circuits (13.3k neurons)
   - Node 3: Motor & Descending Output (2.7k neurons)
2. Closed-Loop Innovation Update (Predict-Arrive-Correct):
   \\varepsilon_{i,t} = E_i(S_{i,t}) - \\hat{z}_{i,t|t-1}
3. 14-Horizon Graph Neural Dynamics simulating message passing along macro-adjacency A^T.
4. Motor Node Multi-Horizon Dynamic Delay Attention Readout to Decoder.
"""

from __future__ import annotations
from typing import Any
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from information_boltzmann.core.sigreg import SIGReg
from information_boltzmann.core.lejepa_predictor import DynamicDelayAttention


class MacroRegionGraph(nn.Module):
    """Encodes the 4-node macro-connectome topology with empirical E/I synaptic delay tiers.

    Derived from MaleCNS_v1 (25.32M synapses):
    - 4 Delay Tiers: d in {1, 2, 3, 4} ticks
    - Separate Excitatory (A_E) and Inhibitory (A_I) directed adjacencies [4, 4, 4].
    """

    def __init__(self, graph_npz_path: str | Path | None = None):
        super().__init__()
        macro_npz_path = Path("data/malecns_v1/malecns_macro_4region_delayed.npz")
        if macro_npz_path.exists():
            data = np.load(macro_npz_path, allow_pickle=False)
            raw_E = torch.from_numpy(data["A_E"].astype(np.float32))  # [4, 4, 4]
            raw_I = torch.from_numpy(data["A_I"].astype(np.float32))  # [4, 4, 4]
            # Normalize by total outgoing weight per region across all delays and targets
            W_tot = raw_E.sum(dim=(0, 2), keepdim=True) + raw_I.sum(dim=(0, 2), keepdim=True) + 1e-6
            norm_E = raw_E / W_tot
            norm_I = raw_I / W_tot
        else:
            # Fallback to normalized male CNS empirical distributions
            norm_E = torch.zeros(4, 4, 4)
            norm_I = torch.zeros(4, 4, 4)
            norm_E[0] = torch.tensor([[0.21, 0.24, 0.23, 0.02],
                                      [0.01, 0.47, 0.01, 0.01],
                                      [0.01, 0.03, 0.35, 0.07],
                                      [0.01, 0.08, 0.01, 0.07]])
            norm_I[0] = torch.tensor([[0.05, 0.14, 0.00, 0.00],
                                      [0.02, 0.33, 0.00, 0.01],
                                      [0.08, 0.05, 0.46, 0.07],
                                      [0.00, 0.04, 0.00, 0.04]])

        self.register_buffer("A_E", norm_E)  # [4, 4, 4]
        self.register_buffer("A_I", norm_I)  # [4, 4, 4]
        # Aggregate directed adjacency A for backward compatibility
        self.register_buffer("A", norm_E.sum(dim=0) + norm_I.sum(dim=0))
        self.num_nodes = 4
        self.num_tiers = 4


class RegionalPhysicalEncoders(nn.Module):
    """Maps physical membrane potential states of the 4 macro-regions into latent space."""

    def __init__(
        self,
        graph_npz_path: str | Path,
        d_model: int = 768,
        sample_per_region: int = 2048,
        output_read: nn.Module | None = None,
        read_indices: torch.Tensor | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self.d_model = d_model
        data = np.load(graph_npz_path, allow_pickle=False)
        sc = data["superclass_id"]

        # 4 macro-region superclass groupings
        groups = {
            0: [6, 12, 13, 14, 15, 24, 25],          # Sensory
            1: [0, 2, 3, 4, 1, 11, 16, 17, 18, 19],  # Central Association & Navigation Hub
            2: [20, 21, 22, 26],                      # Premotor VNC
            3: [5, 8, 9, 10, 23],                     # Motor Output & Descending
        }

        rng = np.random.RandomState(seed)
        sampled_indices = []
        for g in range(4):
            idx = np.where(np.isin(sc, groups[g]))[0]
            n_sample = min(sample_per_region, len(idx))
            s_idx = rng.choice(idx, size=n_sample, replace=False)
            sampled_indices.append(np.sort(s_idx))

        # Register sampled index tensors as buffers
        self.register_buffer("idx_sensory", torch.as_tensor(sampled_indices[0], dtype=torch.long))
        self.register_buffer("idx_central", torch.as_tensor(sampled_indices[1], dtype=torch.long))
        self.register_buffer("idx_premotor", torch.as_tensor(sampled_indices[2], dtype=torch.long))

        # Lightweight regional projection layers
        self.proj_sensory = nn.Linear(len(sampled_indices[0]), d_model)
        self.proj_central = nn.Linear(len(sampled_indices[1]), d_model)
        self.proj_premotor = nn.Linear(len(sampled_indices[2]), d_model)

        if output_read is not None:
            self.proj_motor = output_read
            if read_indices is not None:
                self.register_buffer("idx_motor", torch.as_tensor(read_indices, dtype=torch.long))
            else:
                self.register_buffer("idx_motor", torch.as_tensor(sampled_indices[3], dtype=torch.long))
        else:
            self.register_buffer("idx_motor", torch.as_tensor(sampled_indices[3], dtype=torch.long))
            self.proj_motor = nn.Linear(len(sampled_indices[3]), d_model)

        # Multi-channel auxiliary physical flux adapters:
        # Channels: E conductance (ge), I conductance (gi), ALIF/STP adaptation (b, x-1, u-u0), and 4 delay tiers (ring 1..4)
        # Separate projections prevent 8D null-space collapse between E/I and across arrival delays.
        self.proj_ge_sensory = nn.Linear(len(sampled_indices[0]), d_model, bias=False)
        self.proj_ge_central = nn.Linear(len(sampled_indices[1]), d_model, bias=False)
        self.proj_ge_premotor = nn.Linear(len(sampled_indices[2]), d_model, bias=False)
        self.proj_ge_motor = nn.Linear(len(self.idx_motor), d_model, bias=False)

        self.proj_gi_sensory = nn.Linear(len(sampled_indices[0]), d_model, bias=False)
        self.proj_gi_central = nn.Linear(len(sampled_indices[1]), d_model, bias=False)
        self.proj_gi_premotor = nn.Linear(len(sampled_indices[2]), d_model, bias=False)
        self.proj_gi_motor = nn.Linear(len(self.idx_motor), d_model, bias=False)

        self.proj_adapt_sensory = nn.Linear(3 * len(sampled_indices[0]), d_model, bias=False)
        self.proj_adapt_central = nn.Linear(3 * len(sampled_indices[1]), d_model, bias=False)
        self.proj_adapt_premotor = nn.Linear(3 * len(sampled_indices[2]), d_model, bias=False)
        self.proj_adapt_motor = nn.Linear(3 * len(self.idx_motor), d_model, bias=False)

        self.proj_ring_sensory = nn.Linear(4 * len(sampled_indices[0]), d_model, bias=False)
        self.proj_ring_central = nn.Linear(4 * len(sampled_indices[1]), d_model, bias=False)
        self.proj_ring_premotor = nn.Linear(4 * len(sampled_indices[2]), d_model, bias=False)
        self.proj_ring_motor = nn.Linear(4 * len(self.idx_motor), d_model, bias=False)

        # Legacy aliases for backward compatibility:
        self.proj_flux_sensory = self.proj_ge_sensory
        self.proj_flux_central = self.proj_ge_central
        self.proj_flux_premotor = self.proj_ge_premotor
        self.proj_flux_motor = self.proj_ge_motor
        self.num_flux_channels = 9

        # Zero-initialization: exact zero-shock guarantee on existing decoder
        for proj in [
            self.proj_ge_sensory, self.proj_ge_central, self.proj_ge_premotor, self.proj_ge_motor,
            self.proj_gi_sensory, self.proj_gi_central, self.proj_gi_premotor, self.proj_gi_motor,
            self.proj_adapt_sensory, self.proj_adapt_central, self.proj_adapt_premotor, self.proj_adapt_motor,
            self.proj_ring_sensory, self.proj_ring_central, self.proj_ring_premotor, self.proj_ring_motor,
        ]:
            nn.init.zeros_(proj.weight)

        # Layer norms for regional physical latents
        self.norm_sensory = nn.LayerNorm(d_model)
        self.norm_central = nn.LayerNorm(d_model)
        self.norm_premotor = nn.LayerNorm(d_model)
        self.norm_motor = nn.LayerNorm(d_model)

    def _extract_regional_flux(
        self,
        idx: torch.Tensor,
        ge: torch.Tensor,
        gi: torch.Tensor,
        b: torch.Tensor,
        x: torch.Tensor,
        u: torch.Tensor,
        ring: tuple[torch.Tensor, ...] | list[torch.Tensor],
        proj_ge: nn.Linear,
        proj_gi: nn.Linear,
        proj_adapt: nn.Linear,
        proj_ring: nn.Linear,
        u0: float = 0.25,
    ) -> torch.Tensor:
        """Extracts and independently projects E/I, adaptation, and 4 delay tiers for the specified region."""
        ge_r = ge[:, idx]
        gi_r = gi[:, idx]
        b_r = b[:, idx] if b.numel() > 0 else torch.zeros_like(ge_r)
        x_r = (x[:, idx] - 1.0) if x.numel() > 0 else torch.zeros_like(ge_r)
        u_r = (u[:, idx] - u0) if u.numel() > 0 else torch.zeros_like(ge_r)
        ring_r = [p[:, idx] for p in ring[:4]]
        while len(ring_r) < 4:
            ring_r.append(torch.zeros_like(ge_r))

        adapt_cat = torch.cat([b_r, x_r, u_r], dim=-1)
        ring_cat = torch.cat(ring_r, dim=-1)

        return proj_ge(ge_r) + proj_gi(gi_r) + proj_adapt(adapt_cat) + proj_ring(ring_cat)

    def forward(
        self,
        state_or_h: Any,
        ge: torch.Tensor | None = None,
        gi: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        u: torch.Tensor | None = None,
        ring: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        """Args:
            state_or_h: [T, n_neurons] or [B, n_neurons] tensor, or a FlyPhysicalState,
                        or list of FlyPhysicalState objects.

        Returns:
            Z_physical: [T, 4, d_model], observed regional latent vectors for all 4 nodes.
        """
        if isinstance(state_or_h, list) and len(state_or_h) > 0 and hasattr(state_or_h[0], 'h'):
            h = torch.cat([s.h if s.h.dim() == 2 else s.h.unsqueeze(0) for s in state_or_h], dim=0)
            ge = torch.cat([s.ge if s.ge.dim() == 2 else s.ge.unsqueeze(0) for s in state_or_h], dim=0)
            gi = torch.cat([s.gi if s.gi.dim() == 2 else s.gi.unsqueeze(0) for s in state_or_h], dim=0)
            b = torch.cat([s.b if s.b.dim() == 2 else s.b.unsqueeze(0) for s in state_or_h], dim=0) if state_or_h[0].b.numel() > 0 else torch.empty(0, device=h.device)
            x = torch.cat([s.x if s.x.dim() == 2 else s.x.unsqueeze(0) for s in state_or_h], dim=0) if state_or_h[0].x.numel() > 0 else torch.empty(0, device=h.device)
            u = torch.cat([s.u if s.u.dim() == 2 else s.u.unsqueeze(0) for s in state_or_h], dim=0) if state_or_h[0].u.numel() > 0 else torch.empty(0, device=h.device)
            ring = tuple(
                torch.cat([s.ring[d] if s.ring[d].dim() == 2 else s.ring[d].unsqueeze(0) for s in state_or_h], dim=0)
                for d in range(min(4, len(state_or_h[0].ring)))
            ) if (hasattr(state_or_h[0], 'ring') and state_or_h[0].ring is not None and len(state_or_h[0].ring) > 0) else ()
        elif hasattr(state_or_h, 'h') and hasattr(state_or_h, 'ge') and hasattr(state_or_h, 'ring'):
            h = state_or_h.h if state_or_h.h.dim() == 2 else state_or_h.h.unsqueeze(0)
            ge = state_or_h.ge if state_or_h.ge.dim() == 2 else state_or_h.ge.unsqueeze(0)
            gi = state_or_h.gi if state_or_h.gi.dim() == 2 else state_or_h.gi.unsqueeze(0)
            b = state_or_h.b if (state_or_h.b.dim() == 2 or state_or_h.b.numel() == 0) else state_or_h.b.unsqueeze(0)
            x = state_or_h.x if (state_or_h.x.dim() == 2 or state_or_h.x.numel() == 0) else state_or_h.x.unsqueeze(0)
            u = state_or_h.u if (state_or_h.u.dim() == 2 or state_or_h.u.numel() == 0) else state_or_h.u.unsqueeze(0)
            ring = tuple(p if p.dim() == 2 else p.unsqueeze(0) for p in state_or_h.ring) if (state_or_h.ring is not None and len(state_or_h.ring) > 0) else ()
        else:
            h = state_or_h if state_or_h.dim() == 2 else state_or_h.unsqueeze(0)

        # h: [T, n_neurons]
        h_s = h[:, self.idx_sensory]
        h_c = h[:, self.idx_central]
        h_p = h[:, self.idx_premotor]
        h_m = h[:, self.idx_motor]

        z_s = self.proj_sensory(h_s)
        z_c = self.proj_central(h_c)
        z_p = self.proj_premotor(h_p)
        z_m = self.proj_motor(h_m)

        if ge is not None and gi is not None and ring is not None and len(ring) > 0:
            flux_s = self._extract_regional_flux(
                self.idx_sensory, ge, gi, b, x, u, ring,
                self.proj_ge_sensory, self.proj_gi_sensory, self.proj_adapt_sensory, self.proj_ring_sensory
            )
            flux_c = self._extract_regional_flux(
                self.idx_central, ge, gi, b, x, u, ring,
                self.proj_ge_central, self.proj_gi_central, self.proj_adapt_central, self.proj_ring_central
            )
            flux_p = self._extract_regional_flux(
                self.idx_premotor, ge, gi, b, x, u, ring,
                self.proj_ge_premotor, self.proj_gi_premotor, self.proj_adapt_premotor, self.proj_ring_premotor
            )
            flux_m = self._extract_regional_flux(
                self.idx_motor, ge, gi, b, x, u, ring,
                self.proj_ge_motor, self.proj_gi_motor, self.proj_adapt_motor, self.proj_ring_motor
            )

            z_s = z_s + flux_s
            z_c = z_c + flux_c
            z_p = z_p + flux_p
            z_m = z_m + flux_m

        z_s = self.norm_sensory(z_s)
        z_c = self.norm_central(z_c)
        z_p = self.norm_premotor(z_p)
        z_m = self.norm_motor(z_m)

        # Stack into [T, 4, d_model]
        return torch.stack([z_s, z_c, z_p, z_m], dim=1)

    def make_admissible_perturbation(
        self,
        h_ref: torch.Tensor,
        delta_z: torch.Tensor,
        alpha: float = 0.05,
        max_delta: float = 0.2,
    ) -> torch.Tensor:
        """Constructs an admissible perturbed whole-brain physical state h_q from delta_z.

        delta_z: [4, d_model], student discrepancy in latent space.
        h_ref: [n_neurons], reference physical membrane potentials.
        """
        if delta_z.norm() < 1e-7:
            return h_ref.clone()
        h_q = h_ref.clone()
        projs = [self.proj_sensory, self.proj_central, self.proj_premotor, self.proj_motor]
        indices = [self.idx_sensory, self.idx_central, self.idx_premotor, self.idx_motor]

        for r in range(4):
            W_r = projs[r].weight  # [d_model, n_r]
            pullback = torch.matmul(delta_z[r], W_r)  # [n_r]
            p_norm = pullback.norm()
            if p_norm < 1e-7:
                continue
            unit_dir = pullback / (p_norm + 1e-6)
            delta_h_r = alpha * torch.clamp(unit_dir, -max_delta, max_delta)
            h_q[indices[r]] = torch.clamp(h_q[indices[r]] + delta_h_r, min=-0.5, max=1.5)

        return h_q


class GraphNeuralTransition(nn.Module):
    """Simulates delayed E/I message passing along connectome macro-edges."""

    def __init__(self, d_model: int = 768):
        super().__init__()
        self.d_model = d_model
        # Node update function conditioned on:
        # 1. Local state H_k [d_model]
        # 2. Incoming delayed Excitatory message M_E [d_model]
        # 3. Incoming delayed Inhibitory message M_I [d_model]
        self.node_mlp = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.normal_(self.node_mlp[-1].weight, std=1e-3)
        nn.init.zeros_(self.node_mlp[-1].bias)

    def forward_step(
        self,
        history_H: list[torch.Tensor],
        A_E: torch.Tensor,
        A_I: torch.Tensor,
    ) -> torch.Tensor:
        """Single tick step conditioned on historical states and delayed E/I channels.

        Args:
            history_H: list of [T, 4, d_model] states up to current tick k.
                       history_H[-1] is H_k.
            A_E: [4, 4, 4] (tier d in 1..4, pre, post)
            A_I: [4, 4, 4] (tier d in 1..4, pre, post)
        """
        cur_H = history_H[-1]

        # Accumulate incoming delayed messages across tiers d = 1..4
        M_E = torch.zeros_like(cur_H)
        M_I = torch.zeros_like(cur_H)

        for d in range(1, 5):
            past_H = history_H[-d] if len(history_H) >= d else torch.zeros_like(cur_H)
            # A_E[d-1]: [4, 4], pre -> post. Message to post j is sum_i A_{i, j} past_H_i, i.e., A^T past_H
            a_e = A_E[d - 1].t()
            a_i = A_I[d - 1].t()
            M_E = M_E + torch.matmul(a_e, past_H)
            M_I = M_I + torch.matmul(a_i, past_H)

        combined = torch.cat([cur_H, M_E, M_I], dim=-1)
        delta = self.node_mlp(combined)

        # Unconstrained physical residual: allows incoming messages to excite quiescent / zero-norm nodes.
        return cur_H + delta

    def forward(
        self,
        H: torch.Tensor,
        A_E: torch.Tensor,
        A_I: torch.Tensor,
    ) -> torch.Tensor:
        """Single-step forward for 1-step prediction (where history is just [H])."""
        return self.forward_step([H], A_E, A_I)


class MotorDelayAttention(nn.Module):
    """Dynamic Delay Attention across K=0..max_horizon motor horizons."""

    def __init__(self, d_model: int = 768, max_horizon: int = 14, d_attn: int = 128):
        super().__init__()
        self.d_model = d_model
        self.max_horizon = max_horizon
        self.d_attn = d_attn

        self.q_proj = nn.Linear(d_model, d_attn)
        self.k_proj = nn.Linear(d_model, d_attn)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.hop_positions = nn.Parameter(torch.randn(max_horizon + 1, d_model) * 0.02)
        self.q_norm = nn.LayerNorm(d_attn)
        self.k_norm = nn.LayerNorm(d_attn)

    def forward(self, hops_stack: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """hops_stack: [T, K+1, d_model]"""
        z_0 = hops_stack[:, 0]
        q = self.q_norm(self.q_proj(z_0)).unsqueeze(1)  # [T, 1, d_attn]

        k_in = hops_stack + self.hop_positions.unsqueeze(0)
        k = self.k_norm(self.k_proj(k_in))              # [T, K+1, d_attn]

        scores = torch.bmm(q, k.transpose(1, 2)) / (self.d_attn ** 0.5)
        attn_weights = F.softmax(scores.squeeze(1), dim=-1)  # [T, K+1]

        # Extract predicted future displacement relative to present state z_0
        delta_hops = hops_stack - z_0.unsqueeze(1)           # [T, K+1, d_model], horizon 0 is exactly 0
        v = self.v_proj(delta_hops)                          # [T, K+1, d_model]
        context = torch.bmm(attn_weights.unsqueeze(1), v).squeeze(1)  # [T, d_model]
        return context, attn_weights


class FlyGraphObserver(nn.Module):
    """HX-1: Graph-Constrained Distributed State Observer with Closed-Loop Calibration."""

    def __init__(
        self,
        graph_npz_path: str | Path,
        d_model: int = 768,
        max_horizon: int = 14,
        lambda_obs: float = 1.0,
        lambda_sigreg: float = 0.2,
        sample_per_region: int = 2048,
        output_read: nn.Module | None = None,
        read_indices: torch.Tensor | None = None,
        init_gamma: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_horizon = max_horizon
        self.lambda_obs = lambda_obs
        self.lambda_sigreg = lambda_sigreg

        # Connectome macro-topology with delayed E/I channels
        self.macro_graph = MacroRegionGraph(graph_npz_path)

        # Physical regional encoders E_i(S_{i,t})
        self.encoders = RegionalPhysicalEncoders(
            graph_npz_path,
            d_model=d_model,
            sample_per_region=sample_per_region,
            output_read=output_read,
            read_indices=read_indices,
        )

        # Graph neural transition dynamics
        self.transition = GraphNeuralTransition(d_model=d_model)

        # Learnable Kalman innovation gain parameter K_i in (0, 1)
        self.raw_gain = nn.Parameter(torch.zeros(4, d_model))

        # Dynamic Delay Attention over motor node (Node 3) multi-horizon predictions
        self.delay_attn = MotorDelayAttention(d_model=d_model, max_horizon=max_horizon)

        # Bounded scalar coupling factor raw_gamma in R; gamma = tanh(raw_gamma) in (-1, 1).
        # Initialized to init_gamma (default: 0.0). When raw_gamma == 0, readout exactly restores baseline.
        self.raw_gamma = nn.Parameter(torch.full((1,), float(init_gamma)))

        # SIGReg anti-collapse regularizer
        self.sigreg = SIGReg(dim=d_model, num_slices=128)

    @property
    def gamma(self) -> torch.Tensor:
        """Bounded scalar coupling factor gamma in (-1, 1)."""
        return torch.tanh(self.raw_gamma)

    @property
    def kalman_gain(self) -> torch.Tensor:
        """Constrained Kalman gain K in (0, 1)."""
        return torch.sigmoid(self.raw_gain)

    def simulate_hops(
        self,
        z_start: torch.Tensor,
        num_hops: int,
        history: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Simulates num_hops forward ticks along delayed connectome edges.

        Args:
            z_start: [B, 4, d_model] or [4, d_model] initial regional state.
            num_hops: Number of internal physical ticks to simulate.
            history: Optional list/tuple/tensor of historical states prior to z_start.

        Returns:
            list of [B, 4, d_model] simulated regional latent states for k=1..num_hops.
        """
        A_E, A_I = self.macro_graph.A_E, self.macro_graph.A_I
        z_0 = z_start.squeeze(0) if (z_start.dim() == 3 and z_start.shape[0] == 1) else z_start
        if history is not None and len(history) > 0:
            if isinstance(history, (tuple, list)):
                history_H = [h.squeeze(0) if (h.dim() == 3 and h.shape[0] == 1) else h for h in history] + [z_0]
            elif isinstance(history, torch.Tensor) and history.numel() > 0:
                history_H = [history[i].squeeze(0) if (history[i].dim() == 3 and history[i].shape[0] == 1) else history[i]
                             for i in range(history.shape[0])] + [z_0]
            else:
                history_H = [z_0]
        else:
            history_H = [z_0]

        simulated = []
        for _ in range(num_hops):
            next_H = self.transition.forward_step(history_H, A_E, A_I)
            history_H.append(next_H)
            simulated.append(next_H)
        return simulated

    def forward_window(
        self,
        h_sequence: torch.Tensor,
        prior_state: torch.Tensor | None = None,
        prior_history: tuple[torch.Tensor, ...] | list[torch.Tensor] | torch.Tensor | None = None,
        teacher_quiet_h: torch.Tensor | None = None,
        query_pairs: list[Any] | None = None,
        physical_states: list | None = None,
        teacher_quiet_states: list | None = None,
        return_history: bool = True,
        anchor_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, float], torch.Tensor]:
        """Runs the Predict-Arrive-Correct observer across the sequence window.

        Args:
            h_sequence: [T, n_neurons], whole-brain physical membrane potentials for all tokens in window.
            prior_state: Optional [4, d_model], prior observer state from previous window.
            prior_history: Optional recent up to 4 posterior states [H, 4, d_model] from previous window.
            teacher_quiet_h: Optional [K, n_neurons], ground-truth whole-brain physical quiet evolution.
            query_pairs: Optional list of (q_history, z_target) pairs from student-error-guided physical queries.
            physical_states: Optional list of T FlyPhysicalState objects with complete physical dynamics.
            teacher_quiet_states: Optional list of K FlyPhysicalState objects for quiet trajectory.
            return_history: Whether to return next_history as the 5th element.

        Returns:
            z_readout: [T, d_model], motor node multi-delay attended representation for Decoder.
            total_loss: Observer innovation MSE loss + OPD loss + Query aggregation loss + SIGReg.
            metrics: Diagnostic metrics dictionary.
            next_prior: [4, d_model], prior state for the next window boundary.
            next_history: (optional) [H, 4, d_model], recent posterior history for the next window boundary.
        """
        T = h_sequence.shape[0]
        K_horizons = self.max_horizon
        A_E, A_I = self.macro_graph.A_E, self.macro_graph.A_I

        # 1. ARRIVE: Extract ground-truth physical regional states Z_star [T, 4, d_model]
        if physical_states is not None and len(physical_states) == T:
            Z_star = self.encoders(physical_states)
        else:
            Z_star = self.encoders(h_sequence)

        # 2. PREDICT-ARRIVE-CORRECT CLOSED-LOOP
        innovations = []
        Z_posterior_list = []

        # Parse prior_history: list of [4, d_model]
        if prior_history is not None:
            if isinstance(prior_history, (tuple, list)):
                running_history = [h.squeeze(0) if (h.dim() == 3 and h.shape[0] == 1) else h for h in prior_history]
            elif isinstance(prior_history, torch.Tensor) and prior_history.numel() > 0:
                running_history = [prior_history[i].squeeze(0) if (prior_history[i].dim() == 3 and prior_history[i].shape[0] == 1) else prior_history[i]
                                   for i in range(prior_history.shape[0])]
            else:
                running_history = []
        else:
            running_history = []

        if prior_state is not None and prior_state.numel() > 0:
            cur_prior = prior_state.squeeze(0) if (prior_state.dim() == 3 and prior_state.shape[0] == 1) else prior_state
        elif len(running_history) > 0:
            cur_prior = self.transition.forward_step(running_history, A_E, A_I)
        else:
            cur_prior = Z_star[0]

        gain = self.kalman_gain  # [4, d_model]

        for t in range(T):
            z_star_t = Z_star[t]  # [4, d_model]
            # Innovation error: \varepsilon_t = Z_star_t - \hat{Z}_{t|t-1}
            eps_t = z_star_t - cur_prior
            innovations.append(eps_t)

            # Posterior state update: Z_t = \hat{Z}_{t|t-1} + K * \varepsilon_t
            z_post_t = cur_prior + gain * eps_t
            Z_posterior_list.append(z_post_t)
            running_history.append(z_post_t)

            # Advance 1 step to form prior for t+1 conditioned on real multi-tier history
            cur_prior = self.transition.forward_step(running_history, A_E, A_I)

        Z_posterior = torch.stack(Z_posterior_list, dim=0)  # [T, 4, d_model]
        eps_stack = torch.stack(innovations, dim=0)          # [T, 4, d_model]

        # Observer 1-step innovation MSE loss
        loss_obs = F.mse_loss(eps_stack, torch.zeros_like(eps_stack))

        # 3. FORWARD MULTI-HOP GRAPH SIMULATION (along connectome macro-edges with delayed E/I channels)
        # Starting from posterior state Z_posterior [T, 4, d_model]:
        motor_hops = [Z_posterior[:, 3]]  # Horizon 0 is immediate motor node (Node 3)
        simulated_regional_hops = []

        all_posts = running_history
        N_prior = len(running_history) - T
        zero_pad = torch.zeros_like(Z_posterior[0])

        history_H = []
        for d in (3, 2, 1):
            past_t = []
            for t in range(T):
                idx = N_prior + t - d
                if idx >= 0:
                    past_t.append(all_posts[idx])
                else:
                    past_t.append(zero_pad)
            history_H.append(torch.stack(past_t, dim=0))
        history_H.append(Z_posterior)

        for k in range(1, K_horizons + 1):
            next_H = self.transition.forward_step(history_H, A_E, A_I)
            history_H.append(next_H)
            # Node 3 is Motor & Descending output
            motor_hops.append(next_H[:, 3])
            simulated_regional_hops.append(next_H)

        motor_stack = torch.stack(motor_hops, dim=1)  # [T, K_horizons + 1, d_model]

        # 4. TERMINAL ANCHOR PHYSICAL SUPERVISION (OPD) ACROSS 14 TICKS
        # Supervise LeWM multi-hop simulation against real physical quiet trajectory
        if teacher_quiet_states is not None and len(teacher_quiet_states) > 0:
            K_quiet = min(K_horizons, len(teacher_quiet_states))
            Z_quiet_target = self.encoders(teacher_quiet_states[:K_quiet]).detach()  # [K_quiet, 4, d_model]
            anchor_simulated = torch.stack([hop[-1] for hop in simulated_regional_hops[:K_quiet]], dim=0)
            loss_opd = F.mse_loss(anchor_simulated, Z_quiet_target)
        elif teacher_quiet_h is not None and teacher_quiet_h.shape[0] > 0:
            K_quiet = min(K_horizons, teacher_quiet_h.shape[0])
            Z_quiet_target = self.encoders(teacher_quiet_h[:K_quiet]).detach()  # [K_quiet, 4, d_model]
            anchor_simulated = torch.stack([hop[-1] for hop in simulated_regional_hops[:K_quiet]], dim=0)
            loss_opd = F.mse_loss(anchor_simulated, Z_quiet_target)
        else:
            loss_opd = torch.tensor(0.0, device=h_sequence.device)

        # 5. DAgger-INSPIRED STUDENT-ERROR-GUIDED PHYSICAL QUERY AGGREGATION
        # Training target: G_theta(E(S_query)) -> E(F_physical(S_query))
        if query_pairs is not None and len(query_pairs) > 0:
            q_losses = []
            for item in query_pairs:
                if len(item) == 2:
                    q_hist, z_target = item
                    if isinstance(q_hist, (list, tuple)):
                        q_list = [h.squeeze(0) if (h.dim() == 3 and h.shape[0] == 1) else h for h in q_hist]
                        z_pred = self.transition.forward_step(q_list, A_E, A_I)
                    elif isinstance(q_hist, torch.Tensor):
                        if q_hist.dim() == 2:
                            z_pred = self.transition.forward(q_hist.unsqueeze(0), A_E, A_I)
                        elif q_hist.dim() == 3 and q_hist.shape[0] > 1 and q_hist.shape[1] == 4:
                            z_pred = self.transition.forward_step(list(q_hist), A_E, A_I)
                        else:
                            z_pred = self.transition.forward(q_hist, A_E, A_I)
                    else:
                        z_pred = self.transition.forward(q_hist, A_E, A_I)

                    if z_target.dim() == 2 and z_pred.dim() == 3:
                        z_target = z_target.unsqueeze(0)
                    elif z_target.dim() == 3 and z_pred.dim() == 2:
                        z_pred = z_pred.unsqueeze(0)
                    q_losses.append(F.mse_loss(z_pred, z_target))
            loss_query = torch.stack(q_losses).mean() if q_losses else torch.tensor(0.0, device=h_sequence.device)
        else:
            loss_query = torch.tensor(0.0, device=h_sequence.device)

        # 6. MOTOR PREDICTIVE LOOKAHEAD CORRECTION & ANCHORED READOUT
        delta_z, attn_weights = self.delay_attn(motor_stack)

        if anchor_features is not None:
            z_anchor = anchor_features
        else:
            z_anchor = Z_star[:, 3]

        gamma = torch.tanh(self.raw_gamma)
        z_readout = z_anchor + gamma * delta_z

        # 7. REGIONAL SIGReg ANTI-COLLAPSE REGULARIZER
        # Apply SIGReg directly across all 4 anatomical regions + terminal lookahead draft
        loss_sigreg_regions = torch.stack([
            self.sigreg(Z_star[:, r, :]) for r in range(4)
        ]).mean()
        loss_sigreg_lookahead = self.sigreg(motor_stack[:, -1])
        loss_sigreg = 0.5 * (loss_sigreg_regions + loss_sigreg_lookahead)

        total_loss = self.lambda_obs * (loss_obs + loss_opd + loss_query) + self.lambda_sigreg * loss_sigreg

        w = attn_weights.detach()
        metrics = {
            "obs_mse": float(loss_obs.item()),
            "obs_opd": float(loss_opd.item()),
            "obs_query": float(loss_query.item()),
            "obs_sigreg": float(loss_sigreg.item()),
            "obs_sigreg_regions": float(loss_sigreg_regions.item()),
            "obs_total": float(total_loss.item()),
            "obs_gain_mean": float(gain.mean().item()),
            "obs_gamma": float(gamma.item()),
            "obs_delta_norm": float(delta_z.detach().norm(dim=-1).mean().item()),
            "attn_k0_immediate": float(w[:, 0].mean().item()),
            "attn_k1_fast": float(w[:, 1].mean().item()),
            "attn_k4_median": float(w[:, min(4, self.max_horizon)].mean().item()),
            "attn_k7_mid": float(w[:, min(7, self.max_horizon)].mean().item()),
            "attn_k14_recurrent": float(w[:, -1].mean().item()),
        }

        next_prior = cur_prior.detach()
        next_history = torch.stack(running_history[-4:], dim=0).detach() if len(running_history) > 0 else torch.empty(0, device=h_sequence.device)

        if return_history:
            return z_readout, total_loss, metrics, next_prior, next_history
        return z_readout, total_loss, metrics, next_prior
