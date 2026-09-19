"""Continuous Boltzmann Information Medium (CBIM) Field Model.

Key physical equations and architectural invariants:
1. State representation:
   Real-valued continuous memory field h in R^{B x L x d}.
   Energy defined by Frobenius norm: E(h) = 0.5 * ||h||_F^2.
2. Step 1: Localized input write with total energy budget clamp:
   u_t = Embedding(X_t)
   u_x = MLP_u(u_t)
   eta(x) = eta_max * exp(-dist_circ(x, mu_t)^2 / (2 * sigma_t^2))
   h_hat = (1 - eta) * h + eta * u_x
   h* = h_hat if ||h_hat||_F <= R else R * h_hat / ||h_hat||_F
   Guarantees E(h_t) <= R^2 / 2 for all t.
3. Step 2: Cayley Transport (Unitary Streaming):
   A = V_theta D_x where V_theta is a learnable dispersion spectrum and D_x is the skew-symmetric circular difference.
   Diagonalized in spatial Fourier domain along grid length L:
   U_{k, c} = (1 - i * mu_{k, c}) / (1 + i * mu_{k, c}) with mu_{k, c} = (dt / 2) * v_c * sin(2 * pi * k / L).
   Since |U_{k, c}| == 1.0 identically, ||h**||_F == ||h*||_F to machine precision.
   Zero numerical dissipation, zero explosive instability.
4. Step 3: State-dependent local conservative scattering:
   Adjacent spatial pairs (p, q) are transformed via:
   s = (p + q) / sqrt(2),  d = (p - q) / sqrt(2)
   theta = MLP_scatter([p, q])
   d' = R(theta) d via 2D Givens rotation blocks.
   p' = (s + d') / sqrt(2),  q' = (s - d') / sqrt(2)
   Strictly conserves:
   p' + q' == p + q (linear momentum / mass)
   ||p'||^2 + ||q'||^2 == ||p||^2 + ||q||^2 (quadratic energy)
   Alternates even-odd and odd-even brickwork pairing across L in parallel.
5. Step 4: State readout:
   logits_{t+1} = Readout(h_{t+1}) in R^{B x V}.
"""
import math
import os
import sys

# Prevent local scripts/ib_local from shadowing standard library 'types'
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
from torch import nn
from torch.nn import functional as F


class CayleyTransport(nn.Module):
    """Exact L2 norm-preserving unitary streaming along periodic spatial grid L."""
    def __init__(self, L: int = 64, d: int = 128, dt: float = 1.0):
        super().__init__()
        self.L = L
        self.d = d
        self.dt = dt

        # Learnable dispersion velocity spectrum across channels
        self.velocity = nn.Parameter(torch.randn(d) * 0.1)

        # Precompute spatial normalized angular frequencies
        n_freqs = L // 2 + 1
        k = torch.arange(n_freqs, dtype=torch.float32)
        omega = torch.sin(2.0 * math.pi * k / L)  # [n_freqs]
        omega[0] = 0
        if L % 2 == 0:
            omega[-1] = 0  # Real rFFT endpoints cannot carry a complex phase.
        self.register_buffer('omega', omega.unsqueeze(-1))  # [n_freqs, 1]

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Unitary streaming forward pass.

        h: [B, L, d]
        Returns: h_streamed [B, L, d] with ||h_streamed||_F == ||h||_F
        """
        B, L, d = h.shape
        # mu_{k, c} = (dt / 2) * v_c * omega_k
        mu = (self.dt * 0.5) * (self.omega * self.velocity.unsqueeze(0))  # [n_freqs, d]

        # Cayley multiplier: U = (1 - i * mu) / (1 + i * mu)
        # Real: (1 - mu^2) / (1 + mu^2), Imag: -2 * mu / (1 + mu^2)
        denom = 1.0 + mu.square()
        u_real = (1.0 - mu.square()) / denom
        u_imag = (-2.0 * mu) / denom
        U = torch.complex(u_real, u_imag).unsqueeze(0)  # [1, n_freqs, d]

        # rFFT along periodic memory grid L
        h_fft = torch.fft.rfft(h, dim=1)
        h_streamed_fft = h_fft * U
        h_streamed = torch.fft.irfft(h_streamed_fft, n=L, dim=1)

        diag = {
            'mean_abs_velocity': self.velocity.detach().abs().mean(),
            'max_abs_velocity': self.velocity.detach().abs().max(),
            'zero_velocity_modes_count': (self.velocity.detach().abs() < 0.01).sum(),
        }
        return h_streamed, diag


class ConservativeScattering(nn.Module):
    """State-dependent local conservative scattering preserving both sum and quadratic energy."""
    def __init__(self, L: int = 64, d: int = 128, n_layers: int = 2):
        super().__init__()
        assert d % 2 == 0, f"Channel dimension d={d} must be even for 2D Givens rotations"
        assert L % 2 == 0, f"Grid length L={L} must be even for brickwork pairing"
        self.L = L
        self.d = d
        self.n_layers = n_layers
        self.d_pairs = d // 2

        # State-dependent angle networks for each scattering layer
        self.angle_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * d, d),
                nn.SiLU(),
                nn.Linear(d, self.d_pairs),
            )
            for _ in range(n_layers)
        ])

        # Initialize angle nets to small weights
        for net in self.angle_nets:
            nn.init.normal_(net[-1].weight, std=1e-3)
            nn.init.zeros_(net[-1].bias)

    def scatter_pairs(self, p: torch.Tensor, q: torch.Tensor, angle_net: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply state-dependent rotation to adjacent pairs (p, q).

        p: [B, K, d]
        q: [B, K, d]
        """
        B, K, d = p.shape
        # Sum and difference coordinates
        s = (p + q) * (1.0 / math.sqrt(2.0))
        d_vec = (p - q) * (1.0 / math.sqrt(2.0))

        # Predict rotation angles from current local state [p, q]
        pq = torch.cat([p, q], dim=-1)  # [B, K, 2d]
        theta = angle_net(pq)            # [B, K, d_pairs]

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # 2D Givens rotation blocks: (d_{2c}, d_{2c+1})
        d_pairs = d_vec.view(B, K, self.d_pairs, 2)
        d_prime_0 = cos_t * d_pairs[..., 0] - sin_t * d_pairs[..., 1]
        d_prime_1 = sin_t * d_pairs[..., 0] + cos_t * d_pairs[..., 1]
        d_prime = torch.stack([d_prime_0, d_prime_1], dim=-1).view(B, K, d)

        # Reconstruct: p', q' strictly conserve p + q and ||p||^2 + ||q||^2
        p_prime = (s + d_prime) * (1.0 / math.sqrt(2.0))
        q_prime = (s - d_prime) * (1.0 / math.sqrt(2.0))

        return p_prime, q_prime

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Apply alternating brickwork scattering layers across memory grid L."""
        B, L, d = h.shape
        curr_h = h

        for layer_idx in range(self.n_layers):
            angle_net = self.angle_nets[layer_idx]
            # Alternate channel planes as well as spatial pairs. Undo after
            # each layer so the field retains its original channel coordinates.
            curr_h = torch.roll(curr_h, shifts=-layer_idx, dims=-1)
            if layer_idx % 2 == 0:
                # Even-Odd pairing: (0, 1), (2, 3), ..., (L-2, L-1)
                p = curr_h[:, 0::2, :]  # [B, L//2, d]
                q = curr_h[:, 1::2, :]  # [B, L//2, d]
                p_p, q_p = self.scatter_pairs(p, q, angle_net)

                # Reassemble into grid
                next_h = torch.empty_like(curr_h)
                next_h[:, 0::2, :] = p_p
                next_h[:, 1::2, :] = q_p
                curr_h = next_h
            else:
                # Odd-Even pairing: (1, 2), (3, 4), ..., (L-1, 0) with circular wrap
                p = curr_h[:, 1::2, :]                    # [B, L//2, d]
                q = torch.roll(curr_h[:, 0::2, :], shifts=-1, dims=1)  # [B, L//2, d]
                p_p, q_p = self.scatter_pairs(p, q, angle_net)

                next_h = torch.empty_like(curr_h)
                next_h[:, 1::2, :] = p_p
                next_h[:, 0::2, :] = torch.roll(q_p, shifts=1, dims=1)
                curr_h = next_h

            curr_h = torch.roll(curr_h, shifts=layer_idx, dims=-1)

        diag = {
            'scattering_layers_applied': self.n_layers,
        }
        return curr_h, diag


class LocalFieldWriter(nn.Module):
    """Local, state-conditioned input writer with total Frobenius energy budget clamp."""
    def __init__(self, vocab_size: int = 50257, L: int = 64, d: int = 128, R_budget: float = 10.0):
        super().__init__()
        self.L = L
        self.d = d
        self.R_budget = R_budget

        self.embedding = nn.Embedding(vocab_size, d)

        # Content generator
        self.content_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        # Address and intensity heads
        self.addr_head = nn.Linear(d, 1)    # predicts center mu in [0, L)
        self.width_head = nn.Linear(d, 1)   # predicts packet width sigma
        self.rate_head = nn.Linear(d, 1)    # predicts peak rate eta_max in [0, 1]
        self.state_encoder = nn.Sequential(
            nn.Linear(3 * d, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU())
        self.state_content = nn.Linear(d, d)
        self.state_rate = nn.Linear(d, 1)
        nn.init.normal_(self.state_content.weight, std=1e-3)
        nn.init.zeros_(self.state_content.bias)
        nn.init.normal_(self.state_rate.weight, std=1e-3)
        nn.init.zeros_(self.state_rate.bias)

        # Spatial grid indices for circular distance
        x_grid = torch.arange(L, dtype=torch.float32)
        self.register_buffer('x_grid', x_grid.view(1, L, 1))

    def forward(self, h: torch.Tensor, token_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Write token into local region of memory field h, then clamp total energy.

        h: [B, L, d]
        token_ids: [B]
        """
        B, L, d = h.shape
        u_emb = self.embedding(token_ids)  # [B, d]
        u_content = self.content_mlp(u_emb).unsqueeze(1)  # [B, 1, d]

        # Predict write center mu in [0, L), width sigma, and rate eta_max in [0, 1]
        mu = self.L * torch.sigmoid(self.addr_head(u_emb)).view(B, 1, 1)
        sigma = 1.0 + 2.0 * torch.sigmoid(self.width_head(u_emb)).view(B, 1, 1)
        eta_max = torch.sigmoid(self.rate_head(u_emb)).view(B, 1, 1)

        # Circular spatial distance: min(|x - mu|, L - |x - mu|)
        dx = (self.x_grid - mu).abs()
        dist_circ = torch.minimum(dx, self.L - dx)
        # Local write rate profile eta(x)
        envelope = torch.exp(-0.5 * (dist_circ / sigma).square())
        neighborhood = (torch.roll(h, 1, 1) + h + torch.roll(h, -1, 1)) / 3
        context = self.state_encoder(torch.cat(
            (u_emb[:, None].expand(-1, L, -1), h, neighborhood), dim=-1))
        eta = envelope * torch.sigmoid(
            self.rate_head(u_emb)[:, None] + self.state_rate(context))
        u_content = u_content + self.state_content(context)

        # Unconstrained write candidate
        h_hat = (1.0 - eta) * h + eta * u_content

        # Energy budget clamping (SPEC §1)
        norm_hat = h_hat.norm(dim=(-2, -1), keepdim=True)  # [B, 1, 1]
        clamping_scale = torch.clamp(self.R_budget / norm_hat.clamp_min(1e-8), max=1.0)
        h_star = h_hat * clamping_scale

        clamped = (clamping_scale.detach() < 1.0).flatten()
        diag = {
            'clamped': clamped,
            'norm_before_clamp': norm_hat.detach().mean(),
            'norm_after_clamp': h_star.detach().norm(dim=(-2, -1)).mean(),
            'peak_write_rate': eta.detach().amax(dim=1).mean(),
        }
        return h_star, diag


class CBIMFieldModel(nn.Module):
    """Continuous Boltzmann Information Medium Language Model."""
    def __init__(
        self,
        vocab_size: int = 50257,
        L: int = 64,
        d: int = 128,
        R_budget: float = 10.0,
        dt_stream: float = 1.0,
        n_scatter_layers: int = 2,
        n_queries: int = 4,
        readout_heads: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.L = L
        self.d = d
        self.R_budget = R_budget
        if d % readout_heads or n_queries < 1 or R_budget <= 0:
            raise ValueError('Invalid readout dimensions or energy budget')
        self.architecture_version = 2
        self.n_queries = n_queries
        self.readout_heads = readout_heads

        self.writer = LocalFieldWriter(vocab_size=vocab_size, L=L, d=d, R_budget=R_budget)
        self.transport = CayleyTransport(L=L, d=d, dt=dt_stream)
        self.scattering = ConservativeScattering(L=L, d=d, n_layers=n_scatter_layers)

        # Readout: spatial attention pool over memory grid -> vocabulary logits
        self.readout_query = nn.Parameter(torch.randn(1, n_queries, d))
        self.readout_norm = nn.LayerNorm(d)
        self.readout_key = nn.Linear(d, d)
        self.readout_value = nn.Linear(d, d)
        self.readout_position = nn.Parameter(torch.randn(1, L, d) * 0.02)
        self.readout_merge = nn.Linear(n_queries * d, d)
        self.readout_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.decoder = nn.Linear(d, vocab_size, bias=True)
        # Tie decoder weight to writer embedding
        self.decoder.weight = self.writer.embedding.weight

    def read_state(self, h: torch.Tensor) -> torch.Tensor:
        """State-only multi-query retrieval; no token-to-logit bypass."""
        B, L, d = h.shape
        H = self.readout_heads
        encoded = self.readout_norm(h)
        q = self.readout_query.expand(B, -1, -1)
        k = self.readout_key(encoded + self.readout_position)
        v = self.readout_value(encoded)
        split = lambda a: a.reshape(B, -1, H, d // H).transpose(1, 2)
        att = F.scaled_dot_product_attention(split(q), split(k), split(v), dropout_p=0.)
        merged = att.transpose(1, 2).reshape(B, self.n_queries * d)
        return self.readout_mlp(self.readout_merge(merged))

    def step(self, h: torch.Tensor, token_id: torch.Tensor, *,
             disable_scattering: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """One causal autoregressive token update.

        h: [B, L, d]
        token_id: [B]
        Returns:
            logits: [B, vocab_size]
            h_next: [B, L, d]
            diag: dict
        """
        # Step 1: Local write with energy budget clamp
        h_star, diag_w = self.writer(h, token_id)

        # Step 2: Cayley Unitary Streaming (norm strictly preserved)
        h_streamed, diag_t = self.transport(h_star)

        # Step 3: Local Conservative Scattering (linear sum and energy strictly preserved)
        if disable_scattering:
            h_next, diag_s = h_streamed, {'scattering_layers_applied': 0}
        else:
            h_next, diag_s = self.scattering(h_streamed)

        # Step 4: State readout from evolved field
        B, L, d = h_next.shape
        readout_feat = self.read_state(h_next)
        logits = self.decoder(readout_feat)

        diag = {
            'write': diag_w,
            'transport': diag_t,
            'scattering': diag_s,
            'energy': 0.5 * h_next.detach().square().sum(dim=(-2, -1)).mean(),
        }
        return logits, h_next, diag

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor,
                initial_h: torch.Tensor | None = None, *,
                disable_scattering: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Autoregressive sequence forward pass.

        input_ids: [B, T]
        targets: [B, T]
        Returns:
            loss: scalar Cross Entropy
            h_final: [B, L, d]
            diagnostics: dict
        """
        B, T = input_ids.shape
        device = input_ids.device

        if initial_h is None:
            curr_h = torch.zeros(B, self.L, self.d, device=device)
        else:
            curr_h = initial_h

        all_features = []
        energies = []
        clamp_counts = []
        initial_energy = .5 * curr_h.detach().square().sum((-2, -1)).mean()

        for t in range(T):
            token_t = input_ids[:, t]
            # Decode all token features together after recurrent evolution.
            curr_h, diag_w = self.writer(curr_h, token_t)
            curr_h, _ = self.transport(curr_h)
            if not disable_scattering:
                curr_h, _ = self.scattering(curr_h)
            all_features.append(self.read_state(curr_h))
            diag_t = {'energy': .5 * curr_h.detach().square().sum((-2, -1)).mean()}
            energies.append(diag_t['energy'])
            clamp_counts.append(diag_w['clamped'])

        logits_seq = self.decoder(torch.stack(all_features, dim=1))
        loss = F.cross_entropy(logits_seq.view(B * T, self.vocab_size), targets.view(B * T))

        diagnostics = {
            'initial_energy': initial_energy,
            'final_energy': energies[-1],
            'mean_energy': torch.stack(energies).mean(),
            'clamp_trigger_rate': torch.stack(clamp_counts).float().mean(),
        }
        return loss, curr_h, diagnostics
