"""Variational Rolling Stream Learning Mechanism.

Implements continuous lifelong learning over infinite token streams using:
1. Rolling overlapping windows (Window W=128, Stride S=64): Context A -> Target B.
2. Variational Free Energy (VFE) objective:
     F_B(q) = - E_q[log p(B | z, A)] + D_KL(q(z) || p_A^-(z))
3. Inner assimilation loop optimizing posterior q(z) until plateau (|Delta F| < epsilon)
   or budget limit (K_max), while holding prior p_A^-(z) strictly detached.
4. Single-counting evidence: outer update on slow model parameters occurs exactly once
   per window under the converged posterior q_B^*(z).
5. Continuous temporal state transition: posterior q_B^*(z) decays via Ornstein-Uhlenbeck
   retention dynamics into the prior for the subsequent window p_B^-(z).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VariationalGaussianBelief:
    """Represents a diagonal Gaussian belief distribution in latent space.

    Parameters:
        mean: Tensor of shape (..., latent_dim)
        log_std: Tensor of shape (..., latent_dim)
    """
    mean: torch.Tensor
    log_std: torch.Tensor

    @property
    def std(self) -> torch.Tensor:
        return torch.exp(self.log_std)

    @property
    def var(self) -> torch.Tensor:
        return torch.exp(2.0 * self.log_std)

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """Sample from q(z) via the reparameterization trick.

        Returns:
            Tensor of shape (num_samples, ..., latent_dim) if num_samples > 1
            else (..., latent_dim)
        """
        eps = torch.randn_like(self.mean)
        if num_samples == 1:
            return self.mean + self.std * eps
        samples = []
        for _ in range(num_samples):
            eps_i = torch.randn_like(self.mean)
            samples.append(self.mean + self.std * eps_i)
        return torch.stack(samples, dim=0)

    def kl_divergence(self, prior: "VariationalGaussianBelief") -> torch.Tensor:
        """Compute exact analytical KL divergence D_KL(q || p) between two diagonal Gaussians.

        Formula:
            D_KL(q || p) = 0.5 * sum( (var_q + (mu_q - mu_p)^2) / var_p - 1 + 2 * (log_std_p - log_std_q) )

        Returns:
            Scalar tensor with total KL divergence summed over latent dimensions.
        """
        var_q = self.var
        var_p = prior.var
        mu_diff_sq = (self.mean - prior.mean).pow(2)

        term1 = (var_q + mu_diff_sq) / (var_p + 1e-8)
        term2 = 2.0 * (prior.log_std - self.log_std)
        kl_per_dim = 0.5 * (term1 - 1.0 + term2)
        return kl_per_dim.sum()

    def transition(
        self,
        retention: float = 0.90,
        base_mean: float = 0.0,
        base_log_std: float = 0.0,
    ) -> "VariationalGaussianBelief":
        """Compute continuous temporal Ornstein-Uhlenbeck state transition to form next prior.

        Carries over posterior belief with exponential retention rho, relaxing toward
        resting baseline (base_mean, exp(base_log_std)) to prevent unbounded overconfidence.

        Formulas:
            mu_next = rho * mu_q + (1 - rho) * mu_0
            var_next = rho^2 * var_q + (1 - rho^2) * var_0
            log_std_next = 0.5 * log(var_next)
        """
        rho = float(retention)
        rho_sq = rho * rho
        base_var = math.exp(2.0 * base_log_std)

        with torch.no_grad():
            mu_next = rho * self.mean.detach() + (1.0 - rho) * base_mean
            var_next = rho_sq * self.var.detach() + (1.0 - rho_sq) * base_var
            log_std_next = 0.5 * torch.log(var_next.clamp_min(1e-8))

        return VariationalGaussianBelief(mean=mu_next, log_std=log_std_next)

    def detach(self) -> "VariationalGaussianBelief":
        """Return a detached copy of this belief."""
        return VariationalGaussianBelief(
            mean=self.mean.detach(),
            log_std=self.log_std.detach(),
        )

    def clone(self) -> "VariationalGaussianBelief":
        """Return a cloned copy of this belief."""
        return VariationalGaussianBelief(
            mean=self.mean.clone(),
            log_std=self.log_std.clone(),
        )

    @classmethod
    def standard_normal(
        cls, latent_dim: int, device: torch.device | str = "cpu"
    ) -> "VariationalGaussianBelief":
        """Create standard normal resting prior N(0, I)."""
        mean = torch.zeros(latent_dim, device=device)
        log_std = torch.zeros(latent_dim, device=device)
        return cls(mean=mean, log_std=log_std)


class VariationalBeliefModulator(nn.Module):
    """Modulates neural representations and output heads using continuous belief z.

    Maps latent z into:
    1. Feature-wise affine scale/shift (FiLM) for hidden representations.
    2. Low-rank projection bias for logits readout.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Affine FiLM projection for hidden layers
        self.film_proj = nn.Linear(latent_dim, hidden_dim * 2)

        # Low-rank readout bias: z -> bottleneck -> vocab_size
        bottleneck = min(latent_dim * 2, 128)
        self.readout_bottleneck = nn.Linear(latent_dim, bottleneck)
        self.readout_head = nn.Linear(bottleneck, vocab_size, bias=False)

        # Initialize with standard variance so z receives clean gradients
        nn.init.normal_(self.film_proj.weight, std=0.02)
        nn.init.zeros_(self.film_proj.bias)
        nn.init.normal_(self.readout_bottleneck.weight, std=0.02)
        nn.init.zeros_(self.readout_bottleneck.bias)
        nn.init.normal_(self.readout_head.weight, std=0.02)

    def modulate_features(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation: h_mod = h * (1 + tanh(gamma)) + beta."""
        # z: (latent_dim,) or (batch, latent_dim)
        # h: (batch, seq_len, hidden_dim)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        film = self.film_proj(z)  # (batch, hidden_dim * 2)
        gamma, beta = film.chunk(2, dim=-1)
        # Expand for sequence length
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return h * (1.0 + torch.tanh(gamma)) + beta

    def compute_logits_bias(self, z: torch.Tensor) -> torch.Tensor:
        """Compute additive vocabulary bias from belief z."""
        if z.ndim == 1:
            z = z.unsqueeze(0)
        h_mid = F.gelu(self.readout_bottleneck(z))
        bias = self.readout_head(h_mid)  # (batch, vocab_size)
        return bias.unsqueeze(1)  # (batch, 1, vocab_size)


class RollingStreamTransformer(nn.Module):
    """Compact causal transformer designed for continuous rolling stream learning."""

    def __init__(
        self,
        vocab_size: int = 50257,
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_len: int = 128,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_len = max_len
        self.latent_dim = latent_dim

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.tok_emb.weight

        self.modulator = VariationalBeliefModulator(
            latent_dim=latent_dim, hidden_dim=dim, vocab_size=vocab_size
        )

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((max_len, max_len), float("-inf")), diagonal=1),
        )

        self._init_weights()

    def _init_weights(self):
        """Standard small-variance initialization for stable causal language modeling."""
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        for p in self.layers.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=0.02)

    def forward(
        self, tokens: torch.Tensor, z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass conditioned on tokens and optional latent belief z.

        Args:
            tokens: (batch, seq_len)
            z: optional latent belief tensor of shape (latent_dim,) or (batch, latent_dim)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        b, t = tokens.shape
        device = tokens.device
        pos = torch.arange(t, device=device)

        h = self.tok_emb(tokens) + self.pos_emb(pos)

        if z is not None:
            h = self.modulator.modulate_features(h, z)

        mask = self.causal_mask[:t, :t]
        h = self.layers(h, mask=mask, is_causal=True)
        h = self.norm(h)

        logits = self.head(h)

        if z is not None:
            logits = logits + self.modulator.compute_logits_bias(z)

        return logits


@dataclass
class AssimilationStepResult:
    """Detailed telemetry for an inner assimilation iteration."""
    step: int
    free_energy: float
    nll_target: float
    kl_divergence: float
    delta_fe: float
    plateau_triggered: bool


@dataclass
class WindowAssimilationReport:
    """Summary report for an assimilated stream window."""
    window_idx: int
    token_start: int
    token_end: int
    prequential_nll: float
    plateau_nll: float
    adaptation_gain: float
    final_kl: float
    final_free_energy: float
    inner_steps_taken: int
    plateau_reached: bool
    wall_time_ms: float


class VariationalRollingStreamLearner:
    """Orchestrates rolling stream lifelong learning with variational free energy assimilation.

    Maintains:
    - Overlapping rolling windows (Window W=128, Stride S=64).
    - Persistent variational belief transitioned across window boundaries via Ornstein-Uhlenbeck.
    - Strictly frozen prior during inner posterior optimization.
    - Plateau detection for inner loop early stopping.
    - Single-counting outer update on model structural parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        latent_dim: int = 64,
        window_size: int = 128,
        stride: int = 64,
        inner_lr: float = 0.05,
        inner_max_steps: int = 15,
        plateau_delta: float = 1e-3,
        plateau_patience: int = 2,
        kl_weight: float = 0.1,
        retention: float = 0.90,
        outer_lr: float = 1e-4,
        device: torch.device | str = "cpu",
    ):
        self.model = model.to(device)
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.stride = stride
        self.prompt_size = window_size - stride
        self.target_size = stride

        self.inner_lr = inner_lr
        self.inner_max_steps = inner_max_steps
        self.plateau_delta = plateau_delta
        self.plateau_patience = plateau_patience
        self.kl_weight = kl_weight
        self.retention = retention
        self.device = torch.device(device)

        # Initialize current prior as resting standard normal
        self.current_prior = VariationalGaussianBelief.standard_normal(
            latent_dim=latent_dim, device=self.device
        )

        # Outer optimizer for slow structural parameters
        self.outer_optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=outer_lr, weight_decay=1e-2
        )

        # Telemetry
        self.total_windows = 0
        self.total_tokens_seen = 0
        self.history: list[WindowAssimilationReport] = []

    def compute_target_nll(
        self,
        tokens: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute cross-entropy NLL strictly over the target segment B (indices prompt_size..window_size).

        Tokens shape: (1, seq_len)
        Prompt: tokens[0 : prompt_size]
        Target: tokens[prompt_size : window_size]
        """
        logits = self.model(tokens, z=z)  # (1, seq_len, vocab_size)
        # Position t predicts token t+1.
        # Target tokens are at indices [prompt_size .. seq_len - 1].
        # Corresponding logits are at indices [prompt_size - 1 .. seq_len - 2].
        p_start = self.prompt_size - 1
        p_end = tokens.shape[1] - 1

        pred_logits = logits[:, p_start:p_end, :].contiguous()  # (1, target_len, vocab_size)
        target_tokens = tokens[:, self.prompt_size:].contiguous()  # (1, target_len)

        nll = F.cross_entropy(
            pred_logits.view(-1, pred_logits.size(-1)),
            target_tokens.view(-1),
            reduction="mean",
        )
        return nll

    def evaluate_prequential(self, tokens: torch.Tensor) -> float:
        """Evaluate prequential NLL on target continuation using current prior mean (unbiased score)."""
        self.model.eval()
        with torch.no_grad():
            prior_mean = self.current_prior.mean
            nll = self.compute_target_nll(tokens, z=prior_mean)
        return float(nll.item())

    def inner_assimilation_loop(
        self,
        tokens: torch.Tensor,
        prior: VariationalGaussianBelief,
    ) -> Tuple[VariationalGaussianBelief, list[AssimilationStepResult]]:
        """Run inner loop optimizing posterior q(z) until plateau or budget exhaustion.

        The prior is strictly detached/frozen throughout.
        """
        # Strictly detach and freeze prior
        frozen_prior = prior.detach()

        # Initialize posterior parameters from prior
        mu_q = frozen_prior.mean.clone().detach().requires_grad_(True)
        log_std_q = frozen_prior.log_std.clone().detach().requires_grad_(True)

        # Fast inner optimizer for variational parameters
        inner_opt = torch.optim.Adam([mu_q, log_std_q], lr=self.inner_lr)

        step_history: list[AssimilationStepResult] = []
        prev_fe: Optional[float] = None
        consecutive_plateaus = 0

        # Model is in eval mode during posterior-only search
        self.model.eval()

        for step in range(1, self.inner_max_steps + 1):
            inner_opt.zero_grad()

            current_q = VariationalGaussianBelief(mean=mu_q, log_std=log_std_q)

            # Sample latent z via reparameterization trick
            z_sample = current_q.sample(num_samples=1)

            # 1. Prediction likelihood term (Target NLL)
            nll_target = self.compute_target_nll(tokens, z=z_sample)

            # 2. Complexity / Memory constraint term (KL divergence against frozen prior)
            kl_val = current_q.kl_divergence(frozen_prior)

            # 3. Variational Free Energy objective
            # Scaled KL weight: kl_weight * KL / target_size
            fe = nll_target + (self.kl_weight / self.target_size) * kl_val

            fe.backward()
            inner_opt.step()

            # Evaluate expected free energy at the updated posterior mean (free of MC jitter)
            with torch.no_grad():
                updated_q = VariationalGaussianBelief(mean=mu_q, log_std=log_std_q)
                eval_nll = self.compute_target_nll(tokens, z=mu_q)
                eval_kl = updated_q.kl_divergence(frozen_prior)
                eval_fe = float((eval_nll + (self.kl_weight / self.target_size) * eval_kl).item())

            delta_fe = abs(eval_fe - prev_fe) if prev_fe is not None else float("inf")

            plateau_triggered = False
            if delta_fe < self.plateau_delta:
                consecutive_plateaus += 1
                if consecutive_plateaus >= self.plateau_patience:
                    plateau_triggered = True
            else:
                consecutive_plateaus = 0

            step_history.append(
                AssimilationStepResult(
                    step=step,
                    free_energy=eval_fe,
                    nll_target=float(eval_nll.item()),
                    kl_divergence=float(eval_kl.item()),
                    delta_fe=delta_fe,
                    plateau_triggered=plateau_triggered,
                )
            )

            prev_fe = eval_fe

            if plateau_triggered:
                break

        converged_q = VariationalGaussianBelief(
            mean=mu_q.detach(),
            log_std=log_std_q.detach(),
        )
        return converged_q, step_history

    def outer_evidence_update(
        self,
        tokens: torch.Tensor,
        converged_q: VariationalGaussianBelief,
        prior: VariationalGaussianBelief,
    ) -> float:
        """Single-counting outer update on slow model parameters using converged posterior.

        Computes gradient of F_B(q*) w.r.t model weights theta, steps optimizer once,
        and clears gradients.
        """
        self.model.train()
        self.outer_optimizer.zero_grad()

        # Evaluate at converged posterior mean
        z_star = converged_q.mean
        nll_target = self.compute_target_nll(tokens, z=z_star)

        # Structural update on weights
        nll_target.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.outer_optimizer.step()

        return float(nll_target.item())

    def step(self, window_tokens: torch.Tensor) -> WindowAssimilationReport:
        """Process a single rolling window: prequential eval -> inner assimilate -> outer update -> transition.

        Args:
            window_tokens: 1D or 2D tensor containing window_size tokens.

        Returns:
            WindowAssimilationReport with comprehensive diagnostic metrics.
        """
        import time

        t_start = time.perf_counter()

        if window_tokens.ndim == 1:
            tokens = window_tokens.unsqueeze(0).to(self.device)
        else:
            tokens = window_tokens.to(self.device)

        assert tokens.shape[1] == self.window_size, (
            f"Expected window length {self.window_size}, got {tokens.shape[1]}"
        )

        token_start = self.total_tokens_seen
        token_end = token_start + self.window_size

        # 1. Prequential Evaluation (before any adaptation to target B)
        prequential_nll = self.evaluate_prequential(tokens)

        # 2. Inner Assimilation Loop (digest target B into posterior q, holding prior frozen)
        prior_for_window = self.current_prior
        converged_q, inner_steps = self.inner_assimilation_loop(
            tokens=tokens, prior=prior_for_window
        )

        last_step = inner_steps[-1]
        plateau_nll = last_step.nll_target
        final_kl = last_step.kl_divergence
        final_fe = last_step.free_energy
        plateau_reached = last_step.plateau_triggered
        adaptation_gain = prequential_nll - plateau_nll

        # 3. Outer Evidence Update (single-counting evidence into slow weights)
        self.outer_evidence_update(
            tokens=tokens, converged_q=converged_q, prior=prior_for_window
        )

        # 4. Continuous Temporal State Transition (posterior q -> next prior p)
        next_prior = converged_q.transition(
            retention=self.retention, base_mean=0.0, base_log_std=0.0
        )
        self.current_prior = next_prior

        # 5. Advance Stream Position
        self.total_tokens_seen += self.stride
        self.total_windows += 1

        t_end = time.perf_counter()
        wall_time_ms = (t_end - t_start) * 1000.0

        report = WindowAssimilationReport(
            window_idx=self.total_windows,
            token_start=token_start,
            token_end=token_end,
            prequential_nll=prequential_nll,
            plateau_nll=plateau_nll,
            adaptation_gain=adaptation_gain,
            final_kl=final_kl,
            final_free_energy=final_fe,
            inner_steps_taken=len(inner_steps),
            plateau_reached=plateau_reached,
            wall_time_ms=wall_time_ms,
        )

        self.history.append(report)
        return report
