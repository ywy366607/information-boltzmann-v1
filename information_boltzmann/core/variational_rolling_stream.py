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
        retention: float | torch.Tensor = 0.90,
        base_mean: float | torch.Tensor = 0.0,
        base_log_std: float | torch.Tensor = 0.0,
    ) -> "VariationalGaussianBelief":
        """Compute continuous temporal Ornstein-Uhlenbeck state transition to form next prior.

        Carries over posterior belief with exponential retention rho, relaxing toward
        resting baseline (base_mean, exp(base_log_std)) to prevent unbounded overconfidence.
        Supports scalar retention or multi-scale vector retention across latent dimensions.

        Formulas:
            mu_next = rho * mu_q + (1 - rho) * mu_0
            var_next = rho^2 * var_q + (1 - rho^2) * var_0
            log_std_next = 0.5 * log(var_next)
        """
        if isinstance(retention, torch.Tensor):
            rho = retention.detach().to(self.mean.device)
            rho_sq = rho * rho
        else:
            rho = float(retention)
            rho_sq = rho * rho

        if isinstance(base_mean, torch.Tensor):
            b_mean = base_mean.detach().to(self.mean.device)
        else:
            b_mean = float(base_mean)

        if isinstance(base_log_std, torch.Tensor):
            base_var = torch.exp(2.0 * base_log_std.detach().to(self.mean.device))
        else:
            base_var = math.exp(2.0 * float(base_log_std))

        with torch.no_grad():
            mu_next = rho * self.mean.detach() + (1.0 - rho) * b_mean
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

        # GDN-style channel-wise learnable forget gate parameters
        # 1. Per-channel learnable base logit bias
        clamped_r = 0.90
        logit_center = math.log(clamped_r / (1.0 - clamped_r))
        spread = torch.linspace(-1.0, 1.0, latent_dim)
        self.channel_retention_bias = nn.Parameter(logit_center + spread)

        # 2. Per-channel learnable surprise sensitivity
        self.channel_sensitivity = nn.Parameter(torch.ones(latent_dim))

        # 3. Content projection: hidden features -> latent channel gate offsets
        self.content_gate_proj = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.content_gate_proj.weight)
        nn.init.zeros_(self.content_gate_proj.bias)

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

    def compute_channel_retention(
        self,
        log_precision_ratio: float = 0.0,
        content_feat: Optional[torch.Tensor] = None,
        retention_min: float = 0.10,
        retention_max: float = 0.98,
    ) -> torch.Tensor:
        """Compute GDN-style channel-wise, content-dependent and surprise-modulated retention vector.

        Formulation:
            logits_d = channel_retention_bias_d + channel_sensitivity_d * log_precision_ratio + content_gate_proj(h)_d
            rho_d = clamp(sigmoid(logits_d), retention_min, retention_max)
        """
        logits = self.channel_retention_bias + self.channel_sensitivity * float(log_precision_ratio)
        if content_feat is not None:
            if content_feat.ndim == 3:
                c_vec = content_feat.mean(dim=(0, 1))
            elif content_feat.ndim == 2:
                c_vec = content_feat.mean(dim=0)
            else:
                c_vec = content_feat
            logits = logits + self.content_gate_proj(c_vec)
        return torch.sigmoid(logits).clamp(min=retention_min, max=retention_max)


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
        self,
        tokens: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass conditioned on tokens and optional latent belief z.

        Args:
            tokens: (batch, seq_len)
            z: optional latent belief tensor of shape (latent_dim,) or (batch, latent_dim)
            return_features: whether to return normalized hidden features alongside logits

        Returns:
            logits: (batch, seq_len, vocab_size) if not return_features
            (logits, h): tuple of logits and normalized representations if return_features
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

        if return_features:
            return logits, h
        return logits


def compute_friston_adaptive_retention(
    base_retention: float | torch.Tensor,
    prediction_error: float,
    baseline_error: float,
    error_sensitivity: float = 1.0,
    retention_min: float = 0.10,
    retention_max: float = 0.98,
) -> Tuple[torch.Tensor | float, float]:
    """Compute Friston precision-weighted adaptive retention.

    In Active Inference and predictive coding (Friston 2008, 2010),
    optimal belief precision Pi* is inversely proportional to squared prediction error:
        Pi* = 1 / error^2

    When encountering high surprise / unexpected uncertainty (prediction_error > baseline_error),
    precision collapses (Pi_rel < 1.0), triggering neuromodulatory reset by lowering retention rho.
    Conversely, during consistent, low-error streams (prediction_error < baseline_error),
    precision is high (Pi_rel > 1.0), consolidating prior retention.

    Parameters:
        base_retention: Nominal retention scalar or multi-scale tensor across latent dims.
        prediction_error: Current window's prequential prediction error (e.g. NLL).
        baseline_error: Running expected prediction error baseline.
        error_sensitivity: Exponent scale kappa regulating responsiveness to surprise shocks.
        retention_min: Minimal retention floor (prevents complete numerical collapse).
        retention_max: Maximum retention ceiling (preserves headroom for future learning).

    Returns:
        (effective_retention, precision_ratio)
    """
    eps = 1e-4
    curr_err = max(float(prediction_error), eps)
    base_err = max(float(baseline_error), eps)

    # Relative precision ratio Pi_rel = (base_err / curr_err)^2
    precision_ratio = (base_err / curr_err) ** 2
    log_precision_ratio = math.log(max(precision_ratio, 1e-8))

    if isinstance(base_retention, torch.Tensor):
        clamped_base = base_retention.clamp(1e-4, 1.0 - 1e-4)
        logit_base = torch.log(clamped_base / (1.0 - clamped_base))
        modulated_logit = logit_base + error_sensitivity * log_precision_ratio
        effective_rho = torch.sigmoid(modulated_logit).clamp(min=retention_min, max=retention_max)
    else:
        clamped_base = max(1e-4, min(1.0 - 1e-4, float(base_retention)))
        logit_base = math.log(clamped_base / (1.0 - clamped_base))
        modulated_logit = logit_base + error_sensitivity * log_precision_ratio
        effective_rho = 1.0 / (1.0 + math.exp(-modulated_logit))
        effective_rho = max(retention_min, min(retention_max, effective_rho))

    return effective_rho, precision_ratio


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
    retention_mean: float = 0.90
    precision_ratio: float = 1.0


class VariationalRollingStreamLearner:
    """Orchestrates rolling stream lifelong learning with variational free energy assimilation.

    Maintains:
    - Overlapping rolling windows (Window W=128, Stride S=64).
    - Persistent variational belief transitioned across window boundaries via Ornstein-Uhlenbeck.
    - Strictly frozen prior during inner posterior optimization.
    - Plateau detection for inner loop early stopping.
    - Single-counting outer update on model structural parameters.
    - Friston-style precision-adaptive retention based on prediction error surprisal.
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
        adaptive_retention: bool = True,
        retention_min: float = 0.10,
        retention_max: float = 0.98,
        error_sensitivity: float = 1.0,
        multi_scale_retention: bool = True,
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
        self.adaptive_retention = adaptive_retention
        self.retention_min = retention_min
        self.retention_max = retention_max
        self.error_sensitivity = error_sensitivity
        self.multi_scale_retention = multi_scale_retention
        self.device = torch.device(device)

        # Baseline error tracker for Friston precision dynamics
        self.running_error_baseline: Optional[float] = None
        self.error_ema_beta = 0.85

        if self.multi_scale_retention:
            clamped_r = max(1e-3, min(1.0 - 1e-3, retention))
            logit_base = math.log(clamped_r / (1.0 - clamped_r))
            spread = torch.linspace(-1.0, 1.0, latent_dim, device=self.device)
            self.base_retention = torch.sigmoid(logit_base + spread)
        else:
            self.base_retention = float(retention)

        # Initialize current prior as resting standard normal
        self.current_prior = VariationalGaussianBelief.standard_normal(
            latent_dim=latent_dim, device=self.device
        )

        # Outer optimizer for slow structural parameters
        self.outer_optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=outer_lr, weight_decay=1e-2
        )

        # Previous state tracking for empirical Bayes GDN channel gate learning
        self.previous_posterior: Optional[VariationalGaussianBelief] = None
        self.previous_content_feat: Optional[torch.Tensor] = None
        self.previous_log_prec_ratio: float = 0.0

        # Telemetry
        self.total_windows = 0
        self.total_tokens_seen = 0
        self.history: list[WindowAssimilationReport] = []

    def compute_target_nll(
        self,
        tokens: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute cross-entropy NLL strictly over the target segment B (indices prompt_size..window_size).

        Tokens shape: (1, seq_len)
        Prompt: tokens[0 : prompt_size]
        Target: tokens[prompt_size : window_size]
        """
        if return_features:
            out = self.model(tokens, z=z, return_features=True)
            if isinstance(out, tuple):
                logits, h = out
                feat = h[:, self.prompt_size:].mean(dim=1).squeeze(0)
            else:
                logits = out
                feat = None
        else:
            logits = self.model(tokens, z=z)
            feat = None

        p_start = self.prompt_size - 1
        p_end = tokens.shape[1] - 1

        pred_logits = logits[:, p_start:p_end, :].contiguous()  # (1, target_len, vocab_size)
        target_tokens = tokens[:, self.prompt_size:].contiguous()  # (1, target_len)

        nll = F.cross_entropy(
            pred_logits.view(-1, pred_logits.size(-1)),
            target_tokens.view(-1),
            reduction="mean",
        )
        if return_features:
            return nll, feat
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

        # Explicit no-inner-loop mode: carry the predictive prior directly into
        # the outer update and temporal transition. This is different from a
        # one-step inner update and is useful for measuring sample efficiency
        # supplied by the persistent state alone.
        if self.inner_max_steps <= 0:
            with torch.no_grad():
                nll_target = self.compute_target_nll(tokens, z=frozen_prior.mean)
            return (
                frozen_prior,
                [
                    AssimilationStepResult(
                        step=0,
                        free_energy=float(nll_target.item()),
                        nll_target=float(nll_target.item()),
                        kl_divergence=0.0,
                        delta_fe=0.0,
                        plateau_triggered=True,
                    )
                ],
            )

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
        prev_posterior: Optional[VariationalGaussianBelief] = None,
        prev_content_feat: Optional[torch.Tensor] = None,
        log_precision_ratio: float = 0.0,
    ) -> Tuple[float, Optional[torch.Tensor]]:
        """Single-counting outer update on slow model parameters using converged posterior.

        Computes gradient of F_B(q*) w.r.t model weights theta, steps optimizer once,
        and clears gradients. Also optimizes GDN-style channel-wise forget gate parameters.
        """
        self.model.train()
        self.outer_optimizer.zero_grad()

        # Evaluate at converged posterior mean
        z_star = converged_q.mean
        nll_out = self.compute_target_nll(tokens, z=z_star, return_features=True)
        if isinstance(nll_out, tuple):
            nll_target, feat = nll_out
        else:
            nll_target, feat = nll_out, None

        total_loss = nll_target

        # Empirical Bayes alignment for channel-wise GDN forget gate:
        if (
            self.adaptive_retention
            and self.multi_scale_retention
            and prev_posterior is not None
            and hasattr(self.model, "modulator")
            and hasattr(self.model.modulator, "compute_channel_retention")
        ):
            rho_vec = self.model.modulator.compute_channel_retention(
                log_precision_ratio=log_precision_ratio,
                content_feat=prev_content_feat,
                retention_min=self.retention_min,
                retention_max=self.retention_max,
            )
            rho_sq = rho_vec * rho_vec
            mu_pred = rho_vec * prev_posterior.mean.detach()
            var_pred = rho_sq * prev_posterior.var.detach() + (1.0 - rho_sq) * 1.0

            var_q = converged_q.var.detach()
            mu_q = converged_q.mean.detach()
            kl_gate = 0.5 * torch.sum(
                (var_q + (mu_q - mu_pred).pow(2)) / (var_pred + 1e-8)
                - 1.0
                + torch.log(var_pred.clamp_min(1e-8) / var_q.clamp_min(1e-8))
            )
            total_loss = total_loss + (0.05 / self.latent_dim) * kl_gate

        # Structural update on weights and gate parameters
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.outer_optimizer.step()

        return float(nll_target.item()), (feat.detach() if feat is not None else None)

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

        # Update running error baseline for Friston precision dynamics
        if self.running_error_baseline is None:
            self.running_error_baseline = prequential_nll
        else:
            self.running_error_baseline = (
                self.error_ema_beta * self.running_error_baseline
                + (1.0 - self.error_ema_beta) * prequential_nll
            )

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

        # 3. Outer Evidence Update (single-counting evidence into slow weights & channel gate)
        _, current_feat = self.outer_evidence_update(
            tokens=tokens,
            converged_q=converged_q,
            prior=prior_for_window,
            prev_posterior=self.previous_posterior,
            prev_content_feat=self.previous_content_feat,
            log_precision_ratio=self.previous_log_prec_ratio,
        )

        # 4. Continuous Temporal State Transition with GDN channel-wise adaptation
        if self.adaptive_retention:
            eps = 1e-4
            curr_err = max(float(prequential_nll), eps)
            base_err = max(float(self.running_error_baseline), eps)
            precision_ratio = (base_err / curr_err) ** 2
            log_precision_ratio = math.log(max(precision_ratio, 1e-8))

            if hasattr(self.model, "modulator") and hasattr(self.model.modulator, "compute_channel_retention"):
                effective_retention = self.model.modulator.compute_channel_retention(
                    log_precision_ratio=log_precision_ratio,
                    content_feat=current_feat,
                    retention_min=self.retention_min,
                    retention_max=self.retention_max,
                ).detach()
                retention_mean_val = float(effective_retention.mean().item())
            else:
                effective_retention, precision_ratio = compute_friston_adaptive_retention(
                    base_retention=self.base_retention,
                    prediction_error=prequential_nll,
                    baseline_error=self.running_error_baseline,
                    error_sensitivity=self.error_sensitivity,
                    retention_min=self.retention_min,
                    retention_max=self.retention_max,
                )
                if isinstance(effective_retention, torch.Tensor):
                    retention_mean_val = float(effective_retention.mean().item())
                else:
                    retention_mean_val = float(effective_retention)
        else:
            effective_retention = self.base_retention
            precision_ratio = 1.0
            log_precision_ratio = 0.0
            if isinstance(effective_retention, torch.Tensor):
                retention_mean_val = float(effective_retention.mean().item())
            else:
                retention_mean_val = float(effective_retention)

        next_prior = converged_q.transition(
            retention=effective_retention, base_mean=0.0, base_log_std=0.0
        )
        self.current_prior = next_prior

        # Save for next window's empirical Bayes GDN gate alignment
        self.previous_posterior = converged_q.detach()
        self.previous_content_feat = current_feat
        self.previous_log_prec_ratio = log_precision_ratio

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
            inner_steps_taken=(0 if self.inner_max_steps <= 0 else len(inner_steps)),
            plateau_reached=plateau_reached,
            wall_time_ms=wall_time_ms,
            retention_mean=retention_mean_val,
            precision_ratio=precision_ratio,
        )

        self.history.append(report)
        return report
