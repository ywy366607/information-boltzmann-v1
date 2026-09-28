"""Active-inference M-path ponder wrapper for the canonical CBIM T^3 field.

The wrapper leaves the canonical CBIM write, transport, collision, bath, and
read operators intact.  Each external event writes once, launches M complete
latent-controlled fields, evolves each for elapsed internal time T using K
numerical substeps (dt=T/K), and predicts the immediate next observation.
Observed targets only infer posterior responsibilities over already-generated
prior paths, so deployment never receives future-token information.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .torus3d import CBIMTorus3D, UnifiedTorusDissipation


@dataclass
class MTPonderOutput:
    """One external event through M complete internal field trajectories."""

    log_probs: torch.Tensor
    next_field: torch.Tensor
    branch_fields: torch.Tensor
    branch_log_probs: torch.Tensor
    route_weights: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class _GaussianPathPrior(nn.Module):
    """State-conditioned Gaussian path generator over z and log elapsed time."""

    def __init__(self, d: int, latent_dim: int, branches: int) -> None:
        super().__init__()
        self.d = int(d)
        self.latent_dim = int(latent_dim)
        self.branches = int(branches)
        self.net = nn.Sequential(
            nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 2 * latent_dim + 2),
        )
        # Standard path noise at start; branch carriers make M complete fields
        # distinct even under deterministic evaluation.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.branch_carrier = nn.Parameter(torch.randn(branches, latent_dim) * 0.02)

    def forward(
        self, field_summary: torch.Tensor, token_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(torch.cat((field_summary, token_embed), dim=-1))
        mu_z, logvar_z, mu_log_t, logvar_log_t = torch.split(
            raw, (self.latent_dim, self.latent_dim, 1, 1), dim=-1)
        return (
            mu_z[:, None] + self.branch_carrier[None],
            logvar_z[:, None].expand(-1, self.branches, -1),
            mu_log_t[:, None].expand(-1, self.branches, -1),
            logvar_log_t[:, None].expand(-1, self.branches, -1),
        )


class CBIMActivePonder3D(nn.Module):
    """M full-field hypotheses with particle VFE and K-resolution evolution.

    ``K`` is only numerical quadrature.  Each path advances an elapsed time
    ``T_m=exp(log_T_m)`` through all kinetic operators using exactly
    ``dt_m=T_m/K``.  Every branch starts after the same write and evolves a
    full T^3 field.  For an observed target y, the particle posterior is

        q(m) proportional to p(m | F, x) p(y | F_m).

    This supplies active-inference free energy while retaining a deployment
    path generator that only conditions on the existing field and token.
    """

    architecture = "CBIM-Torus3D-active-ponder-v2-particle-vfe"

    def __init__(
        self,
        core: CBIMTorus3D,
        *,
        branches: int = 8,
        integration_steps: int = 64,
        latent_dim: int = 32,
        vfe_beta: float = 1.0,
    ) -> None:
        super().__init__()
        if branches < 2:
            raise ValueError("Active pondering requires at least two paths")
        if integration_steps < 1:
            raise ValueError("integration_steps must be positive")
        self.core = core
        self.branches = int(branches)
        self.integration_steps = int(integration_steps)
        self.latent_dim = int(latent_dim)
        self.vfe_beta = float(vfe_beta)
        d = core.d
        self.prior = _GaussianPathPrior(d, latent_dim, branches)
        self.path_to_token = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, d))
        self.router = nn.Sequential(
            nn.LayerNorm(3 * d), nn.Linear(3 * d, d), nn.SiLU(), nn.Linear(d, 1))
        self.free_energy_critic = nn.Sequential(
            nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, 1))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        nn.init.zeros_(self.free_energy_critic[-1].weight)
        nn.init.zeros_(self.free_energy_critic[-1].bias)

    @property
    def vocab_size(self) -> int:
        return self.core.vocab_size

    def initial_state(self, batch_size: int, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.core.initial_state(batch_size, *args, **kwargs)

    def set_ness_prior(self, mature_state: torch.Tensor) -> None:
        self.core.set_ness_prior(mature_state)

    @staticmethod
    def _sample(
        mu_z: torch.Tensor,
        logvar_z: torch.Tensor,
        mu_log_t: torch.Tensor,
        logvar_log_t: torch.Tensor,
        *,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if deterministic:
            z, log_t = mu_z, mu_log_t
        else:
            z = mu_z + torch.randn_like(mu_z) * (0.5 * logvar_z).exp()
            log_t = mu_log_t + torch.randn_like(mu_log_t) * (0.5 * logvar_log_t).exp()
        return z, log_t, log_t.exp().squeeze(-1)

    def _evolve(
        self,
        fields: torch.Tensor,
        controlled_token: torch.Tensor,
        duration: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evolve every path with requested total elapsed time exactly ``T``."""
        dt = duration / float(self.integration_steps)
        last_collision: dict[str, torch.Tensor] = {}
        last_bath: dict[str, torch.Tensor] = {}
        last_direction: torch.Tensor | None = None
        for _ in range(self.integration_steps):
            direction = (
                self.core.direction_controller(fields, controlled_token)
                if self.core.continuous_velocities else None
            )
            multiplier, _ = self.core.transport.multiplier(dt, direction=direction)
            fields = self.core.transport.apply_multiplier(fields, multiplier)
            fields, last_collision = self.core.collision(fields, dt)
            if isinstance(self.core.bath, UnifiedTorusDissipation):
                fields, last_bath = self.core.bath(fields, dt, tok_embed=controlled_token)
            else:
                fields, last_bath = self.core.bath(fields, dt)
            last_direction = direction
        diagnostics: dict[str, torch.Tensor] = {
            "internal_time_mean": duration.detach().mean(),
            "integration_dt_mean": dt.detach().mean(),
            "integration_dt_sum": (dt * self.integration_steps).detach().mean(),
            **last_collision,
            **last_bath,
        }
        if last_direction is not None:
            pairwise = torch.bmm(last_direction, last_direction.transpose(1, 2))
            q = last_direction.shape[1]
            off_diag = (pairwise.sum((-1, -2)) - q) / (q * (q - 1))
            diagnostics["dir_pairwise_sep_deg"] = torch.rad2deg(
                torch.acos(off_diag.clamp(-1.0, 1.0))).mean().detach()
        return fields, diagnostics

    def event(
        self,
        field: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        target_ids: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> MTPonderOutput:
        """Assimilate one token, create M full paths, and predict its successor.

        Passing a target evaluates VFE and learning diagnostics.  It does not
        alter the proposed paths, router input, duration, or selected state.
        """
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be [batch]")
        batch = token_ids.shape[0]
        written, _, write_diag = self.core.source(field, token_ids)
        token_embed = self.core.source.embedding(token_ids)
        summary = written.reshape(batch, -1, self.core.d).mean(1)
        p_args = self.prior(summary, token_embed)
        z, log_t, duration = self._sample(*p_args, deterministic=deterministic)
        control = self.path_to_token(z)
        controlled_token = (token_embed[:, None] + control).reshape(batch * self.branches, self.core.d)
        fields = written[:, None].expand(
            -1, self.branches, -1, -1, -1, -1).reshape(batch * self.branches, *self.core.state_shape)
        evolved, dynamics_diag = self._evolve(fields, controlled_token, duration.reshape(-1))
        if self.core.readout_type == "baseline":
            branch_feature = self.core.readout(evolved)
        else:
            branch_feature, _ = self.core.readout(evolved, controlled_token, return_diag=False)
        branch_logits = self.core.decoder(branch_feature).reshape(batch, self.branches, self.vocab_size)
        branch_log_probs = branch_logits.log_softmax(-1)
        branch_summary = evolved.reshape(batch, self.branches, -1, self.core.d).mean(2)
        router_input = torch.cat((branch_summary, summary[:, None].expand_as(branch_summary), control), -1)
        route_logits = self.router(router_input).squeeze(-1)
        route_weights = route_logits.softmax(-1)
        log_probs = torch.logsumexp(route_weights.clamp_min(1e-12).log()[:, :, None] + branch_log_probs, dim=1)
        selected = route_logits.argmax(-1)
        evolved_5d = evolved.reshape(batch, self.branches, *self.core.state_shape)
        next_field = evolved_5d[torch.arange(batch, device=field.device), selected]
        diagnostics: dict[str, torch.Tensor] = {
            **write_diag,
            **dynamics_diag,
            "field_energy": (0.5 * next_field.detach().square().sum(-1).mean()).detach(),
            "branch_effective_count": (-(route_weights * route_weights.clamp_min(1e-12).log()).sum(-1)).exp().mean().detach(),
            "branch_field_spread": (evolved_5d - evolved_5d[:, :1]).square().mean().sqrt().detach(),
            "selected_branch": selected.detach(),
            # Derived device scalar: host-originated ``torch.tensor`` is not
            # CUDA-graph capture safe.
            "posterior_active": field.reshape(-1)[0].detach() * 0.0 + float(target_ids is not None),
        }
        critic = self.free_energy_critic(torch.cat((branch_summary, control), dim=-1)).squeeze(-1)
        diagnostics["predicted_free_energy"] = critic.detach().mean()
        if target_ids is not None:
            target_index = target_ids[:, None, None].expand(-1, self.branches, 1)
            branch_nll = -branch_log_probs.gather(-1, target_index).squeeze(-1)
            log_route = route_weights.clamp_min(1e-12).log()
            posterior_log_weights = log_route - branch_nll
            posterior_weights = posterior_log_weights.softmax(-1)
            accuracy = (posterior_weights * branch_nll).sum(-1)
            complexity = (posterior_weights * (
                posterior_weights.clamp_min(1e-12).log() - log_route)).sum(-1)
            free_energy = accuracy + self.vfe_beta * complexity
            reward = -branch_nll.detach()
            advantage = (reward - reward.mean(1, keepdim=True)) / reward.std(1, keepdim=True).clamp_min(1e-4)
            p_mu_z, p_logvar_z, p_mu_t, p_logvar_t = p_args
            path_log_prob = -0.5 * (
                ((z.detach() - p_mu_z).square() / p_logvar_z.exp()) + p_logvar_z
            ).sum(-1) - 0.5 * (
                ((log_t.detach() - p_mu_t).square() / p_logvar_t.exp()) + p_logvar_t
            ).squeeze(-1)
            diagnostics.update({
                "vfe": free_energy.mean(),
                "vfe_accuracy": accuracy.mean(),
                "vfe_complexity": complexity.mean(),
                "branch_nll_mean": branch_nll.mean(),
                "branch_nll_best": branch_nll.amin(1).mean(),
                "critic_loss": F.mse_loss(critic, branch_nll.detach()),
                "grpo_route_loss": -(log_route * advantage).sum(-1).mean(),
                "grpo_path_loss": -(path_log_prob * advantage).mean(),
                "posterior_effective_count": (-(posterior_weights * posterior_weights.clamp_min(1e-12).log()).sum(-1)).exp().mean(),
            })
        return MTPonderOutput(
            log_probs=log_probs,
            next_field=next_field,
            branch_fields=evolved_5d,
            branch_log_probs=branch_log_probs,
            route_weights=route_weights,
            diagnostics=diagnostics,
        )


