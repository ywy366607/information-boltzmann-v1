"""Active-inference GDN-2 memory for causal Slice evolution.

The official GDN-2 recurrence is a useful fixed-size fast-weight
approximation, but its erase/write gates are not by themselves a variational
posterior.  This module stores Gaussian natural parameters on a stable spatial
atlas.  Process noise lowers old precision (erase); likelihood precision adds
new natural parameters (write).  Suppressing the precision normalization and
making both precisions state-independent recovers the usual gated delta-rule
shape.

Transient Slice vectors are observations of this state, not the state itself.
The state may therefore be passed between *physical* frames without tying
network depth to time.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ActiveInferenceState:
    """Factorized Gaussian belief over an address-by-content fast weight."""

    natural: torch.Tensor       # [B,K,D], eta = precision * mean
    precision: torch.Tensor     # [B,K,1]
    velocity_mean: torch.Tensor # [B,2] in normalized (y,x) coordinates
    velocity_precision: torch.Tensor  # [B,1]
    last_tokens: Optional[torch.Tensor] = None
    last_centers: Optional[torch.Tensor] = None
    last_observation_precision: Optional[torch.Tensor] = None


def _atlas(n_slots: int) -> torch.Tensor:
    """Approximately square, fixed Eulerian address atlas in [-1,1]^2."""
    side = int(math.ceil(math.sqrt(int(n_slots))))
    # Periodic cell centers, not inclusive endpoints: -1 and +1 are the same
    # address under periodic_delta and would create duplicate memory slots.
    axis = -1.0 + (torch.arange(side, dtype=torch.float32) + 0.5) * (2.0 / side)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([yy, xx], dim=-1).reshape(-1, 2)[: int(n_slots)]


def periodic_delta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Shortest a-b displacement on the periodic normalized chart."""
    return torch.remainder(a - b + 1.0, 2.0) - 1.0


class ActiveInferenceGDN2(nn.Module):
    """Bayesian fast-weight memory with a GDN-2-compatible degeneration.

    The memory distribution is

        q(M) = N(eta / Lambda, Lambda^-1)

    on K fixed address slots.  A dynamic prior adds process variance Q before
    an observation contributes likelihood precision Pi.  Both operations are
    closed-form coordinate updates of variational free energy for the
    factorized Gaussian family.
    """

    def __init__(
        self,
        d_model: int,
        n_slots: int,
        res: int,
        address_temperature: float = 0.18,
        initial_prior_trust: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_slots = int(n_slots)
        self.res = int(res)
        self.register_buffer("atlas", _atlas(n_slots), persistent=True)
        self.address_temperature_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(float(address_temperature))))
        )
        # Position defines identity at initialization. Content may refine an
        # address, but cannot silently create a different atlas per depth.
        self.content_address = nn.Linear(d_model, n_slots, bias=False)
        nn.init.zeros_(self.content_address.weight)
        self.content_address_scale = nn.Parameter(torch.tensor(-3.0))

        self.log_likelihood_precision = nn.Linear(d_model, 1)
        nn.init.zeros_(self.log_likelihood_precision.weight)
        nn.init.zeros_(self.log_likelihood_precision.bias)
        self.prior_precision_raw = nn.Parameter(torch.tensor(-4.0))
        self.process_variance_raw = nn.Parameter(torch.tensor(-4.0))
        self.action_process_variance_raw = nn.Parameter(torch.tensor(-3.0))

        # Shared motion features compare observations in the same content
        # chart. Identity initialization makes the probe meaningful before a
        # long training run.
        self.motion_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.motion_proj.weight)
        self.motion_salience = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.motion_salience.weight)
        self.motion_precision_raw = nn.Parameter(torch.tensor(0.0))
        # One discrete micro-world cell is 2/3 of the normalized nine-grid.
        # It remains learnable for continuous worlds.
        self.action_scale_raw = nn.Parameter(torch.tensor(math.log(math.expm1(2.0 / 3.0))))
        # The atlas already lives in X-content coordinates. A dense map would
        # relearn and destroy that chart. Start from exact identity behavior
        # and learn only per-channel trust in the dynamic prior.
        trust = float(initial_prior_trust)
        if not -0.99 < trust < 0.99:
            raise ValueError("initial_prior_trust must be in (-0.99, 0.99)")
        self.prior_channel_gain = nn.Parameter(
            torch.full((d_model,), math.atanh(trust))
        )

        self.last_diagnostics: Dict[str, torch.Tensor] = {}

    @staticmethod
    def slice_centers(weights: torch.Tensor, point_coords: torch.Tensor) -> torch.Tensor:
        """Soft Slice centroids in the persistent point-field chart."""
        mass = weights.sum(dim=1).clamp_min(1e-6).unsqueeze(-1)
        return torch.einsum("bnm,bnd->bmd", weights, point_coords) / mass

    def initial_state(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ActiveInferenceState:
        precision = F.softplus(self.prior_precision_raw).to(device=device, dtype=dtype)
        precision = precision.expand(batch, self.n_slots, 1).clone()
        natural = torch.zeros(batch, self.n_slots, self.d_model, device=device, dtype=dtype)
        return ActiveInferenceState(
            natural=natural,
            precision=precision,
            velocity_mean=torch.zeros(batch, 2, device=device, dtype=dtype),
            velocity_precision=precision[:, :1, 0].clone(),
        )

    def address(
        self,
        tokens: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        """Key-address transient slices without assigning identity by index."""
        atlas = self.atlas.to(device=centers.device, dtype=centers.dtype)
        delta = periodic_delta(centers.unsqueeze(-2), atlas.view(1, 1, -1, 2))
        temp = F.softplus(self.address_temperature_raw).clamp_min(1e-3)
        positional = -delta.square().sum(dim=-1) / temp
        content = self.content_address(tokens)
        scale = torch.sigmoid(self.content_address_scale)
        return torch.softmax(positional + scale * content, dim=-1)

    def _infer_velocity(
        self,
        previous_tokens: torch.Tensor,
        previous_centers: torch.Tensor,
        tokens: torch.Tensor,
        centers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Infer a Gaussian velocity observation by content correspondence."""
        prev = F.normalize(self.motion_proj(previous_tokens), dim=-1)
        cur = F.normalize(self.motion_proj(tokens), dim=-1)
        match = torch.softmax(torch.matmul(cur, prev.transpose(-1, -2)) * 4.0, dim=-1)
        matched_centers = torch.matmul(match, previous_centers)
        displacement = periodic_delta(centers, matched_centers)
        # Novel/non-background slices receive more voice once learned. Uniform
        # is the conservative initialization and has no class-specific mask.
        salience = torch.softmax(self.motion_salience(tokens).squeeze(-1), dim=-1)
        velocity = (salience.unsqueeze(-1) * displacement).sum(dim=1)
        precision = F.softplus(self.motion_precision_raw).expand(tokens.shape[0], 1)
        return velocity, precision

    def transport_state(
        self,
        state: ActiveInferenceState,
        displacement_yx: torch.Tensor,
    ) -> ActiveInferenceState:
        """Predict M forward on the fixed atlas before likelihood correction.

        Destination addresses query the old belief at destination-displacement.
        Mean and variance are both transported; this is the missing causal
        predict step, not an extra observation or a depth update.
        """
        batch = state.natural.shape[0]
        displacement = torch.as_tensor(
            displacement_yx, device=state.natural.device, dtype=state.natural.dtype,
        ).reshape(batch, 2)
        destination = self.atlas.to(
            device=state.natural.device, dtype=state.natural.dtype,
        ).unsqueeze(0).expand(batch, -1, -1)
        source = torch.remainder(destination - displacement.unsqueeze(1) + 1.0, 2.0) - 1.0
        zeros = state.natural.new_zeros(batch, self.n_slots, self.d_model)
        transition = self.address(zeros, source)  # B,K_dst,K_src
        old_mean = state.natural / state.precision.clamp_min(1e-6)
        old_var = 1.0 / state.precision.clamp_min(1e-6)
        mean = torch.einsum("bij,bjd->bid", transition, old_mean)
        variance = torch.einsum("bij,bjq->biq", transition.square(), old_var)
        precision = variance.clamp_min(1e-6).reciprocal()
        natural = precision * mean
        self.last_diagnostics = {
            **self.last_diagnostics,
            "state_transition": transition.detach(),
            "state_displacement_yx": displacement.detach(),
        }
        return ActiveInferenceState(
            natural=natural,
            precision=precision,
            velocity_mean=state.velocity_mean,
            velocity_precision=state.velocity_precision,
            last_tokens=state.last_tokens,
            last_centers=state.last_centers,
            last_observation_precision=state.last_observation_precision,
        )

    def assimilate(
        self,
        state: ActiveInferenceState,
        tokens: torch.Tensor,
        centers: torch.Tensor,
        observation_precision,
        update_motion: bool = True,
        transport_memory: bool = True,
    ) -> ActiveInferenceState:
        """Minimize observation VFE by a closed-form natural update."""
        batch, n_events, _ = tokens.shape
        obs_pi = torch.as_tensor(
            observation_precision, device=tokens.device, dtype=tokens.dtype,
        )
        if obs_pi.ndim == 0:
            obs_pi = obs_pi.expand(batch)
        obs_pi = obs_pi.reshape(batch, 1).clamp_min(0.0)
        velocity_mean = state.velocity_mean
        velocity_precision = state.velocity_precision
        if update_motion and state.last_tokens is not None and state.last_centers is not None:
            v_obs, v_pi = self._infer_velocity(
                state.last_tokens, state.last_centers, tokens, centers,
            )
            previous_pi = state.last_observation_precision
            if previous_pi is not None:
                v_pi = v_pi * previous_pi.reshape(batch, 1).clamp(0.0, 1.0)
            v_pi = v_pi * obs_pi
            post_pi = velocity_precision + v_pi
            velocity_mean = (
                velocity_precision * velocity_mean + v_pi * v_obs
            ) / post_pi.clamp_min(1e-6)
            velocity_precision = post_pi
            if transport_memory:
                # Optional predict-before-correct filter. It remains an
                # explicit ablation until velocity correspondence is reliable;
                # confidence estimation alone must not silently move memory.
                displacement = velocity_mean * (obs_pi > 0).to(tokens.dtype)
                state = self.transport_state(state, displacement)

        address = self.address(tokens, centers)
        local_pi = F.softplus(self.log_likelihood_precision(tokens))
        event_pi = obs_pi.unsqueeze(1) * local_pi / max(1, n_events)
        write = address.unsqueeze(-1) * event_pi.unsqueeze(-2)  # B,M,K,1
        precision_add = write.sum(dim=1)
        natural_add = (write * tokens.unsqueeze(-2)).sum(dim=1)
        precision = state.precision + precision_add
        natural = state.natural + natural_add

        live = obs_pi > 0
        last_tokens = tokens if state.last_tokens is None else torch.where(
            live.unsqueeze(-1), tokens, state.last_tokens,
        )
        last_centers = centers if state.last_centers is None else torch.where(
            live.unsqueeze(-1), centers, state.last_centers,
        )
        last_pi = obs_pi
        if state.last_observation_precision is not None:
            last_pi = torch.where(live, obs_pi, state.last_observation_precision)
        self.last_diagnostics = {
            "write_precision": precision_add.detach(),
            "observation_precision": obs_pi.detach(),
        }
        return ActiveInferenceState(
            natural=natural,
            precision=precision,
            velocity_mean=velocity_mean,
            velocity_precision=velocity_precision,
            last_tokens=last_tokens,
            last_centers=last_centers,
            last_observation_precision=last_pi,
        )

    def dynamic_prior(
        self,
        state: ActiveInferenceState,
        horizon,
        action: Optional[torch.Tensor] = None,
        action_precision=None,
    ) -> ActiveInferenceState:
        """Apply Gaussian process noise: exact precision release/erase."""
        batch = state.natural.shape[0]
        tau = torch.as_tensor(horizon, device=state.natural.device, dtype=state.natural.dtype)
        if tau.ndim == 0:
            tau = tau.expand(batch)
        tau = tau.reshape(batch, 1, 1).clamp_min(0.0)
        if action is None:
            action_norm = tau.new_zeros(batch, 1, 1)
        else:
            act = torch.as_tensor(action, device=tau.device, dtype=tau.dtype)
            if act.ndim == 3:
                act = act[:, -1]
            act = act.reshape(batch, -1)
            action_norm = act.norm(dim=-1, keepdim=True).unsqueeze(-1)
        if action_precision is not None:
            api = torch.as_tensor(action_precision, device=tau.device, dtype=tau.dtype)
            if api.ndim > 1:
                api = api.reshape(batch, -1).amax(dim=-1)
            action_norm = action_norm * api.reshape(batch, 1, 1).clamp_min(0.0)
        process_var = tau * (
            F.softplus(self.process_variance_raw)
            + action_norm * F.softplus(self.action_process_variance_raw)
        )
        old_mean = state.natural / state.precision.clamp_min(1e-6)
        new_precision = 1.0 / (1.0 / state.precision.clamp_min(1e-6) + process_var)
        new_natural = new_precision * old_mean
        retention = new_precision / state.precision.clamp_min(1e-6)
        self.last_diagnostics = {
            **self.last_diagnostics,
            "erase_retention": retention.detach(),
            "process_variance": process_var.detach(),
        }
        return ActiveInferenceState(
            natural=new_natural,
            precision=new_precision,
            velocity_mean=state.velocity_mean,
            velocity_precision=state.velocity_precision,
            last_tokens=state.last_tokens,
            last_centers=state.last_centers,
            last_observation_precision=state.last_observation_precision,
        )

    def query(
        self,
        state: ActiveInferenceState,
        tokens: torch.Tensor,
        target_centers: torch.Tensor,
        horizon,
        action: Optional[torch.Tensor] = None,
        action_precision=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Read the prior at causal source addresses for target Slice slots."""
        batch = tokens.shape[0]
        tau = torch.as_tensor(horizon, device=tokens.device, dtype=tokens.dtype)
        if tau.ndim == 0:
            tau = tau.expand(batch)
        tau = tau.reshape(batch, 1)
        displacement = tau * state.velocity_mean
        if action is not None:
            act = torch.as_tensor(action, device=tokens.device, dtype=tokens.dtype)
            if act.ndim == 3:
                act = act[:, -1]
            act_yx = act.reshape(batch, -1)[..., [1, 0]]
            if action_precision is None:
                api = torch.ones(batch, 1, device=tokens.device, dtype=tokens.dtype)
            else:
                api = torch.as_tensor(action_precision, device=tokens.device, dtype=tokens.dtype)
                if api.ndim > 1:
                    api = api.reshape(batch, -1).amax(dim=-1)
                api = api.reshape(batch, 1).clamp_min(0.0)
            displacement = displacement + tau * api * F.softplus(self.action_scale_raw) * act_yx
        source_centers = torch.remainder(target_centers - displacement.unsqueeze(1) + 1.0, 2.0) - 1.0
        address = self.address(tokens, source_centers)
        memory_mean = state.natural / state.precision.clamp_min(1e-6)
        mean = torch.einsum("bmk,bkd->bmd", address, memory_mean)
        variance = torch.einsum(
            "bmk,bkq->bmq", address.square(), 1.0 / state.precision.clamp_min(1e-6),
        )
        logvar = variance.clamp_min(1e-6).log().expand_as(mean)
        diagnostics = {
            **self.last_diagnostics,
            "address": address.detach(),
            "source_centers": source_centers.detach(),
            "displacement_yx": displacement.detach(),
            "query_precision": variance.detach().reciprocal(),
        }
        self.last_diagnostics = diagnostics
        return mean, logvar, diagnostics

    def query_atlas(
        self,
        state: ActiveInferenceState,
        horizon,
        action: Optional[torch.Tensor] = None,
        action_precision=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Query K persistent destination addresses in their own chart."""
        batch = state.natural.shape[0]
        centers = self.atlas.to(
            device=state.natural.device, dtype=state.natural.dtype,
        ).unsqueeze(0).expand(batch, -1, -1)
        tokens = state.natural / state.precision.clamp_min(1e-6)
        return self.query(
            state, tokens, centers, horizon=horizon, action=action,
            action_precision=action_precision,
        )

    def point_to_atlas_weights(self, point_coords: torch.Tensor) -> torch.Tensor:
        """Persistent-coordinate Deslice weights [B,N,K], independent of content."""
        zeros = point_coords.new_zeros(
            point_coords.shape[0], point_coords.shape[1], self.d_model,
        )
        # content_address(0)=0, so only the fixed Eulerian atlas participates.
        return self.address(zeros, point_coords)

    def pool_field(
        self,
        field: torch.Tensor,
        point_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fixed physical SliceRead of the full-resolution field.

        Returns K atlas observations and their persistent coordinates. Unlike
        learned transient SliceRead, address identity is stable across frames.
        """
        if field.shape[-1] != self.d_model:
            raise ValueError(
                f"field width {field.shape[-1]} != memory width {self.d_model}"
            )
        weights = self.point_to_atlas_weights(point_coords)
        mass = weights.sum(dim=1).clamp_min(1e-6).unsqueeze(-1)
        tokens = torch.einsum(
            "bnk,bnd->bkd", weights, field,
        ) / mass
        centers = self.atlas.to(
            device=field.device, dtype=field.dtype,
        ).unsqueeze(0).expand(field.shape[0], -1, -1)
        return tokens, centers

    def apply_prior_action(
        self,
        current: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        """Land the causal prior as an exactly zero-initialized residual."""
        if current.shape != prior.shape:
            raise ValueError(f"prior shape {tuple(prior.shape)} != {tuple(current.shape)}")
        gain = torch.tanh(self.prior_channel_gain).view(1, 1, -1)
        return current + gain * (prior - current)

    def prior_residual(
        self,
        posterior: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-initialized causal delta in the persistent atlas chart."""
        if posterior.shape != prior.shape:
            raise ValueError(
                f"prior shape {tuple(prior.shape)} != posterior {tuple(posterior.shape)}"
            )
        gain = torch.tanh(self.prior_channel_gain).view(1, 1, -1)
        return gain * (prior - posterior)

    @staticmethod
    def gaussian_kl(
        q_mean: torch.Tensor,
        q_logvar: torch.Tensor,
        p_mean: torch.Tensor,
        p_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Elementwise KL(q||p), the transition-complexity part of VFE."""
        return 0.5 * (
            p_logvar - q_logvar
            + torch.exp(q_logvar - p_logvar)
            + (q_mean - p_mean).square() * torch.exp(-p_logvar)
            - 1.0
        )
