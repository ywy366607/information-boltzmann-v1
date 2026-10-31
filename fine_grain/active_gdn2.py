"""Active-inference GDN-2 memory for causal Slice evolution.

The official GDN-2 recurrence is a useful fixed-size fast-weight
approximation, but its erase/write gates are not by themselves a variational
posterior.  This module stores Gaussian natural parameters on a stable spatial
atlas.  Process noise lowers old precision (erase); likelihood precision adds
new natural parameters (write). With a one-hot address, no global decay and
a matching precision-derived gain, the mean update agrees with a delta write;
mixed addresses generally give different updates.

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
    # Cached causal address input; with the legacy shared representation this
    # is exactly the last event value tensor.
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
    """Bayesian fast-weight memory with a conditional delta-write relation.

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
    def association_write(
        keys: torch.Tensor,
        values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the value-by-key outer-product sum ``V Kᵀ``.

        ``keys`` are the soft atlas addresses ``[B,M,K]`` and ``values`` are
        event contents ``[B,M,D]``.  The result ``[B,D,K]`` is the exact
        associative state used by the Lean `associationState` reference;
        observation weights only scale the arrived events.
        """
        if weights is not None:
            values = values * weights
        return torch.einsum("bmk,bmd->bdk", keys, values)

    @staticmethod
    def association_state(state: ActiveInferenceState) -> torch.Tensor:
        """Expose the persistent natural state as a value-by-key matrix."""
        return state.natural.transpose(-1, -2)

    @staticmethod
    def association_readout(
        state: ActiveInferenceState,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """Read value vectors from soft keys in the associative state.

        `keys` has shape ``[B,M,K]`` and the result has ``[B,M,D]``.  The
        precision-normalized value-by-key matrix is queried by the same
        contraction as the Lean associative state, so probe error can be
        measured without changing the causal update.
        """
        value_by_key = ActiveInferenceGDN2.association_state(state)
        value_by_key = value_by_key / state.precision.transpose(-1, -2).clamp_min(1e-6)
        return torch.einsum("bdk,bmk->bmd", value_by_key, keys)

    @staticmethod
    def association_probe_error(
        state: ActiveInferenceState,
        keys: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-batch worst absolute error on a declared probe set."""
        readout = ActiveInferenceGDN2.association_readout(state, keys)
        return (readout - targets).abs().amax(dim=(-1, -2))

    @staticmethod
    def association_interference_bound(
        effective_keys: torch.Tensor, queries: torch.Tensor,
    ) -> torch.Tensor:
        """Row-sum error bound for bounded values, on a declared probe family.

        Both inputs are ``[B, M, K]``. The result ``[B, M]`` is the
        absolute row sum of ``queries @ effective_keys.T - I``. It applies
        to values in [-1, 1] and excludes any old-state contribution and
        floating-point roundoff. This diagnostic materializes an M-by-M
        temporary; it is not persistent organism state or a per-tick update.
        """
        cross = torch.matmul(queries, effective_keys.transpose(-1, -2))
        identity = torch.eye(cross.shape[-1], dtype=cross.dtype, device=cross.device)
        return (cross - identity).abs().sum(dim=-1)

    @staticmethod
    def association_delta_update(
        state: ActiveInferenceState,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: float = 1.0,
    ) -> ActiveInferenceState:
        """Reference GDN delta write: erase the key's prediction, then write value.

        In mean value-by-key form this is ``S ← S + β(v − S k)kᵀ``, where
        ``S`` is natural state divided by slot precision, as in readout. The
        changed mean is re-encoded into natural parameters using the same
        clamped precision. The Bayesian assimilate path remains unchanged.
        """
        state_matrix = ActiveInferenceGDN2.association_state(state)
        precision = state.precision.transpose(-1, -2).clamp_min(1e-6)
        mean = state_matrix / precision
        prediction = torch.einsum("bdk,bk->bd", mean, key)
        mean_delta = float(beta) * torch.einsum("bd,bk->bdk", value - prediction, key)
        state_matrix = state_matrix + precision * mean_delta
        return ActiveInferenceState(
            natural=state_matrix.transpose(-1, -2),
            precision=state.precision,
            velocity_mean=state.velocity_mean,
            velocity_precision=state.velocity_precision,
            last_tokens=state.last_tokens,
            last_centers=state.last_centers,
            last_observation_precision=state.last_observation_precision,
        )

    @staticmethod
    def association_dual_keys(keys: torch.Tensor, ridge: float = 0.0) -> torch.Tensor:
        """Return the exact Gram-dual query keys when the probe Gram is full rank.

        For row keys ``K``, the returned ``Q`` satisfies ``K @ Q.T = I``
        when ``ridge=0`` and the rows are independent.  A positive ridge is
        an explicit approximate-decoding choice, not part of the exact Lean
        capacity theorem.
        """
        gram = torch.matmul(keys, keys.transpose(-1, -2))
        if ridge:
            eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
            gram = gram + float(ridge) * eye
        return torch.linalg.solve(gram, keys)

    @staticmethod
    def association_dual_readout(
        state: ActiveInferenceState,
        keys: torch.Tensor,
        ridge: float = 0.0,
    ) -> torch.Tensor:
        """Read an associative state with the Gram-dual probe decoder."""
        return ActiveInferenceGDN2.association_readout(
            state, ActiveInferenceGDN2.association_dual_keys(keys, ridge=ridge),
        )

    @staticmethod
    def association_probe_uncertainty(
        state: ActiveInferenceState,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-query variance induced by the slot precision."""
        slot_variance = 1.0 / state.precision.clamp_min(1e-6)
        return torch.einsum("bmk,bkq->bmq", keys.square(), slot_variance)

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
        *,
        address_tokens: Optional[torch.Tensor] = None,
        likelihood_tokens: Optional[torch.Tensor] = None,
    ) -> ActiveInferenceState:
        """Minimize observation VFE by a closed-form natural update.

        ``tokens`` are values written into the associative state.  The optional
        address and likelihood views expose the three distinct terms of the
        update while defaulting to the historical shared representation.  The
        explicit views allow a proven value-only learning path without silently
        changing where the event is written or how strongly it is trusted.
        """
        batch, n_events, _ = tokens.shape
        if address_tokens is None:
            address_tokens = tokens
        if likelihood_tokens is None:
            likelihood_tokens = tokens
        if tuple(address_tokens.shape) != tuple(tokens.shape):
            raise ValueError("address tokens must match written-value shape")
        if tuple(likelihood_tokens.shape) != tuple(tokens.shape):
            raise ValueError("likelihood tokens must match written-value shape")
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
                state.last_tokens, state.last_centers, address_tokens, centers,
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

        address = self.address(address_tokens, centers)
        local_pi = F.softplus(self.log_likelihood_precision(likelihood_tokens))
        event_pi = obs_pi.unsqueeze(1) * local_pi / max(1, n_events)
        write = address.unsqueeze(-1) * event_pi.unsqueeze(-2)  # B,M,K,1
        precision_add = write.sum(dim=1)
        natural_add = self.association_write(
            address, tokens, event_pi,
        ).transpose(-1, -2)
        precision = state.precision + precision_add
        natural = state.natural + natural_add

        live = obs_pi > 0
        # The retained causal key is the address view.  With the defaults it
        # is exactly the old ``tokens`` cache; a value-only update therefore
        # cannot alter the next query through this cache.
        last_tokens = address_tokens if state.last_tokens is None else torch.where(
            live.unsqueeze(-1), address_tokens, state.last_tokens,
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
        mean = self.association_readout(state, address)
        variance = self.association_probe_uncertainty(state, address)
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
