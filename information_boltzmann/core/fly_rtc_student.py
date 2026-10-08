"""Motor-response drafts over physical ticks, with explicit driving inputs.

The sampled codec is fixed and includes delayed pulses and synaptic state.
This is an approximate reduced model, not an invertible physical state or an
exact Markov quotient. Only motor-region forecasts reach the vocabulary head.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .fly_streaming_observer import StreamingGraphObserver


class PhysicalObservationCodec(StreamingGraphObserver):
    """Reuse the audited full-local-state codec without its old observer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.transition, self.innovation_gate, self.motor_adapter

    def encode_drive(self, drive: torch.Tensor) -> torch.Tensor:
        # h is channel zero of the fixed regional projection. No token or
        # embedding is supplied directly to a motor head.
        sampled = drive[:, self.sample_indices] * self.sample_mask[None]
        projection = self.observation_projection[:, :self.sample_indices.shape[1]]
        return torch.einsum('brs,rsl->brl', sampled, projection)


class TickResponseStudent(nn.Module):
    def __init__(self, codec, output_dim: int, *, horizon: int = 14):
        super().__init__()
        if horizon < 1:
            raise ValueError('Positive physical forecast horizon required')
        self.codec, self.horizon = codec, int(horizon)
        dim = codec.latent_dim
        self.transition = nn.Sequential(
            nn.LayerNorm(4 * dim), nn.Linear(4 * dim, dim),
            nn.GELU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.transition[-1].weight)
        nn.init.zeros_(self.transition[-1].bias)
        motor_dim = len(codec.graph.motor_regions) * dim
        self.query = nn.Linear(motor_dim, dim, bias=False)
        self.key = nn.Linear(motor_dim, dim, bias=False)
        self.horizon_bias = nn.Parameter(torch.zeros(horizon + 1))
        self.motor_adapter = nn.Linear(motor_dim, output_dim, bias=False)
        nn.init.zeros_(self.motor_adapter.weight)

    @classmethod
    def from_model(cls, model, *, latent_dim=128, sample_per_region=64,
                   horizon=14, seed=11):
        codec = PhysicalObservationCodec.from_model(
            model, latent_dim=latent_dim, sample_per_region=sample_per_region,
            seed=seed)
        return cls(codec, model.embedding.embedding_dim, horizon=horizon)

    def initial_history(self, batch, device, dtype):
        return torch.zeros(self.codec.graph.max_delay, batch,
                           self.codec.graph.num_regions, self.codec.latent_dim,
                           device=device, dtype=dtype)

    def transition_step(self, history, drive=None):
        """history[0] is the current state; delay d uses history[d-1]."""
        excitatory, inhibitory = self.codec.graph.incoming(history)
        current = history[0]
        if drive is None:
            drive = torch.zeros_like(current)
        delta = self.transition(torch.cat(
            (current, excitatory, inhibitory, drive), dim=-1))
        # Residual scale is learned by the final layer, not a biological clock.
        following = current + delta
        return following, torch.cat((following[None], history[:-1]), dim=0)

    def rollout(self, history, controls=None, *, horizon=None):
        """Predict a declared drive sequence; omitted controls mean zero drive.

        Unknown future stimuli are NOT inferred from later labels. Quiet drafts
        represent the response to already-in-flight input only. A new stimulus
        requires a new revision of the uncommitted forecast suffix.
        """
        length = self.horizon if horizon is None else horizon
        if not 0 <= length <= self.horizon:
            raise ValueError('Forecast exceeds configured compute horizon')
        if controls is not None and len(controls) != length:
            raise ValueError('One explicit control per predicted physical tick')
        states = [history[0]]
        for tick in range(length):
            control = None if controls is None else controls[tick]
            following, history = self.transition_step(history, control)
            states.append(following)
        return torch.stack(states, dim=0), history

    def read_motor_drafts(self, drafts):
        # [horizon+1, batch, region, latent] -> [batch, horizon+1, motor*latent]
        motor = drafts[:, :, self.codec.graph.motor_regions].flatten(2).transpose(0, 1)
        q = self.query(motor[:, 0])[:, None]
        keys = self.key(motor)
        logits = (q * keys).sum(-1) / self.codec.latent_dim**0.5
        weights = F.softmax(logits + self.horizon_bias, dim=-1)
        pooled = (weights[..., None] * (motor - motor[:, :1])).sum(1)
        return self.motor_adapter(pooled), weights

    def draft_features(self, drafts, current_motor_features):
        motor = drafts[:, :, self.codec.graph.motor_regions].flatten(2)
        return current_motor_features[None] + self.motor_adapter(motor - motor[:1])

    def issue(self, history):
        drafts, _ = self.rollout(history)
        response, weights = self.read_motor_drafts(drafts)
        return response, drafts, weights

    def perturb_copy(self, physical, discrepancy, *, relative_radius: float):
        """Adjoint steering, followed by mandatory re-encoding at query time.

        No inverse-codec claim. Radius is a declared query budget, relative to
        the actual sampled membrane RMS. Other full physical fields are cloned.
        The external perturbation's squared norm is reported as work proxy.
        """
        from .fly_bptt_learning import FlyPhysicalState
        if not 0 <= relative_radius <= 1:
            raise ValueError('Relative physical-query radius must be in [0, 1]')
        payload = physical.detached().state_dict()
        copied = {name: tuple(t.clone() for t in value) if name == 'ring'
                  else value.clone() for name, value in payload.items()}
        codec = self.codec
        projection = codec.observation_projection[:, :codec.sample_indices.shape[1]]
        direction = torch.einsum('brl,rsl->brs', discrepancy.detach(), projection)
        direction = direction * codec.sample_mask[None]
        direction_rms = direction.square().mean(-1, keepdim=True).sqrt()
        observed = physical.h[:, codec.sample_indices] * codec.sample_mask[None]
        scale = observed.square().mean(-1, keepdim=True).sqrt()
        change = relative_radius * scale * direction / direction_rms.clamp_min(
            torch.finfo(direction.dtype).eps)
        change = change * codec.sample_mask[None]
        flat_idx = codec.sample_indices.flatten()[None].expand(physical.h.shape[0], -1)
        delta = torch.zeros_like(physical.h).scatter_add(1, flat_idx, change.flatten(1))
        copied['h'] = copied['h'] + delta
        return FlyPhysicalState(**copied), delta.square().sum().detach()
