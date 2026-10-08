"""A continuing, delay-constrained reduced observer, updated once per tick.

This is an approximate regional observer, not a replacement for the physical
connectome. Its one-step forecast supplies an explicitly additional predictive
read path. Long histories travel through continuing state and real delay tiers;
there is no per-token horizon rollout. Frozen local projections make arrival
targets independent of learned encoder shrinkage.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class StreamingObserverState:
    history: torch.Tensor
    prior: torch.Tensor
    pending_source: torch.Tensor
    issued_forecast: torch.Tensor
    previous_observation: torch.Tensor
    tick: int = 0

    def detached(self):
        return type(self)(**{
            f.name: getattr(self, f.name).detach() if f.name != 'tick'
            else self.tick for f in fields(self)})

    def to(self, device):
        return type(self)(**{
            f.name: getattr(self, f.name).to(device) if f.name != 'tick'
            else self.tick for f in fields(self)})

    def state_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_state_dict(cls, payload):
        if set(payload) != {f.name for f in fields(cls)}:
            raise ValueError('Incomplete streaming observer continuation')
        return cls(**payload)


class DelayRegionGraph(nn.Module):
    """Initial real edge-weight aggregates, separated by delay and E/I.

    Regions are existing superclass labels split by the actual read surface.
    No edge or delay is invented. Aggregation can shorten within-region paths,
    so regional arrival times must not be called exact single-cell latencies.
    Magnitudes are normalized by total incoming weight across both signs and
    every delay; absent edges remain exactly zero.
    """

    def __init__(self, region_ids, motor_regions, excitatory, inhibitory):
        super().__init__()
        a_e = torch.as_tensor(excitatory, dtype=torch.float32)
        a_i = torch.as_tensor(inhibitory, dtype=torch.float32)
        if a_e.shape != a_i.shape or a_e.ndim != 3:
            raise ValueError('Expected [delay, pre-region, post-region] arrays')
        if a_e.shape[1] != a_e.shape[2] or a_e.shape[0] < 1:
            raise ValueError('Positive delay tiers and square graph required')
        if (a_e < 0).any() or (a_i < 0).any():
            raise ValueError('E/I magnitudes must be nonnegative')
        denominator = (a_e + a_i).sum(dim=(0, 1)).clamp_min(
            torch.finfo(a_e.dtype).tiny)
        self.register_buffer('a_e', a_e / denominator[None, None, :])
        self.register_buffer('a_i', a_i / denominator[None, None, :])
        self.register_buffer('region_ids', torch.as_tensor(region_ids, dtype=torch.long))
        self.register_buffer('motor_regions', torch.as_tensor(motor_regions, dtype=torch.long))
        self.num_regions = a_e.shape[1]
        self.max_delay = a_e.shape[0]
        if not len(motor_regions):
            raise ValueError('A motor-only observer requires a motor surface')

    @classmethod
    def from_model(cls, model):
        if model.synapse_model != 'coba' or model.superclass_id is None:
            raise ValueError('This route requires COBA edges and superclass metadata')
        classes = model.superclass_id.detach().cpu().numpy()
        motor = np.zeros(len(classes), dtype=np.int64)
        motor[model.read_indices.detach().cpu().numpy()] = 1
        _, region_ids = np.unique(np.stack((classes, motor), axis=1),
                                  axis=0, return_inverse=True)
        num_regions = int(region_ids.max()) + 1
        motor_regions = np.unique(region_ids[motor.astype(bool)])
        # The current physical kernel carries four pulse slots, even for graphs
        # with fewer occupied delay tiers. Splits include the initial zero.
        tiers = 4
        arrays = []
        for sign in ('e', 'i'):
            pre = getattr(model, 'edge_pre_' + sign).detach().cpu().numpy()
            post = getattr(model, 'edge_post_' + sign).detach().cpu().numpy()
            weights = getattr(model, 'edge_weight_' + sign).detach().cpu().numpy()
            splits = getattr(model, 'splits_' + sign)
            aggregate = np.zeros((tiers, num_regions, num_regions), np.float64)
            if not splits or splits[0] != 0 or len(splits) > tiers + 1:
                raise ValueError('Expected cumulative delay splits starting at zero')
            start = 0
            for delay, end in enumerate(splits[1:]):
                # The physical kernel uses cumulative split endpoints.
                codes = region_ids[pre[start:end]] * num_regions + region_ids[post[start:end]]
                aggregate[delay] = np.bincount(
                    codes, weights=np.abs(weights[start:end]),
                    minlength=num_regions * num_regions).reshape(num_regions, num_regions)
                start = end
            if start != len(weights):
                raise ValueError('Delay splits do not cover the physical edge array')
            arrays.append(aggregate)
        return cls(region_ids, motor_regions, *arrays)

    def incoming(self, history):
        # history[d-1] is the state whose message will arrive next tick at delay d.
        if history.shape[0] != self.max_delay:
            raise ValueError('Observer history does not match physical delay tiers')
        return (torch.einsum('dij,dbil->bjl', self.a_e, history),
                torch.einsum('dij,dbil->bjl', self.a_i, history))


class StreamingGraphObserver(nn.Module):
    def __init__(self, graph, output_dim, *, latent_dim=128,
                 sample_per_region=64, seed=11, initial_u=None):
        super().__init__()
        if latent_dim < 1 or sample_per_region < 1:
            raise ValueError('Positive observer capacity required')
        self.graph = graph
        self.latent_dim = int(latent_dim)
        region_ids = graph.region_ids.cpu().numpy()
        rng = np.random.default_rng(seed)
        indices = np.zeros((graph.num_regions, sample_per_region), dtype=np.int64)
        mask = np.zeros(indices.shape, dtype=np.float32)
        for region in range(graph.num_regions):
            candidates = np.flatnonzero(region_ids == region)
            size = min(len(candidates), sample_per_region)
            indices[region, :size] = np.sort(rng.choice(candidates, size, replace=False))
            mask[region, :size] = 1.0
        self.register_buffer('sample_indices', torch.from_numpy(indices))
        self.register_buffer('sample_mask', torch.from_numpy(mask))
        channels = 6 + graph.max_delay
        input_dim = channels * sample_per_region
        projection = rng.standard_normal((graph.num_regions, input_dim, latent_dim))
        projection /= np.sqrt(np.maximum(mask.sum(1), 1) * channels)[:, None, None]
        self.register_buffer('observation_projection', torch.tensor(projection, dtype=torch.float32))
        u0 = np.zeros_like(mask) if initial_u is None else np.asarray(initial_u)[indices]
        self.register_buffer('initial_u', torch.tensor(u0, dtype=torch.float32))
        # Shared local computation; incoming signs are distinct features.
        self.transition = nn.Sequential(nn.Linear(3 * latent_dim, latent_dim),
                                        nn.GELU(), nn.Linear(latent_dim, latent_dim))
        for layer in self.transition:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.bias)
        nn.init.normal_(self.transition[-1].weight, std=1e-3)
        self.innovation_gate = nn.Linear(2 * latent_dim, latent_dim)
        nn.init.zeros_(self.innovation_gate.weight)
        nn.init.zeros_(self.innovation_gate.bias)
        self.motor_adapter = nn.Linear(len(graph.motor_regions) * latent_dim,
                                       output_dim, bias=False)
        # Preserve the physical-read baseline exactly at initialization.
        nn.init.zeros_(self.motor_adapter.weight)

    @classmethod
    def from_model(cls, model, *, latent_dim=128, sample_per_region=64, seed=11):
        graph = DelayRegionGraph.from_model(model)
        u0 = model.get_stp_params()[0].detach().cpu().expand(1, model.n_neurons)[0].numpy()
        return cls(graph, model.embedding.embedding_dim, latent_dim=latent_dim,
                   sample_per_region=sample_per_region, seed=seed, initial_u=u0)

    def initial_state(self, batch, device, dtype):
        z = torch.zeros(batch, self.graph.num_regions, self.latent_dim,
                        device=device, dtype=dtype)
        return StreamingObserverState(
            z.unsqueeze(0).expand(self.graph.max_delay, -1, -1, -1).clone(),
            z.clone(), torch.cat((z, z, z), dim=-1), z.clone(), z.clone())

    def encode(self, physical):
        idx, mask = self.sample_indices, self.sample_mask
        values = [physical.h[:, idx], physical.ge[:, idx], physical.gi[:, idx],
                  physical.b[:, idx], physical.x[:, idx] - 1.0,
                  physical.u[:, idx] - self.initial_u]
        if len(physical.ring) != self.graph.max_delay:
            raise ValueError('Physical and observed ring depths differ')
        values.extend(pulse[:, idx] for pulse in physical.ring)
        local = torch.stack(values, dim=2) * mask[None, :, None, :]
        return torch.einsum('brf,rfl->brl', local.flatten(2), self.observation_projection)

    def predict(self, source):
        return source[..., :self.latent_dim] + self.transition(source)

    def step(self, physical, state):
        """Issue one forecast and assimilate only evidence already present.

        Stored issued_forecast is scored unchanged when its deadline arrives.
        Training recomputes the pending source with current parameters, including
        across window boundaries. This gives bounded online replay, not unlimited
        physical credit. Within a BPTT window source history still has its graph.
        The future-input convention is conditional mean under the observed stream.
        """
        observed = self.encode(physical)
        if state.history.shape[1:] != observed.shape:
            raise ValueError('Observer batch/region/latent shape mismatch')
        target = observed.detach()
        if state.tick:
            replay_prediction = self.predict(state.pending_source)
            auxiliary = F.mse_loss(replay_prediction, target)
            issued_mse = F.mse_loss(state.issued_forecast.detach(), target)
            persistence_mse = F.mse_loss(state.previous_observation.detach(), target)
        else:
            # No forecast was issued before birth; no artificial target scored.
            auxiliary = self.predict(state.pending_source).sum() * 0.0
            issued_mse = persistence_mse = target.sum() * 0.0
        innovation = observed - state.prior
        gain = torch.sigmoid(self.innovation_gate(torch.cat((observed, state.prior), -1)))
        posterior = state.prior + gain * innovation
        history = torch.cat((posterior[None], state.history[:-1]), dim=0)
        incoming_e, incoming_i = self.graph.incoming(history)
        source = torch.cat((posterior, incoming_e, incoming_i), dim=-1)
        forecast = self.predict(source)
        motor = forecast[:, self.graph.motor_regions].flatten(1)
        output = self.motor_adapter(motor)
        next_state = StreamingObserverState(history, forecast, source,
                                             forecast.detach(), observed.detach(), state.tick + 1)
        metrics = {
            'observer_arrival_replay_mse': auxiliary.detach(),
            'observer_issued_forecast_mse': issued_mse,
            'observer_persistence_mse': persistence_mse,
            'observer_forecast_gain_over_persistence': persistence_mse - issued_mse,
            'observer_innovation_rms': innovation.detach().square().mean().sqrt(),
            'observer_gain_mean': gain.detach().mean(),
            'observer_evidence_rms': target.square().mean().sqrt(),
            'observer_adapter_rms': output.detach().square().mean().sqrt(),
            'observer_scored_arrival': target.new_tensor(float(state.tick > 0)),
        }
        return output, auxiliary, next_state, metrics
