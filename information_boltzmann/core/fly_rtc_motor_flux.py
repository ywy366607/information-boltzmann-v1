"""Experimental local physics closure with learned unresolved arrival flux.

Keep exact motor-local COBA/ALIF/STP state and motor-motor delay edges. Pulses
already in flight supply a known external arrival prefix. Only newly emitted
outside pulses require prediction. This does not close outside-brain dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy

import torch
from torch import nn

from .fly_reservoir import SpikeFn
from .triton_synapse import execute_delayed_synaptic_transmission


@dataclass
class MotorLocalState:
    h: torch.Tensor
    ge: torch.Tensor
    gi: torch.Tensor
    b: torch.Tensor
    x: torch.Tensor
    u: torch.Tensor
    mean: torch.Tensor
    ring: tuple[torch.Tensor, ...]


class MotorPhysicalDomain(nn.Module):
    """Exact physical equations on a selected set of anatomical neurons."""

    def __init__(self, model, *, state_indices=None, mean_decay=.99):
        super().__init__()
        if model.synapse_model != 'coba' or not model.use_alif or not model.use_stp:
            raise ValueError('This bounded arm requires the registered COBA/ALIF/STP oracle')
        self.mean_decay = mean_decay
        self.E_E, self.E_I = model.E_E, model.E_I
        indices = model.read_indices if state_indices is None else state_indices
        self.register_buffer('indices', indices.detach().clone())
        inverse = torch.full((model.n_neurons,), -1, dtype=torch.long, device=indices.device)
        inverse[self.indices] = torch.arange(len(self.indices), device=indices.device)
        self.register_buffer('inverse', inverse)
        for sign in ('e', 'i'):
            pre, post, weights = (getattr(model, 'edge_' + field + '_' + sign).detach()
                                  for field in ('pre', 'post', 'weight'))
            splits = getattr(model, 'splits_' + sign)
            inside = (inverse[pre.long()] >= 0) & (inverse[post.long()] >= 0)
            external = (inverse[pre.long()] < 0) & (inverse[post.long()] >= 0)
            self.register_buffer('pre_' + sign, inverse[pre[inside].long()].int())
            self.register_buffer('post_' + sign, inverse[post[inside].long()].int())
            self.register_buffer('weight_' + sign, weights[inside].clone())
            self.register_buffer('external_pre_' + sign, pre[external].long())
            self.register_buffer('external_post_' + sign, inverse[post[external].long()])
            self.register_buffer('external_weight_' + sign, weights[external].clone())
            internal_splits, external_splits = [0], [0]
            for d, (start, end) in enumerate(zip(splits[:-1], splits[1:])):
                internal_splits.append(internal_splits[-1] + int(inside[start:end].sum()))
                external_splits.append(external_splits[-1] + int(external[start:end].sum()))
            setattr(self, 'splits_' + sign, tuple(internal_splits))
            setattr(self, 'external_splits_' + sign, tuple(external_splits))

    def local_state(self, physical):
        idx = self.indices
        return MotorLocalState(*(getattr(physical, k)[:, idx] for k in ('h', 'ge', 'gi', 'b', 'x', 'u')),
                               physical.h_mean[:, idx], tuple(p[:, idx] for p in physical.ring))

    def coefficients(self, model):
        def cut(t):
            t = t.detach()
            return t[..., self.indices] if t.shape[-1] == model.n_neurons else t
        return dict(rates=tuple(cut(t) for t in model.get_decay_rates()),
                    threshold=cut(model.get_thresholds()), gains=tuple(t.detach() for t in model.get_conductance_gains()),
                    alif=tuple(cut(t) for t in model.get_alif_params()),
                    stp=tuple(cut(t) for t in model.get_stp_params()))

    def known_queue(self, physical):
        """Existing origin pulses only; no teacher future states are consulted."""
        outputs = []
        for k in range(1, 5):
            signs = []
            for sign in ('e', 'i'):
                q = torch.zeros_like(physical.h[:, self.indices])
                pre = getattr(self, 'external_pre_' + sign)
                post = getattr(self, 'external_post_' + sign)
                weight = getattr(self, 'external_weight_' + sign)
                split = getattr(self, 'external_splits_' + sign)
                for d in range(k, 5):
                    start, end = split[d-1:d+1]
                    if start == end:
                        continue
                    q = q.index_add(1, post[start:end],
                                    physical.ring[d-k][:, pre[start:end]] * weight[start:end])
                signs.append(q)
            outputs.append(torch.stack(signs, -1))
        return torch.stack(outputs)

    def external_arrival(self, physical):
        return self.known_queue(physical)[0]

    def transmit(self, ring):
        return [execute_delayed_synaptic_transmission(ring,
                    getattr(self, 'pre_' + s), getattr(self, 'post_' + s),
                    getattr(self, 'weight_' + s), getattr(self, 'splits_' + s)) for s in ('e', 'i')]

    def integrate(self, current, external, coefficients):
        """Same local equations as the production COBA kernel, given arrivals."""
        internal = self.transmit(current.ring)
        leak_m, leak_e, leak_i = coefficients['rates']
        ge = leak_e * current.ge + (1-leak_e) * (internal[0] + external[..., 0])
        gi = leak_i * current.gi + (1-leak_i) * (internal[1] + external[..., 1])
        gain_e, gain_i = coefficients['gains']
        G_E, G_I = gain_e * ge, gain_i * gi
        total = -torch.log(leak_m.clamp(min=1e-5, max=1-1e-7)) + G_E + G_I
        alpha = torch.exp(-total.clamp(min=1e-5, max=20.))
        beta = (1-alpha) / total.clamp_min(1e-5)
        rho_a, beta_a = coefficients['alif']
        spike = SpikeFn.apply(alpha * current.h + beta * (G_E*self.E_E + G_I*self.E_I)
                              - coefficients['threshold'] - beta_a*current.b)
        h = (alpha * current.h + beta * (G_E*self.E_E + G_I*self.E_I)) * (1-spike)
        b = rho_a * current.b + (1-rho_a)*spike
        u0, rho_fac, rho_rec, norm = coefficients['stp']
        active_u = current.u + u0*(1-current.u)*spike
        pulse = ((active_u * current.x / norm)*spike).clamp(max=3.)
        x = 1 + (current.x-active_u*current.x*spike-1)*rho_rec
        u = u0 + (active_u-u0)*rho_fac
        mean = self.mean_decay*current.mean+(1-self.mean_decay)*h
        return MotorLocalState(h, ge, gi, b, x, u, mean, (pulse, *current.ring[:3]))


class MotorFluxForecaster(MotorPhysicalDomain):
    def __init__(self, model, *, mean_decay=.99):
        super().__init__(model, mean_decay=mean_decay)
        self.latent_student = copy.deepcopy(model.rtc_student)
        for name in ('query', 'key', 'horizon_bias', 'motor_adapter'):
            delattr(self.latent_student, name)
        codec = self.latent_student.codec
        r, n, dim = codec.graph.num_regions, len(self.indices), codec.latent_dim
        for sign in ('e', 'i'):
            graph = torch.zeros(4, r, n, device=self.indices.device)
            pre = getattr(self, 'external_pre_' + sign)
            post = getattr(self, 'external_post_' + sign)
            weights = getattr(self, 'external_weight_' + sign)
            splits = getattr(self, 'external_splits_' + sign)
            for d, (start, end) in enumerate(zip(splits[:-1], splits[1:])):
                codes = codec.graph.region_ids[pre[start:end]] * n + post[start:end]
                graph[d] = torch.bincount(codes, weights=weights[start:end], minlength=r*n).reshape(r, n)
            norm = graph.sum((0, 1)).clamp_min(torch.finfo(graph.dtype).tiny)
            self.register_buffer('graph_' + sign, graph / norm[None, None])
            self.register_buffer('external_total_' + sign, graph.sum((0, 1)))
        self.flux_network = nn.Sequential(nn.Linear(2 * dim + 10, dim), nn.GELU(), nn.Linear(dim, 2))
        nn.init.zeros_(self.flux_network[-1].weight)
        nn.init.zeros_(self.flux_network[-1].bias)

    def feedback(self, history, local):
        """Encode actual/predicted local pulses back into regional state."""
        codec = self.latent_student.codec
        regions = codec.graph.motor_regions
        idx = self.inverse[codec.sample_indices[regions]]
        mask = codec.sample_mask[regions]
        values = [local.h[:, idx], local.ge[:, idx], local.gi[:, idx], local.b[:, idx],
                  local.x[:, idx]-1, local.u[:, idx]-codec.initial_u[regions]]
        values.extend(p[:, idx] for p in local.ring)
        observed = torch.einsum('brf,rfl->brl',
                (torch.stack(values, 2)*mask[None, :, None]).flatten(2),
                codec.observation_projection[regions])
        current = history[0].clone()
        current[:, regions] = observed
        return torch.cat((current[None], history[1:]))

    def forward(self, history, origin, coefficients, known_queue):
        """Future outside flux is learned; local physics uses own predicted state."""
        local = origin
        motors, means, fluxes, latents = [local.h], [local.mean], [], [history[0]]
        zero = torch.zeros_like(known_queue[0])
        for k in range(self.latent_student.horizon):
            history = self.feedback(history, local)
            incoming = [torch.einsum('drm,dbrl->bml', getattr(self, 'graph_'+s), history) for s in ('e','i')]
            state = torch.stack((local.h, local.ge, local.gi, local.b, local.x, local.u, *local.ring), -1)
            unknown = self.flux_network(torch.cat((*incoming, state), -1)).clamp_min(0)
            # At the first future tick every outside pulse is already in flight.
            # Known queue is a lower bound at later ticks; after4 all is modeled.
            if k == 0:
                unknown = torch.zeros_like(unknown)
            scales = torch.stack((self.external_total_e, self.external_total_i), -1)
            external = (known_queue[k] if k < 4 else zero) + unknown * scales[None]
            local = self.integrate(local, external, coefficients)
            z, history = self.latent_student.transition_step(history)
            history = self.feedback(history, local)
            fluxes.append(external)
            latents.append(history[0])
            motors.append(local.h)
            means.append(local.mean)
        return torch.stack(latents), torch.stack(motors), torch.stack(means), torch.stack(fluxes)
