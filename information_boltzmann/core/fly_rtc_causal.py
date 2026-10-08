"""Finite-horizon exact motor response without lossy upstream compression.

The domain follows the real directed delay graph. Outside origin pulses are
known boundary data. Newly emitted excluded pulses cannot reach the motor
surface within the declared horizon. This is a physical forecast, not a learned
world model or a prediction of unknown future sensory events.
"""
from __future__ import annotations

from dataclasses import fields

import torch

from .fly_rtc_motor_flux import MotorLocalState, MotorPhysicalDomain


def motor_delay_distances(model, horizon: int) -> torch.Tensor:
    """Reverse shortest delay distances, exact for distances <= horizon.

    Every edge delay is at least one, so at most horizon relaxations suffice.
    Parallel edges use min, not a sparse-matrix duplicate-summing convention.
    The first new emission costs an extra tick; omitting it is conservative.
    """
    if horizon < 1:
        raise ValueError('Positive physical forecast horizon required')
    device = model.read_indices.device
    distance = torch.full((model.n_neurons,), horizon + 5,
                          dtype=torch.long, device=device)
    distance[model.read_indices] = 0
    edges = []
    for sign in ('e', 'i'):
        pre = getattr(model, 'edge_pre_' + sign).long()
        post = getattr(model, 'edge_post_' + sign).long()
        splits = getattr(model, 'splits_' + sign)
        lengths = torch.tensor([b-a for a,b in zip(splits[:-1], splits[1:])], device=device)
        delays = torch.repeat_interleave(torch.arange(1, 5, device=device), lengths)
        edges.append((pre, post, delays))
    for _ in range(horizon):
        previous = distance.clone()
        for pre, post, delays in edges:
            distance.scatter_reduce_(0, pre, previous[post]+delays,
                                     reduce='amin', include_self=True)
        if torch.equal(distance, previous):
            break
    return distance


def select_local(state: MotorLocalState, indices: torch.Tensor) -> MotorLocalState:
    return MotorLocalState(**{
        f.name: tuple(p[:, indices] for p in state.ring) if f.name == 'ring'
        else getattr(state, f.name)[:, indices] for f in fields(state)})


class CausalMotorForecaster(MotorPhysicalDomain):
    """Exact selected-surface quiet forecast at fixed physical parameters.

    Compile once for the original output surface and delay graph. A sensory
    event requires a new forecast from the updated physical origin. Forecast
    time is physical time; this class adds no learned clock or stopping rule.
    """

    def __init__(self, model, *, horizon=14, fused_transmission=False):
        distances = motor_delay_distances(model, horizon)
        domain = torch.nonzero(distances <= horizon).flatten()
        super().__init__(model, state_indices=domain)
        self.horizon = int(horizon)
        self.register_buffer('motor_indices', self.inverse[model.read_indices].clone())
        self.register_buffer('distances', distances)
        self.source_neurons = model.n_neurons
        self.source_edges = sum(len(getattr(model, 'edge_pre_'+s)) for s in ('e', 'i'))
        self.fused_transmission = bool(fused_transmission)
        if self.fused_transmission:
            rows, columns, values = [], [], []
            n = len(domain)
            for sign_index, sign in enumerate(('e', 'i')):
                splits = getattr(self, 'splits_'+sign)
                counts = torch.tensor([b-a for a,b in zip(splits[:-1], splits[1:])],
                                      device=domain.device)
                slots = torch.repeat_interleave(torch.arange(4, device=domain.device), counts)
                rows.append(sign_index*n+getattr(self, 'post_'+sign).long())
                columns.append(slots*n+getattr(self, 'pre_'+sign).long())
                values.append(getattr(self, 'weight_'+sign))
            operator = torch.sparse_coo_tensor(
                torch.stack((torch.cat(rows), torch.cat(columns))), torch.cat(values),
                size=(2*n, 4*n), device=domain.device).coalesce().to_sparse_csr()
            self.register_buffer('arrival_operator', operator)

    def transmit(self, ring):
        if not self.fused_transmission:
            return super().transmit(ring)
        # Fixed anatomical matrix; multiplication is differentiable with
        # respect to the dense predicted pulse slots. No dense N*N allocation.
        arrivals = torch.sparse.mm(self.arrival_operator, torch.cat(ring, -1).T).T
        n = len(self.indices)
        return [arrivals[:, :n], arrivals[:, n:]]

    def budget(self):
        return dict(horizon=self.horizon, source_neurons=self.source_neurons,
                    retained_neurons=len(self.indices),
                    retained_fraction=len(self.indices)/self.source_neurons,
                    source_edges=self.source_edges,
                    internal_edges=sum(len(getattr(self, 'pre_'+s)) for s in ('e', 'i')),
                    origin_boundary_edges=sum(len(getattr(self, 'external_pre_'+s)) for s in ('e', 'i')),
                    neural_parameters=0, fused_transmission=self.fused_transmission)

    def rollout(self, physical, coefficients, *, horizon=None):
        """Only origin state reaches prediction; no future-teacher argument."""
        length = self.horizon if horizon is None else int(horizon)
        if not 0 <= length <= self.horizon:
            raise ValueError('Forecast exceeds compiled causal horizon')
        current = self.local_state(physical)
        known = self.known_queue(physical)
        zero = torch.zeros_like(known[0])
        outputs = [select_local(current, self.motor_indices)]
        for tick in range(length):
            current = self.integrate(current, known[tick] if tick < 4 else zero, coefficients)
            outputs.append(select_local(current, self.motor_indices))
        return outputs
