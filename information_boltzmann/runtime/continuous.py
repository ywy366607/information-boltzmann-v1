"""One state owner; independent observations, evolution and action requests.

The synchronous engine is differentiable. The background runner is deployment
only and returns Futures. It neither drops observations nor rewinds committed
time. Current-state reads can bypass catch-up; their timestamp/lag is explicit.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
import heapq
import math
from queue import Full
import threading
import time
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from ..core.plastic_medium import EvolutionCoefficients, MediumState
from ..core.plastic_ports import PlasticBelief, PlasticMediumPorts3D
from ..core.temporal_probes import TemporalProbeState


@dataclass(frozen=True)
class ReadResponse:
    value: torch.Tensor
    state_time: float
    requested_time: float

    @property
    def lag(self) -> float:
        return max(0.0, self.requested_time - self.state_time)


class _EvolutionStep(nn.Module):
    def __init__(self, medium):
        super().__init__()
        self.medium = medium

    def forward(self, state, duration, prepared: EvolutionCoefficients):
        return self.medium.advance(state, duration, prepared=prepared, diagnostics=False)[0]


class _GraphEvolutionStep:
    """Fixed-shape deployment CUDA Graph, retaining dynamic dt/state inputs.

    Capture amortizes local PyTorch kernel launch overhead. All returned state
    tensors are owned copies; subsequent replays cannot mutate prior snapshots.
    Differentiable training uses _EvolutionStep rather than this inference graph.
    """

    def __init__(self, medium):
        self.kernel = _EvolutionStep(medium)
        self.graph = None
        self.signature = None

    @staticmethod
    def clone(state):
        return MediumState(state.field.clone(), tuple(x.clone() for x in state.flux),
                           state.elapsed.clone(),
                           None if state.conduction is None else state.conduction.clone(),
                           None if state.receptors is None else state.receptors.clone(),
                           None if state.transmission is None else state.transmission.clone())

    def copy_inputs(self, state, duration):
        for target, value in zip((self.inputs.field, *self.inputs.flux, self.inputs.elapsed),
                                 (state.field, *state.flux, state.elapsed)):
            target.copy_(value)
        if state.conduction is not None:
            self.inputs.conduction.copy_(state.conduction)
        if state.receptors is not None:
            self.inputs.receptors.copy_(state.receptors)
        if state.transmission is not None:
            self.inputs.transmission.copy_(state.transmission)
        self.duration.copy_(duration)

    def __call__(self, state, duration, prepared):
        if torch.is_grad_enabled() or not state.field.is_cuda:
            raise ValueError('cuda_graph backend requires CUDA inference without gradients')
        signature = (state.field.shape, state.field.dtype, state.field.device,
                     self.kernel.medium.execution_backend,
                     state.conduction is not None, state.receptors is not None,
                     state.transmission is not None, id(prepared), id(prepared.material),
                     id(prepared.baseline_log_speed))
        if self.graph is None or signature != self.signature:
            self.graph = None
            self.signature = signature
            # Keep captured coefficient storage alive. Otherwise allocator/
            # Python address reuse can make refreshed coefficients appear to
            # have the old signature and silently replay stale physics.
            self.prepared = prepared
            self.inputs = self.clone(state)
            self.duration = duration.clone()
            current = torch.cuda.current_stream(state.field.device)
            warm = torch.cuda.Stream(device=state.field.device)
            warm.wait_stream(current)
            with torch.cuda.stream(warm):
                for _ in range(3):
                    self.kernel(self.inputs, self.duration, prepared)
            current.wait_stream(warm)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = self.kernel(self.inputs, self.duration, prepared)
        self.copy_inputs(state, duration)
        self.graph.replay()
        return self.clone(self.output)


class ContinuousStream:
    """Differentiable timestamped single-individual stream.

    max_step is a declared solver accuracy budget in model time, not pondering
    duration. All events share the same persisted belief. No method detaches it
    implicitly. Only fixed-weight inference caches spatial coefficients/table.
    """

    def __init__(self, model: PlasticMediumPorts3D, *, max_step: float,
                 belief: PlasticBelief | None = None, cache_inference: bool = True,
                 compile_backend: str | None = None):
        if not math.isfinite(max_step) or max_step <= 0:
            raise ValueError('Positive finite solver max_step required')
        self.model = model
        self.belief = model.initial_belief() if belief is None else belief
        if self.belief.medium.field.shape[0] != 1:
            raise ValueError('Live individual requires batch size one')
        self.max_step = float(max_step)
        self.time = float(self.belief.medium.elapsed.detach().cpu()[0])
        if not math.isfinite(self.time):
            raise ValueError('Finite continuation time required')
        self.cache_inference = cache_inference
        self._stamp = None
        self._prepared = None
        self._token_features = None
        self.steps = 0
        self.coefficient_preparations = 0
        self.table_preparations = 0
        self._kernel = _EvolutionStep(model.medium)
        if compile_backend == 'cuda_graph':
            self._kernel = _GraphEvolutionStep(model.medium)
        elif compile_backend is not None:
            self._kernel = torch.compile(self._kernel, backend=compile_backend, fullgraph=True)

    def invalidate_cache(self) -> None:
        """Call after untracked .data edits; optimizer/no_grad edits track versions."""
        self._stamp = self._prepared = self._token_features = None

    def _check_cache(self) -> bool:
        if torch.is_grad_enabled() or not self.cache_inference:
            return False
        # Structural sampling changes buffers without changing a parameter.
        # Both are part of the physical coefficient cache's identity.
        from itertools import chain
        stamp = tuple((id(p), p._version, p.device, p.dtype)
                      for p in chain(self.model.parameters(), self.model.buffers()))
        if stamp != self._stamp:
            self.invalidate_cache()
            self._stamp = stamp
        return True

    def _coefficients(self) -> EvolutionCoefficients:
        cache = self._check_cache()
        if not cache or self._prepared is None:
            prepared = self.model.medium.prepare_evolution()
            self.coefficient_preparations += 1
            if cache:
                self._prepared = prepared
            return prepared
        return self._prepared

    def _features(self) -> torch.Tensor:
        cache = self._check_cache()
        if not cache or self._token_features is None:
            table = F.normalize(self.model.source.embedding.weight, dim=-1)
            self.table_preparations += 1
            if cache:
                self._token_features = table
            return table
        return self._token_features

    def advance_to(self, timestamp: float) -> None:
        """Evolve elapsed time including intervals without any new input."""
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < self.time:
            raise ValueError('Timestamp precedes committed time or is non-finite')
        if timestamp == self.time:
            return
        prepared = self._coefficients()
        # One shared coefficient graph within this differentiable interval.
        while self.time < timestamp:
            duration = min(self.max_step, timestamp - self.time)
            dt = torch.tensor(duration, device=self.belief.medium.field.device,
                              dtype=torch.float64)
            if self.model.solver_max_step is None:
                state = self._kernel(self.belief.medium, dt, prepared)
                self.belief = self.model.complete_advance(self.belief, state, dt, prepared=prepared)
            else:
                self.belief, _ = self.model.advance(self.belief, dt,
                                                   prepared=prepared, diagnostics=False)
            self.time = min(timestamp, self.time + duration)
            self.steps += 1

    def observe(self, timestamp: float, observed_ids: torch.Tensor, *,
                training_terms: bool = True) -> dict:
        if observed_ids.shape != (1,):
            raise ValueError('One observation id [1] required')
        self.advance_to(timestamp)
        ids = observed_ids.to(device=self.belief.medium.field.device, dtype=torch.long)
        self.belief, info = self.model.assimilate(
            self.belief, ids, token_features=self._features(), diagnostics=False,
            training_terms=training_terms)
        return info

    def read_current(self, *, requested_time: float | None = None,
                     decode: bool = True) -> ReadResponse:
        prepared = self._coefficients() if self.model.read_mode in ('dynamic', 'temporal') else None
        value, _ = self.model.read(self.belief, decode=decode, prepared=prepared)
        return ReadResponse(value, self.time, self.time if requested_time is None else requested_time)

    def read_at(self, timestamp: float, *, decode: bool = True) -> ReadResponse:
        self.advance_to(timestamp)
        return self.read_current(requested_time=timestamp, decode=decode)

    def observe_event(self, observed_ids: torch.Tensor, *, event_duration: float,
                      decode: bool = True, training_terms: bool = True):
        """The training event's causal clock, for a self-paced deployment stream.

        External timestamped observe/read remains available for continuously
        arriving sensors. This helper consumes one observation, evolves once
        over its chosen interval, and emits one response without resetting.
        """
        if observed_ids.shape != (1,):
            raise ValueError('One observation id [1] required')
        duration = self.model.event_time(self.belief, event_duration)
        self.belief, write = self.model.assimilate(self.belief,
            observed_ids.to(device=self.belief.medium.field.device, dtype=torch.long),
            token_features=self._features(), diagnostics=False, training_terms=training_terms)
        self.belief, _, motion = self.model.advance(self.belief, duration,
            prepared=self._coefficients(), diagnostics=False, return_motion=True)
        self.time = float(self.belief.medium.elapsed.detach().item())
        value, _ = self.model.read(self.belief, decode=decode,
                                  prepared=self._coefficients(), motion=motion)
        self.steps += 1
        return ReadResponse(value, self.time, self.time), write

    def interval_contract(self):
        clock = self.model.intrinsic_time
        return {'solver_max_step': self.model.solver_max_step,
                'observer_max_step': self.model.observer_max_step,
                'max_evolution_steps': self.model.max_evolution_steps,
                'intrinsic_reference': None if clock is None else float(clock.reference_duration),
                'intrinsic_max_duration': None if clock is None else clock.max_duration}

    def structure_contract(self):
        posterior = self.model.medium.structural_posterior
        if posterior is None:
            return None
        return {'law': 'gaussian-simplex-capacity-v1',
                'coefficient_count': posterior.coefficient_count,
                **{key: float(getattr(posterior, key)) for key in (
                    'resource_density', 'speed_reference', 'structure_time', 'maintenance_supply')},
                **{key: getattr(posterior, key).detach().cpu().tolist()
                   for key in ('stationary_mean', 'stationary_variance')}}

    def detach(self) -> None:
        """Explicit BPTT boundary: retain experience, structure and timestamps."""
        self.belief = self.belief.detach()
        self.invalidate_cache()

    def state_dict(self) -> dict:
        """Tensor-only continuation payload for torch.load(weights_only=True)."""
        state = self.belief.medium
        return {'schema': 9, 'time': self.time, 'max_step': self.max_step,
                'interval_contract': self.interval_contract(),
                'structure_contract': self.structure_contract(),
                'material_reference_shape': self.model.medium.material.reference_shape,
                'read_mode': self.model.read_mode,
                'read_time_reference': self.model.read_time_reference,
                'temporal_contract': ('sensor_trajectory_endpoint_hold_v1'
                                      if self.model.read_mode == 'temporal' else None),
                'medium_execution': self.model.medium.execution_backend,
                'anisotropic_transport': self.model.medium.anisotropic_transport,
                'transport_capacity_budget': self.model.medium.transport_capacity_limit,
                'port_execution': self.model.port_execution,
                'short_term_plasticity': self.model.short_term_plasticity,
                'activity_adaptation': self.model.activity_adaptation,
                'write_exchange': self.model.write_agent.exchange,
                'port_scope': self.model.port_scope,
                'write_port_radius': (self.model.write_agent.local_ports.physical_radius
                                      if self.model.port_scope == 'compact' else None),
                'read_port_radius': (self.model.readout.physical_radius
                                     if self.model.port_scope == 'compact' else None),
                'field': state.field.detach().clone(),
                'flux': tuple(x.detach().clone() for x in state.flux),
                'elapsed': state.elapsed.detach().clone(),
                'conduction': None if state.conduction is None else state.conduction.detach().clone(),
                'receptors': None if state.receptors is None else state.receptors.detach().clone(),
                'transmission': None if state.transmission is None else state.transmission.detach().clone(),
                'precision': self.belief.precision.detach().clone(),
                'temporal': None if self.belief.temporal is None else
                    {key: value.detach().clone()
                     for key, value in self.belief.temporal.state_dict().items()}}

    @classmethod
    def from_state_dict(cls, model: PlasticMediumPorts3D, saved: dict, **options):
        from ..core.plastic_medium import MediumState
        if saved.get('schema') not in (1, 2, 3, 4, 5, 6, 7, 8, 9):
            raise ValueError('Unsupported continuous-state schema')
        if saved.get('read_mode', 'instantaneous') != model.read_mode:
            raise ValueError('Checkpoint read mode differs; use an explicit architecture branch')
        if (model.read_mode in ('dynamic', 'temporal')
                and saved.get('read_time_reference') != model.read_time_reference):
            raise ValueError('Checkpoint read physical time reference differs')
        if saved.get('anisotropic_transport', False) != model.medium.anisotropic_transport:
            raise ValueError('Checkpoint propagation tensor differs; use an explicit branch')
        if saved.get('transport_capacity_budget') != model.medium.transport_capacity_limit:
            raise ValueError('Checkpoint structural capacity budget differs; use an explicit branch')
        if saved.get('short_term_plasticity', False) != model.short_term_plasticity:
            raise ValueError('Checkpoint STP law differs; use an explicit branch')
        if saved.get('activity_adaptation', False) != model.activity_adaptation:
            raise ValueError('Checkpoint activity feedback differs; use an explicit branch')
        if saved.get('write_exchange', 'global') != model.write_agent.exchange:
            raise ValueError('Checkpoint write exchange differs; changing the port law requires an explicit branch')
        if saved.get('port_scope', 'global') != model.port_scope:
            raise ValueError('Checkpoint port scope differs; spatial locality requires an explicit branch')
        if model.port_scope == 'compact':
            for key, radius in (('write_port_radius', model.write_agent.local_ports.physical_radius),
                                ('read_port_radius', model.readout.physical_radius)):
                if tuple(saved[key]) != tuple(radius):
                    raise ValueError('Checkpoint physical port radius differs')
        reference = next(model.parameters())
        def move(x):
            return x.to(device=reference.device, dtype=reference.dtype)
        state = MediumState(move(saved['field']), tuple(move(x) for x in saved['flux']),
                            saved['elapsed'].to(device=reference.device, dtype=torch.float64),
                            None if saved['conduction'] is None else move(saved['conduction']),
                            None if saved.get('receptors') is None else move(saved['receptors']),
                            None if saved.get('transmission') is None else move(saved['transmission']))
        model.medium.validate_transmission(state)
        if (state.receptors is None) != (model.medium.conductance_response is None):
            raise ValueError('Checkpoint receptor state differs from architecture')
        if state.receptors is not None:
            expected = (*state.field.shape[:-1], 2, model.medium.channels)
            if state.receptors.shape != expected:
                raise ValueError('Checkpoint receptor shape mismatch')
            torch._assert_async((torch.isfinite(state.receptors) & (state.receptors >= 0)
                                 & (state.receptors <= 1)).all(), 'Invalid checkpoint receptor fractions')
        model.medium.edge_log_speeds(state)
        temporal = None
        if model.read_mode == 'temporal':
            if saved.get('temporal_contract') != 'sensor_trajectory_endpoint_hold_v1':
                raise ValueError('Temporal sampling/history contract differs')
            if saved.get('temporal') is None:
                raise ValueError('Temporal checkpoint must retain history')
            complex_dtype = torch.complex128 if reference.dtype == torch.float64 else torch.complex64
            temporal = TemporalProbeState(
                saved['temporal']['value'].to(device=reference.device, dtype=complex_dtype).clone(),
                saved['temporal']['elapsed'].to(device=reference.device, dtype=torch.float64).clone())
            if temporal.value.shape != model.temporal_readout.bank.initial_state(state.field.shape[0]).value.shape:
                raise ValueError('Checkpoint temporal bank shape differs')
            torch.testing.assert_close(temporal.elapsed, state.elapsed, atol=1e-10, rtol=1e-10)
        elif saved.get('temporal') is not None:
            raise ValueError('Temporal checkpoint requires matching read architecture')
        stream = cls(model, max_step=saved['max_step'],
                     belief=PlasticBelief(state, move(saved['precision']), temporal), **options)
        if saved.get('structure_contract') != stream.structure_contract():
            raise ValueError('Continuation changes structural propagation law or its prior')
        if tuple(saved.get('material_reference_shape', model.medium.material.reference_shape)
                 ) != model.medium.material.reference_shape:
            raise ValueError('Continuation changes material bandwidth; use an explicit basis migration')
        if saved.get('interval_contract', stream.interval_contract() if model.solver_max_step is None
                     else None) != stream.interval_contract():
            raise ValueError('Continuation changes physical clock or numerical sampling contract')
        if not math.isclose(stream.time, saved['time'], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError('Continuation timestamp differs from saved physical time')
        return stream


@dataclass(order=True)
class _Event:
    timestamp: float
    sequence: int
    kind: str = field(compare=False)
    future: Future = field(compare=False)
    payload: object = field(compare=False, default=None)


class LiveStream:
    """Nonblocking producers, one serial causal state owner, continuous idle flow.

    Model weights must remain fixed while running. Current reads are served at
    the next safe update boundary, before further catch-up. Timestamped reads
    use exact event ordering. CPU submission never waits for model convergence.
    CUDA work completes in the owner before a Future is published.
    """

    def __init__(self, stream: ContinuousStream, *, model_time_per_second: float,
                 queue_capacity: int = 1024, clock: Callable[[], float] = time.monotonic):
        if not math.isfinite(model_time_per_second) or model_time_per_second <= 0:
            raise ValueError('Positive finite model_time_per_second required')
        if queue_capacity < 1:
            raise ValueError('Positive queue capacity required')
        if stream.model.training:
            raise ValueError('Call model.eval() before deployment; use ContinuousStream for training')
        self.stream = stream
        self.scale = float(model_time_per_second)
        self.capacity = queue_capacity
        self.clock = clock
        self._condition = threading.Condition()
        self._events: list[_Event] = []
        self._reads = deque()
        self._sequence = 0
        self._thread = None
        self._stopping = False
        self._origin_wall = None
        self._origin_model = stream.time
        self._failure = None
        self.max_lag = 0.0

    def now(self) -> float:
        if self._origin_wall is None:
            raise RuntimeError('Start live stream before submitting events')
        return self._origin_model + (self.clock() - self._origin_wall) * self.scale

    def start(self) -> "LiveStream":
        with self._condition:
            if self._thread is not None:
                raise RuntimeError('Live stream already started; resume via a new owner')
            self._origin_wall = self.clock()
            self._thread = threading.Thread(target=self._run, name='persistent-medium', daemon=True)
            self._thread.start()
        return self

    def _submit(self, kind: str, payload, timestamp: float | None) -> Future:
        with self._condition:
            if self._thread is None or self._stopping or self._failure is not None:
                raise RuntimeError('Live stream is not accepting requests')
            if len(self._events) + len(self._reads) >= self.capacity:
                raise Full('Queue saturated; retry or increase service throughput. No observation dropped.')
            current = self.now()
            when = current if timestamp is None else float(timestamp)
            if not math.isfinite(when):
                raise ValueError('Finite event timestamp required')
            future = Future()
            event = _Event(when, self._sequence, kind, future, payload)
            self._sequence += 1
            if kind == 'current_read':
                self._reads.append(event)
            else:
                heapq.heappush(self._events, event)
            self._condition.notify()
            return future

    def submit_observation(self, observed_ids: torch.Tensor, *, timestamp: float | None = None) -> Future:
        # Copy small producer-owned input so later caller mutation cannot change an event.
        if observed_ids.device.type != 'cpu' or observed_ids.shape != (1,):
            raise ValueError('Nonblocking producer supplies CPU observation ids [1]')
        return self._submit('observe', observed_ids.detach().clone(), timestamp)

    def request_read(self, *, timestamp: float | None = None, decode: bool = True) -> Future:
        return self._submit('current_read' if timestamp is None else 'read_at', decode, timestamp)

    def _complete(self, future: Future, result=None, exception=None):
        if self.stream.belief.medium.field.is_cuda:
            torch.cuda.current_stream(self.stream.belief.medium.field.device).synchronize()
        if future.cancelled():
            return
        if exception is not None:
            future.set_exception(exception)
        else:
            future.set_result(result)

    def _run(self):
        event = None
        try:
            with torch.inference_mode():
                while True:
                    event = None
                    with self._condition:
                        if self._stopping:
                            break
                        wall_target = self.now()
                        self.max_lag = max(self.max_lag, wall_target - self.stream.time)
                        if self._reads:
                            event = self._reads.popleft()
                        elif self._events and self._events[0].timestamp <= wall_target:
                            first = self._events[0]
                            if first.timestamp > self.stream.time + self.stream.max_step:
                                event = None
                                target = self.stream.time + self.stream.max_step
                            else:
                                event = heapq.heappop(self._events)
                        elif wall_target >= self.stream.time + self.stream.max_step:
                            event = None
                            target = self.stream.time + self.stream.max_step
                            if self._events:
                                target = min(target, self._events[0].timestamp)
                        else:
                            next_time = self.stream.time + self.stream.max_step
                            if self._events:
                                next_time = min(next_time, self._events[0].timestamp)
                            self._condition.wait(max(0.0, (next_time - wall_target) / self.scale))
                            continue
                    if event is None:
                        self.stream.advance_to(target)
                        if self.stream.belief.medium.field.is_cuda:
                            torch.cuda.current_stream(self.stream.belief.medium.field.device).synchronize()
                        continue
                    if not event.future.cancelled():
                        event.future.set_running_or_notify_cancel()
                    if event.kind == 'current_read':
                        self._complete(event.future, self.stream.read_current(
                            requested_time=event.timestamp, decode=event.payload))
                    elif event.timestamp < self.stream.time:
                        self._complete(event.future, exception=ValueError(
                            'Late event precedes committed state; no implicit rewind or reset'))
                    elif event.kind == 'observe':
                        self.stream.observe(event.timestamp, event.payload, training_terms=False)
                        self._complete(event.future, self.stream.time)
                    else:
                        self._complete(event.future, self.stream.read_at(event.timestamp, decode=event.payload))
        except BaseException as error:
            self._failure = error
        finally:
            with self._condition:
                pending = [*self._events, *self._reads]
                self._events.clear()
                self._reads.clear()
                self._stopping = True
            if event is not None and not event.future.done():
                pending.append(event)
            for event in pending:
                if not event.future.done():
                    event.future.set_exception(self._failure or RuntimeError('Live stream closed before event'))

    def close(self, timeout: float = 5.0) -> PlasticBelief:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError('State owner still finishing its current update; retry close')
        if self._failure is not None:
            raise RuntimeError('Live stream failed') from self._failure
        return self.stream.belief
