"""Deterministic local receptor eligibility for the existing 3D medium.

This is an operator-local e-prop approximation, not a whole-system RTRL
estimator. It preserves the two receptor closing-rate sensitivities at every
site/channel. Voltage/current features act as observed external inputs to the
receptor subsystem. The current event receives exact joint autodiff gradients.
Historical feedback is the actual loss derivative at the incoming receptors,
not a random feedback matrix or an unsigned surprise scalar.
"""
from __future__ import annotations

from dataclasses import replace
import math

import torch

from .training import clone_belief, quiet_training_chunk


def receptor_partials(gates, opening, closing, dt, feedback=None, baseline=None):
    """Jacobian and forcing of the native frozen-feature kinetics flow.

    Return [..., D, 2, 2] matrices: output E/I versus input E/I, and output
    E/I versus physical closing_E/I. Include native I->E adaptation and the
    closing_I-dependent inhibition baseline. No invented eligibility decay.
    """
    rate = opening + closing[None]
    target = opening / rate
    decay = (-dt.unsqueeze(-1) * rate).exp()
    fraction = -torch.expm1(-dt.unsqueeze(-1) * rate)
    common = -dt.unsqueeze(-1) * decay * (gates - target)
    d_open = common + fraction * closing[None] / rate.square()
    d_close = common - fraction * opening / rate.square()
    zero = torch.zeros_like(decay[..., 0, :])
    feedback = zero if feedback is None else feedback
    baseline = zero if baseline is None else baseline
    jac = torch.stack((decay[..., 0, :], d_open[..., 0, :] * feedback,
                       zero, decay[..., 1, :]), -1).reshape(*zero.shape, 2, 2)
    injection = torch.stack((d_close[..., 0, :], d_open[..., 0, :] * baseline,
                             zero, d_close[..., 1, :]), -1).reshape(*zero.shape, 2, 2)
    return jac, injection


class LocalReceptorEligibility:
    """Fixed-size local sensitivities; both kinetic halves share the same trace."""

    def __init__(self, gates):
        self.trace = torch.zeros((*gates.shape[:-2], gates.shape[-1], 2, 2),
                                 device=gates.device, dtype=gates.dtype)
        self.kinetic_steps = 0

    @torch.no_grad()
    def observe(self, gates, opening, closing, dt, feedback, baseline):
        jac, injection = receptor_partials(gates, opening, closing, dt, feedback, baseline)
        self.trace.copy_(jac @ self.trace + injection)
        self.kinetic_steps += 1

    def closing_feedback(self, learning_signal):
        # Signal arrives at incoming gate state: [B,X,Y,Z,2,D]. Sum batch,
        # retain location/content/branch identity until the material pullback.
        signal = learning_signal.transpose(-2, -1)
        return torch.einsum('bxyzdg,bxyzdgc->xyzcd', signal, self.trace)

    def nbytes(self):
        return self.trace.numel() * self.trace.element_size()


class LocalPlasticTrainer:
    """One-event exact joint learning plus persistent local receptor credit.

    This first migration extends delayed credit for closing kinetics and their
    continuous material map. Historical voltage, transport, collision, STP,
    write and precision sensitivities remain outside this approximation; those
    parameters still learn from the exact current-event graph. Physical state
    never resets. Traces persist through optimizer boundaries and checkpoints.
    """

    def __init__(self, model, belief=None, *, event_duration, substeps=1):
        if model.read_mode == 'temporal':
            raise ValueError('Temporal history uses the joint BPTT learner; local receptor traces exclude it')
        if not math.isfinite(event_duration) or event_duration <= 0 or substeps < 1:
            raise ValueError('Positive physical duration and solver resolution required')
        if model.medium.conductance_response is None:
            raise ValueError('Local receptor credit requires the conductance medium')
        if model.medium.execution_backend != 'native' or model.port_execution != 'native':
            raise ValueError('Use native execution for the eligibility observer')
        self.model = model
        self.belief = (model.initial_belief() if belief is None else belief).detach()
        if self.belief.medium.field.shape[0] != 1:
            raise ValueError('One persistent individual required')
        self.event_duration, self.substeps = float(event_duration), int(substeps)
        self.duration = torch.tensor(event_duration, device=self.belief.medium.field.device,
                                     dtype=torch.float64)
        self.names, self.parameters = zip(*((n, p) for n, p in model.named_parameters()
                                           if p.requires_grad))
        self.eligibility = LocalReceptorEligibility(self.belief.medium.receptors)
        self.events = 0
        if model.short_term_plasticity:
            model.medium.short_term_plasticity.fuse_execution = False

    def eligibility_bytes(self):
        return self.eligibility.nbytes()

    def backward_event(self, observed, target, *, loss_scale=1.0, check_finite=True):
        if observed.numel() != 1 or target.numel() != 1 or not math.isfinite(loss_scale) or loss_scale <= 0:
            raise ValueError('One event and positive finite loss scale required')
        response = self.model.medium.conductance_response
        if getattr(response, '_local_credit_observer', None) is not None:
            raise RuntimeError('Eligibility observer already owned by another learner')
        gates = self.belief.medium.receptors.detach().requires_grad_()
        incoming = replace(self.belief, medium=replace(self.belief.medium, receptors=gates))
        old_trace = self.eligibility.trace.clone()
        old_steps = self.eligibility.kinetic_steps
        response._local_credit_observer = self.eligibility.observe
        try:
            loss, output, nll = quiet_training_chunk(
                self.model, observed.reshape(1, 1), target.reshape(1, 1), incoming,
                event_duration=self.duration, substeps=self.substeps,
                health_capture=getattr(self, 'health_capture', None))
            direct_and_signal = torch.autograd.grad(
                loss * loss_scale, (*self.parameters, gates), allow_unused=True)
            signal = direct_and_signal[-1]
            if signal is None:
                signal = torch.zeros_like(gates)
            new_trace = self.eligibility.trace.clone()
            # Pre-event eligibility only. The exact event gradient already
            # includes all of this event's receptor forcing, including both halves.
            self.eligibility.trace.copy_(old_trace)
            feedback = self.eligibility.closing_feedback(signal.detach())
            # Pull local coefficient credit into the existing continuous
            # material/coefficient network. No extra trainable module is added.
            closing = self.model.medium.prepare_evolution().response.closing
            history = torch.autograd.grad(closing, self.parameters,
                                          grad_outputs=feedback, allow_unused=True)
            # Commit recorded trace only after both backward operations succeed.
            # Observer has already accumulated the prospective trace separately.
            if check_finite:
                finite = torch.isfinite(loss) & torch.isfinite(new_trace).all()
                for values in (direct_and_signal[:-1], history):
                    for value in values:
                        if value is not None:
                            finite = finite & torch.isfinite(value).all()
                if not bool(finite):
                    raise RuntimeError('Non-finite local online loss, trace, or gradient')
            for parameter, direct, extra in zip(self.parameters, direct_and_signal[:-1], history):
                if direct is None and extra is None:
                    # Preserve ordinary autodiff/AdamW semantics for a parameter
                    # outside this graph, rather than inventing a zero gradient
                    # that would enable decoupled weight decay on unused weights.
                    continue
                grad = (torch.zeros_like(parameter) if direct is None else direct)
                if extra is not None:
                    grad = grad + extra
                if parameter.grad is None:
                    parameter.grad = grad.detach().clone()
                else:
                    parameter.grad.add_(grad.detach())
            self.eligibility.trace.copy_(new_trace)
            self.belief = output.detach()
            self.events += 1
            historical_norm = sum(g.square().sum() for g in history if g is not None).sqrt()
            return {'loss': loss.detach(), 'token_nll': nll.detach(),
                    'port_objective': (loss - nll).detach(),
                    'history_gradient_norm': historical_norm.detach(),
                    'eligibility_rms': new_trace.square().mean().sqrt()}
        except Exception:
            self.eligibility.trace.copy_(old_trace)
            self.eligibility.kinetic_steps = old_steps
            raise
        finally:
            del response._local_credit_observer

    def state_dict(self):
        return {'version': 1, 'belief': clone_belief(self.belief),
                'trace': self.eligibility.trace.detach().clone(), 'events': self.events,
                'kinetic_steps': self.eligibility.kinetic_steps,
                'event_duration': self.event_duration, 'substeps': self.substeps,
                'names': self.names}

    def load_state_dict(self, saved):
        if (saved['version'] != 1 or tuple(saved['names']) != self.names
                or saved['event_duration'] != self.event_duration or saved['substeps'] != self.substeps
                or saved['trace'].shape != self.eligibility.trace.shape):
            raise ValueError('Local credit continuation layout/cadence mismatch')
        self.belief = clone_belief(saved['belief']).detach()
        self.eligibility.trace.copy_(saved['trace'].to(self.eligibility.trace))
        self.events = saved['events']
        self.eligibility.kinetic_steps = saved['kinetic_steps']


class CapturedLocalEvent:
    """Capture joint event gradients and deterministic local eligibility.

    Capture/warmup restores physical and learning state; replay refreshes all
    token, parameter and trace values. Grad buffers must retain their addresses.
    """

    def __init__(self, online, *, loss_scale=1.0):
        from .training import belief_tensors
        if not online.parameters[0].is_cuda:
            raise ValueError('CUDA parameters required')
        self.online = online
        self.ids = torch.zeros(1, dtype=torch.long, device=online.parameters[0].device)
        self.target = self.ids.clone()
        self.belief = clone_belief(online.belief)
        snapshot = online.state_dict()
        trace = online.eligibility.trace

        def restore():
            for dst, src in zip(belief_tensors(self.belief), belief_tensors(snapshot['belief'])):
                dst.copy_(src)
            trace.copy_(snapshot['trace'])
            online.belief = self.belief
            online.events = snapshot['events']
            online.eligibility.kinetic_steps = snapshot['kinetic_steps']

        def operation():
            metrics = online.backward_event(self.ids, self.target,
                                             loss_scale=loss_scale, check_finite=False)
            with torch.no_grad():
                for dst, src in zip(belief_tensors(self.belief), belief_tensors(online.belief)):
                    dst.copy_(src)
            online.belief = self.belief
            return metrics

        current = torch.cuda.current_stream()
        warm = torch.cuda.Stream()
        warm.wait_stream(current)
        try:
            with torch.cuda.stream(warm):
                for _ in range(2):
                    restore()
                    online.model.zero_grad(set_to_none=True)
                    operation()
            current.wait_stream(warm)
            torch.cuda.synchronize()
            restore()
            online.model.zero_grad(set_to_none=False)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.metrics = operation()
            restore()
            online.model.zero_grad(set_to_none=False)
        except Exception:
            try:
                online.load_state_dict(snapshot)
                online.model.zero_grad(set_to_none=True)
            except Exception:
                pass
            raise
        self.trace = trace
        self.grad_storage = tuple(p.grad for p in online.parameters)

    def backward(self, observed, target):
        if self.online.belief is not self.belief or self.online.eligibility.trace is not self.trace:
            raise RuntimeError('Online storage changed; recapture after continuation loading')
        if any(p.grad is not g for p, g in zip(self.online.parameters, self.grad_storage)):
            raise RuntimeError('Use zero_grad(set_to_none=False) with capture')
        self.ids.copy_(observed.reshape(1))
        self.target.copy_(target.reshape(1))
        self.graph.replay()
        self.online.events += 1
        self.online.eligibility.kinetic_steps += 2 * self.online.substeps
        if not bool(torch.stack([torch.isfinite(v) for v in self.metrics.values()]).all()):
            raise FloatingPointError('Nonfinite local event; stop and retain last checkpoint')
        return {key: value.detach().clone() for key, value in self.metrics.items()}
