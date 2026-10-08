"""Sequence-length-independent online credit for the actual plastic medium.

UORO (Tallec & Ollivier, ICLR 2018, arXiv:1702.05043) compresses the
RTRL sensitivity ds/dtheta into rank-one factors. Each factor follows the
FULL local transition Jacobian, not an invented neuron eligibility formula.
There is one event's autograd graph and no historical activation tape.

Random compression is unbiased for frozen-parameter trajectory sensitivities;
online parameter changes have the usual RTRL interpretation. Finite rank has
sampling variance. This module does not claim exact cheap lifelong gradients.
"""
from __future__ import annotations

from dataclasses import replace
import math

import torch
from torch.autograd import forward_ad

from ..core.plastic_ports import PlasticBelief, PlasticMediumPorts3D
from .training import quiet_training_chunk


def credit_tensors(belief: PlasticBelief) -> tuple[torch.Tensor, ...]:
    """All learned/persistent degrees of freedom; exclude the exogenous clock."""
    state = belief.medium
    return tuple(t for t in (state.field, *state.flux, state.conduction,
                             state.receptors, state.transmission, belief.precision)
                 if t is not None)


def replace_credit(belief: PlasticBelief, tensors) -> PlasticBelief:
    values = iter(tensors)
    state = belief.medium
    result = replace(state, field=next(values), flux=tuple(next(values) for _ in state.flux),
                     conduction=next(values) if state.conduction is not None else None,
                     receptors=next(values) if state.receptors is not None else None,
                     transmission=next(values) if state.transmission is not None else None)
    precision = next(values)
    if next(values, None) is not None:
        raise ValueError('Excess credit-state tensors')
    return PlasticBelief(result, precision)


def squared_norm(tensors):
    return sum(t.square().sum() for t in tensors)


def balance(left, right):
    """Variance-minimizing factor balance; 1 is the neutral zero-factor gauge."""
    left_norm, right_norm = squared_norm(left).sqrt(), squared_norm(right).sqrt()
    tiny = torch.finfo(left_norm.dtype).tiny
    return torch.where((left_norm > 0) & (right_norm > 0),
                       (right_norm / left_norm.clamp_min(tiny)).sqrt(),
                       torch.ones_like(left_norm))


def merge_rank_one(propagated, old_parameter_factor, signs, injected):
    """E[(u' v'^T)] = Ju v^T + B, given E[signs signs^T] = I.

    ``injected`` is B^T signs. Balancing depends only on norms and is even
    under global sign reversal, so cross terms still have zero expectation.
    """
    rho_old = balance(propagated, old_parameter_factor)
    rho_new = balance(signs, injected)
    state_factor = tuple((rho_old * a + rho_new * b).detach()
                         for a, b in zip(propagated, signs))
    parameter_factor = tuple((a / rho_old + b / rho_new).detach()
                             for a, b in zip(old_parameter_factor, injected))
    return state_factor, parameter_factor


class OnlinePlasticTrainer:
    """Joint CE + existing W4 likelihood, no temporal BPTT truncation.

    A token advances exactly the original write/evolve/read chain. Parameter
    updates may be accumulated for throughput; this is an optimizer cadence,
    not a memory horizon. Belief AND eligibility survive every such update.
    Native execution is explicit because forward AD requires JVP-capable ops.
    """

    def __init__(self, model: PlasticMediumPorts3D, belief: PlasticBelief | None = None,
                 *, event_duration: float, substeps: int = 1, rank: int = 1,
                 seed: int = 449):
        if model.read_mode == 'temporal':
            raise ValueError('Temporal history uses the joint BPTT learner; legacy UORO state excludes it')
        if rank < 1 or substeps < 1 or not math.isfinite(event_duration) or event_duration <= 0:
            raise ValueError('Positive rank, substeps and finite event duration required')
        if model.medium.execution_backend != 'native' or model.port_execution != 'native':
            raise ValueError('Online JVP requires native execution; choose it explicitly')
        # The nested STP AOT backward wrapper has no forward-AD rule. Its
        # identical native expression supports both derivative directions.
        if model.short_term_plasticity:
            model.medium.short_term_plasticity.fuse_execution = False
        self.model = model
        self.belief = (model.initial_belief() if belief is None else belief).detach()
        if self.belief.medium.field.shape[0] != 1:
            raise ValueError('One persistent individual required')
        self.names, self.parameters = zip(*((n, p) for n, p in model.named_parameters()
                                           if p.requires_grad))
        self.event_duration, self.substeps, self.rank = float(event_duration), int(substeps), int(rank)
        self.duration = torch.tensor(self.event_duration,device=self.parameters[0].device,dtype=torch.float64)
        self.generator = torch.Generator(device=self.parameters[0].device).manual_seed(seed)
        self.state_factors = [tuple(torch.zeros_like(t) for t in credit_tensors(self.belief))
                              for _ in range(rank)]
        self.parameter_factors = [tuple(torch.zeros_like(p) for p in self.parameters)
                                  for _ in range(rank)]
        self.events = 0

    def _event(self, belief, observed, target):
        loss, output, nll = quiet_training_chunk(
            self.model, observed.reshape(1, 1), target.reshape(1, 1), belief,
            event_duration=self.duration, substeps=self.substeps,
            health_capture=getattr(self, 'health_capture', None))
        return loss, output, nll

    def backward_event(self, observed: torch.Tensor, target: torch.Tensor,
                       *, loss_scale: float = 1.0, check_finite: bool = True) -> dict[str, torch.Tensor]:
        """Accumulate all parameter gradients; free the one-event tape on return.

        Target affects the likelihood only. No reset, trace decay hyperparameter,
        historical graph, fixed credit window, or read-mask feedback shortcut.
        """
        if observed.numel() != 1 or target.numel() != 1 or not math.isfinite(loss_scale) or loss_scale <= 0:
            raise ValueError('One observation/target and positive finite loss scale required')
        leaves = tuple(t.detach().requires_grad_() for t in credit_tensors(self.belief))
        incoming = replace_credit(self.belief, leaves)
        propagated = []
        # Forward-mode directions cost rank local executions, never past events.
        for state_factor in self.state_factors:
            with forward_ad.dual_level():
                dual = tuple(forward_ad.make_dual(x, u) for x, u in zip(leaves, state_factor))
                loss_dual, result_dual, nll_dual = self._event(
                    replace_credit(incoming, dual), observed, target)
                pairs = [forward_ad.unpack_dual(t) for t in credit_tensors(result_dual)]
                propagated.append(tuple(torch.zeros_like(p.primal) if p.tangent is None
                                        else p.tangent.detach() for p in pairs))
                # Last primal graph also computes exact local parameter gradients.
                loss = forward_ad.unpack_dual(loss_dual).primal
                nll = forward_ad.unpack_dual(nll_dual).primal
                output = replace_credit(result_dual, tuple(p.primal for p in pairs))
                output = PlasticBelief(replace(output.medium,
                    elapsed=forward_ad.unpack_dual(output.medium.elapsed).primal), output.precision)
        if check_finite and not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError('Nonfinite online event loss')
        next_tensors = credit_tensors(output)
        gradients = torch.autograd.grad(loss, (*self.parameters, *leaves),
                                        allow_unused=True, retain_graph=True)
        direct = gradients[:len(self.parameters)]
        state_gradient = gradients[len(self.parameters):]
        local = tuple(torch.zeros_like(p) if g is None else g.detach()
                      for p, g in zip(self.parameters, direct))
        history = [torch.zeros_like(p) for p in self.parameters]
        # Use the PRE-event sensitivity for dl(s_old,theta)/dtheta.
        # The current event's direct gradient already includes its transition.
        for u, v in zip(self.state_factors, self.parameter_factors):
            credit = sum((g.detach() * direction).sum()
                         for g, direction in zip(state_gradient, u) if g is not None)
            for h, factor in zip(history, v):
                h.add_(factor * (credit / self.rank))
        new_state, new_parameter = [], []
        active = [(i, t) for i, t in enumerate(next_tensors) if t.requires_grad]
        for index in range(self.rank):
            signs = tuple(torch.empty_like(t).bernoulli_(0.5, generator=self.generator).mul_(2).sub_(1)
                          for t in next_tensors)
            injected = torch.autograd.grad(
                tuple(t for _, t in active), self.parameters,
                grad_outputs=tuple(signs[i] for i, _ in active), allow_unused=True,
                retain_graph=index < self.rank - 1)
            injected = tuple(torch.zeros_like(p) if g is None else g.detach()
                             for p, g in zip(self.parameters, injected))
            u, v = merge_rank_one(propagated[index], self.parameter_factors[index], signs, injected)
            new_state.append(u)
            new_parameter.append(v)
        combined = tuple((g + h) * loss_scale for g, h in zip(local, history))
        if check_finite and not bool(torch.stack([torch.isfinite(g).all() for g in combined]).all()):
            raise FloatingPointError('Nonfinite online gradient; event not committed')
        if check_finite and not bool(torch.stack([torch.isfinite(t).all()
                                for factors in (*new_state, *new_parameter) for t in factors]).all()):
            raise FloatingPointError('Nonfinite online eligibility; event not committed')
        with torch.no_grad():
            for p, gradient in zip(self.parameters, combined):
                if p.grad is None:
                    p.grad = gradient.clone()
                else:
                    p.grad.add_(gradient)
        self.belief = output.detach()
        self.state_factors, self.parameter_factors = new_state, new_parameter
        self.events += 1
        return {'loss': loss.detach(), 'token_nll': nll.detach(),
                'local_gradient_norm': squared_norm(local).sqrt().detach(),
                'history_gradient_norm': squared_norm(history).sqrt().detach()}

    def state_dict(self) -> dict:
        """Save eligibility and its randomness along with physical continuation."""
        from .training import clone_belief
        return {'schema': 1, 'names': list(self.names), 'rank': self.rank,
                'event_duration': self.event_duration, 'substeps': self.substeps,
                'events': self.events, 'generator': self.generator.get_state(),
                'belief': clone_belief(self.belief),
                'state_factors': [tuple(t.detach().clone() for t in u) for u in self.state_factors],
                'parameter_factors': [tuple(t.detach().clone() for t in v) for v in self.parameter_factors]}

    def load_state_dict(self, saved: dict):
        for key, current in (('schema', 1), ('names', list(self.names)), ('rank', self.rank),
                             ('event_duration', self.event_duration), ('substeps', self.substeps)):
            if saved[key] != current:
                raise ValueError(f'Online continuation mismatch: {key}')
        templates = (credit_tensors(self.belief), self.parameters)
        loaded = []
        for key, template in zip(('state_factors', 'parameter_factors'), templates):
            if len(saved[key]) != self.rank:
                raise ValueError(f'Wrong factor rank: {key}')
            group = []
            for factor in saved[key]:
                if len(factor) != len(template) or any(a.shape != b.shape for a, b in zip(factor, template)):
                    raise ValueError(f'Wrong factor shapes: {key}')
                group.append(tuple(a.detach().to(b).clone() for a, b in zip(factor, template)))
            loaded.append(group)
        belief = saved['belief']
        if len(credit_tensors(belief)) != len(templates[0]) or any(
                a.shape != b.shape for a, b in zip(credit_tensors(belief), templates[0])):
            raise ValueError('Online belief layout differs')
        values = tuple(a.detach().to(b).clone() for a, b in zip(credit_tensors(belief), templates[0]))
        self.belief = replace_credit(self.belief, values)
        self.belief = PlasticBelief(replace(self.belief.medium,
            elapsed=belief.medium.elapsed.detach().to(self.belief.medium.elapsed).clone()), self.belief.precision)
        self.state_factors, self.parameter_factors = loaded
        self.events = int(saved['events'])
        self.generator.set_state(saved['generator'].cpu())

    def eligibility_bytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for group in (self.state_factors, self.parameter_factors)
                   for factor in group for t in factor)


class CapturedOnlineEvent:
    """CUDA Graph for one event's JVP/VJPs and forward eligibility update.

    Fixed tape size, dynamic token/parameter/state/factor values. This changes
    launch scheduling only, not compression rank or learning equations. Finite
    checks occur at replay boundaries, outside stream capture.
    """

    def __init__(self, online: OnlinePlasticTrainer, *, loss_scale: float = 1.):
        from .training import belief_tensors, clone_belief
        if not online.parameters[0].is_cuda:
            raise ValueError('CUDA capture requires CUDA parameters')
        self.online = online
        self.ids = torch.zeros(1, dtype=torch.long, device=online.parameters[0].device)
        self.target = self.ids.clone()
        self.belief = clone_belief(online.belief)
        self.state_factors = [tuple(t.clone() for t in factor) for factor in online.state_factors]
        self.parameter_factors = [tuple(t.clone() for t in factor) for factor in online.parameter_factors]
        snapshot = online.state_dict()
        def restore():
            for destination, source in zip(belief_tensors(self.belief), belief_tensors(snapshot['belief'])):
                destination.copy_(source)
            for target_group, source_group in ((self.state_factors, snapshot['state_factors']),
                                               (self.parameter_factors, snapshot['parameter_factors'])):
                for destination, source in zip(target_group, source_group):
                    for d, s in zip(destination, source):
                        d.copy_(s)
            online.belief = self.belief
            online.state_factors = self.state_factors
            online.parameter_factors = self.parameter_factors
            online.events = snapshot['events']
            online.generator.set_state(snapshot['generator'])
        def operation():
            metrics = online.backward_event(self.ids, self.target,loss_scale=loss_scale,check_finite=False)
            with torch.no_grad():
                for destination, source in zip(belief_tensors(self.belief),belief_tensors(online.belief)):
                    destination.copy_(source)
                for target_group, source_group in ((self.state_factors,online.state_factors),
                                                   (self.parameter_factors,online.parameter_factors)):
                    for destination, source in zip(target_group,source_group):
                        for d,s in zip(destination,source):
                            d.copy_(s)
            online.belief = self.belief
            online.state_factors = self.state_factors
            online.parameter_factors = self.parameter_factors
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
            self.graph.register_generator_state(online.generator)
            with torch.cuda.graph(self.graph):
                self.metrics = operation()
            restore()
            online.model.zero_grad(set_to_none=False)
        except Exception:
            # A failed capture may invalidate the CUDA context; preserve the
            # original exception rather than masking it with a restore failure.
            try:
                online.load_state_dict(snapshot)
                online.model.zero_grad(set_to_none=True)
            except Exception:
                pass
            raise
        self.grad_storage = tuple(p.grad for p in online.parameters)

    def backward(self, observed, target):
        if self.online.belief is not self.belief or self.online.state_factors is not self.state_factors:
            raise RuntimeError('Online state storage changed; recapture after loading continuation')
        if any(p.grad is not g for p,g in zip(self.online.parameters,self.grad_storage)):
            raise RuntimeError('Captured gradient storage changed; zero_grad(set_to_none=False) is required')
        self.ids.copy_(observed.reshape(1))
        self.target.copy_(target.reshape(1))
        self.graph.replay()
        self.online.events += 1
        # These diagnostics reflect the unscaled local/history contributions.
        if not bool(torch.stack([torch.isfinite(v) for v in self.metrics.values()]).all()):
            raise FloatingPointError('Nonfinite captured online event; stop and retain last checkpoint')
        return {key:value.detach().clone() for key,value in self.metrics.items()}
