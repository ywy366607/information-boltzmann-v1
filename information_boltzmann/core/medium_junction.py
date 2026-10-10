"""Finite-state, capacity-paid flux junctions compiled from decorated CK forests.

The three existing edge-origin stores are physical coordinates, not copied waves.
A topology edit changes local skew couplings and leaves every stored coordinate
in place. Zero utilization is the exact baseline; no hidden residual is dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from .hopf_recomposition import RootedTree, canonical_forest, graft

PAIRS = ((0, 1), (0, 2), (1, 2))
# A fixed resource owner for each independent skew generator. Each paid row is
# used once; reversing/grafting a route cannot duplicate that resource account.
OWNERS = (0, 1, 2)


def default_junction_forest():
    # Potential couplings, all initially closed. Physical feedback is represented
    # by overlapping decorated atoms; their generators add, not their wave states.
    return canonical_forest(graft((RootedTree(f'q{b}'),), f'q{a}') for a, b in PAIRS)


def compile_junction_forest(forest: Iterable[RootedTree]):
    """Bind tree edges to actual flux pairs; reject duplicated resource owners.

    Singleton nodes preserve their stored modes but do not supply a coupling.
    Grafting and admissible cuts change the resulting mask. Canonical ordering
    only controls representation; overlapping physical generators need splitting.
    """
    mask = [0.0, 0.0, 0.0]
    def visit(tree):
        if tree.label not in ('q0', 'q1', 'q2'):
            raise ValueError('Junction nodes must bind to existing q0/q1/q2 stores')
        source = int(tree.label[1])
        for child in tree.children:
            if child.label not in ('q0', 'q1', 'q2'):
                raise ValueError('Junction nodes must bind to existing q0/q1/q2 stores')
            target = int(child.label[1])
            if source == target:
                raise ValueError('A junction edge must join distinct stored modes')
            pair = tuple(sorted((source, target)))
            index = PAIRS.index(pair)
            if mask[index]:
                raise ValueError('Repeated local coupling would double-charge a resource owner')
            mask[index] = 1.0 if source < target else -1.0
            visit(child)
    result = canonical_forest(forest)
    for tree in result:
        visit(tree)
    return result, tuple(mask)


def _encode(tree):
    return {'label': tree.label, 'children': [_encode(c) for c in tree.children]}


def _decode(value):
    return RootedTree(value['label'], tuple(_decode(c) for c in value['children']))


@dataclass(frozen=True)
class JunctionAllocation:
    factor: torch.Tensor
    rates: torch.Tensor  # [..., pair], radians per unit physical time
    signed_fraction: torch.Tensor
    full_row_capacity: torch.Tensor
    transport_row_capacity: torch.Tensor
    junction_row_capacity: torch.Tensor


class LocalFluxJunction(nn.Module):
    """Local state/material-conditioned routing with three persistent outputs.

    For owner a, theta=(pi/2)*tanh(logits), B'_a=cos(theta) B_a and
    omega_pair=sin(theta)*||B_a||/ell_a. The two squared coupling capacities sum
    to the original paid row norm squared, including at zero utilization. A skew map conserves wave
    energy; this does not certify a norm bound on its full learning Jacobian.
    """

    def __init__(self, material_width: int, shape: tuple[int, int, int]):
        super().__init__()
        if material_width < 1 or len(shape) != 3 or any(n < 2 for n in shape):
            raise ValueError('Positive material width and three spatial dimensions required')
        self.material_width = material_width
        self.reference_shape = tuple(shape)
        # A fixed physical constitutive length, not the current runtime spacing.
        self.register_buffer('inverse_length', torch.tensor(shape, dtype=torch.float64))
        self.gate = nn.Linear(material_width + 3, 3)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.register_buffer('route_mask', torch.zeros(3))
        self.set_topology(default_junction_forest())
        self.register_load_state_dict_pre_hook(self._validate_load)

    @staticmethod
    def validate_checkpoint(saved, prefix=''):
        """Pure validation before parent/child modules copy any stored tensors."""
        keys = tuple(prefix + name for name in ('_extra_state', 'route_mask', 'inverse_length'))
        if not any(key in saved for key in keys):
            return  # Explicit zero-birth adoption has no junction keys yet.
        if not all(key in saved for key in keys):
            raise ValueError('Incomplete saved physical junction structure')
        extra, mask, length = (saved[key] for key in keys)
        if not isinstance(extra, dict) or extra.get('version') != 1:
            raise ValueError('Unsupported physical junction topology version')
        _, compiled = compile_junction_forest(tuple(_decode(t) for t in extra['forest']))
        if not isinstance(mask, torch.Tensor) or mask.shape != (3,) or not torch.equal(
                mask, mask.new_tensor(compiled)):
            raise ValueError('Saved junction forest and route mask disagree')
        if (not isinstance(length, torch.Tensor) or length.shape != (3,) or
                not bool(torch.isfinite(length).all() and (length > 0).all())):
            raise ValueError('Junction physical lengths must be finite and positive')

    def _validate_load(self, module, saved, prefix, *args):
        self.validate_checkpoint(saved, prefix)

    def set_topology(self, forest):
        """A bounded operator edit; existing state is neither resized nor remapped."""
        forest, mask = compile_junction_forest(forest)
        self.forest = forest
        with torch.no_grad():
            self.route_mask.copy_(self.route_mask.new_tensor(mask))

    def get_extra_state(self):
        return {'version': 1, 'forest': [_encode(t) for t in self.forest]}

    def set_extra_state(self, state):
        if state.get('version') != 1:
            raise ValueError('Unsupported physical junction topology version')
        forest, mask = compile_junction_forest(tuple(_decode(t) for t in state['forest']))
        expected = self.route_mask.new_tensor(mask)
        if not torch.equal(expected, self.route_mask):
            raise ValueError('Saved junction forest and route mask disagree')
        if not bool(torch.isfinite(self.inverse_length).all() and (self.inverse_length > 0).all()):
            raise ValueError('Junction physical lengths must be finite and positive')
        self.forest = forest

    def material_logits(self, material):
        return F.linear(material, self.gate.weight[:, :self.material_width], self.gate.bias)

    @staticmethod
    def generator_rate_bound(allocation, runtime_shape):
        """Sum of the norms of six edge colours and three independent pairs.

        This bounds a frozen split-generator frequency, not the derivative of
        state-dependent gates nor a certified global integration error.
        """
        edge_weight = allocation.rates.new_tensor(runtime_shape) * (2 * math.sqrt(2.))
        # The largest transport and junction rates can occur at different sites.
        # Maximize separately before summing; max(a+b) can underestimate this
        # split-operator norm budget when their peaks are spatially separated.
        transport = allocation.transport_row_capacity.amax((1, 2, 3))
        junction = allocation.rates.abs().amax((1, 2, 3))
        return (edge_weight * transport + junction).sum(-1)

    def solver_rate_multiplier(self, runtime_shape, material):
        """Numerical budget relative to the old row-sum rotation-rate bound.

        Use cos(theta)<=1 and abs(sin(theta))<=sin(theta_max) separately,
        since the two operator maxima need not occur at the same spatial site.
        Bound the state's convex energy shares by the content-weight extrema.
        The untouched zero gate has multiplier one; the reserved bound then
        changes continuously with the gate's maximum possible angle. Integer
        quadrature still has its existing ceil boundaries and is held for VJP.
        """
        with torch.no_grad():
            logits = self.material_logits(material).abs().amax((0, 1, 2))
            logits = logits + self.gate.weight[:, self.material_width:].abs().amax(-1)
            angles = ((torch.pi / 2) * logits.tanh() * self.route_mask.abs()).cpu().tolist()
            length = self.inverse_length.detach().cpu().tolist()
        multiplier = 1.
        for angle, inverse, size in zip(angles, length, runtime_shape):
            ratio = inverse / (2 * math.sqrt(2.) * size)
            multiplier = max(multiplier, 1. + ratio * math.sin(angle))
        return multiplier

    def allocate(self, state, full_factor, *, material=None, material_logits=None, route_mask=None):
        if material_logits is None:
            if material is None:
                raise ValueError('Material or prepared material logits required')
            material_logits = self.material_logits(material)
        # Scale all three modes together before squaring. This retains relative
        # energy for finite large amplitudes, and the all-zero case stays zero.
        scale = torch.stack([q.abs().amax(-1) for q in state.flux], -1).amax(-1)
        scale = scale.clamp_min(torch.finfo(state.field.dtype).tiny)[..., None]
        energy = torch.stack([(q / scale).square().mean(-1) for q in state.flux], -1)
        # Bounded local shares, including the zero-flux case; no target or global
        # state enters this content-dependent gate.
        shares = energy / (energy.sum(-1, keepdim=True) + torch.finfo(energy.dtype).eps)
        logits = material_logits + F.linear(shares, self.gate.weight[:, self.material_width:])
        angle = (torch.pi / 2) * logits.tanh() * (self.route_mask if route_mask is None else route_mask)
        fraction = angle.sin()
        # Only suppress the tiny negative rounding of cos(pi/2) in FP32.
        transport_scale = angle.cos().clamp_min(0.)
        paid = torch.linalg.vector_norm(full_factor, dim=-1).expand_as(fraction)
        junction = paid * fraction.abs()
        factor = full_factor * transport_scale[..., :, None]
        transport = paid * transport_scale
        rates = paid * fraction * self.inverse_length.to(fraction)
        return JunctionAllocation(factor, rates, fraction, paid, transport, junction)

    @staticmethod
    def rotate(state, rates, duration):
        """Symmetric exact pair flows; inverse uses the same rates and -duration.

        Existing nonzero q modes are all used and retained. No assumption of a
        vacant child, new grid, state copying, or per-channel dense solve appears.
        """
        dt = torch.as_tensor(duration, device=state.field.device, dtype=state.field.dtype)
        if dt.numel() not in (1, state.field.shape[0]):
            raise ValueError('Duration must be scalar or one value per batch')
        dt = dt.reshape(-1, 1, 1, 1)
        modes = list(state.flux)
        for index, share in ((0, .5), (1, .5), (2, 1.), (1, .5), (0, .5)):
            a, b = PAIRS[index]
            angle = (rates[..., index] * dt * share)[..., None]
            cosine, sine = angle.cos(), angle.sin()
            left, right = modes[a], modes[b]
            modes[a] = cosine * left - sine * right
            modes[b] = sine * left + cosine * right
        from dataclasses import replace
        return replace(state, flux=tuple(modes))

    def descriptions(self):
        return {'protocol': 'capacity_paid_persistent_flux_junction_v2',
                'persistent_modes': 3, 'added_state_elements': 0,
                'inverse_physical_length': self.inverse_length.detach().cpu().tolist(),
                'forest': [_encode(t) for t in self.forest],
                'capacity_metric': 'squared_coupling_norm',
                'capacity': 'transport row squared plus local junction capacity squared equals paid row norm squared',
                'state': 'all existing field/flux and physical auxiliary stores retained',
                'initialization': 'exact zero utilization; unchanged baseline physics'}
