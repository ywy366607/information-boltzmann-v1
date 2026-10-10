"""State-continuous local recomposition for the third-generation medium.

The Hopf objects here are a structural ledger: they describe admissible
branching and decomposition.  ``WaveRecomposer`` is the physical interface
that carries the existing field/flux state through a local two-branch edit.
It uses a fixed two-mode budget; it never allocates a hidden new grid or
duplicates the persistent auxiliary state.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True, order=True)
class RootedTree:
    """An unordered decorated rooted tree used by the commutative CK algebra."""

    label: str
    children: tuple["RootedTree", ...] = ()

    def __post_init__(self):
        if not self.label:
            raise ValueError("A rooted-tree label must be non-empty")
        children = tuple(sorted(self.children, key=tree_key))
        object.__setattr__(self, "children", children)


Forest = tuple[RootedTree, ...]


def tree_key(tree: RootedTree):
    return (tree.label, tuple(tree_key(child) for child in tree.children))


def canonical_forest(forest: Iterable[RootedTree]) -> Forest:
    return tuple(sorted(tuple(forest), key=tree_key))


@dataclass(frozen=True)
class CutTerm:
    """One formal CK coproduct term: pruned forest tensor remainder forest."""

    pruned: Forest
    remainder: Forest

    def key(self):
        return (self.pruned, self.remainder)


def graft(forest: Forest, label: str = "root") -> RootedTree:
    """The CK B_+ grafting operator."""
    return RootedTree(label, canonical_forest(forest))


def _cut_options(tree: RootedTree):
    """All edge cuts inside ``tree``, including the empty cut.

    Each result is (pruned forest, retained tree, has_cut).  Directly cutting a
    child edge ends that branch, while recursive cuts remain independent on
    sibling branches.  This enforces at most one cut per root-to-leaf path.
    """
    states = [((), (), False)]  # pruned, retained children, has_cut
    for child in tree.children:
        # Direct cut plus all descendant cuts.
        # Cutting the parent -> child edge prunes the complete child subtree.
        child_options = [((child,), None, True)] + _cut_options(child)
        folded = []
        for forest, retained_children, used in states:
            for child_pruned, child_retained, child_used in child_options:
                folded.append((forest + tuple(child_pruned),
                               retained_children if child_retained is None
                               else retained_children + (child_retained,),
                               used or child_used))
        states = folded
    return [(canonical_forest(pruned),
             RootedTree(tree.label, canonical_forest(retained_children)), used)
            for pruned, retained_children, used in states]


def admissible_cuts(tree: RootedTree) -> tuple[tuple[Forest, RootedTree], ...]:
    """Return proper admissible cuts as (pruned forest, retained tree)."""
    result = []
    for pruned, retained, used in _cut_options(tree):
        if used:
            result.append((pruned, retained))
    return tuple(result)


def coproduct_tree(tree: RootedTree) -> tuple[CutTerm, ...]:
    """CK coproduct of a rooted tree, including the two unit terms."""
    terms = [CutTerm((), (tree,)), CutTerm((tree,), ())]
    terms.extend(CutTerm(pruned, (retained,))
                 for pruned, retained in admissible_cuts(tree))
    return tuple(terms)


def _multiply_terms(left: Iterable[CutTerm], right: Iterable[CutTerm]):
    return [CutTerm(canonical_forest(a.pruned + b.pruned),
                    canonical_forest(a.remainder + b.remainder))
            for a in left for b in right]


def coproduct(forest: Forest | RootedTree) -> tuple[CutTerm, ...]:
    """Algebra morphism extension of the tree coproduct to forests."""
    if isinstance(forest, RootedTree):
        forest = (forest,)
    terms = [CutTerm((), ())]
    for tree in canonical_forest(forest):
        terms = _multiply_terms(terms, coproduct_tree(tree))
    counts = Counter(term.key() for term in terms)
    expanded = []
    for (pruned, remainder), multiplicity in sorted(counts.items(),
                                                    key=lambda item: item[0]):
        expanded.extend(CutTerm(pruned, remainder) for _ in range(multiplicity))
    return tuple(expanded)


def coproduct_counter(forest: Forest | RootedTree):
    return Counter(term.key() for term in coproduct(forest))


def tensor_coproduct(term: CutTerm) -> Counter:
    """Apply Delta to both tensor factors, used for coassociativity tests."""
    result = Counter()
    for left in coproduct(term.pruned):
        for right in coproduct(term.remainder):
            # ((P1 tensor R1) tensor (P2 tensor R2)) is represented as
            # P1, R1, P2, R2 in the canonical three-factor comparison below.
            result[(left.pruned, left.remainder,
                    right.pruned, right.remainder)] += 1
    return result


def iterated_coproduct_counters(forest: Forest | RootedTree):
    """Canonical three-factor counters for (Delta tensor id)Delta and vice versa."""
    left = Counter()
    right = Counter()
    for term in coproduct(forest):
        for first in coproduct(term.pruned):
            left[(first.pruned, first.remainder, term.remainder)] += 1
        for second in coproduct(term.remainder):
            right[(term.pruned, second.pruned, second.remainder)] += 1
    return left, right


@dataclass(frozen=True)
class WaveResidual:
    field: torch.Tensor
    flux: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    def energy(self, volume: float = 1.0):
        return volume * 0.5 * (self.field.square().flatten(1).sum(1)
                               + sum(x.square().flatten(1).sum(1) for x in self.flux))


@dataclass(frozen=True)
class SplitWave:
    left: torch.Tensor
    right: torch.Tensor
    left_capacity: torch.Tensor
    right_capacity: torch.Tensor


@dataclass(frozen=True)
class MergeWave:
    main: torch.Tensor
    residual: torch.Tensor


class WaveRecomposer:
    """Fixed-budget orthogonal branch/merge maps for packed f,q coordinates.

    ``split`` and ``merge`` operate on a packed last dimension.  The caller
    keeps auxiliary receptor/conduction/STP tensors as one shared medium state;
    they are not amplitude-scaled or silently duplicated here.
    """

    def __init__(self, *, epsilon: float = 1e-4):
        if not math.isfinite(epsilon) or not 0 < epsilon < .5:
            raise ValueError("epsilon must lie in (0, 0.5)")
        self.epsilon = float(epsilon)

    def fraction(self, theta: torch.Tensor):
        return self.epsilon + (1 - 2 * self.epsilon) * theta.sigmoid()

    def split(self, state: torch.Tensor, p: torch.Tensor | float) -> SplitWave:
        p = torch.as_tensor(p, device=state.device, dtype=state.dtype)
        p = p.clamp(self.epsilon, 1 - self.epsilon)
        left, right = p.sqrt() * state, (1 - p).sqrt() * state
        return SplitWave(left, right, p, 1 - p)

    def merge(self, left: torch.Tensor, right: torch.Tensor,
              p: torch.Tensor | float) -> MergeWave:
        if left.shape != right.shape:
            raise ValueError("Branch wave shapes must match")
        p = torch.as_tensor(p, device=left.device, dtype=left.dtype)
        p = p.clamp(self.epsilon, 1 - self.epsilon)
        main = p.sqrt() * left + (1 - p).sqrt() * right
        residual = (1 - p).sqrt() * left - p.sqrt() * right
        return MergeWave(main, residual)

    def unmerge(self, main: torch.Tensor, residual: torch.Tensor,
                p: torch.Tensor | float):
        result = self.merge(main, residual, p)
        # The 2x2 map is symmetric and involutive.
        return result.main, result.residual

    @staticmethod
    def pack_wave(field: torch.Tensor,
                  flux: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        if len(flux) != 3 or any(x.shape != field.shape for x in flux):
            raise ValueError("field and all three flux tensors must have equal shapes")
        return torch.cat((field, *flux), dim=-1)

    @staticmethod
    def unpack_wave(packed: torch.Tensor, channels: int):
        if packed.shape[-1] != 4 * channels:
            raise ValueError("Packed wave last dimension must be 4*channels")
        return (packed[..., :channels],
                tuple(packed[..., channels * (i + 1):channels * (i + 2)]
                      for i in range(3)))

    def split_medium_state(self, state, p: torch.Tensor | float):
        """Split f/q while keeping auxiliary variables shared and untouched."""
        from .plastic_medium import MediumState
        packed = self.pack_wave(state.field, state.flux)
        branch = self.split(packed, p)
        left_field, left_flux = self.unpack_wave(branch.left, state.field.shape[-1])
        right_field, right_flux = self.unpack_wave(branch.right, state.field.shape[-1])
        return (replace(state, field=left_field, flux=left_flux),
                replace(state, field=right_field, flux=right_flux),
                branch.left_capacity, branch.right_capacity)

    def merge_medium_states(self, left, right, p: torch.Tensor | float):
        if left.field.shape != right.field.shape:
            raise ValueError("Branch medium fields must have equal shapes")
        packed_left = self.pack_wave(left.field, left.flux)
        packed_right = self.pack_wave(right.field, right.flux)
        result = self.merge(packed_left, packed_right, p)
        main_field, main_flux = self.unpack_wave(result.main, left.field.shape[-1])
        residual_field, residual_flux = self.unpack_wave(result.residual, left.field.shape[-1])
        main = replace(left, field=main_field, flux=main_flux)
        residual = WaveResidual(residual_field, residual_flux)
        return main, residual


class PairedBranchGenerator:
    """Fixed-budget skew block generator for two learned route modules."""

    def __init__(self, first: torch.Tensor, second: torch.Tensor,
                 coupling: torch.Tensor | None = None):
        if first.ndim != 2 or second.shape != first.shape:
            raise ValueError("Branch generators must be square matrices of equal shape")
        if not torch.compiler.is_compiling():
            if not torch.allclose(first + first.transpose(-1, -2),
                                  torch.zeros_like(first), atol=1e-6, rtol=1e-6):
                raise ValueError("First branch generator must be skew-symmetric")
            if not torch.allclose(second + second.transpose(-1, -2),
                                  torch.zeros_like(second), atol=1e-6, rtol=1e-6):
                raise ValueError("Second branch generator must be skew-symmetric")
        if coupling is None:
            coupling = torch.zeros_like(first)
        if coupling.shape != first.shape:
            raise ValueError("Coupling must have the branch matrix shape")
        self.first, self.second, self.coupling = first, second, coupling

    def matrix(self):
        n = self.first.shape[-1]
        result = self.first.new_zeros(2 * n, 2 * n)
        result[:n, :n] = self.first
        result[n:, n:] = self.second
        result[:n, n:] = self.coupling
        result[n:, :n] = -self.coupling.transpose(-1, -2)
        return result

    def apply(self, state: torch.Tensor):
        return state @ self.matrix().transpose(-1, -2)


def conjugate_generator(generator: torch.Tensor, orthogonal: torch.Tensor):
    """Pure coordinate change A' = Q A Q^T; used as a null control."""
    if generator.shape[-2:] != orthogonal.shape[-2:]:
        raise ValueError("Generator and coordinate map dimensions must match")
    return orthogonal @ generator @ orthogonal.transpose(-1, -2)


class LocalHopfBranchPathway(nn.Module):
    """A trainable 2-mode branching pathway module with exact energy conservation.

    Input wave state s is split into two branches via unitary beam splitter:
        (s, 0) -> (s_left, s_right) = (sqrt(p) s, sqrt(1-p) s)
    Each branch evolves under its own skew generator (Cayley unitary rotation),
    and branches optionally exchange energy through skew cross-coupling C.
    The children merge into main output and residual r:
        a = sqrt(p) s_left + sqrt(1-p) s_right
        r = sqrt(1-p) s_left - sqrt(p) s_right
    Energy conservation holds identically: ||s_in||^2 = ||a||^2 + ||r||^2.
    """

    def __init__(self, channels: int, *, epsilon: float = 1e-4, coupled: bool = True):
        super().__init__()
        self.channels = channels
        self.recomposer = WaveRecomposer(epsilon=epsilon)
        self.theta = nn.Parameter(torch.zeros(()))  # initial 50/50 split
        self.weight_left = nn.Parameter(torch.randn(channels, channels) * 0.01)
        self.weight_right = nn.Parameter(torch.randn(channels, channels) * 0.01)
        self.coupling = nn.Parameter(torch.zeros(channels, channels)) if coupled else None

    def skew_generator(self):
        left = self.weight_left - self.weight_left.T
        right = self.weight_right - self.weight_right.T
        c = (self.coupling if self.coupling is not None else None)
        return PairedBranchGenerator(left, right, c)

    def fraction(self):
        return self.recomposer.fraction(self.theta)

    def cayley_step(self, generator_matrix: torch.Tensor, duration: float | torch.Tensor):
        """Exact unitary Cayley transform: U = (I - tau*A/2)^{-1} (I + tau*A/2)."""
        dim = generator_matrix.shape[-1]
        eye = torch.eye(dim, dtype=generator_matrix.dtype, device=generator_matrix.device)
        half_step = 0.5 * duration * generator_matrix
        lhs = eye - half_step
        rhs = eye + half_step
        return torch.linalg.solve(lhs, rhs)

    def forward(self, state: torch.Tensor, duration: float | torch.Tensor = 1.0):
        """Forward pass preserving exact quadratic wave energy."""
        p = self.fraction()
        split = self.recomposer.split(state, p)
        # Combine branches into paired coordinate [..., 2*channels]
        paired = torch.cat((split.left, split.right), dim=-1)
        gen = self.skew_generator().matrix()
        unitary = self.cayley_step(gen, duration)
        evolved_paired = paired @ unitary.T
        dim = self.channels
        evolved_left = evolved_paired[..., :dim]
        evolved_right = evolved_paired[..., dim:]
        merged = self.recomposer.merge(evolved_left, evolved_right, p)
        return merged.main, merged.residual

    def tree_representation(self, label: str = "pathway") -> RootedTree:
        """CK rooted tree representation of this branching pathway."""
        child_left = RootedTree(f"{label}_left")
        child_right = RootedTree(f"{label}_right")
        return graft((child_left, child_right), label)
