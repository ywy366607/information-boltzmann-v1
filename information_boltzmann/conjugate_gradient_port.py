"""Local gradient-work port for the existing packed material carrier.

This is a numerical reference component, not an admitted training optimizer.
Decoded parameters are offset + scale*(p+q+w). Exact rational forces represent
gradient*scale/energy_unit; they are supplied observations, not memory scores.
The readout-fiber executor declares a separate local proposal clock; its
certificates do not inherit the native executor's time bounds.
The same fields that decode parameters participate in their physical repair.
"""
from __future__ import annotations

from copy import copy
from fractions import Fraction
from bisect import bisect_right
import math
import random
import struct
from typing import Callable

import torch

from .core.cgb_local_medium import LocalMedium, OpenMedium

FULL_LIFELONG_ADMITTED = False
KINETIC_CLOCK = "maximal_reverse_pair_v1"
ACCEPTANCE_SAMPLER = "rational_geometric_mixture_v1"
READOUT_FIBER_CLOCK = "readout_fiber_v1"

# Shared finite geometry, independent of the number of parameters. Adjacent
# levels contain at most13 states; no edge-index or per-parameter table.
_READOUT_FIBERS = tuple(tuple((p, q, w) for p in range(3) for q in range(3)
                               for w in range(3) if p + q + w == level)
                        for level in range(7))

# Exact same-mass minima for the registered quadratic H, proved by
# readout_floor_reduction. These eight tuples are shared immutable geometry.
_READOUT_MINIMA = (((0, 0, 0),), ((0, 0, 1),), ((0, 0, 2),),
                   ((1, 1, 1),), ((1, 1, 2),),
                   ((1, 2, 2), (2, 1, 2)), ((2, 2, 2),))


class _RuntimeRandom(random.Random):
    """Account for proposal rejection sampling as well as acceptance bits."""

    def __new__(cls, seed, ledger):
        return super().__new__(cls, seed)

    def __init__(self, seed, ledger):
        self.ledger = ledger
        super().__init__(seed)

    def getrandbits(self, count):
        self.ledger["random_bits"] += count
        return super().getrandbits(count)


def encode_gradient_force(gradient: Fraction, scale: Fraction,
                          energy_unit: Fraction = Fraction(1)) -> int:
    """Legacy rounding diagnostic, NOT used by the optimizer's force port.

    Retained to reproduce the old dead band. Nearest integer has error <=1/2,
    but erases nonzero small forces, so that bound alone is insufficient.
    """
    if any(not isinstance(v, (int, Fraction)) or isinstance(v, bool)
           for v in (gradient, scale, energy_unit)):
        raise TypeError("force encoding requires explicit rational inputs")
    if scale <= 0 or energy_unit <= 0:
        raise ValueError("scale and energy_unit must be positive")
    value = Fraction(gradient) * scale / energy_unit + Fraction(1, 2)
    return value.numerator // value.denominator


def fractional_probability_bounds(power: Fraction, digits: int):
    """Historical root enclosure; not used by the active acceptance sampler.

    Binary exponent digits select repeated square-root factors. Every root
    and product rounding is enclosed by exact integer inequalities. The
    unexpanded tail uses 1-tail <= 2**(-tail) <=1 (Lean proved). No floating
    exponential or probability rounding, nor fixed force threshold, is used.
    All work here is transient; no per-parameter history is stored.
    """
    power = Fraction(power)
    if not 0 <= power < 1 or type(digits) is not int or digits < 1:
        raise ValueError("requires fractional power in [0,1) and positive precision")
    if power == 0:
        return Fraction(1), Fraction(1)
    unit = 1 << (2 * digits + 8)
    root_lo = root_hi = unit // 2
    product_lo = product_hi = unit
    remainder, denominator, head = power.numerator, power.denominator, 0
    for _ in range(digits):
        old_lo, old_hi = root_lo, root_hi
        root_lo = math.isqrt(old_lo * unit)
        root_hi = min(unit, math.isqrt(old_hi * unit) + 1)
        assert root_lo * root_lo <= old_lo * unit
        assert old_hi * unit <= root_hi * root_hi
        remainder *= 2
        bit, remainder = divmod(remainder, denominator)
        head = 2 * head + bit
        if bit:
            old_lo, old_hi = product_lo, product_hi
            product_lo = old_lo * root_lo // unit
            product_hi = (old_hi * root_hi + unit - 1) // unit
            assert product_lo * unit <= old_lo * root_lo
            assert old_hi * root_hi <= product_hi * unit
    tail = Fraction(remainder, denominator * (1 << digits))
    assert Fraction(head, 1 << digits) + tail == power
    low = Fraction(product_lo, unit) * (1 - tail)
    high = Fraction(product_hi, unit)
    assert 0 <= low <= high <= 1
    assert high - low <= Fraction(2, 1 << digits)
    return low, high


def parameter_value(medium: OpenMedium, edge: int, scale: Fraction,
                    offset: Fraction = Fraction(0)) -> Fraction:
    """A readout of existing state; no protected shadow parameter is stored."""
    if any(not isinstance(v, (int, Fraction)) or isinstance(v, bool) for v in (scale, offset)):
        raise TypeError("parameter decoding requires explicit rational inputs")
    if scale <= 0:
        raise ValueError("scale must be positive")
    return Fraction(offset) + Fraction(scale) * sum(medium.body.get(edge))


class _EditedBody(LocalMedium):
    """Read-only <=2-edge successor overlay, never a copy of the body arrays."""

    def __init__(self, body: LocalMedium, edits):
        self.law, self._base, self._edits = body.law, body, edits

    def get(self, edge):
        for changed, state in self._edits:
            if changed == edge:
                return state
        return self._base.get(edge)

    def total(self, channel, index):
        value = self._base.total(channel, index)
        for edge, state in self._edits:
            location = edge // self.law.labels if channel == 0 else edge % self.law.labels
            if location == index:
                value += state[channel] - self._base.get(edge)[channel]
        return value


def _prepare_gradient_event(medium: OpenMedium, index: int, label: int,
                            force: Callable[[int], int | Fraction], *, maximal_clock: bool):
    """Return the source protocol's (edits, exponent, bound, material_sign).

    Exact dyadic acceptance is 2**(exponent-bound). Both ends of a transfer
    contribute task work. The reverse source exponent is read on a bounded
    successor overlay: source exponents are not assumed to be opposites.
    The caller owns frozen force observations, stochastic tickets and payment;
    this function only prepares a proposal and never mutates the medium.
    """
    edits, exponent, bound, sign = medium.prepare(index, label)
    if exponent is None:
        return edits, exponent, bound, sign
    work = 0
    for edge, state in edits:
        change = sum(state) - sum(medium.body.get(edge))
        if change == 0:
            continue
        coefficient = force(edge)
        if not isinstance(coefficient, (int, Fraction)) or isinstance(coefficient, bool):
            raise TypeError("force callback must return exact rational energy coefficients")
        work += coefficient * change
    if work == 0 and not maximal_clock:
        return edits, exponent, bound, sign
    after = copy(medium)
    after.body = _EditedBody(medium.body, edits)
    after.stock -= sign
    _, reverse_exponent, reverse_bound, _ = after.prepare(medium.reverse(index), label)
    if reverse_exponent is None or reverse_bound != bound:
        raise RuntimeError("source reverse-event contract is violated")
    forward = exponent - work
    reverse = reverse_exponent + work
    paired_bound = max(forward, reverse)
    if not maximal_clock:
        paired_bound = max(bound, paired_bound)
    return edits, forward, paired_bound, sign


def prepare_source_gradient_event(medium: OpenMedium, index: int, label: int,
                                  force: Callable[[int], int | Fraction]):
    """Historical source-clock v2 reference for explicitly scoped audits."""
    return _prepare_gradient_event(medium, index, label, force, maximal_clock=False)


def prepare_gradient_event(medium: OpenMedium, index: int, label: int,
                           force: Callable[[int], int | Fraction]):
    """Lean-derived minimal reverse-pair ceiling, without any state mutation.

    Its rate is 2**min(0, source_forward-source_reverse-2*task_work).
    This maximizes reversible acceptance for the same proposal and energy;
    retention depends on the carrier/drive conditions checked separately.
    """
    return _prepare_gradient_event(medium, index, label, force, maximal_clock=True)


def prepare_readout_fiber(medium: OpenMedium, edge: int, direction: int,
                         force: Callable[[int], int | Fraction]):
    """Prepare the Lean-checked adjacent-readout block, without any mutation.

    Returns (state, rejection_power, material_delta) for each uniform candidate.
    Conditional weights use the SAME global physical energy and one frozen
    task-force observation. Subtracting the block minimum gives a maximal
    common uniform-proposal envelope, without evaluating a partition function.
    This changes kinetics; it is not an exact trajectory acceleration of native
    reactions. Internal changes within a readout level are permitted.
    """
    law = medium.body.law
    if law.charge_cap != 2 or law.slot_cap != 2 or medium.reservoir.stimulus:
        raise ValueError("readout-fiber reference requires cap2 and uniform boundary drive")
    if type(direction) is not int or direction not in (-1, 1):
        raise ValueError("readout direction must be -1 or +1")
    old = medium.body.get(edge)
    level = sum(old)
    if not 0 <= level + direction <= 6 or (direction == 1 and medium.stock == 0):
        return ()
    coefficient = force(edge)
    if not isinstance(coefficient, (int, Fraction)) or isinstance(coefficient, bool):
        raise TypeError("force callback must return an exact rational energy coefficient")
    lower = min(level, level + direction)
    states = _READOUT_FIBERS[lower] + _READOUT_FIBERS[lower + 1]
    row, col = divmod(edge, law.labels)
    row_total, col_total = medium.body.total(0, row), medium.body.total(1, col)
    old_energy = (medium.body.unit_energy(old) + law.pool_energy(0, row_total)
                  + law.pool_energy(1, col_total))
    chemical = medium.reservoir.content_cost - medium.reservoir.chemical_base
    energies = tuple(medium.body.unit_energy(s)
                     + law.pool_energy(0, row_total + s[0] - old[0])
                     + law.pool_energy(1, col_total + s[1] - old[1]) - old_energy
                     + (chemical + coefficient) * (sum(s) - level) for s in states)
    floor = min(energies)
    return tuple((s, Fraction(2 * (energy - floor)), sum(s) - level)
                 for s, energy in zip(states, energies))


def _prepare_reduced_fiber(medium, edge, direction, force, choose):
    """Exact registered-H choice with <=4 local energy evaluations.

    Candidate order and all proposal states match prepare_readout_fiber.
    Only the floor calculation is reduced; no probability is approximated.
    The caller must check the registered immutable law before using this path.
    """
    law = medium.body.law
    old = medium.body.get(edge)
    level = sum(old)
    if not 0 <= level + direction <= 6 or (direction == 1 and medium.stock == 0):
        return None, 0, 0, 0
    coefficient = force(edge)
    if not isinstance(coefficient, (int, Fraction)) or isinstance(coefficient, bool):
        raise TypeError("force callback must return an exact rational energy coefficient")
    lower = min(level, level + direction)
    states = _READOUT_FIBERS[lower] + _READOUT_FIBERS[lower + 1]
    row, col = divmod(edge, law.labels)
    outside_p = medium.body.total(0, row) - old[0]
    outside_q = medium.body.total(1, col) - old[1]
    chemical = medium.reservoir.content_cost - medium.reservoir.chemical_base

    def energy(state):
        return (medium.body.unit_energy(state)
                + law.pool_energy(0, outside_p + state[0])
                + law.pool_energy(1, outside_q + state[1])
                + (chemical + coefficient) * sum(state))

    minima = _READOUT_MINIMA[lower] + _READOUT_MINIMA[lower + 1]
    energies = {state: energy(state) for state in minima}
    floor = min(energies.values())
    state = states[choose(len(states))]
    if state not in energies:
        energies[state] = energy(state)
    return state, Fraction(2 * (energies[state] - floor)), sum(state) - level, len(energies)


class MaterialOptimizer(torch.optim.Optimizer):
    """CPU numerical optimizer reference with content in the existing medium.

    The supplied medium OWNS the weights; parameters are its floating readout
    cache and are initialized from it. This is deliberately not an exact
    continuation of arbitrary preexisting floating weights. ``quantum`` fixes
    representation precision. The medium declares interactions, shared material
    supply and the finite attempts_per_step via opportunities_per_tick.
    ``lr`` scales the external gradient force; ``quantum`` sets actual write
    granularity. At lr=0 or absent gradient, physical dynamics still proceeds.
    Events use the Lean-derived maximal reversible pair clock. Local content
    regeneration additionally needs the declared carrier/drive conditions;
    the clock alone does not confer healthy retention on an arbitrary medium.

    Only numerical/interface execution is admitted. General graph retention,
    useful differentiation, floating/noisy-force bounds, real data and GPU
    efficiency remain open. The cycle optimizer's descent/GPU proofs do not
    transfer to this different physical law.
    """

    checkpoint_version = 4
    kinetic_clock = KINETIC_CLOCK
    reaction_fee_factor = 1

    def __init__(self, params, medium: OpenMedium, *, quantum: Fraction,
                 offset: Fraction = Fraction(0), energy_unit: Fraction = Fraction(1),
                 lr: float = 1.0, seed: int = 0):
        if not isinstance(medium, OpenMedium) or medium.reservoir.stimulus != 0:
            raise ValueError("requires a registered medium with no byte-selective drive")
        encode_gradient_force(Fraction(0), quantum, energy_unit)
        if not isinstance(offset, (int, Fraction)) or isinstance(offset, bool):
            raise TypeError("offset must be rational")
        super().__init__(params, {"lr": lr})
        self.medium = medium
        self.quantum, self.offset, self.energy_unit = map(Fraction, (quantum, offset, energy_unit))
        self.costs = dict(attempts=0, accepted=0, fee_units=0, random_bits=0, refused=0,
                          coefficient_steps=0, max_rate_precision=0)
        self._rng = _RuntimeRandom(seed, self.costs)
        self._layout()
        self._materialize(range(self.medium.body.law.edges))

    def _layout(self):
        self._ends, self._entries = [], []
        storage_ranges = []
        checked_dtypes = set()
        size = 0
        largest = max(abs(float(self.offset)), abs(float(self.offset + self.quantum *
                      (2 * self.medium.body.law.charge_cap + self.medium.body.law.slot_cap))))
        for group in self.param_groups:
            if set(group) != {"params", "lr"}:
                raise ValueError("only the lr parameter-group option is supported")
            if not math.isfinite(group["lr"]) or group["lr"] < 0:
                raise ValueError("lr must be finite and nonnegative")
            for p in group["params"]:
                if (p.device.type != "cpu" or p.dtype not in (torch.float32, torch.float64)
                        or not p.is_contiguous()):
                    raise ValueError("numerical reference requires contiguous CPU FP32/FP64 parameters")
                if not math.isfinite(largest) or largest > torch.finfo(p.dtype).max:
                    raise ValueError("decoded parameter range exceeds floating cache")
                if p.dtype not in checked_dtypes:
                    levels = 2 * self.medium.body.law.charge_cap + self.medium.body.law.slot_cap
                    for n in range(levels + 1):
                        exact = self.offset + self.quantum * n
                        rounded = float(exact)
                        if p.dtype == torch.float32:
                            rounded = struct.unpack("f", struct.pack("f", rounded))[0]
                        if Fraction.from_float(rounded) != exact:
                            raise ValueError("finite parameter alphabet must be exactly representable in cache")
                        if rounded and abs(rounded) < torch.finfo(p.dtype).tiny:
                            raise ValueError("subnormal parameter cache values are outside the reference contract")
                    checked_dtypes.add(p.dtype)
                size += p.numel()
                if p.numel():
                    storage_ranges.append((p.data_ptr(), p.data_ptr() + p.numel() * p.element_size()))
                self._ends.append(size)
                self._entries.append((p, group))
        if size != self.medium.body.law.edges:
            raise ValueError("one material cell is required per decoded scalar parameter")
        self._check_force_units(self.param_groups)
        storage_ranges.sort()
        if any(right[0] < left[1] for left, right in zip(storage_ranges, storage_ranges[1:])):
            raise ValueError("decoded parameter storages must not overlap")

    @staticmethod
    def _check_force_units(groups):
        # A shared conserved transfer reads a gradient DIFFERENCE. Different
        # gradient-unit multipliers could reverse its true task-work sign.
        if len({float(group["lr"]) for group in groups}) != 1:
            raise ValueError("coupled material cells require a common gradient-force scale; local mobility remains state-dependent")

    def add_param_group(self, param_group):
        if hasattr(self, "medium"):
            raise ValueError("the registered material layout has a fixed parameter set")
        return super().add_param_group(param_group)

    def _locate(self, edge):
        entry = bisect_right(self._ends, edge)
        p, group = self._entries[entry]
        return p, group, edge - (self._ends[entry - 1] if entry else 0)

    @torch.no_grad()
    def _materialize(self, edges):
        for edge in edges:
            p, _, index = self._locate(edge)
            p.view(-1)[index] = float(parameter_value(self.medium, edge, self.quantum, self.offset))

    def _force(self, edge):
        p, group, index = self._locate(edge)
        if p.grad is None:
            return 0
        gradient = Fraction.from_float(float(p.grad.view(-1)[index]))
        return gradient * Fraction.from_float(float(group["lr"])) * self.quantum / self.energy_unit

    @torch.no_grad()
    def apply_external_loss(self, edge: int, field: int) -> bool:
        """Export one existing quantum and refresh its owned parameter cache.

        The external world selects the event and accounts for its work/RNG.
        Export returns to the existing material stock and release ledger;
        this method adds no optimizer attempt, protection rule or state array.
        A zero field is a no-op. Invalid indices fail before any mutation.
        """
        if type(field) is not int or not 0 <= field < 3:
            raise ValueError("invalid external-loss field")
        state = self.medium.body.get(edge)
        if state[field] == 0:
            return False
        changed = list(state)
        changed[field] -= 1
        self.medium._commit(((edge, tuple(changed)),), -1)
        self._materialize((edge,))
        return True

    def _zero_ticket(self, bits):
        """Exact 2**(-bits) coin with <=32-bit workspace and early rejection."""
        consumed = 0
        while bits:
            take = min(bits, 32)
            word = self._rng.getrandbits(take)
            consumed += take
            if word:
                return False, consumed
            bits -= take
        return True, consumed

    def _rational_ticket(self, probability):
        """Exact rational coin from fresh fair bits, without growing a prefix.

        The residual numerator stays in [0,denominator]. At most one of two
        next bits can leave a nonterminal residual. The Lean finite-prefix
        reference proves the requested probability, a.s. termination and at
        most two requested bits in expectation. This counts logical bits;
        the Python PRNG's word generation and CPU work are separate costs.
        """
        probability = Fraction(probability)
        if not 0 <= probability <= 1:
            raise ValueError("rational probability must be in [0,1]")
        numerator, denominator = probability.numerator, probability.denominator
        used = 0
        self.costs["max_rate_precision"] = max(
            self.costs["max_rate_precision"], denominator.bit_length())
        while 0 < numerator < denominator:
            bit = self._rng.getrandbits(1)
            used += 1
            numerator *= 2
            if bit:
                if numerator <= denominator:
                    return False, used
                numerator -= denominator
            elif denominator <= numerator:
                return True, used
        return numerator == denominator, used

    def _probability_ticket(self, power):
        """Exact 2**(-power) via the Lean-proved rational geometric mixture.

        For fractional x, c_0=x and c_(n+1)=c_n*(n+1-x)/(n+2).
        Draw N with P(N=n)=2^(-n-1), reject with rational probability c_N.
        The binomial identity gives rejection probability 1-2**(-x).
        Expected fractional work is two geometric bits, <=two rational-coin
        bits, and one coefficient update; no roots or precision cutoff.
        Integer exponents retain the bounded-chunk early-rejection path.
        """
        power = Fraction(power)
        if power < 0:
            raise ValueError("negative rejection power")
        integer = power.numerator // power.denominator
        accepted, used = self._zero_ticket(integer)
        tail = power - integer
        if not accepted or tail == 0:
            return accepted, used
        coefficient, order = tail, 1
        while True:
            bit = self._rng.getrandbits(1)
            used += 1
            if bit:
                break
            coefficient *= (order - tail) / (order + 1)
            order += 1
            self.costs["coefficient_steps"] += 1
            self.costs["max_rate_precision"] = max(
                self.costs["max_rate_precision"], coefficient.denominator.bit_length())
        rejected, rational_bits = self._rational_ticket(coefficient)
        return not rejected, used + rational_bits

    @torch.no_grad()
    def step(self, closure=None, *, pay_write=None, before_attempt=None):
        """Run a finite paid window; optional funding port owns its own state.

        With PrefixReplayWorld.pay_write, publication/maintenance remain with
        the caller and denied funding ends this window before any random draw
        or material update. Without a port this is counted executor-funded
        work, not a certificate of a finite-world energy supply. An optional
        caller-owned before_attempt() runs after funding and before proposal
        sampling; it may apply the proved external loss through the public
        method. Its RNG/state/work belong to the world's checkpoint. Gradient
        observations must remain fixed unless the caller explicitly supplies
        a different admissible force process; this is not a task-risk claim.
        """
        if before_attempt is not None and not callable(before_attempt):
            raise TypeError("before_attempt must be callable")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Gradient observations must stay frozen while readout caches change.
        # These are per-tensor byte ranges, not copies of parameter/gradient data.
        cache_ranges = sorted((p.data_ptr(), p.data_ptr() + p.numel() * p.element_size())
                              for p, _ in self._entries if p.numel())
        cache_starts = [start for start, _ in cache_ranges]
        for p, group in self._entries:
            if not math.isfinite(group["lr"]) or group["lr"] < 0:
                raise ValueError("lr must be finite and nonnegative")
            if p.grad is not None:
                if p.grad.is_sparse or not p.grad.is_contiguous():
                    raise ValueError("requires contiguous dense gradients")
                if p.grad.numel():
                    begin = p.grad.data_ptr()
                    end = begin + p.grad.numel() * p.grad.element_size()
                    other = bisect_right(cache_starts, end - 1) - 1
                    if other >= 0 and cache_ranges[other][1] > begin:
                        raise ValueError("gradient observations must not alias parameter caches")
                lo, hi = torch.aminmax(p.grad) if p.grad.numel() else (0, 0)
                if not math.isfinite(float(lo)) or not math.isfinite(float(hi)):
                    raise ValueError("nonfinite gradient")
        self._check_force_units(self.param_groups)
        for _ in range(self.medium.opportunities_per_tick):
            if pay_write is not None:
                funded = pay_write(self.reaction_fee_factor * self.medium.body.law.attempt_fee)
                if type(funded) is not bool:
                    raise TypeError("funding port must return bool")
                if not funded:
                    self.costs["refused"] += 1
                    break
            self.costs["attempts"] += 1
            self.costs["fee_units"] += self.reaction_fee_factor * self.medium.body.law.attempt_fee
            if before_attempt is not None:
                before_attempt()
            edits, exponent, bound, sign = self._prepare_attempt()
            if exponent is None:
                continue
            accepted, _ = self._probability_ticket(bound - exponent)
            if accepted:
                self.medium._commit(edits, sign)
                self.costs["accepted"] += 1
                self._materialize(edge for edge, _ in edits)
        return loss

    def _prepare_attempt(self):
        index = self.medium.sample_proposal(self._rng)
        return prepare_gradient_event(self.medium, index, 0, self._force)

    def state_dict(self):
        """Contains physical weights and RNG; floating cache is reconstructible."""
        result = super().state_dict()
        result["material_optimizer"] = dict(
            version=self.checkpoint_version, clock=self.kinetic_clock, sampler=ACCEPTANCE_SAMPLER,
            medium=self.medium.state_dict(), rng=self._rng.getstate(),
            quantum=(self.quantum.numerator, self.quantum.denominator),
            offset=(self.offset.numerator, self.offset.denominator),
            energy_unit=(self.energy_unit.numerator, self.energy_unit.denominator),
            costs=dict(self.costs))
        return result

    def load_state_dict(self, state_dict):
        """Restore physical weights too, then rebuild their floating cache."""
        meta = state_dict["material_optimizer"]
        if (meta["version"] != self.checkpoint_version or meta.get("clock") != self.kinetic_clock
                or meta.get("sampler") != ACCEPTANCE_SAMPLER or state_dict["state"]):
            raise ValueError("unsupported material optimizer checkpoint")
        groups = state_dict["param_groups"]
        if len(groups) != len(self.param_groups):
            raise ValueError("checkpoint parameter groups differ")
        for loaded, current in zip(groups, self.param_groups):
            if (set(loaded) != {"params", "lr"} or len(loaded["params"]) != len(current["params"])
                    or not math.isfinite(loaded["lr"]) or loaded["lr"] < 0):
                raise ValueError("invalid checkpoint parameter groups")
        self._check_force_units(groups)
        for name in ("quantum", "offset", "energy_unit"):
            value = getattr(self, name)
            if tuple(meta[name]) != (value.numerator, value.denominator):
                raise ValueError("checkpoint parameter representation differs")
        if (set(meta["costs"]) != set(self.costs)
                or any(type(v) is not int or v < 0 for v in meta["costs"].values())):
            raise ValueError("invalid cost ledger")
        costs = dict(meta["costs"])
        if (costs["accepted"] > costs["attempts"] or
                costs["fee_units"] != costs["attempts"] * self.reaction_fee_factor *
                self.medium.body.law.attempt_fee):
            raise ValueError("inconsistent cost ledger")
        # Shallow successor wrapper: load_state_dict constructs its own packed
        # body; the original live medium is untouched on validation failure.
        restored = copy(self.medium)
        restored.load_state_dict(meta["medium"])
        rng = _RuntimeRandom(0, costs)
        rng.setstate(meta["rng"])
        super().load_state_dict({"state": {}, "param_groups": state_dict["param_groups"]})
        self.medium, self._rng, self.costs = restored, rng, costs
        self._layout()
        self._materialize(range(self.medium.body.law.edges))


class FiberMaterialOptimizer(MaterialOptimizer):
    """CPU numerical reference for the Lean-derived adjacent-readout kernel.

    A selected cell/direction proposes uniformly from at most13 physical
    states, then uses the existing exact coin. One global energy-evaluation
    counter is added; physical storage and parameter readout are unchanged.
    The registered quadratic H computes the exact floor from <=3 states and
    the chosen candidate with <=4 total evaluations. Other cap2 laws retain
    enumeration. Every attempt is still charged13 base units, a declared operation
    tariff rather than measured joules. Blocked/self attempts are paid too.

    Valid Markov/reversibility contracts apply to the cap2 family. Regeneration
    and force bounds require the separately registered scaled quadratic H.
    Native-clock time certificates do not apply. Real-model execution is still
    closed pending the joint functional and practical admission checks.
    """

    checkpoint_version = 5
    kinetic_clock = READOUT_FIBER_CLOCK
    reaction_fee_factor = 13

    def __init__(self, params, medium: OpenMedium, **kwargs):
        if medium.body.law.charge_cap != 2 or medium.body.law.slot_cap != 2:
            raise ValueError("readout-fiber reference requires cap2")
        super().__init__(params, medium, **kwargs)
        law = medium.body.law
        self._reduced_floor_law = (law if law.epsilon == 0 and law.coupling == 12
            and law.pool_reference_size == 2
            and all(value == n*n for n, value in enumerate(law.potential)) else None)
        self.costs["candidate_energy_evals"] = 0

    def _prepare_attempt(self):
        edge = self._rng.randrange(self.medium.body.law.edges)
        direction = 1 if self._rng.randrange(2) else -1
        if self.medium.body.law is self._reduced_floor_law:
            state, power, sign, count = _prepare_reduced_fiber(
                self.medium, edge, direction, self._force, self._rng.randrange)
            self.costs["candidate_energy_evals"] += count
            if state is None:
                return (), None, 0, 0
            return ((edge, state),), -power, 0, sign
        candidates = prepare_readout_fiber(self.medium, edge, direction, self._force)
        self.costs["candidate_energy_evals"] += len(candidates)
        if not candidates:
            return (), None, 0, 0
        state, power, sign = candidates[self._rng.randrange(len(candidates))]
        return ((edge, state),), -power, 0, sign

    def load_state_dict(self, state_dict):
        costs = state_dict["material_optimizer"]["costs"]
        count, attempts = costs.get("candidate_energy_evals"), costs.get("attempts")
        if (type(count) is not int or type(attempts) is not int
                or not 0 <= count <= 13 * attempts):
            raise ValueError("invalid candidate evaluation ledger")
        return super().load_state_dict(state_dict)
