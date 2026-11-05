"""One shared recurrent column body for the CGB language organism.

This is a G2 implementation candidate, not a language-capability claim.  Its
only persistent state is the three projections required by
``GenerationColumnRefinement``:

* ``H``: bounded column activity;
* ``C``: natural/precision causal carrier;
* ``S``: typed, local edge participation.

Eligibility and local demand are *readouts* of that state.  A byte event is
scored before it is written.  The cell uses an implicit periodic lattice, so
it holds no edge-index table or unbounded event history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


SourceKind = Literal["external", "self"]
_EXTERNAL = 0
_SELF = 1
_NORTH, _EAST, _SOUTH, _WEST = range(4)
_OPPOSITE_DIRECTION = (_SOUTH, _WEST, _NORTH, _EAST)
_GRID_DIRECTION_SHIFTS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass(frozen=True)
class ColumnBodyState:
    """The complete persistent state of the executable column body.

    ``natural`` and ``precision`` together form C.  ``participation`` is S:
    for each grid cell, the four local directions, and the two source kinds,
    it stores a bounded local participation reading.  It is not a fourth
    eligibility page; :meth:`CGBColumnLanguageBody.trace` reconstructs the
    usable trace from ``participation`` and the precision already in C.
    """

    activity: torch.Tensor       # H: [B, rows, cols, D], in [-1, 1]
    natural: torch.Tensor        # C: [B, rows, cols, D]
    precision: torch.Tensor      # C: [B, rows, cols, 1], positive
    causal_token: torch.Tensor   # C: [B], previous live byte or -1 at birth
    participation: torch.Tensor  # S: [B, rows, cols, 4, 2, rank]


@dataclass(frozen=True)
class ColumnBodyStep:
    """Transient products of one causal event; none is retained implicitly."""

    state: ColumnBodyState
    logits: torch.Tensor         # prediction prepared from the old state
    features: torch.Tensor       # old-state local readout, transient only
    read_route: torch.Tensor     # old C query's fixed-support sparse route
    read_values: torch.Tensor    # old local H+C values read through that route
    route: torch.Tensor          # [B, rows, cols], sparse post-arrival write participation
    edge_flux: torch.Tensor      # [B, rows, cols, 2], east/south oriented flow
    credit_token: torch.Tensor   # [B], C query responsible for this pre-score
    value_eligibility: torch.Tensor  # [B], reconstructed direct C sensitivity
    contact_response_gate: torch.Tensor  # [B, rows, cols, D], transient local forward gate
    contact_pre: torch.Tensor  # local H/C/S/event pre-participation for the shared contact rule
    contact_return_weight: torch.Tensor | None = None  # optional one-event H/flow <- S snapshot


class CGBColumnLanguageBody(nn.Module):
    """A scalable local recurrent body with a byte prediction boundary.

    The neural maps are shared across columns.  Spatial differentiation comes
    from the finite body state and event route, never from pre-labelled expert
    modules.  The four-neighbour topology is an address formula on a periodic
    grid; runtime state therefore scales as O(columns * width), with constant
    degree and no edge index tensor.
    """

    def __init__(
        self,
        d_model: int,
        grid_shape: tuple[int, int],
        *,
        active_columns: int = 8,
        contact_rank: int = 1,
        byte_vocab: int = 256,
    ) -> None:
        super().__init__()
        rows, cols = (int(grid_shape[0]), int(grid_shape[1]))
        if rows < 1 or cols < 1:
            raise ValueError("grid dimensions must be positive")
        if d_model < 1:
            raise ValueError("d_model must be positive")
        columns = rows * cols
        if not 1 <= int(active_columns) <= columns:
            raise ValueError("active_columns must lie in [1, rows * cols]")
        if int(contact_rank) < 1:
            raise ValueError("contact_rank must be positive")
        if byte_vocab != 256:
            raise ValueError("the current language boundary is the 256-byte alphabet")

        self.rows = rows
        self.cols = cols
        self.d_model = int(d_model)
        self.active_columns = int(active_columns)
        # S uses one bounded basis coordinate per local contact factor.
        # Raising rank only supplies further distinguishable causal directions;
        # it never allocates an event-history page.
        self.contact_rank = int(contact_rank)
        self.byte_vocab = int(byte_vocab)

        # Boundary coding and routing are learned, but all columns share the
        # same local dynamics below.  Keys select transiently active columns;
        # they are neither an MoE label nor a persistent event archive.
        self.byte_embedding = nn.Embedding(byte_vocab, d_model)
        self.column_keys = nn.Parameter(torch.empty(rows, cols, d_model))
        nn.init.normal_(self.column_keys, std=d_model ** -0.5)
        # The causal address is independent of the value write.  This is the
        # separation required by `GDNLife.valuePerturb`: a value update may
        # not silently alter the next query/address path.
        # Before the first arrival no causal language distinction exists.
        # A learned birth-only query could be trained only by restarting lives,
        # so it is not part of a continuously learning organism.
        self.causal_address = nn.Embedding(byte_vocab, d_model)
        # `event_value` is the only online-plastic value path.  Dynamics and
        # confidence have their own maps, so a credit update cannot mutate an
        # address, likelihood, H, or S path that the GDN theorem holds fixed.
        self.event_value = nn.Linear(d_model, d_model, bias=False)
        self.dynamics_value = nn.Linear(d_model, d_model, bias=False)
        self.event_likelihood = nn.Linear(d_model, 1)
        self.readout = nn.Linear(d_model, byte_vocab)

        # One shared reciprocal local contact field replaces the former pile of
        # independent H/S dynamics maps.  At every lattice site it reads the
        # same existing local state: self H, neighbour H, C mean, C demand,
        # typed/rank S, and the arrived event drive.  The one field emits all
        # local effects: H drive, conservative oriented flux, S allocation and
        # S release.  Thus a contact coordinate has one forward use and one
        # local return coefficient, as formalized by
        # `GenerationReciprocalCredit`; there is no per-mechanism optimizer.
        self.local_contact_input_dim = 4 * d_model + 1 + 8 * self.contact_rank
        # Four allocation and four release channels per rank remain parts of
        # the same contact response, so learned timescales are local contact
        # properties rather than a separately tuned global clock.
        self.local_contact_output_dim = d_model + 2 + 8 * self.contact_rank
        self.reciprocal_contact = nn.Parameter(torch.empty(
            self.local_contact_output_dim, self.local_contact_input_dim,
        ))
        nn.init.normal_(self.reciprocal_contact, std=self.local_contact_input_dim ** -0.5)

        # Birth precision is an initial condition, not a continual plastic
        # degree of freedom: a single non-resetting organism cannot revisit
        # its birth to learn it.  Ongoing release is governed by the learned
        # `process_variance_raw` below.
        self.initial_precision = float(F.softplus(torch.tensor(-2.0)))
        self.process_variance_raw = nn.Parameter(torch.tensor(-3.0))

    @property
    def columns(self) -> int:
        return self.rows * self.cols

    def initial_state(
        self,
        batch: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ColumnBodyState:
        if batch < 1:
            raise ValueError("batch must be positive")
        device = self.column_keys.device if device is None else device
        dtype = self.column_keys.dtype if dtype is None else dtype
        prior = torch.as_tensor(self.initial_precision, device=device, dtype=dtype)
        shape = (batch, self.rows, self.cols)
        return ColumnBodyState(
            activity=torch.zeros(*shape, self.d_model, device=device, dtype=dtype),
            natural=torch.zeros(*shape, self.d_model, device=device, dtype=dtype),
            precision=prior.expand(*shape, 1).clone(),
            # This is the compact causal query already required by the GDN
            # value-credit theorem: it belongs to C, alongside natural and
            # precision, rather than creating an eligibility/history page.
            causal_token=torch.full((batch,), -1, device=device, dtype=torch.long),
            participation=torch.zeros(*shape, 4, 2, self.contact_rank, device=device, dtype=dtype),
        )

    @staticmethod
    def detach_state(state: ColumnBodyState) -> ColumnBodyState:
        """Cut numerical training history without removing causal state."""
        return ColumnBodyState(*(value.detach() for value in (
            state.activity, state.natural, state.precision, state.causal_token, state.participation,
        )))

    @staticmethod
    def clone_state(state: ColumnBodyState) -> ColumnBodyState:
        """Create a complete causal checkpoint with no shared mutable storage."""
        return ColumnBodyState(*(value.detach().clone() for value in (
            state.activity, state.natural, state.precision, state.causal_token, state.participation,
        )))

    @staticmethod
    def persistent_state_bytes(state: ColumnBodyState) -> dict[str, int]:
        """Report the physical H/C/S state separately from parameters/workspace."""
        sizes = {
            "H_activity": state.activity.numel() * state.activity.element_size(),
            "C_natural": state.natural.numel() * state.natural.element_size(),
            "C_precision": state.precision.numel() * state.precision.element_size(),
            "C_causal_token": state.causal_token.numel() * state.causal_token.element_size(),
            "S_participation": state.participation.numel() * state.participation.element_size(),
        }
        return {**sizes, "total": sum(sizes.values())}

    def demand(self, state: ColumnBodyState) -> torch.Tensor:
        """Local write availability, reconstructed from C rather than stored."""
        return state.precision.reciprocal() / (1.0 + state.precision.reciprocal())

    def trace(self, state: ColumnBodyState) -> torch.Tensor:
        """Demand-weighted typed participation history, read solely from C/S."""
        return state.participation * self.demand(state).unsqueeze(-1).unsqueeze(-1)

    @property
    def _contact_s_input(self) -> slice:
        """Columns of the shared field read from the existing S coordinate."""
        start = 3 * self.d_model + 1
        return slice(start, start + 8 * self.contact_rank)

    @property
    def _contact_dynamic_input(self) -> slice:
        """Columns through which the arrived event drives the local H field."""
        return slice(self.local_contact_input_dim - self.d_model, self.local_contact_input_dim)

    @property
    def _contact_h_output(self) -> slice:
        """Rows of the shared field that drive bounded H activity."""
        return slice(0, self.d_model)

    def mute_contact_readout_(self) -> None:
        """Ablate only old S -> H influence while preserving S physics.

        This is an intervention on the corresponding block of the same local
        contact field.  It leaves all S allocation/release rows and the whole
        H/C/S state transition intact, so it is a genuine causal read
        ablation rather than a smaller replacement body.
        """
        with torch.no_grad():
            # Flux contributes to H as its divergence, so it belongs to the
            # same causal read intervention as direct H-drive rows.
            self.reciprocal_contact[:self.d_model + 2, self._contact_s_input].zero_()

    @staticmethod
    def arrival_participation(state: ColumnBodyState) -> torch.Tensor:
        """Read the bounded local S occupancy."""
        return state.participation

    def _validate_state(self, state: ColumnBodyState) -> None:
        expected = (self.rows, self.cols)
        if state.activity.ndim != 4 or tuple(state.activity.shape[1:3]) != expected:
            raise ValueError("activity does not match the configured column grid")
        if state.activity.shape[-1] != self.d_model:
            raise ValueError("activity width does not match d_model")
        if state.natural.shape != state.activity.shape:
            raise ValueError("natural carrier must align with activity")
        if state.precision.shape != (*state.activity.shape[:-1], 1):
            raise ValueError("precision carrier must have one local scalar per column")
        if state.causal_token.shape != (state.activity.shape[0],):
            raise ValueError("causal-token carrier must have one entry per body")
        if state.causal_token.dtype != torch.long:
            raise ValueError("causal-token carrier must use integer byte identities")
        if not bool(((-1 <= state.causal_token) & (state.causal_token < self.byte_vocab)).all()):
            raise ValueError("causal-token carrier must be a byte index or the birth sentinel -1")
        if state.participation.shape != (*state.activity.shape[:-1], 4, 2, self.contact_rank):
            raise ValueError("participation must be one four-neighbour typed low-rank S state")
        if (not bool(torch.isfinite(state.participation).all()) or
                not bool(((0.0 <= state.participation) & (state.participation <= 1.0)).all())):
            raise ValueError("S occupancy must lie in [0, 1]")
        bundle_mass = state.participation.flatten(start_dim=-2).sum(dim=-1)
        if not bool((bundle_mass <= 1.0 + 1e-6).all()):
            raise ValueError("each typed/rank S contact bundle must share at most unit mass")
        if not bool(torch.isfinite(state.precision).all()) or not bool((state.precision > 0).all()):
            raise ValueError("carrier precision must be finite and positive")

    def _route_from_query(self, query: torch.Tensor) -> torch.Tensor:
        """Select a bounded, normalized geometric write/read support."""
        query = F.normalize(query, dim=-1, eps=1e-6)
        keys = F.normalize(self.column_keys, dim=-1, eps=1e-6)
        scores = torch.einsum("bd,hwd->bhw", query, keys)
        flat_scores = scores.flatten(1)
        values, indices = flat_scores.topk(self.active_columns, dim=-1)
        sparse = torch.zeros_like(flat_scores)
        sparse.scatter_(1, indices, torch.softmax(values, dim=-1))
        return sparse.view(-1, self.rows, self.cols)

    def _token_route(self, token: torch.Tensor) -> torch.Tensor:
        """The fixed address of an arrived byte, independent of its value."""
        return self._route_from_query(self.causal_address(token))

    def _state_route(self, state: ColumnBodyState) -> torch.Tensor:
        """Read the next causal query from existing C before the arrival."""
        has_causal_token = (state.causal_token >= 0).unsqueeze(-1)
        token = state.causal_token.clamp_min(0)
        query = self.causal_address(token)
        query = query * has_causal_token.to(dtype=query.dtype)
        return self._route_from_query(query)

    def _value_eligibility_tensors(
        self,
        natural: torch.Tensor,
        precision: torch.Tensor,
        causal_token: torch.Tensor,
        participation: torch.Tensor,
    ) -> torch.Tensor:
        """Capture-safe C-only eligibility kernel; caller owns validation."""
        state = ColumnBodyState(torch.empty_like(natural), natural, precision,
                                causal_token, participation)
        valid = causal_token >= 0
        token = causal_token.clamp_min(0)
        query = self._state_route(state)
        incoming = self._token_route(token) * F.softplus(
            self.event_likelihood(self.byte_embedding(token))
        ).view(-1, 1, 1)
        eligibility = (query * incoming / precision.squeeze(-1)).sum(dim=(1, 2))
        return torch.where(valid, eligibility, torch.zeros_like(eligibility))

    def value_eligibility(self, state: ColumnBodyState) -> torch.Tensor:
        """Validated public view of the reconstructed C eligibility."""
        self._validate_state(state)
        return self._value_eligibility_tensors(state.natural, state.precision,
                                               state.causal_token, state.participation)

    def likelihood_eligibility(self, state: ColumnBodyState) -> torch.Tensor:
        """Return C's exact local posterior-mean sensitivity to its last logit.

        The result is reconstructed from the existing post-write C state and
        causal byte.  It adds no eligibility page: ``natural / precision`` is
        the posterior already retained by C, and the same address map supplies
        the read and write weights.  The formula is Lean
        ``likelihood_posterior_mean_deriv`` applied componentwise.
        """
        self._validate_state(state)
        return self._likelihood_eligibility_tensors(
            state.natural, state.precision, state.causal_token, state.participation,
        )

    def _likelihood_eligibility_tensors(
        self,
        natural: torch.Tensor,
        precision: torch.Tensor,
        causal_token: torch.Tensor,
        participation: torch.Tensor,
    ) -> torch.Tensor:
        """Capture-safe, componentwise likelihood-posterior C sensitivity."""
        state = ColumnBodyState(torch.empty_like(natural), natural, precision,
                                causal_token, participation)
        valid = causal_token >= 0
        token = causal_token.clamp_min(0)
        read_route = self._state_route(state)
        write_route = self._token_route(token)
        source = self.byte_embedding(token)
        raw = self.event_likelihood(source).view(-1, 1, 1, 1)
        slope = torch.sigmoid(raw)
        value = self.event_value(source).view(-1, 1, 1, self.d_model)
        posterior_mean = natural / precision
        component = (
            read_route.unsqueeze(-1) * write_route.unsqueeze(-1) * slope /
            precision * (value - posterior_mean)
        )
        eligibility = component.sum(dim=(1, 2))
        return torch.where(valid.unsqueeze(-1), eligibility, torch.zeros_like(eligibility))

    @staticmethod
    def _as_elapsed(
        elapsed: float | torch.Tensor,
        batch: int,
        like: torch.Tensor,
    ) -> torch.Tensor:
        value = torch.as_tensor(elapsed, device=like.device, dtype=like.dtype)
        if value.ndim == 0:
            value = value.expand(batch)
        if tuple(value.shape) != (batch,) or not bool(torch.isfinite(value).all()) or not bool((value >= 0).all()):
            raise ValueError("elapsed must be a finite nonnegative scalar or one value per batch item")
        return value.view(batch, 1, 1, 1)

    @staticmethod
    def _source_index(source: SourceKind | bool | torch.Tensor, batch: int, like: torch.Tensor) -> torch.Tensor:
        if isinstance(source, str):
            if source not in ("external", "self"):
                raise ValueError("source must be external or self")
            return torch.full((batch,), _EXTERNAL if source == "external" else _SELF,
                              device=like.device, dtype=torch.long)
        value = torch.as_tensor(source, device=like.device)
        if value.ndim == 0:
            value = value.expand(batch)
        if tuple(value.shape) != (batch,):
            raise ValueError("tensor source must provide one boolean per batch item")
        if value.dtype != torch.bool:
            raise ValueError("tensor source is boolean: True means external")
        return torch.where(value, torch.full_like(value, _EXTERNAL, dtype=torch.long),
                           torch.full_like(value, _SELF, dtype=torch.long))

    def _edge_flux(self, contact_response: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return conservative flux emitted by the shared local contact field.

        ``flux[..., 0]`` is eastward and ``flux[..., 1]`` southward.  Periodic
        incoming-minus-outgoing construction has zero total before bounded H
        activity; the two flux channels are rows of the same reciprocal field
        that also produces H and S dynamics.
        """
        flux = torch.tanh(contact_response[..., self.d_model:self.d_model + 2])
        east, south = flux[..., 0], flux[..., 1]
        net = (east.roll(1, dims=2) - east + south.roll(1, dims=1) - south)
        return flux, net

    @staticmethod
    def reciprocal_participation_mismatch(participation: torch.Tensor) -> torch.Tensor:
        """Read each directed contact minus its reciprocal endpoint reading.

        This is a diagnostic readout of S, not another state array. Direction
        order is north, east, south, west; source type remains the final
        contact-type axis.
        """
        if participation.ndim != 6 or participation.shape[-3:-1] != (4, 2):
            raise ValueError("participation must have shape [B, rows, cols, 4, 2, rank]")
        reciprocal = []
        for direction, (row_shift, col_shift) in enumerate(_GRID_DIRECTION_SHIFTS):
            opposite = _OPPOSITE_DIRECTION[direction]
            neighbour = participation[..., opposite, :, :].roll(
                shifts=(-row_shift, -col_shift), dims=(1, 2),
            )
            reciprocal.append(participation[..., direction, :, :] - neighbour)
        return torch.stack(reciprocal, dim=-3)

    def _reciprocal_contact_update(
        self,
        participation: torch.Tensor,
        contact_response: torch.Tensor,
        response_gate: torch.Tensor,
        route: torch.Tensor,
        source_index: torch.Tensor,
        diagnostic_totals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update each selected local contact as an adaptive phase exchange.

        The typed/rank coordinates on one *physical contact* share one finite
        pool: their available partner is ``1 - sum(S_contact)``, not one
        independent ``1-h`` per coordinate.  The learned local drive selects
        a transient distribution for an admitted increment; the existing
        H/C-dependent retention readout releases individual coordinates back
        into that same implicit pool.  This is the shared-reservoir exchange
        proved in ``GenerationContactCompetition``.  It creates actual local
        competition without altering the causal byte route, allocating a
        separate trace page, or inventing a global resource scalar.

        ``response_gate`` is this event's bounded H response.  Each endpoint
        pools its activity coordinates, and their maximum is the shared edge
        gate: either endpoint can admit a response trace, while two closed
        endpoints hold the entire bundle.  Both directed readings use the
        same gate and rates, preserving reciprocal equality.  This retains
        typed/rank admitted participation, not the individual H response
        directions or a historical parameter gradient.  It is an instance of
        ``shared_pool_closed_gate_holds``, not an outcome signal or a second
        learning state.
        """
        allocation_start = self.d_model + 2
        allocation_stop = allocation_start + 4 * self.contact_rank
        raw_drive = contact_response[..., allocation_start:allocation_stop].unflatten(
            -1, (4, self.contact_rank),
        )
        endpoint_retain = torch.sigmoid(contact_response[..., allocation_stop:]).unflatten(
            -1, (4, self.contact_rank),
        )
        if response_gate.shape != (*participation.shape[:3], self.d_model):
            raise ValueError("response gate must align with local H activity")
        endpoint_gate = response_gate.to(dtype=participation.dtype).mean(dim=-1)
        edge_drive, edge_retain, edge_active, edge_gate = [], [], [], []
        for direction, (row_shift, col_shift) in enumerate(_GRID_DIRECTION_SHIFTS):
            opposite = _OPPOSITE_DIRECTION[direction]
            neighbour_drive = raw_drive[..., opposite, :].roll(
                shifts=(-row_shift, -col_shift), dims=(1, 2),
            )
            neighbour_retain = endpoint_retain[..., opposite, :].roll(
                shifts=(-row_shift, -col_shift), dims=(1, 2),
            )
            neighbour_route = route.roll(shifts=(-row_shift, -col_shift), dims=(1, 2))
            neighbour_gate = endpoint_gate.roll(shifts=(-row_shift, -col_shift), dims=(1, 2))
            edge_drive.append(torch.sigmoid(0.5 * (raw_drive[..., direction, :] + neighbour_drive)))
            edge_retain.append(0.5 * (endpoint_retain[..., direction, :] + neighbour_retain))
            edge_active.append(torch.maximum(route, neighbour_route))
            edge_gate.append(torch.maximum(endpoint_gate, neighbour_gate))
        drive = torch.stack(edge_drive, dim=-2)
        # Rank-specific retention is emitted by the same reciprocal field.  No
        # rank clock or additional state tensor exists: a rank is only an S
        # contact coordinate with its own learned local release response.
        retain = torch.stack(edge_retain, dim=-2)
        active = torch.stack(edge_active, dim=-1)
        source_one_hot = F.one_hot(source_index, num_classes=2).to(dtype=participation.dtype)
        raw_target = (drive.unsqueeze(-2) *
                      source_one_hot[:, None, None, None, :, None])
        target_flat = raw_target.flatten(start_dim=-2)
        # Sigmoid drive is strictly positive, so its normalized typed/rank
        # distribution is defined without a stored allocation record.
        target = target_flat / target_flat.sum(dim=-1, keepdim=True)
        # Only the selected source type receives an admission, and its mean
        # local drive is an in-range total up-rate for the shared contact pool.
        # This is an edge statistic: the two directed readings must use the
        # same response fraction even when only one endpoint can respond.
        gate = torch.stack(edge_gate, dim=-1)
        admit = active * drive.mean(dim=-1)
        # Retention has one value per direction/rank.  It is shared by the
        # two existing source types of that physical contact, then flattened
        # only over source-type × rank (never over the direction axis).
        retain_flat = retain.unsqueeze(-2).expand(
            *retain.shape[:-1], 2, self.contact_rank,
        ).flatten(start_dim=-2)
        release = active.unsqueeze(-1) * (1.0 - retain_flat)
        occupied = participation.flatten(start_dim=-2)
        next_occupied = self._shared_contact_pool_exchange(
            occupied, target, gate * admit, gate.unsqueeze(-1) * release,
        )
        if diagnostic_totals is not None:
            # Observer-owned counters: count each reciprocal edge once.  The
            # local VJP does not pass them, so recomputation records no event.
            with torch.no_grad():
                free = 1.0 - occupied.sum(dim=-1, keepdim=True)
                incoming = (gate * admit).unsqueeze(-1) * target * free
                outgoing = gate.unsqueeze(-1) * release * occupied
                diagnostic_totals[0].add_(incoming.sum() * 0.5)
                diagnostic_totals[1].add_(outgoing.sum() * 0.5)
        return next_occupied.unflatten(-1, (2, self.contact_rank))

    @staticmethod
    def _shared_contact_pool_exchange(
        occupied: torch.Tensor,
        target: torch.Tensor,
        admit: torch.Tensor,
        release: torch.Tensor,
    ) -> torch.Tensor:
        """Exchange one existing typed contact bundle with its implicit pool.

        ``occupied`` and ``target`` end in the same flattened contact-factor
        axis.  The target is normalized on that axis, ``admit`` is a scalar
        up-rate per physical contact, and ``release`` is a per-factor local
        down-rate.  If the initial bundle mass is in ``[0, 1]``, each rate is
        in ``[0, 1]``, and the target sums to one, the update keeps its total
        mass in that interval.  No pool tensor is materialized.
        """
        if occupied.shape != target.shape or release.shape != occupied.shape:
            raise ValueError("contact pool factors must have matching shapes")
        if admit.shape != occupied.shape[:-1]:
            raise ValueError("contact pool admission must be scalar per contact")
        free = 1.0 - occupied.sum(dim=-1, keepdim=True)
        return occupied + admit.unsqueeze(-1) * target * free - release * occupied

    @staticmethod
    @torch.no_grad()
    def evidence_gated_s_exchange(
        participation: torch.Tensor,
        teaching: torch.Tensor,
        step_size: float,
    ) -> torch.Tensor:
        """Apply one bounded language receipt to the existing local S pool.

        ``teaching`` is the signed, one-receipt return for the physical S
        coordinates.  Positive components admit the indicated factor from the
        bundle's implicit free partner; negative components release that
        factor.  The exponential rate map is a bounded kinetic response to
        receipt magnitude, so this is an instance of the existing vector
        shared-pool exchange, rather than a gradient update of the common
        contact law or an extra eligibility state.

        The update is deliberately local and one-receipt only.  It proves the
        conservation/competition admission of the delayed third factor; it
        does not claim an arbitrary-horizon parameter gradient.
        """
        if participation.shape != teaching.shape:
            raise ValueError("language return must align with S participation")
        if not isinstance(step_size, (float, int)) or isinstance(step_size, bool) or step_size < 0:
            raise ValueError("step_size must be a finite nonnegative scalar")
        if not torch.isfinite(torch.as_tensor(step_size)):
            raise ValueError("step_size must be finite")
        # Do not reduce a CUDA tensor to a Python boolean here: this function
        # is part of the captured event step.  The CPU check keeps the public
        # input contract helpful, while CUDA non-finites remain observable by
        # the runner's numerical audit without breaking CUDA Graph capture.
        if not teaching.is_cuda and not torch.isfinite(teaching).all():
            raise ValueError("language return must be finite")
        occupied = participation.flatten(start_dim=-2)
        returned = teaching.detach().flatten(start_dim=-2)
        positive = returned.clamp_min(0.0)
        positive_total = positive.sum(dim=-1, keepdim=True)
        # `target` is always a probability distribution.  With no positive
        # return it is immaterial because the admission rate below is zero;
        # using uniform avoids a hidden undefined 0/0 branch.
        uniform = torch.full_like(positive, 1.0 / positive.shape[-1])
        target = torch.where(
            positive_total > 0.0,
            positive / positive_total.clamp_min(torch.finfo(positive.dtype).tiny),
            uniform,
        )
        # Positive and negative evidence select opposite directions through
        # the same finite pool.  1-exp(-x) lies in [0,1) for x >= 0.
        admit = -torch.expm1(-float(step_size) * positive_total.squeeze(-1))
        release = -torch.expm1(-float(step_size) * (-returned).clamp_min(0.0))
        next_occupied = CGBColumnLanguageBody._shared_contact_pool_exchange(
            occupied, target, admit, release,
        )
        return next_occupied.unflatten(-1, (2, participation.shape[-1]))

    @torch.no_grad()
    def update_s_participation_from_following(
        self,
        previous: ColumnBodyStep | None,
        current: ColumnBodyStep,
        target: torch.Tensor,
        *,
        step_size: float,
    ) -> ColumnBodyState:
        """Consume one boundary receipt as a conserved S redistribution.

        The returned state is the already-written current S carrier with the
        old receipt's language evidence applied.  No model parameter or new
        persistent carrier is touched.  The caller installs this state for the
        next event, preserving causal score-before-write order.
        """
        if step_size == 0 or previous is None:
            return current.state
        if previous.contact_return_weight is None:
            raise ValueError("the previous event requires record_contact_return=True")
        if target.shape != (current.state.activity.shape[0],) or target.dtype != torch.long:
            raise ValueError("target must contain one byte index per body")
        if not bool(((0 <= target) & (target < self.byte_vocab)).all()):
            raise ValueError("target must contain byte indices")
        index, _, gate, unique = self._pack_local_receipt(
            previous.contact_pre, previous.contact_response_gate, previous.route,
        )
        teaching = self._s_contact_return_from_tensors(
            current.logits, current.read_route, index, gate, unique,
            previous.edge_flux, previous.contact_return_weight, target,
        )
        return ColumnBodyState(
            activity=current.state.activity,
            natural=current.state.natural,
            precision=current.state.precision,
            causal_token=current.state.causal_token,
            participation=self.evidence_gated_s_exchange(
                current.state.participation, teaching, step_size,
            ),
        )

    @staticmethod
    def _adaptive_phase_exchange(
        occupied: torch.Tensor,
        drive: torch.Tensor,
        retain: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """One local, adaptive two-state exchange on an existing S coordinate.

        ``drive`` and ``retain`` are local, learned values in ``[0, 1]``;
        ``active`` is the sparse routed contact participation.  With
        ``k_plus = active * drive`` and ``k_minus = active * (1 - retain)``,
        this is exactly ``Generation.twoStateStep k_plus k_minus occupied``.
        Its unoccupied partner is ``1 - occupied`` and remains implicit.
        """
        scale = active.unsqueeze(-1).unsqueeze(-1)
        # A scalar hold is shared across rank in the smallest body.  A
        # generated bank supplies one hold per rank without allocating a new
        # state coordinate, so insert only the source-type axis.
        if retain.ndim == drive.ndim - 2:
            hold = retain.unsqueeze(-1).unsqueeze(-1)
        elif retain.ndim == drive.ndim - 1:
            hold = retain.unsqueeze(-2)
        else:
            raise ValueError("retain must provide one value per edge or per edge/rank")
        k_plus = scale * drive
        k_minus = scale * (1.0 - hold)
        return occupied + k_plus * (1.0 - occupied) - k_minus * occupied

    def advance_time(
        self,
        state: ColumnBodyState,
        elapsed: float | torch.Tensor,
    ) -> ColumnBodyState:
        """Apply only the declared continuous C release during an empty gap."""
        self._validate_state(state)
        dt = self._as_elapsed(elapsed, state.activity.shape[0], state.precision)
        process = F.softplus(self.process_variance_raw).to(dtype=state.precision.dtype)
        precision = state.precision / (1.0 + dt * process * state.precision)
        return ColumnBodyState(
            activity=state.activity,
            natural=precision * (state.natural / state.precision),
            precision=precision,
            causal_token=state.causal_token,
            participation=state.participation,
        )

    @torch.no_grad()
    def _update_value_from_tensors(
        self,
        logits: torch.Tensor,
        credit_token: torch.Tensor,
        value_eligibility: torch.Tensor,
        target: torch.Tensor,
        step_size: float,
    ) -> None:
        """Capture-safe value update; public wrapper owns input validation."""
        innovation = self._language_innovation(logits, target)
        language_error = innovation @ self.readout.weight.detach()
        source = self.byte_embedding(credit_token.clamp_min(0)).detach()
        valid = (credit_token >= 0).to(dtype=language_error.dtype).unsqueeze(-1)
        scaled_error = language_error * value_eligibility.detach().unsqueeze(-1) * valid
        self.event_value.weight.add_(scaled_error.transpose(0, 1) @ source, alpha=float(step_size))

    @torch.no_grad()
    def _update_likelihood_from_tensors(
        self,
        logits: torch.Tensor,
        natural: torch.Tensor,
        precision: torch.Tensor,
        credit_token: torch.Tensor,
        participation: torch.Tensor,
        target: torch.Tensor,
        step_size: float,
    ) -> None:
        """Apply one current language score to the last C likelihood logit."""
        innovation = self._language_innovation(logits, target)
        language_error = innovation @ self.readout.weight.detach()
        eligibility = self._likelihood_eligibility_tensors(
            natural, precision, credit_token, participation,
        )
        scalar_score = (language_error * eligibility.detach()).sum(dim=-1)
        source = self.byte_embedding(credit_token.clamp_min(0)).detach()
        valid = (credit_token >= 0).to(dtype=scalar_score.dtype)
        scalar_score = scalar_score * valid
        self.event_likelihood.weight.add_(scalar_score.unsqueeze(0) @ source,
                                          alpha=float(step_size))
        if self.event_likelihood.bias is not None:
            self.event_likelihood.bias.add_(scalar_score.sum().view_as(self.event_likelihood.bias),
                                            alpha=float(step_size))

    @torch.no_grad()
    def update_likelihood_from_arrival(
        self,
        state: ColumnBodyState,
        prepared: ColumnBodyStep,
        target: torch.Tensor,
        *,
        step_size: float,
    ) -> None:
        """Use the following byte score to update C's preceding confidence.

        ``state`` is the real prepared C body before ``target`` arrives;
        ``prepared`` supplies that body's logits.  No old parameter tensor,
        autograd graph, or trace is stored.  Value and likelihood are both C
        write coordinates, so this is called before any route/value update can
        change their shared issuing transition.
        """
        self._validate_state(state)
        if not isinstance(step_size, (float, int)) or isinstance(step_size, bool) or step_size < 0:
            raise ValueError("step_size must be a finite nonnegative scalar")
        if target.ndim != 1 or tuple(target.shape) != (state.activity.shape[0],):
            raise ValueError("target must contain one byte index per prepared prediction")
        if target.dtype != torch.long or not bool(((0 <= target) & (target < self.byte_vocab)).all()):
            raise ValueError("target must contain byte indices")
        if step_size == 0:
            return
        self._update_likelihood_from_tensors(
            prepared.logits.detach(), state.natural, state.precision,
            state.causal_token, state.participation, target, float(step_size),
        )

    @torch.no_grad()
    def update_value_from_arrival(
        self,
        prepared: ColumnBodyStep,
        target: torch.Tensor,
        *,
        step_size: float,
    ) -> None:
        """Apply the Lean-admitted previous-value credit before head learning.

        `prepared.credit_token` and `prepared.value_eligibility` are transient
        views of old C, never another persistent page.  With address,
        likelihood, H and S fixed with respect to `event_value`, this outer
        product equals the direct next-byte log-likelihood derivative.
        """
        if not isinstance(step_size, (float, int)) or isinstance(step_size, bool) or step_size < 0:
            raise ValueError("step_size must be a finite nonnegative scalar")
        if target.ndim != 1 or tuple(target.shape) != (prepared.logits.shape[0],):
            raise ValueError("target must contain one byte index per prepared prediction")
        if target.dtype != torch.long or not bool(((0 <= target) & (target < self.byte_vocab)).all()):
            raise ValueError("target must contain byte indices")
        if step_size == 0:
            return
        self._update_value_from_tensors(
            prepared.logits.detach(), prepared.credit_token,
            prepared.value_eligibility, target, float(step_size),
        )

    @staticmethod
    def _language_innovation(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the arrived-minus-predicted byte vector for one receipt."""
        innovation = -torch.softmax(logits.detach(), dim=-1)
        innovation[torch.arange(target.shape[0], device=target.device), target] += 1
        return innovation

    @torch.no_grad()
    def update_readout_from_arrival(
        self,
        prepared: ColumnBodyStep,
        target: torch.Tensor,
        *,
        step_size: float,
    ) -> None:
        """Apply Lean G14 to the already prepared byte prediction.

        This is the exact observed-minus-predicted affine head update.  Its
        feature belongs to the returned event receipt, never to persistent Z;
        calling it after the target arrives cannot reopen an autograd history.
        It is the immediate supervised language path, not delayed action
        credit or a replacement for the later three-factor S readout.
        """
        if not isinstance(step_size, (float, int)) or isinstance(step_size, bool) or step_size < 0:
            raise ValueError("step_size must be a finite nonnegative scalar")
        if not torch.isfinite(torch.as_tensor(step_size)):
            raise ValueError("step_size must be finite")
        if target.ndim != 1 or tuple(target.shape) != (prepared.logits.shape[0],):
            raise ValueError("target must contain one byte index per prepared prediction")
        if target.dtype != torch.long or not bool(((0 <= target) & (target < self.byte_vocab)).all()):
            raise ValueError("target must contain byte indices")
        if step_size == 0:
            return
        innovation = self._language_innovation(prepared.logits, target)
        self.readout.weight.add_(innovation.transpose(0, 1) @ prepared.features.detach(), alpha=float(step_size))
        if self.readout.bias is not None:
            self.readout.bias.add_(innovation.sum(dim=0), alpha=float(step_size))

    @staticmethod
    def _normalize_pullback(
        vector: torch.Tensor,
        normalized: torch.Tensor,
        upstream: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Pull a score derivative through the exact local `F.normalize` rule.

        Above its epsilon floor, normalization projects a change to the
        tangent plane and divides by the vector norm.  Below the floor the
        executable rule is simply division by ``eps``.  Keeping both branches
        makes the receipt agree with the real route even at a degenerate
        birth vector; it does not introduce a separate optimizer state.
        """
        norm = vector.norm(dim=-1, keepdim=True)
        tangent = (upstream - normalized * (normalized * upstream).sum(dim=-1, keepdim=True))
        return torch.where(norm > eps, tangent / norm, upstream / eps)

    @torch.no_grad()
    def _update_route_from_tensors(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        read_route: torch.Tensor,
        read_values: torch.Tensor,
        credit_token: torch.Tensor,
        target: torch.Tensor,
        step_size: float,
    ) -> None:
        """Capture-safe fixed-support route receipt; public wrapper validates."""
        innovation = self._language_innovation(logits, target)
        feature_score = innovation @ self.readout.weight.detach()
        centered_value = read_values - features[:, None, None, :]
        route_score = read_route * (feature_score[:, None, None, :] * centered_value).sum(dim=-1)
        valid = credit_token >= 0
        token = credit_token.clamp_min(0)
        query = self.causal_address(token)
        query_normalized = F.normalize(query, dim=-1, eps=1e-6)
        keys = self.column_keys.reshape(self.columns, self.d_model)
        key_normalized = F.normalize(keys, dim=-1, eps=1e-6)
        route_score_flat = route_score.reshape(route_score.shape[0], self.columns)
        route_score_flat = route_score_flat * valid.to(dtype=route_score.dtype).unsqueeze(-1)
        query_upstream = route_score_flat @ key_normalized
        query_grad = self._normalize_pullback(query, query_normalized, query_upstream)
        key_upstream = route_score_flat.transpose(0, 1) @ query_normalized
        key_grad = self._normalize_pullback(keys, key_normalized, key_upstream)
        # The birth sentinel shares row zero only after its factor is exactly
        # masked out, which keeps this fixed-shape path CUDA-graph safe.
        self.causal_address.weight.index_add_(0, token, query_grad, alpha=step_size)
        self.column_keys.add_(key_grad.view_as(self.column_keys), alpha=step_size)

    def _pack_local_receipt(
        self,
        contact_pre: torch.Tensor,
        contact_gate: torch.Tensor,
        write_route: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack exactly the prior write neighbourhood for a fixed O(1) receipt.

        Only routed sites and their four lattice neighbours can have a true
        H-response gate.  This packs at most ``5 * active_columns`` sites,
        marks periodic duplicates once, and leaves all other cells implicit
        zeros.  It is an exact reindexing of the local derivative sum, not a
        sparse approximation or a new state carrier.
        """
        batch = write_route.shape[0]
        base = write_route.flatten(1).topk(self.active_columns, dim=-1).indices
        rows = torch.div(base, self.cols, rounding_mode="floor")
        cols = base.remainder(self.cols)
        sites = [base]
        for row_shift, col_shift in _GRID_DIRECTION_SHIFTS:
            sites.append(((rows + row_shift).remainder(self.rows) * self.cols
                          + (cols + col_shift).remainder(self.cols)))
        index = torch.cat(sites, dim=-1)
        count = index.shape[-1]
        prior = torch.tril(torch.ones(count, count, dtype=torch.bool, device=index.device), diagonal=-1)
        unique = ~((index.unsqueeze(-1) == index.unsqueeze(-2)) & prior).any(dim=-1)
        pre_flat = contact_pre.reshape(batch, self.columns, contact_pre.shape[-1])
        gate_flat = contact_gate.reshape(batch, self.columns, self.d_model)
        pre = pre_flat.gather(1, index.unsqueeze(-1).expand(-1, -1, pre_flat.shape[-1]))
        gate = gate_flat.gather(1, index.unsqueeze(-1).expand(-1, -1, self.d_model))
        return index, pre, gate, unique

    @torch.no_grad()
    def _s_contact_return_from_tensors(
        self,
        logits: torch.Tensor,
        read_route: torch.Tensor,
        previous_index: torch.Tensor,
        previous_gate: torch.Tensor,
        previous_unique: torch.Tensor,
        previous_flux: torch.Tensor,
        previous_weight: torch.Tensor,
        target: torch.Tensor,
        has_previous: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the issued score through exactly one old spatial contact.

        The old numerical H/flow <- S weights and flux belong to one consumed
        pipeline receipt.  They are not current weights or a historical
        state/parameter Jacobian.  The flux adjoint can reach outside the packed
        H-write neighbourhood, so reconstruct its gate on the full lattice.
        """
        innovation = self._language_innovation(logits, target)
        read_coefficient = innovation @ self.readout.weight.detach()
        batch = read_route.shape[0]
        packed_gate = previous_gate.to(dtype=read_coefficient.dtype)
        packed_gate = packed_gate * previous_unique.unsqueeze(-1)
        gate = torch.zeros(
            batch, self.columns, self.d_model,
            device=read_route.device, dtype=read_coefficient.dtype,
        )
        gate.scatter_add_(
            1, previous_index.unsqueeze(-1).expand(-1, -1, self.d_model), packed_gate,
        )
        h_return = (gate.reshape(batch, self.rows, self.cols, self.d_model)
                    * read_route.unsqueeze(-1) * read_coefficient[:, None, None, :])
        if has_previous is not None:
            h_return = h_return * has_previous.view(-1, 1, 1, 1)
        direct = h_return @ previous_weight[:self.d_model].detach()
        potential = h_return.sum(dim=-1)
        flow_return = torch.stack((
            potential.roll(-1, dims=2) - potential,
            potential.roll(-1, dims=1) - potential,
        ), dim=-1)
        flow_return = flow_return * (1.0 - previous_flux.detach().square())
        returned = direct + flow_return @ previous_weight[self.d_model:].detach()
        return returned.unflatten(-1, (4, 2, self.contact_rank))

    @torch.no_grad()
    def _contact_input_return_from_tensors(
        self,
        logits: torch.Tensor,
        read_route: torch.Tensor,
        previous_index: torch.Tensor,
        previous_gate: torch.Tensor,
        previous_unique: torch.Tensor,
        previous_flux: torch.Tensor,
        previous_weight: torch.Tensor,
        target: torch.Tensor,
        has_previous: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one score through the complete old H/flow contact input.

        This is the shared local adjoint behind the S and event-drive paths.
        It consumes only the immediately preceding contact gate and flux; the
        returned tensor is a current-event workspace, never a body coordinate
        or an accumulated temporal derivative.
        """
        innovation = self._language_innovation(logits, target)
        read_coefficient = innovation @ self.readout.weight.detach()
        batch = read_route.shape[0]
        packed_gate = previous_gate.to(dtype=read_coefficient.dtype)
        packed_gate = packed_gate * previous_unique.unsqueeze(-1)
        gate = torch.zeros(
            batch, self.columns, self.d_model,
            device=read_route.device, dtype=read_coefficient.dtype,
        )
        gate.scatter_add_(
            1, previous_index.unsqueeze(-1).expand(-1, -1, self.d_model), packed_gate,
        )
        h_return = (gate.reshape(batch, self.rows, self.cols, self.d_model)
                    * read_route.unsqueeze(-1) * read_coefficient[:, None, None, :])
        if has_previous is not None:
            h_return = h_return * has_previous.view(-1, 1, 1, 1)
        direct = h_return @ previous_weight[:self.d_model].detach()
        potential = h_return.sum(dim=-1)
        flow_return = torch.stack((
            potential.roll(-1, dims=2) - potential,
            potential.roll(-1, dims=1) - potential,
        ), dim=-1)
        flow_return = flow_return * (1.0 - previous_flux.detach().square())
        return direct + flow_return @ previous_weight[self.d_model:].detach()

    @torch.no_grad()
    def _dynamics_value_direction_from_tensors(
        self,
        logits: torch.Tensor,
        read_route: torch.Tensor,
        previous_index: torch.Tensor,
        previous_gate: torch.Tensor,
        previous_unique: torch.Tensor,
        previous_flux: torch.Tensor,
        previous_token: torch.Tensor,
        target: torch.Tensor,
        has_previous: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct the exact one-transition score direction for event drive.

        The event's byte identity is already C's causal token, and its local
        address is reconstructed through the fixed address map.  Thus this
        adds neither an eligibility page nor an event archive.  The method is
        deliberately a direction only: an online step is admitted separately
        after its receipt lifetime and shared contact schedule are audited.
        """
        drive_return = self._dynamics_value_return_from_tensors(
            logits, read_route, previous_index, previous_gate, previous_unique,
            previous_flux, previous_token, target, has_previous,
        )
        source = self.byte_embedding(previous_token.clamp_min(0)).detach()
        return torch.einsum("bo,bi->oi", drive_return, source)

    @torch.no_grad()
    def _dynamics_value_return_from_tensors(
        self,
        logits: torch.Tensor,
        read_route: torch.Tensor,
        previous_index: torch.Tensor,
        previous_gate: torch.Tensor,
        previous_unique: torch.Tensor,
        previous_flux: torch.Tensor,
        previous_token: torch.Tensor,
        target: torch.Tensor,
        has_previous: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the one-event score coefficient at `dynamics_value` output."""
        returned = self._contact_input_return_from_tensors(
            logits, read_route, previous_index, previous_gate, previous_unique,
            previous_flux, self.reciprocal_contact[:self.d_model + 2], target,
            has_previous,
        )
        token = previous_token.clamp_min(0)
        route = self._token_route(token).unsqueeze(-1)
        drive_return = (returned[..., self._contact_dynamic_input] * route).sum(dim=(1, 2))
        if has_previous is not None:
            drive_return = drive_return * has_previous.to(dtype=drive_return.dtype).unsqueeze(-1)
        return drive_return

    @torch.no_grad()
    def _event_embedding_direction_from_tensors(
        self,
        logits: torch.Tensor,
        natural: torch.Tensor,
        precision: torch.Tensor,
        credit_token: torch.Tensor,
        participation: torch.Tensor,
        read_route: torch.Tensor,
        previous_index: torch.Tensor,
        previous_gate: torch.Tensor,
        previous_unique: torch.Tensor,
        previous_flux: torch.Tensor,
        target: torch.Tensor,
        has_previous: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sum every admitted one-event return at the shared byte embedding.

        Value, confidence and local dynamics all transform the same arrived
        byte.  Their source derivatives must therefore be combined *before*
        either map changes.  This is an exact one-transition product-rule sum
        at a frozen body point; it does not identify an arbitrary-horizon
        derivative or allocate a shared optimiser state.
        """
        innovation = self._language_innovation(logits, target)
        language_error = innovation @ self.readout.weight.detach()
        valid = (credit_token >= 0).to(dtype=language_error.dtype).unsqueeze(-1)
        value_return = language_error * self._value_eligibility_tensors(
            natural, precision, credit_token, participation,
        ).detach().unsqueeze(-1) * valid
        likelihood = self._likelihood_eligibility_tensors(
            natural, precision, credit_token, participation,
        )
        likelihood_return = (language_error * likelihood.detach()).sum(dim=-1, keepdim=True) * valid
        dynamics_return = self._dynamics_value_return_from_tensors(
            logits, read_route, previous_index, previous_gate, previous_unique,
            previous_flux, credit_token, target, has_previous,
        )
        source_return = (
            value_return @ self.event_value.weight.detach()
            + likelihood_return @ self.event_likelihood.weight.detach()
            + dynamics_return @ self.dynamics_value.weight.detach()
        )
        direction = torch.zeros_like(self.byte_embedding.weight)
        direction.index_add_(0, credit_token.clamp_min(0), source_return)
        return direction

    @torch.no_grad()
    def _update_event_maps_from_tensors(
        self,
        logits: torch.Tensor,
        natural: torch.Tensor,
        precision: torch.Tensor,
        credit_token: torch.Tensor,
        participation: torch.Tensor,
        read_route: torch.Tensor,
        previous_index: torch.Tensor,
        previous_gate: torch.Tensor,
        previous_unique: torch.Tensor,
        previous_flux: torch.Tensor,
        target: torch.Tensor,
        step_size: float,
        has_previous: torch.Tensor | None = None,
    ) -> None:
        """Apply one common C-event step to every admitted event coordinate.

        All directions are formed before a parameter changes.  The shared
        step is the existing C write scale, not a fourth optimiser or an
        independently tuned dynamics rate.  This remains a one-transition
        learning rule: callers must provide the consumed prior contact receipt.
        """
        innovation = self._language_innovation(logits, target)
        language_error = innovation @ self.readout.weight.detach()
        valid = (credit_token >= 0).to(dtype=language_error.dtype).unsqueeze(-1)
        source = self.byte_embedding(credit_token.clamp_min(0)).detach()
        value_return = language_error * self._value_eligibility_tensors(
            natural, precision, credit_token, participation,
        ).detach().unsqueeze(-1) * valid
        likelihood = self._likelihood_eligibility_tensors(
            natural, precision, credit_token, participation,
        )
        likelihood_return = (language_error * likelihood.detach()).sum(dim=-1, keepdim=True) * valid
        dynamics_return = self._dynamics_value_return_from_tensors(
            logits, read_route, previous_index, previous_gate, previous_unique,
            previous_flux, credit_token, target, has_previous,
        )
        embedding_direction = (
            value_return @ self.event_value.weight.detach()
            + likelihood_return @ self.event_likelihood.weight.detach()
            + dynamics_return @ self.dynamics_value.weight.detach()
        )
        self.event_value.weight.add_(value_return.transpose(0, 1) @ source, alpha=float(step_size))
        self.event_likelihood.weight.add_(likelihood_return.transpose(0, 1) @ source,
                                          alpha=float(step_size))
        if self.event_likelihood.bias is not None:
            self.event_likelihood.bias.add_(likelihood_return.sum(dim=0).view_as(self.event_likelihood.bias),
                                            alpha=float(step_size))
        self.dynamics_value.weight.add_(dynamics_return.transpose(0, 1) @ source,
                                        alpha=float(step_size))
        self.byte_embedding.weight.index_add_(
            0, credit_token.clamp_min(0), embedding_direction, alpha=float(step_size),
        )

    @torch.no_grad()
    def _s_contact_direction_from_tensors(
        self,
        occupied: torch.Tensor,
        precision: torch.Tensor,
        contact_pre: torch.Tensor,
        response_gate: torch.Tensor,
        write_route: torch.Tensor,
        source_index: torch.Tensor,
        teaching: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiate only the current pool, modulated by its existing tag.

        Phi = sum(stopgrad(teaching * trace(old C/S)) * current_pool(W)).
        This is a local language-feedback surrogate, not the issued CE or its
        full-history gradient.  Current pre is never passed off as historic pre.
        The same spatial kernel generates allocation and retain derivatives;
        its VJP is consumed here and never carried to another event.
        """
        occupied, precision = occupied.detach(), precision.detach()
        pre, gate = contact_pre.detach(), response_gate.detach()
        route, source = write_route.detach(), source_index.detach()
        demand = precision.reciprocal() / (1.0 + precision.reciprocal())
        tag = occupied * demand.unsqueeze(-1).unsqueeze(-1)
        cotangent = (teaching.detach() * tag).detach()

        def local_pool(weight: torch.Tensor) -> torch.Tensor:
            return self._reciprocal_contact_update(
                occupied, F.linear(pre, weight), gate, route, source,
            )

        _, pullback = torch.func.vjp(local_pool, self.reciprocal_contact.detach())
        direction, = pullback(cotangent)
        return direction.detach()

    @torch.no_grad()
    def update_s_contact_from_following(
        self,
        previous: ColumnBodyStep | None,
        current: ColumnBodyStep,
        state: ColumnBodyState,
        target: torch.Tensor,
        *,
        step_size: float,
        source: SourceKind | bool | torch.Tensor = "external",
    ) -> None:
        """Use old contact feedback and current C/S tags to learn the S rows.

        ``state`` is the input to ``current``, not its newly written state.
        A nonzero step requires the previous event's optional return snapshot.
        Only one spatial VJP is formed: the long-lived part is the physical S
        tag, not a saved graph, reconstructed old pre or delayed total gradient.
        """
        if not isinstance(step_size, (float, int)) or isinstance(step_size, bool) or step_size < 0:
            raise ValueError("step_size must be a finite nonnegative scalar")
        if not torch.isfinite(torch.as_tensor(step_size)):
            raise ValueError("step_size must be finite")
        if step_size == 0 or previous is None:
            return
        self._validate_state(state)
        if previous.contact_return_weight is None:
            raise ValueError("the previous event requires record_contact_return=True")
        if target.shape != (state.activity.shape[0],) or target.dtype != torch.long:
            raise ValueError("target must contain one byte index per body")
        if not bool(((0 <= target) & (target < self.byte_vocab)).all()):
            raise ValueError("target must contain byte indices")
        index, _, gate, unique = self._pack_local_receipt(
            previous.contact_pre, previous.contact_response_gate, previous.route,
        )
        teaching = self._s_contact_return_from_tensors(
            current.logits, current.read_route, index, gate, unique,
            previous.edge_flux, previous.contact_return_weight, target,
        )
        direction = self._s_contact_direction_from_tensors(
            state.participation, state.precision, current.contact_pre,
            current.contact_response_gate, current.route,
            self._source_index(source, state.activity.shape[0], state.activity), teaching,
        )
        start = self.d_model + 2
        self.reciprocal_contact[start:].add_(direction[start:], alpha=float(step_size))

    @torch.no_grad()
    def _update_reciprocal_h_from_tensors(
        self,
        logits: torch.Tensor,
        read_route: torch.Tensor,
        previous_index: torch.Tensor,
        previous_pre: torch.Tensor,
        previous_gate: torch.Tensor,
        previous_unique: torch.Tensor,
        target: torch.Tensor,
        step_size: float,
        has_previous: torch.Tensor | None = None,
    ) -> None:
        """Apply the exact next-score H-contact direction from one receipt.

        The preceding event leaves only an event-local pipeline receipt
        ``(previous_gate, previous_pre)``.  The present language score supplies
        the third factor and its already-issued C route supplies the local post
        coefficient.  This is an O(1) one-transition instance of
        ``GenerationReciprocalCredit`` plus the hard-saturation condition in
        ``GenerationContactEligibility``.  It deliberately does not claim to
        solve an arbitrary later credit assignment; that requires a proved
        H/C/S carrier rather than extending this receipt.
        """
        innovation = self._language_innovation(logits, target)
        read_coefficient = innovation @ self.readout.weight.detach()
        post = read_coefficient[:, None, None, :] * read_route.unsqueeze(-1)
        post_flat = post.reshape(post.shape[0], self.columns, self.d_model)
        packed_post = post_flat.gather(
            1, previous_index.unsqueeze(-1).expand(-1, -1, self.d_model),
        )
        if has_previous is not None:
            packed_post = packed_post * has_previous.to(dtype=packed_post.dtype).view(-1, 1, 1)
        response = packed_post * previous_gate.to(dtype=packed_post.dtype)
        response = response * previous_unique.to(dtype=response.dtype).unsqueeze(-1)
        direction = torch.einsum("bkd,bkp->dp", response, previous_pre)
        self.reciprocal_contact[self._contact_h_output].add_(direction, alpha=float(step_size))

    @torch.no_grad()
    def update_reciprocal_h_from_following(
        self,
        previous: ColumnBodyStep,
        current: ColumnBodyStep,
        target: torch.Tensor,
        *,
        step_size: float,
    ) -> None:
        """Consume one prior local receipt at the next language boundary.

        ``previous`` must be the immediately preceding event from this body;
        it is a consumed boundary receipt, not a field added to H/C/S.  The
        current score is evaluated before this update, so its result cannot
        rewrite its own prediction.  Fixed sparse support is carried by
        ``current.read_route`` and follows ``GenerationRoutingCredit``.
        """
        if not isinstance(step_size, (float, int)) or isinstance(step_size, bool) or step_size < 0:
            raise ValueError("step_size must be a finite nonnegative scalar")
        if not torch.isfinite(torch.as_tensor(step_size)):
            raise ValueError("step_size must be finite")
        if target.ndim != 1 or tuple(target.shape) != (current.logits.shape[0],):
            raise ValueError("target must contain one byte index per body")
        if target.dtype != torch.long or not bool(((0 <= target) & (target < self.byte_vocab)).all()):
            raise ValueError("target must contain byte indices")
        if previous.contact_pre.shape[:3] != current.read_route.shape:
            raise ValueError("previous local receipt and current causal route do not align")
        if step_size == 0:
            return
        index, pre, gate, unique = self._pack_local_receipt(
            previous.contact_pre, previous.contact_response_gate, previous.route,
        )
        self._update_reciprocal_h_from_tensors(
            current.logits, current.read_route, index, pre, gate, unique,
            target, float(step_size),
        )

    @torch.no_grad()
    def update_route_from_arrival(
        self,
        prepared: ColumnBodyStep,
        target: torch.Tensor,
        *,
        step_size: float,
    ) -> None:
        """Apply the exact one-event language credit to the causal route.

        The existing C byte identity selects a query; the selected finite
        column support reads local H+C values.  With that support held fixed
        for this already-issued receipt, ``GenerationRoutingCredit`` gives
        the centered softmax factor.  Pulling it through the normalized query
        and normalized keys updates both slow route maps without BPTT or a
        persistent trace page.  A later support change is a new event, never
        treated as a differentiable retrospective update.
        """
        if not isinstance(step_size, (float, int)) or isinstance(step_size, bool) or step_size < 0:
            raise ValueError("step_size must be a finite nonnegative scalar")
        if not torch.isfinite(torch.as_tensor(step_size)):
            raise ValueError("step_size must be finite")
        if target.ndim != 1 or tuple(target.shape) != (prepared.logits.shape[0],):
            raise ValueError("target must contain one byte index per body")
        if target.dtype != torch.long or not bool(((0 <= target) & (target < self.byte_vocab)).all()):
            raise ValueError("target must contain byte indices")
        if step_size == 0:
            return

        self._update_route_from_tensors(
            prepared.logits, prepared.features, prepared.read_route,
            prepared.read_values, prepared.credit_token, target, float(step_size),
        )

    @torch.no_grad()
    def contact_response_factor(
        self,
        state: ColumnBodyState,
        prepared: ColumnBodyStep,
    ) -> torch.Tensor:
        """Return this arrival's exact local response factor, without storing it.

        The factor has shape ``[B, rows, cols, D, P]`` where ``P`` is the
        shared contact field's complete local pre-participation.  It is exactly
        ``post-response-gate × pre`` for every H-driving contact coordinate.
        A later boundary coefficient may contract its ``D`` axis, but this
        method does not retain that coefficient, add a state page, or prescribe
        an update.  It is the concrete bounded probe for whether existing H/C/S
        can carry an unresolved local eligibility across a delay.
        """
        self._validate_state(state)
        if prepared.contact_response_gate.shape != state.activity.shape:
            raise ValueError("prepared response gate does not match the source state")
        contact_pre = prepared.contact_pre
        if contact_pre.shape[:-1] != state.activity.shape[:-1]:
            raise ValueError("prepared local contact pre-state does not match the source state")
        return (prepared.contact_response_gate.to(dtype=contact_pre.dtype).unsqueeze(-1)
                * contact_pre.unsqueeze(-2))

    @torch.no_grad()
    def one_step_contact_score_direction(
        self,
        state: ColumnBodyState,
        prepared: ColumnBodyStep,
        following: torch.Tensor,
        *,
        elapsed: float | torch.Tensor = 0.0,
        source: SourceKind | bool | torch.Tensor = "external",
    ) -> torch.Tensor:
        """Return the exact one-next-score S-to-H direction without an update.

        This is the executable local contract behind
        ``GenerationContactEligibility``.  It uses only the pre-arrival S
        reading and the transient response gate produced by that arrival;
        the following language innovation supplies the current third factor.
        No factor is stored after this call and no parameter is changed.  It
        is therefore an admission oracle for a later multi-event carrier,
        never an online BPTT substitute.
        """
        self._validate_state(state)
        if following.ndim != 1 or tuple(following.shape) != (state.activity.shape[0],):
            raise ValueError("following must contain one byte index per body")
        if following.dtype != torch.long or not bool(((0 <= following) & (following < self.byte_vocab)).all()):
            raise ValueError("following must contain byte indices")
        if prepared.contact_response_gate.shape != state.activity.shape:
            raise ValueError("prepared response gate does not match the source state")

        next_prepared = self.step(prepared.state, following, elapsed=elapsed, source=source)
        innovation = self._language_innovation(next_prepared.logits, following)
        read_coefficient = innovation @ self.readout.weight
        next_read_route = self._state_route(prepared.state)
        local_score_coefficient = (
            read_coefficient[:, None, None, :]
            * next_read_route.unsqueeze(-1)
        )
        response_factor = self.contact_response_factor(state, prepared)
        return torch.einsum("brcd,brcdf->df", local_score_coefficient, response_factor)

    @staticmethod
    def _local_neighbourhood(route: torch.Tensor) -> torch.Tensor:
        """The event's routed columns and their four geometric neighbours."""
        selected = route > 0
        return (selected | selected.roll(1, dims=1) | selected.roll(-1, dims=1) |
                selected.roll(1, dims=2) | selected.roll(-1, dims=2))

    def _prepare_carrier_tensors(
        self,
        natural: torch.Tensor,
        precision: torch.Tensor,
        elapsed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the known continuous-time C prior before language is read.

        This is the executable `GDNLife.prepare` phase.  It has no access to
        the arriving byte and retains the same H/C/S body rather than creating
        a predictive workspace.
        """
        dt = elapsed.view(natural.shape[0], 1, 1, 1)
        process = F.softplus(self.process_variance_raw).to(dtype=precision.dtype)
        released_precision = precision / (1.0 + dt * process * precision)
        prior_precision = torch.where(dt == 0, precision, released_precision)
        prior_natural = torch.where(
            dt == 0, natural, prior_precision * (natural / precision),
        )
        return prior_natural, prior_precision

    def _predict_tensors(
        self,
        activity: torch.Tensor,
        natural: torch.Tensor,
        precision: torch.Tensor,
        causal_token: torch.Tensor,
        participation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read one prediction only from an already-prepared H/C/S state."""
        state = ColumnBodyState(activity, natural, precision, causal_token, participation)
        read_route = self._state_route(state)
        read_values = natural / precision + activity
        features = (read_route.unsqueeze(-1) * read_values).sum(dim=(1, 2))
        value_eligibility = self._value_eligibility_tensors(
            natural, precision, causal_token, participation,
        )
        return features, read_route, read_values, causal_token.clone(), value_eligibility

    def _assimilate_tensors(
        self,
        activity: torch.Tensor,
        natural: torch.Tensor,
        precision: torch.Tensor,
        participation: torch.Tensor,
        token: torch.Tensor,
        source_index: torch.Tensor,
        diagnostic_totals: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Write one arrived byte into the prepared local H/C/S body.

        The returned gate/pre/flux are the bounded receipt issued by this
        arrival.  They are transient event products, not a fourth body state.
        """
        batch = token.shape[0]
        event = self.byte_embedding(token)
        write_route = self._token_route(token)
        write_route_d = write_route.unsqueeze(-1)
        evidence_precision = F.softplus(self.event_likelihood(event)).view(batch, 1, 1, 1)
        write_precision = write_route_d * evidence_precision
        value = self.event_value(event).view(batch, 1, 1, self.d_model)
        dynamic_value = self.dynamics_value(event).view(batch, 1, 1, self.d_model)
        next_natural = natural + write_precision * value
        next_precision = precision + write_precision

        # The GDN value-credit theorem holds the H/S dynamics fixed with
        # respect to `event_value`.  Feed the reciprocal field C only from the
        # prepared prior; current-event content enters through `dynamics_value`.
        prior_mean = natural / precision
        prior_demand = precision.reciprocal() / (1.0 + precision.reciprocal())
        neighbour_activity = (
            activity.roll(1, dims=1) + activity.roll(-1, dims=1) +
            activity.roll(1, dims=2) + activity.roll(-1, dims=2)
        ) * 0.25
        contact_pre = torch.cat((
            activity, neighbour_activity, prior_mean, prior_demand,
            participation.flatten(start_dim=-3), write_route_d * dynamic_value,
        ), dim=-1)
        contact_response = F.linear(contact_pre, self.reciprocal_contact)
        flux, net_flux = self._edge_flux(contact_response)
        unmodulated_activity = contact_response[..., self._contact_h_output] + net_flux.unsqueeze(-1)
        proposed_activity = F.hardtanh(unmodulated_activity)
        local_write_support = self._local_neighbourhood(write_route).unsqueeze(-1)
        contact_response_gate = (
            local_write_support & (unmodulated_activity > -1.0) & (unmodulated_activity < 1.0)
        )
        next_activity = torch.where(local_write_support, proposed_activity, activity)
        if diagnostic_totals is not None:
            with torch.no_grad():
                diagnostic_totals[2].add_(contact_response_gate.sum())
                diagnostic_totals[3].add_(local_write_support.sum() * self.d_model)
        next_participation = self._reciprocal_contact_update(
            participation, contact_response, contact_response_gate, write_route, source_index,
            diagnostic_totals=diagnostic_totals,
        )
        return (
            next_activity, next_natural, next_precision, next_participation,
            write_route, flux, contact_response_gate, contact_pre,
        )

    def _step_tensors(
        self,
        activity: torch.Tensor,
        natural: torch.Tensor,
        precision: torch.Tensor,
        causal_token: torch.Tensor,
        participation: torch.Tensor,
        token: torch.Tensor,
        elapsed: torch.Tensor,
        source_index: torch.Tensor,
        diagnostic_totals: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compose `prepare → predict → assimilate` for one causal arrival."""
        prior_natural, prior_precision = self._prepare_carrier_tensors(
            natural, precision, elapsed,
        )
        features, _, _, credit_token, value_eligibility = self._predict_tensors(
            activity, prior_natural, prior_precision, causal_token, participation,
        )
        (activity, natural, precision, participation, write_route, flux,
         contact_response_gate, contact_pre) = self._assimilate_tensors(
            activity, prior_natural, prior_precision, participation, token, source_index,
            diagnostic_totals=diagnostic_totals,
        )
        # A live arrival becomes the query identity for the following causal
        # prediction.  No score reads this new token in the current event.
        return (activity, natural, precision, token.clone(), participation,
                features, write_route, flux, credit_token, value_eligibility,
                contact_response_gate, contact_pre)

    def step(
        self,
        state: ColumnBodyState,
        token: torch.Tensor,
        *,
        elapsed: float | torch.Tensor = 1.0,
        source: SourceKind | bool | torch.Tensor = "external",
        record_contact_return: bool = False,
        diagnostic_totals: torch.Tensor | None = None,
    ) -> ColumnBodyStep:
        """Score one byte from old Z, then assimilate that arrived event locally."""
        self._validate_state(state)
        if token.ndim != 1 or token.shape[0] != state.activity.shape[0]:
            raise ValueError("token must be one byte index per body")
        if token.dtype != torch.long or not bool(((0 <= token) & (token < self.byte_vocab)).all()):
            raise ValueError("token must contain byte indices")

        batch = token.shape[0]
        dt = self._as_elapsed(elapsed, batch, state.precision).reshape(batch)
        source_index = self._source_index(source, batch, state.activity)
        # These are one-event receipts, returned to the just-arrived language
        # boundary and never folded into H/C/S.  They are prepared from the
        # same elapsed-time prior as `_step_tensors`, so a route credit never
        # differentiates a pre-time state while the body writes a post-time
        # state.  This is the concrete fixed support/value pair required by
        # GenerationRoutingCredit.
        prior_state = self.advance_time(state, elapsed)
        read_route = self._state_route(prior_state)
        read_values = prior_state.natural / prior_state.precision + prior_state.activity
        (activity, natural, precision, causal_token, participation,
         features, route, flux, credit_token, value_eligibility,
         contact_response_gate, contact_pre) = self._step_tensors(
            state.activity, state.natural, state.precision, state.causal_token, state.participation,
            token, dt, source_index, diagnostic_totals=diagnostic_totals,
        )

        return ColumnBodyStep(
            state=ColumnBodyState(activity, natural, precision, causal_token, participation),
            logits=self.readout(features),
            features=features,
            read_route=read_route,
            read_values=read_values,
            route=route,
            edge_flux=flux,
            credit_token=credit_token,
            value_eligibility=value_eligibility,
            contact_response_gate=contact_response_gate,
            contact_pre=contact_pre,
            contact_return_weight=(
                self.reciprocal_contact[:self.d_model + 2, self._contact_s_input].detach().clone()
                if record_contact_return else None
            ),
        )
