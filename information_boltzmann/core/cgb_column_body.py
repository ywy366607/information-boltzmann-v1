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
    route: torch.Tensor          # [B, rows, cols], sparse post-arrival write participation
    edge_flux: torch.Tensor      # [B, rows, cols, 2], east/south oriented flow
    credit_token: torch.Tensor   # [B], C query responsible for this pre-score
    value_eligibility: torch.Tensor  # [B], reconstructed direct C sensitivity


@dataclass(frozen=True)
class ContactTeachingReceipt:
    """One consumed boundary result expressed through the existing body.

    This is an event-local receipt, never a fourth state page.  It makes the
    three factors required by an eventual S teaching rule explicit: language
    innovation at the boundary, the pre-arrival read support, and the C/S
    trace.  It deliberately contains no feedback matrix or update rule: such
    a route must first meet the reciprocal-alignment admission.
    """

    innovation: torch.Tensor    # [B, byte_vocab], arrived minus predicted
    route_index: torch.Tensor   # [B, active_columns], flattened old-C addresses
    route_weight: torch.Tensor  # [B, active_columns], old-C read weights
    trace: torch.Tensor         # [B, active_columns, 4, 2, rank], old C/S only


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
        self.birth_query = nn.Parameter(torch.zeros(d_model))
        self.causal_address = nn.Embedding(byte_vocab, d_model)
        # `event_value` is the only online-plastic value path.  Dynamics and
        # confidence have their own maps, so a credit update cannot mutate an
        # address, likelihood, H, or S path that the GDN theorem holds fixed.
        self.event_value = nn.Linear(d_model, d_model, bias=False)
        self.dynamics_value = nn.Linear(d_model, d_model, bias=False)
        self.event_likelihood = nn.Linear(d_model, 1)
        self.activity_carrier = nn.Linear(d_model, d_model, bias=False)
        self.readout = nn.Linear(d_model, byte_vocab)

        # Shared column rule F_theta.  It reads only local H/C and local
        # neighbour transfers.  A gain above one with hard saturation permits
        # retention; the learned release gate reads C, so contradictory local
        # evidence can release a held activity rather than freeze it forever.
        self.neighbour_mix = nn.Linear(d_model, d_model, bias=False)
        # S is a standing local contact, not a diagnostic-only ledger.  Its
        # old-state typed reading enters the same shared activity rule on the
        # next event; no extra feedback state is introduced.
        self.contact_to_activity = nn.Linear(4 * 2 * self.contact_rank, d_model, bias=False)
        self.activity_gain = nn.Linear(2 * d_model + 1, 1)
        self.activity_release = nn.Linear(2 * d_model + 1, 1)
        self.edge_flow = nn.Linear(2 * d_model + 1, 2)
        self.edge_participation = nn.Linear(2 * d_model + 1, 4 * self.contact_rank)
        self.participation_retain = nn.Linear(d_model + 1, 1)
        # A single local release and one trainable shared ratio generate the lifetime
        # ladder of the existing S rank coordinates.  This is not a rank-wise
        # clock or another memory array: rank k releases at r / gamma**k, with
        # gamma > 1.  It is the executable instance of the geometric two-state
        # bank required only when the language stream presents broad re-entry
        # delays (GenerationCascadeMemory).
        self.participation_scale_raw = nn.Parameter(torch.tensor(0.0))

        self.prior_precision_raw = nn.Parameter(torch.tensor(-2.0))
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
        prior = F.softplus(self.prior_precision_raw).to(device=device, dtype=dtype)
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
        """Typed generalized eligibility reconstructed solely from C and S."""
        return state.participation * self.demand(state).unsqueeze(-1).unsqueeze(-1)

    def participation_scale(self) -> torch.Tensor:
        """The trainable geometric separation of adjacent existing S ranks."""
        return 1.0 + F.softplus(self.participation_scale_raw)

    def _rank_retention(self, base_retain: torch.Tensor) -> torch.Tensor:
        """Generate rank-wise S retention from one local release.

        ``base_retain`` is the local F_theta output for rank zero.  Every
        higher rank receives the same local evidence but releases more slowly
        by a learned geometric factor.  The returned tensor is a readout of
        existing local quantities; S itself remains the only stored trace.
        """
        release = 1.0 - base_retain
        ranks = torch.arange(self.contact_rank, device=release.device, dtype=release.dtype)
        scale = self.participation_scale().to(dtype=release.dtype)
        return 1.0 - release.unsqueeze(-1) / scale.pow(ranks)

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
        query = torch.where(has_causal_token, self.causal_address(token),
                            self.birth_query.unsqueeze(0).expand(token.shape[0], -1))
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

    def _edge_flux(self, local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return oriented local flux and its exact endpoint net contribution.

        ``flux[..., 0]`` is eastward and ``flux[..., 1]`` southward.  Periodic
        incoming minus outgoing construction has zero total before any bounded
        activity nonlinearity, the executable form of local conservative
        transport rather than a separate global resource account.
        """
        flux = torch.tanh(self.edge_flow(local))
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
        local: torch.Tensor,
        route: torch.Tensor,
        source_index: torch.Tensor,
    ) -> torch.Tensor:
        """Update each selected local contact as an adaptive phase exchange.

        Each arrival coordinate ``h`` is the occupied part of an existing
        contact; its available partner is represented implicitly by ``1-h``.
        The learned local drive supplies the up-rate, and the existing
        H/C-dependent retention readout supplies the down-rate.  Thus the
        update is the already-admitted two-state exchange
        ``h + k_plus * (1-h) - k_minus * h``.  It forms a bounded local
        trace from routed events without a separately allocated trace page or
        a clock-selected decay constant.

        The two directed readings of an edge receive the same local rates, so
        reciprocal equality is preserved.
        """
        raw_drive = self.edge_participation(local).unflatten(-1, (4, self.contact_rank))
        endpoint_retain = torch.sigmoid(
            self.participation_retain(torch.cat(
                (local[..., self.d_model:2 * self.d_model], local[..., -1:]), dim=-1,
            ))
        ).squeeze(-1)
        edge_drive, edge_retain, edge_active = [], [], []
        for direction, (row_shift, col_shift) in enumerate(_GRID_DIRECTION_SHIFTS):
            opposite = _OPPOSITE_DIRECTION[direction]
            neighbour_drive = raw_drive[..., opposite, :].roll(
                shifts=(-row_shift, -col_shift), dims=(1, 2),
            )
            neighbour_retain = endpoint_retain.roll(
                shifts=(-row_shift, -col_shift), dims=(1, 2),
            )
            neighbour_route = route.roll(shifts=(-row_shift, -col_shift), dims=(1, 2))
            edge_drive.append(torch.sigmoid(0.5 * (raw_drive[..., direction, :] + neighbour_drive)))
            edge_retain.append(0.5 * (endpoint_retain + neighbour_retain))
            edge_active.append(torch.maximum(route, neighbour_route))
        drive = torch.stack(edge_drive, dim=-2)
        retain = self._rank_retention(torch.stack(edge_retain, dim=-1))
        active = torch.stack(edge_active, dim=-1)
        source_one_hot = F.one_hot(source_index, num_classes=2).to(dtype=participation.dtype)
        target = (drive.unsqueeze(-2) *
                  source_one_hot[:, None, None, None, :, None])
        return self._adaptive_phase_exchange(participation, target, retain, active)

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

    def contact_teaching_receipt(
        self,
        state: ColumnBodyState,
        prepared: ColumnBodyStep,
        target: torch.Tensor,
    ) -> ContactTeachingReceipt:
        """Expose only the causal factors available after a language arrival.

        `prepared.logits` was read from `state`; `target` is the just-arrived
        byte.  The returned factors are sufficient to audit a proposed local
        teaching path without keeping an autograd graph, event history, or a
        persistent feedback/eligibility tensor.  The method has no side
        effects and does not make S result-sensitive by itself.
        """
        self._validate_state(state)
        if target.ndim != 1 or tuple(target.shape) != (prepared.logits.shape[0],):
            raise ValueError("target must contain one byte index per prepared prediction")
        if target.dtype != torch.long or not bool(((0 <= target) & (target < self.byte_vocab)).all()):
            raise ValueError("target must contain byte indices")
        if prepared.logits.shape != (state.activity.shape[0], self.byte_vocab):
            raise ValueError("prepared prediction does not match state or byte boundary")
        read_route = self._state_route(state)
        route_weight, route_index = read_route.flatten(1).topk(self.active_columns, dim=-1)
        participation = state.participation.reshape(
            state.activity.shape[0], self.columns, 4, 2, self.contact_rank,
        )
        contact_index = route_index[..., None, None, None].expand_as(
            participation[:, :self.active_columns],
        )
        local_participation = participation.gather(1, contact_index)
        demand = self.demand(state).reshape(state.activity.shape[0], self.columns, 1)
        local_demand = demand.gather(1, route_index.unsqueeze(-1))
        return ContactTeachingReceipt(
            innovation=self._language_innovation(prepared.logits, target),
            route_index=route_index.detach(),
            route_weight=route_weight.detach(),
            trace=(local_participation * local_demand[..., None, None]).detach(),
        )

    @torch.no_grad()
    def contact_teaching_direction(
        self,
        state: ColumnBodyState,
        prepared: ColumnBodyStep,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Propose, but do not apply, the existing-value path's S direction.

        `event_value(byte_embedding(.))` is already the learned language-to-C
        value map.  Applying the arrived language innovation to that map gives
        an endogenous width-D teaching vector without a preset transpose or a
        new feedback parameter.  Multiplying it by the sparse C/S trace gives
        one direction for the existing S-to-H contact weights.

        This is intentionally *not* an online update.  Lean only admits it
        after its dot product with the exact future language score is measured
        positive.  The returned tensor is a short-horizon audit candidate,
        not a replacement for the proved direct C-value path.
        """
        receipt = self.contact_teaching_receipt(state, prepared, target)
        byte_values = self.event_value(self.byte_embedding.weight).detach()
        teaching = receipt.innovation @ byte_values
        direction = torch.einsum(
            "bd,ba,baqsr->dqsr",
            teaching, receipt.route_weight, receipt.trace,
        )
        return direction.flatten(start_dim=1)

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
    def _local_neighbourhood(route: torch.Tensor) -> torch.Tensor:
        """The event's routed columns and their four geometric neighbours."""
        selected = route > 0
        return (selected | selected.roll(1, dims=1) | selected.roll(-1, dims=1) |
                selected.roll(1, dims=2) | selected.roll(-1, dims=2))

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """Unchecked tensor kernel for one causal arrival.

        ``step`` remains the public, validated interface.  This factorisation
        exists so a long, fixed-shape online stream can compile the local
        tensor work without compiling Python validation or changing the
        H/C/S recurrence.  It deliberately returns no logits: prediction is a
        boundary readout from the returned *old-state* features.
        """
        state = ColumnBodyState(activity, natural, precision, causal_token, participation)
        batch = token.shape[0]
        event = self.byte_embedding(token)
        read_route = self._state_route(state)
        read_route_d = read_route.unsqueeze(-1)
        mean = natural / precision
        # H is dynamically useful but independent of `event_value`; this makes
        # the direct C contribution to the scored feature analytically exact.
        features = (read_route_d * (mean + activity)).sum(dim=(1, 2))
        value_eligibility = self._value_eligibility_tensors(
            natural, precision, causal_token, participation,
        )
        credit_token = causal_token.clone()

        dt = elapsed.view(batch, 1, 1, 1)
        process = F.softplus(self.process_variance_raw).to(dtype=precision.dtype)
        released_precision = precision / (1.0 + dt * process * precision)
        prior_precision = torch.where(dt == 0, precision, released_precision)
        prior_natural = torch.where(dt == 0, natural, prior_precision * mean)
        write_route = self._token_route(token)
        write_route_d = write_route.unsqueeze(-1)
        evidence_precision = F.softplus(self.event_likelihood(event)).view(batch, 1, 1, 1)
        write_precision = write_route_d * evidence_precision
        value = self.event_value(event).view(batch, 1, 1, self.d_model)
        dynamic_value = self.dynamics_value(event).view(batch, 1, 1, self.d_model)
        natural = prior_natural + write_precision * value
        precision = prior_precision + write_precision

        demand = precision.reciprocal() / (1.0 + precision.reciprocal())
        local = torch.cat((activity, self.activity_carrier(activity), demand), dim=-1)
        flux, net_flux = self._edge_flux(local)
        neighbour = self.neighbour_mix(
            (activity.roll(1, dims=1) + activity.roll(-1, dims=1) +
             activity.roll(1, dims=2) + activity.roll(-1, dims=2)) * 0.25
        )
        contact_message = self.contact_to_activity(participation.flatten(start_dim=-3))
        recurrent_gain = 1.0 + F.softplus(self.activity_gain(local))
        release = torch.sigmoid(self.activity_release(local))
        unmodulated_activity = (
            (1.0 - release) * recurrent_gain * activity + neighbour +
            contact_message + write_route_d * dynamic_value + net_flux.unsqueeze(-1)
        )
        proposed_activity = F.hardtanh(unmodulated_activity)
        activity = torch.where(
            self._local_neighbourhood(write_route).unsqueeze(-1), proposed_activity, activity,
        )

        participation = self._reciprocal_contact_update(
            participation, local, write_route, source_index,
        )
        # A live arrival becomes the query identity for the following causal
        # prediction.  No score reads this new token in the current event.
        return (activity, natural, precision, token.clone(), participation,
                features, write_route, flux, credit_token, value_eligibility)

    def step(
        self,
        state: ColumnBodyState,
        token: torch.Tensor,
        *,
        elapsed: float | torch.Tensor = 1.0,
        source: SourceKind | bool | torch.Tensor = "external",
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
        (activity, natural, precision, causal_token, participation,
         features, route, flux, credit_token, value_eligibility) = self._step_tensors(
            state.activity, state.natural, state.precision, state.causal_token, state.participation,
            token, dt, source_index,
        )

        return ColumnBodyStep(
            state=ColumnBodyState(activity, natural, precision, causal_token, participation),
            logits=self.readout(features),
            features=features,
            route=route,
            edge_flux=flux,
            credit_token=credit_token,
            value_eligibility=value_eligibility,
        )
