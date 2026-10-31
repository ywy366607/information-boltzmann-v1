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
_ARRIVAL = 0
_TEACHING = 1
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
    participation: torch.Tensor  # S: [B, rows, cols, 4, 2, 2, rank]


@dataclass(frozen=True)
class ColumnBodyStep:
    """Transient products of one causal event; none is retained implicitly."""

    state: ColumnBodyState
    logits: torch.Tensor         # prediction prepared from the old state
    features: torch.Tensor       # old-state local readout, transient only
    route: torch.Tensor          # [B, rows, cols], sparse event participation
    edge_flux: torch.Tensor      # [B, rows, cols, 2], east/south oriented flow


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
        # S uses one bounded basis coordinate per local contact factor.  Its
        # two *typed phases* are arrival participation and signed teaching
        # participation.  They are values of the same local contact carrier,
        # not a fourth persistent page.  Raising rank only supplies further
        # distinguishable causal directions.
        self.contact_rank = int(contact_rank)
        self.byte_vocab = int(byte_vocab)

        # Boundary coding and routing are learned, but all columns share the
        # same local dynamics below.  Keys select transiently active columns;
        # they are neither an MoE label nor a persistent event archive.
        self.byte_embedding = nn.Embedding(byte_vocab, d_model)
        self.column_keys = nn.Parameter(torch.empty(rows, cols, d_model))
        nn.init.normal_(self.column_keys, std=d_model ** -0.5)
        self.birth_query = nn.Parameter(torch.zeros(d_model))
        self.state_query = nn.Linear(d_model, d_model, bias=False)
        self.event_value = nn.Linear(d_model, d_model, bias=False)
        self.event_likelihood = nn.Linear(d_model, 1)
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

        self.prior_precision_raw = nn.Parameter(torch.tensor(-2.0))
        self.process_variance_raw = nn.Parameter(torch.tensor(-3.0))
        # These are deliberately registered *after* every pre-existing map:
        # turning the teaching path off therefore preserves the seeded default
        # body initialization used by prior H/C/S admission runs.
        # The teaching phase is modulatory: it can scale a local H transition
        # already driven by H/C/arrival, but cannot create activity from zero.
        # The two maps are initialized independently.  A matched local update
        # makes their splice mismatch contract; they are never copied as
        # transposes.
        contact_width = 4 * 2 * self.contact_rank
        self.teaching_to_activity = nn.Linear(contact_width, d_model, bias=False)
        self.activity_to_teaching = nn.Linear(d_model, contact_width, bias=False)
        # This boundary map carries an already-arrived language residual into
        # the existing typed S carrier.  Its full feedback alignment is an
        # admission measurement, not assumed by this implementation.
        self.error_to_teaching = nn.Linear(byte_vocab, contact_width, bias=False)

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
            participation=torch.zeros(*shape, 4, 2, 2, self.contact_rank, device=device, dtype=dtype),
        )

    @staticmethod
    def detach_state(state: ColumnBodyState) -> ColumnBodyState:
        """Cut numerical training history without removing causal state."""
        return ColumnBodyState(*(value.detach() for value in (
            state.activity, state.natural, state.precision, state.participation,
        )))

    @staticmethod
    def clone_state(state: ColumnBodyState) -> ColumnBodyState:
        """Create a complete causal checkpoint with no shared mutable storage."""
        return ColumnBodyState(*(value.detach().clone() for value in (
            state.activity, state.natural, state.precision, state.participation,
        )))

    @staticmethod
    def persistent_state_bytes(state: ColumnBodyState) -> dict[str, int]:
        """Report the physical H/C/S state separately from parameters/workspace."""
        sizes = {
            "H_activity": state.activity.numel() * state.activity.element_size(),
            "C_natural": state.natural.numel() * state.natural.element_size(),
            "C_precision": state.precision.numel() * state.precision.element_size(),
            "S_participation": state.participation.numel() * state.participation.element_size(),
        }
        return {**sizes, "total": sum(sizes.values())}

    def demand(self, state: ColumnBodyState) -> torch.Tensor:
        """Local write availability, reconstructed from C rather than stored."""
        return state.precision.reciprocal() / (1.0 + state.precision.reciprocal())

    def trace(self, state: ColumnBodyState) -> torch.Tensor:
        """Typed generalized eligibility reconstructed solely from C and S."""
        return state.participation * self.demand(state).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    @staticmethod
    def arrival_participation(state: ColumnBodyState) -> torch.Tensor:
        """Read the nonnegative arrival phase of the one S carrier."""
        return state.participation[..., _ARRIVAL, :]

    @staticmethod
    def teaching_participation(state: ColumnBodyState) -> torch.Tensor:
        """Read the signed, post-arrival teaching phase of the one S carrier."""
        return state.participation[..., _TEACHING, :]

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
        if state.participation.shape != (*state.activity.shape[:-1], 4, 2, 2, self.contact_rank):
            raise ValueError("participation must be one four-neighbour typed low-rank S state")
        arrival = self.arrival_participation(state)
        teaching = self.teaching_participation(state)
        if (not bool(torch.isfinite(state.participation).all()) or
                not bool(((0.0 <= arrival) & (arrival <= 1.0)).all()) or
                not bool(((-1.0 <= teaching) & (teaching <= 1.0)).all())):
            raise ValueError("arrival S must lie in [0, 1] and teaching S in [-1, 1]")
        if not bool(torch.isfinite(state.precision).all()) or not bool((state.precision > 0).all()):
            raise ValueError("carrier precision must be finite and positive")

    def _state_route(self, state: ColumnBodyState) -> torch.Tensor:
        """Prepare a sparse route from old C/H, before the byte arrives."""
        mean = state.natural / state.precision
        query = self.birth_query + self.state_query((state.activity + mean).mean(dim=(1, 2)))
        query = F.normalize(query, dim=-1, eps=1e-6)
        keys = F.normalize(self.column_keys, dim=-1, eps=1e-6)
        scores = torch.einsum("bd,hwd->bhw", query, keys)
        flat_scores = scores.flatten(1)
        values, indices = flat_scores.topk(self.active_columns, dim=-1)
        sparse = torch.zeros_like(flat_scores)
        sparse.scatter_(1, indices, torch.softmax(values, dim=-1))
        return sparse.view_as(scores)

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
    def reciprocal_participation_mismatch(
        participation: torch.Tensor,
        *,
        phase: Literal["arrival", "teaching"] = "arrival",
    ) -> torch.Tensor:
        """Read each directed contact minus its reciprocal endpoint reading.

        This is a diagnostic readout of S, not another state array.  Direction
        order is north, east, south, west; source type remains the final axis.
        A zero result says that the two endpoint readings of every typed local
        contact agree.
        """
        if participation.ndim != 7 or participation.shape[-4:-1] != (4, 2, 2):
            raise ValueError("participation must have shape [B, rows, cols, 4, 2, 2, rank]")
        phase_index = _ARRIVAL if phase == "arrival" else _TEACHING if phase == "teaching" else None
        if phase_index is None:
            raise ValueError("phase must be arrival or teaching")
        participation = participation[..., phase_index, :]
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
        """Update each selected local contact from one endpoint-shared statistic.

        The two directed readings of an edge receive the same learned
        heterosynaptic drive and retention coefficient.  Hence their mismatch
        contracts whenever that edge participates, exactly matching the
        ``reciprocal_shared_update_mismatch`` admission kernel.  This creates
        no feedback page: it rewrites the existing source-typed S contact.
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
        retain = torch.stack(edge_retain, dim=-1)
        active = torch.stack(edge_active, dim=-1)
        arrival = participation[..., _ARRIVAL, :]
        source_one_hot = F.one_hot(source_index, num_classes=2).to(dtype=arrival.dtype)
        target = (drive.unsqueeze(-2) *
                  source_one_hot[:, None, None, None, :, None])
        arrival = arrival + active.unsqueeze(-1).unsqueeze(-1) * (
            retain.unsqueeze(-1).unsqueeze(-1) * arrival + target - arrival
        )
        return torch.stack((arrival.clamp(0.0, 1.0), participation[..., _TEACHING, :]), dim=-2)

    def _reciprocal_teaching_update(
        self,
        participation: torch.Tensor,
        local: torch.Tensor,
        route: torch.Tensor,
        teaching: torch.Tensor,
    ) -> torch.Tensor:
        """Write an arrived language residual into signed S at both edge ends.

        ``teaching`` is transient boundary evidence with one contact coordinate
        per batch item.  Its edge target and release are symmetrised from the
        two endpoints, so the resulting directed S readings satisfy the same
        reciprocal splice condition as the arrival phase.  The write happens
        only after the byte's old-state prediction was scored.
        """
        if teaching.shape != (participation.shape[0], 4 * 2 * self.contact_rank):
            raise ValueError("teaching must provide one signed local contact vector per body")
        batch = participation.shape[0]
        teaching = teaching.reshape(batch, 4, 2, self.contact_rank)
        endpoint_retain = torch.sigmoid(
            self.participation_retain(torch.cat(
                (local[..., self.d_model:2 * self.d_model], local[..., -1:]), dim=-1,
            ))
        ).squeeze(-1)
        edge_target, edge_retain, edge_active = [], [], []
        for direction, (row_shift, col_shift) in enumerate(_GRID_DIRECTION_SHIFTS):
            opposite = _OPPOSITE_DIRECTION[direction]
            neighbour_route = route.roll(shifts=(-row_shift, -col_shift), dims=(1, 2))
            # The same batch-level residual is offered to both physical ends;
            # only local route participation decides whether that edge writes.
            edge_target.append(0.5 * (teaching[:, direction] + teaching[:, opposite])[:, None, None])
            edge_retain.append(0.5 * (
                endpoint_retain + endpoint_retain.roll(shifts=(-row_shift, -col_shift), dims=(1, 2))
            ))
            edge_active.append(torch.maximum(route, neighbour_route))
        target = torch.stack(edge_target, dim=-3)
        retain = torch.stack(edge_retain, dim=-1)
        active = torch.stack(edge_active, dim=-1)
        previous = participation[..., _TEACHING, :]
        teaching_state = previous + active.unsqueeze(-1).unsqueeze(-1) * (
            retain.unsqueeze(-1).unsqueeze(-1) * previous + target - previous
        )
        return torch.stack((participation[..., _ARRIVAL, :], teaching_state.clamp(-1.0, 1.0)), dim=-2)

    def reciprocal_teaching_mismatch(self) -> torch.Tensor:
        """The learnable splice error covered by the Lean reciprocal theorem."""
        return self.teaching_to_activity.weight - self.activity_to_teaching.weight.transpose(0, 1)

    def _teaching_contact_from_arrival(
        self,
        prepared: ColumnBodyStep,
        innovation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read a local eligibility and gate it by the arrived third factor."""
        mass = prepared.route.sum(dim=(1, 2)).clamp_min(
            torch.finfo(prepared.state.activity.dtype).eps
        )
        local_activity = (
            prepared.route.unsqueeze(-1) * prepared.state.activity
        ).sum(dim=(1, 2)) / mass.unsqueeze(-1)
        eligibility = self.activity_to_teaching(local_activity)
        third_factor = self.error_to_teaching(innovation)
        # This is the operational three-factor contact write: existing local
        # H-to-S sensitivity, multiplied by a residual only available after
        # the byte arrives.  It is bounded before entering physical S.
        return torch.tanh(eligibility * third_factor), local_activity

    def teaching_alignment_to_g14(
        self,
        prepared: ColumnBodyStep,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine of local teaching and exact immediate affine G14 directions.

        This is an admission measurement.  It neither substitutes the exact
        G14 update nor turns splice contraction into an unearned claim of
        whole-path alignment.
        """
        innovation = -torch.softmax(prepared.logits.detach(), dim=-1)
        innovation[torch.arange(target.shape[0], device=target.device), target] += 1
        exact = innovation @ self.readout.weight.detach()
        contact, _ = self._teaching_contact_from_arrival(prepared, innovation)
        teaching = self.teaching_to_activity(contact)
        return F.cosine_similarity(teaching, exact, dim=-1, eps=1e-8)

    def _teaching_release(self, state: ColumnBodyState, route: torch.Tensor) -> torch.Tensor:
        """One shared event release read from already-written local H/C."""
        mean = state.natural / state.precision
        local = torch.cat((state.activity, mean, self.demand(state)), dim=-1)
        mass = route.sum(dim=(1, 2)).clamp_min(torch.finfo(state.activity.dtype).eps)
        release = ((route * torch.sigmoid(self.activity_release(local)).squeeze(-1)).sum(dim=(1, 2)) /
                   mass).mean()
        return release.clamp(0.0, 1.0)

    @torch.no_grad()
    def settle_reciprocal_teaching_from_arrival(
        self,
        prepared: ColumnBodyStep,
        target: torch.Tensor,
        *,
        step_size: float,
    ) -> ColumnBodyState:
        """Settle one post-arrival, three-factor reciprocal teaching event.

        The byte residual is the already-arrived third factor.  The route-
        weighted local activity is the first factor and the signed S write is
        the finite history/participation factor.  Both H-to-S and S-to-H maps
        receive the identical local outer product and identical release; thus
        their splice mismatch contracts exactly as
        ``reciprocal_shared_update_mismatch`` states.  No transpose is copied
        and no past autograd graph is retained.  This is deliberately narrower
        than a claim that the complete feedback path is already aligned with
        the exact G14 score.
        """
        if not isinstance(step_size, (float, int)) or isinstance(step_size, bool) or step_size < 0:
            raise ValueError("step_size must be a finite nonnegative scalar")
        if not torch.isfinite(torch.as_tensor(step_size)):
            raise ValueError("step_size must be finite")
        if target.ndim != 1 or tuple(target.shape) != (prepared.logits.shape[0],):
            raise ValueError("target must contain one byte index per prepared prediction")
        if target.dtype != torch.long or not bool(((0 <= target) & (target < self.byte_vocab)).all()):
            raise ValueError("target must contain byte indices")

        innovation = -torch.softmax(prepared.logits.detach(), dim=-1)
        innovation[torch.arange(target.shape[0], device=target.device), target] += 1
        residual_contact, local_activity = self._teaching_contact_from_arrival(prepared, innovation)
        state = prepared.state
        mean = state.natural / state.precision
        demand = self.demand(state)
        local = torch.cat((state.activity, mean, demand), dim=-1)
        written = self._reciprocal_teaching_update(
            state.participation, local, prepared.route, residual_contact,
        )

        # A common event-level release gate is enough for the theorem: every
        # entry of the reciprocal pair receives the same scalar and shared
        # outer product at this event.  The physical release is on the same
        # update scale as the outer-product write; applying the raw [0,1]
        # gate as full per-byte weight decay would be a different, unjustified
        # parameter dynamics.
        release = float(step_size) * self._teaching_release(state, prepared.route)
        shared = (local_activity.transpose(0, 1) @ residual_contact) / local_activity.shape[0]
        factor = 1.0 - release
        self.teaching_to_activity.weight.mul_(factor).add_(shared, alpha=float(step_size))
        self.activity_to_teaching.weight.mul_(factor).add_(shared.transpose(0, 1), alpha=float(step_size))
        return ColumnBodyState(state.activity, state.natural, state.precision, written)

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
            participation=state.participation,
        )

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
        innovation = -torch.softmax(prepared.logits.detach(), dim=-1)
        innovation[torch.arange(target.shape[0], device=target.device), target] += 1
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
        participation: torch.Tensor,
        token: torch.Tensor,
        elapsed: torch.Tensor,
        source_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unchecked tensor kernel for one causal arrival.

        ``step`` remains the public, validated interface.  This factorisation
        exists so a long, fixed-shape online stream can compile the local
        tensor work without compiling Python validation or changing the
        H/C/S recurrence.  It deliberately returns no logits: prediction is a
        boundary readout from the returned *old-state* features.
        """
        state = ColumnBodyState(activity, natural, precision, participation)
        batch = token.shape[0]
        event = self.byte_embedding(token)
        route = self._state_route(state)
        route_d = route.unsqueeze(-1)
        mean = natural / precision
        features = (route_d * (mean + activity)).sum(dim=(1, 2))

        dt = elapsed.view(batch, 1, 1, 1)
        process = F.softplus(self.process_variance_raw).to(dtype=precision.dtype)
        released_precision = precision / (1.0 + dt * process * precision)
        prior_precision = torch.where(dt == 0, precision, released_precision)
        prior_natural = torch.where(dt == 0, natural, prior_precision * mean)
        evidence_precision = F.softplus(self.event_likelihood(event)).view(batch, 1, 1, 1)
        write_precision = route_d * evidence_precision
        value = self.event_value(event).view(batch, 1, 1, self.d_model)
        natural = prior_natural + write_precision * value
        precision = prior_precision + write_precision

        demand = precision.reciprocal() / (1.0 + precision.reciprocal())
        local = torch.cat((activity, mean, demand), dim=-1)
        flux, net_flux = self._edge_flux(local)
        neighbour = self.neighbour_mix(
            (activity.roll(1, dims=1) + activity.roll(-1, dims=1) +
             activity.roll(1, dims=2) + activity.roll(-1, dims=2)) * 0.25
        )
        arrival_contacts = participation[..., _ARRIVAL, :]
        teaching_contacts = participation[..., _TEACHING, :]
        contact_message = self.contact_to_activity(arrival_contacts.flatten(start_dim=-3))
        teaching_modulation = torch.tanh(
            self.teaching_to_activity(teaching_contacts.flatten(start_dim=-3))
        )
        recurrent_gain = 1.0 + F.softplus(self.activity_gain(local))
        release = torch.sigmoid(self.activity_release(local))
        unmodulated_activity = (
            (1.0 - release) * recurrent_gain * activity + neighbour +
            contact_message + route_d * value + net_flux.unsqueeze(-1)
        )
        # Teaching is a later, modulatory stream.  It cannot fabricate an H
        # event from an otherwise zero local transition; any effect appears
        # only on this subsequent recurrence after the residual was written.
        proposed_activity = F.hardtanh(unmodulated_activity * (1.0 + teaching_modulation))
        activity = torch.where(
            self._local_neighbourhood(route).unsqueeze(-1), proposed_activity, activity,
        )

        participation = self._reciprocal_contact_update(
            participation, local, route, source_index,
        )
        return (activity, natural, precision, participation,
                features, route, flux)

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
        (activity, natural, precision, participation,
         features, route, flux) = self._step_tensors(
            state.activity, state.natural, state.precision, state.participation,
            token, dt, source_index,
        )

        return ColumnBodyStep(
            state=ColumnBodyState(activity, natural, precision, participation),
            logits=self.readout(features),
            features=features,
            route=route,
            edge_flux=flux,
        )
