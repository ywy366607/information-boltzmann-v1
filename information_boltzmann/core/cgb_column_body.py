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
    participation: torch.Tensor  # S: [B, rows, cols, 4, 2, rank], in [0, 1]


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
        # S uses one bounded basis coordinate per local contact factor.  It is
        # the same carrier at every rank: raising this only supplies further
        # distinguishable causal directions, never a separate trace page.
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
            participation=torch.zeros(*shape, 4, 2, self.contact_rank, device=device, dtype=dtype),
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
        return state.participation * self.demand(state).unsqueeze(-1).unsqueeze(-1)

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
        if state.participation.shape != (*state.activity.shape[:-1], 4, 2, self.contact_rank):
            raise ValueError("participation must be one four-neighbour typed low-rank S state")
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
    def reciprocal_participation_mismatch(participation: torch.Tensor) -> torch.Tensor:
        """Read each directed contact minus its reciprocal endpoint reading.

        This is a diagnostic readout of S, not another state array.  Direction
        order is north, east, south, west; source type remains the final axis.
        A zero result says that the two endpoint readings of every typed local
        contact agree.
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
        source_one_hot = F.one_hot(source_index, num_classes=2).to(dtype=participation.dtype)
        target = (drive.unsqueeze(-2) *
                  source_one_hot[:, None, None, None, :, None])
        return participation + active.unsqueeze(-1).unsqueeze(-1) * (
            retain.unsqueeze(-1).unsqueeze(-1) * participation + target - participation
        )

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
        prior_precision = precision / (1.0 + dt * process * precision)
        prior_natural = prior_precision * mean
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
        contact_message = self.contact_to_activity(participation.flatten(start_dim=-3))
        recurrent_gain = 1.0 + F.softplus(self.activity_gain(local))
        release = torch.sigmoid(self.activity_release(local))
        proposed_activity = F.hardtanh(
            (1.0 - release) * recurrent_gain * activity + neighbour +
            contact_message + route_d * value + net_flux.unsqueeze(-1)
        )
        activity = torch.where(
            self._local_neighbourhood(route).unsqueeze(-1), proposed_activity, activity,
        )

        participation = self._reciprocal_contact_update(
            participation, local, route, source_index,
        )
        return (activity, natural, precision, participation.clamp(0.0, 1.0),
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
