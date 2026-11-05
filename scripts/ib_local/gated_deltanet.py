"""Capacity-matched and parameter-matched Gated DeltaNet baseline for BPE language modeling.

This model serves as the unconstrained (non-conservative) baseline to benchmark against
the Continuous Boltzmann Information Medium (CBIM). While CBIM enforces exact unitary
scattering, energy conservation, Cayley spatial transport, and passive quadratic radiation,
Gated DeltaNet (GDN) utilizes unconstrained associative memory matrices updated via
the delta rule:

    S_t = alpha_t * S_{t-1} + beta_t * k_t (v_t - S_{t-1}^T k_t)^T

Parameters, vocabulary, embedding tying, and training conditions are matched 1:1 against
CBIM Torus3D (8x8x4, d=128) with ~332k non-embedding parameters.

The original defaults remain the historical baseline. Explicit ``decay_then_delta``
and ``mlp_hidden`` options provide a reusable recurrent language backbone. Optional
``conv_kernel`` adds causal depthwise QKV convolution followed by SiLU, with its
raw projected-input cache packed alongside the recurrent matrix. These options do
not certify CGB conservation, long-horizon credit or healthy latching. Persist
``config_dict()`` alongside parameters AND recurrent state when checkpointing.

Optional ``content_budget`` admits actual signed memory into row AND column L1
capacity balls after each write. Empty capacity is implicit, with no allocation
array. This is a capacity constraint, not equal-marginal Sinkhorn transport or
the persistent evidence-exchange/latch rule. Disabled by default.

``evidence_exchange`` is a different opt-in recurrence in the same neural stack:
weighted content and one positive share per key row evolve jointly. It adds
heads*key_dimension scalars per layer, no learned parameters. It uses a Gaussian
latent-value likelihood; it is not mathematically equivalent to Gated DeltaNet,
a next-byte posterior, or a validated language organism. Its same-law stable
memories/release have conditional numerical evidence, not a general guarantee.

``contact_exchange`` uses the same state size, storing conditional content C
instead of M=a*C. The shared posterior assigns local write rates gamma*r_i;
the actual read still uses a*C. Configured recurrence is part of checkpoint
semantics: old weighted matrices must not be reinterpreted as conditional ones.

``feedback_delta`` keeps the same associative matrix. Its pre-write read feeds
softsign(gain*read + arrived_value), mixed with arrived_value before delta write.
Two learned scalars per head control mixture and gain; no new recurrent array.
The language value projection supplies the cue, never future labels. The fixed-key
Lean certificate is conditional and is not a language-latch guarantee. Optional
capacity admission follows the write and can change the certified dynamics.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def admit_association_content(proposed: torch.Tensor, budget: float) -> torch.Tensor:
    """Retract [...,key,value] into simultaneous row/column L1 budgets.

    Exact reference: reviews/budgeted_association_write_20261103.lean.
    U_ij *= b / max(b, sum_l |U_il|, sum_l |U_lj|). Feasible content is
    unchanged. This is not the nearest Euclidean projection; an overfull
    contact can alter old associations sharing either endpoint. Its correction
    must be included in readout/retention accounting. No data-dependent Python
    branch or persistent tensor is introduced; CUDA Graph capture is supported.
    Floating-point sums/casts incur ordinary rounding, not exact conservation.
    """
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("content budget must be finite and positive")
    work = proposed.float() if proposed.dtype in (torch.float16, torch.bfloat16) else proposed
    magnitude = work.abs()
    row = magnitude.sum(-1, keepdim=True)
    column = magnitude.sum(-2, keepdim=True)
    denominator = torch.maximum(row, column).clamp_min(budget)
    return (work * (budget / denominator)).to(proposed.dtype)


def exchange_association_content(content, share, address, value, gain, rate, budget):
    """One joint update of weighted content M and its row shares a.

    M_i = a_i c_i, ||c_i||_1 <= budget. The bounded arrived value and the
    stored conditional contents define Gaussian *latent* evidence, not the
    next-byte likelihood or a dopamine surrogate. Actual language CE trains
    address/value/gain/rate through the real readout. No target labels enter.
    gain/rate have shape [...,1]; address/share [...,keys].
    """
    magnitude = value.abs().sum(-1, keepdim=True)
    value = value * (budget / magnitude.clamp_min(budget))
    # Zero share has zero weighted content. Its forgotten conditional mean is
    # represented by zero; the fresh softmax prior can recruit it again.
    denominator = torch.where(share > 0, share, torch.ones_like(share))
    conditional = content / denominator.unsqueeze(-1)
    log_evidence = -0.5 * (conditional - value.unsqueeze(-2)).square().sum(-1)
    posterior = (gain * share + address + log_evidence).softmax(-1)
    next_share = (1-rate) * share + rate * posterior
    next_content = ((1-rate).unsqueeze(-1) * content
                    + rate.unsqueeze(-1) * posterior.unsqueeze(-1) * value.unsqueeze(-2))
    return next_content, next_share


def contact_exchange_content(conditional, share, address, value, gain, rate, budget,
                             availability=False):
    """Shared evidence, allocation and contact-local content learning.

    State is (C,a), with ||C_i||_1<=budget and a on the simplex. Both updates
    read the SAME pre-event snapshot. Sum_i gamma*r_i=gamma distributes a
    finite write rate; M=a*C is decoded only when needed, avoiding division
    by tiny shares. Zero-share content can influence subsequent recruitment.
    Exact-real reference: budgeted_association_write_20261103.lean.
    Optional availability uses the SAME share's unoccupied fraction, so
    posterior_i is proportional to exp(evidence_i)*(1-a_i). It attenuates
    admission at fixed evidence and releases a completely occupied contact.
    It does not guarantee uniform usage or useful long-term memory.
    """
    magnitude = value.abs().sum(-1, keepdim=True)
    value = value * (budget / magnitude.clamp_min(budget))
    log_evidence = -0.5 * (conditional-value.unsqueeze(-2)).square().sum(-1)
    logits = gain*share + address + log_evidence
    if availability:
        room = (1-share).clamp_min(0)
        # Keep exact zeros excluded and avoid log(0) derivatives. With >=2
        # contacts on the simplex, at least one room is positive. Log space
        # avoids underflow of raw exp(logits)*(1-a) at a concentrated state.
        logits = logits + room.clamp_min(torch.finfo(room.dtype).tiny).log()
        logits = logits.masked_fill(room == 0, -torch.inf)
    posterior = logits.softmax(-1)
    local_rate = (rate*posterior).unsqueeze(-1)
    next_conditional = (1-local_rate)*conditional + local_rate*value.unsqueeze(-2)
    next_share = (1-rate)*share + rate*posterior
    return next_conditional, next_share


def feedback_delta_step(memory, key, value, alpha, beta, strength, gain):
    """Same-state renewal, before optional capacity admission.

    key/value are column vectors; alpha/beta/strength/gain broadcast over the
    last two dimensions. This is the exact family of feedback_potential_step,
    with cue=value. Full derivatives include the read-back path. Fixed event
    potential descent does not imply future language improvement.
    """
    read = torch.matmul(memory.transpose(-1, -2), key)
    target = value + strength * (F.softsign(gain * read + value) - value)
    error = target - alpha * read
    return alpha * memory + beta * torch.matmul(key, error.transpose(-1, -2)), error


class GatedDeltaNetLayer(nn.Module):
    """Single layer of Gated DeltaNet with data-dependent decay and learning rate."""

    def __init__(self, d: int = 128, heads: int = 4,
                 recurrence: str = "legacy", mlp_hidden: int = 0,
                 conv_kernel: int = 0, content_budget: float = 0.0,
                 contact_availability: bool = False, decay_init: str = "constant",
                 feedback_init: str = "constant"):
        super().__init__()
        if decay_init not in ("constant", "log_timescale"):
            raise ValueError("decay_init must be 'constant' or 'log_timescale'")
        if feedback_init not in ("constant", "threshold_spread"):
            raise ValueError("feedback_init must be 'constant' or 'threshold_spread'")
        if feedback_init != "constant" and recurrence != "feedback_delta":
            raise ValueError("feedback_init requires feedback_delta")
        if decay_init != "constant" and recurrence not in ("decay_then_delta", "feedback_delta"):
            raise ValueError("decay_init requires decay_then_delta or feedback_delta")
        self.decay_init, self.feedback_init = decay_init, feedback_init
        if d <= 0 or heads <= 0 or d % heads != 0:
            raise ValueError(f"d ({d}) must be divisible by heads ({heads})")
        if recurrence not in ("legacy", "decay_then_delta", "feedback_delta", "evidence_exchange", "contact_exchange"):
            raise ValueError("unknown recurrence")
        if mlp_hidden < 0:
            raise ValueError("mlp_hidden must be nonnegative")
        if not isinstance(conv_kernel, int) or conv_kernel < 0:
            raise ValueError("conv_kernel must be a nonnegative integer (0 disables it)")
        if not math.isfinite(content_budget) or content_budget < 0:
            raise ValueError("content_budget must be finite and nonnegative")
        if content_budget and recurrence == "legacy":
            raise ValueError("content_budget requires a modern recurrence")
        if recurrence in ("evidence_exchange", "contact_exchange") and not content_budget:
            raise ValueError("exchange recurrences require positive content_budget")
        if contact_availability and (recurrence != "contact_exchange" or d // heads < 2):
            raise ValueError("contact_availability requires contact_exchange and at least two rows")
        self.contact_availability = bool(contact_availability)
        self.content_budget = float(content_budget)
        self.conv_kernel = conv_kernel
        self.recurrence = recurrence
        self.d, self.heads = d, heads
        self.dk = self.dv = d // heads
        self.matrix_size = heads * self.dk * self.dv
        self.share_size = heads * self.dk if recurrence in ("evidence_exchange", "contact_exchange") else 0
        self.history_size = 3 * d * max(0, conv_kernel - 1)
        self.state_size = self.matrix_size + self.share_size + self.history_size

        self.norm = nn.RMSNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.g_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

        self.alpha_proj = nn.Linear(d, heads, bias=decay_init == "constant")
        self.beta_proj = nn.Linear(d, heads, bias=True)

        # Initializations:
        if decay_init == "constant":
            # Retention bias positive: default to holding memory (~0.92 retention on step 0)
            nn.init.constant_(self.alpha_proj.bias, -2.5 if self.share_size else 2.5)
        else:
            # Official Gated DeltaNet (Mamba-2) parameterization:
            # log alpha = -exp(A_log) * softplus(a(x) + dt_bias), A ~ U(0, 16),
            # dt log-uniform on [1e-3, 1e-1] with floor 1e-4. Heads are born with
            # timescales from about one event to tens of thousands, and the timescale
            # moves on a log scale instead of through a saturating sigmoid.
            A = torch.empty(heads).uniform_(0, 16)
            self.A_log = nn.Parameter(torch.log(A))
            dt = torch.exp(torch.rand(heads) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
            dt = dt.clamp_min(1e-4)
            self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        nn.init.constant_(self.beta_proj.bias, 0.0)
        nn.init.normal_(self.o_proj.weight, std=0.02 / math.sqrt(2 * 4))
        self.mlp = None
        if mlp_hidden:
            self.mlp_norm = nn.RMSNorm(d)
            self.mlp = nn.ModuleDict({
                "gate": nn.Linear(d, mlp_hidden, bias=False),
                "up": nn.Linear(d, mlp_hidden, bias=False),
                "down": nn.Linear(mlp_hidden, d, bias=False),
            })
        if recurrence == "feedback_delta" and feedback_init == "constant":
            # Constants do not consume RNG: shared weights match the baseline.
            self.feedback_logit = nn.Parameter(torch.full((heads,), math.log(.1/.9)))
            self.feedback_gain_raw = nn.Parameter(torch.full((heads,), math.log(math.expm1(2.))))
        elif recurrence == "feedback_delta":
            # Spread the heads across the latch threshold beta*eta*g > 1 - alpha(1 - beta)
            # (section 9.5), evaluated at the birth retention with a(x)=0 and beta=1/2:
            # ratio 0 (plain delta), then 0.5 to 3 geometrically. Gain 8 keeps eta <= 3/4.
            # Deterministic from the drawn timescales: no extra RNG, so the shared
            # weights match a decay_then_delta control with the same decay_init.
            gain, beta0 = 8.0, 0.5
            alpha0 = self._birth_retention()
            threshold = 1 - alpha0 * (1 - beta0)
            ratio = torch.zeros(heads)
            if heads > 1:
                steps = torch.arange(heads - 1, dtype=torch.float32)
                ratio[1:] = 0.5 * 6.0 ** (steps / max(heads - 2, 1))
            eta = (ratio * threshold / (beta0 * gain)).clamp(1e-3, 0.999)
            self.feedback_logit = nn.Parameter(torch.log(eta / (1 - eta)))
            self.feedback_gain_raw = nn.Parameter(torch.full((heads,), math.log(math.expm1(gain))))
        self.qkv_conv = None
        if conv_kernel:
            self.qkv_conv = nn.Conv1d(3 * d, 3 * d, conv_kernel,
                                      groups=3 * d, bias=False)

    def retention(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Per-event, per-head retention alpha in (0, 1), shape [B, T, H]."""
        if self.decay_init == "constant":
            return torch.sigmoid(self.alpha_proj(x_norm))
        rate = self.A_log.exp() * F.softplus(self.alpha_proj(x_norm) + self.dt_bias)
        return torch.exp(-rate)

    @torch.no_grad()
    def _birth_retention(self) -> torch.Tensor:
        """Retention at a zero input projection, per head (used only for initialization)."""
        if self.decay_init == "constant":
            return torch.sigmoid(self.alpha_proj.bias.detach().float())
        return torch.exp(-self.A_log.exp() * F.softplus(self.dt_bias)).float()

    def _mix_qkv(self, projected: torch.Tensor, history: torch.Tensor):
        """[B,3d,T] raw projections; oldest-to-newest history [B,3d,K-1].

        No right padding: output t sees only history and inputs through t.
        Cache raw projections, not convolved/activated values; do not detach here.
        """
        joined = torch.cat((history, projected), dim=-1)
        mixed = F.silu(self.qkv_conv(joined))
        keep = self.conv_kernel - 1
        following = joined[..., -keep:] if keep else joined[..., :0]
        return mixed, following

    def forward(
        self, x: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass over sequence of length T with recurrent state S.

        Args:
            x: Input tensor of shape [B, T, d].
            state: [B,H,dk,dv] for ordinary recurrence without convolution;
                otherwise [B,N] packed matrix, optional row shares, raw QKV
                history. N is fixed by configuration.

        Returns:
            x_next: Updated tensor [B, T, d].
            state_next: Updated matrix, optional shares and convolution history.
            diagnostics: Dict of scalar monitoring metrics.
        """
        B, T, d = x.shape
        if T == 0:
            raise ValueError("at least one input event is required")
        x_norm = self.norm(x)

        q = self.q_proj(x_norm).view(B, T, self.heads, self.dk)
        k = self.k_proj(x_norm).view(B, T, self.heads, self.dk)
        v = self.v_proj(x_norm).view(B, T, self.heads, self.dv)
        if self.conv_kernel or self.share_size:
            S = state[:, :self.matrix_size].reshape(B, self.heads, self.dk, self.dv)
        else:
            S = state
        if self.share_size:
            share = state[:, self.matrix_size:self.matrix_size+self.share_size].reshape(B, self.heads, self.dk)
        if self.conv_kernel:
            history = state[:, self.matrix_size+self.share_size:].reshape(B, 3*d, self.conv_kernel-1)
            projected = torch.cat((q.reshape(B, T, d), k.reshape(B, T, d),
                                   v.reshape(B, T, d)), dim=-1).transpose(1, 2)
            mixed, next_history = self._mix_qkv(projected, history)
            q, k, v = [part.reshape(B, T, self.heads, self.dk)
                       for part in mixed.transpose(1, 2).split(d, dim=-1)]
        g = self.g_proj(x_norm)

        alpha = self.retention(x_norm)                  # [B, T, H]
        beta = torch.sigmoid(self.beta_proj(x_norm))    # [B, T, H]

        q = F.normalize(q, dim=-1)
        if not self.share_size:
            k = F.normalize(k, dim=-1)

        outputs = []
        errors = []
        if self.recurrence == "feedback_delta":
            strength = self.feedback_logit.sigmoid()[None, :, None, None]
            feedback_gain = F.softplus(self.feedback_gain_raw)[None, :, None, None]

        for t in range(T):
            if self.share_size:
                # Size-normalized interaction strength spans the symmetry-breaking
                # threshold g=keys. Initialize below it; CE can learn either regime.
                exchange = (contact_exchange_content if self.recurrence == "contact_exchange"
                            else exchange_association_content)
                options = {"availability": self.contact_availability} if self.recurrence == "contact_exchange" else {}
                S, share = exchange(
                    S, share, k[:, t], v[:, t],
                    2*self.dk*alpha[:, t, :, None], beta[:, t, :, None],
                    self.content_budget, **options)
                # a*C is the actual bounded read; low-share retention does not
                # magically confer high retrieval weight.
                query = q[:, t] * share if self.recurrence == "contact_exchange" else q[:, t]
                ot = torch.matmul(S.transpose(-1, -2), query.unsqueeze(-1)).squeeze(-1)
                outputs.append(ot)
                errors.append((v[:, t] - ot).detach().square().mean())
                continue
            kt = k[:, t].unsqueeze(-1)  # [B, H, dk, 1]
            vt = v[:, t].unsqueeze(-1)  # [B, H, dv, 1]
            qt = q[:, t].unsqueeze(-1)  # [B, H, dk, 1]
            a_t = alpha[:, t, :, None, None]  # [B, H, 1, 1]
            b_t = beta[:, t, :, None, None]   # [B, H, 1, 1]

            # Modern delta recurrence erases from the already decayed state.
            # Keep the old rule explicit so old checkpoints/results do not change.
            if self.recurrence == "feedback_delta":
                S, e_t = feedback_delta_step(S, kt, vt, a_t, b_t, strength, feedback_gain)
            else:
                prior = a_t * S if self.recurrence == "decay_then_delta" else S
                v_hat = torch.matmul(prior.transpose(-1, -2), kt)
                e_t = vt - v_hat  # [B, H, dv, 1]
                delta = torch.matmul(kt, e_t.transpose(-1, -2))
                S = a_t * S + b_t * delta
            errors.append(e_t.detach().square().mean())
            if self.content_budget:
                # Constrain the actual matrix used NOW and in the next event.
                # No detached shadow budget or independent allocation ledger.
                proposed = S
                S = admit_association_content(S, self.content_budget)
                if t == T - 1:
                    correction = (S.detach()-proposed.detach()).square().mean().sqrt()

            # Readout
            ot = torch.matmul(S.transpose(-1, -2), qt).squeeze(-1)  # [B, H, dv]
            outputs.append(ot)

        out = torch.stack(outputs, dim=1).reshape(B, T, d)
        out = self.o_proj(out * F.silu(g))
        result = x + out
        if self.mlp is not None:
            normalized = self.mlp_norm(result)
            result = result + self.mlp["down"](
                F.silu(self.mlp["gate"](normalized)) * self.mlp["up"](normalized)
            )

        diagnostics = {
            "alpha_mean": alpha.detach().mean(),
            "beta_mean": beta.detach().mean(),
            "error_norm": torch.stack(errors).mean().sqrt(),
            "state_norm": S.detach().square().mean().sqrt(),
        }
        if self.content_budget and not self.share_size:
            diagnostics.update(
                budget_correction_rms=correction,
                content_mass_max=torch.maximum(S.detach().abs().sum(-1).max(),
                                               S.detach().abs().sum(-2).max()))
        if self.recurrence == "feedback_delta":
            diagnostics.update(feedback_strength_mean=strength.detach().mean(),
                               feedback_gain_mean=feedback_gain.detach().mean())
        if self.share_size:
            observed_share = share.detach()
            diagnostics.update(
                interaction_gain_mean=2*self.dk*alpha.detach().mean(),
                share_max=observed_share.max(),
                share_min=observed_share.min(),
                share_entropy=-(observed_share * observed_share.clamp_min(
                    torch.finfo(observed_share.dtype).tiny).log()).sum(-1).mean(),
                content_mass=((S.detach().abs()*observed_share.unsqueeze(-1)
                               if self.recurrence == "contact_exchange" else S.detach().abs())
                              .sum((-2, -1)).mean()),
            )
        if self.conv_kernel or self.share_size:
            # A fresh fixed-size tensor owns its storage: no hidden whole-chunk view.
            pieces = [S.reshape(B, self.matrix_size)]
            if self.share_size:
                pieces.append(share.reshape(B, self.share_size))
            if self.conv_kernel:
                pieces.append(next_history.reshape(B, self.history_size))
            S = torch.cat(pieces, dim=-1)
        return result, S, diagnostics


class GatedDeltaNetLM(nn.Module):
    """Language model using stacked Gated DeltaNet layers matching CBIM capacity."""

    architecture = "GatedDeltaNet-d128-L4-BPE"

    def __init__(
        self,
        vocab_size: int = 50257,
        d: int = 128,
        layers: int = 4,
        heads: int = 4,
        recurrence: str = "legacy",
        mlp_hidden: int = 0,
        conv_kernel: int = 0,
        content_budget: float = 0.0,
        contact_availability: bool = False,
        decay_init: str = "constant",
        feedback_init: str = "constant",
    ):
        super().__init__()
        if vocab_size < 2 or layers < 1:
            raise ValueError("vocab_size >= 2 and layers >= 1 are required")
        if d <= 0 or heads <= 0 or d % heads:
            raise ValueError("positive d must be divisible by positive heads")
        self.vocab_size = int(vocab_size)
        self.d = int(d)
        self.num_layers = int(layers)
        self.heads = int(heads)
        self.recurrence = recurrence
        self.mlp_hidden = int(mlp_hidden)
        self.conv_kernel = conv_kernel
        self.content_budget = float(content_budget)
        self.contact_availability = bool(contact_availability)
        self.decay_init, self.feedback_init = decay_init, feedback_init
        self.dk = self.dv = d // heads

        # We nest embedding inside source to match TruncatedInternalTimeGraphTrainer's
        # lexical parameter group convention ("source.embedding.weight").
        self.source = nn.Module()
        self.source.embedding = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.source.embedding.weight, std=0.02)

        self.layers = nn.ModuleList(
            [GatedDeltaNetLayer(d, heads, recurrence, mlp_hidden, conv_kernel, content_budget,
                               contact_availability, decay_init, feedback_init)
             for _ in range(layers)]
        )
        self.final_norm = nn.RMSNorm(d)
        self.decoder = nn.Linear(d, vocab_size, bias=False)
        self.decoder.weight = self.source.embedding.weight
        if recurrence != "legacy" or mlp_hidden or conv_kernel:
            self.architecture = f"GatedDeltaNet-{recurrence}-d{d}-L{layers}-MLP{mlp_hidden}-V{vocab_size}"
            if conv_kernel:
                self.architecture += f"-Conv{conv_kernel}"
            if content_budget:
                self.architecture += f"-ContentBudget{content_budget:g}"
            if contact_availability:
                self.architecture += "-AvailableContact"
            if decay_init != "constant":
                self.architecture += "-LogTimescale"
            if feedback_init != "constant":
                self.architecture += "-ThresholdSpread"

    def config_dict(self) -> dict:
        """Constructor fields required to reproduce semantics on restoration."""
        return dict(vocab_size=self.vocab_size, d=self.d, layers=self.num_layers,
                    heads=self.heads, recurrence=self.recurrence, mlp_hidden=self.mlp_hidden,
                    conv_kernel=self.conv_kernel, content_budget=self.content_budget,
                    contact_availability=self.contact_availability,
                    decay_init=self.decay_init, feedback_init=self.feedback_init)

    def state_shape(self, batch_size: int) -> tuple:
        """Packed states keep the existing tensor-based CUDA Graph/checkpoint API."""
        if self.conv_kernel or self.recurrence in ("evidence_exchange", "contact_exchange"):
            return (self.num_layers, batch_size, self.layers[0].state_size)
        return (self.num_layers, batch_size, self.heads, self.dk, self.dv)

    def initial_state(
        self, batch_size: int, device=None, dtype=None
    ) -> torch.Tensor:
        parameter = next(self.parameters())
        dev = device or parameter.device
        dt = dtype or parameter.dtype
        result = torch.zeros(
            self.state_shape(batch_size),
            device=dev, dtype=dt,
        )
        if self.recurrence in ("evidence_exchange", "contact_exchange"):
            layer = self.layers[0]
            result[..., layer.matrix_size:layer.matrix_size+layer.share_size] = 1 / self.dk
        return result

    def predict_logits(
        self,
        input_ids: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Consume known events and predict their successors; no targets read.

        Input [B,T], logits [B,T,V], state shape from ``state_shape(B)``. Caller retains the
        numerical state across windows and explicitly detaches its graph at the
        chosen fixed training horizon. This method never resets/detaches it.
        """
        B, T = input_ids.shape
        if state is None:
            state = self.initial_state(B, device=input_ids.device)
        expected = self.state_shape(B)
        if tuple(state.shape) != expected:
            raise ValueError(f"expected recurrent state {expected}, got {tuple(state.shape)}")

        h = self.source.embedding(input_ids)
        next_states = []
        diagnostics = {}

        for l, layer in enumerate(self.layers):
            h, next_s, layer_diag = layer(h, state[l])
            next_states.append(next_s)
            for k, v in layer_diag.items():
                diagnostics[f"l{l}_{k}"] = v

        h = self.final_norm(h)
        logits = self.decoder(h)
        next_state = torch.stack(next_states, dim=0)

        # Global summary diagnostics
        diagnostics["alpha_mean"] = torch.stack(
            [diagnostics[f"l{l}_alpha_mean"] for l in range(self.num_layers)]
        ).mean()
        diagnostics["beta_mean"] = torch.stack(
            [diagnostics[f"l{l}_beta_mean"] for l in range(self.num_layers)]
        ).mean()
        diagnostics["state_norm"] = torch.stack(
            [diagnostics[f"l{l}_state_norm"] for l in range(self.num_layers)]
        ).mean()
        diagnostics["error_norm"] = torch.stack(
            [diagnostics[f"l{l}_error_norm"] for l in range(self.num_layers)]
        ).mean()

        return logits, next_state, diagnostics

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """Historical trainer API; targets score predictions but never write state."""
        logits, next_state, diagnostics = self.predict_logits(input_ids, state)
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1)
            )
        return loss, next_state, diagnostics


if __import__("os").environ.get("CGB_FUSED_DELTA") == "1":
    # Subprocesses of a fused run (paired arms, evaluators) inherit this switch.
    try:
        from scripts.ib_local import gdn_fused as _gdn_fused
    except ImportError:
        import gdn_fused as _gdn_fused
    _gdn_fused.enable()
