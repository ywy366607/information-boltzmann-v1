"""H-Net Dynamic Chunking (arXiv:2507.07955v1) grafted onto MengLi's 32-layer GDN body.

Implements the formally verified 4 + 24 + 4 hierarchical chunking architecture
proved in `lean/CGBAxioms/reviews/HNetDynamicChunking.lean`:

- Outer Encoder (`layers[0..enc_layers-1]`): runs at full byte resolution `T`
  (preserving Fast Table injection at `inject_layer = 1`).
- Cosine Boundary Router: computes causal boundary probabilities
  `p_t = sigmoid(scale * (0.5 * (1 - cos(W_q x_t, W_k x_{t-1})) - thresh))`
  with `b_0 = 1` and `b_t = 1[p_t >= 0.5]`.
- Downsampler & Deep Core (`layers[enc_layers..num_layers-dec_layers-1]`):
  physically packs only the `M <= T` boundary representations (and `PastPresentField`
  incoming slices) and runs all 24 deep GDN layers on `[B, M, d]` instead of `[B, T, d]`.
- Differentiable Triton EMA Dechunker + STE Confidence + Zero-Init Skip Residual:
  uses MengLi's existing fused Triton `causal_scan` for `z_bar_m = p_m z_m + (1-p_m) z_bar_{m-1}`,
  causally upsamples back to length `T`, weights by `STE(c_t)` (`value == 1`, exact
  gradient into `p_t`), and adds `W_res x_t` (zero-initialized for smooth upcycling).
- Outer Decoder (`layers[num_layers-dec_layers..num_layers-1]`): runs at full byte
  resolution `T` (preserving experience/memory injection at `experience_layer = 31`
  and Fast Table output residual).
- Ratio Load-Balancing Loss (H-Net Eq. 10, Theorems 1 & 2):
  `L_ratio = (N / (N - 1)) * ((N - 1) * F * G + (1 - F) * (1 - G))`.

Default OFF; enabled explicitly via `--hnet-chunking` / `configure()`.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F

from scripts.ib_local.causal_workspace_scan import causal_scan


@contextmanager
def hnet_bypass(model):
    """Temporarily bypass H-Net chunking (e.g. for ablation/diagnostics)."""
    hnet = getattr(model, 'hnet_chunker', None)
    if hnet is None and hasattr(model, 'body'):
        hnet = getattr(model.body, 'hnet_chunker', None)
    if hnet is None:
        yield
        return
    prev = getattr(hnet, '_bypass', False)
    hnet._bypass = True
    try:
        yield
    finally:
        hnet._bypass = prev


class HNetStepState:
    """Ephemeral H-Net §2.4 single-step autoregressive router + EMA dechunker state.

    Zero learnable parameters; carries only `prev_k` (router previous key),
    `prev_z_bar` (EMA dechunker last output), and optional `prev_inc`
    (PastPresentField workspace held between chunk boundaries).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.has_prev: bool = False
        self.prev_k: torch.Tensor | None = None
        self.prev_z_bar: torch.Tensor | None = None
        self.prev_inc: torch.Tensor | None = None
        self.steps: int = 0
        self.boundaries: int = 0


@contextmanager
def hnet_step_scope(model, step_state: HNetStepState | None = None):
    """Enable H-Net §2.4 stateful 1-byte chunked decoding on `model.body`."""
    body = getattr(model, 'body', model)
    hnet = getattr(body, 'hnet_chunker', None)
    if hnet is None:
        yield step_state
        return
    st = HNetStepState() if step_state is None else step_state
    prev = getattr(body, '_hnet_step_state', None)
    body._hnet_step_state = st
    try:
        yield st
    finally:
        if prev is None:
            if '_hnet_step_state' in vars(body):
                delattr(body, '_hnet_step_state')
        else:
            body._hnet_step_state = prev


class HNetDynamicChunker(nn.Module):
    """End-to-end differentiable dynamic chunking router + Triton EMA dechunker."""

    def __init__(
        self,
        d: int,
        *,
        enc_layers: int = 4,
        dec_layers: int = 4,
        target_ratio: float = 4.0,
        ratio_loss_weight: float = 0.03,
        min_seq_len: int = 1,
        router_scale: float = 6.0,
        router_thresh_init: float = 0.25,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if enc_layers < 1 or dec_layers < 1:
            raise ValueError('enc_layers >= 1 and dec_layers >= 1 required')
        if target_ratio < 1.0:
            raise ValueError('target_ratio >= 1.0 required')
        self.d = int(d)
        self.enc_layers = int(enc_layers)
        self.dec_layers = int(dec_layers)
        self.target_ratio = float(target_ratio)
        self.ratio_loss_weight = float(ratio_loss_weight)
        self.min_seq_len = int(min_seq_len)
        self.router_scale = float(router_scale)
        self._bypass = False

        dt = dtype or torch.float32
        self.q_proj = nn.Linear(d, d, bias=False, device=device, dtype=dt)
        self.k_proj = nn.Linear(d, d, bias=False, device=device, dtype=dt)
        self.residual_proj = nn.Linear(d, d, bias=False, device=device, dtype=dt)
        self.router_thresh = nn.Parameter(torch.tensor(float(router_thresh_init), device=device, dtype=dt))

        with torch.no_grad():
            nn.init.eye_(self.q_proj.weight)
            nn.init.eye_(self.k_proj.weight)
            # Zero-init skip residual for smooth identity-safe upcycling (Theorem 3).
            nn.init.zeros_(self.residual_proj.weight)

        # Telemetry buffers (checkpoint-restorable, zero schema break).
        self.register_buffer('calibrated', torch.zeros((), device=device, dtype=torch.bool))
        self.register_buffer('calls', torch.zeros((), device=device, dtype=torch.long))
        self.register_buffer('last_boundary_rate', torch.zeros((), device=device, dtype=dt))
        self.register_buffer('last_compression_ratio', torch.ones((), device=device, dtype=dt))
        self.register_buffer('last_ratio_loss', torch.ones((), device=device, dtype=dt))
        self.register_buffer('mean_boundary_rate_ema', torch.tensor(1.0 / max(1.0, target_ratio), device=device, dtype=dt))

        # Ephemeral per-forward ratio loss (consumed by model.objective, not persisted).
        self._pending_ratio_loss: torch.Tensor | None = None

    def active_for(self, seq_len: int) -> bool:
        return (not self._bypass) and (int(seq_len) >= self.min_seq_len) and (self.target_ratio > 1.0)

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute causal boundary probabilities `p`, binary masks `b`, and ratio loss."""
        B, T, _ = x.shape
        q = F.normalize(self.q_proj(x), dim=-1, eps=1e-6)
        k = F.normalize(self.k_proj(x), dim=-1, eps=1e-6)
        # Causal cosine dissimilarity between position t and t-1.
        cos_sim = (q[:, 1:] * k[:, :-1]).sum(dim=-1)  # [B, T-1]
        dissim = 0.5 * (1.0 - cos_sim)                # [B, T-1] in [0, 1]

        # On first real training forward pass, calibrate router_thresh to the (1 - 1/N)
        # quantile of dissim so step-0 starts cleanly near target_ratio without collapse.
        if not bool(self.calibrated) and torch.is_grad_enabled() and T >= 8:
            with torch.no_grad():
                q_level = max(0.05, min(0.95, 1.0 - 1.0 / self.target_ratio))
                init_th = float(torch.quantile(dissim.detach().float().reshape(-1), q_level))
                self.router_thresh.fill_(init_th)
                self.calibrated.fill_(True)

        p_rest = torch.sigmoid(self.router_scale * (dissim - self.router_thresh))
        p0 = torch.ones((B, 1), device=x.device, dtype=x.dtype)
        p = torch.cat((p0, p_rest), dim=1)            # [B, T] in (0, 1], p[:, 0] == 1
        b = (p >= 0.5).to(x.dtype)
        b[:, 0] = 1.0                                 # Guarantee M >= 1 per sequence (Theorem 4)
        # Strictly causal VRAM safety cap (zero future leakage; inactive on short utterances T < 128
        # so single-step §2.4 decode and utterance replay match identically).
        if self.target_ratio >= 2.5 and T >= 128:
            m_cap = max(2, int(math.ceil(0.5 * T)))
            b = b * (b.cumsum(dim=1) <= m_cap).to(b.dtype)

        # H-Net Eq. 10 load-balancing loss (Theorems 1 & 2).
        N = self.target_ratio
        F_rate = b.detach().mean()
        G_rate = p.mean()
        ratio_loss = (N / (N - 1.0)) * ((N - 1.0) * F_rate * G_rate + (1.0 - F_rate) * (1.0 - G_rate))

        if T >= 32:
            with torch.no_grad():
                fr = float(F_rate)
                self.calls.add_(1)
                self.last_boundary_rate.fill_(fr)
                self.last_compression_ratio.fill_(1.0 / max(fr, 1e-4))
                self.last_ratio_loss.fill_(float(ratio_loss))
                self.mean_boundary_rate_ema.mul_(0.95).add_(0.05 * fr)

        return p, b, ratio_loss

    def route_step(
        self,
        x_t: torch.Tensor,
        prev_k: torch.Tensor | None,
        has_prev: bool | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-byte causal router step (H-Net §2.4), bit-exact with `route()`.

        Supports both Python `bool` `has_prev` (eager mode) and 0-D / [1] CUDA
        tensor `has_prev` (static CUDA Graph capture without host branching).
        """
        x0 = x_t[:, 0]  # [B, D]
        q_t = F.normalize(self.q_proj(x0), dim=-1, eps=1e-6)
        k_t = F.normalize(self.k_proj(x0), dim=-1, eps=1e-6)
        if torch.is_tensor(has_prev):
            cos_sim = (q_t * prev_k).sum(dim=-1)
            dissim = 0.5 * (1.0 - cos_sim)
            p_cand = torch.sigmoid(self.router_scale * (dissim - self.router_thresh))
            ones = torch.ones_like(p_cand)
            p_t = torch.where(has_prev > 0.5, p_cand, ones)
            b_t = p_t >= 0.5
            return p_t, b_t, k_t
        if not has_prev or prev_k is None:
            p_t = torch.ones((x0.shape[0],), device=x0.device, dtype=x0.dtype)
            b_t = torch.ones((x0.shape[0],), device=x0.device, dtype=torch.bool)
            return p_t, b_t, k_t
        cos_sim = (q_t * prev_k).sum(dim=-1)
        dissim = 0.5 * (1.0 - cos_sim)
        p_t = torch.sigmoid(self.router_scale * (dissim - self.router_thresh))
        b_t = p_t >= 0.5
        return p_t, b_t, k_t

    def dechunk_update(
        self,
        z_t: torch.Tensor,
        p_t: torch.Tensor,
        prev_z_bar: torch.Tensor | None,
    ) -> torch.Tensor:
        """Single-step EMA dechunker update: z_bar = p_t * z_t + (1 - p_t) * prev_z_bar."""
        p_w = p_t.reshape(-1, 1, 1)
        if prev_z_bar is None:
            return p_w * z_t
        return p_w * z_t + (1.0 - p_w) * prev_z_bar

    def combine_skip(self, z_bar: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """Combine EMA-smoothed deep-core state with trained skip residual W_res x_t."""
        return z_bar + self.residual_proj(x_t)

    def chunk(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        b: torch.Tensor,
        incoming: torch.Tensor | None = None,
    ) -> dict:
        """Pack boundary positions `[B, M_max, d]` for the 24 deep core layers."""
        B, T, D = x.shape
        b_bool = b > 0.5
        counts = b_bool.sum(dim=1)  # [B] >= 1
        M_max = int(counts.max().item())

        # Causal chunk index c_t in [0, M_b - 1] (Theorem 4).
        chunk_idx = b_bool.long().cumsum(dim=1) - 1  # [B, T]

        if B == 1:
            # Zero-padding fast path for B=1 (95% of MengLi sequence calls!).
            sel = torch.nonzero(b_bool[0], as_tuple=False).flatten()  # [M]
            x_c = x[:, sel]                                           # [1, M, D]
            p_c = p[:, sel]                                           # [1, M]
            inc_c = None if incoming is None else incoming[:, sel]
            return dict(
                x_chunk=x_c, p_chunk=p_c, incoming_chunk=inc_c,
                chunk_idx=chunk_idx, M_max=M_max, counts=counts,
                p=p, b=b, x_enc=x,
            )

        # General B > 1 path (e.g. B=8 native_replay_batch): right-pad to M_max.
        # Causality guarantees positions 0..M_b-1 are completely unaffected by padding >= M_b.
        x_c = x.new_zeros((B, M_max, D))
        p_c = p.new_ones((B, M_max))
        inc_c = None if incoming is None else incoming.new_zeros((B, M_max, *incoming.shape[2:]))
        for bi in range(B):
            sel = torch.nonzero(b_bool[bi], as_tuple=False).flatten()
            m = len(sel)
            x_c[bi, :m] = x[bi, sel]
            p_c[bi, :m] = p[bi, sel]
            if inc_c is not None:
                inc_c[bi, :m] = incoming[bi, sel]
        return dict(
            x_chunk=x_c, p_chunk=p_c, incoming_chunk=inc_c,
            chunk_idx=chunk_idx, M_max=M_max, counts=counts,
            p=p, b=b, x_enc=x,
        )

    def dechunk(
        self,
        z_chunk: torch.Tensor,
        pack: dict,
        incoming_chunk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Triton causal EMA dechunking + upsampling + STE confidence + skip residual."""
        p_c = pack['p_chunk']                 # [B, M]
        B, M, D = z_chunk.shape

        # Causal EMA: z_bar_m = p_m * z_m + (1 - p_m) * z_bar_{m-1}, with z_bar_0 = z_0.
        # p_c[:, 0] == 1.0 so a[:, 0] == 0.0 -> initial zero state yields exact z_bar_0 = z_0!
        a = (1.0 - p_c).unsqueeze(-1).expand(B, M, D)
        u = p_c.unsqueeze(-1) * z_chunk
        init = z_chunk.new_zeros((B, D))
        z_smooth = causal_scan(a, u, init)    # [B, M, D] (fused Triton CUDA forward+backward!)

        # Causal upsampling via gather on chunk_idx [B, T].
        chunk_idx = pack['chunk_idx']         # [B, T]
        T = chunk_idx.shape[1]
        if B == 1:
            z_up = z_smooth[:, chunk_idx[0]]  # [1, T, D]
            inc_up = None if incoming_chunk is None else incoming_chunk[:, chunk_idx[0]]
        else:
            idx_exp = chunk_idx.unsqueeze(-1).expand(B, T, D)
            z_up = torch.gather(z_smooth, 1, idx_exp)
            if incoming_chunk is not None:
                s_slots, s_rank = incoming_chunk.shape[2], incoming_chunk.shape[3]
                inc_flat = incoming_chunk.reshape(B, M, s_slots * s_rank)
                idx_inc = chunk_idx.unsqueeze(-1).expand(B, T, s_slots * s_rank)
                inc_up = torch.gather(inc_flat, 1, idx_inc).reshape(B, T, s_slots, s_rank)
            else:
                inc_up = None

        # Straight-through confidence weighting (H-Net Eq. 5, Theorem 3).
        p, b = pack['p'], pack['b']
        conf = p * b + (1.0 - p) * (1.0 - b)
        ste_conf = (1.0 + conf - conf.detach()).unsqueeze(-1)  # value == 1.0, grad -> p!
        y = ste_conf * z_up + self.residual_proj(pack['x_enc'])
        return y, inc_up

    def pop_ratio_loss(self) -> torch.Tensor | None:
        loss = self._pending_ratio_loss
        self._pending_ratio_loss = None
        return loss

    def diagnostics(self, optimizer=None) -> dict:
        wiener_gate = None
        if optimizer is not None:
            gates = []
            for p in self.parameters():
                st = optimizer.state.get(p)
                if st is not None and torch.is_tensor(st.get('budget_gate')):
                    gates.append(float(st['budget_gate'].detach().float().mean()))
            if gates:
                wiener_gate = float(sum(gates) / len(gates))
        return dict(
            schema=1,
            enc_layers=self.enc_layers,
            dec_layers=self.dec_layers,
            target_ratio=self.target_ratio,
            ratio_loss_weight=self.ratio_loss_weight,
            min_seq_len=self.min_seq_len,
            router_scale=self.router_scale,
            router_thresh=float(self.router_thresh.detach()),
            calibrated=bool(self.calibrated),
            calls=int(self.calls),
            last_boundary_rate=float(self.last_boundary_rate),
            last_compression_ratio=float(self.last_compression_ratio),
            mean_boundary_rate_ema=float(self.mean_boundary_rate_ema),
            effective_compression_ema=1.0 / max(float(self.mean_boundary_rate_ema), 1e-4),
            last_ratio_loss=float(self.last_ratio_loss),
            residual_weight_rms=float(self.residual_proj.weight.detach().float().square().mean().sqrt()),
            wiener_mean_gate=wiener_gate,
        )


def enabled(engine) -> bool:
    model = getattr(engine, 'model', engine)
    body = getattr(model, 'body', model)
    return hasattr(body, 'hnet_chunker') and body.hnet_chunker is not None


def configure(
    engine,
    config=None,
    *,
    enc_layers: int = 4,
    dec_layers: int = 4,
    target_ratio: float = 4.0,
    ratio_loss_weight: float = 0.03,
    min_seq_len: int = 1,
    router_scale: float = 6.0,
    router_thresh_init: float = 0.25,
    lr: float | None = None,
    initialize: bool = True,
) -> HNetDynamicChunker:
    """Attach opt-in H-Net dynamic chunking cleanly to MengLi's 32-layer TableGDN."""
    model = engine.model
    body = model.body
    base = body.base
    if enc_layers + dec_layers >= base.num_layers:
        raise ValueError(f'enc_layers({enc_layers}) + dec_layers({dec_layers}) must be < num_layers({base.num_layers})')
    w = base.source.embedding.weight
    if not hasattr(body, 'hnet_chunker') or body.hnet_chunker is None:
        chunker = HNetDynamicChunker(
            base.d,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            target_ratio=target_ratio,
            ratio_loss_weight=ratio_loss_weight,
            min_seq_len=min_seq_len,
            router_scale=router_scale,
            router_thresh_init=router_thresh_init,
            device=w.device,
            dtype=w.dtype,
        )
        body.hnet_chunker = chunker
        added = list(chunker.parameters())
        use_lr = float(lr) if lr is not None else float(engine.optimizer.param_groups[0]['lr']) * 3.0
        engine.optimizer.add_param_group(dict(params=added, lr=use_lr))
        engine.params.extend(added)
        engine.groups = (*engine.groups, added)
    else:
        chunker = body.hnet_chunker
        chunker.target_ratio = float(target_ratio)
        chunker.ratio_loss_weight = float(ratio_loss_weight)
        chunker.min_seq_len = int(min_seq_len)
        chunker.router_scale = float(router_scale)

    # Rebuild engine.decision so H-Net §2.4 3-stage CUDA graphs (enc / conditional core / dec)
    # are active whenever configure() runs after load_strict_engine().
    dec = getattr(engine, 'decision', None)
    if dec is not None and not getattr(dec, '_hnet_section24', False):
        from scripts.ib_local.scoped_decision_graph import ScopedDecisionGraph
        from scripts.ib_local.language_navigation_turns_v3 import DecisionGraphV3
        if isinstance(dec, ScopedDecisionGraph):
            engine.decision = ScopedDecisionGraph(
                engine.model,
                capture=dec.capture,
                optimize=dec.optimize,
                pack_inputs=(dec._input_host is not None),
                capture_handover=hasattr(dec, '_capture_handover_budget'),
                fused_step=bool(dec.step_bundles),
            )
        elif isinstance(dec, DecisionGraphV3):
            engine.decision = DecisionGraphV3(engine.model, capture=dec.capture)

    if config is not None:
        config['hnet_chunking'] = dict(
            schema=1,
            enc_layers=int(enc_layers),
            dec_layers=int(dec_layers),
            target_ratio=float(target_ratio),
            ratio_loss_weight=float(ratio_loss_weight),
            min_seq_len=int(min_seq_len),
            router_scale=float(router_scale),
            router_thresh_init=float(router_thresh_init),
            lr=lr,
        )
    return chunker


def diagnostics(engine) -> dict | None:
    if not enabled(engine):
        return None
    return engine.model.body.hnet_chunker.diagnostics(getattr(engine, 'optimizer', None))
