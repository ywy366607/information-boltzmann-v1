"""Pre-linguistic Biological Object-Pointing / Gaze Steering (`GazeObjectPointer`).

Inspired by Drosophila Central Complex steering (`E-PG` / `PFL3`) and human infant
pre-linguistic gaze & pointing (direction precedes syntax and self-other models):

  1. Strictly O(1) parameters independent of the number of Umwelt objects M.
     Zero per-person or per-tool output heads — any new person, agent, or tool
     can be hot-plugged into the scene with zero architecture changes.
  2. Each in-scene object `i` maintains a live sensory fingerprint `k_i in R^d`
     computed purely from that object's actual incoming UTF-8 bytes & style
     through MengLi's own byte embedding + 32-layer GDN bodily state, paired
     with a stable `target_object_id` for evidence provenance and settlement.
  3. Unified Target-Conditioned Action `a = (target_id, utterance)`:
     `target_to_action(k_i)` maps object fingerprint `k_i` to a residual action
     condition `delta(k_i) in R^d` (zero-init at birth), forming
     `a_tilde(i, u) = a(u) + delta(k_i)`.
     - Slow causal model (`causal_relations`) evaluates `p(response | Z, i, u)`
       for every in-scene object `i` BEFORE world delivery.
     - Fast gaze pointer provides rapid habit preference `q(h_t, u_t) · W_k k_i`.
     - Fast habit + slow causal prior jointly select `target_id`.
  4. Dual learning & exact optimizer recovery:
     - Gaze habit updates from real settled `metal_g` ONLY when valid response
       boundary evidence was available (`evidence_available == True`).
     - `target_to_action` updates from real response-boundary prediction MSE
       across ALL sources (`book`, `mother`, `parent`, `tool`, `object`),
       teaching the causal space to distinguish outcomes of the same utterance
       aimed at different objects.
     - Full Adam optimizer state (`exp_avg`, `exp_avg_sq`, `step`) is serialized
       bit-exactly in float32 base64 so save + reload reproduces the exact next
       learning step (`< 1e-6`).
"""
from __future__ import annotations

import base64
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _resolve_model_dim_and_device(engine) -> tuple[int, torch.device]:
    model = getattr(engine, 'model', None)
    body = getattr(model, 'body', None) if model is not None else None
    base = getattr(body, 'base', None) if body is not None else None
    if base is not None and hasattr(base, 'source') and hasattr(base.source, 'embedding'):
        w = base.source.embedding.weight
        return int(w.shape[1]), w.device
    enc = getattr(model, 'encoder', None) if model is not None else None
    if enc is not None and hasattr(enc, 'in_features'):
        dev = next(enc.parameters()).device
        return int(enc.in_features), dev
    dev = getattr(engine, 'device', torch.device('cpu'))
    return 64, torch.device(dev)


def _tensor_to_b64(t: torch.Tensor) -> dict:
    arr = t.detach().cpu().contiguous().numpy()
    return dict(
        dtype=str(arr.dtype),
        shape=list(arr.shape),
        b64=base64.b64encode(arr.tobytes()).decode('ascii'),
    )


def _b64_to_tensor(spec: dict, *, device: torch.device) -> torch.Tensor:
    raw = base64.b64decode(spec['b64'].encode('ascii'))
    arr = np.frombuffer(raw, dtype=np.dtype(spec['dtype'])).reshape(spec['shape']).copy()
    return torch.from_numpy(arr).to(device=device)


@torch.no_grad()
def encode_sensory_bytes(engine, raw_bytes: bytes, *, d_model: int, device: torch.device) -> torch.Tensor:
    """Encode raw UTF-8 bytes through MengLi's own byte embedding table (no extra forward pass)."""
    if not raw_bytes:
        return torch.zeros(d_model, device=device, dtype=torch.float32)
    sample = raw_bytes[:512]
    model = getattr(engine, 'model', None)
    body = getattr(model, 'body', None) if model is not None else None
    base = getattr(body, 'base', None) if body is not None else None
    if base is not None and hasattr(base, 'source') and hasattr(base.source, 'embedding'):
        w = base.source.embedding.weight
        ids = torch.as_tensor(np.frombuffer(sample, dtype=np.uint8).astype(np.int64), device=w.device)
        emb = F.embedding(ids, w).float()  # [L, d]
        mean_v = emb.mean(dim=0)
        if emb.shape[0] >= 8:
            tail_v = emb[-min(64, emb.shape[0]):].mean(dim=0)
            vec = 0.65 * mean_v + 0.35 * tail_v
        else:
            vec = mean_v
        return F.normalize(vec.to(device=device, dtype=torch.float32), dim=-1, eps=1e-6)

    arr = np.frombuffer(sample, dtype=np.uint8)
    hist = np.bincount(arr, minlength=256).astype(np.float32)
    hist /= max(float(hist.sum()), 1.0)
    out = np.zeros(d_model, dtype=np.float32)
    for idx in range(256):
        out[idx % d_model] += hist[idx]
    t = torch.from_numpy(out).to(device=device, dtype=torch.float32)
    return F.normalize(t, dim=-1, eps=1e-6)


@torch.no_grad()
def read_body_state(engine, *, d_model: int, device: torch.device) -> torch.Tensor:
    """Read MengLi's live recurrent body state vector cleanly in R^d."""
    if engine is not None and hasattr(engine, '_read_source'):
        try:
            h = engine._read_source()
            if torch.is_tensor(h) and h.numel() >= d_model:
                v = h.detach().float().reshape(-1)[:d_model].to(device)
                return F.normalize(v, dim=-1, eps=1e-6)
        except Exception:
            pass
    return torch.zeros(d_model, device=device, dtype=torch.float32)


def classify_interaction_mode(spoke: bool, send_target: str, observe_target: str) -> str:
    """Derive human-readable telemetry label purely from orthogonal `(spoke, out, in)`."""
    out_t = str(send_target or 'none')
    in_t = str(observe_target or 'none')
    if not spoke:
        if in_t == 'none':
            return 'quiet_self_evolution'
        if in_t == 'book':
            return 'quiet_reading'
        if in_t == 'mother':
            return 'look_at_mother'
        return f'observe_{in_t}'
    if out_t == 'none':
        if in_t == 'none':
            return 'inner_monologue'
        if in_t == 'book':
            return 'read_aloud_solo'
        return f'self_talk_observing_{in_t}'
    if out_t == 'book' and in_t == 'book':
        return 'book_search'
    if out_t == 'book' and in_t == 'mother':
        return 'read_aloud_to_mother'
    if out_t == 'mother' and in_t == 'mother':
        return 'face_to_face_chat'
    if out_t == 'mother' and in_t == 'book':
        return 'co_reading_talk'
    if in_t == 'none':
        return f'call_{out_t}_and_pause'
    return f'send_{out_t}_observe_{in_t}'


class _OtherWorldModel(nn.Module):
    """Per-object persistent latent Other/World boundary predictor & state-persistence gate.

    Separates persistent conversational Others (`mother`, `parent`) from resetting World
    streams (`book`) using O(1) object-conditioned parameters independent of object count M:
      - `pred_in -> SiLU -> pred_out` (zero-init output): predicts latent boundary residual
        `Δz(base_pred, z_obj, k_in, k_out)` with exact zero shock at initialization.
      - `gate_net` (zero-init -> sigmoid(0)=0.5): learned per-object update gate
        `g = σ(W_g [z_obj, z_actual, k_in] + b_g)` trained by exact 1-step BPTT so
        persistent Others learn smooth state continuity while Books learn rapid refresh.
      - `state_to_out_key`, `state_to_in_key` (zero-init): inject live persistent object
        state `z_obj` into MengLi's `(out, in)` gaze queries without perturbing step 0.
    """

    def __init__(self, d_latent: int, d_gaze: int, device: torch.device):
        super().__init__()
        self.d_latent = int(d_latent)
        self.d_gaze = int(d_gaze)
        d_hidden = max(32, min(64, self.d_latent * 2))
        self.pred_in = nn.Linear(
            self.d_latent * 2 + self.d_gaze * 2, d_hidden, bias=True, device=device, dtype=torch.float32,
        )
        self.pred_out = nn.Linear(d_hidden, self.d_latent, bias=True, device=device, dtype=torch.float32)
        self.gate_net = nn.Linear(
            self.d_latent * 2 + self.d_gaze, self.d_latent, bias=True, device=device, dtype=torch.float32,
        )
        self.state_to_out_key = nn.Linear(self.d_latent, self.d_gaze, bias=False, device=device, dtype=torch.float32)
        self.state_to_in_key = nn.Linear(self.d_latent, self.d_gaze, bias=False, device=device, dtype=torch.float32)

        nn.init.zeros_(self.pred_out.weight)
        nn.init.zeros_(self.pred_out.bias)
        nn.init.zeros_(self.gate_net.weight)
        nn.init.zeros_(self.gate_net.bias)
        nn.init.zeros_(self.state_to_out_key.weight)
        nn.init.zeros_(self.state_to_in_key.weight)

    def predict_residual(
        self,
        base_pred: torch.Tensor,
        z_obj: torch.Tensor,
        k_in: torch.Tensor,
        k_out: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat((base_pred, z_obj, k_in, k_out), dim=-1)
        return self.pred_out(F.silu(self.pred_in(x)))

    def state_gate(self, z_prev: torch.Tensor, z_actual: torch.Tensor, k_in: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate_net(torch.cat((z_prev, z_actual, k_in), dim=-1)))


class _DualChannelHead(nn.Module):
    """Companion O(1) observation-pointer (`in`) + null-channel (`∅`) head.

    Kept outside `GazeObjectPointer`'s `nn.Module` child tree and serialized bit-exactly
    inside `export_dynamic_state()` so existing schema-2 checkpoints load with `strict=True`.
    """

    def __init__(self, d_model: int, d_gaze: int, device: torch.device):
        super().__init__()
        self.q_in_state = nn.Linear(d_model, d_gaze, bias=False, device=device, dtype=torch.float32)
        self.q_in_utt = nn.Linear(d_model, d_gaze, bias=False, device=device, dtype=torch.float32)
        self.q_in_ctx = nn.Linear(4, d_gaze, bias=True, device=device, dtype=torch.float32)
        self.k_in_proj = nn.Linear(d_model, d_gaze, bias=False, device=device, dtype=torch.float32)
        self.recency_in = nn.Parameter(torch.zeros(2, device=device, dtype=torch.float32))
        self.observe_to_action = nn.Linear(d_model, d_model, bias=False, device=device, dtype=torch.float32)
        # Mild negative initial bias on `∅` (~15% initial self-talk / self-evolution curiosity, learned freely from metal_g).
        self.out_null_logit = nn.Parameter(torch.tensor(-0.7, device=device, dtype=torch.float32))
        self.in_null_logit = nn.Parameter(torch.tensor(-0.9, device=device, dtype=torch.float32))

        nn.init.zeros_(self.q_in_state.weight)
        nn.init.zeros_(self.q_in_utt.weight)
        nn.init.zeros_(self.q_in_ctx.weight)
        nn.init.zeros_(self.q_in_ctx.bias)
        nn.init.orthogonal_(self.k_in_proj.weight)
        nn.init.zeros_(self.observe_to_action.weight)


class GazeObjectPointer(nn.Module):
    """Constant-parameter O(1) dynamic cross-attention pointer + target-action conditioning."""

    def __init__(
        self,
        d_model: int,
        *,
        d_gaze: int = 64,
        lr: float = 1.5e-3,
        entropy_coef: float = 0.025,
        causal_prior_scale: float = 1.0,
        boundary_loss_weight: float = 0.25,
        fp_ema: float = 0.25,
        baseline_decay: float = 0.90,
        enable_other_world: bool = False,
        device: torch.device | str = 'cpu',
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_gaze = int(d_gaze)
        self.lr = float(lr)
        self.entropy_coef = float(entropy_coef)
        self.causal_prior_scale = float(causal_prior_scale)
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.fp_ema = float(fp_ema)
        self.baseline_decay = float(baseline_decay)
        self.enable_other_world = bool(enable_other_world)

        dev = torch.device(device)
        # Zero-init query, recency & target_to_action -> exact uniform curiosity & zero distortion at birth.
        self.q_state = nn.Linear(self.d_model, self.d_gaze, bias=False, device=dev, dtype=torch.float32)
        self.q_utt = nn.Linear(self.d_model, self.d_gaze, bias=False, device=dev, dtype=torch.float32)
        self.q_ctx = nn.Linear(4, self.d_gaze, bias=True, device=dev, dtype=torch.float32)
        self.k_proj = nn.Linear(self.d_model, self.d_gaze, bias=False, device=dev, dtype=torch.float32)
        self.target_to_action = nn.Linear(self.d_model, self.d_model, bias=False, device=dev, dtype=torch.float32)
        self.recency_weight = nn.Parameter(torch.zeros(2, device=dev, dtype=torch.float32))

        nn.init.zeros_(self.q_state.weight)
        nn.init.zeros_(self.q_utt.weight)
        nn.init.zeros_(self.q_ctx.weight)
        nn.init.zeros_(self.q_ctx.bias)
        nn.init.orthogonal_(self.k_proj.weight)
        nn.init.zeros_(self.target_to_action.weight)

        self.register_buffer('baseline_g', torch.tensor(0.0, device=dev, dtype=torch.float32))
        self.register_buffer('step_count', torch.tensor(0, device=dev, dtype=torch.long))
        self.register_buffer('mean_entropy', torch.tensor(0.0, device=dev, dtype=torch.float32))
        self.register_buffer('boundary_steps', torch.tensor(0, device=dev, dtype=torch.long))

        self._opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        # Companion `(out, in)` dual-channel head kept outside `state_dict()` for strict=True compatibility.
        dual = _DualChannelHead(self.d_model, self.d_gaze, dev)
        self.__dict__['_dual_channel'] = dual
        self._dual_opt = torch.optim.Adam(dual.parameters(), lr=self.lr)
        # Companion per-object persistent Other/World latent state model (lazily sized to `encoder.out_features`).
        self.__dict__['_other_world_model'] = None
        self._other_opt: torch.optim.Optimizer | None = None
        self.other_states: dict[str, torch.Tensor] = {}
        self._other_prev_transitions: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.other_model_ema: dict[str, dict[str, float]] = {}
        self.fingerprints: dict[str, torch.Tensor] = {}
        self.fp_center: torch.Tensor | None = None
        self.fp_center_ema: float = 0.05
        self.max_rel_delta_norm: float = 0.45
        self.target_counts: dict[str, int] = {}
        self.target_g_ema: dict[str, float] = {}
        self.interaction_mode_counts: dict[str, int] = {}
        self.last_decision: dict | None = None
        self._pending: dict | None = None

    @property
    def device(self) -> torch.device:
        return self.recency_weight.device

    @torch.no_grad()
    def _refresh_fp_center(self, *, snap: bool = False) -> None:
        """Track smooth unweighted centroid across known objects to strip shared cone anisotropy."""
        if len(self.fingerprints) < 2:
            return
        centroid = torch.stack(
            [v.to(device=self.device, dtype=torch.float32) for v in self.fingerprints.values()],
            dim=0,
        ).mean(dim=0)
        if self.fp_center is None or snap:
            self.fp_center = centroid.detach()
        else:
            a = self.fp_center_ema
            self.fp_center = ((1.0 - a) * self.fp_center.to(self.device) + a * centroid).detach()

    def relative_fingerprint(self, obj_id: str) -> torch.Tensor:
        """Return unit-normalized global-mean-centered object fingerprint `k_rel_i`.

        Strips the shared byte/body common component (`cos ~ 0.995`) without batch-only
        collapse: persists `fp_center` across turns so a lone in-scene object retains its
        exact coordinate when another object leaves.
        """
        key = str(obj_id)
        if key not in self.fingerprints:
            return torch.zeros(self.d_model, device=self.device, dtype=torch.float32)
        raw = self.fingerprints[key].to(device=self.device, dtype=torch.float32)
        if self.fp_center is None:
            if len(self.fingerprints) >= 2:
                self._refresh_fp_center(snap=True)
            else:
                return torch.zeros_like(raw)
        diff = raw - self.fp_center.to(device=self.device, dtype=torch.float32)
        dn = float(diff.norm().item())
        if dn < 1e-4:
            return torch.zeros_like(raw)
        return diff / dn

    def _dual(self) -> _DualChannelHead:
        dual: _DualChannelHead = self.__dict__['_dual_channel']
        if dual.out_null_logit.device != self.device:
            dual.to(self.device)
        return dual

    def _ensure_other_world(self, d_latent: int) -> _OtherWorldModel:
        owm: _OtherWorldModel | None = self.__dict__.get('_other_world_model')
        if owm is None or int(owm.d_latent) != int(d_latent):
            owm = _OtherWorldModel(int(d_latent), self.d_gaze, self.device)
            self.__dict__['_other_world_model'] = owm
            self._other_opt = torch.optim.Adam(owm.parameters(), lr=self.lr)
        elif owm.pred_out.bias.device != self.device:
            owm.to(self.device)
        return owm

    def _other_state_batch(self, ids: list[str]) -> torch.Tensor | None:
        if not bool(getattr(self, 'enable_other_world', False)):
            return None
        owm: _OtherWorldModel | None = self.__dict__.get('_other_world_model')
        if owm is None or not ids:
            return None
        d_lat = int(owm.d_latent)
        rows = []
        for cid in ids:
            st = self.other_states.get(str(cid))
            if st is not None and st.numel() == d_lat:
                rows.append(st.to(device=self.device, dtype=torch.float32).reshape(d_lat))
            else:
                rows.append(torch.zeros(d_lat, device=self.device, dtype=torch.float32))
        return torch.stack(rows, dim=0)  # [M, d_latent]

    def _compute_delta(
        self,
        obj_id: str | None,
        act_scale: torch.Tensor | float,
        *,
        observe_id: str | None = None,
    ) -> torch.Tensor:
        """Differentiable scale-matched bounded residual `delta_out(k_rel_out) + delta_in(k_rel_in)`."""
        raw_v = torch.zeros(self.d_model, device=self.device, dtype=torch.float32)
        if obj_id is not None and str(obj_id) != 'none' and str(obj_id) in self.fingerprints:
            raw_v = raw_v + self.target_to_action(self.relative_fingerprint(str(obj_id)))
        if observe_id is not None and str(observe_id) != 'none' and str(observe_id) in self.fingerprints:
            raw_v = raw_v + self._dual().observe_to_action(self.relative_fingerprint(str(observe_id)))
        v_norm = raw_v.norm(dim=-1, keepdim=True)
        bounded = raw_v / (1.0 + v_norm / self.max_rel_delta_norm)
        return bounded * act_scale

    def ensure_object(self, engine, obj_id: str, intro_bytes: bytes) -> torch.Tensor:
        key = str(obj_id)
        if key == 'none':
            return torch.zeros(self.d_model, device=self.device, dtype=torch.float32)
        if key not in self.fingerprints:
            vec = encode_sensory_bytes(engine, intro_bytes, d_model=self.d_model, device=self.device)
            if float(vec.norm()) < 1e-5:
                g = torch.Generator(device='cpu')
                g.manual_seed(abs(hash(key)) % (2 ** 31 - 1) + 1)
                vec = F.normalize(torch.randn(self.d_model, generator=g), dim=-1).to(self.device)
            self.fingerprints[key] = vec.detach()
            self._refresh_fp_center()
        return self.fingerprints[key]

    @torch.no_grad()
    def target_offset(
        self,
        obj_id: str | None,
        *,
        observe_id: str | None = None,
        like: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return detached scale-matched residual `delta_out + delta_in` in body action space."""
        dev = like.device if like is not None else self.device
        dt = like.dtype if like is not None else torch.float32
        has_out = (obj_id is not None and str(obj_id) != 'none' and str(obj_id) in self.fingerprints)
        has_in = (observe_id is not None and str(observe_id) != 'none' and str(observe_id) in self.fingerprints)
        if not has_out and not has_in:
            if like is not None and like.ndim >= 2:
                return torch.zeros_like(like)
            return torch.zeros(self.d_model, device=dev, dtype=dt)
        if like is not None:
            act_scale = like.detach().to(device=self.device, dtype=torch.float32).norm(dim=-1, keepdim=True).clamp_min(1.0)
        else:
            act_scale = 1.0
        delta = self._compute_delta(obj_id, act_scale, observe_id=observe_id)
        return delta.to(device=dev, dtype=dt).detach()

    @torch.no_grad()
    def condition_action(
        self,
        action: torch.Tensor,
        obj_id: str | None,
        *,
        observe_id: str | None = None,
    ) -> torch.Tensor:
        """Form target-conditioned action a_tilde(out, in, u) = a(u) + delta_out + delta_in."""
        if action.shape[-1] != self.d_model:
            return action
        delta = self.target_offset(obj_id, observe_id=observe_id, like=action)
        return action + delta

    @torch.no_grad()
    def observe_object(self, engine, obj_id: str, raw_bytes: bytes, *, fuse_body_state: bool = True) -> None:
        """Update object `obj_id`'s sensory fingerprint from actual bytes crossing the Membrane."""
        key = str(obj_id)
        if not raw_bytes:
            return
        byte_vec = encode_sensory_bytes(engine, raw_bytes, d_model=self.d_model, device=self.device)
        if float(byte_vec.norm()) < 1e-5:
            return
        if fuse_body_state:
            body_vec = read_body_state(engine, d_model=self.d_model, device=self.device)
            if float(body_vec.norm()) > 1e-5:
                obs = F.normalize(0.65 * byte_vec + 0.35 * body_vec, dim=-1, eps=1e-6)
            else:
                obs = byte_vec
        else:
            obs = byte_vec
        prev = self.fingerprints.get(key)
        if prev is None:
            self.fingerprints[key] = obs.detach()
        else:
            a = self.fp_ema
            self.fingerprints[key] = F.normalize((1.0 - a) * prev.to(self.device) + a * obs, dim=-1, eps=1e-6).detach()
        self._refresh_fp_center()

    def _compute_habit_logits(
        self,
        h: torch.Tensor,
        u: torch.Tensor,
        ctx: torch.Tensor,
        keys: torch.Tensor,
        recency: torch.Tensor,
        obj_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.q_state(h) + self.q_utt(u) + self.q_ctx(ctx)  # [d_g]
        k = self.k_proj(keys)                                   # [M, d_g]
        owm: _OtherWorldModel | None = (
            self.__dict__.get('_other_world_model')
            if bool(getattr(self, 'enable_other_world', False)) else None
        )
        if owm is not None and obj_states is not None:
            k = k + owm.state_to_out_key(obj_states.to(device=k.device, dtype=k.dtype))
        scores = torch.matmul(k, q) / math.sqrt(float(self.d_gaze))
        scores = scores + torch.matmul(recency, self.recency_weight)
        return scores

    def _compute_observe_logits(
        self,
        h: torch.Tensor,
        u: torch.Tensor,
        ctx: torch.Tensor,
        keys: torch.Tensor,
        recency: torch.Tensor,
        obj_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dual = self._dual()
        q = dual.q_in_state(h) + dual.q_in_utt(u) + dual.q_in_ctx(ctx)  # [d_g]
        k = dual.k_in_proj(keys)                                        # [M, d_g]
        owm: _OtherWorldModel | None = (
            self.__dict__.get('_other_world_model')
            if bool(getattr(self, 'enable_other_world', False)) else None
        )
        if owm is not None and obj_states is not None:
            k = k + owm.state_to_in_key(obj_states.to(device=k.device, dtype=k.dtype))
        scores = torch.matmul(k, q) / math.sqrt(float(self.d_gaze))
        scores = scores + torch.matmul(recency, dual.recency_in)
        return scores

    @torch.no_grad()
    def select_interaction(
        self,
        engine,
        *,
        candidates: list[tuple[str, bytes]],
        utterance_bytes: bytes,
        causal_values: dict[str, float] | None = None,
        observe_causal_values: dict[str, float] | None = None,
        stumble_ratio: float = 1.0,
        nll_norm: float = 0.0,
        last_speaker: str | None = None,
        awaiting_from: str | None = None,
        allow_null_out: bool = True,
        allow_null_in: bool = True,
        rng: np.random.Generator | None = None,
    ) -> dict:
        """Factorized O(1) `(out, in)` interaction selection over `(O ∪ {∅})²`.

        Orthogonal separation:
          - `utterance_bytes` decides whether MengLi speaks and what she says.
          - `send_target` (`out`) selects who receives her speech (`'none'` = pure self-talk / monologue).
          - `observe_target` (`in`) selects who she actively observes (`'none'` = no external pull / quiet self-evolution).
        """
        if not candidates:
            raise ValueError('gaze_pointer requires at least one in-scene candidate object')
        ids = [cid for cid, _ in candidates]
        for cid, intro in candidates:
            self.ensure_object(engine, cid, intro)

        dev = self.device
        dual = self._dual()
        spoke = bool(utterance_bytes)
        h = read_body_state(engine, d_model=self.d_model, device=dev)
        u = encode_sensory_bytes(engine, utterance_bytes, d_model=self.d_model, device=dev)
        has_speech = 1.0 if spoke else 0.0
        stumble_c = max(-1.0, min(3.0, float(stumble_ratio) - 1.0))
        nll_c = max(-2.0, min(4.0, float(nll_norm)))
        awaiting_flag = 1.0 if awaiting_from is not None else 0.0
        ctx = torch.tensor([stumble_c, has_speech, nll_c, awaiting_flag], device=dev, dtype=torch.float32)

        keys = torch.stack([self.relative_fingerprint(cid) for cid in ids], dim=0)  # [M, d_model]
        obj_states = self._other_state_batch(ids)
        rec_rows = [
            [1.0 if (last_speaker is not None and cid == last_speaker) else 0.0,
             1.0 if (awaiting_from is not None and cid == awaiting_from) else 0.0]
            for cid in ids
        ]
        recency = torch.tensor(rec_rows, device=dev, dtype=torch.float32)  # [M, 2]

        def _build_prior(vals_dict: dict[str, float] | None, include_null: bool) -> tuple[torch.Tensor, dict[str, float] | None]:
            n_tot = len(ids) + (1 if include_null else 0)
            if not vals_dict:
                return torch.zeros(n_tot, device=dev, dtype=torch.float32), None
            raw_vals = [float(vals_dict.get(cid, 0.0)) for cid in ids]
            if not all(math.isfinite(v) for v in raw_vals):
                return torch.zeros(n_tot, device=dev, dtype=torch.float32), None
            # Include `'none'` at explicit baseline 0.0 inside the centering set so even when
            # only 1 external object is in scene (e.g. solo book reading), Wood's epistemic prior
            # genuinely distinguishes `book` from `none` instead of collapsing to `[0.0, 0.0]`.
            all_vals = raw_vals + ([0.0] if include_null else [])
            mean_v = sum(all_vals) / float(len(all_vals))
            centered = [max(-2.0, min(2.0, (v - mean_v) * self.causal_prior_scale)) for v in all_vals]
            return (
                torch.tensor(centered, device=dev, dtype=torch.float32),
                {cid: round(raw_vals[i], 5) for i, cid in enumerate(ids)},
            )

        out_prior_full, causal_map = _build_prior(causal_values, bool(allow_null_out))
        obs_vals_eff = observe_causal_values if observe_causal_values is not None else causal_values
        in_prior_full, _ = _build_prior(obs_vals_eff, bool(allow_null_in))

        # 1) OUT channel (`send_target`): active whenever MengLi speaks; `'none'` when silent.
        out_ids = list(ids) + (['none'] if allow_null_out else [])
        out_sampled = False
        out_idx = None
        out_causal_prior = None
        if not spoke:
            send_target = 'none'
            send_probs_map = {'none': 1.0}
            send_habit_map = {'none': 0.0}
            out_entropy = 0.0
        else:
            out_obj_logits = self._compute_habit_logits(h, u, ctx, keys, recency, obj_states=obj_states)
            out_habit = (
                torch.cat((out_obj_logits, dual.out_null_logit.reshape(1)), dim=0)
                if allow_null_out else out_obj_logits
            )
            out_causal_prior = out_prior_full
            out_total = out_habit + out_causal_prior
            out_probs_t = F.softmax(out_total, dim=-1)
            out_probs_np = out_probs_t.detach().cpu().numpy().astype(np.float64)
            out_probs_np = out_probs_np / max(float(out_probs_np.sum()), 1e-12)
            out_idx = int(rng.choice(len(out_ids), p=out_probs_np)) if rng is not None else int(torch.multinomial(out_probs_t.clamp_min(1e-8), 1).item())
            send_target = out_ids[out_idx]
            out_sampled = (len(out_ids) > 1)
            out_entropy = float((-(out_probs_t * F.log_softmax(out_total, dim=-1)).sum()).item())
            send_probs_map = {cid: round(float(out_probs_np[i]), 4) for i, cid in enumerate(out_ids)}
            send_habit_map = {cid: round(float(out_habit[i].item()), 4) for i, cid in enumerate(out_ids)}

        # 2) IN channel (`observe_target`): always selects who MengLi actively observes (`O ∪ {∅}`).
        in_ids = list(ids) + (['none'] if allow_null_in else [])
        in_obj_logits = self._compute_observe_logits(h, u, ctx, keys, recency, obj_states=obj_states)
        in_habit = (
            torch.cat((in_obj_logits, dual.in_null_logit.reshape(1)), dim=0)
            if allow_null_in else in_obj_logits
        )
        in_causal_prior = in_prior_full
        in_total = in_habit + in_causal_prior
        in_probs_t = F.softmax(in_total, dim=-1)
        in_probs_np = in_probs_t.detach().cpu().numpy().astype(np.float64)
        in_probs_np = in_probs_np / max(float(in_probs_np.sum()), 1e-12)
        in_idx = int(rng.choice(len(in_ids), p=in_probs_np)) if rng is not None else int(torch.multinomial(in_probs_t.clamp_min(1e-8), 1).item())
        observe_target = in_ids[in_idx]
        in_sampled = (len(in_ids) > 1)
        in_entropy = float((-(in_probs_t * F.log_softmax(in_total, dim=-1)).sum()).item())
        in_probs_map = {cid: round(float(in_probs_np[i]), 4) for i, cid in enumerate(in_ids)}
        in_habit_map = {cid: round(float(in_habit[i].item()), 4) for i, cid in enumerate(in_ids)}

        # Primary legacy `target_id` for backwards-compatible telemetry & Wood consumers.
        primary_id = observe_target if observe_target != 'none' else (send_target if send_target != 'none' else 'none')
        mode_tag = classify_interaction_mode(spoke, send_target, observe_target)

        self._pending = dict(
            dual_channel=True,
            target_id=primary_id,
            send_target=send_target,
            observe_target=observe_target,
            interaction_mode=mode_tag,
            spoke=spoke,
            ids=list(ids),
            out_ids=list(out_ids),
            in_ids=list(in_ids),
            out_idx=out_idx,
            in_idx=in_idx,
            out_sampled=out_sampled,
            in_sampled=in_sampled,
            allow_null_out=bool(allow_null_out),
            allow_null_in=bool(allow_null_in),
            h=h.detach(),
            u=u.detach(),
            ctx=ctx.detach(),
            keys=keys.detach(),
            obj_states=(None if obj_states is None else obj_states.detach()),
            recency=recency.detach(),
            out_causal_prior=(None if out_causal_prior is None else out_causal_prior.detach()),
            in_causal_prior=in_causal_prior.detach(),
            probs=in_probs_map,
            send_probs=send_probs_map,
            observe_probs=in_probs_map,
            entropy=0.5 * (out_entropy + in_entropy) if out_sampled else in_entropy,
        )
        dec = dict(
            target_id=primary_id,
            send_target=send_target,
            observe_target=observe_target,
            interaction_mode=mode_tag,
            probs=in_probs_map,
            send_probs=send_probs_map,
            observe_probs=in_probs_map,
            habit_logits=in_habit_map,
            send_habit_logits=send_habit_map,
            causal_values=causal_map,
            entropy=round(float(self._pending['entropy']), 4),
            num_candidates=len(ids),
            sampled=bool(out_sampled or in_sampled),
            dual_channel=True,
        )
        self.last_decision = dec
        return dec

    @torch.no_grad()
    def select_target(
        self,
        engine,
        *,
        candidates: list[tuple[str, bytes]],
        utterance_bytes: bytes,
        causal_values: dict[str, float] | None = None,
        observe_causal_values: dict[str, float] | None = None,
        stumble_ratio: float = 1.0,
        nll_norm: float = 0.0,
        last_speaker: str | None = None,
        awaiting_from: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict:
        """Select in-scene Umwelt target by fusing fast gaze habit + slow causal consequence prior."""
        if not candidates:
            raise ValueError('gaze_pointer requires at least one in-scene candidate object')

        ids = [cid for cid, _ in candidates]
        for cid, intro in candidates:
            self.ensure_object(engine, cid, intro)

        if len(ids) == 1:
            self._pending = None
            dec = dict(
                target_id=ids[0],
                probs={ids[0]: 1.0},
                habit_logits={ids[0]: 0.0},
                causal_values=({ids[0]: float(causal_values[ids[0]])} if causal_values and ids[0] in causal_values else None),
                entropy=0.0,
                num_candidates=1,
                sampled=False,
            )
            self.last_decision = dec
            return dec

        dev = self.device
        h = read_body_state(engine, d_model=self.d_model, device=dev)
        u = encode_sensory_bytes(engine, utterance_bytes, d_model=self.d_model, device=dev)
        has_speech = 1.0 if utterance_bytes else 0.0
        stumble_c = max(-1.0, min(3.0, float(stumble_ratio) - 1.0))
        nll_c = max(-2.0, min(4.0, float(nll_norm)))
        awaiting_flag = 1.0 if awaiting_from is not None else 0.0
        ctx = torch.tensor([stumble_c, has_speech, nll_c, awaiting_flag], device=dev, dtype=torch.float32)

        key_list = [self.relative_fingerprint(cid) for cid in ids]
        keys = torch.stack(key_list, dim=0)  # [M, d_model]

        rec_rows = []
        for cid in ids:
            is_last = 1.0 if (last_speaker is not None and cid == last_speaker) else 0.0
            is_wait = 1.0 if (awaiting_from is not None and cid == awaiting_from) else 0.0
            rec_rows.append([is_last, is_wait])
        recency = torch.tensor(rec_rows, device=dev, dtype=torch.float32)  # [M, 2]

        habit_logits = self._compute_habit_logits(h, u, ctx, keys, recency)
        causal_prior = torch.zeros_like(habit_logits)
        causal_map = None
        if causal_values:
            raw_vals = [float(causal_values.get(cid, 0.0)) for cid in ids]
            if all(math.isfinite(v) for v in raw_vals):
                mean_v = sum(raw_vals) / float(len(raw_vals))
                centered = [max(-2.0, min(2.0, (v - mean_v) * self.causal_prior_scale)) for v in raw_vals]
                causal_prior = torch.tensor(centered, device=dev, dtype=torch.float32)
                causal_map = {cid: round(raw_vals[i], 5) for i, cid in enumerate(ids)}

        total_logits = habit_logits + causal_prior
        probs_t = F.softmax(total_logits, dim=-1)
        probs_np = probs_t.detach().cpu().numpy().astype(np.float64)
        probs_np = probs_np / max(float(probs_np.sum()), 1e-12)

        if rng is not None:
            chosen_idx = int(rng.choice(len(ids), p=probs_np))
        else:
            chosen_idx = int(torch.multinomial(probs_t.clamp_min(1e-8), 1).item())
        chosen_id = ids[chosen_idx]

        log_p = F.log_softmax(total_logits, dim=-1)
        entropy = float((-(probs_t * log_p).sum()).item())
        probs_map = {cid: round(float(probs_np[i]), 4) for i, cid in enumerate(ids)}
        habit_map = {cid: round(float(habit_logits[i].item()), 4) for i, cid in enumerate(ids)}

        self._pending = dict(
            target_id=chosen_id,
            chosen_idx=chosen_idx,
            ids=list(ids),
            h=h.detach(),
            u=u.detach(),
            ctx=ctx.detach(),
            keys=keys.detach(),
            recency=recency.detach(),
            causal_prior=causal_prior.detach(),
            probs=probs_map,
            entropy=entropy,
        )
        dec = dict(
            target_id=chosen_id,
            probs=probs_map,
            habit_logits=habit_map,
            causal_values=causal_map,
            entropy=round(entropy, 4),
            num_candidates=len(ids),
            sampled=True,
        )
        self.last_decision = dec
        return dec

    def settle_credit(
        self,
        metal_g: float | None,
        *,
        evidence_available: bool = True,
        boundary_context: dict | None = None,
    ) -> dict | None:
        """Update GazeObjectPointer from real `metal_g` AND target-conditioned response boundary MSE.

        Never updates gaze policy from missing-evidence zero gain (`evidence_available=False`).
        """
        pending = self._pending
        self._pending = None

        g_valid = (
            bool(evidence_available)
            and metal_g is not None
            and isinstance(metal_g, (int, float))
            and math.isfinite(metal_g)
        )

        # Check if we have a target-conditioned boundary supervision signal for `target_to_action` / `observe_to_action`.
        has_boundary = (
            isinstance(boundary_context, dict)
            and boundary_context.get('boundary_target') is not None
            and boundary_context.get('source') is not None
            and boundary_context.get('raw_action') is not None
            and (
                str(boundary_context.get('actual_target_id') or '') in self.fingerprints
                or str(boundary_context.get('actual_observe_id') or '') in self.fingerprints
            )
            and boundary_context.get('model') is not None
        )

        # Even when an un-scored internal turn (`in == 'none'`) has `g_valid == False`,
        # keep symmetric Active-Inference entropy regularization active on dual-channel
        # decisions (with ZERO reward/advantage) to prevent one-way dark-room collapse.
        dual_entropy_only = (
            pending is not None
            and bool(pending.get('dual_channel'))
            and (not g_valid)
            and bool(pending.get('out_sampled') or pending.get('in_sampled'))
        )

        if (pending is None or (not g_valid and not dual_entropy_only)) and not has_boundary:
            return None

        dev = self.device
        total_loss = torch.zeros((), device=dev, dtype=torch.float32)
        gaze_adv = None
        entropy_val = None
        g_val = float(metal_g) if g_valid else None
        boundary_mse_val = None
        other_model_receipt = None
        target_id = (
            pending['target_id'] if pending is not None
            else str(boundary_context['actual_target_id']) if has_boundary
            else None
        )

        # Deferred state update payload for `_OtherWorldModel` (applied after optimizer step).
        pending_other_update = None

        with torch.enable_grad():
            if pending is not None and (g_valid or dual_entropy_only):
                if g_valid:
                    steps = int(self.step_count.item())
                    base_g = float(self.baseline_g.item()) if steps > 0 else g_val
                    gaze_adv = max(-0.25, min(0.25, g_val - base_g))
                if pending.get('dual_channel'):
                    dual = self._dual()
                    ent_terms = []
                    p_obj_states = pending.get('obj_states')
                    if pending.get('out_sampled') and pending.get('out_idx') is not None:
                        o_obj = self._compute_habit_logits(
                            pending['h'], pending['u'], pending['ctx'], pending['keys'], pending['recency'],
                            obj_states=p_obj_states,
                        )
                        o_h = (
                            torch.cat((o_obj, dual.out_null_logit.reshape(1)), dim=0)
                            if pending.get('allow_null_out') else o_obj
                        )
                        o_logits = o_h + pending['out_causal_prior']
                        o_lp = F.log_softmax(o_logits, dim=-1)
                        o_ent = -(o_lp.exp() * o_lp).sum()
                        if g_valid:
                            total_loss = total_loss - (gaze_adv * o_lp[int(pending['out_idx'])])
                        total_loss = total_loss - (self.entropy_coef * o_ent)
                        ent_terms.append(float(o_ent.item()))
                    if pending.get('in_sampled') and pending.get('in_idx') is not None:
                        i_obj = self._compute_observe_logits(
                            pending['h'], pending['u'], pending['ctx'], pending['keys'], pending['recency'],
                            obj_states=p_obj_states,
                        )
                        i_h = (
                            torch.cat((i_obj, dual.in_null_logit.reshape(1)), dim=0)
                            if pending.get('allow_null_in') else i_obj
                        )
                        i_logits = i_h + pending['in_causal_prior']
                        i_lp = F.log_softmax(i_logits, dim=-1)
                        i_ent = -(i_lp.exp() * i_lp).sum()
                        if g_valid:
                            total_loss = total_loss - (gaze_adv * i_lp[int(pending['in_idx'])])
                        total_loss = total_loss - (self.entropy_coef * i_ent)
                        ent_terms.append(float(i_ent.item()))
                    entropy_val = (sum(ent_terms) / len(ent_terms)) if ent_terms else 0.0
                else:
                    chosen_idx = int(pending['chosen_idx'])
                    habit_logits = self._compute_habit_logits(
                        pending['h'], pending['u'], pending['ctx'], pending['keys'], pending['recency'],
                    )
                    logits = habit_logits + pending['causal_prior']
                    log_probs = F.log_softmax(logits, dim=-1)
                    probs = log_probs.exp()
                    entropy = -(probs * log_probs).sum()
                    entropy_val = float(entropy.item())
                    total_loss = total_loss - (gaze_adv * log_probs[chosen_idx]) - (self.entropy_coef * entropy)

            if has_boundary:
                model = boundary_context['model']
                src = boundary_context['source'].detach().to(device=dev, dtype=torch.float32).reshape(1, -1)
                raw_act = boundary_context['raw_action'].detach().to(device=dev, dtype=torch.float32).reshape(1, -1)
                tgt_feat = boundary_context['boundary_target'].detach().to(device=dev, dtype=torch.float32).reshape(1, -1)
                tid = str(boundary_context['actual_target_id'])
                obs_tid = boundary_context.get('actual_observe_id')
                if raw_act.shape[-1] == self.d_model and hasattr(model, 'encoder'):
                    act_scale = raw_act.detach().norm(dim=-1, keepdim=True).clamp_min(1.0)
                    cond_act = raw_act + self._compute_delta(tid, act_scale, observe_id=obs_tid)
                    enc_w = model.encoder.weight
                    enc_dev, enc_dt = enc_w.device, enc_w.dtype
                    src_m = src.to(device=enc_dev, dtype=enc_dt)
                    cond_m = cond_act.to(device=enc_dev, dtype=enc_dt)
                    tgt_m = tgt_feat.to(device=enc_dev, dtype=enc_dt)
                    enc_tgt = model.encoder(tgt_m).detach()
                    # Match Wood's actual causal prior (`causal_relation_program.py`):
                    # `long_predictor(cat(z_s, z_a)) + mother_from_boundary(...)`.
                    def _base_boundary_pred(act_tensor_m: torch.Tensor) -> torch.Tensor:
                        if hasattr(model, 'long_predictor'):
                            z_s = model.encoder(src_m).detach()
                            z_a = model.encoder(act_tensor_m)
                            bp = model.long_predictor(torch.cat((z_s, z_a), -1))
                            return bp + model.mother_from_boundary(bp) if hasattr(model, 'mother_from_boundary') else bp
                        if hasattr(model, 'internal_transition'):
                            z_s = model.encoder(src_m).detach()
                            z_a = model.encoder(act_tensor_m)
                            bp = z_s + model.internal_transition(torch.cat((z_s, z_a - z_s), -1))
                            return bp + model.mother_from_boundary(bp) if hasattr(model, 'mother_from_boundary') else bp
                        return model.encoder(act_tensor_m)

                    pred = _base_boundary_pred(cond_m)
                    b_mse = F.mse_loss(pred.float(), enc_tgt.float()).to(device=dev, dtype=torch.float32)
                    boundary_mse_val = float(b_mse.item())
                    total_loss = total_loss + self.boundary_loss_weight * b_mse

                    # Per-object persistent Other/World latent state prediction + 1-step BPTT gate learning (opt-in).
                    eff_obs = (
                        str(obs_tid)
                        if bool(getattr(self, 'enable_other_world', False))
                        and obs_tid not in (None, '', 'none') and str(obs_tid) in self.fingerprints
                        else (
                            tid
                            if bool(getattr(self, 'enable_other_world', False))
                            and tid not in (None, '', 'none') and tid in self.fingerprints
                            else None
                        )
                    )
                    if eff_obs is not None:
                        d_lat = int(enc_tgt.shape[-1])
                        owm = self._ensure_other_world(d_lat)
                        dual = self._dual()
                        k_in = dual.k_in_proj(self.relative_fingerprint(eff_obs)).detach().reshape(1, -1)
                        k_out = (
                            self.k_proj(self.relative_fingerprint(tid)).detach().reshape(1, -1)
                            if tid not in (None, '', 'none') and tid in self.fingerprints
                            else torch.zeros(1, self.d_gaze, device=dev, dtype=torch.float32)
                        )
                        prev_tr = self._other_prev_transitions.get(eff_obs)
                        if prev_tr is not None and prev_tr[0].numel() == d_lat:
                            z_pp, z_pa, k_prev = (t.to(device=dev, dtype=torch.float32) for t in prev_tr)
                            gate_prev = owm.state_gate(z_pp, z_pa, k_prev)
                            z_obj_for_pred = (1.0 - gate_prev) * z_pp + gate_prev * z_pa
                        else:
                            z_cur0 = self.other_states.get(eff_obs)
                            if z_cur0 is not None and z_cur0.numel() == d_lat:
                                z_obj_for_pred = z_cur0.to(device=dev, dtype=torch.float32).reshape(1, d_lat)
                            else:
                                z_obj_for_pred = torch.zeros(1, d_lat, device=dev, dtype=torch.float32)
                        base_f = pred.detach().float().to(device=dev).reshape(1, d_lat)
                        tgt_f = enc_tgt.detach().float().to(device=dev).reshape(1, d_lat)
                        z_pred = base_f + owm.predict_residual(base_f, z_obj_for_pred, k_in, k_out)
                        obj_mse = F.mse_loss(z_pred, tgt_f)
                        total_loss = total_loss + self.boundary_loss_weight * obj_mse

                        # Counterfactual action-conditioned sensitivity: how much MengLi's speech
                        # shifts her prediction of this object's response boundary vs silence.
                        action_shift_val = 0.0
                        with torch.no_grad():
                            if float((raw_act - src).norm().item()) > 1e-4:
                                sil_scale = src.norm(dim=-1, keepdim=True).clamp_min(1.0)
                                sil_cond = (src + self._compute_delta(tid, sil_scale, observe_id=obs_tid)).to(
                                    device=enc_dev, dtype=enc_dt,
                                )
                                sil_base_f = _base_boundary_pred(sil_cond).float().to(device=dev).reshape(1, d_lat)
                                sil_z_pred = sil_base_f + owm.predict_residual(
                                    sil_base_f, z_obj_for_pred.detach(), k_in, k_out,
                                )
                                action_shift_val = float((z_pred.detach() - sil_z_pred).norm().item())

                        pending_other_update = dict(
                            obj_id=eff_obs,
                            d_lat=d_lat,
                            tgt_f=tgt_f.detach(),
                            k_in=k_in.detach(),
                            base_mse=float(b_mse.item()),
                            pred_mse=float(obj_mse.item()),
                            gain=float(b_mse.item()) - float(obj_mse.item()),
                            action_shift=action_shift_val,
                        )

            if total_loss.requires_grad:
                # Restrict autograd strictly to GazeObjectPointer + `_dual_channel` + `_other_world_model`
                # so frozen body modules (`encoder`, `long_predictor`, `mother_from_boundary`) never get `.grad`.
                main_params = [p for p in self.parameters() if p.requires_grad]
                dual_params = [p for p in self._dual().parameters() if p.requires_grad]
                owm_cur: _OtherWorldModel | None = (
                    self.__dict__.get('_other_world_model')
                    if bool(getattr(self, 'enable_other_world', False)) else None
                )
                other_params = [p for p in owm_cur.parameters() if p.requires_grad] if owm_cur is not None else []
                all_params = main_params + dual_params + other_params
                grads = torch.autograd.grad(total_loss, all_params, allow_unused=True)
                self._opt.zero_grad(set_to_none=True)
                self._dual_opt.zero_grad(set_to_none=True)
                if self._other_opt is not None:
                    self._other_opt.zero_grad(set_to_none=True)
                main_has_grad = False
                dual_has_grad = False
                other_has_grad = False
                n_main = len(main_params)
                n_dual = len(dual_params)
                for p, g in zip(main_params, grads[:n_main]):
                    if g is not None:
                        p.grad = g.detach()
                        main_has_grad = True
                for p, g in zip(dual_params, grads[n_main:n_main + n_dual]):
                    if g is not None:
                        p.grad = g.detach()
                        dual_has_grad = True
                for p, g in zip(other_params, grads[n_main + n_dual:]):
                    if g is not None:
                        p.grad = g.detach()
                        other_has_grad = True
                if main_has_grad:
                    torch.nn.utils.clip_grad_norm_(main_params, 1.0)
                    self._opt.step()
                if dual_has_grad:
                    torch.nn.utils.clip_grad_norm_(dual_params, 1.0)
                    self._dual_opt.step()
                    with torch.no_grad():
                        dual_m = self._dual()
                        dual_m.in_null_logit.clamp_(-3.0, 1.5)
                        dual_m.out_null_logit.clamp_(-3.0, 1.5)
                if other_has_grad and self._other_opt is not None:
                    torch.nn.utils.clip_grad_norm_(other_params, 1.0)
                    self._other_opt.step()

        with torch.no_grad():
            if pending_other_update is not None:
                eff_obs = pending_other_update['obj_id']
                d_lat = pending_other_update['d_lat']
                owm = self._ensure_other_world(d_lat)
                z_cur = self.other_states.get(eff_obs)
                if z_cur is None or z_cur.numel() != d_lat:
                    z_cur = torch.zeros(1, d_lat, device=dev, dtype=torch.float32)
                else:
                    z_cur = z_cur.to(device=dev, dtype=torch.float32).reshape(1, d_lat)
                tgt_f = pending_other_update['tgt_f']
                k_in = pending_other_update['k_in']
                gate_now = owm.state_gate(z_cur, tgt_f, k_in)
                z_next = (1.0 - gate_now) * z_cur + gate_now * tgt_f
                self._other_prev_transitions[eff_obs] = (z_cur.detach(), tgt_f.detach(), k_in.detach())
                self.other_states[eff_obs] = z_next.detach()
                gate_mean = float(gate_now.mean().item())
                prev_ema = self.other_model_ema.get(eff_obs)
                if prev_ema is None:
                    self.other_model_ema[eff_obs] = dict(
                        count=1,
                        base_mse=pending_other_update['base_mse'],
                        pred_mse=pending_other_update['pred_mse'],
                        gain=pending_other_update['gain'],
                        gate_mean=gate_mean,
                        action_shift=pending_other_update['action_shift'],
                    )
                else:
                    a_ema = 0.1
                    prev_ema['count'] = int(prev_ema.get('count', 0)) + 1
                    prev_ema['base_mse'] = (1.0 - a_ema) * float(prev_ema['base_mse']) + a_ema * pending_other_update['base_mse']
                    prev_ema['pred_mse'] = (1.0 - a_ema) * float(prev_ema['pred_mse']) + a_ema * pending_other_update['pred_mse']
                    prev_ema['gain'] = (1.0 - a_ema) * float(prev_ema['gain']) + a_ema * pending_other_update['gain']
                    prev_ema['gate_mean'] = (1.0 - a_ema) * float(prev_ema['gate_mean']) + a_ema * gate_mean
                    prev_ema['action_shift'] = (1.0 - a_ema) * float(prev_ema['action_shift']) + a_ema * pending_other_update['action_shift']
                other_model_receipt = dict(
                    object_id=eff_obs,
                    base_mse=round(pending_other_update['base_mse'], 6),
                    pred_mse=round(pending_other_update['pred_mse'], 6),
                    gain=round(pending_other_update['gain'], 6),
                    gate_mean=round(gate_mean, 4),
                    action_shift=round(pending_other_update['action_shift'], 5),
                )
            if pending is not None and g_valid and entropy_val is not None:
                steps = int(self.step_count.item())
                if steps == 0:
                    self.baseline_g.fill_(g_val)
                    self.mean_entropy.fill_(entropy_val)
                else:
                    d = self.baseline_decay
                    self.baseline_g.mul_(d).add_((1.0 - d) * g_val)
                    self.mean_entropy.mul_(d).add_((1.0 - d) * entropy_val)
                self.step_count.add_(1)
                self.target_counts[target_id] = int(self.target_counts.get(target_id, 0)) + 1
                prev_ema = self.target_g_ema.get(target_id)
                self.target_g_ema[target_id] = g_val if prev_ema is None else (0.8 * prev_ema + 0.2 * g_val)
                mode_tag = pending.get('interaction_mode')
                if isinstance(mode_tag, str) and mode_tag:
                    self.interaction_mode_counts[mode_tag] = int(self.interaction_mode_counts.get(mode_tag, 0)) + 1
            if boundary_mse_val is not None:
                self.boundary_steps.add_(1)

        return dict(
            target_id=target_id,
            send_target=(pending.get('send_target') if pending is not None else None),
            observe_target=(pending.get('observe_target') if pending is not None else None),
            interaction_mode=(pending.get('interaction_mode') if pending is not None else None),
            metal_g=(None if g_val is None else round(g_val, 6)),
            evidence_available=bool(g_valid),
            advantage=(None if gaze_adv is None else round(gaze_adv, 6)),
            boundary_mse=(None if boundary_mse_val is None else round(boundary_mse_val, 6)),
            other_model=other_model_receipt,
            entropy=(None if entropy_val is None else round(entropy_val, 4)),
            step_count=int(self.step_count.item()),
            boundary_steps=int(self.boundary_steps.item()),
            probs=(dict(pending['probs']) if pending is not None else None),
        )

    def _export_opt(self, opt: torch.optim.Optimizer) -> dict:
        raw = opt.state_dict()
        enc_state = {}
        for pid, st in raw.get('state', {}).items():
            enc_st = {}
            for k, v in st.items():
                if torch.is_tensor(v):
                    enc_st[k] = dict(__tensor__=True, **_tensor_to_b64(v))
                else:
                    enc_st[k] = v
            enc_state[str(pid)] = enc_st
        return dict(state=enc_state, param_groups=copy.deepcopy(raw.get('param_groups', [])))

    def _load_opt(self, opt: torch.optim.Optimizer, payload: dict | None) -> None:
        if not isinstance(payload, dict) or 'state' not in payload:
            return
        dec_state = {}
        for pid_str, st in payload['state'].items():
            dec_st = {}
            for k, v in st.items():
                if isinstance(v, dict) and v.get('__tensor__'):
                    target_dev = torch.device('cpu') if k == 'step' else self.device
                    dec_st[k] = _b64_to_tensor(v, device=target_dev)
                else:
                    dec_st[k] = v
            dec_state[int(pid_str)] = dec_st
        restored = dict(state=dec_state, param_groups=copy.deepcopy(payload.get('param_groups', [])))
        opt.load_state_dict(restored)

    def _export_optimizer_state(self) -> dict:
        return self._export_opt(self._opt)

    def _load_optimizer_state(self, payload: dict | None) -> None:
        self._load_opt(self._opt, payload)

    def export_dynamic_state(self) -> dict:
        """Export exact fingerprints, global center, dual-channel & optional Other/World weights + bit-exact Adam state."""
        dual = self._dual()
        dual_weights = {k: _tensor_to_b64(v) for k, v in dual.state_dict().items()}
        owm: _OtherWorldModel | None = (
            self.__dict__.get('_other_world_model')
            if bool(getattr(self, 'enable_other_world', False)) else None
        )
        other_payload = {}
        if owm is not None:
            other_payload = dict(
                other_world_v1=True,
                other_model_d_latent=int(owm.d_latent),
                other_model_weights_b64={k: _tensor_to_b64(v) for k, v in owm.state_dict().items()},
                other_model_optimizer_state_b64=(
                    None if self._other_opt is None else self._export_opt(self._other_opt)
                ),
                other_states_b64={k: _tensor_to_b64(v.detach().cpu().float()) for k, v in self.other_states.items()},
                other_prev_transitions_b64={
                    k: [_tensor_to_b64(t.detach().cpu().float()) for t in trip]
                    for k, trip in self._other_prev_transitions.items()
                },
                other_model_ema=copy.deepcopy(self.other_model_ema),
            )
        return dict(
            schema=2,
            centered_target_v1=True,
            dual_channel_v1=True,
            fingerprints={k: v.detach().cpu().float().tolist() for k, v in self.fingerprints.items()},
            fingerprints_b64={k: _tensor_to_b64(v.detach().cpu().float()) for k, v in self.fingerprints.items()},
            fp_center=(None if self.fp_center is None else self.fp_center.detach().cpu().float().tolist()),
            fp_center_b64=(None if self.fp_center is None else _tensor_to_b64(self.fp_center.detach().cpu().float())),
            target_counts=dict(self.target_counts),
            target_g_ema={k: float(v) for k, v in self.target_g_ema.items()},
            interaction_mode_counts=dict(self.interaction_mode_counts),
            optimizer_state_b64=self._export_optimizer_state(),
            dual_weights_b64=dual_weights,
            dual_optimizer_state_b64=self._export_opt(self._dual_opt),
            **other_payload,
        )

    def load_dynamic_state(self, state: dict | None) -> None:
        if not isinstance(state, dict) or state.get('schema') not in (1, 2):
            return
        self.fingerprints = {}
        fp_b64 = state.get('fingerprints_b64')
        if isinstance(fp_b64, dict) and fp_b64:
            for k, payload in fp_b64.items():
                t = _b64_to_tensor(payload, device=self.device).float()
                if t.numel() == self.d_model:
                    self.fingerprints[str(k)] = t.detach()
        else:
            for k, vals in (state.get('fingerprints') or {}).items():
                t = torch.tensor(vals, device=self.device, dtype=torch.float32)
                if t.numel() == self.d_model:
                    self.fingerprints[str(k)] = F.normalize(t, dim=-1, eps=1e-6).detach()
        center_b64 = state.get('fp_center_b64')
        center_vals = state.get('fp_center')
        if isinstance(center_b64, dict) and center_b64.get('b64'):
            ct = _b64_to_tensor(center_b64, device=self.device).float()
            self.fp_center = ct.detach() if ct.numel() == self.d_model else None
        elif center_vals is not None:
            ct = torch.tensor(center_vals, device=self.device, dtype=torch.float32)
            self.fp_center = ct.detach() if ct.numel() == self.d_model else None
        else:
            self.fp_center = None
            self._refresh_fp_center(snap=True)
        self.target_counts = {str(k): int(v) for k, v in (state.get('target_counts') or {}).items()}
        self.target_g_ema = {str(k): float(v) for k, v in (state.get('target_g_ema') or {}).items()}
        self.interaction_mode_counts = {str(k): int(v) for k, v in (state.get('interaction_mode_counts') or {}).items()}
        if state.get('schema') == 2 and state.get('optimizer_state_b64') is not None:
            self._load_optimizer_state(state['optimizer_state_b64'])
        if state.get('dual_channel_v1') and isinstance(state.get('dual_weights_b64'), dict):
            dual = self._dual()
            restored_w = {k: _b64_to_tensor(v, device=self.device) for k, v in state['dual_weights_b64'].items()}
            dual.load_state_dict(restored_w, strict=False)
            if state.get('dual_optimizer_state_b64') is not None:
                self._load_opt(self._dual_opt, state['dual_optimizer_state_b64'])
        self.other_states = {}
        self._other_prev_transitions = {}
        self.other_model_ema = {}
        if state.get('other_world_v1') and isinstance(state.get('other_model_weights_b64'), dict):
            d_lat = int(state.get('other_model_d_latent') or 0)
            if d_lat > 0:
                owm = self._ensure_other_world(d_lat)
                owm_w = {k: _b64_to_tensor(v, device=self.device) for k, v in state['other_model_weights_b64'].items()}
                owm.load_state_dict(owm_w, strict=False)
                if state.get('other_model_optimizer_state_b64') is not None and self._other_opt is not None:
                    self._load_opt(self._other_opt, state['other_model_optimizer_state_b64'])
                for k, payload in (state.get('other_states_b64') or {}).items():
                    self.other_states[str(k)] = _b64_to_tensor(payload, device=self.device).float().detach()
                for k, trip in (state.get('other_prev_transitions_b64') or {}).items():
                    if isinstance(trip, list) and len(trip) == 3:
                        self._other_prev_transitions[str(k)] = tuple(
                            _b64_to_tensor(p, device=self.device).float().detach() for p in trip
                        )
                if isinstance(state.get('other_model_ema'), dict):
                    self.other_model_ema = copy.deepcopy(state['other_model_ema'])
        if not state.get('centered_target_v1'):
            with torch.no_grad():
                self.target_to_action.weight.zero_()
            self._opt.state.pop(self.target_to_action.weight, None)
        self._pending = None

    def _supply_metrics(self) -> dict:
        """SADL-inspired upstream separation telemetry (raw cone vs centered relative keys)."""
        keys = sorted(self.fingerprints.keys())
        if len(keys) < 2:
            return dict(raw_fp_max_cos=None, rel_fp_max_cos=None, delta_unit_sep=None)
        raw_max = -1.0
        rel_max = -1.0
        d_sep_max = 0.0
        with torch.no_grad():
            raws = [self.fingerprints[k].to(self.device) for k in keys]
            rels = [self.relative_fingerprint(k) for k in keys]
            deltas = [self._compute_delta(k, 1.0) for k in keys]
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    rc = float(F.cosine_similarity(raws[i][None], raws[j][None]).item())
                    raw_max = max(raw_max, rc)
                    if float(rels[i].norm()) > 1e-4 and float(rels[j].norm()) > 1e-4:
                        kc = float(F.cosine_similarity(rels[i][None], rels[j][None]).item())
                        rel_max = max(rel_max, kc)
                    d_sep_max = max(d_sep_max, float((deltas[i] - deltas[j]).norm().item()))
        return dict(
            raw_fp_max_cos=round(raw_max, 5),
            rel_fp_max_cos=round(rel_max, 5),
            delta_unit_sep=round(d_sep_max, 5),
        )

    def diagnostics(self) -> dict:
        supply = self._supply_metrics()
        return dict(
            schema=2,
            centered_target_v1=True,
            d_model=self.d_model,
            d_gaze=self.d_gaze,
            param_count=sum(p.numel() for p in self.parameters()),
            step_count=int(self.step_count.item()),
            boundary_steps=int(self.boundary_steps.item()),
            target_to_action_norm=round(float(self.target_to_action.weight.detach().norm().item()), 6),
            raw_fp_max_cos=supply['raw_fp_max_cos'],
            rel_fp_max_cos=supply['rel_fp_max_cos'],
            delta_unit_sep=supply['delta_unit_sep'],
            baseline_g=round(float(self.baseline_g.item()), 6),
            mean_entropy=round(float(self.mean_entropy.item()), 4),
            known_objects=sorted(self.fingerprints.keys()),
            target_counts=dict(self.target_counts),
            target_g_ema={k: round(float(v), 6) for k, v in self.target_g_ema.items()},
            interaction_mode_counts=dict(self.interaction_mode_counts),
            other_world_model={
                k: {mk: (int(mv) if mk == 'count' else round(float(mv), 5)) for mk, mv in v.items()}
                for k, v in self.other_model_ema.items()
            },
            last_decision=copy.deepcopy(self.last_decision),
        )


def enabled(engine) -> bool:
    return (
        engine is not None
        and getattr(engine, 'gaze_pointer_schema', None) in (1, 2)
        and hasattr(getattr(engine, 'model', None), 'gaze_pointer')
        and isinstance(engine.model.gaze_pointer, GazeObjectPointer)
    )


def configure(
    engine,
    config: dict | None = None,
    *,
    d_gaze: int = 64,
    lr: float = 1.5e-3,
    entropy_coef: float = 0.025,
    causal_prior_scale: float = 1.0,
    boundary_loss_weight: float = 0.25,
    fp_ema: float = 0.25,
    baseline_decay: float = 0.90,
    enable_other_world: bool = False,
    initialize: bool = True,
) -> GazeObjectPointer:
    """Attach opt-in O(1) GazeObjectPointer cleanly to MengLi's model."""
    d_model, device = _resolve_model_dim_and_device(engine)
    if hasattr(engine.model, 'gaze_pointer') and isinstance(engine.model.gaze_pointer, GazeObjectPointer):
        mod = engine.model.gaze_pointer
        mod.enable_other_world = bool(enable_other_world)
    else:
        mod = GazeObjectPointer(
            d_model,
            d_gaze=d_gaze,
            lr=lr,
            entropy_coef=entropy_coef,
            causal_prior_scale=causal_prior_scale,
            boundary_loss_weight=boundary_loss_weight,
            fp_ema=fp_ema,
            baseline_decay=baseline_decay,
            enable_other_world=enable_other_world,
            device=device,
        )
        engine.model.gaze_pointer = mod
    engine.gaze_pointer_schema = 2
    engine._gaze_boundary_context = None
    try:
        from scripts.ib_local import run_navigation_residual_wood as _legacy
        _legacy._DERIVED.add('_gaze_boundary_context')
    except Exception:
        pass
    if config is not None:
        config['gaze_pointer'] = dict(
            schema=2,
            d_gaze=int(d_gaze),
            lr=float(lr),
            entropy_coef=float(entropy_coef),
            causal_prior_scale=float(causal_prior_scale),
            boundary_loss_weight=float(boundary_loss_weight),
            fp_ema=float(fp_ema),
            baseline_decay=float(baseline_decay),
            enable_other_world=bool(enable_other_world),
        )
    return mod


def diagnostics(engine) -> dict | None:
    if not enabled(engine):
        return None
    return engine.model.gaze_pointer.diagnostics()
