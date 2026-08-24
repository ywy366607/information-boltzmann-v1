"""One DualStream field loop for native text, image, and dense readouts."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fine_grain.bayesian_surprise import compute_point_vfe, reduce_observation_f
from fine_grain.flow_match import v_from_x_pred
from fine_grain.native_mot import slice_mass_loss_weights
from fine_grain.active_gdn2 import ActiveInferenceGDN2
from fine_grain.sigreg import compute_sigreg_loss
from fine_grain.unified_arch import unified_kwargs
from fine_grain.vlm_data import COLORS, KINK_KS, OCR_DIGITS
from scripts.run_v0_surprise_eval import DualStreamVQAModel


def token_nll_per_sample(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shifted causal NLL, one scalar per row. Empty rows are 0, not NaN."""
    # Pythia runs in fp16 on the 4 GB development GPU. Reducing hundreds of
    # answer-token losses in fp16 quantizes matched-vs-shuffled gaps to zero
    # even when the frozen decoder has a small visual preference. Keep the LM
    # frozen and its forward cheap, but evaluate its categorical likelihood in
    # fp32 so the variational accuracy term and its counterfactual gradient are
    # numerically identifiable.
    shift_logits = logits[:, :-1].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    vocab = shift_logits.shape[-1]
    nll = F.cross_entropy(
        shift_logits.reshape(-1, vocab),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    valid = shift_labels.ne(-100)
    counts = valid.sum(dim=1)
    summed = (nll * valid).sum(dim=1)
    return torch.where(
        counts > 0,
        summed / counts.clamp_min(1).to(dtype=nll.dtype),
        nll.new_zeros(nll.shape[0]),
    )


def rgb_energy_weights(tgt: torch.Tensor, signed: bool) -> torch.Tensor:
    """Per-pixel weights so a miss on yellow ≈ a miss on red/green/blue.

    Signed black is -1. One-channel ink vs black has L2²=4; yellow [1,1,0]
    has 8. Weight = ref / ||tgt−bg||² on non-bg pixels.
    """
    bg = -1.0 if signed else 0.0
    energy = (tgt - bg).pow(2).sum(dim=1, keepdim=True)
    ref = 4.0 if signed else 1.0
    return torch.where(
        energy > 1e-6,
        energy.new_tensor(ref) / energy.clamp_min(1e-6),
        torch.ones_like(energy),
    )


def balanced_observation_bce(
    pred: torch.Tensor,
    target: torch.Tensor,
    signed: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    """Equal-weight observed figure/ground BCE for black-field generation.

    Figure membership comes only from deviation from the known null boundary;
    it is not a digit/class mask. This prevents thin observations from being
    drowned by the full-resolution ground while charging flood equally.
    """
    if signed:
        pred = (pred + 1.0) * 0.5
        target = (target + 1.0) * 0.5
    pred = pred.clamp(1e-5, 1.0 - 1e-5)
    target = target.clamp(0.0, 1.0)
    err = F.binary_cross_entropy(pred, target, reduction="none").mean(dim=1)
    figure = target.abs().amax(dim=1) > 1e-5
    ground = ~figure

    def pane(mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(err.dtype)
        return (err * weight).sum(dim=(1, 2)) / weight.sum(dim=(1, 2)).clamp_min(1.0)

    value = 0.5 * pane(figure) + 0.5 * pane(ground)
    if reduction == "none":
        return value
    if reduction != "mean":
        raise ValueError(f"unknown reduction={reduction!r}")
    return value.mean()


class DualStreamOmni(DualStreamVQAModel):
    """Champion-B DualStream + per-point RGB head on the live field X."""

    EXTRA_WORDS = (
        "Draw", "Reconstruct", "Restore", "Paint", "image", "blank",
        "noisy", "this", "to", "a", "has", "polyline",
    )
    EXTRA_SPATIAL_WORDS = (
        "at", "top", "middle", "bottom", "left", "center", "right",
    )
    EXTRA_CAPABILITY_WORDS = (
        "Change", "change", "next", "color", "frame", "Predict",
        "current", "future", "foreground", "Read",
    )

    @classmethod
    def unified(cls, **overrides):
        """Single residual-write graph. Overrides are size / data only."""
        return cls(**unified_kwargs(**overrides))

    def __init__(self, **kwargs):
        self.fm_pred = str(kwargs.pop("fm_pred", "x")).lower()
        self.fm_signed = bool(kwargs.pop("fm_signed", False))
        self.spatial_prompt_vocab = bool(kwargs.pop("spatial_prompt_vocab", False))
        self.capability_vocab = bool(kwargs.pop("capability_vocab", False))
        self.pixel_loss_mode = str(kwargs.pop("pixel_loss_mode", "vfe")).lower()
        if self.pixel_loss_mode not in ("vfe", "balanced_bce", "gaussian_nll"):
            raise ValueError(f"unknown pixel_loss_mode={self.pixel_loss_mode!r}")
        self.s0_acc_coef = float(kwargs.pop("s0_acc_coef", 1.0))
        self.seg_classes = int(kwargs.pop("seg_classes", 0))
        self.seg_loss_coef = float(kwargs.pop("seg_loss_coef", 1.0))
        self.transition_loss_coef = float(kwargs.pop("transition_loss_coef", 0.0))
        self.transition_posterior_loss_coef = float(
            kwargs.pop("transition_posterior_loss_coef", 1.0)
        )
        self.transition_detach_q = bool(kwargs.pop("transition_detach_q", True))
        self.causal_memory_loss_coef = float(
            kwargs.pop("causal_memory_loss_coef", 0.0)
        )
        language = str(kwargs.pop("language", "toy")).lower()
        lm_device = str(kwargs.pop("lm_device", "cpu"))
        injected_lm = kwargs.pop("lm", None)
        injected_tok = kwargs.pop("lm_tok", None)
        lm_id = kwargs.pop("lm_id", None)
        lm_revision = kwargs.pop("lm_revision", None)
        self.lm_note = "toy"
        self.lm_id = None
        self.lm_revision = None
        kwargs.setdefault("surprise_mode", "v1_bayes")
        kwargs.setdefault("s_update", "rms_dir")
        lm = tok = note = None
        d_llm = None
        if injected_lm is not None:
            lm = injected_lm
            tok = injected_tok
            cfg = getattr(lm, "config", None)
            d_llm = int(
                getattr(cfg, "hidden_size", None)
                or lm.get_input_embeddings().embedding_dim
            )
            kwargs["d_llm"] = d_llm
            note = f"injected d_llm={d_llm}"
            self.lm_id = lm_id or "injected"
            self.lm_revision = lm_revision or "injected"
        elif language.startswith("pythia") or language in ("lm", "frozen_lm"):
            from fine_grain.llm_backend import canonical_pythia_id, load_frozen_pythia
            model_id = canonical_pythia_id(lm_id or language)
            lm, tok, d_llm, note, meta = load_frozen_pythia(
                model_id, device=lm_device, dtype=None, revision=lm_revision,
            )
            kwargs["d_llm"] = d_llm
            self.lm_id = meta["id"]
            self.lm_revision = meta.get("revision")
        super().__init__(**kwargs)
        if lm is not None:
            self.lm = lm
            self.lm_tok = tok
            self.lm_note = note
            self.add_module("lm", lm)
            self._freeze_lm()
        # grow vocab for generation prompts (old rows stay aligned)
        requested_words = self.EXTRA_WORDS
        if self.spatial_prompt_vocab:
            requested_words = requested_words + self.EXTRA_SPATIAL_WORDS
        if self.capability_vocab:
            # New rows must come last so published spatial-generator rows keep
            # their token IDs during capability fine-tuning.
            requested_words = requested_words + self.EXTRA_CAPABILITY_WORDS
        extra = [w for w in requested_words if w not in self.vocab]
        if extra:
            old = self.embed
            for w in extra:
                self.vocab[w] = len(self.vocab)
            new = nn.Embedding(len(self.vocab) + 10, self.d_model)
            with torch.no_grad():
                new.weight[: old.num_embeddings] = old.weight
                nn.init.normal_(new.weight[old.num_embeddings :], std=0.02)
            self.embed = new
        self.pix_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 3),
        )
        # Per-point log σ for p(rgb|X)=N(μ(X),σ(X)²). Zero linear ⇒ σ=1 at init
        # (NLL ≈ ½ MSE). Missing in old ckpts; load with strict=False.
        self.pix_logσ = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 3),
        )
        nn.init.zeros_(self.pix_logσ[-1].weight)
        nn.init.zeros_(self.pix_logσ[-1].bias)
        self.seg_head = None
        if self.seg_classes > 0:
            self.seg_head = nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.seg_classes),
            )
        # The same chart parameterizes posterior and dynamic-prior
        # uncertainty. Observation precision can therefore tighten q while
        # the unobserved rollout remains broad.
        self.belief_logvar_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
        )
        nn.init.zeros_(self.belief_logvar_head[-1].weight)
        nn.init.zeros_(self.belief_logvar_head[-1].bias)
        if self.fm_signed:
            # Official JiT FinalLayer: linear head starts at 0.
            nn.init.zeros_(self.pix_head[-1].weight)
            nn.init.zeros_(self.pix_head[-1].bias)

    def _freeze_lm(self) -> None:
        lm = getattr(self, "lm", None)
        if lm is None:
            return
        lm.eval()
        for p in lm.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_lm()
        return self

    def language_meta(self) -> Dict:
        return {
            "lm_id": self.lm_id,
            "lm_revision": self.lm_revision,
            "d_llm": int(self.d_llm),
            "lm_note": self.lm_note,
        }

    def non_lm_state_dict(self) -> Dict:
        return {k: v for k, v in self.state_dict().items() if not k.startswith("lm.")}

    def load_visual_champion(self, path, skip_language_interface: bool = True) -> Dict:
        from fine_grain.pythia_bridge import load_visual_champion
        return load_visual_champion(
            self, path, skip_language_interface=skip_language_interface,
        )

    def set_optimization_phase(self, phase: str):
        from fine_grain.pythia_bridge import set_optimization_phase
        return set_optimization_phase(self, phase)

    def decode_gauss(self, X: torch.Tensor):
        """Per-point diagonal Gaussian in RGB. X [B,N,d] → μ, lv [B,N,3]."""
        mu = self.pix_head(X)
        lv = torch.clamp(self.pix_logσ(X), -3.0, 2.0)
        return mu, lv

    def decode_field(self, X: torch.Tensor) -> torch.Tensor:
        """Per-point 3-ch mean [B,3,R,R] (velocity or pre-sigmoid RGB)."""
        B = X.shape[0]
        mu, _ = self.decode_gauss(X)
        return mu.transpose(1, 2).reshape(B, 3, self.res, self.res)

    def _pts_to_img(self, pts: torch.Tensor) -> torch.Tensor:
        B = pts.shape[0]
        return pts.transpose(1, 2).reshape(B, 3, self.res, self.res)

    def _img_to_pts(self, img: torch.Tensor) -> torch.Tensor:
        B = img.shape[0]
        return img.reshape(B, 3, -1).transpose(1, 2)

    def _obs_assign(self, tgt: torch.Tensor) -> torch.Tensor:
        """SliceRead the *observation* (any structure: stroke, blob, predator).

        Must not use last_w on the current canvas: if X is already black, that
        assignment is uniform and the 1px in the label never gets a vote.
        """
        t1 = torch.ones(tgt.shape[0], device=tgt.device, dtype=tgt.dtype)
        Xy = self.mot_stack.encode_X(tgt, t=t1)
        _, w = self.mot_stack.layers[-1].read(Xy)
        return w

    def _obs_rarity(self, tgt: torch.Tensor) -> torch.Tensor:
        """How unlike the rest of this image a point is. [B,N,1].

        Untrained SliceRead is nearly uniform, so rarity keeps small outliers
        (1px, distant tiger) loud from step 0 — still no stroke/class mask.
        """
        mu = tgt.mean(dim=(2, 3), keepdim=True)
        r = (tgt - mu).pow(2).mean(dim=1, keepdim=True)
        return r.reshape(tgt.shape[0], 1, -1).transpose(1, 2).clamp_min(1e-8)

    def _observation_pi(self, tgt: torch.Tensor) -> torch.Tensor:
        """Normalized figure measure on the observation. Sum_n π = 1."""
        pi = slice_mass_loss_weights(self._obs_assign(tgt)) * self._obs_rarity(tgt)
        return pi / pi.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def _two_pane(
        self,
        fig: torch.Tensor,
        ground: torch.Tensor,
        pi: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Equal voice for figure and ground; π only splits *inside* each pane."""
        wa = pi
        wb = (1.0 - pi).clamp_min(0.0)
        a = (fig * wa).sum(dim=1) / wa.sum(dim=1).clamp_min(1e-8)
        b = (ground * wb).sum(dim=1) / wb.sum(dim=1).clamp_min(1e-8)
        value = 0.5 * (a + b)
        value = value.reshape(value.shape[0], -1).mean(dim=1)
        if reduction == "none":
            return value
        if reduction != "mean":
            raise ValueError(f"unknown reduction={reduction!r}")
        return value.mean()

    def _structure_panes(self, tgt: torch.Tensor):
        """Figure / local band / far ground from o only. No class mask.

        Two-pane ground is an area mean: filling the bounding box is almost
        free. The band is blur(π)·(1−π) — neighbors of structure, where
        flood actually happens.
        """
        pi = self._observation_pi(tgt)
        b, n, _ = pi.shape
        r = int(round(n ** 0.5))
        p = pi.reshape(b, 1, r, r)
        blur = F.avg_pool2d(p, kernel_size=5, stride=1, padding=2)
        fig = p.reshape(b, n, 1)
        band = (blur * (1.0 - p)).reshape(b, n, 1)
        far = (1.0 - blur).reshape(b, n, 1)
        return fig, band, far

    def _three_pane(
        self,
        err: torch.Tensor,
        fig: torch.Tensor,
        band: torch.Tensor,
        far: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        def _m(val, w):
            return (val * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-8)
        value = (_m(err, fig) + _m(err, band) + _m(err, far)) / 3.0
        value = value.reshape(value.shape[0], -1).mean(dim=1)
        if reduction == "none":
            return value
        if reduction != "mean":
            raise ValueError(f"unknown reduction={reduction!r}")
        return value.mean()

    def _s0_contrast(self, s0: torch.Tensor, xo: torch.Tensor) -> torch.Tensor:
        """S0_i must match encode(o_i), not encode(o_j). Generic, any images."""
        if s0.shape[0] < 2:
            return s0.new_zeros(())
        a = F.normalize(s0.reshape(s0.shape[0], -1), dim=-1)
        b = F.normalize(xo.reshape(xo.shape[0], -1), dim=-1)
        logits = a @ b.T * 10.0
        y = torch.arange(s0.shape[0], device=s0.device)
        return F.cross_entropy(logits, y)

    def _sy_sigreg(self, tgt: torch.Tensor) -> torch.Tensor:
        """LeJEPA SIGReg on encoded observation slices (keeps 1px directions alive)."""
        t1 = torch.ones(tgt.shape[0], device=tgt.device, dtype=tgt.dtype)
        Xc = self.mot_stack.encode_X(tgt, t=t1)
        terms = []
        for layer in self.mot_stack.layers:
            Sy, _ = layer.read(Xc)
            terms.append(compute_sigreg_loss(Sy)["sigreg_total"])
        return torch.stack([t.reshape(()) for t in terms]).mean()

    def _reduce_points(self, val_n: torch.Tensor, idx: torch.Tensor, extra=None, w=None) -> torch.Tensor:
        """Mean over points with SliceRead's measure (each slice one vote)."""
        if w is None:
            pi = val_n.new_full((val_n.shape[0], val_n.shape[1], 1), 1.0 / val_n.shape[1])
        else:
            pi = slice_mass_loss_weights(w)
        if extra is not None:
            pi = pi * extra
        pi = pi / pi.sum(dim=1, keepdim=True).clamp_min(1e-8)
        if val_n.shape[-1] != 1:
            val_n = val_n.mean(dim=-1, keepdim=True)
        return (val_n * pi).sum(dim=1).mean()

    def decode_rgb(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decode_field(X))

    def action_writes(
        self,
        images: torch.Tensor,
        prompts: List[str],
        need_pix=None,
        n_steps: int = 2,
        t=None,
        pi_x=None,
    ) -> Dict:
        """Differentiable action: K writes stay in the graph (pragmatic G).

        Sampling may still use no_grad F-descent. Learning must not.
        """
        n = max(1, int(n_steps))
        if t is None:
            t = torch.ones(images.shape[0], device=images.device, dtype=images.dtype)
        x = images
        out: Dict = {}
        for _ in range(n):
            out = self(x, prompts, pi_x=pi_x, need_pix=need_pix, t=t)
            nxt = out.get("x_pred")
            if nxt is None:
                nxt = out["rgb"]
            x = nxt
        out["x_canvas"] = x
        out["n_action"] = n
        return out

    def _s0_accuracy(
        self,
        rgb_tgt: torch.Tensor,
        idx: torch.Tensor,
        device: torch.device,
    ):
        """S0 matches o: miss / local flood / far bg, plus batch identity.

        Measured failure: two-pane ground hid in-box flood, attention bound
        the color word, S0 became a filled rectangle. Band + contrast force
        structure and which image, still no stroke/class mask.
        """
        terms = []
        o = rgb_tgt.to(device=device)
        fig, band, far = self._structure_panes(o)
        xo = self.mot_stack.encode_X(o, t=None).detach()
        s0_ref = None
        for layer in self.mot_stack.layers:
            s0 = getattr(layer, "last_s0", None)
            if s0 is None:
                continue
            if s0_ref is None:
                s0_ref = s0[idx]
            err_x = (s0[idx] - xo[idx]).pow(2).mean(dim=-1, keepdim=True)
            terms.append(self._three_pane(err_x, fig[idx], band[idx], far[idx]))
            rgb = self.decode_field(s0[idx])
            err_rgb = (rgb - o[idx]).pow(2).mean(dim=1, keepdim=True)
            err_rgb = err_rgb.reshape(err_rgb.shape[0], 1, -1).transpose(1, 2)
            terms.append(self._three_pane(err_rgb, fig[idx], band[idx], far[idx]))
        if not terms:
            return None
        acc = torch.stack([t.reshape(()) for t in terms]).mean()
        if s0_ref is not None and s0_ref.shape[0] >= 2:
            acc = acc + self._s0_contrast(s0_ref, xo[idx])
        return acc

    def forward(
        self,
        images: torch.Tensor,
        prompts: List[str],
        pi_x=None,
        need_pix=None,
        t=None,
        image_precision=None,
        text_precision=None,
        target_time=None,
        history_images=None,
        history_precision=None,
        action=None,
        action_precision=None,
        causal_state=None,
    ) -> Dict:
        # Port readouts do not define the internal dynamics. Perception ports
        # still evolve the latent visual field with language; an explicit pi_x
        # is reserved for evidence clamping / controlled causal ablations.
        if pi_x is None:
            pi_x = 1.0
        out = super().forward(
            images,
            prompts,
            pi_x=pi_x,
            t=t,
            image_precision=image_precision,
            text_precision=text_precision,
            target_time=target_time,
            history_images=history_images,
            history_precision=history_precision,
            action=action,
            action_precision=action_precision,
            causal_state=causal_state,
        )
        return self._fill_visual_outputs(out, images, t)

    def _fill_visual_outputs(
        self,
        out: Dict,
        images: torch.Tensor,
        t=None,
    ) -> Dict:
        """RGB / segmentation / belief from the same terminal X*."""
        X = out.get("X")
        if X is None:
            X = self.mot_stack._last_X
        out["belief_mu"] = X
        out["belief_logvar"] = torch.clamp(
            self.belief_logvar_head(X), -6.0, 3.0,
        )
        mu_q, lv_q = self.decode_gauss(X)
        field = self._pts_to_img(mu_q)
        if self.seg_head is not None:
            seg = self.seg_head(X)
            out["seg_logits"] = seg.transpose(1, 2).reshape(
                X.shape[0], self.seg_classes, self.res, self.res,
            )
        else:
            out["seg_logits"] = None
        out["rgb_lv"] = self._pts_to_img(lv_q)
        Xp = getattr(self.mot_stack, "_last_X_prior", None)
        if Xp is not None:
            mu_p, lv_p = self.decode_gauss(Xp)
            out["rgb_mu_p"] = self._pts_to_img(mu_p)
            out["rgb_lv_p"] = self._pts_to_img(lv_p)
        else:
            out["rgb_mu_p"] = None
            out["rgb_lv_p"] = None
        if t is None:
            out["v"] = None
            out["x_pred"] = None
            out["rgb"] = torch.sigmoid(field)
            return out
        if self.fm_pred == "v":
            out["v"] = field
            tb = t.reshape(-1, 1, 1, 1).to(dtype=field.dtype)
            out["x_pred"] = (images + (1.0 - tb) * field).clamp(0.0, 1.0)
            out["rgb"] = out["x_pred"]
        elif self.fm_signed:
            # JiT chart: x ∈ [-1,1], linear head, no sigmoid.
            out["x_pred"] = field
            out["rgb"] = (field + 1.0) * 0.5
            out["v"] = v_from_x_pred(field, images, t)
        else:
            # Unit-interval x-pred (legacy).
            x_hat = torch.sigmoid(field)
            out["x_pred"] = x_hat
            out["rgb"] = x_hat
            out["v"] = v_from_x_pred(x_hat, images, t)
        return out

    def forward_tokens(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        visual_prompt_mask: Optional[torch.Tensor] = None,
        pi_x=None,
        t=None,
        image_precision=None,
        text_precision=None,
        target_time=None,
        history_images=None,
        history_precision=None,
        action=None,
        action_precision=None,
        causal_state=None,
        n_loops=None,
        need_pix=None,
        score_tokens: bool = True,
    ) -> Dict:
        """Joint Slice–MoT step with frozen causal token NLL on H*.

        Language evidence is the frozen LM input embedding. Visual queries
        see only prompt tokens. The frozen decoder scores
        [terminal Slice tokens; H*_llm] and must not be called via generate().
        """
        if self.lm is None:
            raise RuntimeError(
                "forward_tokens requires a frozen causal LM; "
                "construct DualStreamOmni(language='pythia') or pass lm="
            )
        self._freeze_lm()
        if pi_x is None:
            pi_x = 1.0
        text_mask = attention_mask.bool()
        if labels is None:
            # Inference prefix: every valid token is observed. Do not compute NLL
            # (that would let visual queries train on tokens they may also read).
            text_lab = None
            inferred_prompt_mask = text_mask
        else:
            text_lab = labels
            inferred_prompt_mask = text_lab.eq(-100) & text_mask
        if visual_prompt_mask is None:
            prompt_mask = inferred_prompt_mask
        else:
            if visual_prompt_mask.shape != attention_mask.shape:
                raise ValueError(
                    "visual_prompt_mask must have the same [B,T] shape as "
                    "attention_mask"
                )
            prompt_mask = visual_prompt_mask.to(
                device=attention_mask.device, dtype=torch.bool,
            ) & text_mask
            if text_lab is not None and bool(
                (prompt_mask & text_lab.ne(-100)).any().item()
            ):
                raise ValueError(
                    "supervised answer tokens cannot be visible to visual queries"
                )
        emb = self.lm.get_input_embeddings()(input_ids)
        X, H_out, tok, traces = self.mot_stack.forward_native(
            img=images,
            text_emb=emb.float(),
            text_mask=text_mask,
            prompt_mask=prompt_mask,
            pi_x=pi_x,
            n_loops=n_loops,
            t=t,
            image_precision=image_precision,
            text_precision=text_precision,
            target_time=target_time,
            history_images=history_images,
            history_precision=history_precision,
            action=action,
            action_precision=action_precision,
            causal_state=causal_state,
        )
        lm_dtype = emb.dtype
        vis = tok.to(dtype=lm_dtype)
        text = H_out.to(dtype=lm_dtype)
        inputs = torch.cat([vis, text], dim=1)
        batch, n_vis = vis.shape[0], vis.shape[1]
        vis_m = torch.ones(
            batch, n_vis, device=images.device, dtype=attention_mask.dtype,
        )
        attn = torch.cat([vis_m, attention_mask], dim=1)
        if score_tokens:
            lm_out = self.lm(
                inputs_embeds=inputs,
                attention_mask=attn,
                use_cache=False,
            )
            token_logits = lm_out.logits
        else:
            token_logits = None
        if text_lab is None or not score_tokens:
            token_nll = None
            n_loss_tokens = 0
        else:
            ignore = torch.full(
                (batch, n_vis), -100, device=images.device, dtype=input_ids.dtype,
            )
            lm_labels = torch.cat([ignore, text_lab], dim=1)
            token_nll = token_nll_per_sample(token_logits, lm_labels)
            n_loss_tokens = int(text_lab.ne(-100).sum().item())
        mask_f = text_mask.unsqueeze(-1).to(dtype=H_out.dtype)
        pooled = (H_out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        zero = images.new_zeros(())
        pred_loss = getattr(self.mot_stack, "_last_pred_loss", None)
        if pred_loss is None:
            pred_loss = zero
        vfe_loss = getattr(self.mot_stack, "_last_vfe_train_loss", None)
        if vfe_loss is None:
            vfe_loss = zero
        sigreg_loss = getattr(self.mot_stack, "_last_sigreg_loss", None)
        if sigreg_loss is None:
            sigreg_loss = zero
        out = {
            "logits": self.head(pooled),
            "logits_k": [],
            "traces": traces,
            "X": X,
            "H": H_out,
            "tokens": tok,
            "pred_loss": pred_loss,
            "vfe_loss": vfe_loss,
            "sigreg_loss": sigreg_loss,
            "token_logits": token_logits,
            "token_nll": token_nll,
            "n_loss_tokens": n_loss_tokens,
            "n_vis_tokens": n_vis,
            "prompt_mask": prompt_mask,
            "text_mask": text_mask,
            "rho": getattr(self.mot_stack, "_last_rho", {"rho_x": 0.0, "rho_h": 0.0}),
            "looks": getattr(self.mot_stack, "_last_looks", None),
            "s2a_g_logits": [],
            "s2a_ig": [],
            "causal_state": getattr(self.mot_stack, "_last_causal_state", None),
            "causal_prior_mu": getattr(self.mot_stack, "_last_causal_prior_mu", None),
            "causal_prior_logvar": getattr(
                self.mot_stack, "_last_causal_prior_logvar", None,
            ),
            "causal_posterior_mu": getattr(
                self.mot_stack, "_last_causal_posterior_mu", None,
            ),
            "causal_posterior_logvar": getattr(
                self.mot_stack, "_last_causal_posterior_logvar", None,
            ),
            "causal_diagnostics": getattr(
                self.mot_stack, "_last_causal_diagnostics", {},
            ),
        }
        return self._fill_visual_outputs(out, images, t)

    def forward_with_future_posterior(
        self,
        images: torch.Tensor,
        prompts: List[str],
        future_images: torch.Tensor,
        **kwargs,
    ) -> Dict:
        """Predict a future prior and infer q with the exact same graph/chart.

        The observed-future pass runs first; the predictive pass runs second so
        layer caches used by the ordinary losses still describe the prediction.
        """
        batch = images.shape[0]
        history = kwargs.get("history_images")
        q_history = history
        if history is not None:
            if history.ndim != 5:
                raise ValueError("history_images must have shape [B,T,3,R,R]")
            q_history = torch.cat(
                [history[:, 1:], future_images.unsqueeze(1)], dim=1,
            )
        q_kwargs = dict(kwargs)
        q_kwargs["image_precision"] = torch.ones(
            batch, device=images.device, dtype=images.dtype,
        )
        # q(s_{t+1}|o_{t+1}) is a perceptual posterior, not a second rollout.
        # The action and horizon have already produced o_{t+1}; applying them
        # again here would advance the posterior twice.  The predictive p pass
        # below still receives the original action/horizon boundary.
        q_kwargs["text_precision"] = torch.zeros(
            batch, device=images.device, dtype=images.dtype,
        )
        q_kwargs["action_precision"] = torch.zeros(
            batch, device=images.device, dtype=images.dtype,
        )
        q_kwargs["target_time"] = torch.zeros(
            batch, device=images.device, dtype=images.dtype,
        )
        # The observed-future branch constructs its own posterior from the
        # shifted observed history. Reusing a caller's predictive state here
        # would make q depend on p and weaken the variational target.
        q_kwargs["causal_state"] = None
        q_kwargs["history_images"] = q_history
        q_out = self.forward(future_images, [""] * batch, **q_kwargs)
        q_mu = q_out["belief_mu"]
        q_lv = q_out["belief_logvar"]
        q_causal_mu = q_out.get("causal_posterior_mu")
        q_causal_lv = q_out.get("causal_posterior_logvar")
        out = self.forward(images, prompts, **kwargs)
        out["transition_q_mu"] = q_mu
        out["transition_q_logvar"] = q_lv
        out["transition_q_rgb"] = q_out["rgb"]
        out["transition_q_seg_logits"] = q_out.get("seg_logits")
        out["transition_q_causal_mu"] = q_causal_mu
        out["transition_q_causal_logvar"] = q_causal_lv
        return out

    @staticmethod
    def _precision_mean(value: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
        """Per-sample likelihood weighting; pi=0 removes the evidence exactly."""
        v = value.reshape(value.shape[0], -1).mean(dim=1)
        w = precision.to(device=v.device, dtype=v.dtype).reshape(-1).clamp_min(0.0)
        # Do not renormalize by sum(pi): absolute precision must scale the
        # accuracy term in free energy, not merely select examples.
        return (v * w).mean()

    def omni_loss(self, out: Dict, batch: Dict, device: torch.device) -> Tuple[torch.Tensor, Dict]:
        need_text = batch["need_text"]
        need_pix = batch["need_pix"]
        need_seg = batch.get("need_seg", [False] * len(need_text))
        batch_size = len(need_text)
        text_like_pi = torch.as_tensor(
            batch.get("target_text_precision", torch.ones(batch_size)),
            device=device, dtype=out["logits"].dtype,
        )
        image_like_pi = torch.as_tensor(
            batch.get("target_image_precision", torch.ones(batch_size)),
            device=device, dtype=out["logits"].dtype,
        )
        seg_like_pi = torch.as_tensor(
            batch.get("target_seg_precision", torch.ones(batch_size)),
            device=device, dtype=out["logits"].dtype,
        )
        if "answer" in batch:
            targets = torch.tensor(
                [self.ans_to_idx.get(a, 0) for a in batch["answer"]], device=device,
            )
        else:
            targets = torch.zeros(batch_size, device=device, dtype=torch.long)
        rgb_tgt = batch.get("target_rgb")
        if rgb_tgt is not None:
            rgb_tgt = rgb_tgt.to(device)
        stroke = batch.get("stroke")
        if stroke is not None:
            stroke = stroke.to(device)
        losses = []
        meta = {
            "ce": 0.0, "pix": 0.0, "seg": 0.0, "transition_kl": 0.0,
            "transition_q_obs": 0.0,
            "transition_q_sigreg": 0.0,
            "causal_transition_kl": 0.0,
            "n_text": 0, "n_pix": 0, "n_seg": 0, "n_transition": 0,
        }
        if any(need_text):
            idx = torch.tensor([i for i, t in enumerate(need_text) if t], device=device)
            if out.get("token_nll") is not None:
                nll = out["token_nll"].to(device=device, dtype=out["logits"].dtype)
                if nll.ndim == 0:
                    nll = nll.expand(batch_size)
                ce = self._precision_mean(nll[idx], text_like_pi[idx])
                meta["token_nll"] = float(ce.item())
            else:
                ce_each = F.cross_entropy(
                    out["logits"][idx], targets[idx], reduction="none",
                )
                ce = self._precision_mean(ce_each, text_like_pi[idx])
            losses.append(ce)
            meta["ce"] = float(ce.item())
            meta["n_text"] = int(idx.numel())
        if any(need_pix):
            idx = torch.tensor([i for i, t in enumerate(need_pix) if t], device=device)
            tgt = rgb_tgt[idx]
            t = batch.get("t")
            if self.pixel_loss_mode == "gaussian_nll":
                pred = out.get("x_pred")
                if pred is None or self.fm_pred == "v":
                    pred = out["rgb"]
                pred = pred[idx]
                lv = out.get("rgb_lv")
                if lv is None:
                    lv = torch.zeros_like(pred)
                else:
                    lv = lv[idx].clamp(-6.0, 3.0)
                nll = 0.5 * (lv + (tgt - pred).pow(2) * torch.exp(-lv))
                pix_each = nll.flatten(1).mean(dim=1)
                pix = self._precision_mean(pix_each, image_like_pi[idx])
                meta["gaussian_nll"] = float(pix.detach())
                meta["n_obs"] = 1
                out["point_vfe"] = None
                out["point_vfe_terms"] = None
            elif out.get("x_pred") is not None and self.fm_pred != "v":
                xh = out["x_pred"][idx]
                if self.pixel_loss_mode == "balanced_bce":
                    pix_each = balanced_observation_bce(
                        xh, tgt, signed=self.fm_signed, reduction="none",
                    )
                    pix = self._precision_mean(pix_each, image_like_pi[idx])
                    meta["balanced_bce"] = float(pix.detach())
                    meta["n_obs"] = 1
                    out["point_vfe"] = None
                    out["point_vfe_terms"] = None
                else:
                    mu_q = self._img_to_pts(xh)
                    y = self._img_to_pts(tgt)
                    lv = out.get("rgb_lv")
                    if lv is not None:
                        lv_q = self._img_to_pts(lv[idx])
                    else:
                        lv_q = torch.zeros_like(mu_q)
                    mu_p_img = out.get("rgb_mu_p")
                    if mu_p_img is not None:
                        mu_pr = self._img_to_pts(mu_p_img[idx])
                        lv_pr = self._img_to_pts(out["rgb_lv_p"][idx])
                    else:
                        mu_pr = mu_q.detach()
                        lv_pr = lv_q.detach()
                    vfe = compute_point_vfe(mu_pr, lv_pr, mu_q, lv_q, y, sigma_r=1.0)
                    # One o = the target image (stroke + black). Split that same
                    # F into figure/ground panes so 1px is not drowned and flood
                    # is not free. Not a second all-black observation.
                    pi = self._observation_pi(tgt)
                    pix_each = self._two_pane(
                        vfe["F"], vfe["F"], pi, reduction="none",
                    )
                    if t is not None:
                        tw = 1.0 / (1.0 - t.to(device)[idx]).clamp(min=0.05)
                        pix_each = pix_each * tw
                    pix = self._precision_mean(pix_each, image_like_pi[idx])
                    meta["point_F"] = float(pix.detach())
                    meta["point_gap"] = float(vfe["gap"].mean().item())
                    meta["point_U"] = float(vfe["U"].mean().item())
                    meta["see_acc"] = float(
                        (vfe["acc_mean"] + vfe["acc_tr"]).mean().detach()
                    )
                    meta["n_obs"] = 1
                    out["point_vfe"] = pix
                    out["point_vfe_terms"] = vfe
            elif out.get("v") is not None and batch.get("v_tgt") is not None:
                v_hat = out["v"][idx]
                v_tgt = batch["v_tgt"].to(device)[idx]
                pix_each = (v_hat - v_tgt).pow(2).flatten(1).mean(dim=1)
                pix = self._precision_mean(pix_each, image_like_pi[idx])
                meta["n_obs"] = 1
            else:
                pred = out["rgb"][idx]
                pix_each = (pred - tgt).pow(2).flatten(1).mean(dim=1)
                pix = self._precision_mean(pix_each, image_like_pi[idx])
                meta["n_obs"] = 1
            losses.append(pix)
            meta["pix"] = float(pix.item())
            meta["n_pix"] = int(idx.numel())
            s0 = self._s0_accuracy(rgb_tgt, idx, device)
            if s0 is not None and float(getattr(self, "s0_acc_coef", 0.0)) != 0.0:
                s0_weight = image_like_pi[idx].mean()
                losses.append(self.s0_acc_coef * s0_weight * s0)
                meta["s0_acc"] = float(s0.detach())
            if float(getattr(self, "sigreg_coef", 0.0)) != 0.0:
                sy = self._sy_sigreg(tgt)
                losses.append(self.sigreg_coef * sy)
                meta["sigreg_sy"] = float(sy.detach())
        if any(need_seg):
            if out.get("seg_logits") is None:
                raise RuntimeError("batch requests segmentation but seg_classes=0")
            idx = torch.tensor([i for i, flag in enumerate(need_seg) if flag], device=device)
            seg_tgt = batch["target_seg"].to(device=device, dtype=torch.long)
            seg_map = F.cross_entropy(
                out["seg_logits"][idx], seg_tgt[idx], reduction="none",
            )
            fg = seg_tgt[idx] > 0
            bg = ~fg

            def _seg_pane(mask: torch.Tensor) -> torch.Tensor:
                weight = mask.to(seg_map.dtype)
                return (seg_map * weight).sum(dim=(1, 2)) / weight.sum(
                    dim=(1, 2)
                ).clamp_min(1.0)

            seg_each = 0.5 * _seg_pane(fg) + 0.5 * _seg_pane(bg)
            seg = self._precision_mean(seg_each, seg_like_pi[idx])
            losses.append(self.seg_loss_coef * seg)
            meta["seg"] = float(seg.detach())
            meta["n_seg"] = int(idx.numel())
        if self.transition_loss_coef != 0.0 and out.get("X") is not None:
            horizon = torch.as_tensor(
                batch.get("target_time", 0.0), device=device, dtype=out["X"].dtype,
            ).reshape(-1)
            idx = torch.nonzero(horizon > 0.0, as_tuple=False).flatten()
            if idx.numel() > 0:
                if out.get("transition_q_mu") is None:
                    raise RuntimeError(
                        "transition KL requires forward_with_future_posterior(); "
                        "a raw sensory stem is not a closed posterior chart"
                    )
                # Full conditional VFE term KL(q_{t+1} || p_{t+1}) with both
                # distributions inferred by the same graph and diagonal,
                # state-dependent covariance.
                mu_p = out["belief_mu"][idx]
                lv_p = out["belief_logvar"][idx]
                mu_q = out["transition_q_mu"][idx]
                lv_q = out["transition_q_logvar"][idx]
                if self.transition_detach_q:
                    # The observed branch is anchored by its own accuracy term
                    # below; stop-gradient makes it a stable variational target
                    # for the dynamic prior instead of letting q chase p.
                    mu_q = mu_q.detach()
                    lv_q = lv_q.detach()
                point_kl = 0.5 * (
                    lv_p - lv_q
                    + torch.exp(lv_q - lv_p)
                    + (mu_q - mu_p).pow(2) * torch.exp(-lv_p)
                    - 1.0
                ).mean(dim=-1, keepdim=True)
                tgt = rgb_tgt[idx]
                fig, band, far = self._structure_panes(tgt)
                transition_each = self._three_pane(
                    point_kl, fig, band, far, reduction="none",
                )
                transition_kl = self._precision_mean(
                    transition_each, image_like_pi[idx],
                )
                losses.append(self.transition_loss_coef * transition_kl)
                meta["transition_kl"] = float(transition_kl.detach())
                meta["transition_coef"] = self.transition_loss_coef
                meta["n_transition"] = int(idx.numel())
                if self.transition_posterior_loss_coef != 0.0:
                    q_rgb = out.get("transition_q_rgb")
                    if q_rgb is None:
                        raise RuntimeError(
                            "transition posterior accuracy requires the observed "
                            "future pass"
                        )
                    q_obs_each = balanced_observation_bce(
                        q_rgb[idx], tgt, signed=False, reduction="none",
                    )
                    q_obs = self._precision_mean(
                        q_obs_each, image_like_pi[idx],
                    )
                    losses.append(self.transition_posterior_loss_coef * q_obs)
                    meta["transition_q_obs"] = float(q_obs.detach())
                    meta["transition_q_obs_coef"] = (
                        self.transition_posterior_loss_coef
                    )
                if self.causal_memory_loss_coef != 0.0:
                    p_causal_mu = out.get("causal_prior_mu")
                    p_causal_lv = out.get("causal_prior_logvar")
                    q_causal_mu = out.get("transition_q_causal_mu")
                    q_causal_lv = out.get("transition_q_causal_logvar")
                    if any(x is None for x in (
                        p_causal_mu, p_causal_lv, q_causal_mu, q_causal_lv,
                    )):
                        raise RuntimeError(
                            "causal memory KL requires use_active_gdn2 and the "
                            "observed future posterior pass"
                        )
                    pc_mu = p_causal_mu[idx]
                    pc_lv = p_causal_lv[idx]
                    qc_mu = q_causal_mu[idx]
                    qc_lv = q_causal_lv[idx]
                    if self.transition_detach_q:
                        qc_mu = qc_mu.detach()
                        qc_lv = qc_lv.detach()
                    causal_kl_each = ActiveInferenceGDN2.gaussian_kl(
                        qc_mu, qc_lv, pc_mu, pc_lv,
                    ).mean(dim=(1, 2))
                    causal_kl = self._precision_mean(
                        causal_kl_each, image_like_pi[idx],
                    )
                    losses.append(self.causal_memory_loss_coef * causal_kl)
                    meta["causal_transition_kl"] = float(causal_kl.detach())
                    meta["causal_memory_coef"] = self.causal_memory_loss_coef
        if not losses:
            return out["logits"].sum() * 0.0, meta
        loss = torch.stack([x.reshape(()) for x in losses]).sum()
        pred = out.get("pred_loss")
        pw = float(getattr(self.mot_stack, "prior_write", 0.0))
        # Noisy ||S_t − μp||² fights the bridge: μp must be the clean
        # terminal, not the interpolant. Skip it when W^H is on.
        if pred is not None and self.prior_loss_coef != 0.0 and pw == 0.0:
            loss = loss + self.prior_loss_coef * pred
        vfe = out.get("vfe_loss")
        # Point F already is the landing of free energy when we decoded
        # p(rgb|X). Slice gap stays workspace-only in that case.
        if out.get("point_vfe") is None and vfe is not None and self.vfe_coef != 0.0:
            loss = loss + self.vfe_coef * vfe
        if pw != 0.0 and any(need_pix):
            idx = torch.tensor([i for i, t in enumerate(need_pix) if t], device=device)
            cl = self._prior_clean_loss(
                rgb_tgt,
                idx,
                device,
                t=batch.get("t"),
                target_time=batch.get("target_time"),
                history_images=batch.get("history_images"),
                history_precision=batch.get("history_precision"),
                action=batch.get("action"),
                action_precision=batch.get("action_precision"),
            )
            if cl is not None and self.prior_loss_coef != 0.0:
                cl_weight = image_like_pi[idx].mean()
                loss = loss + self.prior_loss_coef * cl_weight * cl
                meta["prior_clean"] = float(cl.item())
                meta["prior_clean_coef"] = self.prior_loss_coef
        return loss, meta

    def _prior_clean_loss(
        self,
        rgb_tgt: torch.Tensor,
        idx: torch.Tensor,
        device: torch.device,
        t=None,
        target_time=None,
        history_images=None,
        history_precision=None,
        action=None,
        action_precision=None,
    ):
        """μp(H) must predict slices of the *clean* image, not the noisy field."""
        terms = []
        t_ref = None if t is None else t.to(device=device)
        encode_kw = {"t": t_ref}
        if getattr(self.mot_stack, "use_modal_precision", False):
            encode_kw["image_precision"] = 1.0
        if getattr(self.mot_stack, "use_target_time", False):
            encode_kw["target_time"] = target_time
        if getattr(self.mot_stack, "history_size", 0) > 0:
            encode_kw["history_images"] = history_images
            encode_kw["history_precision"] = history_precision
        if getattr(self.mot_stack, "action_dim", 0) > 0:
            encode_kw["action"] = action
            encode_kw["action_precision"] = action_precision
        Xc = self.mot_stack.encode_X(rgb_tgt.to(device), **encode_kw)
        for layer in self.mot_stack.layers:
            mu = getattr(layer, "last_mu_p", None)
            if mu is None:
                continue
            Sc, _ = layer.read(Xc)
            terms.append((mu[idx] - Sc[idx].detach()).pow(2).mean())
        if not terms:
            return None
        return torch.stack(terms).mean()
