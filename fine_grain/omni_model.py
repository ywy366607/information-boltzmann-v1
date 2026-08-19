"""One DualStream field loop, five I/O ports: T2T / IT2T / recon / T2I / I2I."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fine_grain.bayesian_surprise import compute_point_vfe
from fine_grain.flow_match import v_from_x_pred
from fine_grain.native_mot import need_pix_to_pi_x, slice_mass_loss_weights
from fine_grain.sigreg import compute_sigreg_loss
from fine_grain.unified_arch import unified_kwargs
from fine_grain.vlm_data import COLORS, KINK_KS, OCR_DIGITS
from scripts.run_v0_surprise_eval import DualStreamVQAModel


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


class DualStreamOmni(DualStreamVQAModel):
    """Champion-B DualStream + per-point RGB head on the live field X."""

    EXTRA_WORDS = (
        "Draw", "Reconstruct", "Restore", "Paint", "image", "blank",
        "noisy", "this", "to", "a", "has", "polyline",
    )

    @classmethod
    def unified(cls, **overrides):
        """Single residual-write graph. Overrides are size / data only."""
        return cls(**unified_kwargs(**overrides))

    def __init__(self, **kwargs):
        self.fm_pred = str(kwargs.pop("fm_pred", "x")).lower()
        self.fm_signed = bool(kwargs.pop("fm_signed", False))
        kwargs.setdefault("surprise_mode", "v1_bayes")
        kwargs.setdefault("s_update", "rms_dir")
        super().__init__(**kwargs)
        # grow vocab for generation prompts (old rows stay aligned)
        extra = [w for w in self.EXTRA_WORDS if w not in self.vocab]
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
        if self.fm_signed:
            # Official JiT FinalLayer: linear head starts at 0.
            nn.init.zeros_(self.pix_head[-1].weight)
            nn.init.zeros_(self.pix_head[-1].bias)

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

    def _two_pane(self, fig: torch.Tensor, ground: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        """Equal voice for figure and ground; π only splits *inside* each pane."""
        wa = pi
        wb = (1.0 - pi).clamp_min(0.0)
        a = (fig * wa).sum(dim=1) / wa.sum(dim=1).clamp_min(1e-8)
        b = (ground * wb).sum(dim=1) / wb.sum(dim=1).clamp_min(1e-8)
        return (0.5 * (a + b)).mean()

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

    def forward(
        self,
        images: torch.Tensor,
        prompts: List[str],
        pi_x=None,
        need_pix=None,
        t=None,
    ) -> Dict:
        if pi_x is None and need_pix is not None:
            pi_x = need_pix_to_pi_x(need_pix, images)
        out = super().forward(images, prompts, pi_x=pi_x, t=t)
        X = out.get("X")
        if X is None:
            X = self.mot_stack._last_X
        mu_q, lv_q = self.decode_gauss(X)
        field = self._pts_to_img(mu_q)
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

    def omni_loss(self, out: Dict, batch: Dict, device: torch.device) -> Tuple[torch.Tensor, Dict]:
        need_text = batch["need_text"]
        need_pix = batch["need_pix"]
        targets = torch.tensor(
            [self.ans_to_idx.get(a, 0) for a in batch["answer"]], device=device,
        )
        rgb_tgt = batch["target_rgb"].to(device)
        stroke = batch["stroke"].to(device)
        losses = []
        meta = {"ce": 0.0, "pix": 0.0, "n_text": 0, "n_pix": 0}
        if any(need_text):
            idx = torch.tensor([i for i, t in enumerate(need_text) if t], device=device)
            ce = F.cross_entropy(out["logits"][idx], targets[idx])
            losses.append(ce)
            meta["ce"] = float(ce.item())
            meta["n_text"] = int(idx.numel())
        if any(need_pix):
            idx = torch.tensor([i for i, t in enumerate(need_pix) if t], device=device)
            if out.get("x_pred") is not None and self.fm_pred != "v":
                xh = out["x_pred"][idx]
                tgt = rgb_tgt[idx]
                ew = rgb_energy_weights(tgt, signed=self.fm_signed)
                t = batch.get("t")
                lv = out.get("rgb_lv")
                mu_p_img = out.get("rgb_mu_p")
                if lv is not None and mu_p_img is not None:
                    mu_q = self._img_to_pts(xh)
                    lv_q = self._img_to_pts(lv[idx])
                    mu_pr = self._img_to_pts(mu_p_img[idx])
                    lv_pr = self._img_to_pts(out["rgb_lv_p"][idx])
                    y = self._img_to_pts(tgt)
                    bg = torch.full_like(y, -1.0 if self.fm_signed else 0.0)
                    vfe_y = compute_point_vfe(mu_pr, lv_pr, mu_q, lv_q, y, sigma_r=1.0)
                    vfe_bg = compute_point_vfe(mu_pr, lv_pr, mu_q, lv_q, bg, sigma_r=1.0)
                    acc_y = vfe_y["acc_mean"] + vfe_y["acc_tr"]
                    acc_bg = vfe_bg["acc_mean"] + vfe_bg["acc_tr"]
                    pi = self._observation_pi(tgt)
                    see = self._two_pane(acc_y, acc_bg, pi)
                    pix = vfe_y["U"].mean() + see
                    if t is not None:
                        tw = 1.0 / (1.0 - t.to(device)[idx]).clamp(min=0.05)
                        pix = pix * tw.mean()
                    meta["point_F"] = float((vfe_y["U"].mean() + see).detach())
                    meta["point_gap"] = float(vfe_y["gap"].mean().item())
                    meta["point_U"] = float(vfe_y["U"].mean().item())
                    meta["see_acc"] = float(see.detach())
                    out["point_vfe"] = pix
                else:
                    err_y = (xh - tgt).pow(2).mean(dim=1, keepdim=True)
                    bg_img = torch.full_like(tgt, -1.0 if self.fm_signed else 0.0)
                    err_bg = (xh - bg_img).pow(2).mean(dim=1, keepdim=True)
                    err_y = err_y.reshape(err_y.shape[0], 1, -1).transpose(1, 2)
                    err_bg = err_bg.reshape(err_bg.shape[0], 1, -1).transpose(1, 2)
                    pix = self._two_pane(err_y, err_bg, self._observation_pi(tgt))
            elif out.get("v") is not None and batch.get("v_tgt") is not None:
                v_hat = out["v"][idx]
                v_tgt = batch["v_tgt"].to(device)[idx]
                lv = (v_hat - v_tgt).abs().mean()
                x1h = out["rgb"][idx]
                tgt = rgb_tgt[idx]
                lx = (x1h - tgt).abs().mean()
                m = stroke[idx].unsqueeze(1)
                ls = ((x1h - tgt).abs() * m).sum() / m.sum().clamp_min(1.0) if m.sum() > 0 else x1h.new_zeros(())
                bg = 1.0 - m
                v_bg = (v_hat.abs() * bg).sum() / bg.sum().clamp_min(1.0) if bg.sum() > 0 else v_hat.new_zeros(())
                pix = lv + 0.25 * lx + ls + v_bg
            else:
                pred = out["rgb"][idx]
                tgt = rgb_tgt[idx]
                pix = (pred - tgt).abs().mean()
                m = stroke[idx].unsqueeze(1)
                if m.sum() > 0:
                    pix = pix + 4.0 * ((pred - tgt).abs() * m).sum() / m.sum().clamp_min(1.0)
            losses.append(pix)
            meta["pix"] = float(pix.item())
            meta["n_pix"] = int(idx.numel())
            if float(getattr(self, "sigreg_coef", 0.0)) != 0.0:
                sy = self._sy_sigreg(tgt if out.get("x_pred") is not None else rgb_tgt[idx])
                losses.append(self.sigreg_coef * sy)
                meta["sigreg_sy"] = float(sy.detach())
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
            cl = self._prior_clean_loss(rgb_tgt, idx, device)
            if cl is not None:
                loss = loss + cl
                meta["prior_clean"] = float(cl.item())
        return loss, meta

    def _prior_clean_loss(
        self, rgb_tgt: torch.Tensor, idx: torch.Tensor, device: torch.device,
    ):
        """μp(H) must predict slices of the *clean* image, not the noisy field."""
        terms = []
        t1 = torch.ones(rgb_tgt.shape[0], device=device)
        Xc = self.mot_stack.encode_X(rgb_tgt.to(device), t=t1)
        for layer in self.mot_stack.layers:
            mu = getattr(layer, "last_mu_p", None)
            if mu is None:
                continue
            Sc, _ = layer.read(Xc)
            terms.append((mu[idx] - Sc[idx].detach()).pow(2).mean())
        if not terms:
            return None
        return torch.stack(terms).mean()
