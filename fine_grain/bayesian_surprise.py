"""Bayesian Surprise & JEPA Proxy Gate for Dual-Stream Vision-Language Co-evolution.

Hierarchy:
  - V0 (Deterministic JEPA Proxy):
      Language predicts slice expectation S_hat = P(H).
      Surprise reduces to prediction error U = (1/d) ||S - S_hat||^2.
      Gate g(U) = 1 - exp(-beta * U_detach).
  - V1 (Gaussian Bayesian Surprise):
      Language outputs prior p(z) = N(mu_p, diag(sigma_p^2)) from H.
      Visual observation outputs posterior q(z) = N(mu_q, diag(sigma_q^2)) from (S, H).
      Surprise is analytical KL: U = D_KL(q || p).
  - Control & Ablation Modes:
      - 'baseline': g(U) = 1.0 (standard un-gated dual-stream update).
      - 'random': g(U) ~ Uniform(0, 1).
      - 'constant': g(U) = c (default 0.5).
      - 'v0_shuffled': V0 JEPA error with slice indices randomly permuted.
      - 'v0_reverse': g_rev(U) = 1.0 - g(U) = exp(-beta * U_detach).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fine_grain.sigreg import compute_sigreg_loss


class _RMSNorm(nn.Module):
    """Local RMSNorm — native_mot imports this file, so do not import back."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(var + self.eps)


def compute_slice_vfe(
    mu_p: torch.Tensor,
    lv_p: torch.Tensor,
    mu_q: torch.Tensor,
    lv_q: torch.Tensor,
    S: torch.Tensor,
    sigma_r: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Diagonal-Gaussian slice VFE and free-energy gap.

    Generative model: p(z)=N(μp,σp²), p(S|z)=N(z,σr²), q=N(μq,σq²).

      F = KL(q‖p) + (‖S-μq‖² + tr Σq) / (2 σr²) + ½ log σr²
      gap = F − F_min = KL(q ‖ p(z|S))     with Kalman q*

    All inputs [..., d]. Energy terms are mean-reduced over the last dim to
    [..., 1], matching the existing per-slice U convention.
    σr is a fixed scalar (the ½ log σr² piece is then a constant).
    """
    lv_p = torch.clamp(lv_p, -5.0, 2.0)
    lv_q = torch.clamp(lv_q, -5.0, 2.0)
    var_p = torch.exp(lv_p)
    var_q = torch.exp(lv_q)
    sr2 = float(sigma_r) ** 2 + 1e-12

    u_mu = ((mu_q - mu_p) ** 2) / (2.0 * var_p + 1e-7)
    u_sig = 0.5 * ((lv_p - lv_q) + var_q / (var_p + 1e-7) - 1.0)
    U = u_mu.mean(dim=-1, keepdim=True).clamp(min=0.0) + u_sig.mean(dim=-1, keepdim=True).clamp(min=0.0)

    acc_mean = (0.5 / sr2) * (S - mu_q).pow(2).mean(dim=-1, keepdim=True)
    acc_tr = (0.5 / sr2) * var_q.mean(dim=-1, keepdim=True)
    acc_const = 0.5 * math.log(sr2)
    F = U + acc_mean + acc_tr + acc_const

    prec_p = 1.0 / (var_p + 1e-7)
    prec_r = 1.0 / sr2
    var_star = 1.0 / (prec_p + prec_r)
    mu_star = var_star * (prec_r * S + prec_p * mu_p)
    lv_star = torch.log(var_star.clamp(min=1e-12))

    g_mu = ((mu_q - mu_star) ** 2) / (2.0 * var_star + 1e-7)
    g_sig = 0.5 * ((lv_star - lv_q) + var_q / (var_star + 1e-7) - 1.0)
    gap = g_mu.mean(dim=-1, keepdim=True).clamp(min=0.0) + g_sig.mean(dim=-1, keepdim=True).clamp(min=0.0)

    F_min = gaussian_prior_predictive_nll(
        mu_p, lv_p, S, sigma_r=sigma_r,
    )
    # Observation Kalman gain: S' = μp + K (S − μp), K = σp² / (σp² + σr²).
    K = var_p / (var_p + sr2)

    return {
        "U": U,
        "u_mu": u_mu.mean(dim=-1, keepdim=True).clamp(min=0.0),
        "u_sigma": u_sig.mean(dim=-1, keepdim=True).clamp(min=0.0),
        "acc_mean": acc_mean,
        "acc_tr": acc_tr,
        "F": F,
        "gap": gap,
        "F_min": F_min,
        "mu_star": mu_star,
        "lv_star": lv_star,
        "K": K,
        "s_err": (S - mu_q).pow(2).mean(dim=-1, keepdim=True),
    }


def gaussian_prior_predictive_nll(
    mu_p: torch.Tensor,
    lv_p: torch.Tensor,
    observation: torch.Tensor,
    sigma_r: float = 1.0,
) -> torch.Tensor:
    """Negative log p(observation | prior) for the diagonal Gaussian model.

    This is the minimum VFE after analytically optimizing q. Keeping it as a
    standalone function lets one fixed observer score every state in a field
    trajectory without comparing layer-private posterior coordinates.
    """
    lv_p = torch.clamp(lv_p, -5.0, 2.0)
    var_m = torch.exp(lv_p) + float(sigma_r) ** 2 + 1e-12
    nll = 0.5 * (
        (observation - mu_p).pow(2) / (var_m + 1e-7)
        + var_m.log()
        + math.log(2.0 * math.pi)
    )
    return nll.mean(dim=-1, keepdim=True)


def compute_point_vfe(
    mu_p: torch.Tensor,
    lv_p: torch.Tensor,
    mu_q: torch.Tensor,
    lv_q: torch.Tensor,
    rgb: torch.Tensor,
    sigma_r: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Gaussian VFE on the point field (RGB or any per-point vector).

    Slices are only a workspace. Cognition / seeing lands here:
    q_n = N(μq, σq²), p_n = N(μp, σp²), observation = rgb_n.
    Same algebra as compute_slice_vfe; last dim is 3 (or d_x), not slice d.
    """
    return compute_slice_vfe(mu_p, lv_p, mu_q, lv_q, rgb, sigma_r=sigma_r)


def reduce_observation_f(
    vfe: Dict[str, torch.Tensor],
    pi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scalar F against one observation.

    ``pi`` is an optional measure *on that same o* (SliceRead dual / rarity).
    It is not a second observation and not a second F.
    """
    F_n = vfe["F"]
    if pi is None:
        return F_n.mean()
    w = pi.to(dtype=F_n.dtype, device=F_n.device)
    if w.shape != F_n.shape:
        w = w.reshape(F_n.shape)
    return (F_n * w).sum() / w.sum().clamp_min(1e-8)


def _init_slice_queries(n_slices: int, d: int) -> nn.Parameter:
    """Orthonormal rows in R^d. Scale no longer matters after RMSNorm, but
    orthogonal init keeps the 32 probes linearly independent at step 0."""
    q = torch.empty(n_slices, d)
    nn.init.orthogonal_(q)
    return nn.Parameter(q.unsqueeze(0))


class BayesianSurpriseGate(nn.Module):
    """Computes prediction error / Bayesian surprise from Language Prior and Vision Evidence,

    and modulates visual slice updates S_{l+1} = S_l + g(U) * delta_S.
    """

    def __init__(
        self,
        d_model: int,
        n_slices: int = 32,
        beta: float = 1.0,
        mode: str = "v0_jepa",
        detach_gate: bool = True,
        detach_pred_target: bool = True,
        constant_val: float = 0.5,
        stiefel_queries: bool = False,
        n_heads: int = 4,
        sigma_r: float = 1.0,
        gate_on: str = "u",
        use_prior_step_condition: bool = False,
        gaussian_head_layout: str = "legacy",
    ):
        super().__init__()
        assert mode in (
            "baseline",
            "random",
            "constant",
            "v0_jepa",
            "v0_shuffled",
            "v0_reverse",
            "v0_global_only",
            "v0_spatial_only",
            "v1_bayes",
        ), f"Unknown mode: {mode}"

        self.d = int(d_model)
        self.n_slices = int(n_slices)
        self.beta = float(beta)
        self.mode = mode
        self.detach_gate = bool(detach_gate)
        self.detach_pred_target = bool(detach_pred_target)
        self.constant_val = float(constant_val)
        self.stiefel_queries = bool(stiefel_queries)
        self.use_prior_step_condition = bool(use_prior_step_condition)
        if gaussian_head_layout not in ("legacy", "per_head"):
            raise ValueError("gaussian_head_layout must be legacy or per_head")
        self.gaussian_head_layout = gaussian_head_layout
        assert self.d % int(n_heads) == 0, (self.d, n_heads)
        self.n_heads = int(n_heads)
        self.dh = self.d // self.n_heads
        self.sigma_r = float(sigma_r)
        gate_on = str(gate_on).lower()
        if gate_on not in ("u", "gap", "f"):
            raise ValueError(f"gate_on must be 'u', 'gap', or 'f', got {gate_on!r}")
        self.gate_on = gate_on
        # Inference-only: snap amortized q to Kalman q*.
        # "amortized" | "star" (all layers) | "residual" (only if gap > q_star_thresh)
        self.q_infer = "amortized"
        self.q_star_thresh = 0.05

        # V0 & V1: Language Prior Predictor H -> Prior on S
        # Language sequence [B, T, d] -> pool or project to M slices [B, M, d]
        if self.mode in ("v0_jepa", "v0_shuffled", "v0_reverse", "v0_global_only", "v0_spatial_only"):
            self.lang_to_prior = nn.Sequential(
                nn.Linear(self.d, self.d),
                nn.GELU(),
                nn.Linear(self.d, self.d),
            )
            self.slice_queries = _init_slice_queries(self.n_slices, self.d)
            self.q_norm = _RMSNorm(self.dh)
            self.h_norm = _RMSNorm(self.dh)

        elif self.mode == "v1_bayes":
            self.slice_queries = _init_slice_queries(self.n_slices, self.d)
            self.q_norm = _RMSNorm(self.dh)
            self.h_norm = _RMSNorm(self.dh)
            # Prior head (from language) -> (mu_p, logvar_p)
            # Per-head Gaussian readouts (not one shared MLP over the mixed d).
            self.prior_head = nn.Sequential(
                nn.Linear(self.dh, self.dh),
                nn.GELU(),
                nn.Linear(self.dh, 2 * self.dh),
            )
            self.post_head = nn.Sequential(
                nn.Linear(self.dh, self.dh),
                nn.GELU(),
                nn.Linear(self.dh, 2 * self.dh),
            )
            # Step-conditioned prior content (opt-in, zero-init = identity).
            # The single-shot F2 prior is a canvas-blind attractor: with
            # prior_write gain 1 every sequential write applies the SAME
            # mu_p, so repeated passes stamp one blob instead of composing.
            # This projection lets the prior content depend on the write
            # step so set-point semantics complete progressively.
            if self.use_prior_step_condition:
                self.prior_step_proj = nn.Linear(1, self.d)
                nn.init.zeros_(self.prior_step_proj.weight)
                nn.init.zeros_(self.prior_step_proj.bias)
            else:
                self.prior_step_proj = None

    def _get_queries(self) -> torch.Tensor:
        """Returns [1, M, d] slice query probes, optionally projected to Stiefel manifold."""
        if getattr(self, "stiefel_queries", False) and hasattr(self, "slice_queries"):
            q_flat = self.slice_queries.squeeze(0)  # [M, d]
            if q_flat.shape[0] > q_flat.shape[1]:
                raise ValueError(
                    "Stiefel slice queries require n_slices <= d_model for "
                    "orthonormal rows"
                )
            # Reduced QR on Q^T gives exact orthonormal rows in Q.  The Muon
            # Newton-Schulz zeropower is useful for optimizer updates but its
            # finite-step polynomial does not satisfy Q @ Q.T == I exactly.
            q_ortho = torch.linalg.qr(q_flat.transpose(0, 1), mode="reduced").Q
            q_ortho = q_ortho.transpose(0, 1)
            return q_ortho.unsqueeze(0)
        return self.slice_queries

    def _predict_prior_from_h(
        self,
        H: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        t=None,
    ) -> torch.Tensor:
        """MoT-aligned multi-head SDPA: M probes attend to language H.

        Per-head RMSNorm on Q/K (QK-Norm). Values stay raw H. No extra in_proj.
        ``t`` optionally conditions the prior content on the write step
        (zero-init projection, exact identity when absent or untrained).
        """
        B, T, _ = H.shape
        h, dh, M = self.n_heads, self.dh, self.n_slices
        q = self._get_queries().expand(B, -1, -1).view(B, M, h, dh)
        k = H.view(B, T, h, dh)
        q = self.q_norm(q).permute(0, 2, 1, 3)  # [B, h, M, dh]
        k = self.h_norm(k).permute(0, 2, 1, 3)  # [B, h, T, dh]
        v = H.view(B, T, h, dh).permute(0, 2, 1, 3)
        scale = dh ** -0.5
        logits = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B, h, M, T]
        none = None
        if text_mask is not None:
            keep = text_mask.bool()
            none = ~keep.any(dim=-1)
            logits = logits.masked_fill(
                ~keep[:, None, None, :], torch.finfo(logits.dtype).min,
            )
            if bool(none.any()):
                logits = logits.masked_fill(none[:, None, None, None], 0.0)
        attn = torch.softmax(logits, dim=-1)
        if none is not None and bool(none.any()):
            attn = attn.masked_fill(none[:, None, None, None], 0.0)
        self.last_attn = attn.detach()
        out = torch.matmul(attn, v)  # [B, h, M, dh]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, M, self.d)
        if t is not None and self.prior_step_proj is not None:
            step = t.reshape(-1, 1).to(dtype=out.dtype, device=out.device)
            out = out + self.prior_step_proj(step).unsqueeze(1)
        return out

    def _head_mlp(self, x: torch.Tensor, mlp: nn.Module) -> torch.Tensor:
        """Apply an MLP independently on each of the n_heads feature blocks."""
        B, M, _ = x.shape
        y = x.view(B, M, self.n_heads, self.dh)
        params = mlp(y)
        if self.gaussian_head_layout == "per_head":
            # [head, (mu, logvar), channel] -> [(mu, logvar), head, channel].
            # Flattening heads first then chunking globally silences half of
            # the context heads for mu and the other half for logvar.
            params = params.reshape(B, M, self.n_heads, 2, self.dh).transpose(2, 3)
        return params.reshape(B, M, -1)

    def prior_predictive(
        self,
        S: torch.Tensor,
        H: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        t=None,
    ) -> Dict[str, torch.Tensor]:
        """Score S under the V1 language prior in one fixed Slice coordinate.

        The returned ``F_min`` is ``-log p(S|H)`` for the same Gaussian model
        used by F2. This method performs no belief update or field write.
        """
        if self.mode != "v1_bayes":
            raise RuntimeError("prior_predictive requires mode='v1_bayes'")
        H_ctx = self._predict_prior_from_h(H, text_mask=text_mask, t=t)
        prior_params = self._head_mlp(H_ctx, self.prior_head)
        mu_p, lv_p = prior_params.chunk(2, dim=-1)
        lv_p = torch.clamp(lv_p, -5.0, 2.0)
        self.last_H_ctx = H_ctx.detach()
        return {
            "F_min": gaussian_prior_predictive_nll(
                mu_p, lv_p, S, sigma_r=self.sigma_r,
            ),
            "mu_p": mu_p,
            "lv_p": lv_p,
            "h_ctx": H_ctx,
        }

    def forward(
        self,
        S: torch.Tensor,
        H: torch.Tensor,
        delta_S: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        t=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """S: [B, M, d] visual slices observed from point field

        H: [B, T, d] language hidden states
        delta_S: optional [B, M, d] update direction from MoT interaction
        text_mask: optional [B, T] 1=keep (pads excluded from prior readout)
        Returns:
          gate: [B, M, 1] modulation multiplier
          meta: dict with 'surprise', 'mean_u', 'mean_gate', 'pred_loss', etc.
                pred_loss is a live tensor: ||stopgrad(S) - prior||^2 so the
                language prior branch receives a real gradient. Gate stays
                detached when detach_gate=True.
        """
        B, M, d = S.shape

        if self.mode == "baseline":
            gate = torch.ones(B, M, 1, device=S.device, dtype=S.dtype)
            return gate, {
                "surprise": torch.zeros(B, M, 1, device=S.device),
                "mean_u": 0.0,
                "mean_gate": 1.0,
                "pred_loss": S.new_zeros(()),
                "sigreg_loss": S.new_zeros(()),
            }

        if self.mode == "constant":
            gate = torch.full((B, M, 1), self.constant_val, device=S.device, dtype=S.dtype)
            return gate, {
                "surprise": torch.zeros(B, M, 1, device=S.device),
                "mean_u": 0.0,
                "mean_gate": self.constant_val,
                "pred_loss": S.new_zeros(()),
                "sigreg_loss": S.new_zeros(()),
            }

        if self.mode == "random":
            gate = torch.rand((B, M, 1), device=S.device, dtype=S.dtype)
            return gate, {
                "surprise": torch.zeros(B, M, 1, device=S.device),
                "mean_u": 0.0,
                "mean_gate": float(gate.mean().item()),
                "pred_loss": S.new_zeros(()),
                "sigreg_loss": S.new_zeros(()),
            }

        # --- V0: Deterministic JEPA Prediction Error & Decompositions ---
        if self.mode in ("v0_jepa", "v0_shuffled", "v0_reverse", "v0_global_only", "v0_spatial_only"):
            H_ctx = self._predict_prior_from_h(H, text_mask=text_mask)
            S_hat = self.lang_to_prior(H_ctx)  # [B, M, d]
            self.last_H_ctx = H_ctx.detach()

            # Error / Innovation: mean squared error across feature dimension d
            # U in [B, M, 1]
            diff = S - S_hat
            u = (diff ** 2).mean(dim=-1, keepdim=True)  # [B, M, 1]
            # Live aux: prior predicts stopgrad(S). Does not pull S toward a
            # collapsed S_hat; does send grad into Q / cross_attn / lang_to_prior.
            S_tgt = S.detach() if self.detach_pred_target else S
            pred_loss = ((S_tgt - S_hat) ** 2).mean()
            sigreg_loss = compute_sigreg_loss(S)["sigreg_total"]

            if self.mode == "v0_shuffled":
                # Randomly permute surprise values across slice dimension M per batch
                perm = torch.randperm(M, device=S.device)
                u = u[:, perm, :]
            elif self.mode == "v0_global_only":
                # Global layer-average surprise U_bar across all slices (no spatial targeting)
                u_bar = u.mean(dim=1, keepdim=True).expand_as(u)  # [B, M, 1]
                u = u_bar
            elif self.mode == "v0_spatial_only":
                # Pure spatial contrast: U_j - U_bar (relative surprise only)
                u_bar = u.mean(dim=1, keepdim=True)
                u = torch.clamp(u - u_bar, min=0.0)

            u_ctrl = u.detach() if self.detach_gate else u
            gate = 1.0 - torch.exp(-self.beta * u_ctrl)

            if self.mode == "v0_reverse":
                gate = 1.0 - gate  # exp(-beta * u_ctrl)

            return gate, {
                "surprise": u.detach(),
                "mean_u": float(u.mean().item()),
                "mean_gate": float(gate.mean().item()),
                "s_hat": S_hat.detach(),
                "h_ctx": H_ctx.detach(),
                "pred_loss": pred_loss,
                "sigreg_loss": sigreg_loss,
            }

        # --- V1: Full Gaussian Bayesian Surprise & KL Decomposition ---
        if self.mode == "v1_bayes":
            prior = self.prior_predictive(S, H, text_mask=text_mask, t=t)
            H_ctx = prior["h_ctx"]
            mu_p, lv_p = prior["mu_p"], prior["lv_p"]
            var_p = torch.exp(lv_p)

            # Posterior conditioned on observed S and context
            post_params = self._head_mlp(S + H_ctx, self.post_head)  # [B, M, 2*d]
            mu_q, lv_q = post_params.chunk(2, dim=-1)
            lv_q = torch.clamp(lv_q, -5.0, 2.0)
            var_q = torch.exp(lv_q)

            # Analytical diagonal Gaussian KL decomposition:
            # D_KL(q || p) = U_mu (mean surprise) + U_sigma (uncertainty surprise)
            # U_mu = (mu_q - mu_p)^2 / (2 * var_p)
            # U_sigma = 0.5 * ( (lv_p - lv_q) + var_q / var_p - 1 )
            u_mu_raw = ((mu_q - mu_p) ** 2) / (2.0 * var_p + 1e-7)
            u_sigma_raw = 0.5 * ((lv_p - lv_q) + var_q / (var_p + 1e-7) - 1.0)

            u_mu = torch.clamp(u_mu_raw.mean(dim=-1, keepdim=True), min=0.0)        # [B, M, 1]
            u_sigma = torch.clamp(u_sigma_raw.mean(dim=-1, keepdim=True), min=0.0)  # [B, M, 1]
            u = u_mu + u_sigma  # Total KL divergence [B, M, 1]

            vfe = compute_slice_vfe(mu_p, lv_p, mu_q, lv_q, S, sigma_r=self.sigma_r)
            # Keep the live KL already used for U (same formula, same clamps).
            vfe["U"] = u
            vfe["u_mu"] = u_mu
            vfe["u_sigma"] = u_sigma

            # Eval-only: q ← q* so surprise/gate use the exact posterior.
            q_infer = str(getattr(self, "q_infer", "amortized")).lower()
            snap = False
            if not self.training and q_infer in ("star", "qstar", "q_*", "*"):
                snap = True
            elif not self.training and q_infer in ("residual", "resid", "gap"):
                snap = bool(float(vfe["gap"].mean()) > float(getattr(self, "q_star_thresh", 0.05)))
            if snap:
                mu_q, lv_q = vfe["mu_star"], vfe["lv_star"]
                vfe = compute_slice_vfe(mu_p, lv_p, mu_q, lv_q, S, sigma_r=self.sigma_r)
                u_mu = vfe["u_mu"]
                u_sigma = vfe["u_sigma"]
                u = vfe["U"]
                vfe["U"] = u
                vfe["q_star_applied"] = True
            else:
                vfe["q_star_applied"] = False

            # Prior predicts stopgrad(S) (same target as V0). Do not train the
            # prior to match an untrained posterior head.
            S_tgt = S.detach() if self.detach_pred_target else S
            pred_loss = ((S_tgt - mu_p) ** 2).mean()
            sigreg_loss = compute_sigreg_loss(S)["sigreg_total"]

            # F2 train target: gap = KL(q ‖ q*) with p and S frozen so only
            # the amortized posterior head is fit. Same as min_q F.
            vfe_q = compute_slice_vfe(
                mu_p.detach(), lv_p.detach(), mu_q, lv_q, S.detach(),
                sigma_r=self.sigma_r,
            )
            vfe_train_loss = vfe_q["gap"].mean()

            if self.gate_on == "gap":
                energy = vfe["gap"]
            elif self.gate_on == "f":
                energy = vfe["F"]
            else:
                energy = u
            e_ctrl = energy.detach() if self.detach_gate else energy
            gate = 1.0 - torch.exp(-self.beta * e_ctrl)

            return gate, {
                "surprise": u.detach(),
                "u_mu": u_mu.detach(),
                "u_sigma": u_sigma.detach(),
                "mean_u": float(u.mean().item()),
                "mean_u_mu": float(u_mu.mean().item()),
                "mean_u_sigma": float(u_sigma.mean().item()),
                "mean_gate": float(gate.mean().item()),
                "mean_logvar_p": float(lv_p.mean().item()),
                "mean_logvar_q": float(lv_q.mean().item()),
                "mu_p": mu_p,
                "mu_q": mu_q.detach(),
                "lv_p": lv_p.detach(),
                "lv_q": lv_q.detach(),
                "sigreg_loss": sigreg_loss,
                "h_ctx": H_ctx.detach(),
                "pred_loss": pred_loss,
                "vfe_train_loss": vfe_train_loss,
                "F": vfe["F"].detach(),
                "gap": vfe["gap"].detach(),
                "F_min": vfe["F_min"].detach(),
                "acc_mean": vfe["acc_mean"].detach(),
                "acc_tr": vfe["acc_tr"].detach(),
                "s_err": vfe["s_err"].detach(),
                "mean_F": float(vfe["F"].mean().item()),
                "mean_gap": float(vfe["gap"].mean().item()),
                "mean_F_min": float(vfe["F_min"].mean().item()),
                "mean_acc_mean": float(vfe["acc_mean"].mean().item()),
                "mean_acc_tr": float(vfe["acc_tr"].mean().item()),
                # Live μ* / K: Kalman-as-S-update writes these. Detach would
                # make the field write a constant and kill prior/read grads.
                "mu_star": vfe["mu_star"],
                "lv_star": vfe["lv_star"],
                "K": vfe["K"],
                "mean_K": float(vfe["K"].mean().item()),
            }

        raise RuntimeError(f"Unhandled mode: {self.mode}")


def global_gate_from_surprise(
    surprise: torch.Tensor,
    slice_gate: Optional[torch.Tensor] = None,
    beta: float = 1.0,
    kind: str = "lse",
    tau: float = 1.0,
    topk: int = 2,
) -> torch.Tensor:
    """Pool per-slice surprise U_j into a global gate g_G of shape [B, 1, 1].

    Not a plain mean: a single 1px-hot slice must not be diluted by M-1 idle
    slices. kind="lse" is a soft-max (log-sum-exp minus the all-zero floor
    tau log M); kind="topk" is the mean of the top-k U_j. Both then use the
    same map as the slice gate, g = 1 - exp(-beta U).

    Modes that do not populate U (baseline / constant / random) have a
    ~0 surprise field but a meaningful per-slice slice_gate. Those fall
    back to pooling the gates so ungated baseline stays open and a forced-zero
    gate stays closed.
    """
    if kind not in ("lse", "topk"):
        raise ValueError(f"unknown global-gate pool {kind!r}")
    u = surprise
    if u.dim() == 3 and u.shape[-1] == 1:
        u = u.squeeze(-1)
    if u.dim() != 2:
        raise ValueError(f"surprise must be [B,M] or [B,M,1], got {tuple(surprise.shape)}")
    B, M = u.shape
    tau_f = max(float(tau), 1e-6)
    beta_f = float(beta)

    def _from_u(src: torch.Tensor) -> torch.Tensor:
        if kind == "lse":
            u_g = tau_f * torch.logsumexp(src / tau_f, dim=-1, keepdim=True)
            floor = tau_f * math.log(M)
            u_g = (u_g - floor).clamp_min(0.0)
        else:
            k = max(1, min(int(topk), M))
            u_g = src.topk(k, dim=-1).values.mean(dim=-1, keepdim=True).clamp_min(0.0)
        return 1.0 - torch.exp(-beta_f * u_g)

    def _from_g(g: torch.Tensor) -> torch.Tensor:
        gg = g
        if gg.dim() == 3 and gg.shape[-1] == 1:
            gg = gg.squeeze(-1)
        if kind == "lse":
            w = torch.softmax(gg / tau_f, dim=-1)
            return (w * gg).sum(dim=-1, keepdim=True)
        k = max(1, min(int(topk), gg.shape[-1]))
        return gg.topk(k, dim=-1).values.mean(dim=-1, keepdim=True)

    g_from_u = _from_u(u)
    live = (u.detach().abs() > 1e-12).any(dim=-1, keepdim=True)
    if slice_gate is None:
        g_G = g_from_u
    else:
        g_G = torch.where(live, g_from_u, _from_g(slice_gate))
    return g_G.unsqueeze(-1)


def spatial_surprise_map(
    U: torch.Tensor,
    A: torch.Tensor,
    res: Optional[int] = None,
) -> torch.Tensor:
    """Project slice-level surprise U [B, M, 1] (or [B, M]) back to point space via router assignment A.

    A: [B, N, M] or [B, H_heads, N, M]
    Returns:
      u_point: [B, N] or [B, 1, res, res] if res is specified.
    """
    if U.dim() == 3 and U.shape[-1] == 1:
        U = U.squeeze(-1)  # [B, M]

    if A.dim() == 4:
        # [B, H_heads, N, M] -> average heads -> [B, N, M]
        A = A.mean(dim=1)

    # u_point = sum_j A_ij * U_j
    # einsum: b n m, b m -> b n
    u_point = torch.einsum("bnm,bm->bn", A, U)

    if res is not None:
        B, N = u_point.shape
        assert N == res * res, f"N={N} does not match res^2={res*res}"
        return u_point.view(B, 1, res, res)

    return u_point
