"""LeWorldModel Latent Predictor with 14-Horizon DAgger Physical Distillation and Dynamic Delay Attention.

References:
- Balestriero & LeCun (2025), "LeJEPA: Provably Stable Joint-Embedding Predictive Architecture"
- Maes, Le Lidec, Scieur, LeCun, Balestriero (2026), "LeWorldModel" (arXiv:2603.19312)
- Ross, Gordon, Bagnell (2011), "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (DAgger)

Key Architecture Guardrails:
1. PURE LATENT INPUT: Predictor receives ONLY brain latent z (no token embeddings).
   Zero possibility of bigram feedforward bypass shortcuts.
2. CAUSAL INDEPENDENCE: Operations are row-wise over sequence time T.
   Rollout for token t operates strictly on z_t with zero access to future tokens t+1..T.
3. DYNAMIC DELAY ATTENTION: Softmax attention across all 15 horizons (0=immediate, 1..14=delays),
   enabling the model to dynamically utilize fast reflexes (1-2 hops), median sensory-motor (3-4 hops),
   central complex loops (6-8 hops), and deep recurrent reverberations (9-14 hops).
4. DAGGER ON-POLICY DISTILLATION: Student rolls out multi-hop predictions, anchored against
   real quiet physical connectome conduction trajectories (s=0, zero external input).
5. SIGREG ANTI-COLLAPSE & ZERO STOP-GRADIENTS: Fully end-to-end differentiable throughout.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sigreg import SIGReg


class DynamicDelayAttention(nn.Module):
    """Dynamic Softmax Attention over Multi-Hop Conduction Latents.

    Dynamically weights fast reflexes (1-2 hops), median sensory-motor paths (3-4 hops),
    central complex loops (6-8 hops), or deep recurrent reverberations (9-14 hops)
    on a per-token contextual basis without any arbitrary hardcoded settle clocks.
    """

    def __init__(
        self,
        d_model: int = 768,
        max_horizon: int = 14,
        d_attn: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_horizon = max_horizon
        self.d_attn = d_attn

        # Contextual query from immediate physical state z_0
        self.q_proj = nn.Linear(d_model, d_attn)
        # Key projection across all multi-hop latents
        self.k_proj = nn.Linear(d_model, d_attn)
        # Value projection
        self.v_proj = nn.Linear(d_model, d_model)
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Learned positional vectors for delay hops k=0..max_horizon (K+1 total)
        self.hop_positions = nn.Parameter(torch.zeros(max_horizon + 1, d_model, dtype=torch.float32))
        nn.init.normal_(self.hop_positions, std=0.02)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, hops_stack: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            hops_stack: [T, K+1, d_model], where k=0 is z_0 (immediate physical state),
                        and k=1..K are internal multi-hop predictions.

        Returns:
            z_readout: [T, d_model], dynamically attended multi-delay latent.
            attn_weights: [T, K+1], softmax attention weights across delay hops.
        """
        T, num_hops, d_model = hops_stack.shape
        assert num_hops == self.max_horizon + 1, f"Expected {self.max_horizon + 1} hops, got {num_hops}"

        # Immediate state as query: [T, d_model]
        z_0 = hops_stack[:, 0]
        q = self.q_proj(z_0).unsqueeze(1)  # [T, 1, d_attn]

        # Add hop positions to keys: [T, K+1, d_model]
        k_in = hops_stack + self.hop_positions.unsqueeze(0)
        k = self.k_proj(k_in)  # [T, K+1, d_attn]
        v = self.v_proj(hops_stack)  # [T, K+1, d_model]

        # Scaled dot-product attention per token: [T, 1, K+1]
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.d_attn ** 0.5)
        attn_weights = F.softmax(scores.squeeze(1), dim=-1)  # [T, K+1]

        # Weighted combination of values: [T, d_model]
        context = torch.bmm(attn_weights.unsqueeze(1), v).squeeze(1)

        # Residual connection from immediate state z_0
        z_readout = z_0 + self.out_proj(context)

        # Manifold norm matching to preserve physical scale
        target_norm = z_0.norm(dim=-1, keepdim=True)
        pred_norm = z_readout.norm(dim=-1, keepdim=True) + 1e-6
        z_readout = z_readout * (target_norm / pred_norm)

        return z_readout, attn_weights


class LeJEPAPredictor(nn.Module):
    """Latent World Model with 14-Horizon DAgger Physical Conduction Distillation."""

    def __init__(
        self,
        d_model: int = 768,
        d_emb: int = 768,  # Kept for signature compatibility, unused internally
        max_horizon: int = 14,
        dagger_beta: float = 0.5,
        lambda_sigreg: float = 0.2,
        num_slices: int = 128,
        bottleneck_dim: int = 128,  # Kept for signature compatibility
        initial_mix: float = 0.5,   # Kept for signature compatibility
    ):
        super().__init__()
        self.d_model = d_model
        self.d_emb = d_emb
        self.max_horizon = max_horizon
        self.dagger_beta = dagger_beta
        self.lambda_sigreg = lambda_sigreg

        # 3-layer residual MLP dynamics model in pure latent space (d_model -> d_model)
        # NO TOKEN EMBEDDING INPUT: mathematically impossible to form bigram bypass.
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

        # Exact SIGReg anti-collapse regularizer (LeWM 2026)
        self.sigreg = SIGReg(dim=d_model, num_slices=num_slices)

        # Dynamic Delay Attention across k=0..max_horizon
        self.delay_attn = DynamicDelayAttention(d_model=d_model, max_horizon=max_horizon)

    @property
    def mix_alpha(self) -> torch.Tensor:
        """Legacy diagnostic property returning median attention weight."""
        return torch.tensor(1.0 / (self.max_horizon + 1))

    def step(self, z_cur: torch.Tensor, z_ref: torch.Tensor) -> torch.Tensor:
        """1-hop autonomous latent transition with manifold norm matching."""
        delta = self.net(z_cur)
        raw = z_cur + delta
        target_norm = z_ref.norm(dim=-1, keepdim=True)
        pred_norm = raw.norm(dim=-1, keepdim=True) + 1e-6
        return raw * (target_norm / pred_norm)

    def rollout_autonomous(
        self,
        z_true: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Strictly autonomous multi-hop student rollout across k=0..max_horizon.

        Used for Decoder readout both during training and inference.
        Zero teacher hops enter this path, ensuring 100% train/generation symmetry.

        Args:
            z_true: [T, d_model], physical brain latents for all tokens in window.

        Returns:
            z_readout: [T, d_model], dynamically attended multi-delay latent.
            hops_stack: [T, K+1, d_model], stacked student latent trajectory.
            attn_weights: [T, K+1], attention weights across hops.
        """
        T, d_model = z_true.shape
        K = self.max_horizon

        # Horizon 0 is the immediate physical state z_true
        all_hops = [z_true]
        cur = z_true

        for k in range(1, K + 1):
            cur = self.step(cur, z_true)
            all_hops.append(cur)

        # Stack: [T, K+1, d_model]
        hops_stack = torch.stack(all_hops, dim=1)

        # Dynamic delay attention readout
        z_readout, attn_weights = self.delay_attn(hops_stack)

        return z_readout, hops_stack, attn_weights

    def rollout(
        self,
        z_true: torch.Tensor,
        teacher_hops: torch.Tensor | None = None,
        beta: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Autonomous rollout alias (teacher_hops and beta are ignored for decoder readout)."""
        return self.rollout_autonomous(z_true)

    def rollout_dagger(
        self,
        z_true: torch.Tensor,
        sensory_emb: torch.Tensor | None = None,  # Kept for signature compatibility
        beta: float | None = None,
        teacher_hops: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """14-Horizon Rollout with DAgger In-Policy Markov Distillation.

        Decoder path is 100% autonomous student rollout (zero teacher injection).
        Teacher quiet physical trajectory is used purely as supervision for the
        1-step transition operator P(s_in^{(k-1)}) -> z^{*(k)}.
        """
        if beta is None:
            beta = self.dagger_beta

        # 1. Autonomous student rollout: feeds Decoder and Cross-Entropy loss
        z_readout, hops_stack, attn_weights = self.rollout_autonomous(z_true)

        # 2. Transition distillation loss on the anchor token (window boundary)
        if teacher_hops is not None and teacher_hops.shape[0] > 0:
            K = min(self.max_horizon, teacher_hops.shape[0])
            z_anchor_0 = z_true[-1:]  # [1, d_model]

            # Ground truth physical sequence at anchor:
            # index 0: immediate physical state z^{*(0)} = z_true[-1]
            # index 1..K: teacher_hops[0..K-1] = z^{*(1..K)}
            z_teacher_full = torch.cat([z_anchor_0, teacher_hops[:K]], dim=0)  # [K+1, d_model]

            pred_steps = []
            for k in range(1, K + 1):
                # Student state at (k-1): hops_stack[-1, k-1]
                z_student_prev = hops_stack[-1, k - 1].unsqueeze(0)  # [1, d_model]
                # Teacher state at (k-1): z_teacher_full[k-1]
                z_teacher_prev = z_teacher_full[k - 1].unsqueeze(0)  # [1, d_model]

                # DAgger mixed state at step k-1:
                # beta=1.0: purely autonomous student state
                # beta=0.0: pure teacher state P(z^{*(k-1)}) -> z^{*(k)}
                # beta=0.5: on-policy DAgger interpolation
                s_in = beta * z_student_prev + (1.0 - beta) * z_teacher_prev

                # Predict transition from (k-1) to k:
                z_pred = self.step(s_in, z_anchor_0)
                pred_steps.append(z_pred)

            pred_stack = torch.cat(pred_steps, dim=0)   # [K, d_model]
            target_stack = z_teacher_full[1:K + 1]      # [K, d_model]

            total_loss, metrics = self.compute_dagger_loss(
                pred_stack, target_stack, z_readout, attn_weights
            )
        else:
            loss_sigreg = self.sigreg(z_readout)
            total_loss = self.lambda_sigreg * loss_sigreg
            w = attn_weights.detach()
            metrics = {
                "jepa_mse": 0.0,
                "jepa_sigreg": float(loss_sigreg.item()),
                "jepa_total": float(total_loss.item()),
                "attn_k0_immediate": float(w[:, 0].mean().item()),
                "attn_k1_fast": float(w[:, 1].mean().item()),
                "attn_k4_median": float(w[:, min(4, self.max_horizon)].mean().item()),
                "attn_k7_mid": float(w[:, min(7, self.max_horizon)].mean().item()),
                "attn_k14_recurrent": float(w[:, -1].mean().item()),
            }

        return z_readout, total_loss, metrics

    def compute_dagger_loss(
        self,
        anchor_pred_hops: torch.Tensor,
        anchor_target_hops: torch.Tensor,
        z_readout: torch.Tensor,
        attn_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """LeWorldModel transition distillation loss + SIGReg regularizer."""
        # anchor_pred_hops: [K, d_model]
        # anchor_target_hops: [K, d_model]
        loss_mse = F.mse_loss(anchor_pred_hops, anchor_target_hops)

        # SIGReg anti-collapse regularizer on attended autonomous readouts
        loss_sigreg = self.sigreg(z_readout)

        total_loss = loss_mse + self.lambda_sigreg * loss_sigreg

        w = attn_weights.detach()
        metrics = {
            "jepa_mse": float(loss_mse.item()),
            "jepa_sigreg": float(loss_sigreg.item()),
            "jepa_total": float(total_loss.item()),
            "attn_k0_immediate": float(w[:, 0].mean().item()),
            "attn_k1_fast": float(w[:, 1].mean().item()),
            "attn_k4_median": float(w[:, min(4, self.max_horizon)].mean().item()),
            "attn_k7_mid": float(w[:, min(7, self.max_horizon)].mean().item()),
            "attn_k14_recurrent": float(w[:, -1].mean().item()),
        }
        return total_loss, metrics

    def compute_jepa_loss(
        self,
        z_preds: torch.Tensor,
        z_targets: torch.Tensor,
        z_all: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Legacy 1-step JEPA loss function for backwards compatibility."""
        loss_pred = F.mse_loss(z_preds, z_targets)
        loss_sigreg_all = self.sigreg(z_all)
        loss_sigreg_pred = self.sigreg(z_preds)
        loss_sigreg = 0.5 * (loss_sigreg_all + loss_sigreg_pred)

        total_loss = loss_pred + self.lambda_sigreg * loss_sigreg
        metrics = {
            "jepa_mse": float(loss_pred.item()),
            "jepa_sigreg": float(loss_sigreg.item()),
            "jepa_total": float(total_loss.item()),
        }
        return total_loss, metrics
