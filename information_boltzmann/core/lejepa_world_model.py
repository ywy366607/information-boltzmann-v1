"""LeJEPA / LeWorldModel (LeWM) Architecture for Brain Physical Dynamics.

References:
- Maes, Le Lidec, Scieur, LeCun, Balestriero (2026), "LeWorldModel" (arXiv:2603.19312)
- Balestriero & LeCun (2025), "LeJEPA: Provably Stable Joint-Embedding Predictive Architecture"

Architecture:
1. Encoder E_theta:
   Maps raw motor neuron state h_motor in R^{2333} to latent state z_t in R^D.
2. Predictor P_phi:
   Action/token-conditioned residual dynamics model:
   ẑ_{t+1} = P_phi(z_t, token_emb_t) in R^D.
3. Loss Objective:
   L_total = L_pred(ẑ_{t+1}, z_{t+1}) + lambda_sig * SIGReg(Z)
   where L_pred is MSE in latent space, and SIGReg prevents collapse without
   stop-gradients, momentum encoders (EMA), or contrastive pairs.
4. Downstream Language Probe:
   Linear head mapping z_t to vocabulary logits p(x_t).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sigreg import SIGReg


class LeJEPAWorldModel(nn.Module):
    """Lean Joint-Embedding Predictive Architecture (LeJEPA / LeWM)."""

    def __init__(
        self,
        n_motor: int = 2333,
        d_latent: int = 128,
        d_emb: int = 128,
        vocab_size: int = 50257,
        lambda_sigreg: float = 1.0,
        num_slices: int = 64,
    ):
        super().__init__()
        self.n_motor = n_motor
        self.d_latent = d_latent
        self.d_emb = d_emb
        self.lambda_sigreg = lambda_sigreg

        # 1. Professional Encoder: LayerNorm + MLP with residual
        self.encoder = nn.Sequential(
            nn.LayerNorm(n_motor),
            nn.Linear(n_motor, d_latent * 2),
            nn.GELU(),
            nn.Linear(d_latent * 2, d_latent),
            nn.LayerNorm(d_latent),
        )

        # 2. Action/Token-conditioned Predictor: predicts latent transition
        self.predictor = nn.Sequential(
            nn.Linear(d_latent + d_emb, d_latent * 2),
            nn.GELU(),
            nn.Linear(d_latent * 2, d_latent * 2),
            nn.GELU(),
            nn.Linear(d_latent * 2, d_latent),
        )

        # 3. Exact SIGReg Module
        self.sigreg = SIGReg(dim=d_latent, num_slices=num_slices)

        # 4. Downstream Linear Probe for Vocabulary Readout
        self.probe = nn.Linear(d_latent, vocab_size)

    def encode(self, h_motor: torch.Tensor) -> torch.Tensor:
        """Encodes motor neuron state to latent space."""
        return self.encoder(h_motor)

    def predict_next(self, z_cur: torch.Tensor, token_emb: torch.Tensor) -> torch.Tensor:
        """Predicts next latent state with residual skip connection."""
        inp = torch.cat([z_cur, token_emb], dim=-1)
        delta_z = self.predictor(inp)
        return z_cur + delta_z

    def compute_jepa_loss(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        z_all: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes exact LeJEPA / LeWM objective: MSE + SIGReg.

        Args:
            z_pred: [B, D] predicted next latent states.
            z_target: [B, D] actual next latent states from encoder.
            z_all: [B, D] batch of latent embeddings for SIGReg anti-collapse.

        Returns:
            total_loss: scalar tensor.
            metrics: dictionary of component values.
        """
        # Latent MSE prediction error
        loss_pred = F.mse_loss(z_pred, z_target)

        # SIGReg anti-collapse loss on embeddings
        loss_sigreg_enc = self.sigreg(z_all)
        loss_sigreg_pred = self.sigreg(z_pred)
        loss_sigreg = 0.5 * (loss_sigreg_enc + loss_sigreg_pred)

        total_loss = loss_pred + self.lambda_sigreg * loss_sigreg

        metrics = {
            "loss_pred_mse": float(loss_pred.item()),
            "loss_sigreg": float(loss_sigreg.item()),
            "loss_total": float(total_loss.item()),
        }
        return total_loss, metrics
