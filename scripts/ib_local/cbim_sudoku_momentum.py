"""CBIM Sudoku with Phase-Space Momentum and Kinetic Optimization Dynamics.
Implements internal momentum (Polyak Heavy-Ball, Nesterov, and AdamW-style adaptive damping)
in the Hamiltonian pondering loop to suppress microstep zig-zag oscillation and accelerate
constraint-satisfaction convergence on Sudoku-Extreme.

Strict zero-energy-injection: ||F_k||^2 is strictly conserved on the Riemannian sphere.
"""
import os
import sys

# Pop script directory to avoid shadowing standard library 'types'
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Also add external/TinyRecursiveModels for imports
trm_dir = os.path.join(repo_root, "external", "TinyRecursiveModels")
if trm_dir not in sys.path:
    sys.path.insert(0, trm_dir)

import math
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.ib_local.cbim_sudoku import (
    UnitaryCayleyTransport2D,
    GivensCollision2D,
    CBIMSudokuCarry,
    CBIMSudokuInnerCarry
)


class MomentumCBIMSudokuModel(nn.Module):
    """CBIM Sudoku Solver with Internal Phase-Space Momentum."""
    def __init__(self, vocab_size: int = 11, d_channels: int = 128, n_velocities: int = 8,
                 ponder_steps: int = 16, halt_max_steps: int = 16, multi_step_loss: bool = True,
                 momentum_mode: str = "heavy_ball", beta: float = 0.7, beta2: float = 0.99):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d_channels
        self.n_v = n_velocities
        self.d_c = d_channels // n_velocities
        self.ponder_steps = ponder_steps
        self.halt_max_steps = halt_max_steps
        self.multi_step_loss = multi_step_loss
        self.momentum_mode = momentum_mode.lower()  # "none", "heavy_ball", "adam", "nesterov"
        self.beta = beta
        self.beta2 = beta2

        # 1. Clue Input Embedding
        self.embed_tokens = nn.Embedding(vocab_size, d_channels)
        self.clue_proj = nn.Sequential(
            nn.Linear(d_channels, d_channels),
            nn.SiLU(),
            nn.Linear(d_channels, d_channels)
        )

        # 2. Kinetic Operators (Zero-Energy Conditioned Flow)
        self.transport = UnitaryCayleyTransport2D(d_channels=d_channels, n_velocities=n_velocities)
        self.collision = GivensCollision2D(d_channels=d_channels, n_layers=2, conditioned=True)

        # 3. Characteristic Kernel Readout
        self.readout_mlp = nn.Sequential(
            nn.Linear(d_channels, 256),
            nn.SiLU(),
            nn.Linear(256, vocab_size)
        )

        # 4. Q-Halt Head
        self.q_head = nn.Linear(d_channels, 2)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> CBIMSudokuCarry:
        B = batch["inputs"].shape[0]
        device = batch["inputs"].device
        field = torch.zeros(B, 9, 9, self.d, dtype=torch.float32, device=device)
        z_H = torch.zeros(B, 81, self.d, dtype=torch.float32, device=device)
        return CBIMSudokuCarry(
            inner_carry=CBIMSudokuInnerCarry(field=field, z_H=z_H),
            steps=torch.zeros((B,), dtype=torch.int32, device=device),
            halted=torch.ones((B,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(self, carry: CBIMSudokuCarry, batch: Dict[str, torch.Tensor],
                return_keys: list = []) -> Tuple[CBIMSudokuCarry, Dict[str, Any]]:
        device = batch["inputs"].device
        B = batch["inputs"].shape[0]

        inputs = torch.where(carry.halted.view(-1, 1), batch["inputs"], carry.current_data["inputs"])
        labels = torch.where(carry.halted.view(-1, 1), batch["labels"], carry.current_data["labels"])
        new_steps = torch.where(carry.halted, 0, carry.steps)

        # 1. Embed clues [B, 81, D] -> [B, 9, 9, D]
        emb = self.embed_tokens(inputs).view(B, 9, 9, self.d)
        clue_field = self.clue_proj(emb)

        prior_field = torch.where(carry.halted.view(B, 1, 1, 1), clue_field, carry.inner_carry.field)

        # 2. Kinetic Pondering with Internal Phase-Space Momentum
        curr = prior_field
        # Reference norm for strict energy conservation ||F_k||^2 == ||F_0||^2
        orig_norm = torch.linalg.vector_norm(curr.float(), dim=(1, 2, 3), keepdim=True).to(curr.dtype)

        # Momentum buffers
        vel = torch.zeros_like(curr)
        var_s = torch.zeros_like(curr)

        step_logits = []
        stride = max(1, self.ponder_steps // 16)

        for k in range(1, self.ponder_steps + 1):
            if self.momentum_mode == "nesterov" and k > 1:
                # Nesterov lookahead point
                look_cand = curr + self.beta * vel
                look_norm = torch.linalg.vector_norm(look_cand.float(), dim=(1, 2, 3), keepdim=True).to(curr.dtype)
                curr_eval = look_cand * (orig_norm / (look_norm + 1e-8))
            else:
                curr_eval = curr

            # Physical transport + collision step
            f_5d = curr_eval.view(B, 9, 9, self.n_v, self.d_c)
            f_tr = self.transport(f_5d).view(B, 9, 9, self.d)
            f_col = self.collision(f_tr, cond=clue_field)

            # Net displacement / constraint force
            delta = f_col - curr_eval

            # Momentum updates
            if self.momentum_mode == "none":
                curr = f_col

            elif self.momentum_mode == "heavy_ball" or self.momentum_mode == "nesterov":
                # Heavy-Ball Momentum: V_k = beta * V_{k-1} + (1 - beta) * delta
                vel = self.beta * vel + (1.0 - self.beta) * delta
                cand = curr + vel
                # Strict spherical projection to conserve exact field energy
                cand_norm = torch.linalg.vector_norm(cand.float(), dim=(1, 2, 3), keepdim=True).to(curr.dtype)
                curr = cand * (orig_norm / (cand_norm + 1e-8))

            elif self.momentum_mode == "adam":
                # AdamW-style adaptive momentum:
                # First moment (directional consensus)
                vel = self.beta * vel + (1.0 - self.beta) * delta
                # Second moment (coordinate-wise uncertainty / variance)
                var_s = self.beta2 * var_s + (1.0 - self.beta2) * (delta ** 2)

                # Bias correction
                m_hat = vel / (1.0 - (self.beta ** k))
                v_hat = var_s / (1.0 - (self.beta2 ** k))

                # Coordinate-wise noise attenuation
                adapt_dir = m_hat / (torch.sqrt(v_hat) + 1e-6)

                # Scale-matching: preserve natural physical RMS scale of the force
                delta_rms = torch.sqrt(torch.mean(delta.float() ** 2, dim=(1, 2, 3), keepdim=True) + 1e-8).to(curr.dtype)
                adapt_rms = torch.sqrt(torch.mean(adapt_dir.float() ** 2, dim=(1, 2, 3), keepdim=True) + 1e-8).to(curr.dtype)
                step = adapt_dir * (delta_rms / (adapt_rms + 1e-8))

                cand = curr + step
                cand_norm = torch.linalg.vector_norm(cand.float(), dim=(1, 2, 3), keepdim=True).to(curr.dtype)
                curr = cand * (orig_norm / (cand_norm + 1e-8))

            else:
                raise ValueError(f"Unknown momentum_mode: {self.momentum_mode}")

            # Collect logits for multi-step loss supervision
            if k % stride == 0 or k == self.ponder_steps:
                flat_k = curr.view(B, 81, self.d)
                logits_k = self.readout_mlp(flat_k)
                step_logits.append(logits_k)

        logits = step_logits[-1]
        flat_field = curr.view(B, 81, self.d)
        q_logits = self.q_head(flat_field[:, 0])
        q_halt, q_cont = q_logits[:, 0], q_logits[:, 1]

        new_steps = new_steps + 1
        new_halted = (q_halt >= 0) | (new_steps >= self.halt_max_steps)

        new_carry = CBIMSudokuCarry(
            inner_carry=CBIMSudokuInnerCarry(field=curr.detach(), z_H=flat_field.detach()),
            steps=new_steps,
            halted=new_halted,
            current_data={"inputs": inputs, "labels": labels}
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt,
            "q_continue_logits": q_cont,
            "all_step_logits": step_logits
        }
        return new_carry, outputs
