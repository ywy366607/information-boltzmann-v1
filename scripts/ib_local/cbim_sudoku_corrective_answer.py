"""CBIM Sudoku with Learned Isoenergetic Corrective Flow and Answer Self-Conditioning.
Compares:
- Arm C: Problem-Only Corrective Flow (cond = P_problem)
- Arm C+: Problem + Draft Answer Self-Conditioning (cond = P_problem + P_ans^(k-1))

Strictly isoenergetic: ||F_k||^2 == ||F_0||^2 via Gram-Schmidt tangent projection + spherical geodesic rotation.
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


class ArmCPlusCBIMSudokuModel(nn.Module):
    """CBIM Sudoku Solver with Learned Isoenergetic Corrective Flow and optional Answer Conditioning."""
    def __init__(self, vocab_size: int = 11, d_channels: int = 128, n_velocities: int = 8,
                 ponder_steps: int = 16, halt_max_steps: int = 16, multi_step_loss: bool = True,
                 answer_conditioned: bool = True, alpha_max: float = 0.25):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d_channels
        self.n_v = n_velocities
        self.d_c = d_channels // n_velocities
        self.ponder_steps = ponder_steps
        self.halt_max_steps = halt_max_steps
        self.multi_step_loss = multi_step_loss
        self.answer_conditioned = answer_conditioned
        self.alpha_max = alpha_max

        # 1. Clue Input Embedding (P_problem)
        self.embed_tokens = nn.Embedding(vocab_size, d_channels)
        self.clue_proj = nn.Sequential(
            nn.Linear(d_channels, d_channels),
            nn.SiLU(),
            nn.Linear(d_channels, d_channels)
        )

        # 2. Draft Answer Projection (P_ans)
        if self.answer_conditioned:
            self.ans_proj = nn.Sequential(
                nn.Linear(vocab_size, d_channels),
                nn.SiLU(),
                nn.Linear(d_channels, d_channels)
            )

        # 3. Kinetic Operators (TC)
        self.transport = UnitaryCayleyTransport2D(d_channels=d_channels, n_velocities=n_velocities)
        self.collision = GivensCollision2D(d_channels=d_channels, n_layers=2, conditioned=True)

        # 4. Learned Isoenergetic Corrective Flow components
        # G_phi(F*, cond) -> raw relaxation direction
        self.corrective_net = nn.Sequential(
            nn.Linear(2 * d_channels, d_channels),
            nn.SiLU(),
            nn.Linear(d_channels, d_channels)
        )
        # a_phi(F*, cond) -> adaptive relaxation angle alpha_k
        self.angle_gate = nn.Sequential(
            nn.Linear(2 * d_channels, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        # Initialize corrective net output close to zero for smooth warm-start
        nn.init.zeros_(self.corrective_net[-1].weight)
        nn.init.zeros_(self.corrective_net[-1].bias)
        nn.init.zeros_(self.angle_gate[-1].weight)
        nn.init.constant_(self.angle_gate[-1].bias, -2.0)  # sigmoid(-2) ~ 0.12

        # 5. Characteristic Kernel Readout
        self.readout_mlp = nn.Sequential(
            nn.Linear(d_channels, 256),
            nn.SiLU(),
            nn.Linear(256, vocab_size)
        )

        # 6. Q-Halt Head
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

        # 1. Embed clues [B, 81, D] -> [B, 9, 9, D] (P_problem)
        emb = self.embed_tokens(inputs).view(B, 9, 9, self.d)
        clue_field = self.clue_proj(emb)

        # 2. Initial Draft Answer P_ans^(0) from clues
        if self.answer_conditioned:
            initial_logits = self.readout_mlp(clue_field.view(B, 81, self.d))
            current_ans_prob = F.softmax(initial_logits, dim=-1).view(B, 9, 9, self.vocab_size)
            current_ans_field = self.ans_proj(current_ans_prob)

        prior_field = torch.where(carry.halted.view(B, 1, 1, 1), clue_field, carry.inner_carry.field)

        curr = prior_field
        orig_norm = torch.linalg.vector_norm(curr.float(), dim=(1, 2, 3), keepdim=True).to(curr.dtype)

        step_logits = []
        stride = max(1, self.ponder_steps // 16)

        # 3. Kinetic Pondering Loop
        for k in range(1, self.ponder_steps + 1):
            if self.answer_conditioned:
                cond = clue_field + current_ans_field
            else:
                cond = clue_field

            # A. Conservative Physical Transport + Collision: F* = C(T(F); cond)
            f_5d = curr.view(B, 9, 9, self.n_v, self.d_c)
            f_tr = self.transport(f_5d).view(B, 9, 9, self.d)
            f_star = self.collision(f_tr, cond=cond)

            # B. Learned Isoenergetic Corrective Flow
            cat_input = torch.cat([f_star, cond], dim=-1)
            v_k = self.corrective_net(cat_input)

            # C. Gram-Schmidt Tangent Projection to Riemannian Sphere S:
            #    u_k = v_k - F* * (<F*, v_k> / ||F*||^2)
            f_star_f32 = f_star.float()
            v_k_f32 = v_k.float()
            dot_prod = torch.sum(f_star_f32 * v_k_f32, dim=(1, 2, 3), keepdim=True)
            norm_sq = torch.sum(f_star_f32 ** 2, dim=(1, 2, 3), keepdim=True) + 1e-8
            proj = dot_prod / norm_sq
            u_k_f32 = v_k_f32 - proj * f_star_f32
            u_norm = torch.linalg.vector_norm(u_k_f32, dim=(1, 2, 3), keepdim=True)
            u_hat_f32 = u_k_f32 / (u_norm + 1e-8)

            # D. Adaptive relaxation angle alpha_k in [0, alpha_max]
            gate_logits = self.angle_gate(cat_input)
            mean_gate = torch.mean(gate_logits, dim=(1, 2, 3), keepdim=True)
            alpha_k = self.alpha_max * torch.sigmoid(mean_gate).to(curr.dtype)

            # E. Spherical Geodesic Rotation:
            #    F_{k+1} = cos(alpha_k) * F* + ||F*|| * sin(alpha_k) * u_hat
            #    Analytically exact ||F_{k+1}||^2 == ||F*||^2 == ||F_0||^2
            f_norm = torch.linalg.vector_norm(f_star_f32, dim=(1, 2, 3), keepdim=True).to(curr.dtype)
            cos_a = torch.cos(alpha_k)
            sin_a = torch.sin(alpha_k)
            curr = (cos_a * f_star) + (f_norm * sin_a * u_hat_f32.to(curr.dtype))

            # F. Readout and update draft answer for step k+1
            flat_k = curr.view(B, 81, self.d)
            logits_k = self.readout_mlp(flat_k)

            if self.answer_conditioned:
                new_ans_prob = F.softmax(logits_k, dim=-1).view(B, 9, 9, self.vocab_size)
                current_ans_field = self.ans_proj(new_ans_prob)

            # Collect logits for multi-step loss supervision
            if k % stride == 0 or k == self.ponder_steps:
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
