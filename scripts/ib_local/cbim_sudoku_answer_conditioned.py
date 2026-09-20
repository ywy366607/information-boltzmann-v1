"""CBIM Sudoku with Answer Self-Conditioning (P_problem + P_answer).
At each microstep k, the collision operator is conditioned on both:
1. P_problem: The frozen initial problem clues
2. P_answer: The updated draft answer probabilities from step k-1
Strict zero energy injection: ||F_k||^2 is identically conserved.
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


class AnswerConditionedCBIMSudokuModel(nn.Module):
    """CBIM Sudoku Solver with Answer Self-Conditioning."""
    def __init__(self, vocab_size: int = 11, d_channels: int = 128, n_velocities: int = 8,
                 ponder_steps: int = 16, halt_max_steps: int = 16, multi_step_loss: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d_channels
        self.n_v = n_velocities
        self.d_c = d_channels // n_velocities
        self.ponder_steps = ponder_steps
        self.halt_max_steps = halt_max_steps
        self.multi_step_loss = multi_step_loss

        # 1. Problem Clue Projection
        self.embed_tokens = nn.Embedding(vocab_size, d_channels)
        self.clue_proj = nn.Sequential(
            nn.Linear(d_channels, d_channels),
            nn.SiLU(),
            nn.Linear(d_channels, d_channels)
        )

        # 2. Answer Draft Projection (projects soft 11-class probabilities to d_channels)
        self.ans_proj = nn.Sequential(
            nn.Linear(vocab_size, d_channels),
            nn.SiLU(),
            nn.Linear(d_channels, d_channels)
        )

        # 3. Kinetic Operators
        self.transport = UnitaryCayleyTransport2D(d_channels=d_channels, n_velocities=n_velocities)
        self.collision = GivensCollision2D(d_channels=d_channels, n_layers=2, conditioned=True)

        # 4. Characteristic Kernel Readout
        self.readout_mlp = nn.Sequential(
            nn.Linear(d_channels, 256),
            nn.SiLU(),
            nn.Linear(256, vocab_size)
        )

        # 5. Q-Halt Head
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

        # 1. Embed initial clues (P_problem)
        emb = self.embed_tokens(inputs).view(B, 9, 9, self.d)
        clue_field = self.clue_proj(emb)

        # Initial draft answer P_ans^(0) from initial clues
        initial_logits = self.readout_mlp(clue_field.view(B, 81, self.d))
        current_ans_prob = F.softmax(initial_logits, dim=-1).view(B, 9, 9, self.vocab_size)
        current_ans_field = self.ans_proj(current_ans_prob)

        prior_field = torch.where(carry.halted.view(B, 1, 1, 1), clue_field, carry.inner_carry.field)

        # 2. Kinetic Pondering with Answer Self-Conditioning
        curr = prior_field
        step_logits = []
        stride = max(1, self.ponder_steps // 16)

        for k in range(1, self.ponder_steps + 1):
            # Combined condition: Problem + Current Draft Answer
            combined_cond = clue_field + current_ans_field

            f_5d = curr.view(B, 9, 9, self.n_v, self.d_c)
            f_tr = self.transport(f_5d).view(B, 9, 9, self.d)
            curr = self.collision(f_tr, cond=combined_cond)

            # Readout updated draft answer at cell level
            flat_k = curr.view(B, 81, self.d)
            logits_k = self.readout_mlp(flat_k)

            # Update the draft answer for step k+1
            new_ans_prob = F.softmax(logits_k, dim=-1).view(B, 9, 9, self.vocab_size)
            current_ans_field = self.ans_proj(new_ans_prob)

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
