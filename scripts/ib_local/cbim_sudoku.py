"""CBIM Sudoku Continuous Kinetic Operator.
2D Torus (9x9) physical Boltzmann field solver for Sudoku-Extreme.
Uses Unitary Cayley Transport, Givens SO(124) Collision Rotations,
and Reversible Hamiltonian Pondering with O(1) memory in depth K.
"""
from __future__ import annotations

import os
import sys
import math
from typing import Tuple, Dict, Any, Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class CBIMSudokuInnerCarry:
    field: torch.Tensor       # [B, 9, 9, D]
    z_H: torch.Tensor         # [B, 81, D] for ACT compatibility


@dataclass
class CBIMSudokuCarry:
    inner_carry: CBIMSudokuInnerCarry
    steps: torch.Tensor       # [B]
    halted: torch.Tensor      # [B]
    current_data: Dict[str, torch.Tensor]


class UnitaryCayleyTransport2D(nn.Module):
    """Exact Unitary 2D Cayley Transport on 9x9 Torus."""
    def __init__(self, d_channels: int = 128, n_velocities: int = 8):
        super().__init__()
        self.grid_size = 9
        self.d = d_channels
        self.n_v = n_velocities
        self.d_c = d_channels // n_velocities  # e.g. 128 // 8 = 16

        # 8 discrete velocity vectors in 2D (D2Q8: 4 cardinal, 4 diagonal)
        c_v = torch.tensor([
            [ 1.0,  0.0],  # E
            [-1.0,  0.0],  # W
            [ 0.0,  1.0],  # N
            [ 0.0, -1.0],  # S
            [ 1.0,  1.0],  # NE
            [ 1.0, -1.0],  # SE
            [-1.0,  1.0],  # NW
            [-1.0, -1.0],  # SW
        ], dtype=torch.float32)
        self.register_buffer("c_v", c_v)

        # Precompute modified wavenumbers for 9x9 grid: sin(2pi k / 9)
        k = torch.arange(self.grid_size, dtype=torch.float32)
        sin_k = torch.sin(2.0 * math.pi * k / self.grid_size)
        self.register_buffer("sin_k", sin_k)

        # Base time-step parameter
        self.tau_0 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def get_multiplier(self, dt: float | torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Compute exact unitary Cayley multiplier [9, 9, 8, 1]."""
        # sin_kx: [9, 1], sin_ky: [1, 9]
        sin_kx = self.sin_k.view(self.grid_size, 1, 1)
        sin_ky = self.sin_k.view(1, self.grid_size, 1)

        # omega: [9, 9, 8]
        omega = sin_kx * self.c_v[:, 0] + sin_ky * self.c_v[:, 1]
        half_dt_omega = 0.5 * dt * omega  # [9, 9, 8]

        if inverse:
            # Conjugate: (1 + i omega dt/2) / (1 - i omega dt/2)
            num = torch.complex(torch.ones_like(half_dt_omega), half_dt_omega)
            den = torch.complex(torch.ones_like(half_dt_omega), -half_dt_omega)
        else:
            # (1 - i omega dt/2) / (1 + i omega dt/2)
            num = torch.complex(torch.ones_like(half_dt_omega), -half_dt_omega)
            den = torch.complex(torch.ones_like(half_dt_omega), half_dt_omega)

        mult = (num / den).unsqueeze(-1)  # [9, 9, 8, 1]
        return mult

    def forward(self, field: torch.Tensor, dt: Optional[torch.Tensor] = None, inverse: bool = False) -> torch.Tensor:
        """Apply unitary Cayley transport to field [B, 9, 9, 8, d_c]."""
        B, H, W, n_v, d_c = field.shape
        dt_val = dt if dt is not None else self.tau_0

        mult = self.get_multiplier(dt_val, inverse=inverse)  # [9, 9, 8, 1]
        # 2D FFT over spatial dimensions (H, W)
        field_c = torch.fft.fft2(field.to(torch.complex64), dim=(1, 2))
        field_c = field_c * mult
        field_out = torch.fft.ifft2(field_c, dim=(1, 2)).real
        return field_out.to(field.dtype)


class GivensCollision2D(nn.Module):
    """Local Givens SO(d-4) Collision Rotations conserving density and momentum."""
    def __init__(self, d_channels: int = 128, n_layers: int = 2):
        super().__init__()
        self.d = d_channels
        self.layers = n_layers

        # Conserved subspace: 4 invariants (1 density + 2 momentum + 1 energy)
        velocities = 8
        content_dim = self.d // velocities
        c_v = torch.tensor([
            [ 1.0,  0.0], [-1.0,  0.0], [ 0.0,  1.0], [ 0.0, -1.0],
            [ 1.0,  1.0], [ 1.0, -1.0], [-1.0,  1.0], [-1.0, -1.0]
        ], dtype=torch.float64)
        mass = torch.ones(1, velocities, content_dim, dtype=torch.float64)
        momentum = c_v.T[:, :, None].expand(-1, -1, content_dim)
        energy = (c_v**2).sum(-1)[None, :, None].expand(1, -1, content_dim)
        constraints = torch.cat((mass, momentum, energy), 0).reshape(4, self.d)
        _, singular, right = torch.linalg.svd(constraints, full_matrices=True)
        rank = int((singular > 1e-10).sum())
        nullspace = right[rank:].T.float()  # [d, 124]
        if nullspace.shape[1] % 2 != 0:
            nullspace = nullspace[:, :-1]
        self.nullity = nullspace.shape[1]
        self.d_active = self.nullity
        self.register_buffer("nullspace", nullspace)

        # Alternating rotation schedule pairs for active dimensions
        pairs_l0 = torch.stack([torch.arange(0, self.nullity, 2), torch.arange(1, self.nullity, 2)], dim=-1)
        pairs_l1 = torch.stack([
            torch.where(torch.arange(self.nullity // 2) == 0, self.nullity - 1, 2 * torch.arange(self.nullity // 2) - 1),
            2 * torch.arange(self.nullity // 2)
        ], dim=-1)
        self.register_buffer("pairs_l0", pairs_l0)
        self.register_buffer("pairs_l1", pairs_l1)

        # Angle prediction network
        self.angle_net = nn.Sequential(
            nn.Linear(self.d, 128),
            nn.SiLU(),
            nn.Linear(128, self.layers * (self.d_active // 2))
        )

    def forward(self, field: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Apply local Givens collision rotations to field [B, 9, 9, D]."""
        B, H, W, D = field.shape
        flat = field.view(B, H * W, D)

        # Conserved projection
        coeff = torch.einsum("dk,bnd->bnk", self.nullspace, flat)
        conserved = flat - torch.einsum("dk,bnk->bnd", self.nullspace, coeff)

        # Compute rotation angles
        angles = self.angle_net(flat).view(B, H * W, self.layers, self.d_active // 2)

        val = coeff
        # Layer schedules
        schedules = [self.pairs_l0, self.pairs_l1]
        layer_indices = range(self.layers - 1, -1, -1) if inverse else range(self.layers)

        for layer in layer_indices:
            pair = schedules[layer]
            th = angles[:, :, layer]
            if inverse:
                th = -th
            cos, sin = th.cos(), th.sin()
            l = val[..., pair[:, 0]]
            r = val[..., pair[:, 1]]
            upd = val.clone()
            upd[..., pair[:, 0]] = cos * l - sin * r
            upd[..., pair[:, 1]] = sin * l + cos * r
            val = upd

        out = conserved + torch.einsum("dk,bnk->bnd", self.nullspace, val)
        return out.view(B, H, W, D)


class ReversiblePonderFunction2D(torch.autograd.Function):
    """Hamiltonian Reversible Pondering Function for 2D Sudoku field with O(1) memory in depth K."""
    @staticmethod
    def forward(ctx, field: torch.Tensor, k_steps: int, transport: UnitaryCayleyTransport2D,
                collision: GivensCollision2D, dt: Optional[torch.Tensor]) -> torch.Tensor:
        ctx.k_steps = k_steps
        ctx.transport = transport
        ctx.collision = collision
        ctx.dt = dt

        curr = field
        B, H, W, D = curr.shape
        n_v = transport.n_v
        d_c = transport.d_c

        with torch.no_grad():
            for _ in range(k_steps):
                # 1. Transport
                f_5d = curr.view(B, H, W, n_v, d_c)
                f_tr = transport(f_5d, dt=dt, inverse=False).view(B, H, W, D)
                # 2. Collision
                curr = collision(f_tr, inverse=False)

        ctx.save_for_backward(curr.detach())
        return curr

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        k_steps = ctx.k_steps
        transport = ctx.transport
        collision = ctx.collision
        dt = ctx.dt
        curr, = ctx.saved_tensors

        B, H, W, D = curr.shape
        n_v = transport.n_v
        d_c = transport.d_c

        grad_f = grad_output.clone()

        # Invert step by step
        for _ in range(k_steps):
            # 1. Invert collision
            with torch.enable_grad():
                f_in_coll = curr.detach().requires_grad_(True)
                f_coll = collision(f_in_coll, inverse=True)
                vjp_coll, = torch.autograd.grad(f_coll, f_in_coll, grad_f, retain_graph=False)

            curr = f_coll.detach()

            # 2. Invert transport
            with torch.enable_grad():
                f_in_tr = curr.view(B, H, W, n_v, d_c).detach().requires_grad_(True)
                f_tr = transport(f_in_tr, dt=dt, inverse=True).view(B, H, W, D)
                vjp_tr, = torch.autograd.grad(f_tr, f_in_tr, vjp_coll, retain_graph=False)

            curr = f_tr.detach()
            grad_f = vjp_tr.view(B, H, W, D)

        return grad_f, None, None, None, None


class CBIMSudokuModel(nn.Module):
    """CBIM Sudoku Solver with continuous kinetic pondering."""
    def __init__(self, vocab_size: int = 11, d_channels: int = 128, n_velocities: int = 8,
                 ponder_steps: int = 6, halt_max_steps: int = 16,
                 multi_step_loss: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d_channels
        self.n_v = n_velocities
        self.d_c = d_channels // n_velocities
        self.ponder_steps = ponder_steps
        self.halt_max_steps = halt_max_steps
        self.multi_step_loss = multi_step_loss

        # 1. Clue Input Embedding
        self.embed_tokens = nn.Embedding(vocab_size, d_channels)
        self.clue_proj = nn.Sequential(
            nn.Linear(d_channels, d_channels),
            nn.SiLU(),
            nn.Linear(d_channels, d_channels)
        )

        # 2. Kinetic Operators
        self.transport = UnitaryCayleyTransport2D(d_channels=d_channels, n_velocities=n_velocities)
        self.collision = GivensCollision2D(d_channels=d_channels, n_layers=2)

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

    def forward_ponder(self, field: torch.Tensor, k: int) -> torch.Tensor:
        """Run K pondering steps with Hamiltonian Reversible Inversion."""
        return ReversiblePonderFunction2D.apply(field, k, self.transport, self.collision, None)

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
                return_keys: list = []) -> Tuple[CBIMSudokuCarry, torch.Tensor, Dict[str, Any], Any, Any]:
        device = batch["inputs"].device
        B = batch["inputs"].shape[0]

        # Handle halted sequences (reset on halt, continue otherwise)
        inputs = torch.where(carry.halted.view(-1, 1), batch["inputs"], carry.current_data["inputs"])
        labels = torch.where(carry.halted.view(-1, 1), batch["labels"], carry.current_data["labels"])
        new_steps = torch.where(carry.halted, 0, carry.steps)

        # 1. Embed clues [B, 81, D] -> [B, 9, 9, D]
        emb = self.embed_tokens(inputs).view(B, 9, 9, self.d)
        clue_field = self.clue_proj(emb)

        # Reset field for halted sequences, retain field for continuing sequences
        prior_field = torch.where(carry.halted.view(B, 1, 1, 1), clue_field, carry.inner_carry.field)

        # 2. Kinetic Pondering: Run K microsteps
        if self.training and self.multi_step_loss:
            curr = prior_field
            step_logits = []
            stride = max(1, self.ponder_steps // 16)
            for k in range(1, self.ponder_steps + 1):
                f_5d = curr.view(B, 9, 9, self.n_v, self.d_c)
                f_tr = self.transport(f_5d).view(B, 9, 9, self.d)
                curr = self.collision(f_tr)
                if k % stride == 0 or k == self.ponder_steps:
                    flat_k = curr.view(B, 81, self.d)
                    logits_k = self.readout_mlp(flat_k)
                    step_logits.append(logits_k)
            evolved_field = curr
            logits = step_logits[-1]
            all_step_logits = step_logits
        else:
            evolved_field = self.forward_ponder(prior_field, self.ponder_steps)
            flat_field = evolved_field.view(B, 81, self.d)
            logits = self.readout_mlp(flat_field)
            all_step_logits = [logits]

        # 3. Q-Halt logits
        flat_field = evolved_field.view(B, 81, self.d)
        q_logits = self.q_head(flat_field[:, 0])  # [B, 2]
        q_halt, q_cont = q_logits[:, 0], q_logits[:, 1]

        # Determine halting condition: Q-halt > 0 or max steps reached
        new_steps = new_steps + 1
        new_halted = (q_halt >= 0) | (new_steps >= self.halt_max_steps)

        # Construct new carry
        new_carry = CBIMSudokuCarry(
            inner_carry=CBIMSudokuInnerCarry(field=evolved_field.detach(), z_H=flat_field.detach()),
            steps=new_steps,
            halted=new_halted,
            current_data={"inputs": inputs, "labels": labels}
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt,
            "q_continue_logits": q_cont,
            "all_step_logits": all_step_logits,
        }
        return new_carry, outputs
