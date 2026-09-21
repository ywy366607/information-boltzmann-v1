"""Controlled Tri-Branch Ablation: Closed Ponder vs Problem-Conditioned Ponder.
Branch A (Closed): F_{k+1} = Collision(Transport(F_k))
Branch B (Conditioned): F_{k+1} = Collision(Transport(F_k); P_t)
   where P_t is the frozen clue representation conditioning collision angles,
   with STRICT ZERO ENERGY INJECTION (||F||^2 identically conserved).
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

import time
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.ib_local.cbim_sudoku import UnitaryCayleyTransport2D, CBIMSudokuCarry, CBIMSudokuInnerCarry
from models.losses import ACTLossHead, IGNORE_LABEL_ID
from adam_atan2 import AdamATan2


class ConditionedGivensCollision2D(nn.Module):
    """Givens Collision operator with optional problem conditioning (P_t).
    Conserves exact unitary norm (||F||^2 is strictly preserved).
    """
    def __init__(self, d_channels: int = 128, n_layers: int = 2, conditioned: bool = True):
        super().__init__()
        self.d = d_channels
        self.layers = n_layers
        self.conditioned = conditioned

        # Nullspace construction
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
        nullspace = right[rank:].T.float()
        if nullspace.shape[1] % 2 != 0:
            nullspace = nullspace[:, :-1]
        self.nullity = nullspace.shape[1]
        self.d_active = self.nullity
        self.register_buffer("nullspace", nullspace)

        # Alternating pairs
        pairs_l0 = torch.stack([torch.arange(0, self.nullity, 2), torch.arange(1, self.nullity, 2)], dim=-1)
        pairs_l1 = torch.stack([
            torch.where(torch.arange(self.nullity // 2) == 0, self.nullity - 1, 2 * torch.arange(self.nullity // 2) - 1),
            2 * torch.arange(self.nullity // 2)
        ], dim=-1)
        self.register_buffer("pairs_l0", pairs_l0)
        self.register_buffer("pairs_l1", pairs_l1)

        # Angle net: takes [state, clue] if conditioned, else [state]
        in_dim = self.d * 2 if conditioned else self.d
        self.angle_net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Linear(128, self.layers * (self.d_active // 2))
        )

    def forward(self, field: torch.Tensor, cond: Optional[torch.Tensor] = None, inverse: bool = False) -> torch.Tensor:
        B, H, W, D = field.shape
        flat = field.view(B, H * W, D)

        coeff = torch.einsum("dk,bnd->bnk", self.nullspace, flat)
        conserved = flat - torch.einsum("dk,bnk->bnd", self.nullspace, coeff)

        if self.conditioned and cond is not None:
            flat_cond = cond.view(B, H * W, D)
            angle_in = torch.cat([flat, flat_cond], dim=-1)
        else:
            angle_in = flat

        angles = self.angle_net(angle_in).view(B, H * W, self.layers, self.d_active // 2)

        val = coeff
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


class ConditionedCBIMSudokuModel(nn.Module):
    """CBIM Sudoku Solver with Branch A (closed) or Branch B (problem-conditioned)."""
    def __init__(self, vocab_size: int = 11, d_channels: int = 128, n_velocities: int = 8,
                 ponder_steps: int = 16, conditioned: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d_channels
        self.n_v = n_velocities
        self.d_c = d_channels // n_velocities
        self.ponder_steps = ponder_steps
        self.halt_max_steps = 16
        self.conditioned = conditioned

        self.embed_tokens = nn.Embedding(vocab_size, d_channels)
        self.clue_proj = nn.Sequential(
            nn.Linear(d_channels, d_channels),
            nn.SiLU(),
            nn.Linear(d_channels, d_channels)
        )

        self.transport = UnitaryCayleyTransport2D(d_channels=d_channels, n_velocities=n_velocities)
        self.collision = ConditionedGivensCollision2D(d_channels=d_channels, n_layers=2, conditioned=conditioned)

        self.readout_mlp = nn.Sequential(
            nn.Linear(d_channels, 256),
            nn.SiLU(),
            nn.Linear(256, vocab_size)
        )
        self.q_head = nn.Linear(d_channels, 2)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)

    def initial_carry(self, batch):
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

    def forward(self, carry, batch, return_keys: list = []):
        device = batch["inputs"].device
        B = batch["inputs"].shape[0]

        inputs = torch.where(carry.halted.view(-1, 1), batch["inputs"], carry.current_data["inputs"])
        labels = torch.where(carry.halted.view(-1, 1), batch["labels"], carry.current_data["labels"])
        new_steps = torch.where(carry.halted, 0, carry.steps)

        emb = self.embed_tokens(inputs).view(B, 9, 9, self.d)
        clue_field = self.clue_proj(emb)  # P_t (frozen problem representation)

        prior_field = torch.where(carry.halted.view(B, 1, 1, 1), clue_field, carry.inner_carry.field)

        # Pondering loop
        curr = prior_field
        step_logits = []
        stride = max(1, self.ponder_steps // 16)

        for k in range(1, self.ponder_steps + 1):
            f_5d = curr.view(B, 9, 9, self.n_v, self.d_c)
            f_tr = self.transport(f_5d).view(B, 9, 9, self.d)
            # In Branch B, collision angles are conditioned on frozen clue_field P_t
            cond = clue_field if self.conditioned else None
            curr = self.collision(f_tr, cond=cond)

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
            "logits": logits, "q_halt_logits": q_halt, "q_continue_logits": q_cont,
            "all_step_logits": step_logits
        }
        return new_carry, outputs


@torch.no_grad()
def evaluate_branch(model, test_in, test_lbl, horizons=[1, 2, 4, 8, 16, 32, 64, 128], batch_size=32):
    model.eval()
    num_samples = len(test_in)
    results = {}

    for k in horizons:
        orig_k = model.model.ponder_steps
        model.model.ponder_steps = k

        total_correct = 0
        total_cells = 0
        total_loss = 0.0

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            B = end - start
            inp = torch.as_tensor(test_in[start:end], dtype=torch.long, device="cuda")
            lbl = torch.as_tensor(test_lbl[start:end], dtype=torch.long, device="cuda")
            batch = {"inputs": inp, "labels": lbl, "puzzle_identifiers": torch.zeros(B, dtype=torch.long, device="cuda")}
            carry = model.model.initial_carry(batch)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                carry, outputs = model.model(carry, batch)
            logits = outputs["logits"].float()
            pred = torch.argmax(logits, dim=-1)

            total_correct += int((pred == lbl).sum().item())
            total_cells += int(lbl.numel())
            loss = F.cross_entropy(logits.view(-1, 11), lbl.view(-1), ignore_index=0)
            total_loss += float(loss.item()) * B

        model.model.ponder_steps = orig_k
        results[k] = {
            "cell_accuracy": total_correct / total_cells,
            "mean_loss": total_loss / num_samples
        }

    return results


def run_ablation(steps: int = 300, batch_size: int = 32, lr: float = 3e-4):
    print("=" * 85)
    print(f"   CONTROLLED ABLATION: CLOSED PONDER (Branch A) vs CONDITIONED PONDER (Branch B)")
    print(f"   Steps: {steps} | Batch: {batch_size} | LR: {lr}")
    print("=" * 85)

    data_dir = "data/sudoku-extreme-1k-aug-100"
    train_in = np.load(os.path.join(data_dir, "train", "all__inputs.npy"), mmap_mode="r")
    train_lbl = np.load(os.path.join(data_dir, "train", "all__labels.npy"), mmap_mode="r")
    test_in = np.load(os.path.join(data_dir, "test", "all__inputs.npy"), mmap_mode="r")
    test_lbl = np.load(os.path.join(data_dir, "test", "all__labels.npy"), mmap_mode="r")
    num_train = len(train_in)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    branches = ["Branch_A_Closed", "Branch_B_Conditioned"]
    ablation_results = {}

    for branch_name in branches:
        is_conditioned = (branch_name == "Branch_B_Conditioned")
        print("\n" + "=" * 80)
        print(f"   TRAINING: {branch_name} (Conditioned: {is_conditioned})")
        print("=" * 80)

        inner = ConditionedCBIMSudokuModel(vocab_size=11, d_channels=128, n_velocities=8,
                                          ponder_steps=16, conditioned=is_conditioned).to(device)
        model = ACTLossHead(inner, loss_type="stablemax_cross_entropy")
        optim = AdamATan2(model.parameters(), lr=lr, weight_decay=1.0)
        scaler = torch.amp.GradScaler("cuda")

        carry = None
        rng = np.random.default_rng(42)
        t0 = time.perf_counter()

        for step in range(1, steps + 1):
            model.train()
            idx = rng.integers(0, num_train, size=batch_size)
            inp = torch.as_tensor(train_in[idx], dtype=torch.long, device=device)
            lbl = torch.as_tensor(train_lbl[idx], dtype=torch.long, device=device)
            batch = {"inputs": inp, "labels": lbl, "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long, device=device)}

            if carry is None:
                carry = model.model.initial_carry(batch)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

            if step % 50 == 0 or step == 1:
                print(f"[{branch_name}] Step {step:3d}/{steps} | Loss: {loss.item():.4f}", flush=True)

        elapsed = (time.perf_counter() - t0) / 60.0
        print(f"{branch_name} trained in {elapsed:.2f} min. Running evaluation sweep...")

        eval_res = evaluate_branch(model, test_in, test_lbl, horizons=[1, 2, 4, 8, 16, 32, 64, 128])
        ablation_results[branch_name] = eval_res

        print(f"\nResults for {branch_name}:")
        for k, r in eval_res.items():
            print(f"  K = {k:3d} | Cell Acc: {r['cell_accuracy']*100:5.2f}% | Loss: {r['mean_loss']:.4f}")

    out_file = Path("results/ablation_problem_conditioning.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    print("\n" + "=" * 85)
    print("   HEAD-TO-HEAD SUMMARY: Branch A (Closed) vs Branch B (Conditioned)")
    print("=" * 85)
    print(f"{'K':>4s} | {'Branch A Acc':>14s} | {'Branch B Acc':>14s} | {'Branch A Loss':>14s} | {'Branch B Loss':>14s}")
    print("-" * 75)
    for k in [1, 2, 4, 8, 16, 32, 64, 128]:
        a = ablation_results["Branch_A_Closed"][k]
        b = ablation_results["Branch_B_Conditioned"][k]
        print(f"{k:4d} | {a['cell_accuracy']*100:13.2f}% | {b['cell_accuracy']*100:13.2f}% | {a['mean_loss']:14.4f} | {b['mean_loss']:14.4f}")


if __name__ == "__main__":
    run_ablation(steps=300, batch_size=32, lr=3e-4)
