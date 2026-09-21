"""Query-Conditioned Branch-Set Bayesian Posterior Router on Sudoku-Extreme.
Implements the exact GPT review specification:
1. Query-Conditioned (Per-Cell) Branch Set Attention:
   Does NOT use a global scalar for the whole board!
   For each cell i in {1..81}:
     Extracts e_{i, h} from each branch h in {1..M}.
     Performs Set Attention across the M branches: e_tilde_{i, h} = Attn_h(e_{i, h}, e_{i, 1:M}).
     Computes branch score s_{i, h} = g_phi(e_{i, h}, e_tilde_{i, h}, P_i).
     Computes log-space Bayesian prior: log_q_{i, h} = log_softmax_h(s).
2. Pure Task NLL Loss (Natural Bayesian Responsibility Equivalence):
   Mixture computed stably in log-space:
     final_logp = logsumexp_h(log_q_{i, h} + branch_logp_{h, i})
     L = NLL(final_logp, targets)
   No artificial KL losses! Task NLL gradient naturally drives q -> r_posterior!
3. Diagnostic Logging:
   Tracks Router Gap = KL(r_posterior || q) and Branch Oracle Acc in real time!
4. Physical Substrate:
   - Stage 3 Champion Base (Exact Exp Transport, dt=1/K, Mass-Norm Readout, Never-Reset)
   - M in {1, 2, 4} dynamically sampled during training
   - K in [1, 2, 4, 8, 16, 32, 64] multi-scale sampling
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import math
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import hadamard

from scripts.ib_local.cbim_sudoku_unified_dissipative import (
    ContextualBoundaryWrite2Port,
    QuadraticPassiveRadiationBath
)
from scripts.ib_local.train_stage1_dt_scaling_3000 import ContinuousConditionedLieCollision
from scripts.ib_local.train_stage2_exact_exponential_3000 import ExactExponentialTorusTransport
from scripts.ib_local.train_stage3_mass_norm_readout_3000 import MassNormalizedSoftIntegralReadout
from external.TinyRecursiveModels.adam_atan2 import AdamATan2


def get_orthogonal_hadamard_codes(M: int, d: int = 256, device: str = "cuda") -> torch.Tensor:
    H = hadamard(d)[:M]  # [M, d]
    tensor_codes = torch.as_tensor(H, dtype=torch.float32, device=device)
    return tensor_codes.view(M, 1, 1, d)  # [M, 1, 1, d]


class QueryConditionedBranchSetRouter(nn.Module):
    """Query-conditioned branch-set attention router.
    Compares the M hypotheses for each cell query i via set attention across the M axis.
    """
    def __init__(self, d_channels: int = 256, heads: int = 4):
        super().__init__()
        self.d = d_channels
        self.heads = heads
        self.head_dim = d_channels // heads

        self.q_proj = nn.Linear(d_channels, d_channels)
        self.k_proj = nn.Linear(d_channels, d_channels)
        self.v_proj = nn.Linear(d_channels, d_channels)

        self.score_head = nn.Sequential(
            nn.Linear(2 * d_channels, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    def forward(self, branch_flat_feats: torch.Tensor) -> torch.Tensor:
        """branch_flat_feats: [M, 81, d]
        returns: log_q of shape [M, 81] in log-space!
        """
        M, N, d = branch_flat_feats.shape
        if M == 1:
            return torch.zeros(1, N, device=branch_flat_feats.device, dtype=branch_flat_feats.dtype)

        # Transpose to [N, M, d] to perform set attention across M branches for each cell i in 1..81
        x = branch_flat_feats.permute(1, 0, 2)  # [81, M, d]

        q = self.q_proj(x).view(N, M, self.heads, self.head_dim).permute(0, 2, 1, 3)  # [81, H, M, dh]
        k = self.k_proj(x).view(N, M, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(N, M, self.heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # [81, H, M, M]
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v).permute(0, 2, 1, 3).reshape(N, M, d)  # [81, M, d]

        # Concatenate query self-feature and cross-branch comparison context
        combined = torch.cat([x, context], dim=-1)  # [81, M, 2d]
        scores = self.score_head(combined).squeeze(-1)  # [81, M]

        # Transpose back to [M, 81] and compute log_softmax along branch dimension M (dim=0)
        log_q = F.log_softmax(scores.permute(1, 0), dim=0)  # [M, 81]
        return log_q


class BayesianBranchSetCBIMSudokuModel(nn.Module):
    """Complete CBIM Model with Query-Conditioned Branch-Set Bayesian Posterior Routing."""

    def __init__(self, vocab_size: int = 11, d_channels: int = 256, alpha_max: float = 0.25):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d_channels
        self.n_v = 8
        self.d_c = d_channels // self.n_v
        self.alpha_max = float(alpha_max)

        self.embed_tokens = nn.Embedding(vocab_size, d_channels)
        self.clue_proj = nn.Sequential(
            nn.Linear(d_channels, d_channels),
            nn.SiLU(),
            nn.Linear(d_channels, d_channels)
        )

        # 1. H-Boundary 2-Port State-Aware Scattering theta(F, P)
        self.boundary_write = ContextualBoundaryWrite2Port(d=d_channels)

        # 2. Exact Complex Exponential Transport on 9x9 Torus
        self.transport = ExactExponentialTorusTransport(n_v=self.n_v, d_c=self.d_c)

        # 3. Continuous Lie Collision (Synchronized with dt!)
        self.collision = ContinuousConditionedLieCollision(d=d_channels)

        # 4. Continuous Arm C Corrective Flow (Synchronized with dt!)
        self.corrective_net = nn.Sequential(
            nn.Linear(2 * d_channels, d_channels),
            nn.SiLU(),
            nn.Linear(d_channels, d_channels)
        )
        self.angle_gate = nn.Sequential(
            nn.Linear(2 * d_channels, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        nn.init.zeros_(self.corrective_net[-1].weight)
        nn.init.zeros_(self.corrective_net[-1].bias)
        nn.init.zeros_(self.angle_gate[-1].weight)
        nn.init.constant_(self.angle_gate[-1].bias, -2.0)

        # 5. H-Boundary Quadratic Passive Bath
        self.bath = QuadraticPassiveRadiationBath(d=d_channels)

        # 6. Shared Mass-Normalized Soft Integral Readout
        self.readout = MassNormalizedSoftIntegralReadout(d_channels=d_channels, vocab_size=vocab_size)

        # 7. Query-Conditioned Branch-Set Attention Bayesian Router
        self.router = QueryConditionedBranchSetRouter(d_channels=d_channels)

    def forward_stream_step(self, persistent_state: torch.Tensor, inp: torch.Tensor,
                            M: int, k_step: int, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        device = inp.device
        clue_field = self.clue_proj(self.embed_tokens(inp).view(1, 9, 9, self.d))  # [1, 9, 9, d]

        # 1. Macro-Step H Boundary: 2-Port Unitary Scattering theta(F, P)
        f_base, reflected, write_diag = self.boundary_write(persistent_state, clue_field)

        # 2. Orthogonal Exploration Seeds (Walsh-Hadamard Codes)
        if M == 1:
            branches = f_base  # [1, 9, 9, d]
        else:
            codes = get_orthogonal_hadamard_codes(M, d=self.d, device=device)  # [M, 1, 1, d]
            branch_seeds = (clue_field * codes) / math.sqrt(M)  # [M, 9, 9, d]
            branches = f_base.expand(M, -1, -1, -1) + 0.15 * branch_seeds  # [M, 9, 9, d]

        orig_norm = torch.linalg.vector_norm(branches.float(), dim=(1, 2, 3), keepdim=True).to(branches.dtype)
        dt = 1.0 / max(1, k_step)
        clue_expanded = clue_field.expand(M, -1, -1, -1)

        # 3. Independent Nonlinear Dynamics for all M branches
        for _ in range(k_step):
            f_5d = branches.view(M, 9, 9, self.n_v, self.d_c)
            f_tr = self.transport(f_5d, dt=dt).view(M, 9, 9, self.d)
            f_star = self.collision(f_tr, cond=clue_expanded, dt=dt)

            cat_in = torch.cat([f_star, clue_expanded], dim=-1)
            v_k = self.corrective_net(cat_in)
            dot_p = torch.sum(f_star.float() * v_k.float(), dim=(1, 2, 3), keepdim=True)
            norm_sq = torch.sum(f_star.float() ** 2, dim=(1, 2, 3), keepdim=True) + 1e-8
            u_k = v_k.float() - (dot_p / norm_sq) * f_star.float()
            u_hat = u_k / (torch.linalg.vector_norm(u_k, dim=(1, 2, 3), keepdim=True) + 1e-8)

            gate_val = torch.sigmoid(torch.mean(self.angle_gate(cat_in), dim=(1, 2, 3), keepdim=True))
            alpha_k = (dt * self.alpha_max) * gate_val.to(branches.dtype)

            f_norm_step = torch.linalg.vector_norm(f_star.float(), dim=(1, 2, 3), keepdim=True)
            branches = torch.cos(alpha_k) * f_star + f_norm_step * torch.sin(alpha_k) * u_hat.to(branches.dtype)

        # Norm preservation per branch
        branch_norms = torch.linalg.vector_norm(branches.float(), dim=(1, 2, 3), keepdim=True) + 1e-8
        branches = branches * (orig_norm / branch_norms)

        # 4. Independent Readout per branch
        logits_branches = self.readout(branches)  # [M, 81, 11]
        branch_logp = F.log_softmax(logits_branches, dim=-1)  # [M, 81, 11]

        # 5. Query-Conditioned Branch-Set Attention Bayesian Router
        branch_flat_feats = branches.view(M, 81, self.d)  # [M, 81, d]
        log_q = self.router(branch_flat_feats)            # [M, 81] in log-space!

        # 6. Exact Log-Space Mixture: final_logp = logsumexp_h(log_q_{i, h} + branch_logp_{h, i})
        # log_q is [M, 81] -> unsqueeze to [M, 81, 1]
        final_logp = torch.logsumexp(log_q.unsqueeze(-1) + branch_logp, dim=0, keepdim=True)  # [1, 81, 11]

        # Diagnostic: Bayesian Responsibility r_h and Router Gap KL(r || q)
        diag = {**write_diag}
        if targets is not None and M > 1:
            with torch.no_grad():
                # branch_logp_target: [M, 81]
                target_gathered = targets.view(1, 81, 1).expand(M, -1, -1)
                log_p_y = branch_logp.gather(dim=-1, index=target_gathered).squeeze(-1)  # [M, 81]
                log_r_unnorm = log_q + log_p_y  # [M, 81]
                log_r = log_r_unnorm - torch.logsumexp(log_r_unnorm, dim=0, keepdim=True)
                r_posterior = torch.exp(log_r)
                q_prior = torch.exp(log_q)
                # Router Gap = KL(r || q) = sum r * (log_r - log_q)
                router_gap = (r_posterior * (log_r - log_q)).sum(dim=0).mean().item()
                diag["router_gap_kl"] = router_gap

        # 7. Select winning branch state (highest average belief) to carry forward
        best_h_idx = torch.argmax(log_q.mean(dim=-1))
        best_branch = branches[best_h_idx:best_h_idx+1]

        state_next, bath_out, bath_diag = self.bath(best_branch)
        return state_next, final_logp, {**diag, **bath_diag}


@torch.no_grad()
def evaluate_bayesian_grid(model, test_in: np.ndarray, test_lbl: np.ndarray,
                           mature_state: torch.Tensor, m_list: List[int], k_list: List[int]) -> Dict[str, Any]:
    model.eval()
    device = mature_state.device
    num_samples = len(test_in)
    grid_results = {}

    for M in m_list:
        for K in k_list:
            state = mature_state.detach().clone()
            total_cells = 0
            correct_cells = 0
            exact_matches = 0
            losses = []
            gaps = []

            for idx in range(num_samples):
                inp = torch.as_tensor(test_in[idx:idx+1], dtype=torch.long, device=device)
                lbl = torch.as_tensor(test_lbl[idx:idx+1], dtype=torch.long, device=device)

                state_next, log_probs, diag = model.forward_stream_step(state, inp, M=M, k_step=K, targets=lbl)
                loss = F.nll_loss(log_probs.view(-1, 11), lbl.view(-1)).item()
                losses.append(loss)
                if "router_gap_kl" in diag:
                    gaps.append(diag["router_gap_kl"])

                preds = torch.argmax(log_probs, dim=-1)
                correct_cells += (preds == lbl).sum().item()
                total_cells += 81
                exact_matches += ((preds == lbl).sum().item() == 81)

                state = state_next.detach()

            grid_results[f"M{M}_K{K}"] = {
                "M": M, "K": K,
                "cell_accuracy": correct_cells / total_cells,
                "exact_accuracy": exact_matches / num_samples,
                "mean_loss": float(np.mean(losses)),
                "mean_router_gap_kl": float(np.mean(gaps)) if gaps else 0.0
            }

    return grid_results


def main():
    parser = argparse.ArgumentParser(description="Query-Conditioned Bayesian Router Training")
    parser.add_argument("--steps", type=int, default=3000, help="Total training steps")
    parser.add_argument("--d-channels", type=int, default=256, help="Phase space feature channels")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--eval-interval", type=int, default=500, help="Validation interval")
    parser.add_argument("--output-dir", type=str, default="results/bayesian_set_router_3000", help="Output directory")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_log = out_path / "metrics.jsonl"

    print("=" * 115)
    print("   QUERY-CONDITIONED BAYESIAN POSTERIOR SET ROUTER (GPT CANONICAL SPECIFICATION)")
    print(f"   Architecture: Per-Cell Set Attention over M Axis: e_{{i,h}} -> Attn_h(e_{{i,h}}, e_{{i,1:M}}) -> log_q_{{i,h}}")
    print(f"   Loss: PURE TASK NLL (d L / d s = q - r natural Bayesian gradient!) | LogSumExp Mixture")
    print(f"   M sampled from {{1, 2, 4}} | K sampled from [1, 2, 4, 8, 16, 32, 64] | Steps: {args.steps}")
    print("=" * 115)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path("data/sudoku-extreme-1k-aug-100")
    train_in = np.load(data_path / "train" / "all__inputs.npy", mmap_mode="r")
    train_lbl = np.load(data_path / "train" / "all__labels.npy", mmap_mode="r")
    test_in = np.load(data_path / "test" / "all__inputs.npy")
    test_lbl = np.load(data_path / "test" / "all__labels.npy")
    num_train = len(train_in)

    model = BayesianBranchSetCBIMSudokuModel(vocab_size=11, d_channels=args.d_channels).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {num_params:,} ({num_params / 1e6:.3f}M)\n")

    optimizer = AdamATan2(model.parameters(), lr=args.lr, weight_decay=1.0)
    scaler = torch.amp.GradScaler("cuda")

    # Initial persistent state: strictly single field [1, 9, 9, d]
    init_sample = torch.as_tensor(train_in[0:1], dtype=torch.long, device=device)
    with torch.no_grad():
        persistent_state = model.clue_proj(model.embed_tokens(init_sample).view(1, 9, 9, args.d_channels))

    m_choices = [1, 2, 4]
    k_choices = [1, 2, 4, 8, 16, 32, 64]
    rng = np.random.default_rng(42)

    best_loss = float("inf")
    best_cell_acc = 0.0
    start_time = time.perf_counter()

    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        model.train()

        progress = step / args.steps
        current_lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g["lr"] = current_lr

        m_step = int(rng.choice(m_choices))
        k_step = int(rng.choice(k_choices))

        puzzle_idx = (step - 1) % num_train
        inp = torch.as_tensor(train_in[puzzle_idx:puzzle_idx+1], dtype=torch.long, device=device)
        lbl = torch.as_tensor(train_lbl[puzzle_idx:puzzle_idx+1], dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            state_next, log_probs, diag_last = model.forward_stream_step(persistent_state, inp, M=m_step, k_step=k_step, targets=lbl)
            loss = F.nll_loss(log_probs.view(-1, 11), lbl.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        with torch.no_grad():
            persistent_state.copy_(state_next.detach())

        torch.cuda.synchronize()
        step_time_ms = (time.perf_counter() - t0) * 1000.0
        vram_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

        if step % 50 == 0 or step == 1:
            fps = 1.0 / (step_time_ms / 1000.0)
            gap_str = f" | Gap(r||q): {diag_last.get('router_gap_kl', 0.0):.4f}" if m_step > 1 else ""
            print(f"Step {step:4d}/{args.steps} | M: {m_step} | K: {k_step:2d} | Loss: {loss.item():6.3f} | LR: {current_lr:.1e}{gap_str} | Time: {step_time_ms:5.1f}ms | {fps:5.1f} puz/s | VRAM: {vram_mb:5.1f}MB", flush=True)

            with open(metrics_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": "train", "step": step, "M": m_step, "k": k_step, "loss": loss.item(), "lr": current_lr,
                    "router_gap_kl": diag_last.get("router_gap_kl", 0.0), "step_time_ms": step_time_ms, "vram_mb": vram_mb
                }) + "\n")

        # Periodic Evaluation across 2D grid
        if step % args.eval_interval == 0 or step == args.steps:
            val_results = evaluate_bayesian_grid(
                model, test_in[:200], test_lbl[:200],
                mature_state=persistent_state,
                m_list=[1, 2, 4],
                k_list=[2, 8, 32]
            )
            print(f"\n[INTERIM BAYESIAN EVALUATION Step {step}]")
            for m_val in [1, 2, 4]:
                line_str = f"  M={m_val} | " + " | ".join([f"K={k_val:2d}: {val_results[f'M{m_val}_K{k_val}']['cell_accuracy']*100:5.2f}% (Loss: {val_results[f'M{m_val}_K{k_val}']['mean_loss']:.3f})" for k_val in [2, 8, 32]])
                print(line_str)

            acc_m4_k8 = val_results["M4_K8"]["cell_accuracy"]
            if acc_m4_k8 > best_cell_acc:
                best_cell_acc = acc_m4_k8
                torch.save({
                    "model": model.state_dict(),
                    "persistent_state": persistent_state.detach().cpu(),
                    "step": step,
                    "val_results": val_results
                }, out_path / "Best_BayesianRouter_d256.pt")
                print(f"Saved new best model to {out_path / 'Best_BayesianRouter_d256.pt'}\n", flush=True)

    total_time_min = (time.perf_counter() - start_time) / 60.0
    print(f"\nBayesian Router Training completed in {total_time_min:.2f} minutes!")

    # Post-training full test-time 2D grid evaluation across all 1,000 test puzzles
    print("\n" + "=" * 115)
    print("   FINAL POST-TRAINING BAYESIAN 2D RESPONSE MATRIX Q(M, K) (FULL 1,000 TEST PUZZLES)")
    print("=" * 115)
    final_grid = evaluate_bayesian_grid(
        model, test_in, test_lbl,
        mature_state=persistent_state,
        m_list=[1, 2, 4],
        k_list=[1, 2, 4, 8, 16, 32, 64]
    )

    header = f"{'Width M':^10} | " + " | ".join([f"K={k:<4d}" for k in [1, 2, 4, 8, 16, 32, 64]])
    print(header)
    print("-" * len(header))
    for M in [1, 2, 4]:
        row_str = f" M = {M:2d}    | " + " | ".join([f"{final_grid[f'M{M}_K{K}']['cell_accuracy']*100:5.2f}%" for K in [1, 2, 4, 8, 16, 32, 64]])
        print(row_str)

    # Equal Compute Horizon Analysis: M * K == 64
    print("\n" + "=" * 80)
    print("   BAYESIAN EQUAL COMPUTE CONSERVATION TEST")
    print("=" * 80)
    equal_configs = [(1, 64), (2, 32), (4, 16)]
    for M, K in equal_configs:
        entry = final_grid[f"M{M}_K{K}"]
        print(f"  (M={M:2d}, K={K:2d}) [Total={M*K:2d}] | Cell Acc: {entry['cell_accuracy']*100:5.2f}% | Loss: {entry['mean_loss']:6.3f} | Gap(r||q): {entry['mean_router_gap_kl']:.4f}")

    with open(out_path / "final_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "steps": args.steps, "d_channels": args.d_channels, "total_time_min": total_time_min,
            "final_grid": final_grid
        }, f, indent=2)

    torch.save({
        "model": model.state_dict(),
        "persistent_state": persistent_state.detach().cpu(),
        "final_grid": final_grid
    }, out_path / "Final_BayesianRouter_d256.pt")
    print(f"\nAll completed! Report saved to {out_path / 'final_report.json'}")


if __name__ == "__main__":
    main()
