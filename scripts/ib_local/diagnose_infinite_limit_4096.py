"""Phase 1: 4-Curve Limit Set Diagnosis (K = 1 .. 4096)
Quantitatively determines the dynamical destiny of CBIM under ultra-deep pondering:
1. State residual: r_F(k) = ||Phi(F_k) - F_k|| / ||F_k||
2. Output residual: r_Y(k) = TotalVariation(R(F_{k+1}), R(F_k))
3. Multi-scale periodic distance: d_m(k) = ||F_k - F_{k-m}|| / ||F_k|| for m in {1, 2, 4, 8, 16, 32, 64, 128, 256}
4. Performance curves: A(k) (Cell Acc), L(k) (Cross-Entropy Loss)

Verdicts:
- Branch A: r_F -> 0 (True state fixed point -> DEQ / Anderson valid)
- Branch B: r_F >> 0, r_Y -> 0 (Quotient-space equilibrium: internal dynamic vortex, static observable)
- Branch C: d_m -> 0 for some m > 1 (Limit cycle / periodic orbit)
- Branch D: Chaotic / NESS attractor
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.ib_local.cbim_sudoku_corrective import CorrectiveCBIMSudokuModel
from models.losses import ACTLossHead


@torch.no_grad()
def run_limit_set_diagnosis(model, test_in, test_lbl, indices, names, out_path: Path, max_k: int = 4096):
    print("=" * 95)
    print(f"   CBIM INFINITE LIMIT SET DIAGNOSIS: K = 1 .. {max_k}")
    print("=" * 95)

    device = next(model.parameters()).device
    m_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    max_m = max(m_values)

    results = {}

    for idx, name in zip(indices, names):
        print(f"\nScanning {name} (idx={idx}) up to K={max_k}...", flush=True)
        inp = torch.as_tensor(test_in[idx:idx+1], dtype=torch.long, device=device)
        lbl = torch.as_tensor(test_lbl[idx:idx+1], dtype=torch.long, device=device)
        lbl_arr = lbl.view(81).cpu().numpy()

        clue_field = model.model.clue_proj(model.model.embed_tokens(inp).view(1, 9, 9, model.model.d))
        curr = clue_field.clone()

        # Ring buffer for state history up to max_m steps
        state_history = {}

        r_F_list = []
        r_Y_list = []
        acc_list = []
        loss_list = []
        d_m_dict = {m: [] for m in m_values}
        steps_list = []

        # Previous output probabilities
        flat_0 = curr.view(1, 81, model.model.d)
        logits_0 = model.model.readout_mlp(flat_0)
        probs_prev = F.softmax(logits_0, dim=-1)

        # Log-linear sampling across K=1..4096 to save memory & plotting overhead
        # Sample dense at start (1..64), then power-of-2 / geometrically up to 4096
        sample_set = set(list(range(1, 65)))
        for k in range(64, max_k + 1, 4):
            if k <= 256 or k % 16 == 0:
                sample_set.add(k)
        sample_set.add(max_k)
        sample_set = sorted(list(sample_set))

        for k in range(1, max_k + 1):
            f_prev = curr.clone()

            # CBIM Forward step (Arm C: Corrective Flow)
            f_5d = curr.view(1, 9, 9, model.model.n_v, model.model.d_c)
            f_tr = model.model.transport(f_5d).view(1, 9, 9, model.model.d)
            f_star = model.model.collision(f_tr, cond=clue_field)

            cat_in = torch.cat([f_star, clue_field], dim=-1)
            v_k = model.model.corrective_net(cat_in)
            dot_p = torch.sum(f_star.float()*v_k.float(), dim=(1,2,3), keepdim=True)
            norm_sq = torch.sum(f_star.float()**2, dim=(1,2,3), keepdim=True) + 1e-8
            u_k = v_k.float() - (dot_p/norm_sq)*f_star.float()
            u_hat = u_k / (torch.linalg.vector_norm(u_k, dim=(1,2,3), keepdim=True) + 1e-8)
            alpha_k = model.model.alpha_max * torch.sigmoid(torch.mean(model.model.angle_gate(cat_in), dim=(1,2,3), keepdim=True)).to(curr.dtype)
            f_norm = torch.linalg.vector_norm(f_star.float(), dim=(1,2,3), keepdim=True)
            curr = torch.cos(alpha_k)*f_star + f_norm*torch.sin(alpha_k)*u_hat.to(curr.dtype)

            # Store in ring buffer
            state_history[k] = curr.clone()
            if len(state_history) > max_m + 10:
                oldest_key = min(state_history.keys())
                del state_history[oldest_key]

            if k in sample_set:
                # 1. State residual: r_F(k) = ||F_{k} - F_{k-1}|| / ||F_k||
                diff_F = torch.linalg.vector_norm((curr - f_prev).float()).item()
                norm_F = torch.linalg.vector_norm(curr.float()).item() + 1e-8
                r_F = diff_F / norm_F

                # 2. Output residual: r_Y(k) = TotalVariation(probs_k, probs_{k-1})
                flat_k = curr.view(1, 81, model.model.d)
                logits_k = model.model.readout_mlp(flat_k)
                probs_k = F.softmax(logits_k, dim=-1)
                tv_dist = 0.5 * torch.sum(torch.abs(probs_k - probs_prev), dim=-1).mean().item()
                probs_prev = probs_k.clone()

                # 3. Accuracy & Loss
                pred_digits = torch.argmax(probs_k, dim=-1).view(81).cpu().numpy()
                cell_acc = float(np.mean(pred_digits == lbl_arr))
                ce_loss = F.cross_entropy(logits_k.view(81, 11), lbl.view(81)).item()

                # 4. Multi-scale periodic residual: d_m(k) = ||F_k - F_{k-m}|| / ||F_k||
                for m in m_values:
                    if (k - m) in state_history:
                        diff_m = torch.linalg.vector_norm((curr - state_history[k - m]).float()).item()
                        d_m_dict[m].append(diff_m / norm_F)
                    else:
                        d_m_dict[m].append(np.nan)

                r_F_list.append(r_F)
                r_Y_list.append(tv_dist)
                acc_list.append(cell_acc)
                loss_list.append(ce_loss)
                steps_list.append(k)

                if k in [1, 16, 64, 256, 1024, 4096]:
                    print(f"  [K={k:04d}] r_F: {r_F:.6e} | r_Y: {tv_dist:.6e} | Cell Acc: {cell_acc*100:5.2f}% | Loss: {ce_loss:.4f} | d_1: {d_m_dict[1][-1]:.4e} | d_64: {d_m_dict[64][-1]:.4e}", flush=True)

        results[name] = {
            "steps": steps_list,
            "r_F": r_F_list,
            "r_Y": r_Y_list,
            "acc": acc_list,
            "loss": loss_list,
            "d_m": d_m_dict
        }

    # Plot 4-Panel Master Diagnostic Figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#060614")
    fig.suptitle(f"CBIM Infinite Limit Set Dynamical Diagnosis (K = 1 .. {max_k})", fontsize=18, color="#e0e8ff", y=0.98, fontweight="bold")

    colors = {"Easy 35-Clue (idx 909)": "#38d8b8", "Medium 25-Clue (idx 248)": "#2080f0", "Hard 17-Clue (idx 122)": "#ff4488"}

    # Panel 1: State Residual r_F(k)
    ax1 = axes[0, 0]
    ax1.set_facecolor("#080a1c")
    for name, res in results.items():
        ax1.plot(res["steps"], res["r_F"], color=colors[name], linewidth=2.0, label=f"{name}")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title("1. State Residual $r_F(k) = \\|\\Phi(F_k) - F_k\\| / \\|F_k\\|$", color="#d0e0ff", fontsize=13)
    ax1.set_xlabel("Pondering Microstep $K$ (log scale)", color="#90a8d0")
    ax1.set_ylabel("$r_F(k)$ (log scale)", color="#90a8d0")
    ax1.grid(True, which="both", color="#182850", alpha=0.6)
    ax1.legend(facecolor="#080a1c", edgecolor="#203560", labelcolor="#e0e8ff")

    # Panel 2: Output Residual r_Y(k)
    ax2 = axes[0, 1]
    ax2.set_facecolor("#080a1c")
    for name, res in results.items():
        ax2.plot(res["steps"], res["r_Y"], color=colors[name], linewidth=2.0, label=f"{name}")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title("2. Output Residual $r_Y(k) = \\mathrm{TV}(R(F_k), R(F_{k-1}))$", color="#d0e0ff", fontsize=13)
    ax2.set_xlabel("Pondering Microstep $K$ (log scale)", color="#90a8d0")
    ax2.set_ylabel("$r_Y(k)$ (log scale)", color="#90a8d0")
    ax2.grid(True, which="both", color="#182850", alpha=0.6)
    ax2.legend(facecolor="#080a1c", edgecolor="#203560", labelcolor="#e0e8ff")

    # Panel 3: Multi-Scale Periodicity d_m(k) for Hard Puzzle
    ax3 = axes[1, 0]
    ax3.set_facecolor("#080a1c")
    hard_res = results["Hard 17-Clue (idx 122)"]
    m_colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(m_values)))
    for idx_m, m in enumerate([1, 2, 4, 8, 16, 32, 64, 128, 256]):
        ax3.plot(hard_res["steps"], hard_res["d_m"][m], color=m_colors[idx_m], linewidth=1.8, label=f"m={m}")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_title("3. Multi-Scale Periodic Distance $d_m(k) = \\|F_k - F_{k-m}\\| / \\|F_k\\|$ (Hard)", color="#d0e0ff", fontsize=13)
    ax3.set_xlabel("Pondering Microstep $K$ (log scale)", color="#90a8d0")
    ax3.set_ylabel("$d_m(k)$ (log scale)", color="#90a8d0")
    ax3.grid(True, which="both", color="#182850", alpha=0.6)
    ax3.legend(facecolor="#080a1c", edgecolor="#203560", labelcolor="#e0e8ff", ncol=3, fontsize=9)

    # Panel 4: Task Performance (Cell Acc & Loss)
    ax4 = axes[1, 1]
    ax4.set_facecolor("#080a1c")
    for name, res in results.items():
        ax4.plot(res["steps"], [a*100 for a in res["acc"]], color=colors[name], linewidth=2.0, label=f"{name} (Acc %)")
    ax4.set_xscale("log")
    ax4.set_title("4. Task Performance Evolution (Cell Accuracy %)", color="#d0e0ff", fontsize=13)
    ax4.set_xlabel("Pondering Microstep $K$ (log scale)", color="#90a8d0")
    ax4.set_ylabel("Cell Accuracy (%)", color="#90a8d0")
    ax4.set_ylim(20, 100)
    ax4.grid(True, which="both", color="#182850", alpha=0.6)
    ax4.legend(facecolor="#080a1c", edgecolor="#203560", labelcolor="#e0e8ff")

    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(colors="#7088b0")
        for spine in ax.spines.values():
            spine.set_color("#203560")

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    plt.savefig(out_path, dpi=120, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"\nSaved master diagnostic plot to {out_path}!", flush=True)

    # Quantitative Verdict Summary
    print("\n" + "=" * 95)
    print("   QUANTITATIVE VERDICT ON THE INFINITE LIMIT SET")
    print("=" * 95)
    for name, res in results.items():
        final_rf = res["r_F"][-1]
        final_ry = res["r_Y"][-1]
        final_acc = res["acc"][-1]
        final_loss = res["loss"][-1]
        print(f"\n--- {name} at K={max_k} ---")
        print(f"  State Residual r_F:   {final_rf:.6e}")
        print(f"  Output Residual r_Y:  {final_ry:.6e}")
        print(f"  Cell Accuracy:        {final_acc*100:.2f}%")
        print(f"  Cross-Entropy Loss:   {final_loss:.4f}")

        if final_rf < 1e-4:
            verdict = "Branch A (True Fixed Point: State F_k is static, DEQ/Anderson valid)"
        elif final_ry < 1e-5 and final_rf >= 1e-4:
            verdict = "Branch B (Quotient-Space Equilibrium: Internal dynamic rotation/vortex, Observable Readout is static!)"
        else:
            verdict = "Branch D (Non-convergent / Chaotic / NESS attractor)"
        print(f"  >> Verdict: {verdict}")


def main():
    out_dir = Path("present")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cbim_infinite_limit_4096_diagnosis.png"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("results/arm_c_champion_3000/Best_Arm_C_Champion_d256.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    inner = CorrectiveCBIMSudokuModel(vocab_size=11, d_channels=256, ponder_steps=256, arm="corrective_flow").to(device)
    head = ACTLossHead(inner, loss_type="stablemax_cross_entropy").to(device)
    head.load_state_dict(ckpt["model"])
    model = head
    model.eval()

    test_in = np.load("data/sudoku-extreme-1k-aug-100/test/all__inputs.npy")
    test_lbl = np.load("data/sudoku-extreme-1k-aug-100/test/all__labels.npy")

    indices = [909, 248, 122]
    names = ["Easy 35-Clue (idx 909)", "Medium 25-Clue (idx 248)", "Hard 17-Clue (idx 122)"]

    run_limit_set_diagnosis(model, test_in, test_lbl, indices, names, out_path, max_k=4096)


if __name__ == "__main__":
    main()
