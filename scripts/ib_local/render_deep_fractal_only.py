"""Render Deep Fractal Basin Multi-Generation Folding Animation (K=1..64)
with chunked streaming readout (batch_size=512) for minimal VRAM footprint (~40MB).
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
from PIL import Image
import io

from scripts.ib_local.cbim_sudoku_corrective import CorrectiveCBIMSudokuModel
from models.losses import ACTLossHead
from scripts.ib_local.render_true_fractal_basins import find_conflicting_pair


@torch.no_grad()
def generate_deep_fractal_basin_animation(model, inp, out_dir: Path, max_k: int = 64, res: int = 80, radius: float = 2.5):
    print(f"\nGenerating Deep Multi-Generation Fractal Basin Folding (K=1..{max_k}, res={res})...", flush=True)
    clue_field = model.model.clue_proj(model.model.embed_tokens(inp).view(1, 9, 9, model.model.d))
    f_0 = clue_field.clone()
    orig_norm = torch.linalg.vector_norm(f_0.float(), dim=(1,2,3), keepdim=True)

    logits_0 = model.model.readout_mlp(f_0.view(1, 81, model.model.d))[0]
    c1, c2, cand_d = find_conflicting_pair(inp, logits_0)
    r1, col1 = c1 // 9, c1 % 9
    r2, col2 = c2 // 9, c2 % 9

    v1 = torch.zeros_like(f_0)
    v2 = torch.zeros_like(f_0)
    emb_d = model.model.clue_proj(model.model.embed_tokens(torch.tensor([cand_d], device=inp.device)).view(1,1,1,model.model.d))
    v1[0, r1, col1] = emb_d[0, 0, 0]
    v2[0, r2, col2] = emb_d[0, 0, 0]
    v1 = v1 / torch.linalg.vector_norm(v1.float(), dim=(1,2,3), keepdim=True)
    v2 = v2 / torch.linalg.vector_norm(v2.float(), dim=(1,2,3), keepdim=True)

    u_vals = np.linspace(-radius, radius, res)
    v_vals = np.linspace(-radius, radius, res)
    U, V = np.meshgrid(u_vals, v_vals)
    u_f = torch.as_tensor(U.ravel(), dtype=torch.float32, device=inp.device).view(-1, 1, 1, 1)
    v_f = torch.as_tensor(V.ravel(), dtype=torch.float32, device=inp.device).view(-1, 1, 1, 1)
    total_pts = len(u_f)

    frames = []
    batch_sz = 256
    p0_all = f_0 + u_f * v1 + v_f * v2
    p0_all = p0_all * (orig_norm / (torch.linalg.vector_norm(p0_all.float(), dim=(1,2,3), keepdim=True) + 1e-8))

    curr = p0_all.clone()
    cond = clue_field.expand(total_pts, -1, -1, -1)

    # Sample 35 frames across K=1..64
    sampled_k = sorted(list(set(
        list(range(1, 17)) +
        [18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
    )))

    for k in range(1, max_k + 1):
        for s in range(0, total_pts, batch_sz):
            e = min(s + batch_sz, total_pts)
            b = e - s
            cb = curr[s:e]
            cd = cond[s:e]

            f_5d = cb.view(b, 9, 9, model.model.n_v, model.model.d_c)
            f_tr = model.model.transport(f_5d).view(b, 9, 9, model.model.d)
            f_star = model.model.collision(f_tr, cond=cd)
            cat_in = torch.cat([f_star, cd], dim=-1)
            v_k = model.model.corrective_net(cat_in)
            dot_p = torch.sum(f_star.float()*v_k.float(), dim=(1,2,3), keepdim=True)
            norm_sq = torch.sum(f_star.float()**2, dim=(1,2,3), keepdim=True) + 1e-8
            u_k = v_k.float() - (dot_p/norm_sq)*f_star.float()
            u_hat = u_k / (torch.linalg.vector_norm(u_k, dim=(1,2,3), keepdim=True) + 1e-8)
            alpha_k = model.model.alpha_max * torch.sigmoid(torch.mean(model.model.angle_gate(cat_in), dim=(1,2,3), keepdim=True)).to(curr.dtype)
            f_norm = torch.linalg.vector_norm(f_star.float(), dim=(1,2,3), keepdim=True)
            curr[s:e] = torch.cos(alpha_k)*f_star + f_norm*torch.sin(alpha_k)*u_hat.to(curr.dtype)

        if k in sampled_k:
            # Chunked streaming readout with batch size 512 (~40MB VRAM)
            diff_prob_list = []
            readout_bs = 512
            for rs in range(0, total_pts, readout_bs):
                re = min(rs + readout_bs, total_pts)
                flat_chunk = curr[rs:re].view(re - rs, 81, model.model.d)
                logits_chunk = model.model.readout_mlp(flat_chunk)
                probs_c1_chunk = F.softmax(logits_chunk[:, c1], dim=-1)[:, cand_d]
                probs_c2_chunk = F.softmax(logits_chunk[:, c2], dim=-1)[:, cand_d]
                diff_prob_list.append((probs_c1_chunk - probs_c2_chunk).cpu().numpy())
            diff_prob = np.concatenate(diff_prob_list, axis=0).reshape(res, res)

            fig, ax = plt.subplots(figsize=(7, 7), facecolor="#060614")
            fig.suptitle(f"Deep Phase-Space Shear & Folding (Time Axis K = {k:02d} / {max_k})", fontsize=15, color="#e0e8ff", y=0.96, fontweight="bold")
            ax.set_facecolor("#020208")

            im = ax.imshow(diff_prob, extent=[-radius, radius, -radius, radius], origin="lower", cmap="coolwarm", vmin=-0.8, vmax=0.8, interpolation="bicubic")
            ax.contour(U, V, diff_prob, levels=7, colors="#ffffff", alpha=0.3, linewidths=0.8)

            ax.set_title(f"Clash: Cell({r1},{col1}) vs Cell({r2},{col2}) | Microstep {k}", color="#d0e0ff", fontsize=12, pad=8)
            ax.set_xlabel("Perturbation $u \cdot \mathbf{v}_1$ (Cell 1)", color="#90a8d0", fontsize=11)
            ax.set_ylabel("Perturbation $v \cdot \mathbf{v}_2$ (Cell 2 - Conflict)", color="#90a8d0", fontsize=11)
            ax.tick_params(colors="#7088b0")
            for spine in ax.spines.values():
                spine.set_color("#182850")

            plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.93])

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor(), edgecolor="none")
            plt.close()
            buf.seek(0)
            frames.append(Image.open(buf))
            print(f"Rendered frame for K={k:02d} ({len(frames)}/{len(sampled_k)})", flush=True)

    gif_path = out_dir / "cbim_fractal_basin_time_evolution_deep.gif"
    frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=frames[1:],
        duration=130,
        loop=0
    )
    print(f"\nSuccessfully saved deep fractal basin evolution animation to {gif_path}!", flush=True)


def main():
    out_dir = Path("present")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("results/arm_c_champion_3000/Best_Arm_C_Champion_d256.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    inner = CorrectiveCBIMSudokuModel(vocab_size=11, d_channels=256, ponder_steps=256, arm="corrective_flow").to(device)
    head = ACTLossHead(inner, loss_type="stablemax_cross_entropy").to(device)
    head.load_state_dict(ckpt["model"])
    model = head
    model.eval()

    test_in = np.load("data/sudoku-extreme-1k-aug-100/test/all__inputs.npy")
    inp_hard = torch.as_tensor(test_in[122:123], dtype=torch.long, device=device)

    generate_deep_fractal_basin_animation(model, inp_hard, out_dir, max_k=64, res=80, radius=2.5)


if __name__ == "__main__":
    main()
