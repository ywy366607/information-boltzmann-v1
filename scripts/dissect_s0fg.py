"""Open the s0fg ckpt and show where the rectangle comes from."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from fine_grain.native_mot import coords, xy_features
from fine_grain.omni_model import DualStreamOmni
from fine_grain.omni_tasks import one_sample
from scripts.train_omni_probe import f_iterate, pixel_rms


def to_img(t):
    x = t.detach().cpu()
    if x.dim() == 4:
        x = x[0]
    x = x.permute(1, 2, 0).numpy()
    return np.clip(x, 0, 1)


def heat(t, res=32):
    x = t.detach().cpu().float().reshape(res, res).numpy()
    return x


def load_model(dev):
    m = DualStreamOmni(
        d_model=256, n_slices=64, n_layers=4, res=32, n_heads=8,
        surprise_mode="v1_bayes", s_update="rms_dir", use_stiefel=True,
        deslice_topk=2, use_residual_read=True, use_null_slice=True,
        use_ticket_read=False, use_write_yield=False, deslice_write="increment",
        fm_pred="x", fm_signed=True, vfe_coef=0.1,
    ).to(dev)
    raw = torch.load("checkpoints/omni_d256_unified_t2i_s0fg_best.pt", map_location=dev)
    m.load_state_dict(raw, strict=True)
    m.eval()
    return m


def s0_and_attn(model, prompts, device):
    tok, mask = model.tokenize(prompts, device)
    H = model.mot_stack.text_in(model.embed(tok))
    layer = model.mot_stack.layers[0]
    B, T, d = H.shape
    N = model.res * model.res
    xy = coords(model.res, device).to(dtype=H.dtype)
    feat = xy_features(xy, n_freq=layer.lang_s0_freq).expand(B, N, -1)
    q = layer.s0_q(feat)
    k, v = layer.s0_kv(H).chunk(2, dim=-1)
    attn = torch.softmax(q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5), dim=-1)
    attn = attn.masked_fill(~mask.bool().unsqueeze(1), 0.0)
    s0 = layer.language_prior(H, mask.float(), N, device, H.dtype)
    return s0, attn, tok, mask


def snapshot(model, x, prompt, device):
    t = torch.ones(x.shape[0], device=device)
    with torch.no_grad():
        out = model(x, [prompt], need_pix=[True], t=t)
    L = model.mot_stack.layers
    s0 = L[0].last_s0
    xin = model.mot_stack.encode_X(x, t=t)
    e = (xin - s0).pow(2).mean(-1)[0]
    nulls = [Li.last_null[0] for Li in L]
    ws = [Li.last_w[0] for Li in L]
    s0_rgb = ((model.decode_field(s0) + 1) * 0.5).clamp(0, 1)
    return {
        "rgb": ((out["x_pred"] + 1) * 0.5).clamp(0, 1) if model.fm_signed else out["rgb"],
        "s0_rgb": s0_rgb,
        "e": e,
        "nulls": nulls,
        "ws": ws,
        "s0": s0,
    }


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = load_model(dev)
    rng = np.random.default_rng(0)
    s7 = one_sample(rng, 32, "t2i", t2i_canvas="black", t2i_place="center",
                    t2i_digit=7, t2i_color="green")
    s1 = one_sample(rng, 32, "t2i", t2i_canvas="black", t2i_place="center",
                    t2i_digit=1, t2i_color="red")
    s6 = one_sample(rng, 32, "t2i", t2i_canvas="black", t2i_place="center",
                    t2i_digit=6, t2i_color="yellow")
    box = torch.zeros(32, 32)
    box[8:24, 8:24] = 1.0
    box = box.to(dev)

    prompts = [s7["prompt"], s1["prompt"], s6["prompt"]]
    s0, attn, tok, mask = s0_and_attn(m, prompts, dev)
    words = []
    for i, p in enumerate(prompts):
        ws = p.replace("?", " ?").split()
        words.append(ws)
        print("PROMPT", i, ws)

    def cos(i, j):
        a, b = s0[i].reshape(1, -1), s0[j].reshape(1, -1)
        return float(F.cosine_similarity(a, b))

    print(f"S0 cos 7vs1={cos(0,1):.4f}  7vs6={cos(0,2):.4f}")
    s0_rgb = ((m.decode_field(s0) + 1) * 0.5).clamp(0, 1)
    for i, name in enumerate(["7g", "1r", "6y"]):
        img = s0_rgb[i]
        e_in = float(img.mean(0)[box.bool()].mean())
        e_out = float(img.mean(0)[~box.bool()].mean())
        print(f"  S0_dec {name}: mean in-box={e_in:.3f} out-box={e_out:.3f} std={float(img.std()):.3f}")

    # token mass at center vs corner for prompt 7
    att = attn[0].reshape(32, 32, -1)  # H,W,T
    center = att[12:20, 12:20].mean((0, 1))
    corner = att[:4, :4].mean((0, 1))
    print("attn mass prompt7 (center / corner):")
    for t, w in enumerate(words[0]):
        print(f"  [{t:2d}] {w:10s}  c={float(center[t]):.3f}  corner={float(corner[t]):.3f}")

    # F-descent unroll from signed black
    x = torch.full((1, 3, 32, 32), -1.0, device=dev)
    prompt = s7["prompt"]
    ones = torch.ones(1, device=dev)
    frames = [((x + 1) * 0.5).clamp(0, 1)]
    snaps = []
    cur = x
    print("\n=== F-descent unroll prompt 7, start=black ===")
    for k in range(8):
        with torch.no_grad():
            snap = snapshot(m, cur, prompt, dev)
        nxt = m(cur, [prompt], need_pix=[True], t=ones)["x_pred"]
        rms = float(pixel_rms(nxt, cur).mean())
        rgb = ((nxt + 1) * 0.5).clamp(0, 1)
        st = s7["stroke"]
        if st.dim() == 3:
            st = st[0]
        st = st.to(dev)
        flood = float((((rgb - 0).abs().mean(1) > 0.2) * (1 - st)).sum() / (1 - st).sum())
        L0 = m.mot_stack.layers[0]
        w = L0.last_w[0]
        ent = -(w.clamp_min(1e-8) * w.clamp_min(1e-8).log()).sum(-1)
        hnorm = float(ent.mean() / np.log(w.shape[-1]))
        e = snap["e"].reshape(32, 32)
        n0 = snap["nulls"][0].reshape(32, 32)
        print(
            f"  k={k} rms={rms:.4f} pred_std={float(rgb.std()):.3f} "
            f"null_box={float(n0[box.bool()].mean()):.3f}/{float(n0[~box.bool()].mean()):.3f} "
            f"e_box={float(e[box.bool()].mean()):.3f}/{float(e[~box.bool()].mean()):.3f} "
            f"H(w)/logM={hnorm:.3f} wsum={float(w.sum(-1).mean()):.3f}"
        )
        print("    per-layer null:", " ".join(f"{float(n.mean()):.2f}" for n in snap["nulls"]))
        frames.append(rgb)
        snaps.append(snap)
        cur = nxt

    # first-step delta spatial: is it a filled box?
    d0 = (frames[1] - frames[0]).abs().mean(1)[0]
    print(f"\nstep1 |Δ| in-box={float(d0[box.bool()].mean()):.4f} out={float(d0[~box.bool()].mean()):.4f}")

    fig, axes = plt.subplots(4, 8, figsize=(16, 8), facecolor="#0f172a")
    titles0 = ["S0 7g", "S0 1r", "S0 6y", "S0 |7-1|", "e k=0", "∅ k=0", "e k=1", "∅ k=1"]
    vis = [
        to_img(s0_rgb[0]), to_img(s0_rgb[1]), to_img(s0_rgb[2]),
        to_img((s0_rgb[0] - s0_rgb[1]).abs()),
        heat(snaps[0]["e"]), heat(snaps[0]["nulls"][0]),
        heat(snaps[1]["e"]) if len(snaps) > 1 else heat(snaps[0]["e"]),
        heat(snaps[1]["nulls"][0]) if len(snaps) > 1 else heat(snaps[0]["nulls"][0]),
    ]
    for ax, im, title in zip(axes[0], vis, titles0):
        ax.set_facecolor("#0f172a")
        if im.ndim == 2:
            ax.imshow(im, cmap="magma", vmin=0, vmax=max(float(im.max()), 1e-6))
        else:
            ax.imshow(im)
        ax.set_title(title, color="#e2e8f0", fontsize=8)
        ax.axis("off")
    for k in range(8):
        ax = axes[1, k]
        ax.set_facecolor("#0f172a")
        ax.imshow(to_img(frames[k]))
        ax.set_title(f"rgb k={k}", color="#e2e8f0", fontsize=8)
        ax.axis("off")
    # layer null at k=0
    for i in range(4):
        ax = axes[2, i]
        ax.set_facecolor("#0f172a")
        ax.imshow(heat(snaps[0]["nulls"][i]), cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"∅ L{i} k=0", color="#e2e8f0", fontsize=8)
        ax.axis("off")
    for i in range(4):
        w = snaps[0]["ws"][i]
        mass = w.sum(-1).reshape(32, 32)
        ax = axes[2, 4 + i]
        ax.set_facecolor("#0f172a")
        ax.imshow(heat(mass), cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"content-w L{i}", color="#e2e8f0", fontsize=8)
        ax.axis("off")
    tgt = s7["target_rgb"]
    if tgt.dim() == 4:
        tgt = tgt[0]
    axes[3, 0].imshow(to_img(tgt.unsqueeze(0)))
    axes[3, 0].set_title("target 7", color="#e2e8f0", fontsize=8)
    axes[3, 0].axis("off")
    axes[3, 0].set_facecolor("#0f172a")
    axes[3, 1].imshow(to_img(frames[-1]))
    axes[3, 1].set_title("final pred", color="#e2e8f0", fontsize=8)
    axes[3, 1].axis("off")
    axes[3, 1].set_facecolor("#0f172a")
    dlt = (frames[-1] - frames[0]).abs().mean(1)[0].detach().cpu().numpy()
    axes[3, 2].imshow(dlt, cmap="magma")
    axes[3, 2].set_title("|Δ| total", color="#e2e8f0", fontsize=8)
    axes[3, 2].axis("off")
    axes[3, 2].set_facecolor("#0f172a")
    for ax in axes[3, 3:]:
        ax.axis("off")
        ax.set_facecolor("#0f172a")
    fig.suptitle("s0fg dissect: S0 / residual / null / F-descent", color="white")
    out = Path("present/figs/s0fg_dissect.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=140)
    plt.close()
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
