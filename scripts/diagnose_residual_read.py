"""Measure S0, PE, and ∅ on the residual-read checkpoint."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fine_grain.omni_model import DualStreamOmni
from fine_grain.omni_tasks import one_sample


def main() -> None:
    ckpt = Path("checkpoints/omni_d256_unified_t2i_rread_best.pt")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = DualStreamOmni(
        d_model=256, n_slices=64, n_layers=4, res=32, n_heads=8,
        surprise_mode="v1_bayes", s_update="rms_dir", use_stiefel=True,
        deslice_topk=2, use_residual_read=True, use_null_slice=True,
        use_ticket_read=False, use_write_yield=False, deslice_write="increment",
        fm_pred="x", fm_signed=True, vfe_coef=0.1,
    ).to(dev)
    raw = torch.load(ckpt, map_location=dev)
    m.load_state_dict(raw, strict=True)
    m.eval()
    layer0 = m.mot_stack.layers[0]
    tau = float(layer0.read.pe_null_tau.detach())
    gamma = float(F.softplus(layer0.read.pe_null_gamma_raw).detach())
    print(f"tau={tau:.4f} gamma={gamma:.4f}  ell_null=tau-gamma*e")

    rng = np.random.default_rng(0)
    specs = [
        ("7g", 7, "green"),
        ("1r", 1, "red"),
        ("8b", 8, "blue"),
    ]
    samples = {
        k: one_sample(
            rng, 32, "t2i", t2i_canvas="black", t2i_place="center",
            t2i_digit=d, t2i_color=c,
        )
        for k, d, c in specs
    }

    def probe(sample, canvas: str) -> dict:
        img = sample["target_rgb"] if canvas == "target" else sample["image"]
        if canvas == "noise":
            img = torch.randn_like(sample["image"])
            if m.fm_signed:
                pass
            else:
                img = img.clamp(0, 1)
        elif canvas == "black":
            img = torch.zeros_like(sample["image"])
        if img.dim() == 3:
            img = img.unsqueeze(0)
        x = img.to(dev)
        if canvas != "noise":
            x = x * 2 - 1
        # noise already ~N(0,1) in signed chart
        t = torch.ones(x.shape[0], device=dev)
        with torch.no_grad():
            out = m(x, [sample["prompt"]], need_pix=[True], t=t)
        L = m.mot_stack.layers[0]
        s0 = L.last_s0
        xin = m.mot_stack.encode_X(x, t=t)
        e = (xin - s0).pow(2).mean(-1)[0].reshape(32, 32)
        null = L.last_null[0].reshape(32, 32)
        w = L.last_w[0]
        ent = -(w.clamp_min(1e-8) * w.clamp_min(1e-8).log()).sum(-1)
        hnorm = float(ent.mean() / np.log(w.shape[-1]))
        st = sample["stroke"]
        if st.dim() == 3:
            st = st[0]
        st = st.to(dev)
        bg = 1.0 - st
        s0_dec = ((m.decode_field(s0) + 1.0) * 0.5).clamp(0, 1)
        tgt = sample["target_rgb"]
        if tgt.dim() == 3:
            tgt = tgt.unsqueeze(0)
        xo = m.mot_stack.encode_X(tgt.to(dev) * 2 - 1, t=t)
        return {
            "null_all": float(null.mean()),
            "null_stroke": float((null * st).sum() / st.sum().clamp_min(1)),
            "null_bg": float((null * bg).sum() / bg.sum().clamp_min(1)),
            "e_all": float(e.mean()),
            "e_stroke": float((e * st).sum() / st.sum().clamp_min(1)),
            "e_bg": float((e * bg).sum() / bg.sum().clamp_min(1)),
            "hnorm": hnorm,
            "wsum": float(w.sum(-1).mean()),
            "s0_var": float(s0.var(1).mean()),
            "s0_acc": float((s0 - xo).pow(2).mean()),
            "s0_rgb_std": float(s0_dec.std()),
            "s0_rgb_mean": float(s0_dec.mean()),
            "pred_std": float(out["rgb"].std()),
            "pred_mean": float(out["rgb"].mean()),
            "s0": s0,
            "s0_dec": s0_dec,
            "stroke": st,
        }

    print("\n=== S0 identity (black canvas, different prompts) ===")
    p = {k: probe(samples[k], "black") for k in samples}

    def pooled(s):
        return s.mean(1)

    c71 = float(F.cosine_similarity(pooled(p["7g"]["s0"]), pooled(p["1r"]["s0"])))
    c78 = float(F.cosine_similarity(pooled(p["7g"]["s0"]), pooled(p["8b"]["s0"])))
    print(f"cos S0 7g vs 1r = {c71:.4f}   7g vs 8b = {c78:.4f}   (1=collapsed prior)")
    for k, rec in p.items():
        st = rec["stroke"]
        s0m = rec["s0_dec"][0].mean(0)
        on = float((s0m * st).sum() / st.sum().clamp_min(1))
        off = float((s0m * (1 - st)).sum() / (1 - st).sum().clamp_min(1))
        print(
            f"  {k}: s0_var={rec['s0_var']:.5f} s0_acc_vs_tgt={rec['s0_acc']:.4f} "
            f"s0_dec mean stroke/bg={on:.4f}/{off:.4f}  "
            f"null s/bg={rec['null_stroke']:.3f}/{rec['null_bg']:.3f}  "
            f"e s/bg={rec['e_stroke']:.3f}/{rec['e_bg']:.3f}  "
            f"H(w)/logM={rec['hnorm']:.3f} wsum={rec['wsum']:.3f}"
        )

    print("\n=== same prompt 7g, different canvases ===")
    for canvas in ("black", "noise", "target"):
        rec = probe(samples["7g"], canvas)
        print(
            f"  {canvas:7s}: e_all={rec['e_all']:.3f} e_s/bg={rec['e_stroke']:.3f}/{rec['e_bg']:.3f}  "
            f"null_all={rec['null_all']:.3f} null_s/bg={rec['null_stroke']:.3f}/{rec['null_bg']:.3f}  "
            f"H(w)/logM={rec['hnorm']:.3f} pred_std={rec['pred_std']:.3f} s0_acc={rec['s0_acc']:.3f}"
        )

    # later layers: does null stay collapsed?
    rec = probe(samples["7g"], "black")
    print("\n=== per-layer null / entropy on black+7g ===")
    t = torch.ones(1, device=dev)
    x = torch.zeros(1, 3, 32, 32, device=dev)
    with torch.no_grad():
        m(x * 2 - 1, [samples["7g"]["prompt"]], need_pix=[True], t=t)
    for i, L in enumerate(m.mot_stack.layers):
        n = L.last_null
        w = L.last_w
        ent = -(w.clamp_min(1e-8) * w.clamp_min(1e-8).log()).sum(-1)
        h = float(ent.mean() / np.log(w.shape[-1]))
        print(f"  L{i}: null={float(n.mean()):.3f}  H(w)/logM={h:.3f}  wsum={float(w.sum(-1).mean()):.3f}")


if __name__ == "__main__":
    main()
