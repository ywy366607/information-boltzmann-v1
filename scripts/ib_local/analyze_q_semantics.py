"""Deep semantic, spectral, and geometric analysis of conserved charge Q.

Computes:
1. Phase space orbit geometry: S and P norms, inner product, cosine similarity,
   and out-of-phase circulation flux (sin theta, Q_norm).
2. Spectral decomposition of Q_K and Q_V:
   Real antisymmetric Q has 2x2 block diagonal Schur form (planes of rotation).
   Eigenvalues are +/- i * omega_m.
3. Feature alignment:
   Project principal rotation planes of Q_K through layer's k_proj and token embeddings
   to see which byte tokens are coupled into circular orbits.
   Project Q_V through v_proj and out_proj.
4. Comparison across checkpoints (e.g. cycle 2843 vs 2936 vs 2940).
"""
from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import types
if 'numba' not in sys.modules:
    nb = types.ModuleType('numba')
    nb.njit = lambda *a, **k: (lambda f: f)
    sys.modules['numba'] = nb
if 'triton' not in sys.modules:
    tr = types.ModuleType('triton')
    tr.__path__ = []
    tr.jit = lambda *a, **k: (lambda f: f)
    sys.modules['triton'] = tr
if 'triton.language' not in sys.modules:
    tl = types.ModuleType('triton.language')
    tl.constexpr = int
    sys.modules['triton.language'] = tl

from scripts.ib_local.bfss_gdn_capture import matrix_view
from scripts.ib_local.bfss_gdn_reference import rotation_charge, quiet_step
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM


def analyze_checkpoint(cp_path: str):
    print(f"=== Analyzing Checkpoint: {cp_path} ===")
    with gzip.open(cp_path, 'rb') as f:
        st = torch.load(f, map_location='cpu', weights_only=False)

    cycle = st.get('cycle', 'unknown')
    print(f"Cycle: {cycle}")

    engine_dict = st['engine']
    recurrent = engine_dict['recurrent']
    momenta = engine_dict['bfss_momenta']

    # Load model weights
    lm = GatedDeltaNetLM(vocab_size=256, d=128, layers=4, heads=4,
                         recurrence='decay_then_delta', conv_kernel=4, mlp_hidden=256)

    # Check if model state dict is in checkpoint
    if 'model' in st:
        model_st = st['model']
        # The base GDN is in model.body.base or similar
        # Let's inspect model_st keys
        base_keys = {k.replace('body.base.', ''): v for k, v in model_st.items() if 'body.base.' in k}
        if not base_keys:
            base_keys = {k: v for k, v in model_st.items() if k in lm.state_dict()}
        lm.load_state_dict(base_keys, strict=False)
        print(f"Loaded {len(base_keys)} weights into LM.")

    vocab_size = 256
    token_embed = lm.source.embedding.weight.detach() # [256, 128]

    report = {"cycle": cycle, "checkpoint": cp_path, "layers": []}

    for l_idx in range(4):
        layer = lm.layers[l_idx]
        s_mat = matrix_view(layer, recurrent[l_idx]) # [1, 4, 32, 32]
        p_mat = momenta[l_idx] # [1, 4, 32, 32]

        qk, qv = rotation_charge(s_mat, p_mat) # [1, 32, 32]
        qk_0 = qk[0].detach()
        qv_0 = qv[0].detach()

        layer_data = {
            "layer": l_idx,
            "heads": []
        }

        # Analyze per head
        for h in range(4):
            s_h = s_mat[0, h].detach()
            p_h = p_mat[0, h].detach()

            s_norm = float(s_h.norm())
            p_norm = float(p_h.norm())

            # Inner product <S, P>
            dot_sp = float((s_h * p_h).sum())
            cos_theta = dot_sp / (s_norm * p_norm + 1e-12)

            # Commutators / Cross product
            qk_h = 0.5 * (s_h @ p_h.T - p_h @ s_h.T)
            qv_h = 0.5 * (s_h.T @ p_h - p_h.T @ s_h)

            qk_h_norm = float(qk_h.norm())
            qv_h_norm = float(qv_h.norm())

            # Phase flux / cross area:
            # For 2 vectors, ||s x p|| = s*p*sin_theta.
            # In matrix space, does ||S||^2 ||P||^2 = <S,P>^2 + 2 ||Q_K||^2?
            # Let's verify: ||S P^T - P S^T||_F^2 / 2
            head_info = {
                "head": h,
                "s_norm": s_norm,
                "p_norm": p_norm,
                "cos_theta": cos_theta,
                "sin_flux_k": qk_h_norm / (s_norm * p_norm + 1e-12),
                "sin_flux_v": qv_h_norm / (s_norm * p_norm + 1e-12),
                "qk_norm": qk_h_norm,
                "qv_norm": qv_h_norm
            }
            layer_data["heads"].append(head_info)

        # Spectral analysis of layer Q_K and Q_V
        # Real Schur decomposition or Hermitian eigendecomposition of 1j * Q
        # Q = U * Omega * U^T where Omega has 2x2 blocks [[0, omega], [-omega, 0]]
        eigvals_k, eigvecs_k = torch.linalg.eigh(1j * qk_0.to(torch.complex64))
        eigvals_v, eigvecs_v = torch.linalg.eigh(1j * qv_0.to(torch.complex64))

        # Sort by eigenvalue magnitude
        k_indices = torch.argsort(eigvals_k.abs(), descending=True)
        v_indices = torch.argsort(eigvals_v.abs(), descending=True)

        top_modes_k = []
        # Examine top 3 conjugate pairs
        for pair_idx in range(min(3, 16)):
            pos_idx = k_indices[2 * pair_idx]
            neg_idx = k_indices[2 * pair_idx + 1]
            omega = float(eigvals_k[pos_idx].abs())
            if omega < 1e-4:
                continue

            v_pos = eigvecs_k[:, pos_idx]
            # Real basis for this 2D rotation plane:
            u1 = v_pos.real
            u2 = v_pos.imag
            u1 = F.normalize(u1, dim=-1)
            u2 = F.normalize(u2, dim=-1)

            # Let's project u1, u2 into key space across heads
            # layer.k_proj is [heads * dk, 128]
            # For head h, W_k^h is [dk, 128]
            # Effective key feature in 128D embedding space: u @ W_k^h
            # Then dot with token embeddings [256, 128] to find which bytes have highest activation!
            top_tokens_plane = []
            for h in [1, 3] if l_idx == 2 else [1, 2] if l_idx == 3 else [0]:
                W_k_h = layer.k_proj.weight.detach()[h*32:(h+1)*32, :] # [32, 128]
                emb_u1 = u1 @ W_k_h # [128]
                emb_u2 = u2 @ W_k_h # [128]

                # Project onto token embeddings
                tok_scores_1 = (token_embed @ emb_u1).cpu().numpy()
                tok_scores_2 = (token_embed @ emb_u2).cpu().numpy()
                tok_radius = np.sqrt(tok_scores_1**2 + tok_scores_2**2)

                top_tok_ids = np.argsort(tok_radius)[::-1][:6]
                tok_summary = []
                for tid in top_tok_ids:
                    char_repr = repr(chr(tid)) if 32 <= tid <= 126 else f"0x{tid:02x}"
                    tok_summary.append((char_repr, float(tok_radius[tid])))
                top_tokens_plane.append({"head": h, "top_tokens": tok_summary})

            top_modes_k.append({
                "mode_pair": pair_idx,
                "angular_freq_omega": omega,
                "semantic_tokens": top_tokens_plane
            })

        layer_data["qk_norm"] = float(qk_0.norm())
        layer_data["qv_norm"] = float(qv_0.norm())
        layer_data["top_modes_k"] = top_modes_k
        report["layers"].append(layer_data)

    return report


if __name__ == '__main__':
    cps = [
        ('Cycle 2940 (Current Bound Question)', 'results/cgb_bfss_gdn_reference_20261110/bound_question_shadow_admission_4/last.pt.gz'),
        ('Cycle 2936 (Persistent Question)', 'results/cgb_bfss_gdn_reference_20261110/persistent_question_admission_4/last.pt.gz'),
        ('Cycle 2843 (Shadow Calibrated)', 'results/cgb_bfss_gdn_reference_20261110/shadow_preview_calibrated_4/last.pt.gz'),
    ]

    all_reports = {}
    for name, cp_path in cps:
        if Path(cp_path).exists():
            rep = analyze_checkpoint(cp_path)
            all_reports[name] = rep
        else:
            print(f"Skipping non-existent {cp_path}")

    # Print summary
    print("\n" + "="*60)
    print("DETAILED Q CHARGE REPORT")
    print("="*60)
    for name, rep in all_reports.items():
        print(f"\n>>> {name} (Cycle {rep['cycle']}) <<<")
        for l in rep['layers']:
            print(f"  Layer {l['layer']}: ||Q_K|| = {l['qk_norm']:.4f}, ||Q_V|| = {l['qv_norm']:.4f}")
            for h in l['heads']:
                print(f"    Head {h['head']}: S={h['s_norm']:.2f}, P={h['p_norm']:.2f}, cos(S,P)={h['cos_theta']:.4f}, ||Q_K^h||={h['qk_norm']:.4f}, ||Q_V^h||={h['qv_norm']:.4f}")
            for m in l['top_modes_k']:
                print(f"    Mode {m['mode_pair']}: omega = {m['angular_freq_omega']:.4f}")
                for tok in m['semantic_tokens']:
                    tok_str = ", ".join([f"{c}:{r:.2f}" for c, r in tok['top_tokens'][:4]])
                    print(f"      Head {tok['head']} key plane aligns with: {tok_str}")
