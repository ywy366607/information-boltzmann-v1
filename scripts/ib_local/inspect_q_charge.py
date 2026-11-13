"""Inspect conserved matrix Noether charge Q in checkpoint.

Loads the checkpoint, unpacks S and P, calculates Q_K and Q_V,
and analyzes their numerical, spectral, and semantic properties.
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

# Mock numba/triton if needed
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
from scripts.ib_local.bfss_gdn_reference import rotation_charge
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM


def inspect_q(checkpoint_path: str):
    print(f"Loading checkpoint: {checkpoint_path}")
    with gzip.open(checkpoint_path, 'rb') as f:
        st = torch.load(f, map_location='cpu', weights_only=False)

    print("Keys in st:", list(st.keys()))
    engine_dict = st.get('engine', {})
    print("Engine keys:", list(engine_dict.keys()))

    recurrent = engine_dict['recurrent'] # [layers, batch, packed_dim]
    momenta = engine_dict.get('bfss_momenta', [])

    print(f"Recurrent shape: {recurrent.shape}")
    print(f"Momenta count: {len(momenta)}")

    # Instantiate base model to get layers and matrix dimensions
    lm = GatedDeltaNetLM(vocab_size=256, d=128, layers=4, heads=4,
                         recurrence='decay_then_delta', conv_kernel=4, mlp_hidden=256)

    layers = lm.layers
    num_layers = len(layers)

    results = []

    for l_idx in range(num_layers):
        layer = layers[l_idx]
        s_packed = recurrent[l_idx]
        s_mat = matrix_view(layer, s_packed) # [batch, heads, dk, dv]
        p_mat = momenta[l_idx] # [batch, heads, dk, dv]

        batch_size, num_heads, dk, dv = s_mat.shape

        # S and P norms
        s_frob = float(s_mat.norm())
        p_frob = float(p_mat.norm())

        # Q_K and Q_V
        # rotation_charge computes:
        # q_k = sum_h (S_h P_h^T - P_h S_h^T) / 2
        # q_v = sum_h (S_h^T P_h - P_h^T S_h) / 2
        qk, qv = rotation_charge(s_mat, p_mat) # [batch, dk, dk], [batch, dv, dv]

        qk_0 = qk[0].detach() # [dk, dk]
        qv_0 = qv[0].detach() # [dv, dv]

        qk_norm = float(qk_0.norm())
        qv_norm = float(qv_0.norm())

        # Antisymmetry check: Q + Q^T == 0
        qk_anti_err = float((qk_0 + qk_0.T).abs().max())
        qv_anti_err = float((qv_0 + qv_0.T).abs().max())

        # Diagonal check: diag(Q) == 0
        qk_diag_max = float(qk_0.diagonal().abs().max())
        qv_diag_max = float(qv_0.diagonal().abs().max())

        # Eigenspectrum of antisymmetric matrix: eigenvalues are purely imaginary +/- i * lambda_k
        # We can compute eigvals of Qk
        # Since Q is real antisymmetric, Q = -Q^T, i*Q is Hermitian!
        # Eigenvalues of i*Q are real!
        eigvals_k = torch.linalg.eigvalsh(1j * qk_0.to(torch.complex64)).cpu().numpy()
        eigvals_v = torch.linalg.eigvalsh(1j * qv_0.to(torch.complex64)).cpu().numpy()

        # Sort singular values / imaginary frequencies
        freqs_k = np.sort(np.abs(eigvals_k))[::-1]
        freqs_v = np.sort(np.abs(eigvals_v))[::-1]

        # Top 5 frequencies (unique positive values)
        top_freqs_k = freqs_k[::2][:5].tolist()
        top_freqs_v = freqs_v[::2][:5].tolist()

        # Participation rank of Q spectrum: (sum lambda^2)^2 / sum lambda^4
        p_rank_k = float((np.sum(freqs_k**2)**2) / (np.sum(freqs_k**4) + 1e-12))
        p_rank_v = float((np.sum(freqs_v**2)**2) / (np.sum(freqs_v**4) + 1e-12))

        # Check per-head contribution to Q_K and Q_V
        head_contrib_k = []
        head_contrib_v = []
        for h in range(num_heads):
            s_h = s_mat[0, h]
            p_h = p_mat[0, h]
            qk_h = 0.5 * (s_h @ p_h.T - p_h @ s_h.T)
            qv_h = 0.5 * (s_h.T @ p_h - p_h.T @ s_h)
            head_contrib_k.append(float(qk_h.norm()))
            head_contrib_v.append(float(qv_h.norm()))

        # Correlation between Q and S, P
        # What is in Q? Does Q align with key projections?
        layer_summary = {
            "layer": l_idx,
            "s_norm": s_frob,
            "p_norm": p_frob,
            "qk_norm": qk_norm,
            "qv_norm": qv_norm,
            "qk_anti_err": qk_anti_err,
            "qv_anti_err": qv_anti_err,
            "qk_diag_max": qk_diag_max,
            "qv_diag_max": qv_diag_max,
            "top_freqs_k": top_freqs_k,
            "top_freqs_v": top_freqs_v,
            "participation_rank_k": p_rank_k,
            "participation_rank_v": p_rank_v,
            "head_contrib_qk": head_contrib_k,
            "head_contrib_qv": head_contrib_v,
        }
        results.append(layer_summary)
        print(f"\n--- Layer {l_idx} ---")
        print(f"||S||_F = {s_frob:.4f}, ||P||_F = {p_frob:.4f}")
        print(f"||Q_K||_F = {qk_norm:.4f}, ||Q_V||_F = {qv_norm:.4f}")
        print(f"Antisymmetry error: Q_K: {qk_anti_err:.2e}, Q_V: {qv_anti_err:.2e}")
        print(f"Top 5 angular frequencies (K): {[round(x, 4) for x in top_freqs_k]}")
        print(f"Top 5 angular frequencies (V): {[round(x, 4) for x in top_freqs_v]}")
        print(f"Participation rank (modes): K={p_rank_k:.2f}/16, V={p_rank_v:.2f}/16")
        print(f"Per-head ||Q_K^h||: {[round(x, 4) for x in head_contrib_k]}")
        print(f"Per-head ||Q_V^h||: {[round(x, 4) for x in head_contrib_v]}")

    return results

if __name__ == '__main__':
    cp = 'results/cgb_bfss_gdn_reference_20261110/bound_question_shadow_admission_4/last.pt.gz'
    if len(sys.argv) > 1:
        cp = sys.argv[1]
    inspect_q(cp)
