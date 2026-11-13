"""Direct, standalone tests for internal lens and epistemic void on Checkpoint 2940.

Runs entirely on CPU in seconds with no heavy world/optimizer instantiation:
1. Logit Lens vs Local Relation Lens readout (J-space comparison).
2. Epistemic Void Tensor (causal covariance of retrieved relations) pulled back to vocabulary.
3. Alignment between Epistemic Void leading eigenvector and question_to_logits['?'].
4. J-space drift under quiet rotation flow.
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

from scripts.ib_local.bfss_gdn_capture import matrix_view, quiet_packed
from scripts.ib_local.bfss_gdn_reference import rotation_charge
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM


def main():
    cp_path = 'results/cgb_bfss_gdn_reference_20261110/bound_question_shadow_admission_4/last.pt.gz'
    print(f"Loading checkpoint {cp_path}...", flush=True)

    with gzip.open(cp_path, 'rb') as f:
        st = torch.load(f, map_location='cpu', weights_only=False)

    model_st = st['model']
    engine_st = st['engine']
    cycle = st.get('cycle', 2940)
    print(f"Cycle: {cycle}", flush=True)

    # Base LM weights
    lm = GatedDeltaNetLM(vocab_size=256, d=128, layers=4, heads=4,
                         recurrence='decay_then_delta', conv_kernel=4, mlp_hidden=256)
    base_keys = {k.replace('body.base.', ''): v for k, v in model_st.items() if 'body.base.' in k}
    lm.load_state_dict(base_keys, strict=False)

    W_U = lm.decoder.weight.detach() # [256, 128]
    W_enc = model_st['encoder.weight'].detach() # [64, 128]
    W_q2logits = model_st['question_to_logits.weight'].detach() # [256, 64]

    # Memory reservoir
    mem_source = model_st['memory.source'].detach() # [4096, 128]
    mem_action = model_st['memory.action'].detach() # [4096, 128]
    mem_next = model_st['memory.internal_next'].detach() # [4096, 128]
    mem_valid = (model_st['memory.occupied'] &
                 model_st['memory.internal_valid'] &
                 (model_st['memory.count'] == 1) &
                 (model_st['memory.internal_stamp'] == model_st['memory.last_write'])).detach()

    num_valid_records = int(mem_valid.sum().item())
    print(f"Memory reservoir has {num_valid_records} valid active relational records.", flush=True)

    # -------------------------------------------------------------
    # EXPERIMENT 1: Logit Lens vs Local Relation Lens Readout
    # -------------------------------------------------------------
    print("\n" + "="*60, flush=True)
    print("EXPERIMENT 1: Logit Lens vs Local Relation Lens Readout", flush=True)
    print("="*60, flush=True)

    # Find the most recently written valid slot as current live state
    stamps = model_st['memory.last_write'].clone()
    stamps[~mem_valid] = -1
    latest_slot = int(stamps.argmax().item())
    h_curr = mem_source[latest_slot].clone() # [128]

    # 1. Logit Lens: raw h_curr @ W_U.T
    raw_logits = h_curr @ W_U.T
    raw_probs = F.softmax(raw_logits, dim=-1)
    raw_entropy = float(-(raw_probs * (raw_probs + 1e-12).log()).sum())
    top5_raw = torch.topk(raw_probs, 5)

    print(f"Current state taken from memory slot {latest_slot} (stamp {int(stamps[latest_slot].item())})", flush=True)
    print(f"Logit Lens (Raw h) Entropy: {raw_entropy:.4f} nats", flush=True)
    print("Logit Lens Top-5 tokens:", flush=True)
    for idx, (p, tid) in enumerate(zip(top5_raw.values.tolist(), top5_raw.indices.tolist())):
        ch = repr(chr(tid)) if 32 <= tid <= 126 else f"0x{tid:02x}"
        print(f"  {idx+1}. Token {tid:3d} ({ch:6s}): p = {p:.4f}", flush=True)

    # 2. Local Relation Lens:
    # Retrieve top-8 similar experiences to h_curr
    q_enc = F.normalize(h_curr @ W_enc.T, dim=-1) # [64]
    bank_enc = F.normalize(mem_source[mem_valid] @ W_enc.T, dim=-1) # [N, 64]
    sims = bank_enc @ q_enc # [N]
    valid_indices = torch.where(mem_valid)[0]
    top_sims, top_k_local = sims.topk(min(8, num_valid_records))
    top_slots = valid_indices[top_k_local]

    # Local ridge regression to find R: h -> h'
    S_mat = mem_source[top_slots] # [K, 128]
    Y_mat = mem_next[top_slots]   # [K, 128]
    ridge = 0.1
    # R = Y^T S (S^T S + lambda I)^-1
    R_local = Y_mat.T @ S_mat @ torch.linalg.inv(S_mat.T @ S_mat + ridge * torch.eye(128)) # [128, 128]
    h_pred = R_local @ h_curr # [128]

    pred_logits = h_pred @ W_U.T
    pred_probs = F.softmax(pred_logits, dim=-1)
    pred_entropy = float(-(pred_probs * (pred_probs + 1e-12).log()).sum())
    top5_pred = torch.topk(pred_probs, 5)

    print(f"\nLocal Relation Lens (R @ h) Entropy: {pred_entropy:.4f} nats (Entropy change: {pred_entropy - raw_entropy:+.4f})", flush=True)
    print("Local Relation Lens Top-5 tokens (Poised to Verbalize Next):", flush=True)
    for idx, (p, tid) in enumerate(zip(top5_pred.values.tolist(), top5_pred.indices.tolist())):
        ch = repr(chr(tid)) if 32 <= tid <= 126 else f"0x{tid:02x}"
        print(f"  {idx+1}. Token {tid:3d} ({ch:6s}): p = {p:.4f}", flush=True)

    # -------------------------------------------------------------
    # EXPERIMENT 2: Epistemic Void Tensor & Vocabulary Pullback
    # -------------------------------------------------------------
    print("\n" + "="*60, flush=True)
    print("EXPERIMENT 2: Epistemic Void Tensor (Causal Uncertainty X=?)", flush=True)
    print("="*60, flush=True)

    # Differences among the retrieved successor states Y_mat [K, 128]
    diffs = Y_mat - Y_mat.mean(dim=0, keepdim=True) # [K, 128]
    cov_sigma = (diffs.T @ diffs) / (len(top_slots) - 1 + 1e-12) # [128, 128]

    # Pullback to vocabulary: G(i) = W_U[i] @ Cov @ W_U[i].T
    gap_variance = (W_U @ cov_sigma * W_U).sum(dim=-1) # [256]
    top_void = torch.topk(gap_variance, 10)

    print(f"Total Relational Uncertainty Trace: {float(cov_sigma.trace()):.4f}", flush=True)
    print("Top-10 Concepts with HIGHEST Relational Uncertainty (Epistemic Void X=?):", flush=True)
    for idx, (var_val, tid) in enumerate(zip(top_void.values.tolist(), top_void.indices.tolist())):
        ch = repr(chr(tid)) if 32 <= tid <= 126 else f"0x{tid:02x}"
        print(f"  {idx+1:2d}. Token {tid:3d} ({ch:6s}): uncertainty variance = {var_val:.4f}", flush=True)

    q_rank = int((gap_variance > gap_variance[63]).sum().item()) + 1
    print(f"\nQuestion mark '?' (ASCII 63): rank {q_rank} / 256, variance = {float(gap_variance[63]):.4f}", flush=True)

    # -------------------------------------------------------------
    # EXPERIMENT 3: Alignment with question_to_logits Head
    # -------------------------------------------------------------
    print("\n" + "="*60, flush=True)
    print("EXPERIMENT 3: Epistemic Void Leading Mode vs Question Head", flush=True)
    print("="*60, flush=True)

    # Leading eigenvector of epistemic void Cov in 128D
    eigvals, eigvecs = torch.linalg.eigh(cov_sigma)
    v_max_128 = eigvecs[:, -1] # [128] leading direction of uncertainty
    lambda_max = float(eigvals[-1])
    print(f"Leading eigenvalue of Epistemic Void: {lambda_max:.4f} (accounts for {lambda_max / float(cov_sigma.trace()) * 100:.2f}% of total variance)", flush=True)

    # Project v_max_128 to 64D encoded question space via W_enc
    v_max_64 = F.normalize(W_enc @ v_max_128, dim=-1) # [64]

    # The dedicated question vector for '?' in question_to_logits
    # question_to_logits is [256, 64]
    w_q_mark = F.normalize(W_q2logits[63], dim=-1) # [64]

    # Cosine similarity between highest uncertainty direction and question mark trigger!
    cos_sim = float((v_max_64 * w_q_mark).sum().item())

    # Baseline comparison: average absolute cosine similarity against all 256 tokens
    all_w = F.normalize(W_q2logits, dim=-1) # [256, 64]
    all_cos = (all_w @ v_max_64).abs().cpu().numpy()
    mean_cos = float(all_cos.mean())
    max_cos = float(all_cos.max())
    q_cos_rank = int((all_cos > abs(cos_sim)).sum()) + 1

    print(f"Cosine similarity between Causal Void leading direction and '?' question weight: {cos_sim:+.4f}", flush=True)
    print(f"Baseline mean |cos| across all 256 tokens: {mean_cos:.4f}, max |cos|: {max_cos:.4f}", flush=True)
    print(f"'?' Rank among all tokens by alignment to Epistemic Void: {q_cos_rank} / 256", flush=True)

    # -------------------------------------------------------------
    # EXPERIMENT 4: Quiet Flow Thinking Trajectory (J-Space)
    # -------------------------------------------------------------
    print("\n" + "="*60, flush=True)
    print("EXPERIMENT 4: J-Space Token Trajectory under Quiet Thinking", flush=True)
    print("="*60, flush=True)

    recurrent = engine_st['recurrent']
    momenta = engine_st['bfss_momenta']

    # Unpack recurrent S into layers
    for steps in [0, 1, 4, 16, 64]:
        s_next, p_next = quiet_packed(lm, recurrent.clone(), [p.clone() for p in momenta],
                                      dt=0.01, frequency=0.8, coupling=0.01, steps=steps)
        # Check layer 2 and layer 3 S matrices
        s2 = matrix_view(lm.layers[2], s_next[2])[0] # [heads, dk, dv]
        # Contract with W_U to see what Layer 2 state projects onto
        # Key space projection of layer 2
        p2 = p_next[2][0]
        qk2, qv2 = rotation_charge(s2.unsqueeze(0), p2.unsqueeze(0))
        drift = float((s_next - recurrent).norm())
        print(f"Quiet Step {steps:2d}: S-drift = {drift:6.3f}, ||Q_K^(L2)|| = {float(qk2.norm()):.4f}, ||Q_V^(L2)|| = {float(qv2.norm()):.4f}", flush=True)


if __name__ == '__main__':
    main()
