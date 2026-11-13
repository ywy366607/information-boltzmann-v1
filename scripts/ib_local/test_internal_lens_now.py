"""Immediate testable experiments on internal lens and epistemic void.

Loads living checkpoint (Cycle 2940), builds engine on CPU, and tests:
1. Local Internal Lens vs Logit Lens readout.
2. Epistemic Void Tensor (causal uncertainty across retrieved relations) projected onto vocabulary.
3. J-space token trajectory during quiet internal thinking steps (0, 1, 8, 32, 64).
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
from scripts.ib_local.language_action import peek
from scripts.ib_local.navigation_internal_rollout import next_boundary_prediction
from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
from scripts.ib_local.navigation_relation_wood import local_relation
from scripts.ib_local.run_causal_water_reply import build


def run_tests():
    cp_path = Path('results/cgb_bfss_gdn_reference_20261110/bound_question_shadow_admission_4/last.pt.gz')
    print(f"Loading living individual from {cp_path}...")
    torch.set_num_threads(1)

    engine, _, _, provenance = build(cp_path, device='cpu')
    model = engine.model
    recurrent = engine.recurrent
    momenta = engine.bfss_momenta

    print(f"Successfully loaded. Cycle: {provenance['cycle']}, Engine class: {engine.__class__.__name__}")

    # Forward 1 observed byte to get latent state h
    byte = torch.tensor([[engine.pending]], device=engine.device)
    probs = peek(engine.table, engine.pending)
    logp = torch.as_tensor(np.log(np.maximum(probs, 1e-6)), device=engine.device)[None, None]

    with torch.no_grad():
        logits_orig, _, _, h, _ = model.predict_observed(byte, recurrent, logp)
        # h has shape [1, 1, d]
        h_vec = h[0, 0] # [128]

        # -------------------------------------------------------------
        # TEST 1: Internal Lens vs Logit Lens Readout
        # -------------------------------------------------------------
        print("\n" + "="*60)
        print("TEST 1: Internal Lens vs Logit Lens Readout")
        print("="*60)

        # Unembedding matrix: model.body.base.decoder.weight or similar
        # In GatedDeltaNetLM: lm.decoder.weight is [vocab_size, d]
        base_lm = model.body.base.base if hasattr(model.body.base, 'base') else model.body.base
        W_U = base_lm.decoder.weight # [256, 128]

        # Logit Lens (Raw intermediate h)
        raw_logits = h_vec @ W_U.T # [256]
        raw_probs = F.softmax(raw_logits, dim=-1)
        raw_entropy = float(-(raw_probs * (raw_probs + 1e-12).log()).sum())
        raw_top5 = torch.topk(raw_probs, 5)

        print(f"Logit Lens (Raw h) Entropy: {raw_entropy:.4f} nats")
        print("Logit Lens Top-5 tokens:")
        for idx, (p, tid) in enumerate(zip(raw_top5.values.tolist(), raw_top5.indices.tolist())):
            ch = repr(chr(tid)) if 32 <= tid <= 126 else f"0x{tid:02x}"
            print(f"  {idx+1}. Token {tid:3d} ({ch:6s}): p = {p:.4f}")

        # Internal Lens: relation + momentum + boundary prediction
        relation, _ = local_relation(model, h, h, ridge=engine.relation_ridge, scale=engine.relation_scale)
        p_read = decision_momentum_read(model, momenta, h)
        wood = next_boundary_prediction(model, h, h) + model.momentum_to_boundary(p_read) + relation

        # Corrected representation: h_wood
        # Note: wood has shape [1, 128]
        h_internal = h_vec + wood[0]
        internal_logits = h_internal @ W_U.T
        internal_probs = F.softmax(internal_logits, dim=-1)
        internal_entropy = float(-(internal_probs * (internal_probs + 1e-12).log()).sum())
        internal_top5 = torch.topk(internal_probs, 5)

        print(f"\nInternal Lens (h + Wood Lens) Entropy: {internal_entropy:.4f} nats (Entropy change: {internal_entropy - raw_entropy:+.4f})")
        print("Internal Lens Top-5 tokens:")
        for idx, (p, tid) in enumerate(zip(internal_top5.values.tolist(), internal_top5.indices.tolist())):
            ch = repr(chr(tid)) if 32 <= tid <= 126 else f"0x{tid:02x}"
            print(f"  {idx+1}. Token {tid:3d} ({ch:6s}): p = {p:.4f}")

        # -------------------------------------------------------------
        # TEST 2: Epistemic Void (Causal Uncertainty over Relations)
        # -------------------------------------------------------------
        print("\n" + "="*60)
        print("TEST 2: Epistemic Void Tensor & Vocabulary Pullback")
        print("="*60)

        # Retrieve top-k experiences from memory reservoir
        m, enc = model.memory, model.encoder
        eligible = m.occupied & m.internal_valid & (m.count == 1) & (m.internal_stamp == m.last_write)
        qs = enc(h[0]) # [1, enc_dim]
        bank_s = enc(m.source)
        scores = F.normalize(qs, dim=-1) @ F.normalize(bank_s, dim=-1).T
        scores = scores.masked_fill(~eligible[None], -1e4)
        k = min(m.topk, int(eligible.sum().item()))
        values, slots = scores.topk(k, -1)

        old_s = m.source[slots[0]] # [k, d]
        old_a = m.action[slots[0]] # [k, d]
        old_y = m.internal_next[slots[0]] # [k, d]

        zs, za, zy = enc(old_s), enc(old_a), enc(old_y)
        base_old = model.long_predictor(torch.cat((zs, za), -1)) # [k, enc_dim]
        # Residual variations among retrieved relations:
        # In feature space: zy - base_old
        # Or in raw representation space:
        # Let's see differences in predicted successor states old_y
        diffs = old_y - old_y.mean(dim=0, keepdim=True) # [k, d]
        cov_matrix = (diffs.T @ diffs) / (k - 1 + 1e-12) # [d, d]

        # Epistemic Void Pullback to vocabulary:
        # G(i) = W_U[i] @ Cov @ W_U[i].T
        # Variance of each token under relational disagreement
        tok_var = (W_U @ cov_matrix * W_U).sum(dim=-1) # [256]
        top_uncertain = torch.topk(tok_var, 10)

        print(f"Retrieved {k} active relations from memory reservoir.")
        print(f"Covariance trace (Total epistemic uncertainty): {float(cov_matrix.trace()):.4f}")
        print("Top-10 Concepts with HIGHEST Relational Uncertainty (Causal Gap X=?):")
        for idx, (var_val, tid) in enumerate(zip(top_uncertain.values.tolist(), top_uncertain.indices.tolist())):
            ch = repr(chr(tid)) if 32 <= tid <= 126 else f"0x{tid:02x}"
            print(f"  {idx+1:2d}. Token {tid:3d} ({ch:6s}): uncertainty variance = {var_val:.4f}")

        # Check if '?' (63) is in top percentiles
        rank_q = int((tok_var > tok_var[63]).sum().item()) + 1
        print(f"Question mark '?' (ASCII 63) uncertainty rank: {rank_q} / 256 (variance = {float(tok_var[63]):.4f})")

        # -------------------------------------------------------------
        # TEST 3: Mind Wandering during Quiet Flow (No External Inputs)
        # -------------------------------------------------------------
        print("\n" + "="*60)
        print("TEST 3: J-Space Token Trajectory under Quiet Thinking Steps")
        print("="*60)

        s0 = recurrent.clone()
        p0 = [x.clone() for x in momenta]

        for steps in [0, 1, 4, 16, 64]:
            s_q, p_q = quiet_packed(base_lm, s0.clone(), [x.clone() for x in p0],
                                    dt=engine.bfss_dt, frequency=engine.bfss_frequency,
                                    coupling=engine.bfss_coupling, steps=steps)
            _, _, _, h_q, _ = model.predict_observed(byte, s_q, logp)
            h_q_vec = h_q[0, 0]
            rel_q, _ = local_relation(model, h_q, h_q, ridge=engine.relation_ridge, scale=engine.relation_scale)
            p_q_read = decision_momentum_read(model, p_q, h_q)
            wood_q = next_boundary_prediction(model, h_q, h_q) + model.momentum_to_boundary(p_q_read) + rel_q
            h_lens_q = h_q_vec + wood_q[0]

            probs_q = F.softmax(h_lens_q @ W_U.T, dim=-1)
            top3 = torch.topk(probs_q, 3)
            top_str = ", ".join([f"{repr(chr(t)) if 32<=t<=126 else f'0x{t:02x}'}:{p:.3f}"
                                for p, t in zip(top3.values.tolist(), top3.indices.tolist())])
            s_dist = float((s_q - s0).norm())
            print(f"Quiet Step {steps:2d} (S-drift: {s_dist:6.3f}): Top-3 J-space: [{top_str}]")


if __name__ == '__main__':
    run_tests()
