"""Empirical comparison on real OWT text: Standard GDN vs BFSS Formulations.

Uses the official GatedDeltaNetLM.predict_logits API to guarantee 100% numerical parity:
- Baseline: Native Gated DeltaNet (pure sequential forward).
- Arm 1: BFSS Scheme A (Phase-Space Sync Kick: S updated via Delta, P kicked by Delta, tracking Q).
- Arm 2: BFSS Scheme B (Pure Impulse: Delta kicks P; S advances via Hamiltonian momentum flow dt*P).
- Arm 3: BFSS Scheme C (Thought-Interleaved: 2 quiet flow steps at punctuation ., !, ?, \n).

Evaluates on 512 bytes of continuous real OpenWebText (OWT).
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

from scripts.ib_local.bfss_gdn_capture import matrix_view, quiet_packed, observed_momentum
from scripts.ib_local.bfss_gdn_reference import quiet_step, rotation_charge
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM


def load_model_and_data(num_tokens: int = 512):
    cp_path = 'results/cgb_bfss_gdn_reference_20261110/bound_question_shadow_admission_4/last.pt.gz'
    print(f"Loading checkpoint: {cp_path}...", flush=True)
    with gzip.open(cp_path, 'rb') as f:
        st = torch.load(f, map_location='cpu', weights_only=False)

    lm = GatedDeltaNetLM(vocab_size=256, d=128, layers=4, heads=4,
                         recurrence='decay_then_delta', conv_kernel=4, mlp_hidden=256)
    base_keys = {k.replace('body.base.', ''): v for k, v in st['model'].items() if 'body.base.' in k}
    lm.load_state_dict(base_keys, strict=False)
    lm.eval()

    root = Path('data/ib_owt_10000')
    tok_info = json.loads((root / 'tokenizer.json').read_text())
    offset = tok_info['offset']
    train_mmap = np.load(root / 'train.npy', mmap_mode='r')

    # Continuous slice
    start_idx = 10000
    raw = np.asarray(train_mmap[start_idx:start_idx + num_tokens * 2])
    valid_bytes = (raw[raw >= offset] - offset).astype(np.uint8)
    tokens = torch.from_numpy(valid_bytes[:num_tokens]).long().unsqueeze(0) # [1, T]
    print(f"Prepared OWT test sequence of length {tokens.shape[1]} bytes.", flush=True)

    recurrent_init = st['engine']['recurrent'].clone() # [4, 1, 5248]
    momenta_init = [p.clone() for p in st['engine']['bfss_momenta']] # 4 x [1, 4, 32, 32]

    return lm, tokens, recurrent_init, momenta_init


@torch.no_grad()
def evaluate_baseline_gdn(lm, tokens, recurrent_init):
    """Arm 0: Native full sequence Gated DeltaNet forward pass."""
    logits, next_state, _ = lm.predict_logits(tokens, recurrent_init.clone())
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = tokens[:, 1:].contiguous()
    nll = F.cross_entropy(shift_logits.view(-1, 256), shift_targets.view(-1), reduction='mean')
    return {
        "nll": float(nll),
        "ppl": float(math.exp(min(nll, 20))),
        "final_state": next_state
    }


@torch.no_grad()
def evaluate_bfss_modes(lm, tokens, recurrent_init, momenta_init, mode="sync_kick"):
    """Token-by-token evaluation testing BFSS integrations."""
    B, T = tokens.shape
    curr_state = recurrent_init.clone()
    momenta = [p.clone() for p in momenta_init]

    all_logits = []

    for t in range(T - 1):
        x_t = tokens[:, t:t+1]
        byte_val = x_t[0, 0].item()

        # Step 1 token through Gated DeltaNet
        logits_t, next_state, diag = lm.predict_logits(x_t, curr_state)
        all_logits.append(logits_t)

        if mode == "sync_kick":
            # Scheme A: S updated by GDN, momenta P kicked by the innovation
            # Innovation Delta is captured from state change
            for l, layer in enumerate(lm.layers):
                s_old = matrix_view(layer, curr_state[l])
                s_new = matrix_view(layer, next_state[l])
                # In decay_then_delta, alpha is around 0.95~0.99
                # P kick tracks Delta: P_next = 0.95 * P + (s_new - 0.95 * s_old)
                delta_s = s_new - 0.95 * s_old
                momenta[l] = 0.95 * momenta[l] + delta_s
            curr_state = next_state

        elif mode == "pure_impulse":
            # Scheme B: Delta ONLY kicks P; S moves via velocity dt * P!
            for l, layer in enumerate(lm.layers):
                s_old = matrix_view(layer, curr_state[l])
                s_new = matrix_view(layer, next_state[l])
                delta_s = s_new - 0.95 * s_old
                # Impulse kicks momentum:
                momenta[l] = 0.95 * momenta[l] + delta_s
                # Position S is updated by momentum P instead of direct jump:
                # S = 0.95 * S + dt * P_kicked
                dt = 0.1
                s_driven = 0.95 * s_old + dt * momenta[l]
                # Replace the matrix portion in next_state
                next_state[l, :, :layer.matrix_size] = s_driven.flatten(1)
            curr_state = next_state

        elif mode == "bfss_interleaved":
            # Scheme C: Standard write, plus quiet flow Hamiltonian evolution at punctuation
            curr_state = next_state
            if byte_val in [46, 63, 33, 10]: # '.', '?', '!', '\n'
                curr_state, momenta = quiet_packed(lm, curr_state, momenta,
                                                   dt=0.01, frequency=0.8, coupling=0.01, steps=2)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    pred_logits = torch.cat(all_logits, dim=1) # [1, T-1, 256]
    targets = tokens[:, 1:]
    nll = F.cross_entropy(pred_logits.view(-1, 256), targets.view(-1), reduction='mean')

    # Layer 2 Q charge at end
    s2 = matrix_view(lm.layers[2], curr_state[2])
    p2 = momenta[2]
    qk_l2, _ = rotation_charge(s2, p2)
    qk_norm = float(qk_l2[0].norm())
    s_norm = float(s2.norm())

    return {
        "nll": float(nll),
        "ppl": float(math.exp(min(nll, 20))),
        "qk_norm_l2": qk_norm,
        "s_norm_l2": s_norm
    }


def main():
    lm, tokens, recurrent_init, momenta_init = load_model_and_data(num_tokens=512)

    print("\n" + "="*65, flush=True)
    print("RUNNING OWT BENCHMARK: Baseline GDN vs BFSS Formulations", flush=True)
    print("="*65, flush=True)

    # 1. Native Baseline
    res_base = evaluate_baseline_gdn(lm, tokens, recurrent_init)
    print(f"1. Native Gated DeltaNet (Baseline):", flush=True)
    print(f"   NLL: {res_base['nll']:.4f} nats/byte | PPL: {res_base['ppl']:.2f}", flush=True)

    # 2. Scheme A: Synchronous Phase-Space Kick
    res_sync = evaluate_bfss_modes(lm, tokens, recurrent_init, momenta_init, mode="sync_kick")
    print(f"\n2. BFSS Scheme A (Phase-Space Sync Kick):", flush=True)
    print(f"   NLL: {res_sync['nll']:.4f} nats/byte (Delta NLL: {res_sync['nll'] - res_base['nll']:+.4f}) | PPL: {res_sync['ppl']:.2f}", flush=True)
    print(f"   Layer 2 ||Q_K||: {res_sync['qk_norm_l2']:.4f}, ||S||: {res_sync['s_norm_l2']:.4f}", flush=True)

    # 3. Scheme B: Pure Impulse Injection (Delta to P only, S driven by P)
    res_impulse = evaluate_bfss_modes(lm, tokens, recurrent_init, momenta_init, mode="pure_impulse")
    print(f"\n3. BFSS Scheme B (Pure Impulse: Delta kicks ONLY P, S driven):", flush=True)
    print(f"   NLL: {res_impulse['nll']:.4f} nats/byte (Delta NLL: {res_impulse['nll'] - res_base['nll']:+.4f}) | PPL: {res_impulse['ppl']:.2f}", flush=True)
    print(f"   Layer 2 ||Q_K||: {res_impulse['qk_norm_l2']:.4f}, ||S||: {res_impulse['s_norm_l2']:.4f}", flush=True)

    # 4. Scheme C: BFSS Thought-Interleaving at Punctuation
    res_think = evaluate_bfss_modes(lm, tokens, recurrent_init, momenta_init, mode="bfss_interleaved")
    print(f"\n4. BFSS Scheme C (Thought-Interleaved: 2 Quiet Steps at Stops):", flush=True)
    print(f"   NLL: {res_think['nll']:.4f} nats/byte (Delta NLL: {res_think['nll'] - res_base['nll']:+.4f}) | PPL: {res_think['ppl']:.2f}", flush=True)
    print(f"   Layer 2 ||Q_K||: {res_think['qk_norm_l2']:.4f}, ||S||: {res_think['s_norm_l2']:.4f}", flush=True)

    print("\n" + "="*65, flush=True)
    print("SUMMARY COMPARISON TABLE (Real OWT Text)", flush=True)
    print("="*65, flush=True)
    print(f"{'Variant':<35} | {'NLL (nats)':<10} | {'PPL':<8} | {'||Q_K^(L2)||':<12}", flush=True)
    print("-" * 72, flush=True)
    print(f"{'Native Gated DeltaNet (Baseline)':<35} | {res_base['nll']:<10.4f} | {res_base['ppl']:<8.2f} | {'N/A (No P)':<12}", flush=True)
    print(f"{'BFSS Scheme A (Sync Phase Kick)':<35} | {res_sync['nll']:<10.4f} | {res_sync['ppl']:<8.2f} | {res_sync['qk_norm_l2']:<12.4f}", flush=True)
    print(f"{'BFSS Scheme B (Pure Impulse to P)':<35} | {res_impulse['nll']:<10.4f} | {res_impulse['ppl']:<8.2f} | {res_impulse['qk_norm_l2']:<12.4f}", flush=True)
    print(f"{'BFSS Scheme C (Thinking at Stops)':<35} | {res_think['nll']:<10.4f} | {res_think['ppl']:<8.2f} | {res_think['qk_norm_l2']:<12.4f}", flush=True)


if __name__ == '__main__':
    main()
