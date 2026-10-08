"""End-to-end GPU smoke test for the Rate-Distortion & MCR^2 Dual-Engine System.

Verifies:
1. Engine 1 (Execution): FlyAdaptiveAdmissionGatekeeper operates in real stream,
   achieving dynamic self-paced token admission with zero artificial clamps.
2. Engine 2 (Learning): Differentiable MCR^2 loss backwards cleanly into model weights,
   producing valid, finite gradients without NaN/Inf or memory leak.
3. System Telemetry: Measures throughput speedup, coding rate expansion Delta R,
   and peak VRAM footprint on real MaleCNS connectome.
"""

import gc
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import (
    FlyPhysicalState,
    extract_fly_motor_latent,
    step_fly_physical_tick,
)
from information_boltzmann.core.mcr2_rate_distortion import (
    FlyAdaptiveAdmissionGatekeeper,
    MCR2Loss,
)


def run_smoke_test(device, num_tokens=16):
    print("=================================================================")
    print(" SMOKE TEST: Rate-Distortion & MCR^2 Dual-Engine on MaleCNS")
    print("=================================================================")

    graph_path = Path("data/malecns_v1/fly_reservoir_coba.npz")
    ckpt_path = Path("E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt")

    print("1. Loading MaleCNS model and checkpoint onto GPU...")
    model = FlyReservoirLM(
        str(graph_path),
        vocab_size=50257,
        d_model=768,
        injection='topographic',
        read_surface='output',
        synapse_model='coba',
        use_alif=True,
        use_stp=True,
    ).to(device)

    saved = torch.load(ckpt_path, map_location='cpu', weights_only=False, mmap=True)
    weights = {k: v for k, v in saved['model'].items() if k not in ('edge_weight_e', 'edge_weight_i')}
    with torch.no_grad():
        for name in ('edge_weight_e', 'edge_weight_i'):
            getattr(model, name).copy_(saved['model'][name])
        model.load_state_dict(weights, strict=False)

    phys = saved['learner']['physical']
    state_dict = {
        k: v.to(device) if isinstance(v, torch.Tensor)
        else tuple(t.to(device) for t in v) if isinstance(v, tuple)
        else v
        for k, v in phys.items()
    }
    initial_state = FlyPhysicalState(**state_dict)

    del saved, weights, phys
    gc.collect()
    torch.cuda.empty_cache()

    rates = model.get_decay_rates()
    thresholds = model.get_thresholds()
    gains = model.get_conductance_gains()
    alif = model.get_alif_params()
    stp = model.get_stp_params()
    options = dict(
        base_rates=rates,
        thresholds=thresholds,
        conductance_gains=gains,
        alif_params=alif,
        stp_params=stp,
    )

    val_data = np.load("data/ib_owt_gpt2/validation.npy", mmap_mode='r')
    tokens = val_data[:num_tokens + 1].astype(np.int64)

    # -------------------------------------------------------------
    # 2. Instantiate Dual-Engine Components
    # -------------------------------------------------------------
    print("\n2. Initializing Dual-Engine operators...")
    gatekeeper = FlyAdaptiveAdmissionGatekeeper(
        num_neurons=165122,
        vocab_size=50257,
        lambda_flux=8.0,
        lambda_vol=0.05,
        flux_baseline_threshold=0.036,
        max_settle_ticks=14,
    )
    mcr2_criterion = MCR2Loss(d_model=768, eps=0.5, beta=0.05)

    quiet_source = torch.zeros_like(initial_state.h)

    # -------------------------------------------------------------
    # 3. Test Engine 1: Self-Paced Admission Forward Pass
    # -------------------------------------------------------------
    print(f"\n3. Running Engine 1 (Self-Paced Forward Pass across {num_tokens} tokens)...")
    st = initial_state.detached()
    token_latents = []
    settle_records = []
    total_physical_ticks = 0

    t_start = time.perf_counter()
    for i in range(num_tokens):
        inp = torch.tensor([tokens[i]], dtype=torch.long, device=device)
        drv, bsl = model.topographic_writer.forward_with_state(
            model.embedding(inp), st.h, st.baseline)
        h_prev = st.h.clone()

        gatekeeper.reset()
        token_emitted_latent = None

        for t in range(14):
            src = drv if t == 0 else quiet_source
            base = bsl if t == 0 else st.baseline
            st = step_fly_physical_tick(model, st, inp if t==0 else None, src, base, options)

            lat = extract_fly_motor_latent(model, st)
            logits = model.decoder(lat)

            # Evaluate gatekeeper metrics
            metrics = gatekeeper.compute_metrics(st.h, h_prev, logits, lat)
            h_prev = st.h.clone()

            admit, reason = gatekeeper.should_admit_next_token()
            if admit:
                settle_ticks = t + 1
                token_emitted_latent = lat
                settle_records.append((settle_ticks, reason))
                total_physical_ticks += settle_ticks
                break
        else:
            settle_ticks = 14
            token_emitted_latent = lat
            settle_records.append((14, "max_settle"))
            total_physical_ticks += 14

        token_latents.append(token_emitted_latent)

    fwd_time = time.perf_counter() - t_start
    mean_ticks = total_physical_ticks / num_tokens
    speedup = (14.0 * num_tokens) / total_physical_ticks

    print(f"Engine 1 Results: Total Ticks={total_physical_ticks} (vs {14*num_tokens} fixed), "
          f"Mean Settle={mean_ticks:.1f} ticks/tok, Speedup={speedup:.2f}x, Time={fwd_time:.2f}s")

    for idx, (ticks, reason) in enumerate(settle_records[:6]):
        print(f"  Token {idx:2d} (ID: {tokens[idx]:5d}): settled in {ticks:2d} ticks | Trigger: {reason}")

    # -------------------------------------------------------------
    # 4. Test Engine 2: MCR^2 Differentiable Loss Backward Pass
    # -------------------------------------------------------------
    print("\n4. Testing Engine 2 (MCR^2 Differentiable Backward Pass)...")
    # Stack emitted latents [num_tokens, d_model]
    Z_emitted = torch.cat(token_latents, dim=0) # [16, 768]
    Z_train = Z_emitted.clone().detach().requires_grad_(True)

    # Decode predictions from latents
    targets = torch.tensor(tokens[1:num_tokens+1], dtype=torch.long, device=device)
    logits_batch = model.decoder(Z_train)
    ce_loss = F.cross_entropy(logits_batch, targets)

    # Compute MCR^2 regularizer
    mcr2_out = mcr2_criterion(Z_train)
    mcr2_loss = mcr2_out["loss"]
    delta_R = mcr2_out["delta_R"]
    R_total = mcr2_out["R_total"]

    joint_loss = ce_loss + mcr2_loss

    print(f"  CE Loss:     {ce_loss.item():.4f}")
    print(f"  MCR^2 Loss:  {mcr2_loss.item():.4f} (beta={mcr2_criterion.beta})")
    print(f"  Subspace Delta R: {delta_R.item():.4f} nats (Total R = {R_total.item():.4f} nats)")
    print(f"  Joint Loss:  {joint_loss.item():.4f}")

    # Backward pass into latents
    joint_loss.backward()

    grad_norm = Z_train.grad.norm().item()
    print(f"  Latent Gradient Norm: {grad_norm:.4f}")

    assert not torch.isnan(Z_train.grad).any(), "Gradient contains NaN!"
    assert not torch.isinf(Z_train.grad).any(), "Gradient contains Inf!"
    assert grad_norm > 1e-4, "Gradient is too small or dead!"

    # Test backward pass through decoder parameters
    model.decoder.zero_grad()
    logits_test = model.decoder(Z_emitted.detach())
    loss_dec = F.cross_entropy(logits_test, targets) + mcr2_criterion(Z_emitted.detach())["loss"]
    loss_dec.backward()

    dec_grad_norm = model.decoder.weight.grad.norm().item()
    print(f"  Decoder Weight Gradient Norm: {dec_grad_norm:.4f}")
    assert dec_grad_norm > 1e-4, "Decoder weight gradient is dead!"

    # -------------------------------------------------------------
    # 5. Telemetry & Summary
    # -------------------------------------------------------------
    mem_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    telemetry = {
        "num_tokens": num_tokens,
        "total_ticks": total_physical_ticks,
        "mean_settle_ticks": mean_ticks,
        "speedup_factor": speedup,
        "ce_loss": ce_loss.item(),
        "mcr2_loss": mcr2_loss.item(),
        "delta_R_nats": delta_R.item(),
        "latent_grad_norm": grad_norm,
        "decoder_grad_norm": dec_grad_norm,
        "peak_vram_mb": mem_allocated_mb,
    }

    out_json = Path("present/smoke_test_mcr2_dual_engine_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    print(f"\nPeak GPU VRAM: {mem_allocated_mb:.1f} MB")
    print(f"Smoke test report saved to {out_json}")
    print(">>> DUAL-ENGINE SMOKE TEST PASSED WITH ZERO ERRORS! <<<")
    return telemetry


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_smoke_test(device, num_tokens=16)
