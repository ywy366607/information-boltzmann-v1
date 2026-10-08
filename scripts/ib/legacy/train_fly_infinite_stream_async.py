"""Continuous Infinite-Stream O(1) Online Learning Trainer.

Streams token-by-token continuously from OpenWebText without any sequence batching
or chunk boundaries (no 128, 1024, or 2048 chunking).

Design & Guardrails:
  1. Infinite Stream: Never resets reservoir physical dynamics (h, ring, ge, gi, b, x, u).
  2. Isolated Control: 25.32M internal connectome synapses remain frozen (identical to baseline).
  3. Online Trained: Readout layer (W_read, decoder bias) and 27-superclass hyperparameters
     (timescales tau_m, thresholds theta_0, ALIF adaptation, STP parameters, conductance gains).
  4. Closed Graphs: Completely bypasses backward autograd tape; updates parameters online via
     analytical O(1) gradients + AdamW optimizer.
  5. Constant Memory: Strictly O(1) spatial complexity, VRAM < 600 MB on NVIDIA GTX 1650.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.eprop_credit_assignment import (
    EPropCreditAssignment,
    EPropEligibilityState,
    DopamineReceptorState,
    SensoryWriterEligibilityState,
)
from information_boltzmann.core.life_form_evaluation import (
    FourPillarEvaluator,
    StreamingConnectomeLearner,
)


def atomic_json(path: Path, payload: dict, *, required: bool = False) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    encoded = json.dumps(payload, allow_nan=False)
    for attempt in range(60):
        try:
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, path)
            return
        except PermissionError:
            time.sleep(0.05 * min(attempt + 1, 4))
    if required:
        raise PermissionError(f"Could not update {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    parser.add_argument("--graph", type=Path, default=Path("data/malecns_v1/fly_reservoir_coba.npz"))
    parser.add_argument("--resume", type=Path,
                        default=Path("results/q8_fly_reservoir_topographic_coba_alif_stp_3000/best.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/q8_fly_infinite_stream_owt"))
    parser.add_argument("--total-tokens", type=int, default=100000,
                        help="Total continuous streaming tokens to process")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="AdamW learning rate for online streaming parameters")
    parser.add_argument("--validate-every-tokens", type=int, default=5000,
                        help="Validation interval in streaming tokens")
    parser.add_argument("--log-every-tokens", type=int, default=100,
                        help="Console progress display interval in tokens")
    parser.add_argument("--grad-accum-tokens", type=int, default=4,
                        help="Accumulate analytical gradients over mini-streaming window before AdamW step")
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--pretrained-embedding", type=Path, default=Path("data/gpt2_model.safetensors"),
                        help="Path to official pretrained GPT-2 weights (.safetensors)")
    parser.add_argument("--unfreeze-decoder", action="store_true", default=True,
                        help="Train decoder.weight alongside output_read in AdamW")
    parser.add_argument("--unfreeze-synapses", action="store_true", default=True,
                        help="Unfreeze all 25.32M biological connectome synapses via O(1) e-prop")
    parser.add_argument("--freeze-synapses", dest="unfreeze_synapses", action="store_false",
                        help="Freeze all 25.32M biological connectome synapses")
    parser.add_argument("--lr-synapse", type=float, default=1e-5,
                        help="Plasticity learning rate for whole-brain synapses")
    parser.add_argument("--synapse-update-interval", type=int, default=2,
                        help="Interval in tokens for applying whole-brain synaptic updates (eligibility traces remain updated every token)")
    parser.add_argument("--decoder-bias", action="store_true", default=True,
                        help="Enable explicit learnable decoder.bias to absorb static unigram distribution")
    parser.add_argument("--no-decoder-bias", dest="decoder_bias", action="store_false",
                        help="Disable decoder.bias")
    parser.add_argument("--read-norm-init", type=float, default=0.1,
                        help="Initial scale (gamma) for read_norm to prevent random initial residual logits from overpowering the unigram prior")
    parser.add_argument("--unfreeze-sensory", action="store_true", default=True,
                        help="Unfreeze all 15,912 sensory projection synapses via input eligibility traces")
    parser.add_argument("--freeze-sensory", dest="unfreeze_sensory", action="store_false",
                        help="Freeze sensory projection synapses")
    parser.add_argument("--lr-sensory", type=float, default=1e-4,
                        help="Plasticity learning rate for sensory projection matrices")
    parser.add_argument("--dopamine-k-on", type=float, default=0.2,
                        help="Binding rate for dopamine receptor dynamics")
    parser.add_argument("--dopamine-k-off", type=float, default=0.05,
                        help="Unbinding/decay rate for dopamine receptor dynamics")
    parser.add_argument("--dopamine-q0", type=float, default=0.05,
                        help="Baseline unmodulated plasticity floor for dopamine gating")
    parser.add_argument("--eval-shock-tokens", type=int, default=150,
                        help="Tokens for plastic environmental shock & re-adaptation curve (Pillar 2)")
    parser.add_argument("--eval-ebb-a", type=int, default=40,
                        help="Tokens for Sequence A in continuous Ebbinghaus savings evaluation (Pillar 3)")
    parser.add_argument("--eval-ebb-b", type=int, default=80,
                        help="Tokens for intervening Stream B in continuous Ebbinghaus savings evaluation (Pillar 3)")
    parser.add_argument("--eval-ftle-steps", type=int, default=50,
                        help="Steps for full continuous physical state FTLE via Benettin shadow method (Pillar 4)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required for streaming connectome training")

    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading MaleCNS connectome from {args.graph}...", flush=True)
    model = FlyReservoirLM(
        args.graph,
        vocab_size=50257,
        d_model=args.d_model,
        injection="topographic",
        read_surface="output",
        synapse_model="coba",
        use_alif=True,
        use_stp=True,
        decoder_bias=args.decoder_bias,
    ).cuda()

    val_unigram_nll = 7.6076
    if args.pretrained_embedding is not None and args.pretrained_embedding.exists() and args.d_model == 768:
        print(f"Loading official pretrained GPT-2 weights from {args.pretrained_embedding}...", flush=True)
        from safetensors.torch import load_file
        gpt2_weights = load_file(str(args.pretrained_embedding))
        wte = gpt2_weights["wte.weight"].cuda()
        with torch.no_grad():
            model.embedding.weight.copy_(wte)
            model.decoder.weight.copy_(wte)
            if hasattr(model.decoder, "bias") and model.decoder.bias is not None:
                print(f"Initializing decoder.bias from empirical unigram distribution...", flush=True)
                train_raw = np.load(args.data / "train.npy", mmap_mode="r")
                train_tokens_t = torch.from_numpy(train_raw[:].astype(np.int64))
                counts = torch.bincount(train_tokens_t, minlength=50257).float()
                probs_unigram = (counts + 1.0) / (train_tokens_t.numel() + 50257)
                log_p_unigram = torch.log(probs_unigram).cuda()
                model.decoder.bias.copy_(log_p_unigram)

                val_raw = np.load(args.data / "validation.npy", mmap_mode="r")
                val_tokens_t = torch.from_numpy(val_raw[:].astype(np.int64))
                val_unigram_nll = float(-log_p_unigram.cpu()[val_tokens_t].mean().item())
                print(f"Initialized decoder.bias from smoothed unigram prior! Val Unigram NLL = {val_unigram_nll:.4f} nats", flush=True)
            if hasattr(model, "read_norm"):
                model.read_norm.weight.fill_(args.read_norm_init)
                print(f"Initialized read_norm.weight (gamma) = {args.read_norm_init} to balance residual logits with unigram prior.", flush=True)
        print("Successfully initialized 768-dim embedding and decoder from official GPT-2 weights!", flush=True)
    elif args.resume is not None and args.resume.exists():
        print(f"Resuming trained weights from {args.resume}...", flush=True)
        saved = torch.load(args.resume, map_location="cpu")
        graph_keys = {
            "edge_index", "edge_pre", "edge_post", "edge_weight", "delay_splits",
            "edge_pre_e", "edge_post_e", "edge_weight_e", "delay_splits_e",
            "edge_pre_i", "edge_post_i", "edge_weight_i", "delay_splits_i",
            "dan_edge_pre", "dan_edge_post", "dan_edge_weight", "lambda_0"
        }
        model_dict = {}
        for k, v in saved["model"].items():
            if k in graph_keys:
                continue
            if k == "output_read.weight" and v.shape != model.output_read.weight.shape:
                if hasattr(model, "read_indices") and v.shape[-1] == model.n_neurons:
                    v = v[:, model.read_indices.cpu()]
            model_dict[k] = v
        model.load_state_dict(model_dict, strict=False)
        print(f"Successfully loaded checkpoint Step {saved.get('step', 'N/A')}, Best Val NLL: {saved.get('best_validation_nll', 'N/A')}", flush=True)

    # Disable all autograd graph construction; O(1) streaming updates use analytical gradients
    model.requires_grad_(False)
    torch.set_grad_enabled(False)

    # Trainable parameters grouped cleanly:
    # 1. Weights subject to mild L2 weight decay: output_read.weight and optionally decoder.weight
    # 2. RMSNorm gain & biophysical parameters: zero weight decay (strictly no shrinkage drift)
    param_groups = [
        {"params": [model.output_read.weight], "weight_decay": 1e-4},
    ]
    if hasattr(model, "read_norm"):
        param_groups.append({"params": [model.read_norm.weight], "weight_decay": 0.0})
        grad_norm_acc = torch.zeros_like(model.read_norm.weight)
    else:
        grad_norm_acc = None

    if args.unfreeze_decoder:
        param_groups[0]["params"].append(model.decoder.weight)
        grad_w_dec_acc = torch.zeros_like(model.decoder.weight)
    else:
        grad_w_dec_acc = None

    if hasattr(model.decoder, "bias") and model.decoder.bias is not None:
        param_groups.append({"params": [model.decoder.bias], "weight_decay": 0.0})
        grad_b_dec_acc = torch.zeros_like(model.decoder.bias)
    else:
        grad_b_dec_acc = None

    param_groups.append({
        "params": [
            model.log_tau_m,
            model.log_threshold,
            model.log_beta,
            model.log_tau_a,
            model.logit_u0,
            model.log_tau_fac,
            model.log_tau_rec,
            model.log_g_e,
            model.log_g_i,
        ],
        "weight_decay": 0.0,
    })

    if args.unfreeze_sensory and getattr(model, "topographic_writer", None) is not None:
        param_groups.append({
            "params": [
                model.topographic_writer.gate_linear.weight,
                model.topographic_writer.gate_linear.bias,
            ],
            "weight_decay": 1e-4,
        })
        grad_gate_w_acc = torch.zeros_like(model.topographic_writer.gate_linear.weight)
        grad_gate_b_acc = torch.zeros_like(model.topographic_writer.gate_linear.bias)
    else:
        grad_gate_w_acc = None
        grad_gate_b_acc = None

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    train_data = np.load(args.data / "train.npy", mmap_mode="r")
    val_data = np.load(args.data / "validation.npy", mmap_mode="r")
    superclass_ids = model.superclass_id.cuda()

    config = {
        "architecture": "FlyReservoir-InfiniteStream-O1-MaleCNS-Topographic-COBA-ALIF-STP",
        "neurons": model.n_neurons,
        "edges": int(model.edge_weight_e.numel() + model.edge_weight_i.numel()),
        "stream_paradigm": "Continuous Infinite Streaming (Zero chunking, strictly O(1) memory, closed autograd graphs)",
        "plasticity_scope": "Whole-Brain 25.32M Synapses Unfrozen (O(1) e-prop) + Readout Layer + Decoder Bias + 27-Superclass Hyperparameters" if args.unfreeze_synapses else "Readout layer + 27-superclass hyperparameters (internal 25.32M synapses frozen)",
        "unfreeze_synapses": bool(args.unfreeze_synapses),
        "lr_synapse": float(args.lr_synapse),
        "decoder_bias": bool(args.decoder_bias),
        "dopamine_modulation": "Bio-structural DAN broadcast + Direct Feedback Alignment",
        "total_tokens": args.total_tokens,
        "grad_accum_tokens": args.grad_accum_tokens,
        "lr": args.lr,
    }
    atomic_json(args.output / "config.json", config, required=True)

    file_lock = threading.Lock()

    def log(row: dict) -> None:
        with file_lock:
            with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")

    val_cursor = 0
    state_meta = {
        "best_live_val": 999.0,
    }

    print("Initializing auxiliary evaluation model with shared connectome graph...", flush=True)
    eval_model = FlyReservoirLM(
        args.graph,
        vocab_size=50257,
        d_model=args.d_model,
        injection="topographic",
        read_surface="output",
        synapse_model="coba",
        use_alif=True,
        use_stp=True,
        decoder_bias=args.decoder_bias,
    ).cuda()
    eval_model.requires_grad_(False)

    for k in model._buffers.keys():
        eval_model._buffers[k] = model._buffers[k]
    eval_model.embedding.weight = model.embedding.weight
    if hasattr(eval_model.decoder, "bias") and eval_model.decoder.bias is not None and model.decoder.bias is not None:
        eval_model.decoder.bias.copy_(model.decoder.bias)
    if hasattr(eval_model, "read_norm") and hasattr(model, "read_norm"):
        eval_model.read_norm.weight.copy_(model.read_norm.weight)
    eval_model.decoder.weight.copy_(model.decoder.weight)
    eval_model.output_read.weight.copy_(model.output_read.weight)
    if hasattr(eval_model, "topographic_writer") and getattr(model, "topographic_writer", None) is not None:
        eval_model.topographic_writer.proj_vis.weight = model.topographic_writer.proj_vis.weight
        eval_model.topographic_writer.proj_chemo.weight = model.topographic_writer.proj_chemo.weight
        eval_model.topographic_writer.proj_mech.weight = model.topographic_writer.proj_mech.weight
        eval_model.topographic_writer.gate_linear.weight = model.topographic_writer.gate_linear.weight
        eval_model.topographic_writer.gate_linear.bias = model.topographic_writer.gate_linear.bias
    torch.cuda.empty_cache()

    eval_stream = torch.cuda.Stream()
    eval_learner = StreamingConnectomeLearner(
        model=eval_model,
        optimizer=None,
        lr=args.lr,
        grad_accum_tokens=args.grad_accum_tokens,
        train_decoder_weight=False,
    )
    eval_learner.unigram_baseline_nll = val_unigram_nll
    eval_thread: Optional[threading.Thread] = None

    def run_eval_job(
        token_step: int,
        weights_dict: dict,
        phys_state: tuple,
        cursor_val: int,
        ema_loss_snapshot: float,
    ) -> None:
        torch.set_grad_enabled(False)
        t_eval_start = time.perf_counter()
        with torch.cuda.stream(eval_stream):
            eval_model.load_state_dict(weights_dict, strict=False)
            h_snap, ring_snap, syn_snap = phys_state
            eval_learner.h.copy_(h_snap)
            for idx_r, r in enumerate(ring_snap):
                eval_learner.ring[idx_r].copy_(r)
            for k, v in syn_snap.items():
                eval_learner.syn_state[k].copy_(v)
            eval_learner.update_cached_params()

            four_pillar = FourPillarEvaluator.evaluate(
                learner=eval_learner,
                val_tokens=val_data,
                cursor=cursor_val,
                shock_tokens=args.eval_shock_tokens,
                ebbinghaus_seq_a_tokens=args.eval_ebb_a,
                ebbinghaus_intervene_tokens=args.eval_ebb_b,
                ftle_steps=args.eval_ftle_steps,
            )

        eval_tokens_consumed = args.eval_shock_tokens + args.eval_ftle_steps + (args.eval_ebb_a + args.eval_ebb_b)
        val_elapsed = time.perf_counter() - t_eval_start

        p1 = four_pillar.pillar_1_prequential
        p2 = four_pillar.pillar_2_shock
        p3 = four_pillar.pillar_3_ebbinghaus
        energy = four_pillar.energy_ledger
        info = four_pillar.information_ledger
        crit = four_pillar.criticality
        conv = four_pillar.convergence

        live_val_nll = p2.plateau_surprise
        with file_lock:
            is_best_live = live_val_nll < state_meta["best_live_val"]
            if is_best_live:
                state_meta["best_live_val"] = live_val_nll

        print(f"\n  +------------------------------------------------------------------------------------------------+", flush=True)
        print(f"  | ASYNC FOUR-PILLAR SCIENTIFIC EVALUATION @ {token_step:7,d} TOKENS (Time: {val_elapsed:.2f}s, {eval_tokens_consumed} tokens)       |", flush=True)
        print(f"  +------------------------------------------------------------------------------------------------+", flush=True)
        print(f"  | [P1 Prequential]    EMA Stream NLL: {ema_loss_snapshot:.4f} | Window Mean NLL: {p1.get('window_mean_nll', ema_loss_snapshot):.4f}", flush=True)
        print(f"  | [P2 Shock Adapt]    S_shock: {p2.shock_surprise:.4f} -> S_plat: {p2.plateau_surprise:.4f} | Delta: {p2.delta_shock:+.4f} | t_1/2: {p2.half_life_tokens:.1f} tok | Elasticity: {p2.elasticity_score:.4f}", flush=True)
        print(f"  | [P3 Ebbinghaus]     Init A: {p3.nll_initial_mean:.4f} -> Relearn A: {p3.nll_relearn_mean:.4f} | Recall Drop: {p3.retention_immediate_drop:+.4f} | Savings: {p3.savings_ratio*100:.2f}% ({p3.acceleration_factor:.1f}x)", flush=True)
        print(f"  | [P4 Energy Ledger]  True Spikes: {energy.cumulative_spikes:,} | Density: {energy.mean_firing_rate_density*100:.3f}% | g_tot: {energy.mean_conductance_per_token:.4f} | Eff: {energy.information_efficiency_nats_per_100k_spikes:.2f} nats/100k spk", flush=True)
        print(f"  | [P4 Info Ledger]    R_eff: {info.effective_rank:.2f} (Motor: {info.rank_motor_eff:.2f} -> W_read: {info.rank_w_read_eff:.2f} -> Norm: {info.rank_rmsnorm_eff:.2f}) | Ctx Gain: {info.net_contextual_gain:+.4f} nats | Roughness: {info.temporal_roughness:.4f}", flush=True)
        print(f"  | [P4 Criticality]    Branching σ: {crit.branching_ratio:.4f} | FTLE λ: {crit.finite_time_lyapunov_exponent:+.4f} | Regime: {crit.regime}", flush=True)
        print(f"  | [P4 Convergence]    DNR: {conv.drift_to_noise_ratio_dnr:.4f} | Trend β: {conv.trend_slope_beta:+.2e} | Status: {conv.status_description}", flush=True)
        print(f"  +------------------------------------------------------------------------------------------------+\n", flush=True)

        log_dict = {
            "token": token_step,
            "best_live_nll": state_meta["best_live_val"],
            **four_pillar.to_dict(),
        }
        log({"kind": "four_pillar_evaluation", **log_dict})

        save_weights = {k: v.cpu() for k, v in weights_dict.items()}
        save_weights["embedding.weight"] = eval_model.embedding.weight.cpu()
        if args.unfreeze_synapses:
            save_weights["edge_weight_e"] = model.edge_weight_e.cpu()
            save_weights["edge_weight_i"] = model.edge_weight_i.cpu()
        if args.unfreeze_sensory and getattr(model, "topographic_writer", None) is not None:
            save_weights["topographic_writer.proj_vis.weight"] = model.topographic_writer.proj_vis.weight.cpu()
            save_weights["topographic_writer.proj_chemo.weight"] = model.topographic_writer.proj_chemo.weight.cpu()
            save_weights["topographic_writer.proj_mech.weight"] = model.topographic_writer.proj_mech.weight.cpu()
            save_weights["topographic_writer.gate_linear.weight"] = model.topographic_writer.gate_linear.weight.cpu()
            save_weights["topographic_writer.gate_linear.bias"] = model.topographic_writer.gate_linear.bias.cpu()
        save_ckpt = {
            "model": save_weights,
            "tokens_streamed": token_step,
            "live_nll": live_val_nll,
            "best_live_nll": state_meta["best_live_val"],
            "four_pillar": log_dict,
            "config": config,
        }

        with file_lock:
            if is_best_live:
                torch.save(save_ckpt, args.output / "best.pt")
                print(f"  >>> [Checkpoint Saved] New Best Shock Plateau NLL: {state_meta['best_live_val']:.4f} -> {args.output / 'best.pt'}\n", flush=True)
            torch.save(save_ckpt, args.output / "last.pt")

    # Initialize continuous physical state vectors (NEVER reset across the entire infinite stream)
    h = torch.zeros(1, model.n_neurons, device="cuda")
    ring = tuple(torch.zeros(1, model.n_neurons, device="cuda") for _ in range(4))
    ge = torch.zeros(1, model.n_neurons, device="cuda")
    gi = torch.zeros(1, model.n_neurons, device="cuda")
    b = torch.zeros(1, model.n_neurons, device="cuda")
    x = torch.ones(1, model.n_neurons, device="cuda")
    u0_init, _, _, _ = model.get_stp_params()
    u = u0_init.detach().clone().expand(1, model.n_neurons).contiguous()

    grad_w_read_acc = torch.zeros_like(model.output_read.weight)
    if grad_norm_acc is None and hasattr(model, "read_norm"):
        grad_norm_acc = torch.zeros_like(model.read_norm.weight)
    if grad_b_dec_acc is None and hasattr(model.decoder, "bias") and model.decoder.bias is not None:
        grad_b_dec_acc = torch.zeros_like(model.decoder.bias)
    grad_thresh_acc = torch.zeros_like(model.log_threshold)
    grad_tau_m_acc = torch.zeros_like(model.log_tau_m)
    grad_beta_acc = torch.zeros_like(model.log_beta)
    grad_tau_a_acc = torch.zeros_like(model.log_tau_a)
    learner = StreamingConnectomeLearner(
        model=model,
        optimizer=optimizer,
        lr=args.lr,
        grad_accum_tokens=args.grad_accum_tokens,
    )
    learner.unigram_baseline_nll = val_unigram_nll

    if grad_w_dec_acc is not None:
        buf_probs = torch.zeros(args.grad_accum_tokens, 50257, device="cuda", dtype=torch.float32)
        buf_z_latent = torch.zeros(args.grad_accum_tokens, args.d_model, device="cuda", dtype=torch.float32)
    else:
        buf_probs = None
        buf_z_latent = None

    if args.unfreeze_synapses:
        print("Initializing O(1) e-prop whole-brain credit assignment for 25.32M synapses...", flush=True)
        eprop = EPropCreditAssignment(
            n_neurons=model.n_neurons,
            d_model=args.d_model,
            vocab_size=50257,
            E_E=model.E_E,
            E_I=model.E_I,
        ).cuda()
        eligibility = EPropEligibilityState.init_zero(
            batch_size=1,
            n_neurons=model.n_neurons,
            device="cuda",
        )
        prev_spikes = torch.zeros(1, model.n_neurons, device="cuda")
    else:
        eprop = None
        eligibility = None
        prev_spikes = None

    if getattr(model, "has_dopamine", False):
        print(f"Initializing DopamineReceptorState across {model.dan_edge_pre.numel():,} dopamine synapses...", flush=True)
        dopamine_state = DopamineReceptorState.init_zero(
            n_neurons=model.n_neurons,
            device="cuda",
            k_on=args.dopamine_k_on,
            k_off=args.dopamine_k_off,
            q_0=args.dopamine_q0,
        )
    else:
        dopamine_state = None

    if args.unfreeze_sensory and getattr(model, "topographic_writer", None) is not None:
        print(f"Initializing SensoryWriterEligibilityState for {model.topographic_writer.n_total:,} sensory projection neurons...", flush=True)
        sensory_eligibility = SensoryWriterEligibilityState.init_zero(
            d_model=args.d_model,
            device="cuda",
            n_vis=model.topographic_writer.n_vis,
            n_chemo=model.topographic_writer.n_chemo,
            n_mech=model.topographic_writer.n_mech,
        )
    else:
        sensory_eligibility = None

    print(f"\n================================================================================")
    print(f"STARTING INFINITE STREAMING O(1) ONLINE LEARNING ({args.total_tokens:,} TOKENS)")
    print(f"================================================================================", flush=True)

    # Preallocate buffers and constants to minimize memory churn
    h_prev = torch.zeros_like(h)
    read_mask_1n = model.read_mask.unsqueeze(0)
    pi_const = math.pi

    # Cached superclass constants (recomputed only when parameters change)
    def update_cached_params():
        rho_a = model.get_alif_params()[0] if getattr(model, "use_alif", False) else torch.ones_like(h)
        return (
            model.get_thresholds(),
            model.get_alif_params()[1],
            torch.exp(model.log_threshold),
            torch.exp(model.log_beta),
            rho_a,
        )

    thresholds, beta_a, theta_0, beta_val, rho_a = update_cached_params()
    use_sliced_readout = hasattr(model, "read_indices") and (getattr(model, "read_surface", "all") != "all")
    motor_indices = model.read_indices if use_sliced_readout else None
    motor_superclasses = superclass_ids[model.read_indices] if use_sliced_readout else None

    # Pre-load tokens in streaming buffer on GPU (10,000 tokens per buffer)
    BUFFER_SIZE = 10000
    current_buf_start = 0
    buf_end = min(BUFFER_SIZE, len(train_data), args.total_tokens + 1)
    gpu_tokens = torch.from_numpy(np.array(train_data[0:buf_end], dtype=np.int64)).cuda()

    t_start = time.perf_counter()
    total_val_time = 0.0
    ema_loss_t = torch.tensor(7.5, device="cuda")
    step_loss_t = torch.tensor(7.5, device="cuda")
    accum_count = 0

    for token_idx in range(args.total_tokens):
        # Refresh GPU token prefetch buffer if needed
        local_idx = token_idx - current_buf_start
        if local_idx + 1 >= gpu_tokens.shape[0]:
            current_buf_start = token_idx
            buf_end = min(current_buf_start + BUFFER_SIZE, len(train_data), args.total_tokens + 1)
            gpu_tokens = torch.from_numpy(np.array(train_data[current_buf_start:buf_end], dtype=np.int64)).cuda()
            local_idx = 0

        tok_in = gpu_tokens[local_idx:local_idx + 1]
        tok_tgt = gpu_tokens[local_idx + 1:local_idx + 2]

        # 1. Forward continuous LIF/COBA step with STP & ALIF (Closed graph)
        h_prev.copy_(h)
        res, biophysics = model.step(
            h, tok_in, spike_ring=ring, ge=ge, gi=gi, b=b, x=x, u=u, return_biophysics=True
        )
        h = res[0]
        ring = res[2]
        ge = res[3]
        gi = res[4]
        b = res[5]
        x = res[6]
        u = res[7]

        # 2. Instantaneous readout & in-place cross-entropy error (Prequential Prediction)
        if use_sliced_readout:
            readout_vec = h[:, model.read_indices]
        else:
            readout_vec = h * read_mask_1n
        z_raw = model.output_read(readout_vec)
        if hasattr(model, "read_norm"):
            z_latent = model.read_norm(z_raw)
        else:
            z_latent = z_raw
        logits = model.decoder(z_latent)
        probs = F.softmax(logits, dim=-1)
        tgt_id = tok_tgt[0]

        # In-place GPU scalar EMA loss: strictly 0 CPU-GPU sync barriers during streaming!
        step_loss_t = -torch.log(probs[0, tgt_id].clamp_min(1e-9))
        ema_loss_t.mul_(0.99).add_(step_loss_t, alpha=0.01)

        # In-place error computation: avoids allocating 50k one-hot tensor
        probs[0, tgt_id] -= 1.0  # now probs is e_logits
        error_norm = torch.matmul(probs, model.decoder.weight)

        if hasattr(model, "read_norm"):
            # Analytical backprop through RMSNorm:
            rms = torch.sqrt(torch.mean(z_raw ** 2, dim=-1, keepdim=True) + 1e-6)
            gamma = model.read_norm.weight
            u_norm = error_norm * gamma
            z_hat = z_raw / rms
            proj = (u_norm * z_hat).sum(dim=-1, keepdim=True) / z_raw.shape[-1]
            error_read = (u_norm - z_hat * proj) / rms
            if grad_norm_acc is not None:
                grad_norm_acc.add_((error_norm * z_hat).squeeze(0))
        else:
            error_read = error_norm

        if use_sliced_readout:
            grad_w_read_acc.addr_(error_read.squeeze(0), readout_vec.squeeze(0))
        else:
            grad_w_read_acc.add_(torch.matmul(error_read.t(), readout_vec))
        if grad_w_dec_acc is not None:
            buf_probs[accum_count] = probs.squeeze(0)
            buf_z_latent[accum_count] = z_latent.squeeze(0)
        elif grad_b_dec_acc is not None:
            grad_b_dec_acc.add_(probs.squeeze(0))

        # 3. Superclass hyperparameter analytical gradients (via local surrogate derivative psi)
        v_pre = biophysics["v_pre"]
        eff_thresh = biophysics["eff_threshold"]
        v_diff = v_pre - eff_thresh
        pi_x = pi_const * v_diff
        psi = 1.0 / (1.0 + pi_x * pi_x)  # [1, N]

        # Top-down error projected into reservoir: L_read
        if use_sliced_readout:
            L_motor = torch.matmul(error_read, model.output_read.weight).squeeze(0)
            L_motor_psi = L_motor * psi[0, motor_indices]
            grad_thresh_acc.index_add_(0, motor_superclasses, -L_motor_psi * theta_0[motor_superclasses])
            grad_tau_m_acc.index_add_(0, motor_superclasses, L_motor_psi * h_prev[0, motor_indices])
            grad_beta_acc.index_add_(0, motor_superclasses, -L_motor_psi * b[0, motor_indices] * beta_val[motor_superclasses])
            rho_a_motor = rho_a[0, motor_indices]
            b_prev_m = b[0, motor_indices]
            grad_tau_a_acc.index_add_(
                0, motor_superclasses,
                (-L_motor_psi * beta_val[motor_superclasses]) * b_prev_m * rho_a_motor * (-torch.log(rho_a_motor.clamp_min(1e-6)))
            )
        else:
            L_read = torch.matmul(error_read, model.output_read.weight) * read_mask_1n
            L_psi = (L_read * psi).squeeze(0)  # [N]
            grad_thresh_acc.index_add_(0, superclass_ids, -L_psi * theta_0[superclass_ids])
            grad_tau_m_acc.index_add_(0, superclass_ids, L_psi * h_prev.squeeze(0))
            grad_beta_acc.index_add_(0, superclass_ids, -L_psi * b.squeeze(0) * beta_val[superclass_ids])
            rho_a_all = rho_a.squeeze(0)
            grad_tau_a_acc.index_add_(
                0, superclass_ids,
                (-L_psi * beta_val[superclass_ids]) * b.squeeze(0) * rho_a_all * (-torch.log(rho_a_all.clamp_min(1e-6)))
            )

        # 4. Neuromodulation & Whole-brain 25.32M synaptic plasticity via analytical e-prop (O(1) memory, Dale's law)
        if dopamine_state is not None:
            q_gate = dopamine_state.update_from_dan_activity(
                spikes=res[1],
                h=res[0],
                dan_indices=getattr(model, "dan_indices", None),
                dan_edge_pre=getattr(model, "dan_edge_pre", None),
                dan_edge_post=getattr(model, "dan_edge_post", None),
                dan_edge_weight=getattr(model, "dan_edge_weight", None),
                dan_scale=getattr(model, "dan_scale", 1.0),
            )
        else:
            q_gate = None

        if sensory_eligibility is not None:
            sens_idx = model.topographic_writer.injection_index
            alpha_sens = biophysics["alpha_eff"][0, sens_idx]
            beta_sens = biophysics["beta_int"][0, sens_idx]
            sensory_eligibility.update_input_eligibility(
                token_emb=model.topographic_writer.last_token_emb,
                gates=model.topographic_writer.last_gates,
                alpha_sens=alpha_sens,
                beta_int_sens=beta_sens,
            )

        if args.unfreeze_synapses and eprop is not None:
            phi_e, phi_i, z_bar = eprop.update_eligibility_coba(
                state=eligibility,
                delayed_pulses=biophysics["delayed_pulses"],
                v_pre=biophysics["v_pre"],
                effective_threshold=biophysics["eff_threshold"],
                beta_adaptation=beta_a,
                alpha_eff=biophysics["alpha_eff"],
                rho_a=rho_a,
                beta_int=biophysics.get("beta_int"),
                g_e_gain=biophysics.get("g_e", 1.0),
                g_i_gain=biophysics.get("g_i", 1.0),
                leak_se=biophysics.get("leak_se", 0.0),
                leak_si=biophysics.get("leak_si", 0.0),
                prev_spikes=prev_spikes,
            )
            prev_spikes.copy_(res[1])

            dan_pre = getattr(model, "dan_edge_pre", None)
            dan_post = getattr(model, "dan_edge_post", None)
            dan_weight = getattr(model, "dan_edge_weight", None)
            dan_scale = getattr(model, "dan_scale", 1.0)

            L_total, _, _ = eprop.compute_learning_signal_whole_brain(
                w_read=model.output_read.weight,
                read_mask=model.read_mask,
                error_read=error_read,
                read_indices=motor_indices,
                l_readout=L_motor if use_sliced_readout else None,
                dan_edge_pre=dan_pre,
                dan_edge_post=dan_post,
                dan_edge_weight=dan_weight,
                dan_scale=dan_scale,
            )

            if (token_idx + 1) % args.synapse_update_interval == 0:
                syn_scale = args.lr_synapse * args.synapse_update_interval
                EPropCreditAssignment.apply_synaptic_updates_inplace(
                    edge_weight=model.edge_weight_e,
                    L=L_total,
                    phi=phi_e,
                    z_bar=z_bar,
                    edge_pre=model.edge_pre_e,
                    edge_post=model.edge_post_e,
                    delay_splits=model.splits_e,
                    q_gate=q_gate,
                    scale=syn_scale,
                    min_val=0.0,
                    max_val=5.0,
                )
                EPropCreditAssignment.apply_synaptic_updates_inplace(
                    edge_weight=model.edge_weight_i,
                    L=L_total,
                    phi=phi_i,
                    z_bar=z_bar,
                    edge_pre=model.edge_pre_i,
                    edge_post=model.edge_post_i,
                    delay_splits=model.splits_i,
                    q_gate=q_gate,
                    scale=syn_scale,
                    min_val=0.0,
                    max_val=5.0,
                )
                if sensory_eligibility is not None and args.unfreeze_sensory:
                    sens_scale = args.lr_sensory * args.synapse_update_interval
                    sens_idx = model.topographic_writer.injection_index
                    L_sens = L_total[0, sens_idx]
                    phi_sens = phi_e[0, sens_idx] if phi_e is not None else torch.ones_like(L_sens)
                    q_sens = q_gate[sens_idx] if q_gate is not None else torch.ones_like(L_sens)
                    sensory_eligibility.apply_sensory_updates_inplace(
                        writer=model.topographic_writer,
                        L_sens=L_sens,
                        phi_sens=phi_sens,
                        q_sens=q_sens,
                        scale=sens_scale,
                        weight_decay=1e-4,
                    )

            if sensory_eligibility is not None and args.unfreeze_sensory and grad_gate_w_acc is not None:
                sens_idx = model.topographic_writer.injection_index
                L_sens = L_total[0, sens_idx]
                phi_sens = phi_e[0, sens_idx] if phi_e is not None else torch.ones_like(L_sens)
                q_sens = q_gate[sens_idx] if q_gate is not None else torch.ones_like(L_sens)
                kappa_sens = q_sens * L_sens * phi_sens

                n_vis = model.topographic_writer.n_vis
                n_chemo = model.topographic_writer.n_chemo
                s_chemo = n_vis
                e_chemo = n_vis + n_chemo

                p_vis = model.topographic_writer.last_p_vis.squeeze(0)
                p_chemo = model.topographic_writer.last_p_chemo.squeeze(0)
                p_mech = model.topographic_writer.last_p_mech.squeeze(0)

                dg0 = (kappa_sens[:n_vis] * p_vis).sum()
                dg1 = (kappa_sens[s_chemo:e_chemo] * p_chemo).sum()
                dg2 = (kappa_sens[e_chemo:] * p_mech).sum()
                dg = torch.stack([dg0, dg1, dg2])

                gates = model.topographic_writer.last_gates.squeeze(0)
                p_gate = gates / 3.0
                dz = 3.0 * p_gate * (dg - torch.dot(dg, p_gate))

                token_x = model.topographic_writer.last_token_emb.squeeze(0)
                grad_gate_w_acc.addr_(dz, token_x)
                grad_gate_b_acc.add_(dz)

        accum_count += 1

        # 5. Apply online AdamW update every grad_accum_tokens
        if accum_count >= args.grad_accum_tokens:
            scale = 1.0 / accum_count
            model.output_read.weight.grad = grad_w_read_acc * scale
            if grad_norm_acc is not None:
                model.read_norm.weight.grad = grad_norm_acc * scale
            if grad_w_dec_acc is not None:
                grad_w_dec_acc.addmm_(buf_probs[:accum_count].t(), buf_z_latent[:accum_count])
                model.decoder.weight.grad = grad_w_dec_acc * scale
                if grad_b_dec_acc is not None:
                    grad_b_dec_acc.add_(buf_probs[:accum_count].sum(dim=0))
                    model.decoder.bias.grad = grad_b_dec_acc * scale
            elif grad_b_dec_acc is not None:
                model.decoder.bias.grad = grad_b_dec_acc * scale
            model.log_threshold.grad = grad_thresh_acc.clamp(-5.0, 5.0) * scale
            model.log_tau_m.grad = grad_tau_m_acc.clamp(-5.0, 5.0) * scale
            model.log_beta.grad = grad_beta_acc.clamp(-5.0, 5.0) * scale
            model.log_tau_a.grad = grad_tau_a_acc.clamp(-5.0, 5.0) * scale
            if grad_gate_w_acc is not None:
                model.topographic_writer.gate_linear.weight.grad = grad_gate_w_acc * scale
                model.topographic_writer.gate_linear.bias.grad = grad_gate_b_acc * scale

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            thresholds, beta_a, theta_0, beta_val, rho_a = update_cached_params()

            grad_w_read_acc.zero_()
            if grad_norm_acc is not None:
                grad_norm_acc.zero_()
            if grad_b_dec_acc is not None:
                grad_b_dec_acc.zero_()
            if grad_w_dec_acc is not None:
                grad_w_dec_acc.zero_()
            if grad_gate_w_acc is not None:
                grad_gate_w_acc.zero_()
                grad_gate_b_acc.zero_()
            grad_thresh_acc.zero_()
            grad_tau_m_acc.zero_()
            grad_beta_acc.zero_()
            grad_tau_a_acc.zero_()
            accum_count = 0

        # Periodic logging: reads .item() only once every log_every_tokens
        if (token_idx + 1) % args.log_every_tokens == 0:
            stream_elapsed = max(time.perf_counter() - t_start, 1e-4)
            tok_per_sec = (token_idx + 1) / stream_elapsed
            vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            ema_loss_val = float(ema_loss_t.item())
            step_loss_val = float(step_loss_t.item())
            with file_lock:
                current_best = state_meta["best_live_val"]
            da_str = f" | DA q_mean: {dopamine_state.q.mean().item():.3f}" if dopamine_state is not None else ""
            print(f"Token {token_idx + 1:7d}/{args.total_tokens:,} | "
                  f"EMA Stream NLL: {ema_loss_val:.4f} | "
                  f"Speed: {tok_per_sec:.1f} tok/s | VRAM: {vram_mb:.1f} MB{da_str}", flush=True)

            progress_payload = {
                "status": "running",
                "tokens_streamed": token_idx + 1,
                "total_target_tokens": args.total_tokens,
                "ema_stream_loss": ema_loss_val,
                "step_loss": step_loss_val,
                "speed_tokens_per_sec": tok_per_sec,
                "vram_mb": vram_mb,
                "best_live_nll": current_best,
            }
            if dopamine_state is not None:
                progress_payload["dopamine_mean_q"] = float(dopamine_state.q.mean().item())
                progress_payload["dopamine_max_q"] = float(dopamine_state.q.max().item())
            atomic_json(args.output / "progress.json", progress_payload)

        # Periodic evaluation: Complete Four-Pillar Scientific Suite (Asynchronous & Non-Blocking)
        if (token_idx + 1) % args.validate_every_tokens == 0:
            if eval_thread is not None and eval_thread.is_alive():
                print(f"  [Async Eval Notice] Previous evaluation still active at token {token_idx + 1}; continuing streaming.", flush=True)
            else:
                ema_loss_snapshot = float(ema_loss_t.item())
                # Quick snapshot of plastic weights and physical state
                snapshot_weights = {
                    k: v.detach().clone()
                    for k, v in model.state_dict().items()
                    if not k.startswith("edge_") and not k.startswith("dan_") and k != "lambda_0" and not k.startswith("embedding.")
                }
                snapshot_phys = (
                    h.detach().clone(),
                    tuple(r.detach().clone() for r in ring),
                    {k: v.detach().clone() for k, v in [("ge", ge), ("gi", gi), ("b", b), ("x", x), ("u", u)]},
                )
                cursor_for_eval = val_cursor
                eval_tokens_consumed = args.eval_shock_tokens + args.eval_ftle_steps + (args.eval_ebb_a + args.eval_ebb_b)
                val_cursor = (val_cursor + eval_tokens_consumed) % max(1, (len(val_data) - eval_tokens_consumed - 10))

                eval_thread = threading.Thread(
                    target=run_eval_job,
                    args=(token_idx + 1, snapshot_weights, snapshot_phys, cursor_for_eval, ema_loss_snapshot),
                    daemon=True,
                )
                eval_thread.start()

    if eval_thread is not None and eval_thread.is_alive():
        print("\nWaiting for in-flight background Four-Pillar evaluation to finish...", flush=True)
        eval_thread.join()

    # If the run ended at a point that wasn't just evaluated, run a final evaluation
    if (args.total_tokens % args.validate_every_tokens) != 0:
        print(f"\nRunning final Four-Pillar evaluation at token {args.total_tokens}...", flush=True)
        snapshot_weights = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if not k.startswith("edge_") and not k.startswith("dan_") and k != "lambda_0" and not k.startswith("embedding.")
        }
        snapshot_phys = (
            h.detach().clone(),
            tuple(r.detach().clone() for r in ring),
            {k: v.detach().clone() for k, v in [("ge", ge), ("gi", gi), ("b", b), ("x", x), ("u", u)]},
        )
        run_eval_job(
            token_step=args.total_tokens,
            weights_dict=snapshot_weights,
            phys_state=snapshot_phys,
            cursor_val=val_cursor,
            ema_loss_snapshot=float(ema_loss_t.item()),
        )

    final_loss_val = float(ema_loss_t.item())
    with file_lock:
        current_best = state_meta["best_live_val"]
    atomic_json(args.output / "progress.json", {
        "status": "completed",
        "tokens_streamed": args.total_tokens,
        "total_target_tokens": args.total_tokens,
        "final_ema_loss": final_loss_val,
        "best_live_nll": current_best,
    })
    print("\nInfinite streaming run successfully completed!", flush=True)


if __name__ == "__main__":
    main()
