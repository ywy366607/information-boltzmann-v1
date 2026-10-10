"""Fruit Fly Connectome Champion + Variational Rolling Stream Integration.

Loads the champion checkpoint (at 121,984 tokens) in read-only mode and continues
training using the Variational Rolling Stream lifelong learning protocol:
- Overlapping rolling window (Window W=32, Stride S=16): Prompt A -> Target B
- Biophysical Drosophila connectome dynamics (25k neurons, COBA, ALIF, STP)
- Prequential scoring before adaptation vs converged plateau scoring
- Variational Free Energy assimilation loop on latent working memory z
- Single-counting outer update on fly weights (decoder, read_norm, modulator, synapses)
- Continuous temporal Ornstein-Uhlenbeck retention transition across window boundaries
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F

from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import (
    FlyPhysicalState,
    FlyBPTTLearner,
    STP_PARAMETER_NAMES,
    stable_clip_grad_norm_,
)
from information_boltzmann.core.variational_rolling_stream import (
    VariationalBeliefModulator,
    VariationalGaussianBelief,
    compute_friston_adaptive_retention,
)
from information_boltzmann.runtime.lifelong_evaluation import adaptation_generalization_summary


def run_fly_rolling_vfe(args: argparse.Namespace) -> dict:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Fly Connectome + Variational Rolling Stream Integration ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Loading champion checkpoint: {args.checkpoint}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Load champion checkpoint (READ-ONLY)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    old = saved["learner"]
    cursor = int(saved["train_cursor"])
    trained_tokens = int(saved["bptt_train_tokens"])
    best_live_nll = float(saved.get("best_live_nll", 6.55))

    print(f"Champion state restored: cursor={cursor:,}, trained_tokens={trained_tokens:,}, best_live_nll={best_live_nll:.4f}")

    # Build model
    model = FlyReservoirLM(
        cfg["graph"], vocab_size=50257, d_model=cfg["d_model"],
        injection="topographic", read_surface="output", synapse_model="coba",
        use_alif=True, use_stp=True, decoder_bias=cfg["decoder_bias"],
        read_centering=old.get("read_centering", False),
        use_read_gamma_trace=old.get("use_read_gamma_trace", False),
        init_read_gamma=old.get("init_read_gamma", 0.7),
        use_latent_predictor=old.get("use_latent_predictor", False),
        use_graph_observer=old.get("use_graph_observer", False),
        lambda_obs=old.get("lambda_jepa", 1.0),
        lambda_sigreg=cfg.get("lambda_sigreg", 0.2),
        max_horizon=cfg.get("max_horizon", 14),
        dagger_beta=cfg.get("dagger_beta", 0.5),
        obs_init_gamma=cfg.get("obs_init_gamma", 0.0),
        detach_reset=cfg.get("detach_reset", False),
        surrogate_mode=cfg.get("surrogate_mode", "absolute"),
        transmission_mode=cfg.get("transmission_mode", "atomic"),
    ).to(device)

    # Load weights
    weights = {k: v for k, v in saved["model"].items() if k not in ("edge_weight_e", "edge_weight_i")}
    with torch.no_grad():
        for name in ("edge_weight_e", "edge_weight_i"):
            getattr(model, name).copy_(saved["model"][name])
        model.load_state_dict(weights, strict=False)

    # Restore biophysical state
    st_dict = old["physical"]
    ring = tuple(st_dict["ring"][i].to(device) for i in range(len(st_dict["ring"])))
    physical = FlyPhysicalState(
        h=st_dict["h"].to(device),
        ring=ring,
        ge=st_dict["ge"].to(device),
        gi=st_dict["gi"].to(device),
        b=st_dict["b"].to(device),
        x=st_dict["x"].to(device),
        u=st_dict["u"].to(device),
        baseline=st_dict["baseline"].to(device),
        h_mean=st_dict["h_mean"].to(device),
        dan_gate=st_dict["dan_gate"].to(device),
        gamma_z1=st_dict["gamma_z1"].to(device),
        gamma_z2=st_dict["gamma_z2"].to(device),
        observer_prior=st_dict["observer_prior"].to(device),
        observer_history=st_dict["observer_history"].to(device),
    )

    names = [
        "output_read.weight", "read_norm.weight", "decoder.weight", "decoder.bias",
        "log_threshold", "log_tau_m", "log_beta", "log_tau_a",
        "log_tau_s_e", "log_tau_s_i", "log_g_e", "log_g_i",
        "topographic_writer.gate_linear.weight", "topographic_writer.gate_linear.bias",
    ]
    if old.get("learn_stp", False):
        names.extend(STP_PARAMETER_NAMES)

    learner = FlyBPTTLearner(
        model, physical, adam_names=names,
        lr=args.lr, lr_synapse=args.lr_synapse, lr_sensory=args.lr_sensory,
        plasticity_optimizer="adamw", lr_decoder=args.lr_decoder,
        settle_ticks=args.settle_ticks, writer_baseline_clock="physical",
        learn_stp=bool(old.get("learn_stp", False)),
        lambda_jepa=1.0,
        use_ctm_loss=True,
        use_checkpointing=True,
        lambda_mcr2=0.0,
        eps_mcr2=0.5,
        adaptive_admission=True,
        min_settle_ticks=3,
        flux_baseline=0.048,
    )
    learner.previous_token = int(old["previous_token"])
    learner.load_adam_state(old["optimizer"])

    # Instantiate Variational Belief Modulator on fly readout
    modulator = VariationalBeliefModulator(
        latent_dim=args.latent_dim, hidden_dim=cfg["d_model"], vocab_size=50257
    ).to(device)
    mod_opt = torch.optim.AdamW(modulator.parameters(), lr=args.lr_modulator)

    # Initialize Prior
    current_prior = VariationalGaussianBelief.standard_normal(args.latent_dim, device=device)

    # Load stream data
    data_mmap = np.load(args.data_path, mmap_mode="r")

    W = args.window
    S = args.stride
    target_size = W - S
    print(f"Stream configuration: Window W={W}, Stride S={S} (Prompt={S}, Target={target_size})")
    print(f"Inner Max Steps: {args.inner_max_steps}, Inner LR: {args.inner_lr}, Plateau Delta: {args.plateau_delta}")
    print(f"KL Weight: {args.kl_weight}, Retention Rho: {args.retention}, Adaptive Retention: {args.adaptive_retention}\n")

    if getattr(args, "multi_scale_retention", True):
        clamped_r = max(1e-3, min(1.0 - 1e-3, args.retention))
        logit_base = math.log(clamped_r / (1.0 - clamped_r))
        spread = torch.linspace(-1.0, 1.0, args.latent_dim, device=device)
        base_retention = torch.sigmoid(logit_base + spread)
    else:
        base_retention = float(args.retention)
    running_baseline_err = None
    error_ema_beta = 0.85
    previous_posterior = None
    previous_content_feat = None
    previous_log_prec_ratio = 0.0

    repeat_raw_window = None
    if args.repeat_window:
        repeat_raw_window = np.array(
            data_mmap[cursor:cursor + W], dtype=np.int64
        )

    print("-" * 130)
    print(
        f"{'Win':>4} | {'Tokens':>15} | {'Preq NLL':>9} | {'Plat NLL':>9} | {'Gain (nats)':>11} | "
        f"{'KL (nats)':>9} | {'FE':>8} | {'Steps':>5} | {'Plat?':>5} | {'Prec':>6} | {'Rho':>6} | {'Time':>7}"
    )
    print("-" * 130)

    reports = []
    t0_all = time.perf_counter()
    stream_cursor = cursor

    for w_idx in range(args.num_windows):
        t_start = time.perf_counter()
        raw_window = (
            repeat_raw_window.copy()
            if repeat_raw_window is not None
            else np.array(data_mmap[stream_cursor : stream_cursor + W], dtype=np.int64)
        )
        tokens = torch.from_numpy(raw_window).to(device)

        tok_range_str = f"{stream_cursor}:{stream_cursor + W}"

        # 1. Forward pass through Drosophila connectome
        ids = learner.inputs_for_targets(tokens)
        learner.optimizer.zero_grad(set_to_none=True)
        learner.sgd.zero_grad(set_to_none=True)
        mod_opt.zero_grad()

        scores, next_state, feats = learner.forward_window(ids, tokens[None])

        # Target segment B is tokens[S:]
        target_b = tokens[S:]
        feats_b = feats[S:]

        # 2. Prequential Evaluation under prior mean
        with torch.no_grad():
            f_mod = modulator.modulate_features(feats_b.unsqueeze(0), current_prior.mean).squeeze(0)
            normed = model.read_norm(f_mod)
            logits_b = model.decoder(normed) + modulator.compute_logits_bias(current_prior.mean).squeeze(0)
            preq_nll = float(F.cross_entropy(logits_b, target_b).item())

        # 3. Inner Assimilation Loop on Target B (holding prior frozen)
        frozen_prior = current_prior.detach()
        mu_q = frozen_prior.mean.clone().detach().requires_grad_(True)
        log_std_q = frozen_prior.log_std.clone().detach().requires_grad_(True)
        inner_opt = torch.optim.Adam([mu_q, log_std_q], lr=args.inner_lr)

        prev_eval_fe = None
        consecutive_plateaus = 0
        plateau_reached = False
        steps_taken = 0
        final_plat_nll = preq_nll
        final_kl = 0.0
        final_fe = preq_nll

        # We detach feats for the inner loop so the base graph is untouched during posterior search
        feats_b_inner = feats_b.detach()

        for step in range(1, args.inner_max_steps + 1):
            steps_taken = step
            inner_opt.zero_grad()

            q = VariationalGaussianBelief(mean=mu_q, log_std=log_std_q)
            z_sample = q.sample(1)

            f_mod = modulator.modulate_features(feats_b_inner.unsqueeze(0), z_sample).squeeze(0)
            normed = model.read_norm(f_mod)
            logits_b = model.decoder(normed) + modulator.compute_logits_bias(z_sample).squeeze(0)
            nll_target = F.cross_entropy(logits_b, target_b)
            kl_val = q.kl_divergence(frozen_prior)

            fe = nll_target + (args.kl_weight / target_size) * kl_val
            fe.backward()
            inner_opt.step()

            # Smooth convergence evaluation at posterior mean
            with torch.no_grad():
                updated_q = VariationalGaussianBelief(mean=mu_q, log_std=log_std_q)
                f_mod_eval = modulator.modulate_features(feats_b_inner.unsqueeze(0), mu_q).squeeze(0)
                normed_eval = model.read_norm(f_mod_eval)
                logits_eval = model.decoder(normed_eval) + modulator.compute_logits_bias(mu_q).squeeze(0)
                eval_nll = float(F.cross_entropy(logits_eval, target_b).item())
                eval_kl = float(updated_q.kl_divergence(frozen_prior).item())
                eval_fe = eval_nll + (args.kl_weight / target_size) * eval_kl

            final_plat_nll = eval_nll
            final_kl = eval_kl
            final_fe = eval_fe

            delta_fe = abs(eval_fe - prev_eval_fe) if prev_eval_fe is not None else float("inf")
            if delta_fe < args.plateau_delta:
                consecutive_plateaus += 1
                if consecutive_plateaus >= args.plateau_patience:
                    plateau_reached = True
                    break
            else:
                consecutive_plateaus = 0

            prev_eval_fe = eval_fe

        adaptation_gain = preq_nll - final_plat_nll

        # 4. Outer Structural Update (Single-Counting Evidence into slow fly weights)
        converged_q = VariationalGaussianBelief(mean=mu_q.detach(), log_std=log_std_q.detach())
        z_star = converged_q.mean

        f_mod_outer = modulator.modulate_features(feats_b.unsqueeze(0), z_star).squeeze(0)
        normed_outer = model.read_norm(f_mod_outer)
        logits_outer = model.decoder(normed_outer) + modulator.compute_logits_bias(z_star).squeeze(0)
        outer_loss = F.cross_entropy(logits_outer, target_b)

        current_content_feat = feats_b.detach().mean(dim=0)
        gate_kl = None
        if (
            args.channel_gate
            and args.adaptive_retention
            and args.multi_scale_retention
            and previous_posterior is not None
        ):
            rho_gate = modulator.compute_channel_retention(
                log_precision_ratio=previous_log_prec_ratio,
                content_feat=previous_content_feat,
                retention_min=args.retention_min,
                retention_max=args.retention_max,
            )
            rho_sq = rho_gate * rho_gate
            mu_pred = rho_gate * previous_posterior.mean.detach()
            var_pred = rho_sq * previous_posterior.var.detach() + (1.0 - rho_sq)
            gate_kl = 0.5 * torch.sum(
                (converged_q.var.detach() + (converged_q.mean.detach() - mu_pred).pow(2))
                / (var_pred + 1e-8)
                - 1.0
                + torch.log(var_pred.clamp_min(1e-8) / converged_q.var.detach().clamp_min(1e-8))
            )
            outer_loss = outer_loss + (args.gdn_kl_weight / args.latent_dim) * gate_kl

        outer_loss.backward()

        stable_clip_grad_norm_(learner.trainable, 1.0)
        learner.optimizer.step()
        learner.sgd.step()
        mod_opt.step()

        with torch.no_grad():
            learner.clamp_edges()
            if model.topographic_writer is not None:
                model.topographic_writer.a_adapt.copy_(next_state.baseline)
            learner.state = next_state.detached()
            learner.previous_token = int(tokens[-1].item())

        learner.optimizer.zero_grad(set_to_none=True)
        learner.sgd.zero_grad(set_to_none=True)
        mod_opt.zero_grad()

        # 5. Temporal Belief Transition to form next prior with Friston adaptive precision
        if running_baseline_err is None:
            running_baseline_err = preq_nll
        else:
            running_baseline_err = error_ema_beta * running_baseline_err + (1.0 - error_ema_beta) * preq_nll

        log_precision_ratio = 0.0
        if args.adaptive_retention:
            effective_retention, prec_ratio = compute_friston_adaptive_retention(
                base_retention=base_retention,
                prediction_error=preq_nll,
                baseline_error=running_baseline_err,
                error_sensitivity=args.error_sensitivity,
                retention_min=args.retention_min,
                retention_max=args.retention_max,
            )
            log_precision_ratio = math.log(max(float(prec_ratio), 1e-8))
            if args.channel_gate and args.multi_scale_retention:
                effective_retention = modulator.compute_channel_retention(
                    log_precision_ratio=log_precision_ratio,
                    content_feat=current_content_feat,
                    retention_min=args.retention_min,
                    retention_max=args.retention_max,
                ).detach()
            retention_mean_val = float(effective_retention.mean().item()) if isinstance(effective_retention, torch.Tensor) else float(effective_retention)
        else:
            effective_retention = base_retention
            prec_ratio = 1.0
            retention_mean_val = float(effective_retention.mean().item()) if isinstance(effective_retention, torch.Tensor) else float(effective_retention)

        next_prior = converged_q.transition(retention=effective_retention, base_mean=0.0, base_log_std=0.0)
        current_prior = next_prior
        previous_posterior = converged_q.detach()
        previous_content_feat = current_content_feat
        previous_log_prec_ratio = log_precision_ratio

        # Advance stream cursor by stride S
        stream_cursor += S

        t_end = time.perf_counter()
        wall_time_ms = (t_end - t_start) * 1000.0

        plat_str = "YES" if plateau_reached else "NO"
        print(
            f"{w_idx + 1:4d} | {tok_range_str:>15} | {preq_nll:9.4f} | {final_plat_nll:9.4f} | "
            f"{adaptation_gain:+11.4f} | {final_kl:9.4f} | {final_fe:8.4f} | "
            f"{steps_taken:5d} | {plat_str:>5} | {prec_ratio:6.2f} | {retention_mean_val:6.3f} | {wall_time_ms:6.1f}ms"
        )

        reports.append({
            "window": w_idx + 1,
            "stream_cursor": stream_cursor,
            "prequential_nll": preq_nll,
            "plateau_nll": final_plat_nll,
            "adaptation_gain": adaptation_gain,
            "kl_divergence": final_kl,
            "free_energy": final_fe,
            "inner_steps": steps_taken,
            "plateau_reached": plateau_reached,
            "retention_mean": retention_mean_val,
            "precision_ratio": prec_ratio,
            "wall_time_ms": wall_time_ms,
        })

    t_total = time.perf_counter() - t0_all
    peak_vram_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)

    preq_list = [r["prequential_nll"] for r in reports]
    plat_list = [r["plateau_nll"] for r in reports]
    gain_list = [r["adaptation_gain"] for r in reports]
    kl_list = [r["kl_divergence"] for r in reports]
    steps_list = [r["inner_steps"] for r in reports]
    plat_count = sum(1 for r in reports if r["plateau_reached"])

    total_tokens_advanced = args.num_windows * S
    throughput = total_tokens_advanced / max(t_total, 1e-6)

    print("-" * 115)
    print("\n=== Drosophila Connectome Continuous Assimilation Summary ===")
    print(f"Total Windows: {args.num_windows}")
    print(f"Total Stream Tokens Advanced: {total_tokens_advanced} tokens (Final cursor: {stream_cursor:,})")
    print(f"Total Elapsed Time: {t_total:.2f} s ({throughput:.2f} tokens/s)")
    print(f"Peak VRAM Usage: {peak_vram_mib:.2f} MiB (Budget: 3072 MiB)")
    print(f"Mean Prequential NLL: {np.mean(preq_list):.4f} nats/token")
    print(f"Mean Plateau NLL: {np.mean(plat_list):.4f} nats/token")
    print(f"Mean Adaptation Gain: {np.mean(gain_list):+.4f} nats/token ({np.mean(gain_list) / np.mean(preq_list) * 100:.2f}%)")
    print(f"Mean Belief KL Divergence: {np.mean(kl_list):.4f} nats")
    print(f"Mean Inner Steps to Plateau: {np.mean(steps_list):.2f} / {args.inner_max_steps}")
    print(f"Plateau Trigger Rate: {plat_count}/{args.num_windows} ({plat_count / args.num_windows * 100:.1f}%)")

    summary = {
        "checkpoint": str(args.checkpoint),
        "initial_cursor": cursor,
        "final_cursor": stream_cursor,
        "num_windows": args.num_windows,
        "window": W,
        "stride": S,
        "total_tokens_advanced": total_tokens_advanced,
        "total_elapsed_s": t_total,
        "throughput_tokens_per_sec": throughput,
        "peak_vram_mib": peak_vram_mib,
        "mean_prequential_nll": float(np.mean(preq_list)),
        "mean_plateau_nll": float(np.mean(plat_list)),
        "mean_adaptation_gain": float(np.mean(gain_list)),
        "mean_kl_divergence": float(np.mean(kl_list)),
        "mean_inner_steps": float(np.mean(steps_list)),
        "plateau_trigger_rate": float(plat_count / args.num_windows),
        "mean_retention": float(np.mean([r["retention_mean"] for r in reports])),
        "mean_precision_ratio": float(np.mean([r["precision_ratio"] for r in reports])),
        "reports": reports,
    }

    if args.repeat_window:
        ag = adaptation_generalization_summary(
            preq_list,
            block_tokens=1,
            hold_blocks=3,
            plateau_blocks=8,
            plateau_tolerance_nll=args.ag_plateau_tolerance,
            reference_nll=[args.ag_reference_nll] * len(preq_list),
        )
        ag["recovery_tokens_in_stream"] = (
            None if ag.get("recovery_tokens") is None
            else ag["recovery_tokens"] * S
        )
        summary["ag"] = ag

    if args.report_out:
        out_p = Path(args.report_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Report saved to: {out_p}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Fly Connectome + Variational Rolling Stream Integration")
    parser.add_argument("--checkpoint", type=Path, default=Path("E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt"))
    parser.add_argument("--data-path", type=Path, default=Path("data/ib_owt_gpt2/train.npy"))
    parser.add_argument("--num-windows", type=int, default=10)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--inner-max-steps", type=int, default=8)
    parser.add_argument("--inner-lr", type=float, default=0.05)
    parser.add_argument("--plateau-delta", type=float, default=0.003)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--retention", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--adaptive-retention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retention-min", type=float, default=0.10)
    parser.add_argument("--retention-max", type=float, default=0.98)
    parser.add_argument("--error-sensitivity", type=float, default=1.0)
    parser.add_argument("--multi-scale-retention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channel-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gdn-kl-weight", type=float, default=0.05)
    parser.add_argument("--repeat-window", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ag-reference-nll", type=float, default=7.65082)
    parser.add_argument("--ag-plateau-tolerance", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-decoder", type=float, default=2e-4)
    parser.add_argument("--lr-synapse", type=float, default=1e-4)
    parser.add_argument("--lr-sensory", type=float, default=1e-4)
    parser.add_argument("--lr-modulator", type=float, default=1e-4)
    parser.add_argument("--settle-ticks", type=int, default=14)
    parser.add_argument("--report-out", type=Path, default=Path("results/published/fly_rolling_vfe_pilot_20261009.json"))

    args = parser.parse_args()
    run_fly_rolling_vfe(args)


if __name__ == "__main__":
    main()

