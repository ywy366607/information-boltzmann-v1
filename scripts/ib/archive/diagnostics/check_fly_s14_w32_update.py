"""One real OWT W32 update on a restored S14 individual; source is immutable.

Write diagnostics only, not new weights. This validates numerical execution
and resource cost, not convergence or task improvement.
"""

import argparse
import hashlib
import json
import sys
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

import information_boltzmann.core.fly_bptt_learning as learning
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.gradient_norms import stable_grad_norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(
        "results/published/fly_s14_w32_update_20261008.json"))
    parser.add_argument("--vram-limit-mib", type=float, default=3900)
    parser.add_argument("--detach-reset", action=argparse.BooleanOptionalAction,
                        default=None, help="Override only the reset surrogate derivative")
    parser.add_argument('--surrogate-mode', choices=('absolute', 'threshold'), default=None,
                        help='Override proxy width only; default inherits saved convention')
    parser.add_argument('--transmission-mode', choices=('atomic', 'incoming'), default=None)
    parser.add_argument("--backward-only", action="store_true",
                        help="Suppress optimizer steps; compare derivatives without training")
    parser.add_argument("--audit-recomputation", action="store_true",
                        help="Compare original/recomputed spike decisions in this backward")
    parser.add_argument("--audit-adjoints", action="store_true",
                        help="Trace physical-state adjoints without changing gradients")
    parser.add_argument("--pulse-credit-path", choices=('all', 'active_only', 'silent_only'),
                        default='all', help="Diagnostic backward pulse-path intervention only")
    parser.add_argument("--audit-feedback-bound", action="store_true",
                        help="Collect sparse full-trajectory passive/event block envelopes")
    parser.add_argument("--forward-only", action="store_true",
                        help="No backward or update; required for first feedback feasibility gate")
    args = parser.parse_args()
    if sum((args.audit_recomputation, args.audit_adjoints, args.audit_feedback_bound,
            args.pulse_credit_path != 'all')) > 1:
        parser.error('Choose one observer per backward')
    if args.pulse_credit_path != 'all' and not args.backward_only:
        parser.error('Pulse credit interventions require --backward-only')
    if args.forward_only != args.audit_feedback_bound:
        parser.error('First feedback bound gate requires --audit-feedback-bound --forward-only')
    report = {"scope": ("One real 32-target forward-only block feasibility diagnostic"
                        if args.forward_only else
                        "One real 32-target derivative diagnostic; optimizer steps suppressed"
                        if args.backward_only else "One real 32-target optimizer update")
                       + (", full 480-tick physical forward" if args.forward_only else
                          ", full 480-tick surrogate BPTT"),
              "source_checkpoint": str(args.checkpoint.resolve()),
              "source_checkpoint_bytes": args.checkpoint.stat().st_size,
              "source_checkpoint_modified": False,
              "window": 32, "settle_ticks": 14, "optimizer_updates": 0,
              "device": torch.cuda.get_device_name(), "status": "initializing"}
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    print('Source checkpoint mapped; constructing restored model', flush=True)
    cfg, old = saved["config"], saved["learner"]
    detach_reset = bool(cfg.get("detach_reset", False)) if args.detach_reset is None else args.detach_reset
    report["detach_reset"] = detach_reset
    surrogate_mode = args.surrogate_mode or cfg.get('surrogate_mode', 'absolute')
    report['surrogate_mode'] = surrogate_mode
    transmission_mode = args.transmission_mode or cfg.get('transmission_mode', 'atomic')
    report['transmission_mode'] = transmission_mode
    report["backward_only"] = args.backward_only
    report["forward_only"] = args.forward_only
    report["pulse_credit_path"] = args.pulse_credit_path
    if old.get("use_graph_observer") or old.get("use_latent_predictor"):
        raise ValueError("This calibration requires the plain CTM S14 checkpoint")

    def memory_guard(stage):
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**20
        reserved = torch.cuda.memory_reserved() / 2**20
        report.update(vram_peak_allocated_mib=peak, vram_reserved_mib=reserved,
                      last_stage=stage)
        snapshot = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True)
        # Include other processes and the driver footprint; PyTorch's memory
        # counters alone do not measure occupation of the entire board.
        board_used = max(float(line.strip()) for line in snapshot.stdout.splitlines()
                         if line.strip())
        report.setdefault("board_memory_snapshots_mib", {})[stage] = board_used
        print(f'{stage}: whole-board {board_used:.0f} MiB', flush=True)
        if max(peak, reserved, board_used) > args.vram_limit_mib:
            raise MemoryError(f"{stage}: CUDA memory exceeds {args.vram_limit_mib} MiB")

    model = FlyReservoirLM(
        cfg["graph"], vocab_size=50257, d_model=cfg["d_model"],
        injection="topographic", read_surface="output", synapse_model="coba",
        use_alif=True, use_stp=True, decoder_bias=cfg["decoder_bias"],
        read_centering=old.get("read_centering", False),
        use_read_gamma_trace=cfg.get("use_read_gamma_trace", False),
        use_latent_predictor=False, use_graph_observer=False,
        detach_reset=detach_reset, surrogate_mode=surrogate_mode,
        transmission_mode=transmission_mode).cuda()
    # Before learner promotion, edges are non-persistent buffers and are NOT
    # restored by load_state_dict. Match production's explicit edge restore.
    with torch.no_grad():
        for name in ("edge_weight_e", "edge_weight_i"):
            getattr(model, name).copy_(saved["model"][name])
        weights = {k: v for k, v in saved["model"].items()
                   if k not in ("edge_weight_e", "edge_weight_i")}
        loaded = model.load_state_dict(weights, strict=False)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise ValueError(f"Model restoration mismatch: {loaded}")
    state = learning.FlyPhysicalState(**{
        name: tuple(v.cuda() for v in value) if name == "ring" else value.cuda()
        for name, value in old["physical"].items()})
    learner = learning.FlyBPTTLearner(
        model, state, adam_names=old["adam_names"], settle_ticks=14,
        writer_baseline_clock=old["writer_baseline_clock"],
        learn_stp=old.get("learn_stp", False),
        plasticity_optimizer=old.get("plasticity_optimizer_kind", "adamw"),
        use_ctm_loss=old.get("use_ctm_loss", True), use_checkpointing=True)
    learner.load_edge_signs(old)
    learner.load_adam_state(old["optimizer"])
    learner.sgd.load_state_dict(old["sgd"])
    learner.previous_token = old["previous_token"]
    learner.events, learner.updates = old["events"], old["updates"]
    learner.physical_ticks = old["physical_ticks"]
    learner.ema = old["ema"]
    learner.latent_window.copy_(old["latent_window"].cuda())
    for name, parameter in model.named_parameters():
        if name not in saved["model"] or not torch.equal(parameter.detach().cpu(), saved["model"][name]):
            raise ValueError(f"Restored parameter differs from the source checkpoint: {name}")
    report["all_restored_parameters_exact"] = True
    initial_events, initial_updates, initial_ticks = learner.events, learner.updates, learner.physical_ticks
    initial_cursor = int(saved["train_cursor"])
    train = np.load(Path(cfg["data"]) / "train.npy", mmap_mode="r")
    targets = np.asarray(train[initial_cursor + 1:initial_cursor + 33], dtype=np.int64).copy()
    if len(targets) != 32:
        raise ValueError("Insufficient fresh targets for the full window")
    report.update(source_train_tokens=saved["bptt_train_tokens"],
                  target_interval=[initial_cursor + 1, initial_cursor + 33],
                  target_sha256=hashlib.sha256(targets.tobytes()).hexdigest(),
                  trainable_parameters=sum(p.numel() for p in learner.trainable))
    parameter_names = {id(p): name for name, p in model.named_parameters()}
    original_clip = learning.stable_clip_grad_norm_

    def diagnostic_clip(parameters, max_norm, **kwargs):
        report["seconds_to_clip_including_group_norms"] = time.perf_counter() - begin
        parameters = list(parameters)
        details = {}
        fp32_norms = []
        for parameter in parameters:
            grad = parameter.grad
            if grad is None or not grad.numel():
                continue
            lo, hi = torch.aminmax(grad.detach())
            fp32_norm = grad.norm()
            details[parameter_names[id(parameter)]] = {
                "entries_finite": bool(torch.isfinite(grad).all()),
                "min": float(lo), "max": float(hi),
                "fp32_norm": float(fp32_norm),
                "fp64_norm": float(stable_grad_norm([parameter]))}
            fp32_norms.append(fp32_norm)
        report["gradients_before_clip"] = details
        report["legacy_total_fp32_norm"] = float(torch.linalg.vector_norm(torch.stack(fp32_norms)))
        report["stable_total_fp64_norm"] = float(stable_grad_norm(parameters))
        report["all_gradient_entries_finite"] = all(d["entries_finite"] for d in details.values())
        memory_guard("backward_before_clip")
        norm = original_clip(parameters, max_norm, **kwargs)
        report["total_norm_after_clip"] = float(stable_grad_norm(parameters))
        report["parameter_norms_after_clip"] = {
            parameter_names[id(parameter)]: float(stable_grad_norm([parameter]))
            for parameter in parameters if parameter.grad is not None}
        memory_guard("after_clip")
        return norm

    learning.stable_clip_grad_norm_ = diagnostic_clip
    audit = None
    if args.audit_recomputation:
        from scripts.ib.fly_checkpoint_audit import CheckpointSpikeAudit
        audit = CheckpointSpikeAudit(track_proxy=surrogate_mode == 'threshold')
    elif args.audit_adjoints:
        from scripts.ib.fly_adjoint_audit import FlyAdjointAudit
        audit = FlyAdjointAudit(model)
    elif args.pulse_credit_path != 'all':
        from scripts.ib.fly_pulse_credit_audit import PulseCreditAudit
        audit = PulseCreditAudit(model, args.pulse_credit_path)
    elif args.audit_feedback_bound:
        from scripts.ib.fly_feedback_bound import FlyFeedbackBound
        audit = FlyFeedbackBound(model)
    audit_key = ('feedback_bound' if args.audit_feedback_bound else
                 'pulse_credit_intervention' if args.pulse_credit_path != 'all' else
                 'state_adjoints' if args.audit_adjoints else 'checkpoint_recomputation')
    original_steps = (learner.optimizer.step, learner.sgd.step)
    if args.backward_only:
        learner.optimizer.step = lambda *unused, **kwargs: None
        learner.sgd.step = lambda *unused, **kwargs: None
    elif args.audit_recomputation:
        def guarded_step(step):
            def apply(*positional, **keyword):
                check = audit.summary()
                if (not check['coverage_complete'] or check['total_spike_flips'] or
                        not check['all_voltages_finite'] or not check['all_proxies_finite']):
                    raise AssertionError('Checkpoint recomputation gate failed before update')
                return step(*positional, **keyword)
            return apply
        learner.optimizer.step = guarded_step(original_steps[0])
        learner.sgd.step = guarded_step(original_steps[1])
    begin = time.perf_counter()
    try:
        memory_guard("restored")
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        if audit is None:
            scores, metrics = learner.observe(targets)
        else:
            with audit:
                if args.forward_only:
                    tokens = torch.as_tensor(targets, device=learner.state.h.device)
                    scores, next_state, _ = learner.forward_window(learner.inputs_for_targets(tokens), tokens[None])
                    scores = scores.detach().cpu().flatten().tolist()
                    learner.state = next_state.detached()
                    learner.events += len(targets)
                    learner.physical_ticks += len(targets) * 15
                    metrics = {}
                else:
                    scores, metrics = learner.observe(targets)
            report[audit_key] = audit.summary()
        memory_guard("after_window" if args.forward_only or args.backward_only else
                     "after_optimizer_update")
        report["seconds_update_including_gradient_diagnostics"] = time.perf_counter() - begin
        report.update(status="passed", pre_update_nll=float(np.mean(scores)),
                      metrics=metrics, optimizer_updates=0 if args.backward_only else learner.updates - initial_updates,
                      suppressed_update_calls=learner.updates - initial_updates if args.backward_only else 0,
                      events=learner.events - initial_events,
                      physical_ticks=learner.physical_ticks - initial_ticks,
                      all_parameters_finite=all(bool(torch.isfinite(p).all()) for p in learner.trainable))
        report["pre_update_scores"] = [float(score) for score in scores]
        report["pre_update_training_loss"] = float(learner.last_train_loss.detach().mean())
        # observe carries the pre-update forward state into the next window.
        # Hash every physical channel, including pending delayed pulses.
        state_hashes = {}
        for name, value in learner.state.state_dict().items():
            tensors = value if name == "ring" else (value,)
            state_hashes[name] = [hashlib.sha256(
                tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                for tensor in tensors]
        report["pre_update_physical_state_hashes"] = state_hashes
        updates = {}
        with torch.no_grad():
            for parameter in learner.trainable:
                name = parameter_names[id(parameter)]
                # The checkpoint remains on CPU; compare one parameter at a time.
                delta = parameter.detach().cpu() - saved["model"][name]
                updates[name] = {"delta_l2": float(torch.linalg.vector_norm(delta, dtype=torch.float64)),
                                 "delta_max_abs": float(delta.abs().max()),
                                 "original_l2": float(torch.linalg.vector_norm(saved["model"][name], dtype=torch.float64))}
        report["parameter_updates"] = updates
        if (args.backward_only or args.forward_only) and any(update["delta_max_abs"] != 0 for update in updates.values()):
            raise AssertionError("Backward-only diagnostic changed a parameter")
        if not report["all_parameters_finite"]:
            report["status"] = "failed_nonfinite_parameters"
        if not args.backward_only and not args.forward_only:
            report['all_optimizer_tensors_finite'] = all(
                bool(torch.isfinite(chunk).all())
                for optimizer in (learner.optimizer, learner.sgd)
                for state_values in optimizer.state.values()
                for value in state_values.values() if isinstance(value, torch.Tensor)
                for chunk in value.detach().reshape(-1).split(1 << 20))
            if not report['all_optimizer_tensors_finite']:
                report['status'] = 'failed_nonfinite_optimizer'
        memory_guard('final_verified')
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        try:
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    report.setdefault("nonfinite_gradient_parameters", []).append(name)
        except Exception as secondary:
            report['gradient_inspection_error'] = f'{type(secondary).__name__}: {secondary}'
    finally:
        learning.stable_clip_grad_norm_ = original_clip
        learner.optimizer.step, learner.sgd.step = original_steps
        if audit is not None:
            report[audit_key] = audit.summary()
        report["seconds_including_diagnostics"] = time.perf_counter() - begin
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Strict JSON: non-finite diagnostics are represented explicitly as strings.
        def safe(value):
            if isinstance(value, dict):
                return {k: safe(v) for k, v in value.items()}
            if isinstance(value, list):
                return [safe(v) for v in value]
            if isinstance(value, float) and not np.isfinite(value):
                return str(value)
            return value
        args.output.write_text(json.dumps(safe(report), ensure_ascii=False, indent=2,
                                         allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in safe(report).items()
                          if k not in ("parameter_updates", "gradients_before_clip", "metrics",
                                       "checkpoint_recomputation", "state_adjoints", "pulse_credit_intervention", "feedback_bound",
                                       "pre_update_physical_state_hashes", "pre_update_scores")},
                         ensure_ascii=False, indent=2), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
