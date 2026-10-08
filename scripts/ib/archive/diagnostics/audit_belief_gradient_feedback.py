"""Locate recurrent gradient amplification without changing forward predictions.

Backward-only detach interventions diagnose feedback; they are not proposed
training rules or capability ablations. Use the same saved belief and tokens.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.torus3d import KineticBeliefState
from scripts.ib.evaluate_continuous_owt import _model_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--windows", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--native-collision", action="store_true")
    parser.add_argument("--event-duration", type=float)
    parser.add_argument("--modes", nargs="+", default=[
        "full_backward", "detach_address_feedback",
        "detach_collision_state_feedback", "detach_both_feedbacks"], choices=[
        "full_backward", "detach_address_feedback",
        "detach_collision_state_feedback", "detach_both_feedbacks"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, _, _ = _model_from_checkpoint(saved)
    model.load_state_dict(saved["model"])
    model.to(args.device)
    if args.event_duration is not None:
        model.tau_0 = args.event_duration / model.micro_steps
        model.tau_0_tensor.fill_(model.tau_0)
    if args.native_collision:
        model.collision._triton_givens = None
        model.collision._projection_ops = None
    stream = np.load(Path(saved["config"]["data"]) / "train.npy", mmap_mode="r")
    offset = int(saved["step"]) * int(saved["config"]["tokens"])
    ids = torch.as_tensor(np.array(stream[offset:offset + args.tokens]),
                          dtype=torch.long, device=args.device)[None]
    targets = torch.as_tensor(np.array(stream[offset + 1:offset + args.tokens + 1]),
                              dtype=torch.long, device=args.device)[None]
    original_anchor = model.source.field_anchor
    original_norm = model.collision.norm.forward
    original_step = model.belief_step
    reports = []
    for window in range(args.windows):
        window_offset = offset + window * args.tokens
        ids = torch.as_tensor(np.array(stream[window_offset:window_offset + args.tokens]),
                              dtype=torch.long, device=args.device)[None]
        targets = torch.as_tensor(np.array(stream[window_offset + 1:window_offset + args.tokens + 1]),
                                  dtype=torch.long, device=args.device)[None]
        baseline_loss = None
        for mode in args.modes:
            model.source.field_anchor = original_anchor
            model.collision.norm.forward = original_norm
            if mode in ("detach_address_feedback", "detach_both_feedbacks"):
                model.source.field_anchor = lambda field: original_anchor(field).detach()
            if mode in ("detach_collision_state_feedback", "detach_both_feedbacks"):
                model.collision.norm.forward = lambda field: original_norm(field).detach()
            input_grads = {}
            index = [0]
    
            def trace_step(belief, *positional, **keyword):
                token_index = index[0]
                index[0] += 1
                if belief.field.requires_grad:
                    belief.field.register_hook(
                        lambda gradient, i=token_index: input_grads.__setitem__(
                            i, float(gradient.detach().norm().cpu())))
                return original_step(belief, *positional, **keyword)
    
            model.belief_step = trace_step
            model.zero_grad(set_to_none=True)
            belief = KineticBeliefState(
                saved["state"].to(args.device).detach().requires_grad_(),
                saved["precision"].to(args.device).detach().requires_grad_())
            loss, next_belief, _ = model.forward_belief(ids, targets, belief)
            loss_value = float(loss.detach().cpu())
            if baseline_loss is None:
                continuation = (next_belief.field.detach().cpu().clone(),
                                next_belief.precision.detach().cpu().clone())
            if baseline_loss is None:
                baseline_loss = loss_value
            assert abs(loss_value - baseline_loss) < 1e-6, "Forward changed during audit"
            loss.backward()
            group_squares = {}
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    group = name.split(".")[0]
                    group_squares[group] = group_squares.get(group, 0.0) + float(
                        parameter.grad.detach().double().square().sum().cpu())
            row = {"mode": mode, "window": window, "loss": loss_value,
                   "module_grad_norms": {k: v ** 0.5 for k, v in group_squares.items()},
                   "total_grad_norm": sum(group_squares.values()) ** 0.5,
                   "input_field_grad_by_token": [input_grads[i] for i in range(args.tokens)]}
            reports.append(row)
            print(json.dumps(row), flush=True)
            del loss, next_belief, belief
            model.zero_grad(set_to_none=True)
        saved["state"], saved["precision"] = continuation
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "step": int(saved["step"]),
        "tokens": args.tokens, "windows": args.windows, "offset": offset,
        "native_collision": args.native_collision,
        "event_duration": float(model.micro_steps * model.tau_0),
        "scope": "backward-only feedback audit, identical forward, no optimizer update",
        "reports": reports}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
