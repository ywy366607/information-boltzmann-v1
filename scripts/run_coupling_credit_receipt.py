"""Trace actual coupling credit and executed updates in one private life round.

Only instrumentation is installed in this subprocess. Production operators,
rewards, gradients, policy, gates and parameter values execute unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys

import torch

from information_boltzmann.budgeted_adamw import BudgetedAdamW
from scripts.ib_local.adaptive_hamiltonian import StateBridge, MomentumReadBridge


def main():
    args = sys.argv[1:]
    work = Path(args[args.index("--work")+1]).resolve()
    work.parent.mkdir(parents=True, exist_ok=True)
    receipts = work/"coupling_update_receipts.jsonl"
    sequence = 0
    update = 0

    def write(kind, **fields):
        nonlocal sequence
        sequence += 1
        with receipts.open("a", encoding="utf8") as stream:
            stream.write(json.dumps(dict(sequence=sequence, optimizer_call=update, kind=kind, **fields))+"\n")

    def credit(kind, result):
        parameter = result[1]
        if parameter is not None and parameter.ndim == 2 and parameter.shape == (32, 12):
            write(kind, coupling=parameter[:, 9].detach().cpu().tolist(),
                  role_norms=parameter.detach().norm(dim=0).cpu().tolist())

    original_state = StateBridge.backward
    original_momentum = MomentumReadBridge.backward

    def state_backward(ctx, gradient):
        result = original_state(ctx, gradient)
        credit("state_credit_return", result)
        return result

    def momentum_backward(ctx, gradient):
        result = original_momentum(ctx, gradient)
        credit("momentum_credit_return", result)
        return result

    StateBridge.backward = staticmethod(state_backward)
    MomentumReadBridge.backward = staticmethod(momentum_backward)
    original_step = BudgetedAdamW.step

    def observed_step(optimizer, *positional, **keywords):
        nonlocal update
        update += 1
        matches = [group for group in optimizer.param_groups
                   if [tuple(p.shape) for p in group["params"]] == [(32, 8), (32, 1), (32, 1), (32, 2)]]
        if len(matches) != 1:
            raise RuntimeError("expected one actual split Hamiltonian role group")
        group = matches[0]
        parameter = group["params"][2]
        before = parameter.detach().clone()
        gradient = None if parameter.grad is None else parameter.grad.detach().clone()
        result = original_step(optimizer, *positional, **keywords)
        after = parameter.detach()
        record = dict(lr=group["lr"], weight_decay=group["weight_decay"],
                      before=before.cpu().flatten().tolist(), after=after.cpu().flatten().tolist(),
                      gradient=None if gradient is None else gradient.cpu().flatten().tolist())
        if gradient is not None:
            state = optimizer.state[parameter]
            count = float(state["step"])
            b1, b2 = group["betas"]
            m = state["exp_avg"]/(1-b1**count)
            rms = (state["exp_avg_sq"]/(1-b2**count)).sqrt()
            direction = m/(rms+group["eps"])
            gate = state["budget_gate"]
            proposal = -group["lr"]*(direction+group["weight_decay"]*before)
            expected_after = before+proposal*gate
            error = float((after-expected_after).abs().max())
            assert error <= 1e-6, f"executed step reconstruction failed: {error}"
            relative = (after.exp()-before.exp())/before.exp()
            record.update(step=count, gate=float(gate), adam_direction=direction.cpu().flatten().tolist(),
                          ungated_raw_step=proposal.cpu().flatten().tolist(),
                          executed_raw_step=(after-before).cpu().flatten().tolist(),
                          relative_exp_parameter_change=relative.cpu().flatten().tolist(),
                          reconstructed_parameter_max_error=error,
                          epsilon_fraction=(group["eps"]/(rms+group["eps"])).cpu().flatten().tolist())
        write("optimizer_execution", **record)
        return result

    BudgetedAdamW.step = observed_step
    sys.argv = ["scripts.run_native_throughput_worker", *args]
    runpy.run_module("scripts.run_native_throughput_worker", run_name="__main__")


if __name__ == "__main__":
    main()
