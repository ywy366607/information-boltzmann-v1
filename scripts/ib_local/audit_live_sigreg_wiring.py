"""Check live SIGReg and F target gradients in a continuing real body."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.ib_local import gdn_fused, language_navigation_kanli_v4i
from scripts.ib_local import run_language_navigation_kanli_v4 as runner
from scripts.ib_local.language_navigation_kanli_v4l import LiveSigregBody, LiveTargetEngine


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cycles', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    gdn_fused.enable()
    runner.LatentNavigationV3 = LiveSigregBody
    language_navigation_kanli_v4i.FilmEngine = LiveTargetEngine
    engine, _, _, cfg = runner.build(equal=True, reply=True, film=True, capture=False, validation_bytes=4096)
    original = engine._outcome_loss
    original_objective = engine.model.objective
    last = {}
    history_grads = []

    def observe_objective(*args, **kwargs):
        history_outputs = []
        def hook(module, inputs, output):
            if inputs[0].data_ptr() == engine.model.memory.history.data_ptr():
                history_outputs.append(output)
        handle = engine.model.encoder.register_forward_hook(hook)
        try:
            result = original_objective(*args, **kwargs)
        finally:
            handle.remove()
        if history_outputs and history_outputs[-1].requires_grad:
            grad = torch.autograd.grad(result[0], history_outputs[-1], retain_graph=True,
                                       allow_unused=True)[0]
            history_grads.append(float(grad.norm()) if grad is not None else 0.)
        return result

    engine.model.objective = observe_objective

    def observe(source, action, future):
        outputs = []
        def hook(module, inputs, output):
            outputs.append(output)
        handle = engine.model.encoder.register_forward_hook(hook)
        try:
            loss, diagnostics = original(source, action, future)
        finally:
            handle.remove()
        # In LiveTargetEngine the first encoder call is enc(future).
        target_grad = torch.autograd.grad(loss, outputs[0], retain_graph=True, allow_unused=True)[0]
        last.clear()
        last['target_grad_norm'] = float(target_grad.norm()) if target_grad is not None else 0.
        return loss, diagnostics

    engine._outcome_loss = observe
    rows = []
    for _ in range(args.cycles):
        item = engine.step(torch.randn(64, 32, device='cuda'))
        rows.append(dict(choice=item['choice'], outcome_target_grad=last['target_grad_norm'],
                         sigreg_history_grad=history_grads[-2:],
                         response_nll=item['response_nll'], updates=item['updates'],
                         replay_error=item['policy_replay_max_error']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(config=cfg, records=rows), indent=2), encoding='utf-8')
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
