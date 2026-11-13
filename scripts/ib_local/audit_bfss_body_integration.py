"""One read-only real-checkpoint S/P wiring audit; no optimizer/world step."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.bfss_gdn_capture import matrix_view, quiet_packed
from scripts.ib_local.language_action import peek
from scripts.ib_local.navigation_bfss_dynamics import configure_momentum
from scripts.ib_local.navigation_internal_rollout import next_boundary_prediction
from scripts.ib_local.run_causal_water_reply import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--length', type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.length <= 128 or args.output.exists():
        raise ValueError('length must be 1..128 and output must be new')
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the existing fused body')
    gdn_fused.enable()
    engine, valid, offset, provenance = build(args.checkpoint)
    configure_momentum(engine)
    model, device = engine.model, engine.device
    # Existing independent text; read without moving the individual's table or
    # world. A static actual table distribution is sufficient for wiring only.
    raw = np.asarray(valid[:args.length*4])
    raw = raw[raw >= offset][:args.length+1] - offset
    if len(raw) < args.length+1 or np.any(raw > 255):
        raise ValueError('validation segment is not sufficient byte text')
    ids = torch.as_tensor(raw[:-1].astype(np.int64), device=device)[None]
    targets = torch.as_tensor(raw[1:].astype(np.int64), device=device)[None]
    table_p = peek(engine.table, engine.pending)
    logp = torch.as_tensor(np.log(np.maximum(table_p, 1e-6)), device=device)
    logp = logp[None, None].expand(1, args.length, -1).contiguous()
    directions = torch.randn(64, 32, device=device)
    before = engine.recurrent.clone()
    model._momentum_active = True
    model._momentum_current = engine.bfss_momenta
    loss, written, _, _ = model.objective(ids, targets, before, logp, directions)
    captures = model._momentum_captured
    p = [item[0] for item in captures]
    unchanged, same_p = quiet_packed(model.body.base, written.detach(), p,
        dt=0., frequency=engine.bfss_frequency, coupling=engine.bfss_coupling, steps=1)
    moved, moved_p = quiet_packed(model.body.base, written.detach(), p,
        dt=engine.bfss_dt, frequency=engine.bfss_frequency,
        coupling=engine.bfss_coupling, steps=engine.bfss_steps)
    one = ids[:, -1:]
    table_one = logp[:, -1:]
    with torch.no_grad():
        logits0, _, _, h0, _ = model.predict_observed(one, written.detach(), table_one)
        logits1, _, _, h1, _ = model.predict_observed(one, moved, table_one)
        pred0 = next_boundary_prediction(model, h0, h0)
        pred1 = next_boundary_prediction(model, h1, h1)
    model.zero_grad(set_to_none=True)
    loss.backward()
    norms = [float(layer.q_proj.weight.grad.norm()) for layer in model.body.base.layers]
    capacities = []
    for i, layer in enumerate(model.body.base.layers):
        s = matrix_view(layer, moved[i])
        maximum = torch.maximum(s.abs().sum(-1).max(), s.abs().sum(-2).max())
        capacities.append(dict(budget=layer.content_budget, maximum=float(maximum),
            excess=max(0., float(maximum)-layer.content_budget) if layer.content_budget else 0.))
    report = dict(source=provenance, observed_validation_bytes=args.length,
        scope='same checkpoint, fixed table view, no optimizer or world transition',
        event_replay_max_error=[item[1] for item in captures],
        event_admission_correction_norm=[item[2] for item in captures],
        momentum_norm=[float(item[0].norm()) for item in captures],
        zero_dt_exact=torch.equal(unchanged, written.detach()) and
                      all(torch.equal(x, y) for x, y in zip(p, same_p)),
        quiet_state_difference=float((moved-written.detach()).norm()),
        quiet_momentum_difference=float(torch.stack([(x-y).norm() for x,y in zip(p,moved_p)]).norm()),
        hidden_difference=float((h1-h0).norm()),
        language_probability_tv=float((logits1.softmax(-1)-logits0.softmax(-1)).abs().sum()/2),
        next_boundary_prediction_difference=float((pred1-pred0).norm()),
        quiet_capacity=capacities,
        existing_q_gradient_norms=norms,
        gpu_peak_bytes=torch.cuda.max_memory_allocated())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
