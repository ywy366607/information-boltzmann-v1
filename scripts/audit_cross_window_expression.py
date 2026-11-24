"""Read and withdraw old episodes from the actual post-2612 draft path."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from scripts.audit_felt_expression_2612 import SOURCE, SpeechReady
from scripts.ib_local import variable_expression as expression
from scripts.ib_local.adaptive_hamiltonian import bridge_state
from scripts.ib_local.autonomous_cycle import dispatch
from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation


def measure(engine, draft, probs, receipt):
    model, m = engine.model, engine.model.memory
    field = model.body.coevolution
    valid = field.valid.clone()
    age = m.clock-m.last_write[field.slots_id]
    old = valid & (age > m.long)
    assert bool(old.any()) and torch.equal(m.read_credit_version[field.slots_id][valid], field.versions[valid])
    assert m.read_credit_enabled and not m._read_credit_active
    n = min(16, len(draft))
    ids = torch.tensor([[engine.pending, *draft[:n]]], device=engine.device, dtype=torch.long)
    logp = torch.as_tensor(np.log(np.maximum(np.stack([*probs, receipt['final_probability']][:n+1]), 1e-6)),
                           device=engine.device)[None]
    remaining = 1-torch.arange(n+1, device=engine.device, dtype=logp.dtype)[None]/engine.max_utterance
    state = bridge_state(model, engine.recurrent.detach())

    def forward():
        with expression.budget_scope(model, remaining), expression.read_route_scope(engine, receipt, 0, n+1):
            out = model.predict_observed(ids, state, logp)
        return out[0][0].log_softmax(-1), expression.handover_logits(model, out[3], remaining[0,:,None])

    m.begin_read_credit()
    try:
        with torch.enable_grad():
            native, stop = forward()
            choice = native[:n].gather(1, ids[0,1:,None]).mean()
            participation = next(r for r in m._read_participation if r.slots.shape == field.slots_id.shape
                                 and torch.equal(r.slots, field.slots_id))
            grad = torch.autograd.grad(choice, participation.bias)[0]
        native, stop = native.detach(), stop.detach()
    finally:
        m._read_participation = []
        m._read_credit_entries = 0
        m._read_credit_active = False
    try:
        field.valid.copy_(valid & ~old)
        # Use the ordinary differentiable path in both arms, not the cached CUDA step.
        with torch.enable_grad():
            removed, removed_stop = forward()
        removed, removed_stop = removed.detach(), removed_stop.detach()
    finally:
        field.valid.copy_(valid)
    sampled = native[:n].gather(1, ids[0,1:,None]).flatten()
    tv = (native.exp()-removed.exp()).abs().sum(-1)*.5
    return dict(prefix_bytes=n, window=int(m.long), memory_clock=int(m.clock),
        draft_prefix=draft[:n].hex(), native_sample_max_logp_error=None,
        workset=[dict(slot=int(s), version=int(v), age=int(a), valid=bool(b), withdrawn=bool(o),
                   current_logp_derivative=float(g))
                 for s,v,a,b,o,g in zip(field.slots_id, field.versions, age, valid,old,grad)],
        native_sample_logp=sampled.tolist(), mean_byte_tv=float(tv[:n].mean()),
        max_byte_tv=float(tv[:n].max()), max_handover_probability_delta=float(
            (stop[1:].sigmoid()-removed_stop[1:].sigmoid()).abs().max()))


def run(work):
    opts = continuation_parser().parse_args(['--checkpoint', str(SOURCE), '--output', str(work/'body'),
        '--table-storage', str(work/'table'), '--causal-storage', str(work/'causal'),
        '--world-source', 'text', '--mode', 'real', '--cycles', '1'])
    loaded = prepare_continuation(opts)
    engine = loaded.engine
    original = expression.film
    report = {}

    def probe(owner, words, table):
        result = original(owner, words, table)
        if words is None:
            assert not getattr(owner, '_draft_recall', ())
            report.update(measure(owner, result[0], result[2], owner._last_expression_receipt))
            sampled = torch.tensor(result[1][:report['prefix_bytes']])
            recovered = torch.tensor(report.pop('native_sample_logp'))
            report['native_sample_max_logp_error'] = float((sampled-recovered).abs().max())
            raise SpeechReady
        return result

    try:
        with patch.object(expression, 'film', probe), patch.object(engine.world, 'take',
                side_effect=AssertionError('world access forbidden')):
            try:
                for _ in range(64):
                    dispatch(engine)
            except SpeechReady:
                return dict(source_cycle=2612, updates=0, world_calls=0, **report)
            raise RuntimeError('no draft boundary reached')
    finally:
        engine.model._causal_archive.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    report = run(args.work)
    with args.report.open('x', encoding='utf8') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(report), flush=True)
