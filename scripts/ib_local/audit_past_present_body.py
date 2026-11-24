"""Read-only audit of the real, complete co-evolving navigation checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local.language_action import peek
from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--body', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--storage', type=Path, required=True)
    a = p.parse_args()
    args = continuation_parser().parse_args([
        '--checkpoint', str(a.body/'last.pt.gz'), '--output', str(a.body),
        '--cycles', '1', '--resume-output', '--world-source', 'text',
        '--table-storage', str(a.storage/'table'), '--causal-storage', str(a.storage/'causal')])
    loaded = prepare_continuation(args, resume_output=True)
    e, m = loaded.engine, loaded.engine.model
    f = m.body.coevolution
    state_before = e.recurrent.clone()
    q = np.asarray(peek(e.table, e.pending))
    logp = torch.tensor(np.log(np.maximum(q, 1e-6)), device=e.device)[None, None]
    ids = torch.tensor([[e.pending]], dtype=torch.long, device=e.device)
    with torch.no_grad():
        eager = m.predict_observed(ids, e.recurrent, logp)
        e.decision.start(e.recurrent)
        captured = e.decision.step(e.pending, q)
        graph_error = float((captured[0]-eager[0][0, 0]).abs().max())
        state_error = float((e.decision.state-eager[1]).abs().max())
        f.enabled.zero_()
        disabled = m.predict_observed(ids, e.recurrent, logp)
        e.decision.start(e.recurrent)
        disabled_graph = e.decision.step(e.pending, q)
        disabled_graph_error = float((disabled_graph[0]-disabled[0][0, 0]).abs().max())
        f.enabled.fill_(1)
    # Use genuine upcoming source bytes for a local derivative check only;
    # no optimizer step, evidence commit, world advance or fabricated labels.
    position = int(e.world.cursor)
    raw = np.asarray(e.world.data[position:position+18], dtype=np.uint8)
    if len(raw) < 18:
        raw = np.asarray(e.world.data[:18], dtype=np.uint8)
    byte_ids = torch.tensor(raw[:-1].astype(np.int64), device=e.device)[None]
    targets = torch.tensor(raw[1:].astype(np.int64), device=e.device)
    lp = logp.expand(1, byte_ids.shape[1], 256).contiguous()
    result = m.predict_observed(byte_ids, e.recurrent, lp)
    loss = torch.nn.functional.cross_entropy(result[0][0], targets)
    parameters = list(f.parameters())
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    names = [n for n, _ in f.named_parameters()]
    steps = {name: int(e.optimizer.state.get(param, {}).get('step', 0))
             for name, param in f.named_parameters()}
    layer_values = [{k: float(result[2][f'l{i}_{k}']) for k in (
        'workspace_residual_rms', 'workspace_write', 'workspace_read_mass')}
        for i in range(len(m.base.layers))]
    report = dict(schema=1, checkpoint=str(a.body/'last.pt.gz'), cycle=loaded.cycle,
        source_sha256=loaded.source_sha, optimizer_updates=0, world_calls=0,
        trained_field_optimizer_steps=steps,
        graph_logit_max_error=graph_error, graph_state_max_error=state_error,
        live_disabled_graph_max_error=disabled_graph_error,
        field_logit_effect_l2=float((eager[0]-disabled[0]).norm()),
        field_gradient_norms={n: None if g is None else float(g.norm())
                              for n, g in zip(names, gradients)},
        layer_exchange=layer_values,
        physical_and_work_state_unchanged=torch.equal(state_before, e.recurrent),
        scope='real-body routing/gradient/capture admission; not a learning-benefit comparison')
    if max(graph_error, state_error, disabled_graph_error) > 1e-3:
        raise RuntimeError('captured field does not follow live buffers')
    if not report['physical_and_work_state_unchanged']:
        raise RuntimeError('read-only audit mutated the continuous recurrent state')
    if any(v['workspace_residual_rms'] == 0 for v in layer_values):
        raise RuntimeError('a layer bypassed the actual retrieved workspace')
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in (
        'layer_exchange', 'field_gradient_norms', 'trained_field_optimizer_steps')}))
    m._causal_archive.db.close()


if __name__ == '__main__':
    main()
