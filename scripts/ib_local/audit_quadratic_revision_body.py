"""Check common quadratic revision on the stopped real body's own records.

No training, world interaction, persistent model write or synthetic task.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local.causal_relation_program import posterior
from scripts.ib_local.generative_relations import dataset
from scripts.ib_local.language_action import peek
from scripts.ib_local.run_causal_water_reply import sha256_file
from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--body', type=Path, required=True)
    p.add_argument('--storage', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    checkpoint = args.body/'last.pt.gz'
    options = continuation_parser().parse_args([
        '--checkpoint', str(checkpoint), '--output', str(args.body), '--cycles', '1',
        '--resume-output', '--world-source', 'text', '--table-storage', str(args.storage/'table'),
        '--causal-storage', str(args.storage/'causal')])
    loaded = prepare_continuation(options, resume_output=True)
    engine, model = loaded.engine, loaded.engine.model
    records = []
    bank = model.causal_relations
    with torch.no_grad():
        for i in torch.nonzero(bank.identity > 0).flatten().tolist():
            state = bank.state(i)
            x, y, _, refs = dataset(model, state)
            if len(x) < 2:
                continue
            if any(ref is None for ref in refs):
                raise RuntimeError('numerical audit only accepts individually identified real facts')
            x, y = x.double(), y.double()
            ridge = model.causal_relation_ridge
            result = posterior(x, y, x, ridge)
            # Independent primal solution for the same actual residual data.
            precision = ridge*torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
            primal = torch.linalg.solve(precision+x.T@x, x.T@y)
            # Rank-one assimilation for each real evidence row, with no release
            # or feature drift inside this fixed-coordinate numerical check.
            covariance = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)/ridge
            operator = torch.zeros_like(primal)
            negative = total = 0
            for row, target in zip(x, y):
                direction = covariance@row
                gain = direction/(1+row@direction)
                operator = operator+gain[:, None]*(target-row@operator)[None]
                covariance = covariance-gain[:, None]*direction[None]
                negative += int((gain < 0).sum()); total += gain.numel()
            records.append(dict(identity=state.identity, facts=len(x), coordinates=x.shape[1],
                output_width=y.shape[1],
                dual_primal_error=float((result['mean']-x@primal).abs().max()),
                incremental_batch_error=float((operator-primal).abs().max()),
                gain_negative_entries=negative, gain_entries=total))
    # Observe actual calls to the workspace recurrence; compare each one-byte
    # update to a dense precision-weighted quadratic minimum independently.
    from scripts.ib_local import past_present_coevolution as coev
    original = coev.causal_scan
    captured = []
    def observe(a, u, initial):
        result = original(a, u, initial)
        captured.append(tuple(t.detach().cpu().double() for t in (a, u, initial, result)))
        return result
    recurrent_before = engine.recurrent.clone()
    probability = peek(engine.table, engine.pending)
    ids = torch.tensor([[engine.pending]], device=engine.device)
    lp = torch.tensor(np.log(np.maximum(probability, 1e-6)), device=engine.device)[None, None]
    try:
        coev.causal_scan = observe
        with torch.no_grad():
            model.predict_observed(ids, engine.recurrent, lp)
    finally:
        coev.causal_scan = original
    workspace = []
    for layer, (a, u, old, result) in enumerate(captured):
        a, u, result = a[0, 0], u[0, 0], result[0, 0]
        # alpha is the old-precision fraction. Inactive slots have zero new
        # precision and remain unchanged, using a stable denominator.
        if bool((a <= 0).any()):
            raise RuntimeError('this real audit needs finite positive prior fractions')
        prior_ratio = a[:, 0]
        dense = torch.linalg.solve(torch.diag(1/prior_ratio), old[0]+u/prior_ratio[:, None])
        workspace.append(dict(layer=layer+1, dense_error=float((dense-result).abs().max())))
    report = dict(schema=1, cycle=loaded.cycle, source_sha256=loaded.source_sha,
        checkpoint_unchanged=sha256_file(checkpoint)==loaded.source_sha,
        recurrent_unchanged=torch.equal(recurrent_before, engine.recurrent),
        optimizer_updates=0, world_calls=0, added_parameters=0,
        relations=records, workspace=workspace,
        interpretation='shared quadratic rule; full evidence geometry versus learned diagonal workspace gain; numerical equivalence, not new capability')
    if not records or len(workspace) != len(model.base.layers):
        raise RuntimeError('real-body audit omitted actual relations or layers')
    if (max(r['dual_primal_error'] for r in records)>1e-6 or
            max(r['incremental_batch_error'] for r in records)>1e-6 or
            max(r['dense_error'] for r in workspace)>1e-6 or
            not report['checkpoint_unchanged'] or not report['recurrent_unchanged']):
        raise RuntimeError('shared quadratic numerical equivalence failed')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(cycle=loaded.cycle, relations=len(records), layers=len(workspace),
        max_relation_error=max(r['dual_primal_error'] for r in records),
        max_incremental_error=max(r['incremental_batch_error'] for r in records),
        max_workspace_error=max(r['dense_error'] for r in workspace),
        checkpoint_unchanged=report['checkpoint_unchanged'], optimizer_updates=0)))
    model._causal_archive.db.close()


if __name__ == '__main__':
    main()
