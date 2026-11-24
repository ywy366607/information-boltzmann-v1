"""Diagnose the actual complete-body replay failure in private working stores."""
import argparse
import json
from pathlib import Path
import time

import torch
from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
from scripts.ib_local import variable_expression


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    options = ['--checkpoint', str(args.body/'last.pt.gz'), '--output', str(args.body),
        '--cycles', '1', '--resume-output', '--world-source', 'text',
        '--table-storage', str(args.output/'table'),
        '--causal-storage', str(args.output/'causal')]
    loaded = prepare_continuation(continuation_parser().parse_args(options), resume_output=True)
    engine = loaded.engine
    original = variable_expression.proposal_score
    receipts, updates = [], [0]
    old_step = engine.optimizer.step
    after_draft = {}
    utter = engine._utter
    def snapshot_utter():
        result = utter()
        after_draft['state'] = engine.recurrent.detach().cpu().clone()
        after_draft['buffers'] = {n:b.detach().cpu().clone() for n,b in engine.model.named_buffers()}
        after_draft['parameter_versions'] = {n:p._version for n,p in engine.model.named_parameters()}
        after_draft['question'] = (None if engine.model._question_context is None else
                                   engine.model._question_context.detach().cpu().clone())
        return result
    engine._utter = snapshot_utter
    def counted(*a, **kw):
        result = old_step(*a, **kw)
        updates[0] += 1
        return result
    engine.optimizer.step = counted

    def diagnosed(e, proposal):
        before = getattr(e, '_batched_replay_fallbacks', 0)
        try:
            result = original(e, proposal)
            receipts.append(dict(proposal=len(receipts), bytes=len(proposal['words']),
                                 accepted_error=result[1]))
            return result
        except RuntimeError as error:
            trace = error.__traceback__
            while trace is not None and trace.tb_frame.f_code is not original.__code__:
                trace = trace.tb_next
            if trace is not None:
                values = trace.tb_frame.f_locals
                byte_error = float((values['byte_terms'].detach()-
                    values['byte_terms'].new_tensor(proposal['logp'])).abs().max())
                stop_error = (float((values['logits'].detach()-values['expected']).abs().max())
                              if 'logits' in values else None)
                receipts.append(dict(proposal=len(receipts), bytes=len(proposal['words']),
                    byte_error=byte_error, stop_error=stop_error,
                    exact_fallbacks=getattr(e, '_batched_replay_fallbacks', 0)-before,
                    stop_positions=len(proposal['handover']['stops']), error=str(error),
                    recurrent_change=float((e.recurrent.detach().cpu()-after_draft['state']).abs().max()),
                    question_change=(float((e.model._question_context.detach().cpu()-
                        after_draft['question']).abs().max()) if after_draft['question'] is not None else None),
                    changed_buffers=[n for n,b in e.model.named_buffers()
                        if not torch.equal(b.detach().cpu(),after_draft['buffers'][n])],
                    changed_parameter_versions=[n for n,p in e.model.named_parameters()
                        if p._version!=after_draft['parameter_versions'][n]]))
            raise

    variable_expression.proposal_score = diagnosed
    start = time.perf_counter()
    print(json.dumps(dict(stage='loaded', source_cycle=loaded.cycle, source_sha=loaded.source_sha)), flush=True)
    row, failure = None, None
    try:
        row = engine.step(torch.randn(engine.model.encoder.out_features, 32, device=engine.device))
    except RuntimeError as error:
        failure = str(error)
    finally:
        variable_expression.proposal_score = original
    torch.cuda.synchronize()
    report = dict(source_cycle=loaded.cycle, source_sha=loaded.source_sha,
        seconds=time.perf_counter()-start, failure=failure, proposal_receipts=receipts,
        optimizer_updates=updates[0], main_written=False,
        scope='one actual next complete event on private state; no source save or API calls',
        policy_error=None if row is None else row.get('policy_replay_max_error'),
        replay=None if row is None else row.get('runtime_replay'))
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report), flush=True)
    engine.model._causal_archive.db.close()


if __name__ == '__main__':
    main()
