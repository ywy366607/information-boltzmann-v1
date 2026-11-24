"""Crash/recover actual whole-body natural-text events; never write main body."""
import argparse
import json
from pathlib import Path

import torch

from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation, ROOT
from scripts.ib_local.event_journal import EventJournal, execution_digest, rng_digest
from scripts.ib_local.native_navigation_body import restore_birth_counters, record_birth_event
from scripts.ib_local.mother_checkpoint import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--body', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=('reference','crash','optimized'), required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    args = continuation_parser().parse_args(['--checkpoint', str(a.body/'last.pt.gz'),
        '--output', str(a.body), '--cycles', '2', '--resume-output', '--world-source', 'text',
        '--table-storage', str(a.output/a.arm/'table'),
        '--causal-storage', str(a.output/a.arm/'causal')])
    loaded = prepare_continuation(args, resume_output=True)
    e, config = loaded.engine, loaded.config
    restore_birth_counters(e, loaded.cycle, a.body/'metrics.jsonl')
    counter = [0]; original = e.optimizer.step
    def step(*args, **kw):
        out = original(*args, **kw); counter[0] += 1; return out
    e.optimizer.step = step
    journal_dir = a.output/'recovery'; journal_dir.mkdir(exist_ok=True)
    journal = None if a.arm == 'reference' else EventJournal(journal_dir, cycle=loaded.cycle,
        checkpoint_sha=loaded.source_sha, code_sha=execution_digest(ROOT), config=config)
    pending = [] if journal is None else list(journal.pending)
    rows = []
    for i in range(1 if a.arm == 'crash' else 2):
        before = rng_digest(); count = counter[0]
        row = e.step(torch.randn(e.model.encoder.out_features, 32, device=e.device))
        row['updates'] = counter[0]-count; record_birth_event(e, row)
        after = rng_digest()
        if i < len(pending): journal.verify(pending[i], row, before, after)
        elif journal is not None: journal.append(loaded.cycle+i+1, row, before, after)
        rows.append(row)
    e.optimizer.step = original
    if a.arm != 'crash':
        target = a.output/f'{a.arm}_checkpoint'; target.mkdir(exist_ok=True)
        save(e, target, loaded.cycle+2, config)
    report = dict(source_cycle=loaded.cycle, events=len(rows), recovered=len(pending),
                  optimizer_updates=counter[0], rows=rows, final_rng=rng_digest(), main_written=False)
    (a.output/f'{a.arm}.json').write_text(json.dumps(report,ensure_ascii=True)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}), flush=True)
    e.model._causal_archive.db.close()


if __name__ == '__main__': main()
