"""Actual budgeted whole-body continuation: uninterrupted vs save/reload.

Run reference, first, resume in separate processes, then compare complete
checkpoints with the existing runtime auditors. The main source is read-only.
"""
import argparse
import json
from pathlib import Path

import torch

from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
from scripts.ib_local.native_navigation_body import restore_birth_counters, record_birth_event
from scripts.ib_local.mother_checkpoint import save
from scripts.ib_local.optimizer_policy import diagnostics
from information_boltzmann.budgeted_adamw import BudgetedAdamW


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--body',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=('reference','first','resume'),required=True)
    a = p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    source = a.output/'first_checkpoint' if a.arm == 'resume' else a.body
    opts = ['--checkpoint',str(source/'last.pt.gz'),'--output',str(source),
        '--cycles','1','--resume-output','--world-source','text',
        '--table-storage',str(a.output/a.arm/'table'),
        '--causal-storage',str(a.output/a.arm/'causal')]
    if a.arm != 'resume': opts.extend(['--optimizer-policy','budgeted'])
    loaded = prepare_continuation(continuation_parser().parse_args(opts),resume_output=True)
    e = loaded.engine
    if not isinstance(e.optimizer,BudgetedAdamW): raise RuntimeError('budget execution was not restored')
    restore_birth_counters(e,loaded.cycle,source/'metrics.jsonl')
    updates = [0]; original = e.optimizer.step
    def step(*args,**kwargs):
        value = original(*args,**kwargs); updates[0] += 1; return value
    e.optimizer.step = step
    events = 2 if a.arm == 'reference' else 1
    rows = []
    for _ in range(events):
        before = updates[0]
        row = e.step(torch.randn(e.model.encoder.out_features,32,device=e.device))
        row['updates'] = updates[0]-before
        record_birth_event(e,row); rows.append(row)
    e.optimizer.step = original
    target = a.output/f'{a.arm}_checkpoint'; target.mkdir(exist_ok=True)
    save(e,target,loaded.cycle+events,loaded.config)
    result = dict(source_cycle=loaded.cycle,events=events,updates=updates[0],
                  restored_class=type(e.optimizer).__name__,optimizer=diagnostics(e),
                  main_written=False,rows=rows)
    (a.output/f'{a.arm}.json').write_text(json.dumps(result,ensure_ascii=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','optimizer')}),flush=True)
    e.model._causal_archive.db.close()


if __name__=='__main__': main()
