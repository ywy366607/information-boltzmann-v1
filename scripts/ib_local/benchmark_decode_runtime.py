"""Same real checkpoint/input/state, old versus exact draft-scoped decoding.

This is a numerical/runtime comparison, not a language or memory capability
test. No optimizer step, evidence write, world advance, or checkpoint update.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local.language_action import peek
from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
from scripts.ib_local.scoped_decision_graph import ScopedDecisionGraph


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--body', type=Path, required=True)
    p.add_argument('--storage', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--bytes', type=int, default=128)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--fused-step', action='store_true',
                   help='compare the current scoped graph against five-kernel layer decoding')
    a = p.parse_args()
    args = continuation_parser().parse_args([
        '--checkpoint', str(a.body/'last.pt.gz'), '--output', str(a.body),
        '--cycles', '1', '--resume-output', '--world-source', 'text',
        '--life-card', 'data/harness/life/decode-runtime-readonly.json',
        '--table-storage', str(a.storage/'table'), '--causal-storage', str(a.storage/'causal')])
    loaded = prepare_continuation(args, resume_output=True)
    e, m = loaded.engine, loaded.engine.model
    if next(m.parameters()).device.type != 'cuda':
        raise RuntimeError('this runtime comparison requires CUDA')
    state = e.recurrent.clone()
    probability = np.asarray(peek(e.table, e.pending))
    cursor = int(e.world.cursor)
    words = np.asarray(e.world.data[cursor:cursor+a.bytes], dtype=np.uint8).tolist()
    if len(words) != a.bytes:
        raise RuntimeError('not enough upcoming real text for fixed-input comparison')
    e.decision = None
    gc.collect(); torch.cuda.empty_cache()
    arms, traces = {}, {}
    with torch.no_grad():
        for name, optimize in [('original', a.fused_step), ('cached', True)]:
            graph = ScopedDecisionGraph(m, capture=True, optimize=optimize,
                                        fused_step=a.fused_step and name == 'cached')
            graph.start(state)
            for byte in words[:4]:
                graph.step(byte, probability)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timings, gpu_timings = [], []
            for _ in range(a.repeats):
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                wall = time.perf_counter()
                start.record()
                graph.start(state)
                logits = [graph.step(byte, probability)[0] for byte in words]
                stop.record(); stop.synchronize()
                timings.append(time.perf_counter()-wall)
                gpu_timings.append(start.elapsed_time(stop)/1000)
            traces[name] = (torch.stack(logits).cpu(), graph.state.cpu().clone())
            arms[name] = dict(wall_seconds=timings, gpu_seconds=gpu_timings,
                median_seconds=float(np.median(timings)),
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                bytes_per_second=a.bytes/float(np.median(timings)))
            del graph, logits
            gc.collect(); torch.cuda.empty_cache()
    errors = dict(logit_max=float((traces['original'][0]-traces['cached'][0]).abs().max()),
                  state_max=float((traces['original'][1]-traces['cached'][1]).abs().max()))
    report = dict(schema=1, cycle=loaded.cycle, checkpoint=str(a.body/'last.pt.gz'),
        source_sha256=loaded.source_sha, bytes=a.bytes, repeats=a.repeats,
        arms=arms, errors=errors, speedup=arms['original']['median_seconds']/arms['cached']['median_seconds'],
        optimizer_updates=0, world_advances=0,
        recurrent_unchanged=torch.equal(state, e.recurrent),
        scope='fixed real text numerical/decode runtime comparison; not overall event throughput')
    if max(errors.values()) > 1e-3 or not report['recurrent_unchanged']:
        raise RuntimeError(f'decode equivalence failed: {report}')
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))
    m._causal_archive.db.close()


if __name__ == '__main__':
    main()
