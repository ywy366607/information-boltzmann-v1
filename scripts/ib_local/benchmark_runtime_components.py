"""Interleaved component timing on one actual restored body, no learning."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation


def draw_handover(logit):
    """Rejected packed-readback candidate, retained only for reproduction."""
    ended = torch.rand((), device=logit.device) < logit.sigmoid()
    value, flag = torch.stack((logit, ended.to(logit.dtype))).cpu().tolist()
    return value, bool(flag)


def draw_byte(logits):
    logp = logits.log_softmax(-1)
    chosen = torch.multinomial(logp.exp(), 1)
    byte, value = torch.cat((chosen.to(logp.dtype), logp[chosen])).cpu().tolist()
    return int(byte), value


def measured(call, repetitions):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repetitions):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter()-start)/repetitions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--body', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    args = continuation_parser().parse_args([
        '--checkpoint', str(a.body/'last.pt.gz'), '--output', str(a.body),
        '--cycles', '1', '--resume-output', '--world-source', 'text',
        '--table-storage', str(a.output/'table'), '--causal-storage', str(a.output/'causal')])
    loaded = prepare_continuation(args, resume_output=True)
    e, m = loaded.engine, loaded.engine.model
    from scripts.ib_local.causal_relation_program import mechanism
    from scripts.ib_local.quadratic_revision import evidence_correction
    from scripts.ib_local.language_action import peek
    from scripts.ib_local.scoped_decision_graph import ScopedDecisionGraph
    from scripts.ib_local.mother_checkpoint import save
    from scripts.ib_local.run_causal_water_reply import sha256_file
    reports = {}
    state_before = e.recurrent.clone()
    with torch.no_grad():
        # Use the largest actual active support, rather than accidentally
        # timing only the first one-fact relation in the bank.
        support = m.causal_relations.valid.sum(-1).masked_fill(m.causal_relations.identity == 0, -1)
        i = int(support.argmax())
        slot = int(torch.nonzero(m.memory.occupied)[0])
        distribution = mechanism(m, m.causal_relations.state(i),
            m.memory.source[slot:slot+1], m.memory.action[slot:slot+1])
        x, y, q = distribution['x'], distribution['y'], distribution['query']
        latent = (torch.einsum('nf,hfd->hnd', x,
            m.causal_relations.directions[:, :x.shape[1]])+
            m.causal_relations.observation_directions[:, :len(x)])
        def solve(reuse):
            result = evidence_correction(x, y, q, .1, return_factor=reuse)
            if reuse:
                factor = result[-1]
                return torch.linalg.lu_solve(factor[0][None], factor[1][None], latent)
            return torch.linalg.solve(result[0][None], latent)
        for mode in (False, True):
            for _ in range(5): solve(mode)
        timings = {'before': [], 'after': []}
        for mode in (False, True, True, False, False, True):
            timings['after' if mode else 'before'].append(measured(lambda: solve(mode), 100))
        reports['factor_seconds'] = dict(timings, evidence_rows=len(x))

        probability = np.asarray(peek(e.table, e.pending))
        cursor = int(e.world.cursor)
        words = np.asarray(e.world.data[cursor:cursor+128], dtype=np.uint8).tolist()
        graphs = [ScopedDecisionGraph(m, capture=True, pack_inputs=pack) for pack in (False, True)]
        def decode(graph):
            graph.start(e.recurrent)
            for byte in words:
                graph.step(byte, probability)
        for graph in graphs: decode(graph)
        timings = {'before': [], 'after': []}
        for mode in (0, 1, 1, 0, 0, 1):
            timings['after' if mode else 'before'].append(measured(lambda: decode(graphs[mode]), 1))
        reports['decode_128_seconds'] = timings
        logits = graphs[0].logits[0, 0]
        stop = torch.tensor(-.5, device=e.device)
        def sample(packed):
            if packed:
                _, ended = draw_handover(stop)
                if not ended: draw_byte(logits)
            else:
                ended = bool(torch.rand((), device=e.device) < stop.sigmoid())
                float(stop)
                if not ended:
                    logp = logits.log_softmax(-1)
                    chosen = int(torch.multinomial(logp.exp(), 1))
                    float(logp[chosen])
        timings = {'before': [], 'after': []}
        with torch.random.fork_rng(devices=[0]):
            for mode in (False, True, True, False, False, True):
                torch.cuda.manual_seed(1611)
                timings['after' if mode else 'before'].append(measured(lambda: sample(mode), 200))
        reports['sample_seconds'] = timings

        directory = a.output/'serialization'
        directory.mkdir(exist_ok=True)
        timings = {'before': [], 'after': []}
        for mode in (False, True, True, False, True, False, False, True):
            seconds = measured(lambda: save(e, directory, loaded.cycle,
                loaded.config.copy(), streamed_hash=mode), 1)
            timings['after' if mode else 'before'].append(seconds)
        reports['checkpoint_seconds'] = timings
        metadata = json.loads((directory/'last.metadata.json').read_text())
        reports['checkpoint_hash_valid'] = metadata['checkpoint_sha256'] == sha256_file(directory/'last.pt.gz')
    for group in ('factor_seconds', 'decode_128_seconds', 'sample_seconds', 'checkpoint_seconds'):
        value = reports[group]
        value['before_median'] = float(np.median(value['before']))
        value['after_median'] = float(np.median(value['after']))
        value['speedup'] = value['before_median']/value['after_median']
    report = dict(schema=1, cycle=loaded.cycle, source_sha256=loaded.source_sha, components=reports,
        optimizer_updates=0, main_checkpoint_written=False,
        recurrent_unchanged=torch.equal(e.recurrent, state_before))
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    m._causal_archive.db.close()


if __name__ == '__main__':
    main()
