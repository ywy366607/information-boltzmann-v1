"""Interleaved v2/v3 execution on an actual body; zero optimizer/world steps."""
import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from scripts.ib_local import generative_relations as relations
from scripts.ib_local import preview_runtime
from scripts.ib_local.causal_relation_program import imagine
from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation


def old_scores(model, state, x, y, raw, ridge, noise):
    encoded = model.encoder(raw)
    centered = encoded-encoded.mean(0)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    order = (centered@vh[0]).argsort()
    return (relations.marginal(x, y, ridge, noise_prior=noise),
            relations.partition_evidence(x, y, order, ridge, noise_prior=noise), vh, order)


def old_heldout(x, y, indices, ridge, *, noise_prior=None):
    scores = []
    for i in indices:
        train = [j for j in range(len(x)) if j != i]
        scores.append(relations.log_score(relations.posterior(
            x[train], y[train], x[i:i+1], ridge, noise_prior=noise_prior), y[i:i+1]).sum())
    return torch.stack(scores) if scores else y.new_empty(0)


@contextmanager
def reference():
    with patch.object(preview_runtime, 'preview_scope', lambda model: nullcontext()), \
         patch.object(relations, 'organization_scores', old_scores), \
         patch.object(relations, 'leave_one_out_scores', old_heldout):
        yield


def measured(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    return time.perf_counter()-start, value


def summarize(times):
    result = dict(before=times[False], after=times[True])
    result['before_median'] = float(np.median(times[False]))
    result['after_median'] = float(np.median(times[True]))
    result['saved_seconds'] = result['before_median']-result['after_median']
    return result


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
    reports = {}
    carried = e.recurrent.clone()
    rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    with torch.no_grad():
        bank = m.causal_relations
        counts = bank.valid.sum(-1).masked_fill(bank.identity == 0, -1)
        state = bank.state(int(counts.argmax()))
        slots = torch.nonzero(m.memory.occupied).flatten()[:4]
        source = m.memory.source[slots[:1]].expand(len(slots), -1)
        actions = m.memory.action[slots]
        def preview(new):
            with nullcontext() if new else reference():
                return imagine(m, state, source, actions, ridge=e.relation_ridge,
                               reward_valid=e.reward_label_valid(wood=True))
        for new in (False, True): preview(new)
        timings, values = {False: [], True: []}, {}
        for new in (False, True, True, False, False, True):
            seconds, value = measured(lambda: preview(new))
            timings[new].append(seconds)
            values[new] = value
        errors = {name:float((values[False][name]-values[True][name]).abs().max()) for name in
                  ('expected_after_value', 'future_delta', 'information', 'shadow_state_shift')}
        if max(errors.values()) > 1e-4:
            raise RuntimeError(f'preview mismatch: {errors}')
        reports['four_action_imagination'] = dict(summarize(timings), errors=errors,
                                                 source_facts=int(counts.max()))

        x, y, _, _ = relations.dataset(m, state)
        indices = list(range(len(x)))
        noise = m.relation_basis.noise(mother=True)
        timings = {False: [], True: []}
        for new in (False, True, True, False, False, True):
            def loo():
                for _ in range(20):
                    (relations.leave_one_out_scores if new else old_heldout)(
                        x, y, indices, e.relation_ridge, noise_prior=noise)
            seconds, _ = measured(loo)
            timings[new].append(seconds/20)
        reports['leave_one_out'] = summarize(timings)

        archive = m._causal_archive
        timings = {False: [], True: []}
        for new in (False, True, True, False, False, True):
            seconds, _ = measured(lambda: archive.sync(bulk=new))
            timings[new].append(seconds)
        reports['archive_sync'] = summarize(timings)

        from scripts.ib_local.scoped_decision_graph import ScopedDecisionGraph
        from scripts.ib_local.variable_expression import handover_logits
        from scripts.ib_local.language_action import peek
        probability = np.asarray(peek(e.table, e.pending))
        cursor = int(e.world.cursor)
        words = np.asarray(e.world.data[cursor:cursor+128], dtype=np.uint8).tolist()
        graphs = {new: ScopedDecisionGraph(m, True, capture_handover=new) for new in (False, True)}
        def decode(new):
            graph = graphs[new]
            graph.start(carried)
            outputs = []
            for i, byte in enumerate(words):
                m._expression_remaining = 1-i/128
                _, hidden = graph.step(byte, probability)
                stop = graph.handover_logit if new else handover_logits(m, hidden, 1-i/128)
                outputs.append(float(stop))
            return np.array(outputs)
        for new in (False, True): decode(new)
        timings, values = {False: [], True: []}, {}
        for new in (False, True, True, False, False, True):
            seconds, value = measured(lambda: decode(new))
            timings[new].append(seconds)
            values[new] = value
        error = float(np.max(np.abs(values[False]-values[True])))
        if error > 1e-5: raise RuntimeError(f'stop head mismatch: {error}')
        reports['decode_and_stop_128'] = dict(summarize(timings), stop_error=error)
    result = dict(schema=1, cycle=loaded.cycle, source_sha256=loaded.source_sha,
                  components=reports, optimizer_updates=0, world_steps=0,
                  carried_state_unchanged=torch.equal(carried, e.recurrent),
                  torch_rng_unchanged=torch.equal(rng[0], torch.get_rng_state()),
                  cuda_rng_unchanged=torch.equal(rng[1], torch.cuda.get_rng_state()))
    (a.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    archive.close()


if __name__ == '__main__':
    main()
