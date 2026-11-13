"""Same-body near-distance paired curriculum, before any new long-credit run.

Uses the existing complete individual and standard answer NLL, same128-byte
bound and production optimizer. Pairs differ only in the observed number.
Final windows are answer-anchored. Top1/oracle reads are frozen diagnostics.
"""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.analyze_ruler_generation import numeric_scores
from scripts.ib_local.audit_small_native_failure import provenance
from scripts.ib_local.navigation_checkpoint import save
from scripts.ib_local.run_causal_water_reply import build
from scripts.ib_local.small_gdn_counterfactual import answer_windows, frozen_read, near_pair
from scripts.ib_local.small_gdn_ruler import (
    advance, consume_prompt, encode_case, probability_tensor, request_branch, run_case, sha256,
)
from scripts.ib_local.table_branch import table_branch
from scripts.ib_local.train_small_native_credit import (
    config_of, continuation_digest, old_gradients, read_manifest,
)


def question_step(engine, row, diagnose=False):
    bank = engine.model.local_replay
    prefix, gold = encode_case(row)
    raw, windows = answer_windows(prefix, gold, engine.pending)
    reports = []
    for start, end, score in windows:
        part = raw[start:end+1]
        _, probability, _ = engine.table.run_credit(part, max_window=128)
        ids, targets, logp = engine._inputs(part, probability)
        mask = torch.tensor(score, device=engine.device)[None]
        with bank.observation(engine.optimizer.param_groups[-1], end-start):
            if not bool(mask.any()):
                with torch.no_grad():
                    _, state, _, features, read = engine.model.predict_observed(ids, engine.recurrent, logp)
                    engine.model.commit_observed(features, read)
                    engine.recurrent = state.detach()
                report = dict(observed=end-start, supervised=0)
            else:
                loss, state, features, diag = engine.model.objective(
                    ids, targets, engine.recurrent, logp,
                    torch.randn(64, 32, device=engine.device), score_mask=mask)
                report = dict(observed=end-start, supervised=int(mask.sum()), ce=float(diag['ce']),
                              layers=bank.diagnostics())
                if diagnose:
                    ce = torch.nn.functional.cross_entropy(diag['logits'][mask], targets[mask])
                    report['old_gradients'] = old_gradients(engine, ce)
                report['gradient_norm'] = engine._advance(loss, features, diag, state)
            bank.settle_and_commit()
        engine.pending = int(raw[end])
        reports.append(report)
    assert sum(bool(w['supervised']) for w in reports) == 1
    return dict(source_index=row['source_index'], pair_variant=row['pair_variant'],
                observed=len(raw)-1, windows=reports)


@torch.no_grad()
def score_case(engine, row, alternate, generation_bytes=16):
    prefix, gold = encode_case(row)
    with request_branch(engine) as table:
        first, state, pending = consume_prompt(engine.model, table, prefix, 128)
        with table_branch(table) as teacher:
            current, local, last = first, state, pending
            nll, margin = [], None
            for i, byte in enumerate(gold):
                logp = current.float().log_softmax(-1)
                nll.append(float(-logp[byte]))
                if i == 1:
                    margin = float(logp[byte]-logp[ord(alternate[0])])
                if i+1 < len(gold):
                    current, local, last = advance(engine.model, teacher, local, last, byte)
        current, generated = first, []
        for i in range(generation_bytes):
            byte = int(current.argmax())
            generated.append(byte)
            if i+1 < generation_bytes:
                current, state, pending = advance(engine.model, table, state, pending, byte)
    prediction = bytes(generated).decode('utf-8', errors='replace')
    return dict(index=row['index'], source_index=row['source_index'],
        pair_variant=row['pair_variant'], references=row['outputs'], prediction=prediction,
        prompt_bytes=len(prefix), answer_distance_bytes=len(prefix)-prefix.index(row['outputs'][0].encode()),
        answer_nll=float(np.mean(nll)), first_digit_nll=nll[1],
        first_digit_correct_vs_paired_margin=margin, answer_byte_nll=nll,
        **numeric_scores(prediction, row['outputs']))


@torch.no_grad()
def evaluate(engine, pairs):
    before = continuation_digest(engine)
    cases, matched = [], 0
    for left, right in pairs:
        a = score_case(engine, left, right['outputs'][0])
        b = score_case(engine, right, left['outputs'][0])
        cases.extend((a, b))
        matched += int(a['first_number_exact'] and b['first_number_exact'])
    assert continuation_digest(engine) == before
    return dict(cases=cases, exact=sum(c['first_number_exact'] for c in cases), count=len(cases),
        both_correct_pairs=matched, pairs=len(pairs),
        answer_nll=float(np.mean([c['answer_nll'] for c in cases])),
        first_digit_nll=float(np.mean([c['first_digit_nll'] for c in cases])),
        positive_content_margins=sum(c['first_digit_correct_vs_paired_margin'] > 0 for c in cases),
        mean_content_margin=float(np.mean([c['first_digit_correct_vs_paired_margin'] for c in cases])),
        complete_state_unchanged=True)


@torch.no_grad()
def routing_probe(engine, row, alternate):
    """Identical factual prompt/S/table; override reads at the first answer digit."""
    bank, model = engine.model.local_replay, engine.model
    prefix, _ = encode_case(row)
    needle = prefix.index(row['outputs'][0].encode())
    with request_branch(engine) as table, provenance(bank) as metadata:
        starts = [int(m.memory.clock) for m in bank.memories]
        _, state, pending = consume_prompt(model, table, prefix, 128)
        # Settle the real leading answer space once, not the answer digit.
        table.run_credit(np.array([pending, 32], dtype=np.uint8), max_window=1)
        from scripts.ib_local.language_action import peek
        logp = probability_tensor(model, peek(table, 32)[None])
        ids = torch.tensor([[32]], device=engine.device)
        results = []
        conditions = [('ordinary', None), ('top1', None), ('first', None), ('wrong', None)]
        conditions += [('first', i) for i in range(len(bank.memories))]
        for mode, layer in conditions:
            scope = (nullcontext([]) if mode == 'ordinary' else
                     frozen_read(bank, mode, metadata, starts, needle, layer))
            with scope as traces:
                logits = model.predict(ids, state, logp)[0][0, 0].float()
                lp = logits.log_softmax(-1)
                a, b = ord(row['outputs'][0][0]), ord(alternate[0])
                results.append(dict(mode=mode, layer=None if layer is None else layer+1,
                    first_digit_nll=float(-lp[a]), correct_vs_alternate_margin=float(lp[a]-lp[b]),
                    predicted_byte=int(logits.argmax()), traces=traces))
        # Ordinary repeat detects hidden probe writes/order contamination.
        repeat = model.predict(ids, state, logp)[0][0, 0].float().log_softmax(-1)
        assert float(-repeat[ord(row['outputs'][0][0])]) == results[0]['first_digit_nll']
    return dict(source_index=row['source_index'], variant=row['pair_variant'], results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=512)
    args = parser.parse_args()
    if not 8 <= args.pairs <= 1024:
        parser.error('8..1024 pairs; validation starts at source index3584')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    gdn_fused.enable()
    rows, _ = read_manifest(args.manifest)
    training = [near_pair(r) for r in rows[:args.pairs]]
    validation = [near_pair(r) for r in rows[3584:3600]]
    fresh = [near_pair(r) for r in rows[3840:3872]]
    split_answers = [{r['outputs'][0] for pair in part for r in pair}
                     for part in (training, validation, fresh)]
    assert not (split_answers[0] & split_answers[1] or split_answers[0] & split_answers[2]
                or split_answers[1] & split_answers[2])
    assert len(validation) == 16 and len(fresh) == 32
    for name, data in [('train', training), ('validation', validation), ('fresh', fresh)]:
        (args.output/(name+'.jsonl')).write_text(
            ''.join(json.dumps(row)+'\n' for pair in data for row in pair), encoding='utf-8')
    gaps = [len(encode_case(r)[0])-encode_case(r)[0].index(r['outputs'][0].encode())
            for part in (training, validation, fresh) for pair in part for r in pair]
    report = dict(status='running', protocol=__doc__, checkpoint=str(args.checkpoint),
        source_sha256=sha256(args.checkpoint), manifest_sha256=sha256(args.manifest),
        bptt_bytes=128, answer_anchored=True, answer_gap_min=min(gaps), answer_gap_max=max(gaps),
        training_pairs=args.pairs, optimizer_updates_budget=2*args.pairs,
        auxiliary_loss_unchanged=True, additional_contrastive_loss=False,
        task_variant='official_short_layout_near_named_query_v1', standard_ruler_score=False,
        new_lean_coverage=0, evaluations=[], steps=[], admission={}, routing_probes=[])
    source_files = {Path(m.__file__) for name, m in sys.modules.copy().items()
                    if name.startswith('scripts.ib_local.') and getattr(m, '__file__', None)} | {Path(__file__)}
    report['source_files'] = {str(f): sha256(f) for f in sorted(source_files)}
    sources = args.output/'executed_sources'
    sources.mkdir()
    import shutil
    for f in source_files:
        shutil.copy2(f, sources/f.name)
    destination = args.output/'report.json'
    def write():
        report['seconds'] = time.time()-started
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temporary.replace(destination)
    started = time.time()
    write()
    try:
        cfg = config_of(args.checkpoint)
        assert cfg['small_native_replay']['schema'] == 5
        engine, _, _, source = build(args.checkpoint)
        engine.model.local_replay.set_held(False)
        # Existing source is the neutral complete migration, not a failed task endpoint.
        assert all(float(m.inlet.projection.gain) == 0 and
                   float(m.inlet.projection.output_weight.norm()) == 0
                   for m in engine.model.local_replay.memories)
        for r in [r for pair in training[:4] for r in pair]:
            report['admission'].setdefault('discarded_steps', []).append(question_step(engine, r))
        row = validation[0][0]
        a = score_case(engine, row, validation[0][1]['outputs'][0])
        reference = run_case(engine, row, 16)
        assert a['prediction'] == reference['prediction'] and a['answer_byte_nll'] == reference['answer_byte_nll']
        report['admission']['evaluator_matches_existing'] = True
        resume = args.output/'admission_resume'
        resume.mkdir()
        save(engine, resume, source['cycle'], cfg)
        first = question_step(engine, training[4][0], diagnose=True)
        digest = continuation_digest(engine)
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        engine, _, _, _ = build(resume/'last.pt.gz')
        second = question_step(engine, training[4][0], diagnose=True)
        assert first == second and continuation_digest(engine) == digest
        report['admission']['disk_next_update_exact'] = True
        report['admission']['passed'] = True
        print('ADMISSION PASSED', time.time()-started, flush=True)
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        engine, _, _, _ = build(args.checkpoint)
        engine.model.local_replay.set_held(False)
        report['evaluations'].append(dict(update=0, validation=evaluate(engine, validation)))
        write()
        ordering = np.random.default_rng(71171)
        update = 0
        for pair in training:
            for variant in ordering.permutation(2):
                if (args.output/'STOP').exists():
                    raise InterruptedError('user STOP requested')
                report['steps'].append(question_step(engine, pair[variant], diagnose=update == 0))
                update += 1
            if update in (64, 128, 256, 512, 1024, 2048) or update == 2*args.pairs:
                evaluation = dict(update=update, validation=evaluate(engine, validation))
                report['evaluations'].append(evaluation)
                print('NEAR', update, 'exact', evaluation['validation']['exact'],
                      'pairs', evaluation['validation']['both_correct_pairs'],
                      'nll', evaluation['validation']['answer_nll'], flush=True)
                folder = args.output/'candidate'
                folder.mkdir(exist_ok=True)
                cfg['small_counterfactual_run'] = dict(updates=update, bptt_bytes=128,
                    task_variant=report['task_variant'], answer_anchored=True)
                save(engine, folder, source['cycle'], cfg)
                report['candidate_checkpoint'] = str(folder/'last.pt.gz')
            write()
        report['fresh'] = evaluate(engine, fresh)
        before = continuation_digest(engine)
        for pair in validation[:4]:
            for i, row in enumerate(pair):
                report['routing_probes'].append(routing_probe(engine, row, pair[1-i]['outputs'][0]))
                assert continuation_digest(engine) == before
        report['routing_complete_state_unchanged'] = True
        last = report['evaluations'][-1]['validation']
        fresh_score = report['fresh']
        report['mastery_gate'] = dict(passed=(
            last['both_correct_pairs'] >= 15 and fresh_score['both_correct_pairs'] >= 30),
            requirement='>=15/16 fixed and >=30/32 fresh pairs BOTH exactly correct',
            long_credit_pair_started=False,
            next_action='fork mastered common checkpoint only if gate passes; otherwise diagnose near retrieval')
        report.update(status='complete', source_unchanged=sha256(args.checkpoint)==report['source_sha256'],
            executed_sources_unchanged=all(sha256(p)==h for p,h in report['source_files'].items()),
            observed_bytes=sum(s['observed'] for s in report['steps']),
            supervised_bytes=sum(w['supervised'] for s in report['steps'] for w in s['windows']),
            peak_cuda_bytes=torch.cuda.max_memory_allocated())
        assert report['source_unchanged'] and report['executed_sources_unchanged']
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        write()


if __name__ == '__main__':
    main()
