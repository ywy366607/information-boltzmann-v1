"""Frozen current-source RULER competence and matched interference diagnostic.

Each disposable researcher branch inherits the full existing S, convolution and
experience state. No cache is zeroed, no optimizer step occurs, and no branch
replaces the actual life. This is a diagnostic, not an irreversible-life score.
"""
import argparse
from functools import partial
import gc
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for key, suffix in [('HF_HOME', '/huggingface'), ('TRANSFORMERS_CACHE', '/transformers'),
                        ('TRITON_CACHE_DIR', '/triton'), ('TORCHINDUCTOR_CACHE_DIR', '/inductor'),
                        ('TEMP', '/tmp'), ('TMP', '/tmp')]:
        os.environ[key] = 'D:/ml_cache' + suffix
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import torch
    from transformers import AutoTokenizer
    from scripts.ib_local import run_native_s_credit_retention as old
    from scripts.ib_local import pretrained_gdn_stream_load as loading
    from scripts.ib_local.frozen_mlp_int8 import load_streamed_int8_frozen
    from scripts.ib_local.readonly_checkpoint import load_readonly_checkpoint
    from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
    from scripts.ib_local.native_s_storage import stop_future_joint_archives
    from scripts.ib_local.evaluate_pretrained_gdn_ruler import run_case, encode_case
    from scripts.ib_local.analyze_ruler_generation import numeric_scores
    from scripts.ib_local.audit_pretrained_continued_answers import insert_interference, summarize
    from scripts.ib_local.run_pretrained_gdn_cross_window import read_text

    runtime = argparse.Namespace(source=ROOT/'scratch/joint_next_pending_20261108/pending_recovery.pt',
        runtime_manifest=ROOT/'scratch/bounded_causal_tags_20261108/operator_source_manifest.json',
        model=Path('E:/cgb_gdn340_probe/model'),
        vendor=Path('E:/cgb_gdn340_probe/vendor/flash-linear-attention-0.2.2'),
        bounded_autotune=True, readonly_checkpoints=True)
    manifest_path = Path('E:/cgb_gdn340_probe/ruler_20261107/manifest.json')
    manifest = old.read_json(manifest_path)
    cell = next(c for c in manifest['data'] if c['task']=='niah_single_1' and c['length']==512)
    if old.sha256(cell['path']) != cell['sha256']:
        raise ValueError('registered official RULER data changed')
    rows = [json.loads(line) for line in Path(cell['path']).read_text().splitlines()][:8]
    if len(rows) != 8:
        raise ValueError('eight preregistered cases required')
    report = dict(status='loading', protocol=__doc__, **old.provenance(runtime),
        script_sha256=old.sha256(__file__), manifest_sha256=old.sha256(manifest_path),
        task_sha256=cell['sha256'], official_revision=manifest['official_revision'],
        seed=manifest['seed'], optimizer_updates=0, main_life_updates=0, native_S_resets=0,
        precision='same row_int8_frozen/mapped_cpu as stopped SQuAD',
        max_new_tokens=32, cases=[], summary={},
        short_gate=dict(required=6, cases=8, metric='first_number_exact'),
        registered_interference_tokens=[512, 2048, 8192],
        interference_cases='first four in fixed file order; no success-based filtering',
        pending_teaching='source pending inputs remain archived; no training stream is consumed',
        new_lean_coverage=0, behavioral_credit_benefit_proven=False)
    started = time.monotonic()
    def save():
        report['seconds'] = time.monotonic()-started
        old.write_json(args.output/'report.json', report)
    save()
    try:
        tokenizer = AutoTokenizer.from_pretrained(runtime.model, local_files_only=True)
        original_loader = loading.load_streamed_pure_gdn
        loading.load_streamed_pure_gdn = partial(load_streamed_int8_frozen,
            mapped_directory=ROOT/'scratch/frozen_weight_cache_20261109')
        try:
            _, _, _, model, weight_loading = old.load_runtime(runtime)
        finally:
            loading.load_streamed_pure_gdn = original_loader
        report['weight_loading'] = weight_loading
        owner = load_readonly_checkpoint(runtime.source)
        state = owner['learner']
        learner = PretrainedExperienceLearner(model, layers=state['layers'], capacity=state['capacity'])
        report.update(source_updates=state['updates'], source_seen=state['cache']['seen_tokens'])
        save()

        def measure(row, gap):
            if (args.output/'STOP').exists():
                raise RuntimeError('explicit researcher diagnostic STOP')
            learner.load_state_dict(state)
            stop_future_joint_archives(learner)
            before_seen, before_updates = learner.cache._seen_tokens, learner.updates
            versions = [p._version for p in model.parameters()]
            report.update(status='running', current=dict(index=row['index'], extra_gap_tokens=gap))
            save()
            sample = run_case(learner, tokenizer, row, 32, reset_native=False)
            sample.update(numeric_scores(sample['prediction'], sample['references']))
            number = sample['first_number']
            sample['first_number_first_digit_correct'] = bool(
                number and number[0] == row['outputs'][0][0])
            if learner.updates != before_updates or versions != [p._version for p in model.parameters()]:
                raise AssertionError('frozen diagnostic changed slow parameters')
            if learner.cache._seen_tokens != before_seen+sample['prompt_tokens']:
                raise AssertionError('prompt dropped or native state reset')
            sample.update(extra_gap_tokens=gap, incoming_seen=before_seen,
                          outgoing_seen=learner.cache._seen_tokens, optimizer_updates=0)
            report['cases'].append(sample)
            report['summary'][str(gap)] = summarize(
                [c for c in report['cases'] if c['extra_gap_tokens']==gap])
            save()
            print(json.dumps(dict(index=row['index'], gap=gap, exact=sample['first_number_exact'],
                                  prediction=sample['prediction'], references=sample['references'])), flush=True)
            gc.collect()
            return sample

        for row in rows:
            measure(row, 0)
        report['short_gate']['observed_exact'] = report['summary']['0']['exact']
        report['short_gate']['passed'] = report['summary']['0']['exact'] >= 6
        save()
        if report['short_gate']['passed']:
            data = ROOT/'data/ib_owt_10000'
            offset = old.read_json(data/'tokenizer.json')['offset']
            text, raw_sha = read_text(data/'train.npy', offset, 12_000_000, length=196608)
            ids = tokenizer.encode(text, add_special_tokens=False)
            report['interference_source'] = dict(raw_start=12_000_000, raw_length=196608, sha256=raw_sha)
            if len(ids) < 8192:
                raise ValueError('real interference text too short')
            for gap in (512, 2048, 8192):
                text = tokenizer.decode(ids[:gap], skip_special_tokens=True)
                for row in rows[:4]:
                    revised = insert_interference(row, text)
                    sample = measure(revised, gap)
                    original_len = len(encode_case(tokenizer, row)[0])
                    sample['actual_added_prompt_tokens'] = sample['prompt_tokens']-original_len
                    save()
        else:
            report['interference_skipped'] = 'current source failed preregistered short-distance competence'
        old.verify_preserved(runtime, report)
        report['script_unchanged'] = old.sha256(__file__)==report['script_sha256']
        report.update(status='complete', peak_cuda_allocated=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
