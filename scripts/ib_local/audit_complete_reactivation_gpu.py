"""Frozen real-body delivery check for the explicit complete-sequence reader.

No optimizer call, no weight training, no actual-life reset/advance. Each
research branch restores the same inherited native states and observations.
"""
import argparse
from contextlib import contextmanager
from functools import partial
import gc
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--modes', nargs='+', choices=('legacy','complete','complete_wrong_positions','disabled'),
                   default=['legacy','complete','disabled'])
    p.add_argument('--gaps', nargs='+', type=int, choices=(0,512), default=[0,512])
    p.add_argument('--continuous', action='store_true', help='same128-event stream and inherited pending state')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    for key, suffix in [('HF_HOME','huggingface'), ('TRANSFORMERS_CACHE','transformers'),
                        ('TRITON_CACHE_DIR','triton'), ('TORCHINDUCTOR_CACHE_DIR','inductor'),
                        ('TEMP','tmp'), ('TMP','tmp')]:
        os.environ[key] = 'D:/ml_cache/'+suffix
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import torch
    from scripts.ib_local import run_native_s_credit_retention as old
    from scripts.ib_local import pretrained_gdn_stream_load as loading
    from scripts.ib_local.frozen_mlp_int8 import load_streamed_int8_frozen
    from scripts.ib_local.readonly_checkpoint import load_readonly_checkpoint
    runtime = argparse.Namespace(vendor=Path('E:/cgb_gdn340_probe/vendor/flash-linear-attention-0.2.2'),
        model=Path('E:/cgb_gdn340_probe/model'), bounded_autotune=True)
    source_hash = old.sha256(args.checkpoint)
    files = [Path(__file__), root/'scripts/ib_local/pretrained_gdn_experience.py',
             root/'scripts/ib_local/pretrained_gdn_learning.py',
             root/'scripts/ib_local/continuous_frozen_case.py']
    execution = {str(f): old.sha256(f) for f in files}
    report = dict(status='loading', checkpoint=str(args.checkpoint), source_sha256=source_hash,
        execution=execution, optimizer_updates=0, actual_life_updates=0, native_S_resets=0,
        protocol=__doc__, cases=[], training_ready=False)
    started = time.monotonic()
    def save():
        report['seconds'] = time.monotonic()-started
        old.write_json(args.output/'report.json', report)
    save()
    try:
        original = loading.load_streamed_pure_gdn
        loading.load_streamed_pure_gdn = partial(load_streamed_int8_frozen,
            mapped_directory=root/'scratch/frozen_weight_cache_20261109')
        try:
            _, _, _, model, details = old.load_runtime(runtime)
        finally:
            loading.load_streamed_pure_gdn = original
        from transformers import AutoTokenizer
        from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
        from scripts.ib_local.native_s_storage import stop_future_joint_archives
        from scripts.ib_local.evaluate_pretrained_gdn_ruler import run_case
        from scripts.ib_local.analyze_ruler_generation import numeric_scores
        from scripts.ib_local.probe_continuous_readiness import native_only
        from scripts.ib_local.audit_pretrained_continued_answers import insert_interference
        from scripts.ib_local.run_pretrained_gdn_cross_window import read_text
        tokenizer = AutoTokenizer.from_pretrained(runtime.model, local_files_only=True)
        owner = load_readonly_checkpoint(args.checkpoint)
        state = owner['learner']
        learner = PretrainedExperienceLearner(model, layers=state['layers'], capacity=state['capacity'])
        rows_path = root/'results/cgb_ruler_adaptation_20261109/data_validation/512/niah_single_1/test.jsonl'
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()][:4]
        offset = old.read_json(root/'data/ib_owt_10000/tokenizer.json')['offset']
        text, _ = read_text(root/'data/ib_owt_10000/train.npy', offset, 13_000_000, length=1_500_000)
        traffic = tokenizer.encode(text, add_special_tokens=False)
        background = tokenizer.decode(traffic[-512:], skip_special_tokens=True)
        report.update(weight_loading=details, validation_sha256=old.sha256(rows_path),
                      source_updates=state['updates'], summary={})
        @contextmanager
        def mismatch_positions(enabled):
            saved = []
            try:
                if enabled:
                    for wrapper in learner.memories.values():
                        previous = wrapper.__dict__.get('retrieve')
                        original_read = wrapper.retrieve
                        def wrong(query, replay, original_read=original_read):
                            # Same old response vectors/norms, wrong input/output
                            # pairing; no label enters selection or permutation.
                            return original_read(query, lambda capsule: replay(capsule).flip(1))
                        wrapper.retrieve = wrong
                        saved.append((wrapper, previous))
                yield
            finally:
                for wrapper, previous in saved:
                    if previous is None:
                        del wrapper.retrieve
                    else:
                        wrapper.retrieve = previous
        for gap in args.gaps:
            for mode in args.modes:
                samples = []
                for row in rows:
                    if (args.output/'STOP').exists():
                        raise RuntimeError('explicit frozen diagnostic STOP')
                    learner.load_state_dict(state)
                    stop_future_joint_archives(learner)
                    if mode.startswith('complete'):
                        learner.enable_complete_reactivation()
                    versions = [p._version for p in model.parameters()]
                    report.update(status='running', current=dict(gap=gap, mode=mode, index=row['index']))
                    save()
                    case = insert_interference(row, background) if gap else row
                    with native_only(learner, mode == 'disabled'), mismatch_positions(mode == 'complete_wrong_positions'):
                        if args.continuous:
                            from scripts.ib_local.continuous_frozen_case import run_continuous_frozen_case
                            sample = run_continuous_frozen_case(learner,tokenizer,case,owner['stream'],32)
                        else:
                            sample = run_case(learner, tokenizer, case, 32, reset_native=False)
                    sample.update(numeric_scores(sample['prediction'], sample['references']))
                    sample.update(gap=gap, mode=mode)
                    if learner.updates != state['updates'] or versions != [p._version for p in model.parameters()]:
                        raise AssertionError('frozen delivery probe changed parameters')
                    report['cases'].append(sample)
                    samples.append(sample)
                    save()
                    print(json.dumps(dict(mode=mode,gap=gap,index=row['index'],
                        exact=sample['first_number_exact'],answer_nll=sample['answer_nll'])), flush=True)
                    gc.collect()
                report['summary'][f'{gap}:{mode}'] = dict(exact=sum(s['first_number_exact'] for s in samples),
                    cases=len(samples), answer_nll=sum(s['answer_nll'] for s in samples)/len(samples))
                save()
        report.update(status='complete', source_unchanged=old.sha256(args.checkpoint)==source_hash,
            execution_unchanged=all(old.sha256(f)==h for f,h in execution.items()),
            peak_cuda_allocated=torch.cuda.max_memory_allocated())
        if not report['source_unchanged'] or not report['execution_unchanged']:
            raise AssertionError('source/execution changed during frozen diagnostic')
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
