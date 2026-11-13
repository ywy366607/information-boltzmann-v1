"""Answer accuracy on existing full/held language-learning endpoints.

Frozen branch from saved complete native state, no question-start reset.
First score original heldout single-needle RULER cases. If the shared source
can answer >=2/4 shortest cases, also insert16384 real OWT tokens immediately
before the original question in those same four cases. The augmented cases
are a declared interference diagnostic, not an official RULER score.
No extra optimizer updates, model changes, learned loss, or new task curriculum.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from scripts.ib_local.analyze_ruler_generation import numeric_scores
from scripts.ib_local.evaluate_pretrained_gdn_ruler import encode_case, run_case
from scripts.ib_local.pretrained_gdn import enable_streaming_convolution, load_pure_gdn
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.prepare_pretrained_gdn_ruler import sha256
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same, read_text, write_json


def insert_interference(row, text):
    """Locate the existing question by syntax only, never by the answer value."""
    boundary = row['input'].rfind('\nWhat is ')
    if boundary < 0:
        raise ValueError('expected official question boundary')
    return dict(row, input=row['input'][:boundary]+'\n'+text+'\n'+row['input'][boundary:])


def summarize(cases):
    return dict(cases=len(cases),
        exact=sum(c['first_number_exact'] for c in cases),
        standalone=sum(c['standalone_number'] for c in cases),
        contains=sum(c['official_contains'] for c in cases),
        first_number_first_digit=sum(c['first_number_first_digit_correct'] for c in cases),
        mean_answer_nll=sum(c['answer_nll'] for c in cases)/len(cases) if cases else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('model', 'source', 'held', 'full', 'manifest', 'output'):
        parser.add_argument('--'+field, type=Path, required=True)
    parser.add_argument('--interference-tokens', type=int, default=16384)
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--cpu-load', action='store_true',
                        help='load exact weights on CPU before GPU transfer; avoids Windows GPU safetensor slice crash')
    parser.add_argument('--stream-load', action='store_true',
                        help='ordinary per-tensor binary reads; no safetensor memory mapping')
    args = parser.parse_args()
    if args.output.exists() or args.limit != 4 or args.interference_tokens != 16384:
        raise ValueError('fresh output and registered4cases/16384tokens required')
    torch.set_num_threads(1)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    manifest = json.loads(args.manifest.read_text())
    cases = []
    for item in manifest['data']:
        if item['task'] != 'niah_single_1' or item['length'] not in (512, 2048, 4096):
            continue
        assert sha256(item['path']) == item['sha256']
        rows = [json.loads(l) for l in Path(item['path']).read_text().splitlines()][:args.limit]
        cases.extend((item['length'], row) for row in rows)
    data = Path('data/ib_owt_10000')
    offset = json.loads((data/'tokenizer.json').read_text())['offset']
    text, raw_sha = read_text(data/'train.npy', offset, 8_000_000, length=262144)
    interference = tokenizer.encode(text, add_special_tokens=False)[:args.interference_tokens]
    assert len(interference) == args.interference_tokens
    text = tokenizer.decode(interference, skip_special_tokens=True)
    sources = dict(shared_source=args.source, held=args.held, full=args.full)
    hashes = {k:sha256(v) for k,v in sources.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', protocol=__doc__, sources={k:str(v) for k,v in sources.items()},
        source_hashes=hashes, manifest_sha256=sha256(args.manifest),
        official_revision=manifest['official_revision'], optimizer_updates=0,
        script_sha256=sha256(__file__), reset_native=False, generation_cap_tokens=32,
        loading_device='stream_binary_to_cuda' if args.stream_load else ('cpu_then_cuda' if args.cpu_load else 'cuda'),
        interference=dict(raw_array_start=8_000_000, raw_array_length=262144,
            raw_bytes_sha256=raw_sha, requested_tokens=len(interference),
            text_sha256=hashlib.sha256(text.encode()).hexdigest()),
        records=[], arms={}, new_lean_coverage=0)
    started = time.time()
    def write():
        report['seconds'] = time.time()-started
        write_json(args.output, report)
    write()
    print('CONTINUED ANSWERS LOAD MODEL', flush=True)
    if args.stream_load:
        from scripts.ib_local.pretrained_gdn_stream_load import load_streamed_pure_gdn
        model, audit = load_streamed_pure_gdn(args.model)
        report['weight_loading'] = audit
    else:
        model, _ = load_pure_gdn(args.model, device='cpu' if args.cpu_load else 'cuda')
        if args.cpu_load:
            model.to('cuda')
    print('CONTINUED ANSWERS MODEL LOADED', flush=True)
    enable_streaming_convolution(model)
    learner = None
    try:
        for name, path in sources.items():
            payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            state = payload.get('learner', payload)
            if learner is None:
                learner = PretrainedExperienceLearner(model, layers=state['layers'], capacity=state['capacity'])
            arm = dict(updates=state['updates'], tokens=state['tokens'],
                incoming_seen_tokens=state['cache']['seen_tokens'],
                original_cases=[], interference_cases=[])
            report['arms'][name] = arm
            def measure(length, row, mode):
                learner.load_state_dict(state)
                before_seen = learner.cache._seen_tokens
                prefix, _ = encode_case(tokenizer, row)
                sample = run_case(learner, tokenizer, row, 32, reset_native=False)
                assert learner.cache._seen_tokens == before_seen+len(prefix)
                sample.update(numeric_scores(sample['prediction'], sample['references']))
                number = sample['first_number']
                sample['first_number_first_digit_correct'] = bool(number and number[0] == row['outputs'][0][0])
                sample['generator_budget'] = length
                sample['mode'] = mode
                sample['incoming_seen_tokens'] = before_seen
                sample['outgoing_seen_tokens'] = learner.cache._seen_tokens
                report['records'].append(dict(arm=name, **sample))
                learner.load_state_dict(state)
                assert_same(state, learner.state_dict())
                print('CONTINUED ANSWER', name, mode, length, row['index'],
                      sample['first_number_exact'], repr(sample['prediction']), flush=True)
                write()
                return sample
            for length, row in cases:
                arm['original_cases'].append(measure(length, row, 'original_carried'))
            short = [s for s in arm['original_cases'] if s['generator_budget'] == 512]
            arm['original_summary'] = summarize(arm['original_cases'])
            arm['by_generator_budget'] = {str(n):summarize([s for s in arm['original_cases']
                if s['generator_budget'] == n]) for n in (512, 2048, 4096)}
            if name == 'shared_source':
                report['interference_gate'] = dict(source_short_exact=sum(s['first_number_exact'] for s in short),
                    cases=len(short), required_exact=2)
                report['interference_gate']['passed'] = report['interference_gate']['source_short_exact'] >= 2
            if report['interference_gate']['passed']:
                for length, row in cases:
                    if length == 512:
                        arm['interference_cases'].append(measure(length, insert_interference(row, text),
                                                                'real_owt_interference'))
                arm['interference_summary'] = summarize(arm['interference_cases'])
            else:
                arm['interference_skipped'] = 'shared source cannot answer2/4 short cases in carried context'
            arm['complete_state_restored'] = True
            write()
            del payload, state
        assert all(sha256(v) == hashes[k] for k,v in sources.items())
        assert sha256(__file__) == report['script_sha256']
        report.update(status='complete', sources_unchanged=True,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated())
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        write()


if __name__ == '__main__':
    main()
