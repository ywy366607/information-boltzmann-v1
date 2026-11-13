"""Frozen location-of-memory audit: no standard attention, recall vs native S.

Same existing491-update pretrained branch and existing12 official single-needle
cases. Disable extra experience hooks, then erase only native recurrent states
before the last prompt token (retain short-convolution buffers). A matched
native control uses exactly the same split. No optimizer updates or new rules.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from scripts.ib_local.analyze_ruler_generation import numeric_scores
from scripts.ib_local.evaluate_pretrained_gdn_ruler import encode_case, run_case
from scripts.ib_local.pretrained_gdn import (
    cache_nbytes, clone_cache, enable_streaming_convolution, load_pure_gdn,
)
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.prepare_pretrained_gdn_ruler import sha256
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same, write_json
from scripts.ib_local.small_gdn_counterfactual import counterpart


class NativeOnly:
    """Call the very same model weights without any experience-layer hooks."""
    def __init__(self, model, split_at, *, erase=False):
        self.model, self.memories = model, {}
        self.split_at, self.erase = split_at, erase
        self.position, self.cache = 0, None
        self.erase_audit = None

    @torch.no_grad()
    def observe(self, x, *, return_logits=False):
        # Both controls split the same observation at the final prompt token.
        count = x.shape[1]
        boundary = self.split_at-self.position
        pieces = [x[:, :boundary], x[:, boundary:]] if 0 < boundary < count else [x]
        logits = []
        for piece in pieces:
            if self.position == self.split_at:
                original = self.cache
                self.cache = clone_cache(original)
                states = self.cache.states
                assert len(states) == self.model.config.num_hidden_layers
                self.erase_audit = dict(layers=len(states), erased=self.erase,
                    state_keys=[list(s) for s in states], cache_bytes=cache_nbytes(self.cache),
                    recurrent_shapes=[list(s['recurrent_state'].shape) for s in states],
                    before_recurrent_norms=[float(s['recurrent_state'].float().norm()) for s in states])
                for i, state in enumerate(states):
                    if self.erase:
                        state['recurrent_state'].zero_()
                    # Only S is cut: fixed convolution buffers and source cache
                    # are untouched, and the ordinary arm changes no tensor.
                    assert_same(original.states[i]['conv_state'], state['conv_state'])
                    assert float(original.states[i]['recurrent_state'].float().norm()) > 0
                    if not self.erase:
                        assert_same(original.states[i]['recurrent_state'], state['recurrent_state'])
                self.erase_audit['after_recurrent_norms'] = [
                    float(s['recurrent_state'].float().norm()) for s in states]
            output = self.model(piece, past_key_values=self.cache, use_cache=True)
            self.cache = clone_cache(output.past_key_values)
            logits.append(output.logits)
            self.position += piece.shape[1]
        return ({}, torch.cat(logits, dim=1)) if return_logits else {}

    @torch.no_grad()
    def predict(self, x, cache=None):
        output = self.model(x, past_key_values=clone_cache(self.cache if cache is None else cache),
                            use_cache=True)
        return output.logits, output.past_key_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'checkpoint', 'manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--limit', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    manifest = json.loads(args.manifest.read_text())
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    state = payload.get('learner', payload)
    checkpoint_sha = sha256(args.checkpoint)
    print('LOCATION LOAD MODEL', flush=True)
    model, _ = load_pure_gdn(args.model)
    print('LOCATION MODEL LOADED', flush=True)
    enable_streaming_convolution(model)
    assert model.config.attn is None
    types = [type(b.attn).__module__+'.'+type(b.attn).__name__ for b in model.model.layers]
    assert all(t.endswith('.GatedDeltaNet') for t in types)
    learner = PretrainedExperienceLearner(model, layers=state['layers'], capacity=state['capacity'])
    print('LOCATION WRAPPER CREATED', flush=True)
    learner.load_state_dict(state)
    print('LOCATION COMPLETE SOURCE RESTORED', flush=True)
    report = dict(status='running', protocol=__doc__, source_sha256=checkpoint_sha,
        source_updates=state['updates'], model_attention_config=model.config.attn,
        actual_layer_types=types, standard_softmax_attention_layers=0,
        lexical_fast_table_present=False, extra_experience_layers=[i+1 for i in learner.layers],
        persistent_optimizer_updates=0, generation_cap=32, records=[], new_lean_coverage=0,
        source_script_sha256=sha256(__file__), manifest_sha256=sha256(args.manifest))
    started = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def write():
        report['seconds'] = time.time()-started
        write_json(args.output, report)
    write()
    try:
        for data in manifest['data']:
            if data['task'] != 'niah_single_1' or data['length'] not in (512, 2048, 4096):
                continue
            assert sha256(data['path']) == data['sha256']
            rows = [json.loads(l) for l in Path(data['path']).read_text().splitlines()][:args.limit]
            for row in rows:
                record = dict(length=data['length'], index=row['index'], references=row['outputs'], arms={})
                report['records'].append(record)
                learner.load_state_dict(state)
                record['arms']['complete'] = run_case(learner, tokenizer, row, 32)
                learner.load_state_dict(state)
                for arm in ('native_only', 'erase_native_S', 'native_counterfactual'):
                    view = row
                    if arm == 'native_counterfactual':
                        number = row['outputs'][0]
                        alternative = counterpart(number)
                        assert row['input'].count(number) == 1
                        view = dict(row, input=row['input'].replace(number, alternative, 1), outputs=[alternative])
                    prefix, _ = encode_case(tokenizer, view)
                    native = NativeOnly(model, len(prefix)-1, erase=arm == 'erase_native_S')
                    result = run_case(native, tokenizer, view, 32)
                    assert native.erase_audit is not None
                    record['arms'][arm] = dict(**result, cache_intervention=native.erase_audit)
                for result in record['arms'].values():
                    result.update(numeric_scores(result['prediction'], result['references']))
                # Reader metadata, optimizer and source body all restored.
                assert_same(state, learner.state_dict())
                print('LOCATION', data['length'], row['index'],
                      {k:(v['first_number_exact'],v['first_number']) for k,v in record['arms'].items()}, flush=True)
                write()
        report['summary'] = {}
        for length in (512, 2048, 4096):
            records = [r for r in report['records'] if r['length'] == length]
            report['summary'][str(length)] = {arm:dict(
                cases=len(records), exact=sum(r['arms'][arm]['first_number_exact'] for r in records),
                mean_answer_nll=sum(r['arms'][arm]['answer_nll'] for r in records)/len(records))
                for arm in ('complete', 'native_only', 'erase_native_S', 'native_counterfactual')}
        learner.load_state_dict(state)
        assert_same(state, learner.state_dict())
        report.update(status='complete', complete_state_restored=True,
            source_unchanged=sha256(args.checkpoint)==checkpoint_sha,
            script_unchanged=sha256(__file__)==report['source_script_sha256'],
            peak_cuda_bytes=torch.cuda.max_memory_allocated())
        assert report['source_unchanged'] and report['script_unchanged']
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        write()


if __name__ == '__main__':
    main()
