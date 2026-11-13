"""Frozen continuous-stream vs. per-case reset audit on the 340M pure GDN.

Compares two execution protocols on the very same 12 official single-needle cases:
1. 'reset_native': Pure GDN, recurrent cache S re-initialized to zero before each case.
2. 'continuous_native': Pure GDN, recurrent cache S is never reset, flowing continuously
   across all 12 cases (over 26,000 tokens of cumulative text).
3. 'reset_complete': Learner with experience memory, reset per case.
4. 'continuous_complete': Learner with experience memory, continuing across all cases.

Zero optimizer updates, zero parameter modifications. Evaluates exact match,
answer NLL, and tracks layer-wise recurrent state norm dynamics across the stream.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

vendor_fla = Path('E:/cgb_gdn340_probe/vendor/flash-linear-attention-0.2.2')
if vendor_fla.exists() and str(vendor_fla) not in sys.path:
    sys.path.insert(0, str(vendor_fla))

import torch
from torch.nn import functional as F

from scripts.ib_local.analyze_ruler_generation import numeric_scores
from scripts.ib_local.evaluate_pretrained_gdn_ruler import encode_case, answer_diagnostics, frozen_decode_replays
from scripts.ib_local.evaluate_pretrained_gdn_tasks import chunk_slices
from scripts.ib_local.pretrained_gdn import (
    cache_nbytes, clone_cache, enable_streaming_convolution, load_pure_gdn, new_recurrent_cache,
)
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.prepare_pretrained_gdn_ruler import sha256
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same, write_json


def recurrent_state_norms(cache):
    """Return L2 norm of recurrent state S for each layer."""
    if cache is None or not hasattr(cache, 'states'):
        return []
    return [float(s['recurrent_state'].float().norm()) for s in cache.states]


class NativeStreamRunner:
    """Manages pure GDN state execution in both reset and continuous modes."""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.cache = new_recurrent_cache()
        self.total_tokens_observed = 0

    def reset(self):
        self.cache = new_recurrent_cache()

    @torch.no_grad()
    def observe_prompt(self, prefix_tokens):
        commits = 0
        first_logits = None
        for section in chunk_slices(len(prefix_tokens)):
            piece = torch.tensor(prefix_tokens[section], device=self.device)[None]
            output = self.model(piece, past_key_values=self.cache, use_cache=True)
            self.cache = clone_cache(output.past_key_values)
            commits += 1
            first_logits = output.logits[:, -1:]
        self.total_tokens_observed += len(prefix_tokens)
        return first_logits, commits

    @torch.no_grad()
    def evaluate_answer(self, prompt_cache, first_logits, gold, tokenizer, max_new_tokens=32):
        # Teacher-forced gold NLL
        gold_tensor = torch.tensor(gold, device=self.device)[None]
        rest_output = self.model(gold_tensor, past_key_values=clone_cache(prompt_cache), use_cache=True)
        scores = torch.cat((first_logits, rest_output.logits), dim=1)
        diagnostics = answer_diagnostics(scores, gold, tokenizer.eos_token_id)

        # Free greedy generation
        generated = []
        cache = clone_cache(prompt_cache)
        next_logits = first_logits[0, -1]
        for step in range(max_new_tokens):
            token = int(next_logits.argmax())
            if token == tokenizer.eos_token_id:
                break
            generated.append(token)
            if step + 1 < max_new_tokens:
                output = self.model(torch.tensor([[token]], device=self.device),
                                    past_key_values=cache, use_cache=True)
                cache = clone_cache(output.past_key_values)
                next_logits = output.logits[0, -1]

        prediction_text = tokenizer.decode(generated, skip_special_tokens=True)
        stopped = (len(generated) > 0 and generated[-1] == tokenizer.eos_token_id) or (len(generated) < max_new_tokens and len(generated) > 0)
        return dict(prediction=prediction_text, generated_ids=generated,
                    generated_tokens=len(generated), gold_tokens=len(gold),
                    stopped_on_eos=stopped,
                    max_new_tokens=max_new_tokens, **diagnostics)

    @torch.no_grad()
    def advance_answer(self, gold, separator=None):
        tokens = list(gold) + (list(separator) if separator else [])
        piece = torch.tensor(tokens, device=self.device)[None]
        output = self.model(piece, past_key_values=self.cache, use_cache=True)
        self.cache = clone_cache(output.past_key_values)
        self.total_tokens_observed += len(tokens)


class CompleteStreamRunner:
    """Manages PretrainedExperienceLearner execution in both reset and continuous modes."""
    def __init__(self, learner, device):
        self.learner = learner
        self.device = device
        self.learner.cache = new_recurrent_cache()
        self.total_tokens_observed = 0

    def reset(self, initial_state):
        self.learner.load_state_dict(initial_state)
        self.learner.cache = new_recurrent_cache()

    @torch.no_grad()
    def observe_prompt(self, prefix_tokens):
        commits = 0
        first_logits = None
        for section in chunk_slices(len(prefix_tokens)):
            piece = torch.tensor(prefix_tokens[section], device=self.device)[None]
            _, logits = self.learner.observe(piece, return_logits=True)
            commits += 1
            first_logits = logits[:, -1:]
        self.total_tokens_observed += len(prefix_tokens)
        return first_logits, commits

    @torch.no_grad()
    def evaluate_answer(self, prompt_cache, first_logits, gold, tokenizer, max_new_tokens=32):
        # Teacher-forced gold NLL
        gold_tensor = torch.tensor(gold, device=self.device)[None]
        rest, _ = self.learner.predict(gold_tensor, prompt_cache)
        scores = torch.cat((first_logits, rest), dim=1)
        diagnostics = answer_diagnostics(scores, gold, tokenizer.eos_token_id)

        # Free greedy generation
        generated = []
        cache = prompt_cache
        next_logits = first_logits[0, -1]
        with frozen_decode_replays(self.learner):
            for step in range(max_new_tokens):
                token = int(next_logits.argmax())
                if token == tokenizer.eos_token_id:
                    break
                generated.append(token)
                if step + 1 < max_new_tokens:
                    output, cache = self.learner.predict(torch.tensor([[token]], device=self.device), cache)
                    next_logits = output[0, -1]

        prediction_text = tokenizer.decode(generated, skip_special_tokens=True)
        stopped = (len(generated) > 0 and generated[-1] == tokenizer.eos_token_id) or (len(generated) < max_new_tokens and len(generated) > 0)
        return dict(prediction=prediction_text, generated_ids=generated,
                    generated_tokens=len(generated), gold_tokens=len(gold),
                    stopped_on_eos=stopped,
                    max_new_tokens=max_new_tokens, **diagnostics)

    @torch.no_grad()
    def advance_answer(self, gold, separator=None):
        tokens = list(gold) + (list(separator) if separator else [])
        piece = torch.tensor(tokens, device=self.device)[None]
        self.learner.observe(piece)
        self.total_tokens_observed += len(tokens)


def run_arm(runner, arm_name, cases, tokenizer, initial_state=None):
    is_continuous = 'continuous' in arm_name
    records = []
    if is_continuous:
        if initial_state is not None and hasattr(runner, 'reset'):
            runner.reset(initial_state)
        elif hasattr(runner, 'reset'):
            runner.reset()

    for idx, (length, row) in enumerate(cases):
        if not is_continuous:
            if initial_state is not None and hasattr(runner, 'reset'):
                runner.reset(initial_state)
            elif hasattr(runner, 'reset'):
                runner.reset()

        cache = runner.learner.cache if hasattr(runner, 'learner') else runner.cache
        incoming_norms = recurrent_state_norms(cache)
        prefix, gold = encode_case(tokenizer, row)

        first_logits, commits = runner.observe_prompt(prefix)
        prompt_cache = clone_cache(runner.learner.cache if hasattr(runner, 'learner') else runner.cache)
        prompt_norms = recurrent_state_norms(prompt_cache)

        eval_result = runner.evaluate_answer(prompt_cache, first_logits, gold, tokenizer, max_new_tokens=32)
        eval_result.update(numeric_scores(eval_result['prediction'], row['outputs']))

        if is_continuous:
            separator = tokenizer.encode('\n\n', add_special_tokens=False)
            runner.advance_answer(gold, separator=separator)
            post_answer_norms = recurrent_state_norms(runner.learner.cache if hasattr(runner, 'learner') else runner.cache)
        else:
            post_answer_norms = []

        record = dict(
            case_index=idx,
            length=length,
            index=row['index'],
            references=row['outputs'],
            first_number_exact=eval_result['first_number_exact'],
            first_number=eval_result['first_number'],
            prediction=eval_result['prediction'],
            answer_nll=eval_result['answer_nll'],
            prompt_tokens=len(prefix),
            cumulative_tokens_observed=runner.total_tokens_observed,
            incoming_S_norm_mean=float(sum(incoming_norms)/len(incoming_norms)) if incoming_norms else 0.0,
            prompt_S_norm_mean=float(sum(prompt_norms)/len(prompt_norms)) if prompt_norms else 0.0,
            incoming_S_norms=incoming_norms,
            prompt_S_norms=prompt_norms,
            post_answer_norms=post_answer_norms,
            raw_eval=eval_result
        )
        records.append(record)
        print(f"[{arm_name}] Case {idx+1}/{len(cases)} (len={length}, idx={row['index']}): "
              f"exact={record['first_number_exact']}, pred={record['first_number']}, "
              f"ans_nll={record['answer_nll']:.4f}, cum_tokens={record['cumulative_tokens_observed']}", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'checkpoint', 'manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--arms', nargs='+', default=['reset_native', 'continuous_native'],
                        choices=['reset_native', 'continuous_native', 'reset_complete', 'continuous_complete'])
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Output file already exists: {args.output}")

    torch.set_num_threads(1)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    manifest = json.loads(args.manifest.read_text())
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    state = payload.get('learner', payload)
    checkpoint_sha = sha256(args.checkpoint)

    print("Loading pure 340M GDN model...", flush=True)
    model, _ = load_pure_gdn(args.model)
    enable_streaming_convolution(model)
    device = next(model.parameters()).device
    assert model.config.attn is None
    print(f"Model loaded on {device}. Verified pure GDN (24 layers, attn=None).", flush=True)

    # Collect the 12 cases (4 per length: 512, 2048, 4096)
    cases = []
    for data in manifest['data']:
        if data['task'] != 'niah_single_1' or data['length'] not in (512, 2048, 4096):
            continue
        assert sha256(data['path']) == data['sha256']
        rows = [json.loads(l) for l in Path(data['path']).read_text().splitlines()][:args.limit]
        for row in rows:
            cases.append((data['length'], row))
    print(f"Collected {len(cases)} evaluation cases across lengths (512, 2048, 4096).", flush=True)

    report = dict(
        status='running',
        protocol=__doc__,
        source_sha256=checkpoint_sha,
        source_updates=state['updates'],
        model_attention_config=model.config.attn,
        standard_softmax_attention_layers=0,
        cases_count=len(cases),
        arms_to_run=args.arms,
        results={},
        summary={},
        started_at=time.time()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)

    # Execute selected arms
    for arm in args.arms:
        print(f"\n=======================================================", flush=True)
        print(f"STARTING ARM: {arm}", flush=True)
        print(f"=======================================================", flush=True)
        arm_start = time.time()
        if 'complete' in arm:
            learner = PretrainedExperienceLearner(model, layers=state['layers'], capacity=state['capacity'])
            learner.load_state_dict(state)
            runner = CompleteStreamRunner(learner, device)
            records = run_arm(runner, arm, cases, tokenizer, initial_state=state)
        else:
            runner = NativeStreamRunner(model, device)
            records = run_arm(runner, arm, cases, tokenizer)

        arm_seconds = time.time() - arm_start
        report['results'][arm] = {
            'records': records,
            'seconds': arm_seconds,
            'total_exact': sum(r['first_number_exact'] for r in records),
            'mean_answer_nll': sum(r['answer_nll'] for r in records) / len(records)
        }
        print(f"\nArm {arm} completed in {arm_seconds:.2f}s: "
              f"exact = {report['results'][arm]['total_exact']}/{len(cases)}, "
              f"mean answer NLL = {report['results'][arm]['mean_answer_nll']:.4f}", flush=True)
        write_json(args.output, report)

    # Compute comparison summary
    report['status'] = 'complete'
    report['total_seconds'] = time.time() - report['started_at']
    report['summary'] = {}
    for arm in args.arms:
        by_len = {}
        for length in (512, 2048, 4096):
            recs = [r for r in report['results'][arm]['records'] if r['length'] == length]
            by_len[str(length)] = {
                'cases': len(recs),
                'exact': sum(r['first_number_exact'] for r in recs),
                'mean_nll': sum(r['answer_nll'] for r in recs) / len(recs)
            }
        report['summary'][arm] = {
            'overall_exact': f"{report['results'][arm]['total_exact']}/{len(cases)}",
            'overall_mean_nll': report['results'][arm]['mean_answer_nll'],
            'by_length': by_len
        }
    write_json(args.output, report)
    print("\nALL ARMS COMPLETE! Final summary written to:", args.output, flush=True)


if __name__ == '__main__':
    main()
