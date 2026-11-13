"""Frozen official RULER prompts on the complete bounded-recall GDN.

Each independent request restores the incoming individual, starts its native
cache empty, and consumes its prompt once. Prompt writes are temporary; generated
tokens advance a local native cache but never become stored training episodes.
No optimizer update, gold-dependent stopping, or answer-conditioned retrieval.
"""
import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from scripts.ib_local.prepare_pretrained_gdn_ruler import sha256
from scripts.ib_local.pretrained_gdn import (
    enable_streaming_convolution, load_pure_gdn, new_recurrent_cache,
)
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.evaluate_pretrained_gdn_tasks import chunk_slices
from scripts.ib_local.probe_pretrained_gdn_instruction import encode_answer
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same, write_json


@contextmanager
def frozen_decode_replays(learner):
    """Reuse held local responses only while slots and parameters cannot change.

    At most one response per occupied slot; values are discarded on exit. This
    is valid for frozen decoding, never for optimizer steps or prompt commits.
    """
    if torch.is_grad_enabled():
        raise RuntimeError('frozen decoding only')
    saved = []
    try:
        for wrapper in learner.memories.values():
            old = wrapper.__dict__.get('retrieve')
            original = wrapper.retrieve
            cache = {}
            def retrieve(query, replay, original=original, cache=cache):
                def cached(capsule):
                    identity = id(capsule)
                    if identity not in cache:
                        cache[identity] = replay(capsule)
                    return cache[identity]
                return original(query,cached)
            saved.append((wrapper,old))
            wrapper.retrieve = retrieve
        yield
    finally:
        for wrapper, old in saved:
            if old is None:
                del wrapper.retrieve
            else:
                wrapper.retrieve = old


def encode_case(tokenizer, row):
    text = row['input']+row.get('answer_prefix','')
    # SentencePiece adds a dummy prefix when the answer is encoded separately.
    # Joint encoding must preserve the independently encoded prompt exactly.
    sequence, boundary = encode_answer(tokenizer,text,', '.join(row['outputs']))
    prefix, gold = sequence[:boundary], sequence[boundary:]
    if not prefix or not gold:
        raise ValueError('empty benchmark prompt/answer')
    return prefix, gold


def answer_diagnostics(scores, gold, eos):
    """Score answer and termination separately on the gold-prefix branch."""
    labels = torch.tensor(gold+[eos],device=scores.device)
    logits = scores.float().reshape(-1,scores.shape[-1])
    if logits.shape[0] != len(labels):
        raise ValueError('one logit vector per answer token and EOS required')
    losses = F.cross_entropy(logits,labels,reduction='none')
    end = logits[-1]
    return dict(answer_nll=float(losses[:-1].mean()),
        answer_with_eos_nll=float(losses.mean()),eos_nll=float(losses[-1]),
        eos_probability=float((-losses[-1]).exp()),
        eos_rank=1+int((end>end[eos]).sum()))


@torch.no_grad()
def run_case(learner, tokenizer, row, max_new_tokens, *, reset_native=True):
    prefix, gold = encode_case(tokenizer,row)
    device = next(learner.model.parameters()).device
    if reset_native:
        learner.cache = new_recurrent_cache()
    commits = 0
    for section in chunk_slices(len(prefix)):
        _, logits = learner.observe(torch.tensor(prefix[section],device=device)[None],return_logits=True)
        commits += 1
    first_logits = logits[:, -1:]
    prompt_cache = learner.cache
    # Teacher-forced gold NLL is a separate read-only branch after the prompt.
    rest, _ = learner.predict(torch.tensor(gold,device=device)[None],prompt_cache)
    scores = torch.cat((first_logits,rest),dim=1)
    diagnostics = answer_diagnostics(scores,gold,tokenizer.eos_token_id)
    generated = []
    cache = prompt_cache
    next_logits = first_logits[0,-1]
    # The first generated token is computed before any gold is exposed.
    with frozen_decode_replays(learner):
        # Once per request verify the optimization against an uncached read.
        first = torch.tensor([[int(next_logits.argmax())]],device=device)
        uncached, _ = learner.predict(first,prompt_cache)
        cached, _ = learner.predict(first,prompt_cache)
        torch.testing.assert_close(uncached,cached,atol=0,rtol=0)
        for step in range(max_new_tokens):
            token = int(next_logits.argmax())
            if token == tokenizer.eos_token_id:
                break
            generated.append(token)
            if step+1 < max_new_tokens:
                output, cache = learner.predict(torch.tensor([[token]],device=device),cache)
                next_logits = output[0,-1]
    return dict(index=row['index'],prediction=tokenizer.decode(generated,skip_special_tokens=True),
        references=row['outputs'],prompt_tokens=len(prefix),generated_tokens=len(generated),
        generated_ids=generated,
        **diagnostics,gold_tokens=len(gold),prompt_observations=commits,
        stopped_on_eos=token==tokenizer.eos_token_id,max_new_tokens=max_new_tokens,
        answer_position=row.get('token_position_answer'),
        answer_distance_tokens=(len(prefix)-row['token_position_answer'] if 'token_position_answer' in row else None))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model','checkpoint','manifest','official','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--lengths',nargs='+',type=int,default=[512])
    p.add_argument('--tasks',nargs='+',default=['niah_single_1','niah_multikey_1','vt'])
    p.add_argument('--limit',type=int,default=8)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous evaluation')
    torch.set_num_threads(1)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    manifest = json.loads(args.manifest.read_text())
    spec = importlib.util.spec_from_file_location('official_ruler_scores',
        args.official/'scripts/eval/synthetic/constants.py')
    scores = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scores)
    payload = torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    state = payload.get('learner',payload)
    source_hash = sha256(args.checkpoint)
    model, loading = load_pure_gdn(args.model)
    enable_streaming_convolution(model)
    learner = PretrainedExperienceLearner(model,layers=state['layers'],capacity=state['capacity'])
    learner.load_state_dict(state)
    report = dict(status='running',kind='official_ruler_configured_subset',
        checkpoint=str(args.checkpoint),checkpoint_sha256=source_hash,source_updates=state['updates'],
        manifest_sha256=sha256(args.manifest),official_revision=manifest['official_revision'],
        persistent_optimizer_updates=0,protocol=__doc__,results=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,report)
    start = time.time()
    try:
        for data in manifest['data']:
            if data['task'] not in args.tasks or data['length'] not in args.lengths:
                continue
            if sha256(data['path']) != data['sha256']:
                raise ValueError('benchmark data changed')
            rows = [json.loads(x) for x in Path(data['path']).read_text().splitlines()][:args.limit]
            record = dict(task=data['task'],length=data['length'],samples=[],max_new_tokens=data['max_new_tokens'])
            report['results'].append(record)
            for row in rows:
                learner.load_state_dict(state)
                case_start = time.time()
                result = run_case(learner,tokenizer,row,data['max_new_tokens'])
                result['seconds'] = time.time()-case_start
                record['samples'].append(result)
                record['ruler_score'] = scores.TASKS[data['family']]['metric_fn'](
                    [x['prediction'] for x in record['samples']],[x['references'] for x in record['samples']])
                report['seconds'] = time.time()-start
                write_json(args.output,report)
                print(json.dumps(dict(task=data['task'],length=data['length'],index=row['index'],
                    score=record['ruler_score'],prediction=result['prediction'],
                    references=result['references'],seconds=result['seconds'])),flush=True)
        learner.load_state_dict(state)
        assert_same(state,learner.state_dict())
        report.update(status='complete',complete_state_restored=True,
            source_checkpoint_unchanged=sha256(args.checkpoint)==source_hash,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-start)
        if not report['source_checkpoint_unchanged']:
            raise AssertionError('source checkpoint changed')
    except Exception as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        write_json(args.output,report)


if __name__ == '__main__':
    main()
