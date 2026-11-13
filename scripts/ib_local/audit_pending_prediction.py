"""Read-only reconstruction of saved, actually issued answer predictions."""
import argparse
from contextlib import ExitStack
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.ib_local.delayed_prediction_credit import differentiable_short_recurrence, fork_view
from scripts.ib_local.pretrained_gdn import enable_streaming_convolution, new_recurrent_cache, clone_cache
from scripts.ib_local.pretrained_gdn_experience import copy_tree
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.pretrained_gdn_stream_load import load_streamed_pure_gdn
from scripts.ib_local.run_irreversible_credit import read_cases
from scripts.ib_local.evaluate_pretrained_gdn_ruler import encode_case
from scripts.ib_local.prepare_pretrained_gdn_ruler import sha256
from scripts.ib_local.run_pretrained_gdn_cross_window import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'checkpoint', 'manifest', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    report = dict(source_sha256=sha256(args.checkpoint), rows=[], retained_updates=0)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    _, cases = read_cases(args.manifest, [2048], payload['next_case']+1)
    _, gold = encode_case(tokenizer, cases[payload['next_case']][1])
    labels = gold+[tokenizer.eos_token_id]
    model, _ = load_streamed_pure_gdn(args.model)
    enable_streaming_convolution(model)
    source = payload['learner']
    learner = PretrainedExperienceLearner(model, layers=source['layers'], capacity=source['capacity'])
    learner.load_state_dict(source)
    device = next(model.parameters()).device
    cache = None
    with differentiable_short_recurrence():
        for index, (record, label) in enumerate(zip(payload['pending'].records, labels)):
            if record.boundary is not None:
                cache = new_recurrent_cache(record.boundary['seen_tokens'])
                cache.states = copy_tree(record.boundary['states'], device)
            if cache is None:
                raise ValueError('missing initial held boundary')
            logits = []
            for differentiable in (False, True):
                shadows = {i: fork_view(w, record.views[i]) for i,w in learner.memories.items()}
                with torch.set_grad_enabled(differentiable), ExitStack() as stack:
                    for i, shadow in shadows.items():
                        stack.enter_context(shadow.attach(model, layer=i, capture=False,
                                                          clock_offset=record.clock_offset))
                    output = model(record.inputs.to(device), past_key_values=clone_cache(cache), use_cache=True)
                    logits.append(output.logits[0, -1].detach().float().cpu())
                    if not differentiable:
                        following = clone_cache(output.past_key_values)
                del output, shadows
            issued = record.logits.float()
            def ce(x):
                return float(F.cross_entropy(x[None], torch.tensor([label])))
            row = dict(index=index, length=record.inputs.shape[1], seen=record.prediction_seen,
                issued_ce=ce(issued), inference_ce=ce(logits[0]), differentiable_ce=ce(logits[1]),
                inference_logit_difference=float((logits[0]-issued).abs().max()),
                differentiable_logit_difference=float((logits[1]-issued).abs().max()),
                views_clock=[int(v.memory.clock) for v in record.views.values()])
            report['rows'].append(row)
            print(row, flush=True)
            write_json(args.output, report)
            cache = following
    report['source_unchanged'] = sha256(args.checkpoint)==report['source_sha256']
    report['status'] = 'complete'
    write_json(args.output, report)


if __name__ == '__main__':
    main()
