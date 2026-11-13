"""Frozen question fork using the SAME128-event stream as the live learner.

No observe() calls on arbitrary128..255 prompt chunks, no reset, no padding,
no forced flush. Pending old labels are retained in the source and explicitly
masked only in this disposable frozen diagnostic. Gold scores and free answer
are separate branches from the identical pre-answer state.
"""
import torch

from scripts.ib_local.irreversible_credit_stream import IrreversibleCreditStream, PretrainedStreamBackend
from scripts.ib_local.pretrained_gdn import clone_cache
from scripts.ib_local.pretrained_gdn_experience import copy_tree


@torch.no_grad()
def run_continuous_frozen_case(learner, tokenizer, row, stream_state, max_new_tokens=32):
    from scripts.ib_local.evaluate_pretrained_gdn_ruler import encode_case
    prefix, gold = encode_case(tokenizer, row)
    updates = learner.updates
    versions = [p._version for p in learner.model.parameters()]
    backend = PretrainedStreamBackend(learner)
    stream = IrreversibleCreditStream(backend)
    stream.restore_latest(stream_state)
    frozen_pending_labels = sum(stream.learn_mask)
    stream.learn_mask = [False] * len(stream.learn_mask)
    before_seen = learner.cache._seen_tokens
    try:
        stream.receive(prefix, phase='frozen_prompt', teach=False)
        backend.close()
        # Clone only mutable forward state, not weights or optimizer moments.
        held = dict(memories={i:copy_tree(m.state_dict()) for i,m in learner.memories.items()},
            cache=clone_cache(learner.cache), stream=stream.state_dict(), tokens=learner.tokens,
            rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        scores = stream.receive(gold+[tokenizer.eos_token_id], phase='frozen_gold_branch', teach=False)
        if len(scores) != len(gold)+1:
            raise AssertionError('one score per arrived gold token/EOS required')
        backend.close()
        for i,memory in learner.memories.items():
            memory.load_state_dict(held['memories'][i])
        learner.cache = clone_cache(held['cache'])
        learner.tokens = held['tokens']
        torch.set_rng_state(held['rng'])
        torch.cuda.set_rng_state(held['cuda_rng'])
        backend = PretrainedStreamBackend(learner)
        stream = IrreversibleCreditStream(backend)
        stream.restore_latest(held['stream'])
        emitted = stream.generate(eos_token_id=tokenizer.eos_token_id, max_tokens=max_new_tokens)
        if (learner.updates != updates or versions != [p._version for p in learner.model.parameters()]
                or learner.cache._seen_tokens != before_seen+len(prefix)+len(emitted)):
            raise AssertionError('frozen continuous fork changed parameters/reset/dropped events')
        return dict(index=row['index'],prediction=tokenizer.decode(emitted,skip_special_tokens=True),
            references=row['outputs'],generated_ids=emitted,prompt_tokens=len(prefix),generated_tokens=len(emitted),
            answer_nll=sum(scores[:-1])/len(gold), answer_with_eos_nll=sum(scores)/len(scores),
            eos_nll=scores[-1],gold_tokens=len(gold),stopped_on_eos=bool(emitted and emitted[-1]==tokenizer.eos_token_id),
            max_new_tokens=max_new_tokens,frozen_pending_labels=frozen_pending_labels,
            stream_window=128,source_pending_inputs=len(stream_state['inputs']),
            arbitrary_prompt_chunk_commits=False, optimizer_updates=0)
    finally:
        backend.close()
