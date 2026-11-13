"""Two real continuous windows: feedback edits a checked read program."""
import argparse
from contextlib import ExitStack
from functools import partial
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    for key, suffix in [('HF_HOME','huggingface'),('TRANSFORMERS_CACHE','transformers'),
                        ('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),
                        ('TEMP','tmp'),('TMP','tmp')]:
        os.environ[key] = 'D:/ml_cache/'+suffix
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import torch
    from scripts.ib_local import run_native_s_credit_retention as old
    from scripts.ib_local import pretrained_gdn_stream_load as loading
    from scripts.ib_local.frozen_mlp_int8 import load_streamed_int8_frozen
    from scripts.ib_local.readonly_checkpoint import load_readonly_checkpoint
    from scripts.ib_local.scalar_credit_trial import ScalarCreditLearner
    from scripts.ib_local.pretrained_gdn import clone_cache
    from scripts.ib_local.pretrained_gdn_learning import language_loss
    from scripts.ib_local.causal_relation_runtime import admit, revise_constant_relation

    source = ROOT/'scratch/joint_next_pending_20261108/pending_recovery.pt'
    source_hash = old.sha256(source)
    if source_hash != '514a45cbe1753c8f056ff75cae2b84121f07c2b5b2a70a6e9e24b9aa9eaca1b5':
        raise ValueError('protected checkpoint hash differs')
    # Verify the three possible one-step results before loading the model;
    # a Lean subprocess competing with the 340M model failed on this machine.
    checked = {value:admit([{'op':'const','arg':value}], 2, [[0,0],[1000000,128]])
               for value in (-20000,0,20000)}
    neutral = checked[0]
    config = argparse.Namespace(vendor=Path('E:/cgb_gdn340_probe/vendor/flash-linear-attention-0.2.2'),
                                model=Path('E:/cgb_gdn340_probe/model'),bounded_autotune=True)
    loader = loading.load_streamed_pure_gdn
    loading.load_streamed_pure_gdn = partial(load_streamed_int8_frozen,
        mapped_directory=ROOT/'scratch/frozen_weight_cache_20261109')
    try:
        _, _, _, model, _ = old.load_runtime(config)
    finally:
        loading.load_streamed_pure_gdn = loader
    state = load_readonly_checkpoint(source)['learner']
    learner = ScalarCreditLearner(model,layers=state['layers'],capacity=state['capacity'])
    learner.load_state_dict(state)
    data = json.loads((ROOT/'results/cgb_water_last_attempt_20261109/scalar_curriculum.json').read_text())
    ids = data['training'][0]['prefix']
    if len(ids) < 257:
        raise ValueError('two continuous windows required')
    x = torch.tensor(ids[:128],device='cuda')[None]
    target = torch.tensor(ids[1:129],device='cuda')[None]
    for memory in learner.memories.values():
        memory.install_causal_relation(neutral)
        memory.begin(learner.optimizer.param_groups[1],query_limit=128)
    with ExitStack() as stack:
        for layer,memory in learner.memories.items():
            stack.enter_context(memory.attach(model,layer=layer))
        output = model(x,past_key_values=clone_cache(learner.cache),use_cache=True)
    _,ce = language_loss(output.logits,target)
    layers = list(learner.memories)
    leaves = [learner.memories[layer].causal_receipt[0] for layer in layers]
    gradients = torch.autograd.grad(ce,leaves,allow_unused=True)
    for leaf,gradient in zip(leaves,gradients):
        leaf.grad = gradient
    for layer in layers:
        memory = learner.memories[layer]
        memory.settle_causal_relation()
        memory.memory.settle_read_credit()
    # Commit the actual first window once; no reset or history reconstruction.
    learner.cache = clone_cache(output.past_key_values)
    for memory in learner.memories.values():
        memory.commit()
    next_x = torch.tensor(ids[128:256],device='cuda')[None]
    next_y = torch.tensor(ids[129:257],device='cuda')[None]

    def read_logits(tokens):
        """Same recall path as the scored window. Do not store another episode."""
        with torch.no_grad(), ExitStack() as stack:
            for layer, memory in learner.memories.items():
                stack.enter_context(memory.attach(model, layer=layer, capture=False))
            logits = model(tokens, past_key_values=clone_cache(learner.cache), use_cache=True).logits
        for memory in learner.memories.values():
            memory.current = None
            memory.replayed = []
            memory.replayed_by_slot = {}
            memory.last_read = None
        return logits

    old_logits = read_logits(next_x)
    _,old_ce = language_loss(old_logits,next_y)
    proposals = {}
    for layer in layers:
        memory = learner.memories[layer]
        revised, receipt = revise_constant_relation(memory.causal_relation,prechecked=checked)
        memory.install_causal_relation(revised)
        proposals[str(layer+1)] = receipt
    revised_logits = read_logits(next_x)
    _,revised_ce = language_loss(revised_logits,next_y)
    report = {'status':'complete', 'source_sha256':source_hash,
              'scope':'one real sequential two-window research-copy, fixed native weights',
              'rule_origin':'fixed template local gradient sign edit; not generated by model',
              'first_window_ce':float(ce.detach()), 'edits':proposals,
              'next_window_old_ce':float(old_ce), 'next_window_revised_ce':float(revised_ce),
              'next_window_delta_ce':float(revised_ce-old_ce),
              'next_window_measurement':'attached read, capture false; predict() does not apply the relation',
              'next_window_logit_rms_change':float((revised_logits.float()-old_logits.float()).square().mean().sqrt()),
              'optimizer_updates':0, 'main_body_updates':0,
              'candidate_activation':'research sidecar only', 'peak_cuda_allocated':torch.cuda.max_memory_allocated()}
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
