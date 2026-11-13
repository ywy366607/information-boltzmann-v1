"""Read-only real GDN sidecar: checked program affects recall and receives CE."""
import argparse
from contextlib import ExitStack
from functools import partial
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
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
    from scripts.ib_local.pretrained_gdn_learning import language_loss, PretrainedExperienceLearner
    from scripts.ib_local.causal_relation_runtime import admit

    source = ROOT/'scratch/joint_next_pending_20261108/pending_recovery.pt'
    source_hash = old.sha256(source)
    if source_hash != '514a45cbe1753c8f056ff75cae2b84121f07c2b5b2a70a6e9e24b9aa9eaca1b5':
        raise ValueError('protected checkpoint hash differs')
    relation = admit([{'op':'const','arg':250000}], 2, [[0,0],[500000,128]])
    neutral = admit([{'op':'const','arg':0}], 2, [[0,0],[500000,128]])
    runtime = argparse.Namespace(vendor=Path('E:/cgb_gdn340_probe/vendor/flash-linear-attention-0.2.2'),
                                 model=Path('E:/cgb_gdn340_probe/model'),bounded_autotune=True)
    loader = loading.load_streamed_pure_gdn
    loading.load_streamed_pure_gdn = partial(load_streamed_int8_frozen,
        mapped_directory=ROOT/'scratch/frozen_weight_cache_20261109')
    try:
        _, _, _, model, _ = old.load_runtime(runtime)
    finally:
        loading.load_streamed_pure_gdn = loader
    state = load_readonly_checkpoint(source)['learner']
    learner = ScalarCreditLearner(model,layers=state['layers'],capacity=state['capacity'])
    learner.load_state_dict(state)
    data = json.loads((ROOT/'results/cgb_water_last_attempt_20261109/scalar_curriculum.json').read_text())
    ids = data['training'][0]['prefix']
    x = torch.tensor(ids[:128],device='cuda')[None]
    target = torch.tensor(ids[1:129],device='cuda')[None]
    baseline, _ = PretrainedExperienceLearner.predict(learner,x,clone_cache(learner.cache))
    for memory in learner.memories.values():
        memory.install_causal_relation(neutral)
    neutral_logits, _ = PretrainedExperienceLearner.predict(learner,x,clone_cache(learner.cache))
    neutral_max = float((baseline.float()-neutral_logits.float()).abs().max())
    if neutral_max != 0.:
        raise AssertionError('neutral program changed baseline forward')
    for memory in learner.memories.values():
        memory.install_causal_relation(relation)
        memory.begin(learner.optimizer.param_groups[1],query_limit=x.shape[1])
    # Compare the candidate with the same capture and gradient mode, not only
    # the read-only prediction wrapper.
    with torch.no_grad(), ExitStack() as stack:
        for layer,memory in learner.memories.items():
            stack.enter_context(memory.attach(model,layer=layer))
        candidate_frozen = model(x,past_key_values=clone_cache(learner.cache),use_cache=True).logits
    with ExitStack() as stack:
        for layer,memory in learner.memories.items():
            stack.enter_context(memory.attach(model,layer=layer))
        output = model(x,past_key_values=clone_cache(learner.cache),use_cache=True)
    _,ce = language_loss(output.logits,target)
    layers = list(learner.memories)
    leaves = [learner.memories[i].causal_receipt[0] for i in layers]
    gradients = torch.autograd.grad(ce,leaves,allow_unused=True)
    for leaf,grad in zip(leaves,gradients):
        leaf.grad = grad
    settled = {str(layer+1): learner.memories[layer].settle_causal_relation() for layer in layers}
    _,base_ce = language_loss(baseline,target)
    report = dict(status='complete', scope='research-copy actual checkpoint; no optimizer update',
        source_sha256=source_hash, candidate='fixed +0.25 read multiplier, not learned',
        baseline_ce=float(base_ce), candidate_ce=float(ce.detach()),
        neutral_logit_max_change=neutral_max,
        capture_mode_logit_max_change=float((candidate_frozen.float()-output.logits.detach().float()).abs().max()),
        logit_rms_change=float((baseline.float()-output.logits.detach().float()).square().mean().sqrt()),
        feedback=settled, optimizer_updates=0, main_body_updates=0,
        peak_cuda_allocated=torch.cuda.max_memory_allocated())
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='feedback'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
