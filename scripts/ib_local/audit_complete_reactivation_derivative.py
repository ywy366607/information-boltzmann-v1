"""One real128-input local GPU VJP; no optimizer step or training stream."""
import argparse
from functools import partial
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[2]
    for key, suffix in [('HF_HOME','huggingface'), ('TRANSFORMERS_CACHE','transformers'),
                        ('TRITON_CACHE_DIR','triton'), ('TORCHINDUCTOR_CACHE_DIR','inductor'),
                        ('TEMP','tmp'), ('TMP','tmp')]:
        os.environ[key] = 'D:/ml_cache/'+suffix
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    from scripts.ib_local import run_native_s_credit_retention as old
    from scripts.ib_local import pretrained_gdn_stream_load as loading
    from scripts.ib_local.frozen_mlp_int8 import load_streamed_int8_frozen
    from scripts.ib_local.readonly_checkpoint import load_readonly_checkpoint
    runtime = argparse.Namespace(vendor=Path('E:/cgb_gdn340_probe/vendor/flash-linear-attention-0.2.2'),
        model=Path('E:/cgb_gdn340_probe/model'),bounded_autotune=True)
    digest = old.sha256(args.checkpoint)
    loader = loading.load_streamed_pure_gdn
    loading.load_streamed_pure_gdn = partial(load_streamed_int8_frozen,
        mapped_directory=root/'scratch/frozen_weight_cache_20261109')
    try:
        _,_,_,model,_ = old.load_runtime(runtime)
    finally:
        loading.load_streamed_pure_gdn = loader
    from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
    from scripts.ib_local.pretrained_gdn import new_recurrent_cache
    from scripts.ib_local.pretrained_gdn_experience import copy_tree
    owner = load_readonly_checkpoint(args.checkpoint)
    state = owner['learner']
    learner = PretrainedExperienceLearner(model,layers=state['layers'],capacity=state['capacity'])
    learner.load_state_dict(state)
    learner.enable_complete_reactivation()
    versions = [p._version for p in model.parameters()]
    report = dict(protocol=__doc__,optimizer_updates=0,source_sha256=digest,layers={})
    for layer,wrapper in learner.memories.items():
        block = model.model.layers[layer]
        query = wrapper.memory.history[wrapper.memory.history_valid][-1:].clone().requires_grad_(True)
        executed, inputs = {}, []
        def replay(capsule):
            cache = new_recurrent_cache()
            cache.states = [{} for _ in range(layer)]+[copy_tree(capsule.boundary,'cuda')]
            x = capsule.complete_inputs.to('cuda').detach().clone().requires_grad_(True)
            response = block(x,past_key_values=cache,use_cache=False)[0]
            inputs.append(x)
            executed[id(capsule)] = response
            return response
        full = wrapper.retrieve(query,replay)
        selected = [block.attn.a_proj.weight,block.attn.b_proj.weight]
        gradients = torch.autograd.grad(full.square().mean(),inputs+selected,allow_unused=True)
        tail = [float(g[:,65:].float().norm()) if g is not None else 0. for g in gradients[:len(inputs)]]
        native = [float(g.float().norm()) if g is not None else 0. for g in gradients[len(inputs):]]
        wrapper.detach_replay = True
        held = wrapper.retrieve(query,lambda capsule:executed[id(capsule)])
        held_grad = torch.autograd.grad(held.square().mean(),selected,allow_unused=True)
        difference = float((full.detach()-held.detach()).abs().max())
        if difference != 0 or any(g is not None and bool(g.any()) for g in held_grad):
            raise AssertionError('held changed values or retained old native derivative')
        if not tail or not all(x>0 for x in tail) or not all(x>0 for x in native):
            raise AssertionError('complete native program has no tail/gate derivative')
        report['layers'][str(layer+1)] = dict(replayed_inputs=[int(x.shape[1]) for x in inputs],
            tail_input_gradient_norms=tail,old_forget_write_gradient_norms=native,
            full_held_value_difference=difference,held_old_native_gradient_zero=True)
        wrapper.detach_replay = False
    report.update(status='complete',source_unchanged=old.sha256(args.checkpoint)==digest,
        model_parameters_unchanged=versions==[p._version for p in model.parameters()],
        peak_cuda_allocated=torch.cuda.max_memory_allocated())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
