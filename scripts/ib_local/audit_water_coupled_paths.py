"""Frozen old-gate interventions followed by the full native body's real future.

Only first128 events use a perturbed A_log[0]. Common weights are then restored,
while all24 resulting S/convolution states continue. Not a training algorithm.
"""
import argparse
from functools import partial
import gc
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    root=Path(__file__).resolve().parents[2]
    for key,suffix in [('HF_HOME','huggingface'),('TRANSFORMERS_CACHE','transformers'),
                       ('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),
                       ('TEMP','tmp'),('TMP','tmp')]:
        os.environ[key]='D:/ml_cache/'+suffix
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    from transformers import AutoTokenizer
    from scripts.ib_local import run_native_s_credit_retention as old
    from scripts.ib_local import pretrained_gdn_stream_load as loading
    from scripts.ib_local.frozen_mlp_int8 import load_streamed_int8_frozen
    from scripts.ib_local.readonly_checkpoint import load_readonly_checkpoint
    from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
    from scripts.ib_local.pretrained_gdn import clone_cache
    from scripts.ib_local.run_pretrained_gdn_cross_window import read_text
    source=root/'scratch/joint_next_pending_20261108/pending_recovery.pt'
    protocol=root/'docs/information_boltzmann/CGB_WATER_FINAL_DIAGNOSIS_20261109.md'
    report=dict(status='loading',protocol=__doc__,source_sha256=old.sha256(source),
        script_sha256=old.sha256(__file__),registration_sha256=old.sha256(protocol),
        optimizer_updates=0,main_life_updates=0,native_S_resets=0,
        added_read_injection=False,epsilon=[.01,.005],records=[])
    started=time.monotonic()
    def save():
        report['seconds']=time.monotonic()-started
        old.write_json(args.output/'report.json',report)
    save()
    runtime=argparse.Namespace(vendor=Path('E:/cgb_gdn340_probe/vendor/flash-linear-attention-0.2.2'),
        model=Path('E:/cgb_gdn340_probe/model'),bounded_autotune=True)
    original=loading.load_streamed_pure_gdn
    loading.load_streamed_pure_gdn=partial(load_streamed_int8_frozen,
        mapped_directory=root/'scratch/frozen_weight_cache_20261109')
    try:
        _,_,_,model,_=old.load_runtime(runtime)
    finally:
        loading.load_streamed_pure_gdn=original
    owner=load_readonly_checkpoint(source)
    state=owner['learner']
    learner=PretrainedExperienceLearner(model,layers=state['layers'],capacity=state['capacity'])
    learner.load_state_dict(state)
    model.eval()
    from scripts.ib_local.pretrained_gdn_experience import copy_tree
    inherited=clone_cache(learner.cache)
    rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
    parameters={l:model.model.layers[l].attn.A_log for l in (1,12)}
    baseline_parameters={l:p.detach().clone() for l,p in parameters.items()}
    tokenizer=AutoTokenizer.from_pretrained(runtime.model,local_files_only=True)
    data=root/'data/ib_owt_10000'
    offset=old.read_json(data/'tokenizer.json')['offset']
    baselines={}
    def tensors(value):
        if isinstance(value,torch.Tensor):
            yield value
        elif isinstance(value,dict):
            for k in sorted(value):
                yield from tensors(value[k])
        elif isinstance(value,(tuple,list)):
            for v in value:
                yield from tensors(v)
    def cache_difference(current,reference):
        layers=[]
        for a,b in zip(current,reference):
            numerator=denominator=0.
            for x,y in zip(tensors(a),tensors(b)):
                a32=x.detach().float().cpu();b32=y.float()
                numerator+=float((a32-b32).double().square().sum())
                denominator+=float(b32.double().square().sum())
            layers.append((numerator**.5)/(denominator**.5+1e-30))
        return layers
    @torch.no_grad()
    def run(ids,layer,perturbation,register_baseline=False):
        torch.set_rng_state(rng);torch.cuda.set_rng_state(cuda_rng)
        cache=clone_cache(inherited)
        for l,param in parameters.items():
            param.copy_(baseline_parameters[l])
        if layer is not None:
            parameters[layer][0].add_(perturbation)
        scores={};effects={}
        try:
            for begin in range(0,768,128):
                if (args.output/'STOP').exists():
                    raise RuntimeError('explicit diagnostic STOP')
                if begin in (256,512,640):
                    if register_baseline:
                        baselines[begin]=dict(cache=copy_tree(cache.states,'cpu'))
                    effects[str(begin)]=dict(cache_relative_l2_by_layer=cache_difference(cache.states,baselines[begin]['cache']))
                x=torch.tensor([ids[begin:begin+128]],device='cuda')
                output=model(x,past_key_values=cache,use_cache=True)
                cache=output.past_key_values
                if begin==0:
                    # Past is not undone: retain its entire physical cache.
                    for l,param in parameters.items():
                        param.copy_(baseline_parameters[l])
                if begin in (256,512,640):
                    target=torch.tensor(ids[begin+1:begin+33],device='cuda')
                    loss=torch.nn.functional.cross_entropy(output.logits[0,:32].float(),target,reduction='none')
                    scores[str(begin)]=loss.cpu().tolist()
                    logits=output.logits[0,:32].float().cpu()
                    if register_baseline:
                        baselines[begin]['logits']=logits.clone()
                    base=baselines[begin]['logits']
                    logp=logits.log_softmax(-1);base_logp=base.log_softmax(-1)
                    effects[str(begin)].update(logit_rms=float((logits-base).square().mean().sqrt()),
                        kl_from_baseline=float((base_logp.exp()*(base_logp-logp)).sum(-1).mean()))
                del output,x
        finally:
            for l,param in parameters.items():
                param.copy_(baseline_parameters[l])
        return scores,effects
    try:
        for raw_start in (12_000_000,14_000_000):
            text,digest=read_text(data/'train.npy',offset,raw_start,length=196608)
            ids=tokenizer.encode(text,add_special_tokens=False)
            if len(ids)<769:
                raise ValueError('registered source too short')
            baselines.clear()
            base,base_effect=run(ids,None,0,register_baseline=True)
            repeat,repeat_effect=run(ids,None,0)
            record=dict(raw_start=raw_start,text_sha256=digest,baseline=base,
                        baseline_repeat=repeat,baseline_repeat_effect=repeat_effect,conditions=[])
            report['records'].append(record)
            report['status']='running';save()
            for layer in (1,12):
                for epsilon in (.01,.005):
                    plus,plus_effect=run(ids,layer,epsilon)
                    minus,minus_effect=run(ids,layer,-epsilon)
                    condition=dict(layer=layer+1,head=0,epsilon=epsilon,plus=plus,minus=minus,
                        plus_effect=plus_effect,minus_effect=minus_effect,
                        finite_difference={k:(sum(plus[k])-sum(minus[k]))/(64*epsilon) for k in plus})
                    record['conditions'].append(condition)
                    print(json.dumps(dict(raw_start=raw_start,layer=layer+1,epsilon=epsilon,
                                          finite_difference=condition['finite_difference'])),flush=True)
                    save();gc.collect()
        report.update(status='complete',source_unchanged=old.sha256(source)==report['source_sha256'],
            parameters_restored=all(torch.equal(p,baseline_parameters[l]) for l,p in parameters.items()),
            script_unchanged=old.sha256(__file__)==report['script_sha256'],
            registration_unchanged=old.sha256(protocol)==report['registration_sha256'],
            peak_cuda_allocated=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:
        save()


if __name__=='__main__':
    main()
