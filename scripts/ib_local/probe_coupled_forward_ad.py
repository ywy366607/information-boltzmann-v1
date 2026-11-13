"""Read-only forward-AD admission on one native GDN direction, not training.

The body cache is jointly dualized; experiences and their read/write hooks are
outside this first native-only admission. Fail closed if a kernel lacks JVP.
"""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',action='store_true',help='audit using unfused reference operators')
    parser.add_argument('--reverse-check',action='store_true',help='check one scalar JVP against reverse AD')
    parser.add_argument('--tokens',type=int,default=None,help='positive real-token admission length')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    for key,suffix in [('HF_HOME','huggingface'),('TRANSFORMERS_CACHE','transformers'),
                       ('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),
                       ('TEMP','tmp'),('TMP','tmp')]:
        os.environ[key]='D:/ml_cache/'+suffix
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    from torch.autograd import forward_ad
    from torch.func import functional_call
    from scripts.ib_local import run_native_s_credit_retention as old
    from scripts.ib_local import pretrained_gdn_stream_load as loading
    from scripts.ib_local.frozen_mlp_int8 import load_streamed_int8_frozen
    from scripts.ib_local.readonly_checkpoint import load_readonly_checkpoint
    from scripts.ib_local.scalar_credit_trial import ScalarCreditLearner
    from scripts.ib_local.pretrained_gdn import clone_cache,cache_nbytes

    source=ROOT/'scratch/joint_next_pending_20261108/pending_recovery.pt'
    digest=old.sha256(source)
    if digest!='514a45cbe1753c8f056ff75cae2b84121f07c2b5b2a70a6e9e24b9aa9eaca1b5':
        raise ValueError('protected source differs')
    runtime=argparse.Namespace(vendor=Path('E:/cgb_gdn340_probe/vendor/flash-linear-attention-0.2.2'),
        model=Path('E:/cgb_gdn340_probe/model'),bounded_autotune=True)
    loader=loading.load_streamed_pure_gdn
    loading.load_streamed_pure_gdn=partial(load_streamed_int8_frozen,
        mapped_directory=ROOT/'scratch/frozen_weight_cache_20261109')
    try:
        _,_,_,model,_=old.load_runtime(runtime)
    finally:
        loading.load_streamed_pure_gdn=loader
    owner=load_readonly_checkpoint(source)
    state=owner['learner']
    learner=ScalarCreditLearner(model,layers=state['layers'],capacity=state['capacity'])
    learner.load_state_dict(state)
    named=dict(model.named_parameters())
    matches=[name for name,p in named.items() if p is model.model.layers[1].attn.A_log]
    if len(matches)!=1:
        raise RuntimeError('cannot identify native A_log direction')
    name=matches[0]
    count=args.tokens if args.tokens is not None else (4 if args.reference else 128)
    if count < 2:
        raise ValueError('at least two actual tokens required')
    tokens=json.loads((ROOT/'results/cgb_water_last_attempt_20261109/scalar_curriculum.json').read_text())['training'][0]['prefix'][:count]
    x=torch.tensor(tokens,device='cuda')[None]
    cache=clone_cache(learner.cache)
    original_bytes=cache_nbytes(cache)
    report=dict(status='running',scope='one coupled native GDN forward-mode direction; no experience hooks',
                source_sha256=digest,parameter=name,input_tokens=len(tokens),optimizer_updates=0,
                main_life_updates=0,state_bytes=original_bytes)
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    try:
        model.eval()
        if args.reference:
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                baseline=model(x,past_key_values=clone_cache(cache),use_cache=True)
                baseline_logits=baseline.logits.detach().cpu()
                baseline_states=clone_cache(baseline.past_key_values)
                del baseline
            report['native_baseline_peak_allocated']=torch.cuda.max_memory_allocated()
            from scripts.ib_local.coupled_gdn_reference import install_reference
            report['reference_patches']=install_reference(model)
            with torch.no_grad():
                reference=model(x,past_key_values=clone_cache(cache),use_cache=True)
                delta=(reference.logits.detach().cpu()-baseline_logits).float()
                state_diffs=[]
                for layer_a,layer_b in zip(baseline_states.states,reference.past_key_values.states):
                    a,b=layer_a['recurrent_state'],layer_b['recurrent_state']
                    state_diffs.append(float((a.float()-b.float()).norm()/(a.float().norm()+1e-12)))
                report['reference_vs_native']=dict(logit_max_abs=float(delta.abs().max()),
                    logit_rms=float(delta.square().mean().sqrt()),state_relative_l2=state_diffs)
                del baseline_states,reference,baseline_logits,delta
            torch.cuda.reset_peak_memory_stats()
        untouched_cache=clone_cache(cache)
        with forward_ad.dual_level(), torch.no_grad():
            def dualize(value):
                if isinstance(value,torch.Tensor) and value.is_floating_point():
                    return forward_ad.make_dual(value,torch.zeros_like(value))
                if isinstance(value,tuple):
                    return tuple(dualize(item) for item in value)
                if isinstance(value,list):
                    return [dualize(item) for item in value]
                return value
            cache.states=[{k:dualize(v) for k,v in layer.items()} for layer in cache.states]
            direction=torch.zeros_like(named[name]);direction.reshape(-1)[0]=1
            parameter=forward_ad.make_dual(named[name],direction)
            output=functional_call(model,{name:parameter},(x,),
                                   {'past_key_values':cache,'use_cache':True},strict=False)
            primal,tangent=forward_ad.unpack_dual(output.logits)
            if tangent is None:
                raise RuntimeError('native forward produced no logit tangent')
            tangent_count=0
            tangent_bytes=0
            recurrent_tangents=[]
            for layer in output.past_key_values.states:
                for value in layer.values():
                    for item in value if isinstance(value,(list,tuple)) else (value,):
                        if isinstance(item,torch.Tensor) and forward_ad.unpack_dual(item).tangent is not None:
                            tangent_count+=1
                            t=forward_ad.unpack_dual(item).tangent
                            tangent_bytes+=t.numel()*t.element_size()
                recurrent=forward_ad.unpack_dual(layer['recurrent_state']).tangent
                recurrent_tangents.append(None if recurrent is None else recurrent.detach().float().cpu())
            tangent_cpu=tangent.detach().float().cpu()
            scalar=(output.logits.float().square().mean())
            scalar_primal,scalar_tangent=forward_ad.unpack_dual(scalar)
            report.update(status='jvp_computed',logit_tangent_norm=float(tangent.float().norm()),
                          state_tensors_with_tangent=tangent_count,tangent_state_bytes=tangent_bytes,
                          jvp_peak_allocated=torch.cuda.max_memory_allocated(),logit_shape=list(primal.shape),
                          scalar_primal=float(scalar_primal), scalar_jvp=float(scalar_tangent),
                          per_layer_recurrent_tangent_norm=[None if t is None else float(t.norm()) for t in recurrent_tangents])
            if args.reference:
                split_cache=clone_cache(untouched_cache)
                split_cache.states=[{k:dualize(v) for k,v in layer.items()} for layer in split_cache.states]
                parts=[]
                half=x.shape[1]//2
                for piece in (x[:,:half],x[:,half:]):
                    part=functional_call(model,{name:parameter},(piece,),
                        {'past_key_values':split_cache,'use_cache':True},strict=False)
                    split_cache=part.past_key_values
                    parts.append(part.logits)
                split_logits=torch.cat(parts,dim=1)
                split_primal,split_tangent=forward_ad.unpack_dual(split_logits)
                report['split_stream_check']=dict(
                    primal_max_abs=float((primal.float()-split_primal.float()).abs().max()),
                    tangent_relative_l2=float((tangent.float()-split_tangent.float()).norm()/(tangent.float().norm()+1e-30)),
                    recurrence_relative_l2=[float((expected.to(actual.device)-actual.float()).norm()/(expected.to(actual.device).norm()+1e-30))
                        for expected,layer in zip(recurrent_tangents,split_cache.states)
                        for actual in [forward_ad.unpack_dual(layer['recurrent_state']).tangent]
                        if expected is not None and actual is not None])
        del output, primal, tangent, cache
        if args.reference:
            def measure(estimated,actual):
                a,b=estimated.flatten().double(),actual.flatten().double()
                na,nb=a.norm(),b.norm()
                return dict(reference_norm=float(na),finite_difference_norm=float(nb),
                            relative_l2=float((a-b).norm()/(na+1e-30)),
                            cosine=float(torch.dot(a,b)/(na*nb+1e-30)))
            report['centered_difference']={}
            for eps in (.05,.1):
                values=[]
                for sign in (1,-1):
                    value=named[name].detach().clone()
                    value.reshape(-1)[0]+=sign*eps
                    with torch.no_grad():
                        out=functional_call(model,{name:value},(x,),
                            {'past_key_values':clone_cache(untouched_cache),'use_cache':True},strict=False)
                    values.append(dict(logits=out.logits.detach().float().cpu(),
                        states=[z['recurrent_state'].detach().float().cpu() for z in out.past_key_values.states]))
                    del out,value
                fd=(values[0]['logits']-values[1]['logits'])/(2*eps)
                checks={str(i+1):measure(t,(values[0]['states'][i]-values[1]['states'][i])/(2*eps))
                        for i,t in enumerate(recurrent_tangents) if t is not None and float(t.norm())>1e-12}
                report['centered_difference'][str(eps)]=dict(logits=measure(tangent_cpu,fd),recurrent=checks)
            if args.reverse_check:
                model.requires_grad_(False)
                selected=named[name].detach().clone().requires_grad_(True)
                torch.cuda.reset_peak_memory_stats()
                with torch.enable_grad():
                    out=functional_call(model,{name:selected},(x,),
                        {'past_key_values':clone_cache(untouched_cache),'use_cache':True},strict=False)
                    value=out.logits.float().square().mean()
                    grad,=torch.autograd.grad(value,selected)
                reverse=float(grad.reshape(-1)[0])
                report['reverse_check']=dict(scalar=float(value.detach()),vjp=reverse,
                    abs_difference=abs(reverse-report['scalar_jvp']),
                    peak_allocated=torch.cuda.max_memory_allocated())
    except Exception as exc:
        report.update(status='jvp_unsupported_or_failed',exception_type=type(exc).__name__,
                      exception=str(exc),traceback=traceback.format_exc())
    report.update(seconds=time.monotonic()-started,peak_cuda_allocated=torch.cuda.max_memory_allocated(),
                  source_unchanged=old.sha256(source)==digest)
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
