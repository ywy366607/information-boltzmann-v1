"""Audit local pretrained pure-GDN loading and the CGB recall boundary.

No optimizer updates or navigation training are performed. Run in the pinned
FLA environment, with a verified local weight manifest. Output separates
loading, causal continuation, constant state, port gradients, and generation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from scripts.ib_local.pretrained_gdn import (
    RecallInlet, cache_nbytes, clone_cache, enable_streaming_convolution,
    load_pure_gdn, new_recurrent_cache,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/ib_owt_10000'))
    parser.add_argument('--skip-backward', action='store_true')
    parser.add_argument('--backward-repeats', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int, default=24)
    args = parser.parse_args()
    if args.backward_repeats < 1:
        parser.error('--backward-repeats must be positive')
    report = dict(kind='pretrained_loading_and_ports', optimizer_updates=0,
                  model=str(args.model.resolve()), status='started')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    try:
        from transformers import AutoTokenizer
        import fla
        torch.set_num_threads(1)
        torch.manual_seed(11)
        report['versions'] = {name: importlib.metadata.version(name)
                              for name in ('torch', 'transformers', 'tokenizers')}
        report['versions']['fla'] = fla.__version__
        report['gpu'] = torch.cuda.get_device_name(0)
        report['config'] = json.loads((args.model/'config.json').read_text())
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model, loading = load_pure_gdn(args.model)
        model.requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        report['loading'] = dict(seconds=time.perf_counter()-start,
            parameter_count=sum(p.numel() for p in model.parameters()),
            footprint_bytes=model.get_memory_footprint(),
            cuda_allocated_bytes=torch.cuda.memory_allocated(), strict=loading)
        save()
        print('LOADED', json.dumps(report['loading']), flush=True)

        offset = json.loads((args.data/'tokenizer.json').read_text())['offset']
        raw = np.load(args.data/'validation.npy', mmap_mode='r')[:16384]
        raw_bytes = (raw[raw >= offset]-offset).astype(np.uint8).tobytes()
        text = raw_bytes.decode('utf-8', errors='replace')
        ids = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids.to('cuda')
        if ids.shape[1] < 513:
            raise ValueError('Real-language reference is too short for this interface audit')
        report['reference'] = dict(dataset=str(args.data), split='validation',
            source_bytes=len(raw_bytes), source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            tokens=ids.shape[1], score_unit='nats/token', capability_evaluation=False)

        with torch.no_grad():
            baseline = model(ids[:, :32], past_key_values=new_recurrent_cache(),
                             use_cache=True).logits
        count = enable_streaming_convolution(model)
        with torch.no_grad():
            revised = model(ids[:, :32], past_key_values=new_recurrent_cache(),
                            use_cache=True).logits
        difference = (baseline.float()-revised.float()).abs()
        report['convolution_compatibility'] = dict(patched_modules=count,
            fresh_prefix_logit_max=difference.max().item(),
            fresh_prefix_logit_rms=difference.square().mean().sqrt().item())
        save()
        print('CONV', report['convolution_compatibility'], flush=True)

        with torch.no_grad():
            full = model(ids[:, :32], past_key_values=new_recurrent_cache(), use_cache=True)
            first = model(ids[:, :13], past_key_values=new_recurrent_cache(), use_cache=True)
            split = model(ids[:, 13:32], past_key_values=clone_cache(first.past_key_values), use_cache=True)
        difference = (full.logits[:, 13:].float()-split.logits.float()).abs()
        report['split_continuation'] = dict(logit_max=difference.max().item(),
            logit_rms=difference.square().mean().sqrt().item(),
            equal_argmax_fraction=(full.logits[:, 13:].argmax(-1)==split.logits.argmax(-1)).float().mean().item())
        # This is the same 19 token suffix with the same weights/state, not a
        # capability threshold. Allow half-precision kernel rounding only.
        if not torch.allclose(full.logits[:, 13:].float(), split.logits.float(), atol=.04, rtol=.01):
            raise RuntimeError('Causal split continuation failed the declared FP16 tolerance')
        save()
        print('SPLIT', report['split_continuation'], flush=True)

        report['state_sizes'] = []
        cache = new_recurrent_cache()
        start = time.perf_counter()
        with torch.no_grad():
            for begin in range(0, 512, 32):
                result = model(ids[:, begin:begin+32], past_key_values=cache, use_cache=True)
                cache = result.past_key_values
                if begin+32 in (32,128,512):
                    report['state_sizes'].append(dict(tokens_seen=begin+32,bytes=cache_nbytes(cache)))
        torch.cuda.synchronize()
        report['stream_512_seconds'] = time.perf_counter()-start
        if len({row['bytes'] for row in report['state_sizes']}) != 1:
            raise RuntimeError('Recurrent tensor storage grew with history')
        saved = clone_cache(cache)
        with torch.no_grad():
            left = model(ids[:, 512:513], past_key_values=clone_cache(saved), use_cache=True)
            right = model(ids[:, 512:513], past_key_values=clone_cache(saved), use_cache=True)
        report['state_restore'] = dict(next_logit_max=(left.logits-right.logits).abs().max().item())
        save()
        print('STATE', report['state_sizes'], flush=True)

        report['generation'] = []
        prompt = 'The most important thing I learned today is'
        prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors='pt').input_ids.to('cuda')
        start = time.perf_counter()
        with torch.no_grad():
            result = model(prompt_ids, past_key_values=new_recurrent_cache(), use_cache=True, logits_to_keep=1)
            generated = []
            for _ in range(args.max_new_tokens):
                token = result.logits[:, -1].argmax(-1, keepdim=True)
                generated.append(int(token))
                if int(token) == tokenizer.eos_token_id:
                    break
                result = model(token, past_key_values=result.past_key_values,
                               use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        report['generation'].append(dict(prompt=prompt,
            continuation=tokenizer.decode(generated, clean_up_tokenization_spaces=False),
            new_tokens=len(generated), seconds=elapsed, tokens_per_second=len(generated)/elapsed,
            interpretation='A base-model continuation, not an instruction-following score'))
        save()
        print('GENERATION', json.dumps(report['generation'], ensure_ascii=False), flush=True)

        if not args.skip_backward:
            inlet = RecallInlet(model.config.hidden_size, device='cuda')
            recalled = torch.linspace(-1,1,64,device='cuda')[None,None]
            # Old FLA's single-token fused decoder is inference-only. A bounded
            # 128-token chunk invokes its supported exact backward kernel.
            with torch.no_grad():
                neutral = model(ids[:, :128], past_key_values=new_recurrent_cache(),
                                use_cache=True, logits_to_keep=1).logits
            start = time.perf_counter()
            with inlet.attach(model, recalled, layer=1):
                result = model(ids[:, :128], past_key_values=new_recurrent_cache(),
                               use_cache=True, logits_to_keep=1)
                loss = F.cross_entropy(result.logits[:, -1].float(), ids[:,128])
                zero_error = (result.logits.detach()-neutral).abs().max().item()
                loss.backward()
            torch.cuda.synchronize()
            grad = inlet.projection.weight.grad
            report['recall_inlet'] = dict(parameters=inlet.projection.weight.numel(),
                zero_inlet_logit_max=zero_error, gradient_norm=grad.norm().item(),
                gradient_finite=bool(grad.isfinite().all()),
                base_received_gradients=any(p.grad is not None for p in model.parameters()),
                seconds=time.perf_counter()-start)
            if not report['recall_inlet']['gradient_finite'] or grad.norm()==0:
                raise RuntimeError('Recall inlet did not receive a finite nonzero language gradient')
            warm_times = []
            for _ in range(args.backward_repeats-1):
                inlet.zero_grad(set_to_none=True)
                start = time.perf_counter()
                with inlet.attach(model, recalled, layer=1):
                    result = model(ids[:, :128], past_key_values=new_recurrent_cache(),
                                   use_cache=True, logits_to_keep=1)
                    F.cross_entropy(result.logits[:, -1].float(), ids[:,128]).backward()
                torch.cuda.synchronize()
                warm_times.append(time.perf_counter()-start)
                if not torch.isfinite(inlet.projection.weight.grad).all():
                    raise RuntimeError('Nonfinite recall gradient on warmed execution')
            report['recall_inlet']['warm_forward_backward_seconds'] = warm_times
            save()
            print('INLET', report['recall_inlet'], flush=True)
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        report['status'] = 'interface_audit_complete'
        save()
    except Exception as error:
        import traceback
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        if torch.cuda.is_available():
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        save()
        raise


if __name__ == '__main__':
    main()
