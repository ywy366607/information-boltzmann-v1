"""Bounded real-language audit of old-computation credit to native GDN weights.

This runs actual optimizer updates on a train stream. It is a wiring/engineering
audit, not a capability comparison or replacement of the navigation individual.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from scripts.ib_local.language_navigation_kanli_v4l import live_sigreg
from scripts.ib_local.pretrained_gdn import (
    cache_nbytes, clone_cache, enable_streaming_convolution, load_pure_gdn,
    new_recurrent_cache,
)
from scripts.ib_local.pretrained_gdn_experience import LayerExperienceRecall, copy_tree


def gradient_norms(named, gradients):
    return {name: None if grad is None else float(grad.float().norm())
            for (name, _), grad in zip(named, gradients)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/ib_owt_10000'))
    parser.add_argument('--layers', type=int, nargs='+', default=[1, 12])
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--window', type=int, default=128)
    parser.add_argument('--capacity', type=int, default=16, help='total entries across selected layers')
    args = parser.parse_args()
    if args.steps < 4 or args.window < 128 or args.capacity % len(args.layers):
        parser.error('need >=4 steps, >=128 tokens, and capacity divisible by layer count')
    torch.set_num_threads(1)
    torch.manual_seed(11)
    report = dict(kind='native_gdn_old_computation_credit', status='started',
                  capability_evaluation=False, config=vars(args).copy(), updates=0)
    report['config'] = {k: str(v) if isinstance(v, Path) else v for k, v in report['config'].items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')

    try:
        from transformers import AutoTokenizer
        model, _ = load_pure_gdn(args.model)
        enable_streaming_convolution(model)
        model.requires_grad_(False)
        if len(set(args.layers)) != len(args.layers) or any(not 0 <= i < len(model.model.layers) for i in args.layers):
            raise ValueError('unique existing layers required')
        memories = {i: LayerExperienceRecall(model.config.hidden_size,
            capacity=args.capacity//len(args.layers), device='cuda') for i in args.layers}
        native = []
        for layer in args.layers:
            for name, param in model.model.layers[layer].attn.named_parameters():
                param.requires_grad_(True)
                native.append((f'layer{layer}.{name}', param))
        # FP32 master weights and moments: Adam on half precision parameters can
        # underflow its second moment/epsilon. The executed model remains FP16.
        masters = [torch.nn.Parameter(param.detach().float().clone()) for _, param in native]
        extras = [p for memory in memories.values() for p in memory.parameters()]
        optimizer = torch.optim.AdamW([
            dict(params=masters, lr=1e-4), dict(params=extras, lr=3e-4)],
            eps=1e-8, weight_decay=0, foreach=False)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        offset = json.loads((args.data/'tokenizer.json').read_text())['offset']
        raw = np.load(args.data/'train.npy', mmap_mode='r')[:131072]
        raw_bytes = (raw[raw >= offset]-offset).astype(np.uint8).tobytes()
        ids = tokenizer(raw_bytes.decode('utf-8', errors='replace'),
                        return_tensors='pt', add_special_tokens=False).input_ids.cuda()
        if ids.shape[1] < (args.steps+1)*args.window+1:
            raise ValueError('insufficient real training text')
        report.update(source=dict(split='train', source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            available_tokens=ids.shape[1], score_unit='nats/token'),
            native_trainable_parameters=sum(p.numel() for _, p in native),
            extra_trainable_parameters=sum(p.numel() for p in extras),
            weights=dict(language=10., sigreg=.001), records=[], layer_checks={})
        directions = torch.randn(64, 16)
        cache = new_recurrent_cache()
        torch.cuda.reset_peak_memory_stats()
        save()
        print('LOADED native old-computation audit', flush=True)

        for step in range(args.steps):
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for _, param in native:
                param.grad = None
            for memory in memories.values():
                memory.begin(optimizer.param_groups[1])
            start = step*args.window
            x, target = ids[:, start:start+args.window], ids[:, start+1:start+args.window+1]
            before = clone_cache(cache)
            with ExitStack() as stack:
                for layer, memory in memories.items():
                    stack.enter_context(memory.attach(model, layer=layer))
                output = model(x, past_key_values=cache, use_cache=True)
            ce = F.cross_entropy(output.logits.float().flatten(0, 1), target.flatten())
            regularizers = []
            for memory in memories.values():
                current = memory.encoder(memory.current['source'][0].float().cpu())
                history = memory.encoder(memory.memory.history)
                regularizers.append(live_sigreg(current, history, memory.memory.history_valid, directions))
            reg = torch.stack(regularizers).mean()
            loss = 10*ce + .001*reg.cuda()

            if step == 0:
                # Zero output projection must preserve the same native body,
                # including its incoming state; no memory has been written yet.
                with torch.no_grad():
                    baseline = model(x, past_key_values=before, use_cache=True).logits
                report['zero_inlet_logit_max'] = float((output.logits-baseline).detach().abs().max())
                if report['zero_inlet_logit_max'] != 0:
                    raise AssertionError('neutral recall changed native predictions')

            if step == args.steps-1:
                # Isolate the current CE -> OLD replay branch -> native weights
                # from the ordinary current-window native gradient. No reward
                # or same-window gradient is allowed to masquerade as this path.
                for layer, memory in memories.items():
                    responses = memory.replayed
                    teaching = torch.autograd.grad(ce, responses, retain_graph=True, allow_unused=True)
                    active = [(r, g) for r, g in zip(responses, teaching) if g is not None]
                    named = [(n, p) for n, p in native if n.startswith(f'layer{layer}.')]
                    gradients = torch.autograd.grad([r for r, g in active], [p for _, p in named],
                        grad_outputs=[g.detach() for r, g in active], retain_graph=True, allow_unused=True)
                    ages = (memory.memory.clock-memory.memory.birth[memory.last_read.slots[0]]).tolist()
                    norms = gradient_norms(named, gradients)
                    report['layer_checks'][str(layer)] = dict(old_branch_native_gradient_norms=norms,
                        selected_source_ages_tokens=ages,
                        teaching_gradient_norms=[None if g is None else float(g.float().norm()) for g in teaching])
                    for suffix in ('a_proj.weight', 'b_proj.weight', 'q_proj.weight', 'k_proj.weight', 'v_proj.weight'):
                        if not norms.get(f'layer{layer}.{suffix}', 0):
                            raise AssertionError(f'no old-experience language credit: {layer}/{suffix}')
                    if not any(age > args.window for age in ages):
                        raise AssertionError('no experience actually older than the BPTT window')

            loss.backward()
            credit = {str(layer): memory.memory.settle_read_credit() for layer, memory in memories.items()}
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for _, p in native):
                raise FloatingPointError('nonfinite native GDN gradient')
            for master, (_, param) in zip(masters, native):
                master.grad = None if param.grad is None else param.grad.detach().float().clone()
            torch.nn.utils.clip_grad_norm_(masters, 1.)
            for p in extras:
                if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                    raise FloatingPointError('nonfinite recall gradient')
            optimizer.step()
            with torch.no_grad():
                for master, (_, param) in zip(masters, native):
                    param.copy_(master.to(param.dtype))
            # Cache is detached only at the local BPTT boundary. Observed input
            # and boundary capsules are copied before graph references vanish.
            cache = clone_cache(output.past_key_values)
            for memory in memories.values():
                memory.commit()
            record = dict(step=step+1, ce=float(ce.detach()), sigreg=float(reg.detach()),
                tokens=(step+1)*args.window, seconds=time.perf_counter()-started,
                recurrence_bytes=cache_nbytes(cache),
                experience_cpu_bytes=sum(m.storage_nbytes for m in memories.values()),
                credit=credit)
            report['records'].append(record)
            report['updates'] = step+1
            save()
            print(json.dumps(record), flush=True)
            del output, loss, ce, reg, regularizers

        # Serialize the new trainable portion, memories/capsules, fixed native
        # recurrence and RNG. Frozen original parameters are addressed by the
        # verified source manifest, rather than duplicated in every archive.
        checkpoint = args.output.with_suffix('.pt')
        payload = dict(schema=1, model=str(args.model), config=report['config'],
            native={n: p.detach().cpu() for n, p in native},
            memories={i: m.state_dict() for i, m in memories.items()},
            masters=[p.detach().cpu() for p in masters], optimizer=optimizer.state_dict(),
            cache=clone_cache(cache), tokens=args.steps*args.window,
            rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        torch.save(payload, checkpoint)
        # Same-module restore checks old capsules and preferences as well as
        # the recurrent state. The real next-window forward is deterministic.
        next_x = ids[:, args.steps*args.window:(args.steps+1)*args.window]
        if next_x.shape[1] != args.window:
            raise ValueError('need one additional real-language window for restore check')

        def next_logits():
            with torch.no_grad(), ExitStack() as stack:
                for layer, memory in memories.items():
                    stack.enter_context(memory.attach(model, layer=layer))
                result = model(next_x, past_key_values=clone_cache(cache), use_cache=True).logits.clone()
            for memory in memories.values():
                memory.current, memory.replayed, memory.last_read = None, [], None
            return result

        expected = next_logits()
        disk = torch.load(checkpoint, map_location='cpu', weights_only=False)
        with torch.no_grad():
            for name, param in native:
                param.copy_(disk['native'][name].to(param.device))
            for master, saved in zip(masters, disk['masters']):
                master.copy_(saved.to(master.device))
        optimizer.load_state_dict(disk['optimizer'])
        cache = new_recurrent_cache(disk['cache']._seen_tokens)
        cache.states = copy_tree(disk['cache'].states, 'cuda')
        torch.set_rng_state(disk['rng'])
        torch.cuda.set_rng_state(disk['cuda_rng'])
        for layer, memory in memories.items():
            memory.load_state_dict(disk['memories'][layer])
        actual = next_logits()
        report.update(status='passed', checkpoint=str(checkpoint),
            checkpoint_bytes=checkpoint.stat().st_size,
            restored_next_logit_max=float((expected-actual).abs().max()),
            restore_scope=['native_weights', 'fp32_masters', 'optimizer',
                           'experiences_and_capsules', 'recurrent_cache', 'cpu_cuda_rng'],
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        if report['restored_next_logit_max'] != 0:
            raise AssertionError('experience/capsule restore changed next prediction')
        save()
        print('PASSED', flush=True)
    except Exception as error:
        report.update(status='failed', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
