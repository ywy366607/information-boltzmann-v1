"""Frozen per-layer old-credit direction vs equal-norm control audit.

Three existing real OWT windows; no organism history or parameter is retained
from these researcher branches. Same actual Adam state/clipping. Compare held,
full, shuffled response-to-teaching assignments (matched norm), and an explicit
equal-layer-RMS amplification of both correct and shuffled old directions.
Current-window fit and untrained immediate continuation are reported separately.
Neither is an irreversible-life score or proof of long-term useful credit.
"""
import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from scripts.ib_local.credit_scale_diagnostics import (
    equal_layer_norms, magnitude, match_layer_norms, parameter_effect,
)
from scripts.ib_local.pretrained_gdn import clone_cache, enable_streaming_convolution
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner, language_loss
from scripts.ib_local.pretrained_gdn_stream_load import load_streamed_pure_gdn
from scripts.ib_local.prepare_pretrained_gdn_ruler import sha256
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same, write_json


def cpu_gradients(values, parameters):
    return [torch.zeros_like(p, dtype=torch.float32, device='cpu') if g is None
            else g.detach().float().cpu() for g,p in zip(values, parameters)]


def collect(learner, state, x, target, *, held=False):
    learner.load_state_dict(state, detach_replay=held)
    learner.optimizer.zero_grad(set_to_none=True)
    for _,p in learner.native:
        p.grad = None
    for wrapper in learner.memories.values():
        wrapper.begin(learner.optimizer.param_groups[1], query_limit=x.shape[1])
    parameters = [p for _,p in learner.native]
    with learner.observation(), ExitStack() as stack:
        for layer, wrapper in learner.memories.items():
            stack.enter_context(wrapper.attach(learner.model, layer=layer))
        output = learner.model(x, past_key_values=clone_cache(learner.cache), use_cache=True)
        _, ce = language_loss(output.logits, target)
        reg = learner._regularizer()
        loss = 10*ce+.001*reg.to(ce.device)
        full = cpu_gradients(torch.autograd.grad(loss, parameters, retain_graph=not held,
                                                allow_unused=True), parameters)
        metadata = dict(ce=float(ce.detach()), sigreg=float(reg.detach()), layers={})
        wrong = [torch.zeros_like(g) for g in full]
        if not held:
            outputs, switched = [], []
            for layer,wrapper in learner.memories.items():
                pairs = [(slot,r) for slot,r in wrapper.replayed_by_slot.items() if r.requires_grad]
                responses = [r for _,r in pairs]
                teaching = torch.autograd.grad(loss, responses, retain_graph=True, allow_unused=True)
                teaching = [torch.zeros_like(r) if g is None else g.detach()
                            for r,g in zip(responses, teaching)]
                # A cyclic derangement gives each capsule another capsule's
                # later teaching. Forward observations and addresses stay fixed.
                outputs.extend(responses)
                switched.extend(teaching[1:]+teaching[:1])
                norms = torch.tensor([float(g.float().norm()) for g in teaching])
                squared = norms.square()
                total = float(squared.sum())
                metadata['layers'][str(layer+1)] = dict(
                    capsules=len(pairs), nonzero_teaching_capsules=int((norms>0).sum()),
                    source_ages=[int(wrapper.memory.clock-wrapper.memory.birth[s]) for s,_ in pairs],
                    response_teaching_norms=norms.tolist(),
                    largest_capsule_teaching_energy_share=float(squared.max())/total if total else None,
                    teaching_effective_capsules=total**2/float(squared.square().sum()) if total else 0.)
            wrong = cpu_gradients(torch.autograd.grad(outputs, parameters, grad_outputs=switched,
                                                     allow_unused=True), parameters)
        logits = output.logits.detach().float().cpu()
        after = clone_cache(output.past_key_values)
    # Discard unconsumed read probes and all current graphs before a reload.
    for w in learner.memories.values():
        w.memory._read_participation = []
        w.memory._read_credit_active = False
        w.memory._read_credit_entries = 0
    return full, wrong, metadata, logits, after


@torch.no_grad()
def score(learner, ids, cache):
    logits, _ = learner.predict(ids[:-1][None].cuda(), cache)
    return float(F.cross_entropy(logits.float().flatten(0, 1), ids[1:].cuda()))


def shadow(learner, state, gradients):
    learner.load_state_dict(state)
    learner.optimizer.zero_grad(set_to_none=True)
    for master,g in zip(learner.masters, gradients):
        master.grad = g.to(master.device).clone()
    preclip = float(torch.nn.utils.clip_grad_norm_(learner.masters, 1.))
    learner.optimizer.step()
    with torch.no_grad():
        for master,(_,p) in zip(learner.masters, learner.native):
            p.copy_(master.to(p.dtype))
    return [p.detach().cpu().clone() for _,p in learner.native], preclip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('model', 'checkpoint', 'curriculum', 'output'):
        parser.add_argument('--'+field, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    source_hash = sha256(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=False)
    state = payload.get('learner', payload)
    data = torch.load(args.curriculum, map_location='cpu', weights_only=False)
    report = dict(status='running', protocol=__doc__, persistent_optimizer_updates=0,
        shadow_optimizer_steps=0, checkpoint_sha256=source_hash,
        script_sha256=sha256(__file__), source_updates=state['updates'], samples=[],
        curriculum=str(args.curriculum), curriculum_sha256=sha256(args.curriculum),
        new_lean_coverage=0, normalization='diagnostic equal current/old L2 per layer;not installed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    def write():
        report['seconds'] = time.time()-started
        write_json(args.output, report)
    write()
    try:
        print('CREDIT SCALE LOAD', flush=True)
        model, report['weight_loading'] = load_streamed_pure_gdn(args.model)
        enable_streaming_convolution(model)
        learner = PretrainedExperienceLearner(model, layers=state['layers'], capacity=state['capacity'])
        names = [n for n,_ in learner.native]
        groups = {layer+1:[i for i,n in enumerate(names) if n.startswith(f'layer{layer}.')]
                  for layer in learner.layers}
        total_parameters = sum(p.numel() for p in model.parameters())
        sources = [('stream_41472', data['stream'][41472:41729]),
                   ('validation_0', data['heldouts']['validation'][:257]),
                   ('a_unseen_0', data['heldouts']['a_unseen'][:257])]
        for label, seq in sources:
            if len(seq) != 257:
                raise ValueError('257 actual tokens needed for current+future windows')
            x, target = seq[:128][None].cuda(), seq[1:129][None].cuda()
            full, wrong, info, logits, after = collect(learner, state, x, target)
            held, _, hinfo, hlogits, _ = collect(learner, state, x, target, held=True)
            max_difference = float((logits-hlogits).abs().max())
            assert max_difference == 0 and info['ce'] == hinfo['ce']
            old = [f-h for f,h in zip(full, held)]
            wrong, match = match_layer_norms(old, wrong, groups)
            amplified, scales = equal_layer_norms(held, old, groups)
            amplified_wrong, _ = equal_layer_norms(held, wrong, groups)
            variants = dict(held=held, full=full,
                wrong_matched=[a+b for a,b in zip(held, wrong)],
                equal_rms=[a+b for a,b in zip(held, amplified)],
                equal_rms_wrong=[a+b for a,b in zip(held, amplified_wrong)])
            row = dict(sample=label, preupdate_forward_max_difference=max_difference,
                teaching=info, wrong_norm_match=match, amplification=scales, variants={})
            baseline = None
            for mode, gradient in variants.items():
                native, preclip = shadow(learner, state, gradient)
                report['shadow_optimizer_steps'] += 1
                current = score(learner, seq[:129], clone_cache(learner.cache))
                future = score(learner, seq[128:], after)
                if mode == 'held':
                    baseline = native
                result = dict(current_nll=current, unseen_next128_nll=future,
                    preclip_norm=preclip, clip_coefficient=min(1., 1./(preclip+1e-6)), layers={})
                for layer, indices in groups.items():
                    result['layers'][str(layer)] = parameter_effect(
                        [baseline[i] for i in indices], [native[i] for i in indices],
                        [state['native'][names[i]] for i in indices], total_parameters)
                result['whole_native'] = parameter_effect(baseline, native,
                    [state['native'][n] for n in names], total_parameters)
                row['variants'][mode] = result
                print('CREDIT SCALE', label, mode, 'current', current, 'future', future,
                      'changed', result['whole_native']['changed_coordinates'], flush=True)
                del native
            report['samples'].append(row)
            learner.load_state_dict(state)
            assert_same(state, learner.state_dict())
            write()
            del full, held, old, wrong, amplified, amplified_wrong, variants, baseline, logits, hlogits, after
            gc.collect()
            torch.cuda.empty_cache()
        learner.load_state_dict(state)
        assert_same(state, learner.state_dict())
        assert sha256(args.checkpoint) == source_hash and sha256(__file__) == report['script_sha256']
        report.update(status='complete', source_unchanged=True, complete_state_restored=True,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated())
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        write()


if __name__ == '__main__':
    main()
