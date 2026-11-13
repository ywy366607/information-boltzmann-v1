"""Read-only decomposition of live learning paths on a trained real-text state.

No optimizer step, observation commit, or persistent cache mutation is allowed.
Linear encoder VJPs are summed independently and checked against autograd.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import torch

from scripts.ib_local.language_navigation_kanli_v4l import live_sigreg
from scripts.ib_local.pretrained_gdn import load_pure_gdn, enable_streaming_convolution, clone_cache
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner, language_loss
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same


def norm(value):
    return None if value is None else float(value.detach().double().norm())


def linear_vjp(inputs, output_gradient):
    """Direct parameter contribution for one use of the shared nn.Linear."""
    x = inputs.detach().reshape(-1, inputs.shape[-1]).double()
    g = output_gradient.detach().reshape(-1, output_gradient.shape[-1]).double()
    return torch.cat(((g.T @ x).flatten(), g.sum(0)))


def audit(learner, source, x, target, held):
    learner.load_state_dict(source, detach_replay=held)
    before = learner.state_dict()
    calls = {layer: [] for layer in learner.layers}
    phase = {'name': 'recall'}
    with ExitStack() as stack:
        for layer, memory in learner.memories.items():
            memory.begin(learner.optimizer.param_groups[1])

            def hook(module, args, output, layer=layer):
                calls[layer].append(dict(phase=phase['name'], x=args[0], z=output))

            handle = memory.encoder.register_forward_hook(hook)
            stack.callback(handle.remove)
            stack.enter_context(memory.attach(learner.model, layer=layer))
        output = learner.model(x, past_key_values=clone_cache(learner.cache), use_cache=True)
        _, ce = language_loss(output.logits, target)
        regs = []
        for layer, memory in learner.memories.items():
            phase['name'] = 'regularizer'
            current = memory.encoder(memory.current['source'][0].float().cpu())
            history = memory.encoder(memory.memory.history)
            regs.append(live_sigreg(current, history, memory.memory.history_valid, learner.directions))
        reg = torch.stack(regs).mean()
    tensors, tags = [], []
    for layer, memory in learner.memories.items():
        recall_index = regularizer_index = 0
        for call in calls[layer]:
            if call['phase'] == 'recall':
                names = ['query', 'stored_keys', 'unused_stored_action', 'unused_stored_successor']
                call['role'] = names[recall_index] if recall_index < 4 else f'old_value_{recall_index-4}'
                recall_index += 1
            else:
                call['role'] = ['sigreg_current', 'sigreg_history'][regularizer_index]
                regularizer_index += 1
            tensors.append(call['z']); tags.append((layer, call['role']))
        if recall_index != 4 + len(memory.replayed_by_slot) or regularizer_index != 2:
            raise AssertionError('encoder call topology changed; relabel the audit explicitly')
        for name, parameter in memory.encoder.named_parameters():
            tensors.append(parameter); tags.append((layer, 'encoder.' + name))
        tensors.append(memory.inlet.projection.weight); tags.append((layer, 'inlet.weight'))
        for receipt in memory.memory._read_participation:
            tensors.append(receipt.bias); tags.append((layer, 'read_logit'))
    for name, parameter in learner.native:
        tensors.append(parameter); tags.append(('native', name))
    ce_grads = torch.autograd.grad(ce, tensors, retain_graph=True, allow_unused=True)
    reg_grads = torch.autograd.grad(reg, tensors, allow_unused=True)
    maps = {name: dict(zip(tags, gradients)) for name, gradients in [('ce', ce_grads), ('sigreg', reg_grads)]}
    rows = {}
    for layer, memory in learner.memories.items():
        m, read = memory.memory, memory.last_read
        row = dict(slots=read.slots.tolist(), ages_tokens=(m.clock-m.birth[read.slots]).tolist(),
                   weights=read.weights.detach().tolist(), branches={}, encoder_vjp_checks={})
        for call in calls[layer]:
            role = call['role']
            row['branches'][role] = {
                loss: dict(output_gradient_norm=norm(gmap[(layer, role)]),
                           encoder_gradient_norm=None if gmap[(layer, role)] is None else
                           norm(linear_vjp(call['x'], gmap[(layer, role)])))
                for loss, gmap in maps.items()}
        for loss, gmap in maps.items():
            total = torch.cat((gmap[(layer, 'encoder.weight')].detach().double().flatten(),
                               gmap[(layer, 'encoder.bias')].detach().double().flatten()))
            contributions = [linear_vjp(c['x'], gmap[(layer, c['role'])]) for c in calls[layer]
                             if gmap[(layer, c['role'])] is not None]
            summed = torch.stack(contributions).sum(0)
            relative = float((total-summed).norm()/total.norm().clamp_min(1e-30))
            if relative > 2e-5:
                raise AssertionError(f'{loss} encoder decomposition mismatch: {relative}')
            row['encoder_vjp_checks'][loss] = dict(norm=norm(total),
                weighted_norm=norm(total)*(10 if loss == 'ce' else .001), relative_error=relative)
        row['ce_inlet_gradient_norm'] = norm(maps['ce'][(layer, 'inlet.weight')])
        bias_gradient = maps['ce'].get((layer, 'read_logit'))
        row['ce_read_logit_gradient'] = None if bias_gradient is None else bias_gradient.detach().tolist()
        rows[str(layer)] = row
    native = {name: gradient.detach().float().cpu().clone() for tag, gradient in zip(tags, ce_grads)
              if tag[0] == 'native' and gradient is not None for name in [tag[1]]}
    after = learner.state_dict()
    assert_same(before, after)
    return dict(ce=float(ce.detach()), sigreg=float(reg.detach()), layers=rows,
                persistent_state_unchanged=True), native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--curriculum', type=Path, required=True)
    parser.add_argument('--offset', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous evidence')
    torch.set_num_threads(1)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    curriculum = torch.load(args.curriculum, map_location='cpu', weights_only=False)
    stream = curriculum['stream']
    if len(stream) < args.offset+129:
        raise ValueError('real stream too short')
    model, _ = load_pure_gdn(args.model)
    enable_streaming_convolution(model)
    learner = PretrainedExperienceLearner(model, layers=source['layers'], capacity=source['capacity'])
    x, target = stream[args.offset:args.offset+128][None].cuda(), stream[args.offset+1:args.offset+129][None].cuda()
    full, full_native = audit(learner, source, x, target, False)
    held, held_native = audit(learner, source, x, target, True)
    if full != held:
        # Only metadata external to the reported gradient paths may differ;
        # current encoder/inlet gradients are independent of the old-native cut.
        raise AssertionError('held unexpectedly changed a current encoder path or forward')
    difference = {name: norm(full_native[name]-held_native[name]) for name in full_native}
    report = dict(status='complete', checkpoint=str(args.checkpoint), checkpoint_sha256=digest,
        source_updates=source['updates'], dataset_revision=curriculum['dataset_revision'],
        stream_sha256=curriculum['stream_sha256'], offset=args.offset,
        window=128, optimizer_updates=0, commits=0, full_held_current_paths_exact=True,
        full=full, held=held, ce_native_old_route_gradient_norms=difference,
        source_checkpoint_unchanged=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==digest,
        scope='one trained real-text state; gradient reachability and exact decomposition, not an efficacy trial')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
