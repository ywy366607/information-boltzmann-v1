"""One real continuing OWT window and one saved-Adam update on a CPU fork.

This diagnoses optimization and readout, not long-run capabilities. No production
state/checkpoint is modified. Hard spikes retain their registered ATan backward.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_fly_current_token_causality import load_mapped_model
from diagnose_fly_clock_readout import physical_copy
import information_boltzmann.core.fly_bptt_learning as learning
from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner, FlyPhysicalState


def flatten_state(state):
    return (state.h, *state.ring, state.ge, state.gi, state.b,
            state.x, state.u, state.baseline)


def unflatten_state(values):
    return FlyPhysicalState(values[0], tuple(values[1:5]), *values[5:])


def checkpoint_event(model, state, token, **options):
    """Functional event recomputation, including all registered physical ticks."""
    def event(ids, *values):
        return flatten_state(ORIGINAL_EVENT(model, unflatten_state(values), ids, **options))
    return unflatten_state(checkpoint(event, token, *flatten_state(state),
                                     use_reentrant=False, preserve_rng_state=False))


ORIGINAL_EVENT = learning.advance_fly_input_event


def head_scores(h, weights, targets):
    z = F.linear(h, weights['output_read.weight'])
    q = F.rms_norm(z, (z.shape[-1],), weights['read_norm.weight'])
    logits = F.linear(q, weights['decoder.weight'], weights.get('decoder.bias'))
    mean_logits = logits.mean(0, keepdim=True)
    full = F.cross_entropy(logits, targets, reduction='none')
    common = F.cross_entropy(mean_logits.expand_as(logits), targets, reduction='none')
    fraction = lambda v: float((v-v.mean(0)).square().sum()/v.square().sum().clamp_min(1e-30))
    return {'full_nll': float(full.mean()), 'own_window_common_nll': float(common.mean()),
            'descriptive_conditional_gain': float(common.mean()-full.mean()),
            'variation_fraction': {'motor': fraction(h), 'projected': fraction(z),
                                   'normalized': fraction(q)},
            'scores': full.tolist()}, q, logits


def linear_gradient_parts(errors, inputs):
    """Exact mean/covariance split without a second vocabulary-sized matrix."""
    a, h = errors.double(), inputs.double()
    ac, hc = a-a.mean(0), h-h.mean(0)
    n = len(h)
    common_sq = float(a.mean(0).square().sum()*h.mean(0).square().sum())
    covariance_sq = float(((ac@ac.T)*(hc@hc.T)).sum()/n**2)
    full_sq = float(((a@a.T)*(h@h.T)).sum()/n**2)
    cross = float(((ac@a.mean(0))*(hc@h.mean(0))).sum()/n)
    return {'full_norm': max(full_sq, 0)**.5, 'common_norm': common_sq**.5,
            'covariance_norm': max(covariance_sq, 0)**.5,
            'identity_relative_error': abs(full_sq-common_sq-covariance_sq-2*cross)/max(full_sq, 1e-30),
            'common_to_covariance_ratio': (common_sq/max(covariance_sq, 1e-30))**.5,
            'mean_error': a.mean(0), 'mean_input': h.mean(0)}


def displacement_accounting(before, after, gradient, parts=None):
    """Use actual rounded parameter displacement, not raw gradient magnitude."""
    total_inner = displacement_sq = parameter_sq = common_inner = 0.0
    a, b, g = before.flatten(), after.detach().flatten(), gradient.flatten()
    width = before.shape[-1] if before.ndim == 2 else 1
    block = width*128 if before.ndim == 2 else 262144
    for left in range(0, len(a), block):
        right = min(left+block, len(a))
        delta = b[left:right].double()-a[left:right].double()
        total_inner += float((g[left:right].double()*delta).sum())
        displacement_sq += float(delta.square().sum())
        parameter_sq += float(a[left:right].double().square().sum())
        if parts is not None:
            rows = slice(left//width, right//width)
            common_inner += float((delta.reshape(-1, width)*parts['mean_error'][rows, None]
                                  *parts['mean_input'][None]).sum())
    result = {'update_norm': displacement_sq**.5,
              'relative_update_norm': (displacement_sq/max(parameter_sq, 1e-30))**.5,
              'gradient_dot_actual_update': total_inner,
              'first_order_predicted_loss_reduction': -total_inner}
    if parts is not None:
        result.update(common_gradient_dot_update=common_inner,
                      covariance_gradient_dot_update=total_inner-common_inner)
    return result


def continuing_motor_window(model, initial, inputs, options):
    state, hs, spikes = initial, [], []
    with torch.no_grad():
        for i, token in enumerate(inputs):
            state = ORIGINAL_EVENT(model, state, torch.tensor([int(token)]), **options)
            hs.append(state.h[:, model.read_indices].clone())
            spikes.append((state.ring[0]>0).clone())
            if (i+1)%8 == 0:
                print(f'Continuation {i+1}/{len(inputs)}', flush=True)
    return torch.cat(hs), state, torch.cat(spikes)


def head_weights(model):
    return {name: parameter for name, parameter in model.named_parameters()
            if name in ('output_read.weight', 'read_norm.weight', 'decoder.weight', 'decoder.bias')}


def fixed_unigram(reference, tokens):
    data = np.load(reference, mmap_mode='r')
    counts = np.ones(50257, dtype=np.float64)
    for left in range(0, len(data), 1_000_000):
        counts += np.bincount(np.asarray(data[left:left+1_000_000], dtype=np.int64), minlength=len(counts))
    return float((-np.log(counts/counts.sum()))[tokens.numpy()].mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if saved['format'] != 'fly-bptt-v1':
        raise ValueError('A full lifecycle BPTT checkpoint is required')
    cfg, old = saved['config'], saved['learner']
    model = load_mapped_model(saved)
    initial = physical_copy(old['physical'])
    learner = FlyBPTTLearner(model, initial, lr=cfg['lr'], lr_decoder=cfg['lr_decoder'],
        lr_synapse=cfg['lr_synapse'], lr_sensory=cfg['lr_sensory'],
        adam_names=old['adam_names'], plasticity_optimizer=old['plasticity_optimizer_kind'],
        settle_ticks=old['settle_ticks'], writer_baseline_clock=old['writer_baseline_clock'],
        learn_stp=old.get('learn_stp', False))
    if old['plasticity_optimizer_kind'] != 'adamw':
        raise ValueError('This registered diagnosis requires two saved AdamW optimizers')
    # Group/name equality is checked before loading the complete existing history.
    current_groups = learner.optimizer.state_dict()['param_groups']
    for actual, stored in zip(current_groups, old['optimizer']['param_groups']):
        if actual['parameter_names'] != stored['parameter_names']:
            raise ValueError('Saved Adam ordering mismatch')
    learner.optimizer.load_state_dict(old['optimizer'])
    learner.sgd.load_state_dict(old['sgd'])
    parameter_names = {id(p): n for n, p in model.named_parameters()}
    ledger = []
    for optimizer in (learner.optimizer, learner.sgd):
        for group in optimizer.param_groups:
            group['fused'], group['capturable'], group['foreach'] = False, False, False
            names = [parameter_names[id(p)] for p in group['params']]
            ledger.append({'names': names, 'lr': group['lr'], 'weight_decay': group['weight_decay'],
                           'betas': group['betas'], 'eps': group['eps'],
                           'saved_steps': [float(optimizer.state[p]['step']) for p in group['params']]})
    optimized = [id(p) for o in (learner.optimizer, learner.sgd) for g in o.param_groups for p in g['params']]
    assert len(optimized) == len(set(optimized)) == len(learner.trainable)
    assert set(optimized) == {id(p) for p in learner.trainable}
    train = np.load(ROOT/cfg['data']/'train.npy', mmap_mode='r')
    cursor = int(saved['train_cursor'])
    targets = torch.tensor(np.asarray(train[cursor+1:cursor+65], dtype=np.int64))
    inputs = torch.cat((targets.new_tensor([old['previous_token']]), targets[:-1]))
    options = {'settle_ticks': learner.settle_ticks,
               'writer_baseline_clock': learner.writer_baseline_clock}
    report = {'scope': __doc__, 'checkpoint': str(args.checkpoint.resolve()),
              'checkpoint_train_targets': int(saved['bptt_train_tokens']),
              'checkpoint_events': old['events'], 'checkpoint_physical_ticks': old['physical_ticks'],
              'train_cursor': cursor, 'window': 32, 'follow_window': 32,
              'input_bridge': int(inputs[0]), 'options': options,
              'optimizer_ledger': ledger, 'all_trainable_parameters_covered_once': True,
              'preregistration': 'results/published/fly_one_update_preregistered_20261005.json',
              'surrogate': 'registered ATan backward; hard-spike function is discontinuous',
              'numerical_scope': 'CPU PyTorch forward/AdamW, equivalent registered algorithm; not bitwise CUDA/Triton equivalence'}
    original_buffer = model.topographic_writer.a_adapt.clone()
    if original_buffer.is_meta:
        # The BPTT writer path is purely functional (state carries the
        # baseline), so unused legacy buffers stay on the meta device under
        # the mapped loader.  Their purity is enforced by construction, and
        # the exact-state replay below cross-checks the physical trajectory.
        print('Writer a_adapt is meta (unused by the BPTT path); purity by construction.', flush=True)
    hs = []
    handle = model.output_read.register_forward_pre_hook(lambda _m, args: hs.append(args[0].detach().clone()))
    learning.advance_fly_input_event = checkpoint_event
    try:
        print('Full joint forward with event checkpointing...', flush=True)
        scores, terminal, features = learner.forward_window(inputs[:32][None], targets[:32][None])
        features.retain_grad()
        loss = scores.mean()
        motors = torch.cat(hs)
        handle.remove()
        before_head = {name: p.detach().clone() for name, p in head_weights(model).items()}
        with torch.no_grad():
            train_description, q, logits = head_scores(motors, before_head, targets[:32])
            errors = logits.softmax(-1).double()
            errors[torch.arange(32), targets[:32]] -= 1
            decoder_parts = linear_gradient_parts(errors, q)
            old_spikes = torch.cat([(v>0).detach() for v in terminal.ring[:1]])
        print(f'Before update NLL {float(loss.detach()):.6f}; full backward...', flush=True)
        loss.backward()
        learning.advance_fly_input_event = ORIGINAL_EVENT
        if not model.topographic_writer.a_adapt.is_meta:
            assert torch.equal(model.topographic_writer.a_adapt, original_buffer),                 'Writer adaptation mutated by the training path'
        model.topographic_writer.a_adapt = original_buffer
        read_parts = linear_gradient_parts(features.grad*32, motors)
        gradient_norm = torch.nn.utils.clip_grad_norm_(learner.trainable, learner.max_grad_norm, error_if_nonfinite=True)
        clip_scale = min(1., learner.max_grad_norm/(float(gradient_norm)+1e-6))
        # Partial gradients sum to the unclipped full gradient; account actual
        # displacement against those gradients before the global clip.
        parts = {'output_read.weight': read_parts, 'decoder.weight': decoder_parts}
        before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        gradients = {n: p.grad.detach() / clip_scale for n, p in model.named_parameters() if p.requires_grad}
        terminal = terminal.detached()
        del loss, scores, features, q, logits
        gc.collect()
        print('Measuring next fresh window before update...', flush=True)
        follow_h_old, _, follow_spikes_old = continuing_motor_window(model, terminal, inputs[32:], options)
        with torch.no_grad():
            follow_description_old, _, _ = head_scores(follow_h_old, before_head, targets[32:])
        learner.optimizer.step()
        learner.sgd.step()
        with torch.no_grad():
            for edge in learner.edges:
                edge.clamp_(0., 5.)
        updates = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                updates[name] = displacement_accounting(before[name], p, gradients[name], parts.get(name))
        del gradients
        learner.optimizer.zero_grad(set_to_none=True)
        learner.sgd.zero_grad(set_to_none=True)
        after_head = head_weights(model)
        with torch.no_grad():
            train_head_only, _, _ = head_scores(motors, after_head, targets[:32])
            follow_head_only, _, _ = head_scores(follow_h_old, after_head, targets[32:])
            # Frozen old history is an explicit fixed reference, distinct from
            # each arm's retrospective own-window common.
            q_history = F.rms_norm(old['latent_window'], (cfg['d_model'],), before_head['read_norm.weight']).mean(0, keepdim=True)
            historical_common = F.linear(q_history, before_head['decoder.weight'], before_head.get('decoder.bias'))
            report['fixed_preupdate_history_common'] = {
                'history_events': len(old['latent_window']),
                'train_nll': float(F.cross_entropy(historical_common.expand(32, -1), targets[:32])),
                'follow_nll': float(F.cross_entropy(historical_common.expand(32, -1), targets[32:]))}
        # Retain only old head matrices for crossed readout comparisons.
        del before
        gc.collect()
        print('Replaying fitted window from the exact saved initial state...', flush=True)
        train_h_new, replay_terminal, _ = continuing_motor_window(model, initial, inputs[:32], options)
        print('Continuing fresh window from the production old-parameter terminal...', flush=True)
        follow_h_new, _, follow_spikes_new = continuing_motor_window(model, terminal, inputs[32:], options)
        with torch.no_grad():
            train_joint, _, _ = head_scores(train_h_new, after_head, targets[:32])
            train_body_only, _, _ = head_scores(train_h_new, before_head, targets[:32])
            follow_joint, _, _ = head_scores(follow_h_new, after_head, targets[32:])
            follow_body_only, _, _ = head_scores(follow_h_new, before_head, targets[32:])
        report.update(
            grad_norm_before_clip=float(gradient_norm), global_clip_scale=clip_scale,
            gradient_parts={n: {k: v for k, v in d.items() if not isinstance(v, torch.Tensor)} for n, d in parts.items()},
            actual_update_by_parameter=updates,
            total_first_order_predicted_loss_reduction=sum(u['first_order_predicted_loss_reduction'] for u in updates.values()),
            fitted_window={'old': train_description, 'head_update_only': train_head_only,
                           'body_update_old_head': train_body_only, 'joint_update_replay': train_joint},
            immediate_fresh_window={'old': follow_description_old, 'head_update_only': follow_head_only,
                                    'body_update_old_head': follow_body_only, 'joint_update': follow_joint},
            physical_after_update={'fitted_replay_terminal_h_change_norm': float((replay_terminal.h-terminal.h).norm()),
                'fitted_replay_terminal_pulse_support_changes': int(((replay_terminal.ring[0]>0)!=old_spikes).sum()),
                'follow_all_neuron_pulse_support_changes': int((follow_spikes_old!=follow_spikes_new).sum()),
                'follow_pulse_support_change_fraction': float((follow_spikes_old!=follow_spikes_new).float().mean())},
            baseline={'train_fixed_unigram_nll': fixed_unigram(ROOT/cfg['reference_train'], targets[:32]),
                      'follow_fixed_unigram_nll': fixed_unigram(ROOT/cfg['reference_train'], targets[32:])},
            wall_seconds=time.perf_counter()-start,
            limitations=['One optimizer update, not a training/capability verdict.',
                'Own-window common uses each arm future-inclusive sample mean and is descriptive only.',
                'Fresh-window two forks share the complete pre-update terminal state; label use confined to preceding update.',
                'Gradient/update inner products are surrogate first-order predictions; finite hard-spike changes are measured separately.'])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'output': str(args.output), 'train': {k: v['full_nll'] for k, v in report['fitted_window'].items()},
                          'follow': {k: v['full_nll'] for k, v in report['immediate_fresh_window'].items()},
                          'first_order_reduction': report['total_first_order_predicted_loss_reduction'],
                          'wall_seconds': report['wall_seconds']}, indent=2), flush=True)
    finally:
        learning.advance_fly_input_event = ORIGINAL_EVENT
        handle.remove()


if __name__ == '__main__':
    main()
