"""Full-size motor-head VJP/Adam memory admission on a disposable real-state copy.

No word is sent to the world. A counterfactual score derivative at the first
saved-queue tick stresses native backpropagation and new optimizer allocation.
This is an interface/resource test, never a behavioral or learning result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.ib.fly_interface.native_causal_audit import (
    ensure_gpu_idle, restore_native, pending_queue_tape, sha256,
)
from information_boltzmann.core import fly_bptt_learning as fl
from information_boltzmann.core.fly_native_motor_port import (
    action_logits, MOTOR_PORT_PARAMETER_NAMES,
)
from information_boltzmann.core.fly_native_policy_credit import (
    native_parameter_names, prepare_native_policy_backward, _auxiliary_backward, _finite,
)


def audit(checkpoint: Path, output: Path):
    ensure_gpu_idle([])
    output.mkdir(parents=True, exist_ok=True)
    source_sha = sha256(checkpoint)
    learner, saved, args, changed = restore_native(checkpoint, output, frozen=False)
    if changed or not learner.model.native_motor_port_enabled:
        raise ValueError('Admission must restore the enabled port without migration')
    model = learner.model
    prepare_native_policy_backward(model)
    tape = pending_queue_tape(saved, 1, 1)[0]
    job = learner.native_context_state.get('pending_teaching') or {}
    source = job.get('source', tape['forced']['sources'][0])
    state = fl.prepare_sensory_expectation(model, learner.state, learner.previous_token, source)
    out = fl.advance_fly_token_stream(model, state,
        torch.tensor([learner.previous_token], device=state.h.device), ticks=1,
        hold_input=learner.hold_input, writer_baseline_clock=learner.writer_baseline_clock,
        fresh=True, sensory_on=True, voice=tape['last_voice'],
        executed_action=source == 'self_hearing',
        reward_ticks=fl.reward_ticks_left(model, tape['last_reward'], 0),
        ai_g=learner.ai_g, ai_tape=[False], read_every=True, update_expectation=False)
    if not bool(out[4][0]):
        raise ValueError('No real terminal packet at this tick; do not invent one')
    logits = action_logits(model, out[3][:1]).float()
    # No world submission or fabricated reward: only a finite score derivative.
    score = logits.log_softmax(-1)[0, logits.detach().argmax(-1)[0]]
    names = native_parameter_names(model)
    own = tuple(n for n,p in model.named_parameters()
                if n in MOTOR_PORT_PARAMETER_NAMES and p.requires_grad)
    parameters = dict(model.named_parameters())
    selected = names + own
    with _auxiliary_backward(model):
        gradients = torch.autograd.grad(score, [parameters[n] for n in selected], allow_unused=True)
    norms = {}
    for name, gradient in zip(selected, gradients):
        if gradient is not None:
            _finite(gradient, f'counterfactual derivative: {name}')
            norms[name] = float(gradient.norm())
            parameters[name].grad = gradient
    if not any(norms.get(n, 0.) > 0 for n in names):
        raise ValueError('No native physiology derivative reached from the motor head')
    # Disposable branch only; force actual lazy Adam moments for the new head.
    learner.optimizer.step()
    own_states = {n: sorted(learner.optimizer.state[parameters[n]]) for n in own}
    if any('exp_avg' not in own_states[n] for n in own):
        raise ValueError('New action parameters did not allocate optimizer state')
    torch.cuda.synchronize()
    report = {'checkpoint': str(checkpoint), 'sha256': source_sha,
        'trained': int(saved['bptt_train_tokens']), 'scope': 'counterfactual one-tick VJP/resource test; no world action or learning benefit',
        'actual_first_saved_interval': tape['forced']['intervals'][0], 'executed_diagnostic_ticks': 1,
        'motor_cells': int(model.native_motor_body_ids.numel()),
        'action_parameter_shapes': {n:list(parameters[n].shape) for n in own},
        'gradient_norms': norms, 'optimizer_state': own_states,
        'peak_mib': torch.cuda.max_memory_allocated()/1024**2,
        'allocated_mib': torch.cuda.memory_allocated()/1024**2,
        'source_unchanged': source_sha == sha256(checkpoint),
        'full_long_action_window_admitted': False,
        'status': 'one_tick_native_vjp_and_optimizer_allocation_passed'}
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status':report['status'], 'peak_mib':report['peak_mib']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.checkpoint, args.output)
