"""Matched real-state recognition-route interventions, never train the source."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.ib.fly_interface.native_causal_audit import (
    restore_native, ensure_gpu_idle, sha256, circuit_intervention, pending_queue_tape,
)
from scripts.ib.fly_interface.action_loop_v2_audit import state_digest
from information_boltzmann.core import fly_bptt_learning as fl
from information_boltzmann.core.fly_native_generative import _clone_state, _preserve_rollout
from information_boltzmann.core.fly_native_context import compartment_chart, context_rollout
from information_boltzmann.core.fly_active_inference import motor_logits


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--follow-targets', type=int, default=8)
    parser.add_argument('--idle-desktop-pid', type=int, action='append', default=[])
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.run.resolve()):
        parser.error('Use a separate diagnostic directory')
    admission = ensure_gpu_idle(args.idle_desktop_pid)
    args.output.mkdir(parents=True, exist_ok=True)
    source_sha = sha256(args.checkpoint)
    presence = (args.run / 'caregiver_presence.json').read_bytes()
    learner, saved, cfg, changed = restore_native(args.checkpoint, args.output, frozen=True)
    if changed:
        raise RuntimeError('Unexpected native migration')
    model = learner.model
    job = learner.native_context_state['pending_teaching']
    if job is None:
        raise ValueError('A saved native recognition job is required')
    tape = pending_queue_tape(saved, args.follow_targets, args.follow_targets)[0]
    ticks = tape['forced']['intervals'][0]
    intervals = tape['forced']['intervals']
    inputs = [int(learner.previous_token)] + tape['targets'][:-1]
    input_sources = [job['source']] + tape['forced']['sources'][:-1]
    token, source = int(learner.previous_token), job['source']
    if token != job['token']:
        raise ValueError('Recognition job and held word differ')
    physical = fl.prepare_sensory_expectation(model, _clone_state(learner.state), token, source)
    # Generate the actual current sensory actuator tape at the retained clock.
    # Mother is still speaking, so there is no sampled world command/hidden tail.
    eye_tapes = []
    with _preserve_rollout(model), torch.no_grad():
        direct_state, direct_muscles = _clone_state(physical), learner.ai_g.clone()
        for index, (word, duration, origin) in enumerate(zip(inputs, intervals, input_sources)):
            direct_state = fl.prepare_sensory_expectation(model, direct_state, word, origin)
            direct = fl.advance_fly_token_stream(model, direct_state,
                torch.tensor([word], device=physical.h.device), [duration], ticks=duration,
                hold_input=learner.hold_input, writer_baseline_clock=learner.writer_baseline_clock,
                fresh=True, sensory_on=True, voice=tape['last_voice'] if index == 0 else True,
                reward_ticks=fl.reward_ticks_left(model, tape['last_reward'] if index == 0 else False, 0),
                ai_g=direct_muscles, ai_tape=[], read_every=True, update_expectation=False)
            eye_tapes.append(direct[-4].detach().cpu().reshape(-1).tolist())
            direct_state, direct_muscles = fl.expect_next(model, direct[0], direct[1]), direct[-1]
        actual_field = compartment_chart(model, direct_state)[0].detach().cpu()
        actual_read = direct[1].detach().cpu()
    eye_tape = eye_tapes[0]
    graph = np.load(cfg.graph, allow_pickle=True)
    groups = {'MBON': torch.unique(model.comp_post.long()),
              'DAN': torch.as_tensor(graph['dan_indices'], device=physical.h.device),
              'KC': model.comp_kc.long()}
    parameter_digest = state_digest(model.state_dict())
    physical_digest = state_digest(learner.state.state_dict())
    rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    def route(start, sensory_on):
        # Preserve the saved muscle state as well as the body; observation
        # rollouts use rest muscles only when a fixed sensory tape makes them
        # irrelevant to the field. A motor read must use actual muscles.
        with _preserve_rollout(model), torch.no_grad():
            state, muscles = _clone_state(start), learner.ai_g.clone()
            segment = int(getattr(learner, 'stream_segment_ticks', 16) or 16)
            first_field = None
            for index, (word, duration, origin) in enumerate(zip(inputs, intervals, input_sources)):
                state = fl.prepare_sensory_expectation(model, state, word, origin)
                for at in range(0, duration, segment):
                    length = min(segment, duration - at)
                    out = fl.advance_fly_token_stream(model, state,
                        torch.tensor([word], device=state.h.device), [length], ticks=length,
                        hold_input=learner.hold_input, writer_baseline_clock=learner.writer_baseline_clock,
                        fresh=at == 0, sensory_on=sensory_on,
                        voice=tape['last_voice'] if index == 0 else True,
                        reward_ticks=fl.reward_ticks_left(model, tape['last_reward'] if index == 0 else False, at),
                        ai_g=muscles, ai_tape=[False] * length, read_every=True,
                        eye_forced=eye_tapes[index][at:at + length], update_expectation=False)
                    state, muscles, latent = out[0], out[-1], out[1]
                if first_field is None:
                    first_field = compartment_chart(model, state)[0].detach().double().cpu()
                state = fl.expect_next(model, state, latent)
            lp = model.decoder(model.read_norm(latent)).float().log_softmax(-1)[0]
            return state, lp, muscles, first_field
    quiet, _ = context_rollout(learner, physical, token, ticks, sensory_on=False,
                                voice=False, reward=False)
    drift = (compartment_chart(model, quiet)[0] - compartment_chart(model, physical)[0]).detach().double().cpu()
    target = job['mean'] + drift
    dimension = len(target)
    delay = learner.consequence_state['control']['clock_ticks'] - job['clock'] + ticks
    covariance = job['covariance'] + torch.eye(dimension) * job['process_variance'] * delay
    precision = torch.linalg.inv(covariance)
    report = {'at': datetime.now(timezone.utc).isoformat(), 'source': str(args.checkpoint),
              'source_sha256': source_sha, 'trained': saved['bptt_train_tokens'],
              'scope': 'acute recognition-route delivery; not learned improvement or full exploration',
              'gpu_admission': admission, 'first_interval_ticks': ticks, 'ticks': sum(intervals),
              'intervals': intervals, 'held_inputs': inputs, 'held_token': token, 'held_source': source,
              'eye_min': min(eye_tape), 'eye_max': max(eye_tape), 'arms': {}}
    baseline = None
    try:
        for arm in ('intact', 'sham', 'sensory_off', 'kc_terminal_off', 'DAN_off', 'phi_frozen', 'restored'):
            start = _clone_state(physical)
            context = (nullcontext({}) if arm == 'intact' else
                       circuit_intervention(model, start, groups,
                           'intact' if arm in ('sham', 'restored', 'sensory_off') else arm))
            with context, torch.no_grad():
                result, lp, muscles, first_field = route(start, arm != 'sensory_off')
                field = compartment_chart(model, result)[0].detach().double().cpu()
                error = first_field - target
                words = lp.exp()
                raw = motor_logits(model, muscles)[0, 1]
            if baseline is None:
                baseline = field.clone(), result.dopamine.detach().clone(), words.clone()
            row = {'field_delta_norm': float((field - baseline[0]).norm()),
                   'dopamine_delta_norm': float((result.dopamine - baseline[1]).norm()),
                   'word_distribution_maxabs_delta': float((words - baseline[2]).abs().max()),
                   'native_recognition_kl': float(.5 * error @ precision @ error),
                   'native_speech_probability': float(raw.sigmoid()),
                   'field_change_norm': float((field - compartment_chart(model, physical)[0].detach().cpu()).norm())}
            if arm == 'intact':
                row['actual_field_replay_maxabs'] = float((field - actual_field).abs().max())
                row['actual_read_replay_maxabs'] = float((fl.read_fly_motor(model, result).detach().cpu() - actual_read).abs().max())
            report['arms'][arm] = row
            print(json.dumps({'arm': arm, **row}), flush=True)
        report['controls_passed'] = all(report['arms'][arm]['field_delta_norm'] < 2e-4 and
            report['arms'][arm]['word_distribution_maxabs_delta'] < 2e-6 for arm in ('sham', 'restored'))
        report['controls_passed'] &= max(report['arms']['intact']['actual_field_replay_maxabs'],
            report['arms']['intact']['actual_read_replay_maxabs']) < 2e-4
        report['parameters_restored'] = state_digest(model.state_dict()) == parameter_digest
        report['body_preserved'] = state_digest(learner.state.state_dict()) == physical_digest
        report['rng_preserved'] = torch.equal(rng[0], torch.get_rng_state()) and torch.equal(rng[1], torch.cuda.get_rng_state())
        if not all(report[k] for k in ('controls_passed', 'parameters_restored', 'body_preserved', 'rng_preserved')):
            raise RuntimeError('Matched recognition controls failed')
        report['status'] = 'acute_route_check_passed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report.update(source_unchanged=sha256(args.checkpoint) == source_sha,
            main_pause_unchanged=(args.run / 'caregiver_presence.json').read_bytes() == presence)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
