"""Current-state native evidence-to-choice interventions; never train the source.

This measures finite-expert policy mediation, not language ability or a complete
subjective sensory model. All arms keep the actual clock, ports and decoder.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.ib.fly_interface.native_causal_audit import (
    restore_native, ensure_gpu_idle, sha256, circuit_intervention, pending_queue_tape,
)
from scripts.ib.fly_interface.action_loop_v2_audit import state_digest
from information_boltzmann.core.fly_native_generative import NativeCategoricalWorld, _clone_state
from information_boltzmann.core.fly_native_policy_credit import native_parameter_names
from information_boltzmann.core.fly_bptt_learning import read_fly_motor
from information_boltzmann.core.fly_active_inference import motor_logits, ai_eye


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidates', type=int, default=4)
    parser.add_argument('--idle-desktop-pid', type=int, action='append', default=[])
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.run.resolve()):
        parser.error('Keep diagnostic outputs outside the paused individual')
    admission = ensure_gpu_idle(args.idle_desktop_pid)
    args.output.mkdir(parents=True, exist_ok=True)
    source_sha = sha256(args.checkpoint)
    presence = (args.run / 'caregiver_presence.json').read_bytes()
    learner, saved, cfg, changed = restore_native(
        args.checkpoint, args.output, frozen=False, overrides={'kc_dan_release': True})
    if changed:
        raise RuntimeError('Unexpected native restore migration')
    model = learner.model
    model.eval()
    reference = torch.load(args.reference, mmap=True, map_location='cpu', weights_only=False)
    world = NativeCategoricalWorld(learner, {
        n: reference['parameters'][n] for n in native_parameter_names(model)},
        {'path': str(args.reference), 'origin': reference['provenance']})
    import pandas as pd
    graph = np.load(cfg.graph, allow_pickle=True)
    table = pd.read_feather(cfg.graph.parent / 'body-annotations.feather')
    table = table.set_index('bodyId').reindex(graph['neuron_body_ids'])
    groups = {key: torch.as_tensor(values, device=learner.state.h.device, dtype=torch.long)
              for key, values in {
                  'KC': np.flatnonzero(table['class'].astype(str).values == 'Kenyon_Cell'),
                  'MBON': np.flatnonzero(table['class'].astype(str).values == 'MBON'),
                  'DAN': graph['dan_indices'], 'output': model.read_indices.cpu().numpy(),
              }.items()}
    with torch.no_grad():
        initial_logits = model.decoder(model.read_norm(read_fly_motor(model, learner.state))).float()[0]
    candidates = initial_logits.topk(args.candidates).indices.tolist()
    ticks = max(1, round(cfg.stream_pace_mean))
    eye = float(ai_eye(model, learner.ai_g, getattr(learner, 'eye_forced', None)).detach())
    initial_digest = state_digest(learner.state.state_dict())
    parameter_digest = state_digest(model.state_dict())
    rng_before = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    started = time.monotonic()
    report = {'at': datetime.now(timezone.utc).isoformat(), 'trained': saved['bptt_train_tokens'],
              'source_checkpoint': str(args.checkpoint), 'source_sha256': source_sha,
              'scope': 'acute native-categorical value mediation, no capability claim',
              'gpu_admission': admission, 'ticks': ticks, 'eye_gain': eye,
              'candidate_ids': candidates, 'arms': {}, 'main_training_started': False}

    @torch.no_grad()
    def policy(forecast, physical):
        logits = model.decoder(model.read_norm(read_fly_motor(model, physical))).float()[0]
        # Fixed current motor command isolates value mediation. No speech draw.
        raw = motor_logits(model, learner.ai_g)[0, 1]
        wait = world.information(forecast['log_probs_wait']).to(logits)
        values = torch.full_like(logits, wait)
        values[candidates] = torch.stack([world.information(lp) for lp in forecast['log_probs_words']]).to(logits)
        lp = logits.log_softmax(-1)
        partition = torch.logsumexp(lp + values, -1) - wait
        speak = torch.sigmoid(raw + partition)
        words = (lp + values).softmax(-1)
        return {'wait_information': float(wait), 'word_information': values[candidates].cpu().tolist(),
                'timing_logit_delta': float(partition), 'speak_probability_per_tick': float(speak),
                'base_speak_probability_per_tick': float(raw.sigmoid()),
                'selected_word_probability': words[candidates].cpu().tolist()}, words

    try:
        base_forecast = None
        base_words = None
        for arm in ('plain', 'sham', 'phi_zero', 'KC_off', 'MBON_off', 'DAN_off',
                    'kc_terminal_off', 'restored'):
            physical = _clone_state(learner.state)
            context = (nullcontext() if arm == 'plain' else
                       circuit_intervention(model, physical, groups,
                                            'intact' if arm in ('sham', 'restored') else arm))
            with context, torch.no_grad():
                forecast = world.command_values(physical, candidates, ticks, eye_gain=eye)
                row, word_probability = policy(forecast, physical)
            if base_forecast is None:
                base_forecast, base_words = forecast, word_probability
            row['word_policy_maxabs_delta'] = float((word_probability - base_words).abs().max())
            row['predictive_logp_maxabs_delta'] = float(
                (forecast['log_probs_wait'] - base_forecast['log_probs_wait']).abs().max())
            report['arms'][arm] = row
            (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
            print(json.dumps({'arm': arm, **row}), flush=True)

        # One actual saved external arrival changes subjective model weights;
        # re-evaluate exactly the same proposals to isolate belief-to-choice.
        chunk = pending_queue_tape(saved, 1, 1)[0]
        duration = int(chunk['forced']['intervals'][0])
        predictions = world.predict(learner.state, chunk['previous_token'], duration,
                                    voice=chunk['last_voice'], reward=chunk['last_reward'], eye_gain=eye)
        report['actual_external_evidence'] = world.update(
            predictions, chunk['targets'][0], chunk['forced']['sources'][0])
        after, after_words = policy(base_forecast, learner.state)
        after['word_policy_maxabs_delta'] = float((after_words - base_words).abs().max())
        report['same_state_after_evidence'] = after

        # Identity reference is the no-epistemic-drive control, not a reset brain.
        old_reference = world.state['reference_parameters']
        world.state['reference_parameters'] = {n: dict(model.named_parameters())[n].detach().cpu().clone()
                                               for n in world.names}
        try:
            with torch.no_grad():
                identity = world.command_values(learner.state, candidates, ticks, eye_gain=eye)
                report['identity_reference'], _ = policy(identity, learner.state)
        finally:
            world.state['reference_parameters'] = old_reference
        # CUDA sparse sums may differ at final accumulation bits; explicitly
        # compare both sham and post-intervention restoration with the plain arm.
        report['controls_passed'] = all(
            report['arms'][arm]['predictive_logp_maxabs_delta'] < 2e-4 and
            report['arms'][arm]['word_policy_maxabs_delta'] < 2e-6
            for arm in ('sham', 'restored')) and abs(report['identity_reference']['timing_logit_delta']) < 2e-6
        if not report['controls_passed']:
            raise RuntimeError('Native policy sham/restore control failed; causal claims not accepted')
        report['status'] = 'acute_controls_passed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started,
                      source_unchanged=sha256(args.checkpoint) == source_sha,
                      pause_unchanged=(args.run / 'caregiver_presence.json').read_bytes() == presence,
                      physical_state_unchanged=initial_digest == state_digest(learner.state.state_dict()),
                      model_parameters_unchanged=parameter_digest == state_digest(model.state_dict()),
                      rng_unchanged=(torch.equal(rng_before[0], torch.get_rng_state()) and
                                     torch.equal(rng_before[1], torch.cuda.get_rng_state())),
                      peak_mib=torch.cuda.max_memory_allocated() / 1024**2)
        (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
