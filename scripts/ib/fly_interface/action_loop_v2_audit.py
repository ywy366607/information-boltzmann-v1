"""Bounded real-state interface/resource check; never starts the paused lineage."""
import argparse
import gc
import hashlib
import shutil
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.ib.fly_interface.native_causal_audit import restore_native, ensure_gpu_idle, sha256
from information_boltzmann.core.fly_mother_world import DeepSeekMother, MotherWorld, live_window
from information_boltzmann.core.fly_native_motor_port import checkpoint_model_weights


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(v) for v in value)
    return value


def identical(a, b):
    if torch.is_tensor(a):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    return a == b


def state_digest(value):
    """Hash tensor bytes one at a time; never make another full state copy."""
    if torch.is_tensor(value):
        cpu = value.detach().cpu().contiguous()
        return ('tensor', str(cpu.dtype), list(cpu.shape),
                hashlib.sha256(memoryview(cpu.numpy().reshape(-1)).cast('B')).hexdigest())
    if isinstance(value, np.ndarray):
        return ('numpy', str(value.dtype), list(value.shape),
                hashlib.sha256(memoryview(np.ascontiguousarray(value).reshape(-1)).cast('B')).hexdigest())
    if isinstance(value, dict):
        return {k: state_digest(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [state_digest(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument('--native-credit-only', action='store_true',
                        help='Keep the existing v1 posterior/policy and only add native credit')
    parser.add_argument('--kc-dan-release', action='store_true',
                        help='Explicitly restore the existing learnable KC->DAN terminal route')
    parser.add_argument('--action-clock', choices=('arrival', 'commit'), default='arrival')
    parser.add_argument('--generative-shadow', action='store_true',
                        help='Audit the marked-event likelihood without altering the native policy')
    parser.add_argument('--native-generative', action='store_true',
                        help='Actually actuate the native categorical candidate on this isolated copy')
    parser.add_argument('--native-generative-reference', type=Path)
    parser.add_argument('--native-context', action='store_true')
    parser.add_argument('--context-initial-variance', type=float, default=1.)
    parser.add_argument('--context-process-variance', type=float, default=1e-4)
    parser.add_argument('--native-generative-candidates', type=int, default=4)
    parser.add_argument('--native-generative-ticks', type=int, default=0)
    parser.add_argument('--save-copy', action='store_true',
                        help='Keep a full resumable diagnostic branch, never replace the source')
    parser.add_argument('--idle-desktop-pid', action='append', type=int, default=[],
                        help='Exact inspected WDDM desktop PID; accepted only at <=128MiB and zero GPU utilization')
    args = parser.parse_args()
    if args.tokens < 8 or args.tokens % 8:
        parser.error('Use a bounded whole number of original eight-target windows')
    if args.generative_shadow and args.action_clock != 'commit':
        parser.error('--generative-shadow requires --action-clock commit')
    if args.native_generative and (args.action_clock != 'commit' or args.native_generative_reference is None):
        parser.error('Native categorical actuation needs commit clock and an explicit native reference')
    if args.native_context and (args.action_clock != 'commit' or args.native_generative or args.native_credit_only):
        parser.error('Native context requires commit clock and an exclusive candidate')
    if (args.output.resolve().is_relative_to(args.run.resolve())
            or args.output.resolve().is_relative_to(args.checkpoint.parent.resolve())):
        parser.error('Audit output must be separate from the paused individual')
    gpu_admission = ensure_gpu_idle(args.idle_desktop_pid)
    args.output.mkdir(parents=True, exist_ok=True)
    fingerprint = sha256(args.checkpoint)
    presence = (args.run / 'caregiver_presence.json').read_bytes()
    runtime_paths = [ROOT / p for p in (
        'information_boltzmann/core/fly_action_model.py',
        'information_boltzmann/core/fly_consequence_belief.py',
        'information_boltzmann/core/fly_native_policy_credit.py',
        'information_boltzmann/core/fly_mother_world.py',
        'information_boltzmann/core/fly_bptt_learning.py',
        'information_boltzmann/core/fly_active_inference.py',
        'scripts/ib/train_fly_bptt_stream.py',
        'information_boltzmann/core/fly_event_belief.py',
        'information_boltzmann/core/fly_generative_policy.py',
        'information_boltzmann/core/fly_generative_observer.py',
        'information_boltzmann/core/fly_native_generative.py',
        'information_boltzmann/core/fly_native_generative_control.py',
        'information_boltzmann/core/fly_native_context.py',
        'information_boltzmann/core/fly_context_filter.py')]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in runtime_paths}
    learner, saved, cfg, changed = restore_native(
        args.checkpoint, args.output, frozen=False,
        overrides={'kc_dan_release': True} if args.kc_dan_release else None)
    if changed:
        raise RuntimeError('Native loader requests a learning-mode migration')
    for name, value in saved['model'].items():
        if not torch.equal(learner.model.state_dict()[name].detach().cpu(), value):
            raise RuntimeError(f'Native migration changed an existing saved tensor: {name}')
    learner.model.train()
    learner.speak_credit = 'consequence'
    learner.consequence_native_enabled = not (args.native_credit_only or args.native_generative or args.native_context)
    learner.native_credit_enabled = args.native_credit_only and not args.native_generative and not args.native_context
    learner.native_generative_enabled = args.native_generative
    learner.native_context_enabled = args.native_context
    learner.context_initial_variance = args.context_initial_variance
    learner.context_process_variance = args.context_process_variance
    learner.native_generative_reference = args.native_generative_reference
    learner.native_generative_candidates = args.native_generative_candidates
    learner.native_generative_ticks = args.native_generative_ticks
    learner.mother_action_clock = args.action_clock
    learner.generative_shadow_enabled = args.generative_shadow
    plan = json.loads((ROOT / 'data/harness/raising.json').read_text(encoding='utf8'))['mother']
    mother = DeepSeekMother(model=plan['model'], retries=1)
    import tiktoken
    world = MotherWorld(mother, tiktoken.get_encoding('gpt2'), opening=None,
                        history_turns=plan['history_turns'], max_fly_words=plan['max_fly_words'],
                        end_of_turn=tuple(plan['child_end_of_turn']),
                        contingent=plan['contingent'], ticks_per_beat=plan['ticks_per_beat'],
                        pause_beats_before_reply=plan['pause_beats_before_reply'],
                        silent_beats_threshold=plan['silent_beats_threshold'], call_prompt=plan['call_prompt'])
    world.load_state_dict(saved['mother_world'])
    rng = np.random.default_rng()
    rng.bit_generator.state = saved['mother_rng']
    started = time.monotonic()
    start_updates = learner.updates
    if args.save_copy:
        if shutil.disk_usage(args.output).free < args.checkpoint.stat().st_size + 1024**3:
            raise OSError('Insufficient room for the full candidate plus atomic-save reserve')
        from information_boltzmann.runtime.prequential_meter import FirstPassPredictionMeter
        metadata = saved['first_pass_meter']['reference']
        meter = FirstPassPredictionMeter.from_state_dict(saved['first_pass_meter'], metadata)
        phase_meter = FirstPassPredictionMeter(saved['bptt_train_tokens'], saved['train_cursor'], metadata)
        reference_train = np.load(metadata['train'], mmap_mode='r')
        counts = np.ones(metadata['vocab_size'], dtype=np.float64)
        for left in range(0, len(reference_train), 1_000_000):
            counts += np.bincount(reference_train[left:left + 1_000_000].astype(np.int64), minlength=len(counts))
        if hashlib.sha256(counts.tobytes()).hexdigest() != metadata['counts_sha256']:
            raise ValueError('The retained fixed-unigram training reference changed')
        reference_nll = -np.log(counts / counts.sum())
        del counts, reference_train
        new_targets, new_scores = [], []
    report = {'at': datetime.now(timezone.utc).isoformat(), 'source_checkpoint': str(args.checkpoint),
              'source_sha256': fingerprint, 'source_tokens': saved['bptt_train_tokens'],
              'scope': 'isolated real-state native continuation/interface check, not capability evaluation',
              'main_line_started': False, 'optimizer_state_preserved': bool(learner.optimizer.state),
              'candidate': ('native-continuous-context' if args.native_context else
                            'native-categorical-actuation' if args.native_generative else
                            'native-credit-only' if args.native_credit_only else 'ordered-policy-v2'),
              'action_clock': args.action_clock, 'generative_shadow': args.generative_shadow,
              'gpu_admission': gpu_admission,
              'existing_action_schema': learner.consequence_state.get('action_schema'),
              'existing_posterior_updates': learner.consequence_state.get('belief', {}).get('n', 0),
              'motor_brain_gradient': bool(learner.model.ai_brain_gradient),
              'restored_voice_cells': int(getattr(learner.model, 'voice_cells', torch.empty(0)).numel()),
              'restored_reward_cells': int(getattr(learner.model, 'reward_cells', torch.empty(0)).numel()),
              'native_routes': {key: getattr(cfg, key, None) for key in
                                ('kc_dan_release', 'kc_habituation_tau', 'precision',
                                 'act_dopamine_precision', 'mother_voice', 'mother_reward',
                                 'mother_reward_ticks', 'use_stp')},
              'runtime_sha256': hashes, 'window_targets': cfg.window, 'windows': []}
    try:
        for offset in range(0, args.tokens, cfg.window):
            targets, forced, speakers = live_window(
                learner, world, cfg.window, rng, float(cfg.stream_pace_min), float(cfg.stream_pace_mean),
                temperature=plan['fly_temperature'], speak_act=True, interrupt=bool(cfg.mother_interrupt))
            learner.forced_world = forced
            native_before = (learner.model.log_kc_release.detach().clone()
                             if args.kc_dan_release else None)
            scores, gradients = learner.observe(targets)
            if args.save_copy:
                meter.record(scores, reference_nll[targets])
                phase_meter.record(scores, reference_nll[targets])
                new_targets.extend(int(t) for t in targets)
                new_scores.extend(float(s) for s in scores)
            if torch.cuda.max_memory_allocated() > 4000 * 1024**2:
                raise MemoryError('Candidate exceeded 4000 MiB resource budget')
            metrics = learner.last_adaptive_metrics or {}
            if args.native_credit_only and not args.native_generative:
                head_clock = learner.consequence_state.get('head_clock_ticks')
                native_clock = learner.consequence_state.get('native_credit', {}).get('clock_ticks')
                if head_clock != native_clock:
                    raise RuntimeError('Head and native consequence credit clocks disagree')
            row = {'targets': offset + cfg.window, 'sources': forced['sources'],
                   'intervals': forced['intervals'], 'quiet_ranges': forced['quiet_ranges'],
                   'decisions': forced['decisions'],
                   'world_stats': dict(world.stats),
                   'native_events': [e['kind'] for e in forced['consequence_native_events']],
                   'raw_nll': float(np.mean(scores)),
                   'native': {k: v for k, v in metrics.items()
                              if k.startswith(('native_policy_', 'context_teacher_'))},
                   'consequence': dict(getattr(learner, 'last_epistemic', {})),
                   'generative_events': forced.get('generative_shadow'),
                   'head_clock_ticks': learner.consequence_state.get('head_clock_ticks'),
                   'motor_policy': {k: metrics.get(k) for k in
                                    ('habit_kl', 'habit_kl_decision_ticks', 'speak_decisions')},
                   'terminal_gain_update_maxabs': (float((learner.model.log_kc_release.detach() - native_before).abs().max())
                                                    if native_before is not None else None),
                   'peak_mib': torch.cuda.max_memory_allocated() / 1024**2}
            report['windows'].append(row)
            with (args.output / 'windows.jsonl').open('a', encoding='utf8') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(json.dumps({'targets': row['targets'], 'schema': learner.consequence_state.get('action_schema'),
                              'native_events': row['native_events'], 'peak_mib': row['peak_mib']}), flush=True)
        report['status'] = 'learning_completed_roundtrip_pending'
        (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        print('roundtrip: serializing interaction only', flush=True)
        interaction = cpu_copy(learner.interaction_state_dict())
        digest = state_digest(interaction)
        temporary = args.output / 'interaction_roundtrip.tmp.pt'
        torch.save(interaction, temporary)
        del interaction
        # Drop old GPU archives before restoring their serialized replacements.
        # Retaining two 150MB vocabulary-score archives is not needed for equality.
        learner.consequence_state = {}
        learner.pending_speak_credit = None
        learner.speak_trace = None
        if args.native_generative:
            learner.native_generative_state = {}
        if args.native_context:
            learner.native_context_state = {}
        if args.generative_shadow:
            learner.generative_event_state = {}
        gc.collect(); torch.cuda.empty_cache()
        print('roundtrip: restoring without duplicate GPU credit archives', flush=True)
        recovered = torch.load(temporary, map_location='cpu', weights_only=False)
        learner.load_interaction_state(recovered)
        del recovered
        gc.collect()
        report['interaction_roundtrip_exact'] = digest == state_digest(learner.interaction_state_dict())
        temporary.unlink()
        if not report['interaction_roundtrip_exact']:
            raise RuntimeError('Candidate interaction restore changed state')
        report.update(status='interface_passed', actual_updates=learner.updates - start_updates,
                      world_state=world.state_dict(),
                      native_generative_summary=({
                          'external_updates': learner.native_generative_state['external_updates'],
                          'self_excluded': learner.native_generative_state['self_excluded'],
                          'clock_ticks': learner.native_generative_state['control']['clock_ticks'],
                          'settlements': learner.native_generative_state['control']['settlements'],
                          'response_pending': learner.native_generative_state['control']['response'] is not None}
                          if args.native_generative else None),
                      v2_activated=learner.consequence_state.get('action_schema'),
                      v2_posterior_updates=learner.consequence_state.get('belief', {}).get('n', 0),
                      self_echo=learner.consequence_state.get('self_echo'),
                      pending_response=learner.consequence_state.get('response') is not None)
        # Save readable scalar diagnostics, not hundreds of MB of temporary scores.
        native = learner.consequence_state.get('native_credit', {})
        report['native_credit_summary'] = {'clock_ticks': native.get('clock_ticks'),
            'eligibility_norms': {k: float(v.norm()) for k, v in native.get('E', {}).items()},
            'response_pending': native.get('response_score') is not None}
        if args.native_context:
            ctx = learner.native_context_state
            report['native_context_summary'] = {
                'external_updates': ctx['filter']['updates'], 'self_excluded': ctx['self_excluded'],
                'teaching_jobs': ctx['teaching_jobs'], 'pending_teaching': ctx['pending_teaching'] is not None,
                'process_variance': ctx['filter']['process_variance'],
                'published_clock': ctx['published_clock'],
                'clock_ticks': learner.consequence_state['control']['clock_ticks']}
        if args.save_copy:
            trained = saved['bptt_train_tokens'] + len(new_targets)
            cursor = saved['train_cursor'] + len(new_targets)
            meter.verify_cursor(trained, cursor)
            phase_meter.verify_cursor(trained, cursor)
            config = {**saved['config'], 'mother_action_clock': args.action_clock,
                      'mother_generative_shadow': args.generative_shadow,
                      'mother_native_generative': args.native_generative,
                      'mother_native_context': args.native_context,
                      'mother_context_initial_variance': args.context_initial_variance,
                      'mother_context_process_variance': args.context_process_variance,
                      'mother_native_credit': learner.native_credit_enabled,
                      'mother_consequence_native': learner.consequence_native_enabled,
                      'mother_generative_reference': str(args.native_generative_reference),
                      'mother_generative_candidates': args.native_generative_candidates,
                      'mother_generative_ticks': args.native_generative_ticks,
                      'kc_dan_release': bool(args.kc_dan_release or cfg.kc_dan_release)}
            payload = {**saved, 'model': checkpoint_model_weights(learner.model),
                       'learner': learner.state_dict(), 'config': config,
                       'bptt_train_tokens': trained, 'train_cursor': cursor,
                       'target_bptt_tokens': trained,
                       'first_pass_meter': meter.state_dict(),
                       'phase_first_pass_meter': phase_meter.state_dict(),
                       'recent_a': (saved.get('recent_a', []) + new_targets)[-cfg.replay_tokens:],
                       'recent_scores': (saved.get('recent_scores', []) + new_scores)[-cfg.replay_tokens:],
                       'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state(),
                       'mother_world': world.state_dict(), 'mother_rng': rng.bit_generator.state,
                       'diagnostic_branch': {'parent_sha256': fingerprint, 'scope': report['scope'],
                                             'runtime_sha256': hashes, 'raising': json.loads(
                                                 (ROOT / 'data/harness/raising.json').read_text(encoding='utf8'))}}
            path = args.output / 'candidate.pt'
            temporary = args.output / 'candidate.pending.pt'
            torch.save(payload, temporary)
            temporary.replace(path)
            report['full_candidate'] = {'path': str(path), 'sha256': sha256(path),
                                        'trained': trained, 'bytes': path.stat().st_size}
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started,
                      source_unchanged=sha256(args.checkpoint) == fingerprint,
                      main_pause_unchanged=(args.run / 'caregiver_presence.json').read_bytes() == presence,
                      runtime_unchanged=all(sha256(p) == hashes[str(p.relative_to(ROOT))] for p in runtime_paths))
        (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
