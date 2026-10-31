"""Read-only matched circuit interventions through the actual restored learner.

The source checkpoint and caregiver files are never written. Pending-queue mode
continues only already saved mother's tokens, with the saved native pacing RNG;
historical-tape mode is a new frozen replay from the CURRENT saved physical state,
not a reconstruction of the past. Neither mode requests a mother response.
"""
from __future__ import annotations

import argparse
import ast
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core import fly_bptt_learning as fl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def clone_state(state):
    return fl.FlyPhysicalState(**{
        key: tuple(x.detach().clone() for x in value) if key == 'ring' else value.detach().clone()
        for key, value in state.state_dict().items()})


def state_difference(a, b):
    differences = {}
    for key, left in a.state_dict().items():
        right = getattr(b, key)
        pairs = zip(left, right) if key == 'ring' else [(left, right)]
        differences[key] = max((float((x - y).abs().max()) if x.numel() else 0.0
                                for x, y in pairs), default=0.0)
    return differences


def states_close(a, b, rtol=1e-5, atol=1e-6):
    """Float32 atomic synaptic sums can differ in their last accumulation bits."""
    for key, left in a.state_dict().items():
        right = getattr(b, key)
        pairs = zip(left, right) if key == 'ring' else [(left, right)]
        for x, y in pairs:
            if not torch.allclose(x, y, rtol=rtol, atol=atol):
                return False
    return True


def trainer_restore_program(source: str, *, device='cuda'):
    """Extract the native resume block, stopping before migrations/world/training.

    This is deliberately fail-closed: semantic AST anchors must exist exactly
    once. The parser is also the trainer's own parser, not a second flag schema.
    """
    if device not in ('cpu', 'cuda'):
        raise ValueError('Native audit device must be cpu or cuda')
    module = ast.parse(source)
    main = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    parser_start = next(i for i, n in enumerate(main.body)
                        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'parser'
                                                           for t in n.targets))
    parser_end = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Global))
    parser_nodes = main.body[parser_start:parser_end]
    branches = [n for n in main.body if isinstance(n, ast.If)
                and isinstance(n.test, ast.Name) and n.test.id == 'is_scratch_start' and n.orelse
                and 'Loading archived weights' in ast.unparse(n.orelse[0])]
    if len(branches) != 1:
        raise RuntimeError('Native trainer resume AST anchor changed; review loader before proceeding')
    resume = branches[0].orelse
    stop = [i for i, n in enumerate(resume)
            if isinstance(n, ast.If) and ast.unparse(n.test).startswith('args.dan_plastic_lr > 0')]
    if len(stop) != 1 or 'learner.load_interaction_state' not in ast.unparse(ast.Module(resume[:stop[0]], [])):
        raise RuntimeError('Cannot locate the complete native interaction restore boundary')
    # Args/config are supplied between parser creation and resume, by caller.
    resume_module = ast.Module(resume[:stop[0]], [])
    if device == 'cpu':
        # Explicit audit-only placement; retain every state/optimizer operation
        # in the trainer's block. This is not CUDA resource admission.
        class CPUPlacement(ast.NodeTransformer):
            def visit_Constant(self, node):
                return ast.copy_location(ast.Constant('cpu'), node) if node.value == 'cuda' else node

            def visit_Call(self, node):
                node = self.generic_visit(node)
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
                    if node.args or node.keywords:
                        raise ValueError('Review argument-bearing CUDA placement before CPU audit')
                    return ast.copy_location(ast.Call(ast.Attribute(node.func.value, 'to', ast.Load()),
                                                      [ast.Constant('cpu')], []), node)
                return node
        resume_module = CPUPlacement().visit(resume_module)
    return (compile(ast.fix_missing_locations(ast.Module(parser_nodes, [])), '<native-parser>', 'exec'),
            compile(ast.fix_missing_locations(resume_module), '<native-resume>', 'exec'))


def restore_native(checkpoint: Path, report_dir: Path, *, frozen: bool = True, overrides=None, device='cuda'):
    """Use current trainer's complete checkpoint restore without entering main()."""
    path = ROOT / 'scripts/ib/train_fly_bptt_stream.py'
    spec = importlib.util.spec_from_file_location('_native_audit_trainer', path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if saved['format'] != 'fly-bptt-v1':
        raise ValueError('Audit requires a complete native BPTT checkpoint')
    config = saved['config']
    parser_code, resume_code = trainer_restore_program(path.read_text(encoding='utf-8'), device=device)
    namespace = vars(trainer).copy()
    exec(parser_code, namespace)
    parser = namespace['parser']
    args = parser.parse_args([])
    for action in parser._actions:
        if action.dest in config:
            value = config[action.dest]
            if action.type is Path and value is not None:
                value = Path(value)
            setattr(args, action.dest, value)
    args.resume, args.output, args.checkpoint_dir = checkpoint, report_dir, report_dir
    args.explicit_dests = list(config)
    overrides = overrides or {}
    if set(overrides) - {'kc_dan_release', 'body_proxy_compatibility', 'native_motor_port', 'native_credit_cpu', 'amine_modulation', 'mb_terminal_projection', 'apl_cable', 'kc_terminal', 'kc_terminal_geometry'}:
        raise ValueError('Only explicit terminal-feedback, native motor-port or historical body overrides are supported')
    if overrides.get('body_proxy_compatibility') and not frozen:
        raise ValueError('Body proxy compatibility is for frozen historical diagnostics only')
    if (config.get('body_loop') and 'body_proxy_compatibility' not in config
            and 'body_proxy_compatibility' not in overrides):
        raise ValueError('Historical checkpoint used synthetic body feedback. Explicitly choose '
                         'body_proxy_compatibility=True for frozen historical replay or False '
                         'for a recorded native-only migration; do not silently change the path.')
    args.native_audit_frozen = bool(frozen)
    for name, value in overrides.items():
        setattr(args, name, value)
        if name not in args.explicit_dests:
            args.explicit_dests.append(name)
    args.profile_dests = list(config.get('profile_dests', []))
    namespace.update(args=args, train=None, val=None)
    exec(resume_code, namespace)
    learner = namespace['learner']
    # The trainer restores this diagnostic history below the AST cutoff. Keep
    # helper-based full restores identical as well; it is not a fresh buffer.
    if 'latent_window' in saved['learner']:
        with torch.no_grad():
            learner.latent_window.copy_(saved['learner']['latent_window'])
    trainer.attach_mother_senses(learner.model, args)
    if getattr(learner.model, 'ai_mode', None) != 'motor':
        raise ValueError('This audit requires the native motor muscle head')
    if getattr(learner, 'eye_binary', False) or float(getattr(learner, 'sense_noise_max', 0)):
        raise ValueError('Binary eyes/noisy senses require explicit recorded eye/noise tapes')
    if frozen:
        learner.model.eval().requires_grad_(False)
        # A matched frozen audit does not update weights, optimizer, policy
        # credit or pacemaker homeostat. Local physical states still evolve.
        for optimizer in (learner.optimizer, learner.sgd):
            optimizer.state.clear()
        learner.pending_speak_credit = None
        learner.sigreg = None
        learner.ai_sigreg = 0.0  # loss-only; no effect on physical forward
        torch.cuda.empty_cache()
    # These are restored later than the AST cutoff in the real trainer. Restore
    # them here too so callers using frozen=False obtain the complete RNG state.
    torch.set_rng_state(saved['rng_cpu'])
    torch.cuda.set_rng_state(saved['rng_cuda'])
    return learner, saved, args, namespace.get('mode_changed', False)


def pending_queue_tape(saved, limit: int, window: int):
    world = saved.get('mother_world') or {}
    queue = list(world.get('queue') or [])
    if world.get('turn') != 'mother' or not queue:
        raise ValueError('Pending-queue audit stops before any new mother/network event')
    if world.get('mother_turn_fresh'):
        raise ValueError('Fresh mother turn pause is not fully archived; refuse to invent its real clock')
    limit = min(limit, len(queue))
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(saved['mother_rng'])
    config, old = saved['config'], saved['learner']
    pace_min, pace_mean = float(config['stream_pace_min']), float(config['stream_pace_mean'])
    previous = int(old['previous_token'])
    voice = bool(old.get('interaction', {}).get('last_voice', False))
    reward = bool(old.get('interaction', {}).get('last_reward', False))
    chunks = []
    for offset in range(0, limit, window):
        targets = queue[offset: min(offset + window, limit)]
        pace = (float(math.exp(rng.uniform(math.log(pace_min), math.log(pace_mean))))
                if 0 < pace_min < pace_mean else pace_mean)
        intervals = [int(rng.geometric(1.0 / pace)) for _ in targets]
        quiet = [[] for _ in targets]
        chunks.append({'previous_token': previous, 'targets': targets,
                       'last_voice': voice, 'last_reward': reward,
                       'forced': {'pace': pace, 'intervals': intervals, 'quiet_ranges': quiet,
                                  'speakers': ['mother'] * len(targets),
                                  'sources': [world.get('mother_source', 'mother_reply')] * len(targets),
                                  'reply_onsets': [False] * len(targets), 'decisions': []},
                       'scope': 'saved_pending_queue_native_rng_continuation'})
        previous, voice, reward = targets[-1], True, False
    return chunks


def historical_tape(path: Path, limit: int, before: int):
    """Reconstruct exact input ids from the held-input tick tape, never transcript text."""
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    rows = [r for r in rows if int(r.get('bptt_train_tokens', before)) < before]
    scores = {}
    for score_path in sorted(path.parent.glob('scores*.jsonl')):
        for line in score_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                score = json.loads(line)
                scores[int(score['trained'])] = score
    chunks = []
    for row_index, (row, following) in enumerate(zip(rows, rows[1:])):
        lengths = row.get('interval') or []
        quiet = row.get('quiet_ranges') or [[] for _ in lengths]
        ids, pos = [], 0
        for length, ranges in zip(lengths, quiet):
            sensed = int(length) - sum(int(b) - int(a) for a, b in ranges)
            if sensed < 1 or pos + sensed > len(row.get('token', [])):
                raise ValueError('Historical input tape has an unrecoverable fully quiet/missing token')
            part = row['token'][pos:pos + sensed]
            if len(set(part)) != 1:
                raise ValueError('Held-input token tape is inconsistent with intervals')
            ids.append(int(part[0])); pos += sensed
        if pos != len(row.get('token', [])) or not following.get('token'):
            raise ValueError('Historical tick tape length mismatch')
        speakers = row.get('speakers')
        if not speakers or len(speakers) != len(ids):
            raise ValueError('Historical source/speaker alignment is missing')
        targets = ids[1:] + [int(following['token'][0])]
        score = scores.get(int(row['bptt_train_tokens']) + len(ids))
        if score is None or score.get('reply_onsets') is None:
            continue  # fail closed on missing reward provenance
        if score.get('sources') != row.get('sources'):
            raise ValueError('Score/trace sources disagree')
        prior = scores.get(int(row['bptt_train_tokens']))
        if row_index == 0 or prior is None or prior.get('reply_onsets') is None:
            continue  # input source/reward at the initial bridge must be observed
        chunks.append({'previous_token': ids[0], 'targets': targets,
                       'recorded_speech_decisions': row.get('speech_decisions', []),
                       'last_voice': rows[row_index - 1]['speakers'][-1] == 'mother',
                       'last_reward': bool(prior['reply_onsets'][-1]),
                       'forced': {'pace': row['pace'], 'intervals': lengths, 'quiet_ranges': quiet,
                                  'speakers': speakers, 'sources': row.get('sources'),
                                  'reply_onsets': score['reply_onsets'], 'decisions': []},
                       'trained': row.get('bptt_train_tokens'),
                       'scope': 'historical_clock_replay_from_current_saved_state'})
    # Include a contiguous recent stretch, preserving cross-window bridges.
    chosen, count = [], 0
    for chunk in reversed(chunks):
        chosen.append(chunk); count += len(chunk['targets'])
        if count >= limit:
            break
    chosen = list(reversed(chosen))
    for left, right in zip(chosen, chosen[1:]):
        if (left['trained'] + len(left['targets']) != right['trained']
                or left['targets'][-1] != right['previous_token']):
            raise ValueError('Historical replay must preserve every recorded bridge, without skipped windows')
    return chosen


def fork_learner(learner, state, muscles, chunk):
    result = copy.copy(learner)
    result.state, result.ai_g = clone_state(state), muscles.detach().clone()
    result.previous_token = int(chunk['previous_token'])
    result.last_voice = bool(chunk.get('last_voice', learner.last_voice))
    result.last_reward = bool(chunk.get('last_reward', learner.last_reward))
    result.forced_world = copy.deepcopy(chunk['forced'])
    result.familiarity = copy.deepcopy(getattr(learner, 'familiarity', None))
    if getattr(learner, 'pace_rng', None) is not None:
        result.pace_rng = copy.deepcopy(learner.pace_rng)
    result.pending_speak_credit = None
    return result


@contextmanager
def circuit_intervention(model, state, groups, arm):
    """Hard cell removal acts before dopamine/compartment updates in this tick.

    Zero membrane and all delayed release slots at branch entry; on each native
    COBA finish zero target membrane, spike and new transmitted release. Incoming
    conductance/adaptation remain dynamic. This removes both transmission and
    direct membrane read contributions, not merely a hyperpolarizing current.
    """
    original_finish, original_tick = model.finish_coba_tick, fl.step_fly_physical_tick
    original_terminal = fl.kenyon_terminal_release
    terminal_pre = getattr(model, 'kcd_pre', None)
    if arm == 'kc_terminal_on':
        candidate = getattr(model, '_audit_terminal_pre', None)
        if candidate is None:
            raise ValueError('Terminal feedback intervention was not explicitly prepared')
        model.kcd_pre = candidate
    elif arm == 'kc_terminal_off':
        if terminal_pre is None:
            raise ValueError('Terminal feedback must already be enabled for this intervention')
        model.kcd_pre = None
    cells = groups.get(arm.removesuffix('_off')) if arm.endswith('_off') else None
    frozen_phi = state.comp_memory.detach().clone()
    activity = {key: [] for key in groups}
    phi_updates, raw_target, residual, terminal_release, mbon_phasic = [], [], [], [], []
    if cells is not None:
        state.h[:, cells] = 0
        for ring in state.ring:
            ring[:, cells] = 0

    def finish(context, drive, **kwargs):
        result = list(original_finish(context, drive, **kwargs))
        if cells is not None:
            raw_target.append((result[2][0][:, cells] > 0).float().mean().detach())
            result[0] = result[0].clone(); result[0][:, cells] = 0
            result[1] = result[1].clone(); result[1][:, cells] = 0
            current = result[2][0].clone(); current[:, cells] = 0
            result[2] = (current, *result[2][1:])
        return tuple(result)

    def tick(m, current, *args, **kwargs):
        before_memory = current.comp_memory
        if arm in ('phi_zero', 'phi_frozen'):
            phi = torch.zeros_like(frozen_phi) if arm == 'phi_zero' else frozen_phi
            current = fl.FlyPhysicalState(**{**current.state_dict(), 'comp_memory': phi})
        result = original_tick(m, current, *args, **kwargs)
        if arm in ('phi_zero', 'phi_frozen'):
            result = fl.FlyPhysicalState(**{**result.state_dict(), 'comp_memory': phi})
        phi_updates.append((result.comp_memory - before_memory).square().mean().sqrt().detach()
                           if result.comp_memory.numel() else result.h.new_zeros(()))
        release = result.ring[0] > 0
        if result.dopamine.numel() and 'MBON' in groups:
            d = result.dopamine[:, groups['MBON']]
            mbon_phasic.append((d[0] - d[1]).abs().mean().detach())
        for key, indices in groups.items():
            activity[key].append(release[:, indices].float().mean().detach())
        if cells is not None:
            residual.append(release[:, cells].float().mean().detach())
        return result

    def terminal(m, spikes):
        delivered = original_terminal(m, spikes)
        terminal_release.append(delivered[:, groups['MBON']].mean().detach())
        return delivered

    model.finish_coba_tick, fl.step_fly_physical_tick = finish, tick
    fl.kenyon_terminal_release = terminal
    try:
        yield {'activity': activity, 'phi_updates': phi_updates,
               'raw_target_release': raw_target, 'clamped_target_release': residual,
               'terminal_release_mean': terminal_release, 'mbon_phasic_abs_mean': mbon_phasic}
    finally:
        model.finish_coba_tick, fl.step_fly_physical_tick = original_finish, original_tick
        fl.kenyon_terminal_release = original_terminal
        if arm in ('kc_terminal_on', 'kc_terminal_off'):
            model.kcd_pre = terminal_pre


def run_native(learner, state, muscles, chunk, groups, arm='intact', instrument=True):
    branch = fork_learner(learner, state, muscles, chunk)
    inputs = [chunk['previous_token']] + chunk['targets'][:-1]
    input_ids = torch.tensor([inputs], device=state.h.device)
    targets = torch.tensor([chunk['targets']], device=state.h.device)
    if instrument:
        with circuit_intervention(branch.model, branch.state, groups, arm) as observation:
            scores, next_state, features = branch.forward_window(input_ids, targets)
    else:
        scores, next_state, features = branch.forward_window(input_ids, targets)
        observation = {}
    activity = {k: float(torch.stack(v).mean()) for k, v in observation.get('activity', {}).items() if v}
    diagnostic = {k: float(torch.stack(observation[k]).mean()) if observation.get(k) else None
                  for k in ('phi_updates', 'raw_target_release', 'clamped_target_release',
                            'terminal_release_mean', 'mbon_phasic_abs_mean')}
    return {'state': next_state, 'muscles': branch.ai_g, 'scores': scores.detach().cpu(),
            'features': features.detach().cpu(), 'trace': branch.last_ai_trace,
            'activity': activity, **diagnostic,
            'last_voice': branch.last_voice, 'last_reward': branch.last_reward}


def decision_hazard_stats(base_hazard, changed_hazard, decisions):
    """Condition acute hazard contrasts on actual recorded child decision ticks."""
    a, b = np.asarray(base_hazard), np.asarray(changed_hazard)
    if a.shape != b.shape:
        raise ValueError('Paired hazard clocks differ')
    mask = np.zeros(len(a), dtype=bool)
    for _, start, length, _ in decisions:
        start, length = int(start), int(length)
        if start < 0 or length < 1 or start + length > len(a):
            raise ValueError('Recorded child decision is outside the replay clock')
        mask[start:start + length] = True
    if not mask.any():
        return {'ticks': 0, 'intact_mean': None, 'changed_mean': None,
                'delta_mean': None, 'delta_abs_mean': None}
    delta = b[mask] - a[mask]
    return {'ticks': int(mask.sum()), 'intact_mean': float(a[mask].mean()),
            'changed_mean': float(b[mask].mean()), 'delta_mean': float(delta.mean()),
            'delta_abs_mean': float(np.abs(delta).mean())}


def compare_runs(base, changed):
    hazard_a, hazard_b = (np.asarray(r['trace']['hazard']) for r in (base, changed))
    eye_a, eye_b = (np.asarray(r['trace']['eye_tick']) for r in (base, changed))
    return {'ce_delta_per_target': (changed['scores'] - base['scores']).tolist(),
            'ce_delta_mean': float((changed['scores'] - base['scores']).mean()),
            'read_contrast_norm': float((changed['features'] - base['features']).norm(dim=-1).mean()),
            'hazard_delta_mean': float((hazard_b - hazard_a).mean()),
            'hazard_delta_abs_mean': float(np.abs(hazard_b - hazard_a).mean()),
            'eye_delta_abs_mean': float(np.abs(eye_b - eye_a).mean()),
            'muscle_maxabs': float((changed['muscles'] - base['muscles']).abs().max()),
            'state_maxabs': state_difference(base['state'], changed['state']),
            'state_contrast_rms': {k: float((getattr(changed['state'], k) - getattr(base['state'], k))
                                          .square().mean().sqrt())
                                   for k in ('h', 'dopamine', 'comp_memory')},
            'activity_delta': {k: changed['activity'][k] - base['activity'][k]
                               for k in changed['activity'] if k in base['activity']},
            'activity': changed['activity'],
            **{k: changed[k] for k in ('phi_updates', 'raw_target_release', 'clamped_target_release',
                                       'terminal_release_mean', 'mbon_phasic_abs_mean')}}


def ensure_gpu_idle(desktop_pids=()):
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                             '--format=csv,noheader'], capture_output=True, text=True, check=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if lines:
        # WDDM can list idle desktop apps as compute clients with N/A memory.
        # Only exact PIDs inspected by the caller are eligible; all unknown
        # clients still block. Aggregate memory/utilization must confirm idle.
        allowed = set(desktop_pids)
        fields = [[p.strip() for p in line.split(',')] for line in lines]
        if allowed and all(len(p) == 3 and p[0].isdigit() and int(p[0]) in allowed
                           and p[2] in ('[N/A]', 'N/A') for p in fields):
            load = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, check=True)
            rows = [r.split(',') for r in load.stdout.splitlines() if r.strip()]
            if rows and all(float(r[0]) <= 128 and float(r[1]) == 0 for r in rows):
                return {'idle_desktop_pids': sorted(allowed), 'gpu_load': load.stdout.strip()}
        raise RuntimeError('GPU has another compute process; audit must wait: ' + '\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--mode', choices=('pending', 'history'), default='pending')
    parser.add_argument('--arms', default='KC_off,MBON_off,DAN_off,phi_zero,phi_frozen')
    parser.add_argument('--terminal-release-gain', type=float, default=None,
                        help='Explicit existing KC->DAN terminal gain for the kc_terminal_on arm')
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error('--tokens must be positive')
    checkpoint = args.checkpoint.resolve()
    run = args.run.resolve()
    if args.output.resolve().is_relative_to(run) or args.output.resolve().is_relative_to(checkpoint.parent):
        parser.error('Use an independent report directory, outside the paused run/checkpoint directories')
    ensure_gpu_idle()
    args.output.mkdir(parents=True, exist_ok=True)
    paused = run / 'caregiver_presence.json'
    source_stat, pause_bytes = checkpoint.stat(), paused.read_bytes() if paused.exists() else None
    checksum = sha256(checkpoint)
    runtime_paths = (ROOT / 'scripts/ib/train_fly_bptt_stream.py',
                     ROOT / 'information_boltzmann/core/fly_bptt_learning.py',
                     ROOT / 'information_boltzmann/core/fly_active_inference.py',
                     ROOT / 'information_boltzmann/core/fly_mother_world.py',
                     ROOT / 'information_boltzmann/core/fly_carried_credit.py',
                     ROOT / 'information_boltzmann/core/triton_synapse.py',
                     ROOT / 'information_boltzmann/core/fly_reservoir.py', Path(__file__))
    runtime_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in runtime_paths}
    started = time.monotonic()
    with torch.no_grad():
        learner, saved, restored_args, mode_changed = restore_native(checkpoint, args.output)
        if mode_changed:
            raise RuntimeError('Native restore requests a learning-mode migration; reject before auditing')
        model = learner.model
        restored_difference = state_difference(learner.state, fl.FlyPhysicalState(**{
            key: tuple(t.to('cuda') for t in value) if key == 'ring' else value.to('cuda')
            for key, value in saved['learner']['physical'].items()}))
        if max(restored_difference.values()) != 0 or not torch.equal(learner.ai_g.cpu(), saved['learner']['ai_g']):
            raise RuntimeError('Native restore changed saved physical state or ai_g')
        missing_model_keys = set(saved['model']) - set(model.state_dict())
        if missing_model_keys:
            raise RuntimeError('Native restore dropped saved model tensors: ' + str(sorted(missing_model_keys)))
        for key, value in saved['model'].items():
            if key in model.state_dict() and not torch.equal(model.state_dict()[key].detach().cpu(), value):
                raise RuntimeError('Native restored weight differs: ' + key)
        arms = args.arms.split(',')
        terminal_candidate = None
        if 'kc_terminal_on' in arms:
            if (getattr(model, 'kcd_pre', None) is not None
                    or args.terminal_release_gain is None
                    or not math.isfinite(args.terminal_release_gain)
                    or args.terminal_release_gain <= 0):
                raise ValueError('Use an absent terminal route and an explicit positive finite release gain')
            edges, cells = fl.attach_kenyon_terminal_release(
                model, restored_args.graph, restored_args.graph.parent / 'body-annotations.feather',
                gain=args.terminal_release_gain)
            model.log_kc_release.requires_grad_(False)
            model._audit_terminal_pre = model.kcd_pre
            model.kcd_pre = None  # Intact/sham keep the original route disabled.
            terminal_candidate = {'gain': args.terminal_release_gain, 'edges': edges, 'dan_cells': cells,
                                  'scope': 'acute delivery of existing route, not calibrated learning benefit'}
        graph = np.load(restored_args.graph, allow_pickle=True)
        import pandas as pd
        annotation = pd.read_feather(restored_args.graph.parent / 'body-annotations.feather')
        annotation = annotation.set_index('bodyId').reindex(graph['neuron_body_ids'])
        groups = {key: torch.as_tensor(indices, dtype=torch.long, device='cuda') for key, indices in {
            'KC': np.flatnonzero(annotation['class'].astype(str).values == 'Kenyon_Cell'),
            'MBON': np.flatnonzero(annotation['class'].astype(str).values == 'MBON'),
            'DAN': graph['dan_indices'], 'output': model.read_indices.cpu().numpy()}.items()}
        chunks = (pending_queue_tape(saved, args.tokens, int(saved['config']['window'])) if args.mode == 'pending'
                  else historical_tape(run / 'ai_trace.jsonl', args.tokens, int(saved['bptt_train_tokens'])))
        if not chunks:
            raise ValueError('No recoverable native events')
        if any(a not in ('KC_off', 'MBON_off', 'DAN_off', 'phi_zero', 'phi_frozen',
                        'kc_terminal_on', 'kc_terminal_off') for a in arms):
            raise ValueError('Unknown circuit intervention')
        report = {'checkpoint': str(checkpoint), 'checkpoint_sha256': checksum,
                  'trained_tokens': int(saved['bptt_train_tokens']), 'mode': args.mode,
                  'scope': chunks[0]['scope'], 'training_updates': 0, 'network_calls': 0,
                  'frozen_parameters': True, 'local_physical_plasticity': 'continues except named phi intervention',
                  'native_forward': 'FlyBPTTLearner.forward_window',
                  'loader': 'current trainer resume AST through interaction/ai_g restore, before migration/world',
                  'eye_gate': model.ai_eye_gate, 'initial_ai_g_norm': float(learner.ai_g.norm()),
                  'configuration': {k: saved['config'].get(k) for k in (
                      'time_scale', 'hold_input', 'stream_pace_min', 'stream_pace_mean', 'window',
                      'writer_baseline_clock', 'writer', 'transmission_mode', 'use_stp', 'graded_types',
                      'mother_voice', 'mother_reward', 'kc_dan_release', 'mother_native_credit')},
                  'groups': {k: int(v.numel()) for k, v in groups.items()},
                  'restored_state_maxabs': restored_difference,
                  'intervention_contract': 'hard zero target h and delayed/new release; retain incoming conductance/adaptation',
                  'pairing': 'every window starts all arms from the intact continuation state; effects are acute within window',
                  'policy_credit': 'weights frozen; carried physical sensitivities retained; loss-only SIGReg excluded',
                  'runtime_sha256': runtime_hashes,
                  'terminal_candidate': terminal_candidate,
                  'windows': []}
        # All branches see identical CPU/CUDA random draws. Native motor commits
        # do not reset g; any future behavior change that breaks this must fail.
        first = chunks[0]
        cpu_rng, cuda_rng = saved['rng_cpu'], saved['rng_cuda']
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
        reference = run_native(learner, learner.state, learner.ai_g, first, groups, instrument=False)
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
        native_repeat = run_native(learner, learner.state, learner.ai_g, first, groups, instrument=False)
        report['native_repeat_state_maxabs'] = state_difference(reference['state'], native_repeat['state'])
        if not states_close(reference['state'], native_repeat['state']):
            (args.output / 'native_repeat_failure.json').write_text(json.dumps(
                compare_runs(reference, native_repeat), indent=2), encoding='utf-8')
            raise RuntimeError('Native uninstrumented replay itself fails numerical equivalence')
        del native_repeat
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
        identity = run_native(learner, learner.state, learner.ai_g, first, groups)
        gate = compare_runs(reference, identity)
        if (not states_close(reference['state'], identity['state']) or gate['read_contrast_norm'] != 0
                or gate['ce_delta_mean'] != 0 or gate['hazard_delta_abs_mean'] != 0 or gate['muscle_maxabs'] != 0):
            (args.output / 'identity_failure.json').write_text(json.dumps(gate, indent=2), encoding='utf-8')
            raise RuntimeError('Instrumented intact path differs from native learner')
        report['identity_gate'] = gate
        report['identity_tolerance'] = {'state_rtol': 1e-5, 'state_atol': 1e-6,
                                        'read_ce_hazard_muscle': 'exact equality required'}
        del reference, identity
        state, muscles = clone_state(learner.state), learner.ai_g.detach().clone()
        for i, chunk in enumerate(chunks):
            torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
            base = run_native(learner, state, muscles, chunk, groups)
            next_cpu, next_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
            row = {'index': i, 'tape': chunk, 'intact_ce': base['scores'].tolist(),
                   'intact_activity': base['activity'], 'intact_hazard_mean': float(np.mean(base['trace']['hazard'])),
                   'intact_phi_update_rms_per_tick': base['phi_updates'], 'arms': {}}
            torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
            sham = run_native(learner, state, muscles, chunk, groups)
            row['sham_numerical_floor'] = compare_runs(base, sham)
            if (not states_close(base['state'], sham['state'])
                    or not torch.equal(base['scores'], sham['scores'])
                    or row['sham_numerical_floor']['hazard_delta_abs_mean'] != 0):
                (args.output / 'window_sham_failure.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
                raise RuntimeError(f'Window {i}: native sham replay differs in functional readout')
            del sham
            for arm in arms:
                torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
                changed = run_native(learner, state, muscles, chunk, groups, arm)
                row['arms'][arm] = compare_runs(base, changed)
                row['arms'][arm]['recorded_child_decision_hazard'] = decision_hazard_stats(
                    base['trace']['hazard'], changed['trace']['hazard'],
                    chunk.get('recorded_speech_decisions', []))
                del changed
            state, muscles = base['state'], base['muscles']
            learner.last_voice, learner.last_reward = base['last_voice'], base['last_reward']
            cpu_rng, cuda_rng = next_cpu, next_cuda
            report['windows'].append(row)
            print(json.dumps({'window': i, 'tokens': len(chunk['targets']), 'ticks': sum(chunk['forced']['intervals']),
                              'ce': float(base['scores'].mean()),
                              'arm_ce_deltas': {k: v['ce_delta_mean'] for k, v in row['arms'].items()}}), flush=True)
        report['summary'] = {arm: {
            'mean_ce_delta': float(np.mean([x for r in report['windows'] for x in r['arms'][arm]['ce_delta_per_target']])),
            'mean_absolute_hazard_delta': float(np.mean([r['arms'][arm]['hazard_delta_abs_mean'] for r in report['windows']])),
            'max_clamp_residual': max((r['arms'][arm]['clamped_target_release'] or 0) for r in report['windows'])}
            for arm in arms}
        for arm in arms:
            decisions = [r['arms'][arm]['recorded_child_decision_hazard'] for r in report['windows']]
            count = sum(d['ticks'] for d in decisions)
            report['summary'][arm].update(
                recorded_child_decision_ticks=count,
                mean_child_decision_hazard_delta=(sum(d['delta_mean'] * d['ticks'] for d in decisions if d['ticks']) / count
                                                  if count else None),
                mean_absolute_child_decision_hazard_delta=(sum(d['delta_abs_mean'] * d['ticks'] for d in decisions if d['ticks']) / count
                                                           if count else None))
        report['source_counts'] = {source: sum(c['forced']['sources'].count(source) for c in chunks)
                                    for source in sorted({s for c in chunks for s in c['forced']['sources']})}
        report['ce_delta_by_source'] = {arm: {
            source: float(np.mean([delta for row in report['windows']
                                   for delta, label in zip(row['arms'][arm]['ce_delta_per_target'],
                                                           row['tape']['forced']['sources']) if label == source]))
            for source in report['source_counts']} for arm in arms}
        report['tokens'], report['physical_ticks'] = (sum(len(c['targets']) for c in chunks),
                                                       sum(sum(c['forced']['intervals']) for c in chunks))
        report['quiet_ticks'] = sum(b - a for c in chunks for ranges in c['forced']['quiet_ranges'] for a, b in ranges)
        report['vram_peak_mib'] = torch.cuda.max_memory_allocated() / 1024**2
        report['elapsed_seconds'] = time.monotonic() - started
        for key, value in saved['model'].items():
            if key in model.state_dict() and not torch.equal(model.state_dict()[key].detach().cpu(), value):
                raise RuntimeError('Audit changed a frozen model tensor: ' + key)
        report['model_tensors_unchanged'] = True
    after = checkpoint.stat()
    if (source_stat.st_size, source_stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Source checkpoint changed during read-only audit')
    if pause_bytes is not None and paused.read_bytes() != pause_bytes:
        raise RuntimeError('Caregiver pause file changed during audit')
    report['checkpoint_unchanged'] = sha256(checkpoint) == checksum
    report['caregiver_pause_unchanged'] = True
    report['runtime_unchanged'] = all(sha256(p) == runtime_hashes[str(p.relative_to(ROOT))] for p in runtime_paths)
    if not report['runtime_unchanged']:
        (args.output / 'invalid_runtime_changed.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        raise RuntimeError('Runtime sources changed during audit; results must be rerun on a fixed version')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'summary': report['summary'], 'output': str(args.output / 'report.json')}), flush=True)


if __name__ == '__main__':
    main()
