"""Unified high-throughput MengLi life runner with existing DSH Mother & biofeedback mirror.

Combines:
  - Champion H-Net 4+24+4 + PhaseSpaceBoltzmann + RootLatentPlanning +
    LearnableAlifRhythm + 1600B calm flow (~350+ B/s, ~2.44 GiB peak VRAM),
  - Single UnifiedWorldPort (1 UTF-8 body boundary, zero extra body heads),
  - Existing DSH Mother (`FamilyCaregiver` + `DshMother` on `127.0.0.1:8902`)
    with adaptive developmental parenting plan, third-person bodily mirroring,
    `mirror_visible` / `book_only` perspective alternation, live human/guest
    inbox preemption, optional Chinese sidecar storybook stream, and
    cache-hit + compaction telemetry.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')

import numpy as np
import torch

PROJECT = Path('D:/information-boltzmann-v1').resolve()
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.run_alif_executive_battle import (
    CLASSIC_TEXT_SPEC,
    FRESH_FACTS,
    FRESH_TEXT_SPEC,
    SPOOL,
    evaluate_checkpoint,
    load_strict_engine,
    sha256_file,
    write_json,
)


DEFAULT_CHAMPION = Path('D:/mengli-alif-hnet-1600b-flux-30min-20261124/body/last.pt.gz').resolve()
DEFAULT_HARNESS_URL = 'http://127.0.0.1:8902'


def _restore_unified_state_if_present(port, base_world, *, skip_dsh_session: bool = False) -> bool:
    """Restore any inherited DSH/unified state carried inside `base_world.retained`.

    When `skip_dsh_session=True` (`--fresh-mother-session`), restores the port's
    storybook cursor and developmental plan while starting a clean DSH Mother
    conversation session (breaking any accumulated long-context opening lock-in).
    """
    retained = getattr(base_world, 'retained', None)
    if not isinstance(retained, dict):
        return False
    has_unified = 'mother_unified_port' in retained or 'mother_provider_state' in retained
    if not has_unified:
        return False
    merged = base_world.state_dict()
    merged.update(retained)
    if skip_dsh_session:
        merged.pop('mother_provider_state', None)
        merged.pop('mother_transcript', None)
    port.load_state_dict(merged)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_CHAMPION)
    parser.add_argument('--work', type=Path, default=Path('D:/mengli-unified-life-20261124'))
    parser.add_argument('--temp-root', type=Path, default=Path('D:/mengli-unified-life-tmp'))
    parser.add_argument('--report', type=Path, default=PROJECT / 'results/published/cgb_unified_life_20261124.json')
    parser.add_argument('--budget-seconds', type=float, default=600.0)
    parser.add_argument('--max-mother-turns', type=int, default=0,
                        help='If >0, stop cleanly once this many DSH Mother turns complete (great for smoke tests)')
    parser.add_argument('--eval-every-seconds', type=float, default=0.0)
    parser.add_argument('--save-every-seconds', type=float, default=300.0)

    # Existing DSH Mother Umwelt bridge & Membrane scheduling (NOT a second Mother).
    parser.add_argument('--umwelt-bridge-url', '--mother-harness-url', dest='mother_harness_url',
                        type=str, default=DEFAULT_HARNESS_URL,
                        help='Loopback URL of MengLi Membrane Bridge (default http://127.0.0.1:8902)')
    parser.add_argument('--branch-id', type=str, default=None)
    parser.add_argument('--no-mother', action='store_true', help='Disable DSH Mother (local storybook + inbox/tools only)')
    parser.add_argument('--mother-cooldown', type=int, default=3,
                        help='Min executive spikes between Mother turns when MengLi speaks')
    parser.add_argument('--mother-max-gap', type=int, default=8,
                        help='Max executive spikes before Mother checks in even during quiet reading')
    parser.add_argument('--mirror-ratio', type=float, default=0.7,
                        help='Fraction of Mother turns with bodily mirror visible (rest are book_only)')
    parser.add_argument('--accompany-minutes', type=float, default=0.0,
                        help='Optional natural day-rhythm: minutes Mother stays in co-reading before stepping away (0=always present)')
    parser.add_argument('--solo-minutes', type=float, default=0.0,
                        help='Optional natural day-rhythm: minutes of solo book-navigation reading when Mother steps away')
    parser.add_argument('--book-floor-default', action='store_true',
                        help='Optional Policy B: route unaddressed speech on Book Floor to book navigation even while Mother is present')
    parser.add_argument('--no-gaze-pointer', action='store_true',
                        help='Disable biological O(1) GazeObjectPointer (fall back to host presence/floor routing)')
    parser.add_argument('--care-plan', type=str, default=None,
                        help='Optional explicit user parenting plan overriding adaptive stage plans')
    parser.add_argument('--context-tokens', type=int, default=524288,
                        choices=[8192, 32768, 65536, 131072, 262144, 524288],
                        help='DSH Mother context window budget (default 524288 -> compacts at ~419K via official DSH 0.80/0.16 policy)')
    parser.add_argument('--fresh-mother-session', action='store_true',
                        help='Start a fresh Mother session at --context-tokens & --mother-policy instead of inheriting a legacy 8K session ref')
    parser.add_argument('--mother-policy', type=str, default='umwelt_mother_v1',
                        choices=['umwelt_mother_v1', 'responsive_v3'],
                        help='Mother constitutional system persona for fresh sessions (default umwelt_mother_v1)')
    parser.add_argument('--reflect-every-turns', type=int, default=16,
                        help='Trigger periodic Five-Element Mother Agent reflection & plan review every N Mother turns (0=off)')
    parser.add_argument('--mother-notes', type=Path, default=PROJECT / 'data/harness/mother_notes.md',
                        help='Markdown growth diary where Mother appends periodic MengLi-model reflections')
    parser.add_argument('--inbox-file', type=Path, default=None,
                        help='Optional append-only JSONL inbox for live parent/guest messages during the run')
    parser.add_argument('--chinese-corpus', type=Path, default=None,
                        help='Optional UTF-8 Chinese natural-text storybook file interleaved on calm reading')
    parser.add_argument('--chinese-mix-ratio', type=float, default=0.35,
                        help='Fraction of calm reading pieces drawn from --chinese-corpus when provided')
    parser.add_argument('--chinese-val-file', type=Path, default=None,
                        help='Optional held-out Chinese validation.bin for tracking 8192B Chinese NLL flux')

    # Champion H-Net + ALIF + PSB + RLP defaults.
    parser.add_argument('--piece-bytes', type=int, default=1600)
    parser.add_argument('--gpu-act-mb', type=int, default=768,
                        help='Tiered GPU activation budget in MiB (768 MiB leaves safe WDDM/commit headroom on 4 GiB GPUs)')
    parser.add_argument('--calm-lr-scale', type=float, default=-1.0)
    parser.add_argument('--extra-hippocampal-steps', type=int, default=-1)
    parser.add_argument('--hnet-target-ratio', type=float, default=4.0)
    parser.add_argument('--hnet-ratio-weight', type=float, default=0.25)
    parser.add_argument('--skip-final-eval', action='store_true')
    parser.add_argument('--bind-dashboard', action='store_true',
                        help='Bind this live run to the MengLi HTML terminal (data/harness/workbench/training-observer.json)')
    parser.add_argument('--save-on-mother-turn', action='store_true',
                        help='Also publish checkpoint + life_card after each Mother/Parent turn so the HTML terminal updates live')
    args = parser.parse_args()

    spool_dir = (args.temp_root.resolve() / 'spool') if args.temp_root is not None else SPOOL
    os.environ['CGB_CHECKPOINT_SPOOL'] = str(spool_dir)
    spool_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    work = args.work.resolve()
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    temp_dir = (args.temp_root.resolve() / 'run_tmp') if args.temp_root is not None else (work / 'temporary')
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    source = args.source.resolve()
    source_sha = sha256_file(source)
    print(f'[UNIFIED] Loading champion source {source.name} (sha={source_sha[:12]})...', flush=True)
    loaded = load_strict_engine(source, work, temp_dir=temp_dir, warm_runtime=True)
    engine, config = loaded.engine, loaded.config
    cycle = int(loaded.cycle)

    from scripts.ib_local import alif_executive as alif
    from scripts.ib_local import gaze_pointer as gaze_mod
    from scripts.ib_local import hnet_chunking as hnet_mod
    from scripts.ib_local import learnable_alif_rhythm as lar
    from scripts.ib_local import phase_space_boltzmann as psb
    from scripts.ib_local import root_latent_planning as rlp
    from scripts.ib_local.mengli_harness.family_provider import FamilyCaregiver
    from scripts.ib_local.mengli_harness.mengli_world_client import (
        DEFAULT_SETTINGS, SETTINGS_32K, SETTINGS_64K, SETTINGS_128K, SETTINGS_256K, SETTINGS_512K,
        DshMother, branch_for_output,
    )
    from scripts.ib_local.mother_checkpoint import save as save_ckpt
    from scripts.ib_local.unified_world_port import UnifiedWorldPort
    from scripts.native_policy_activation_storage import activation_storage, policy_storage

    calm_lr_scale = math.sqrt(args.piece_bytes / 640.0) if args.calm_lr_scale < 0 else float(args.calm_lr_scale)
    extra_h_steps = max(0, int(round(args.piece_bytes / 640.0)) - 1) if args.extra_hippocampal_steps < 0 else int(args.extra_hippocampal_steps)

    alif.configure(
        engine, config,
        piece_bytes=args.piece_bytes,
        calm_lr_scale=calm_lr_scale,
        extra_hippocampal_steps=extra_h_steps,
    )
    lar.configure(engine, config)
    psb.configure(engine, config, initialize=not psb.enabled(engine))
    rlp.configure(engine, config)
    hnet_mod.configure(
        engine, config,
        enc_layers=4, dec_layers=4,
        target_ratio=args.hnet_target_ratio,
        ratio_loss_weight=args.hnet_ratio_weight,
    )
    if not args.no_gaze_pointer:
        gaze_mod.configure(engine, config)

    base_world = engine.world
    retained = getattr(base_world, 'retained', None) or {}
    saved_dsh_settings = ((retained.get('mother_provider_state') or {}).get('dsh_harness') or {}).get('settings')
    settings_map = {
        8192: DEFAULT_SETTINGS,
        32768: SETTINGS_32K,
        65536: SETTINGS_64K,
        131072: SETTINGS_128K,
        262144: SETTINGS_256K,
        524288: SETTINGS_512K,
    }
    chosen_settings = dict(settings_map[int(args.context_tokens)])
    saved_dsh_policy = (retained.get('mother_provider_state') or {}).get('policy')
    chosen_policy = args.mother_policy
    # If resuming an older session without --fresh-mother-session, match its stored DSH settings & policy,
    # while permitting a clean upward context_tokens expansion (e.g. 256K -> 512K) on the new branch.
    if not args.fresh_mother_session and isinstance(saved_dsh_settings, dict):
        merged_settings = dict(saved_dsh_settings)
        req_ctx = int(args.context_tokens)
        if (req_ctx > int(saved_dsh_settings.get('context_tokens', 0))
                and {k: v for k, v in saved_dsh_settings.items() if k != 'context_tokens'}
                == {k: v for k, v in chosen_settings.items() if k != 'context_tokens'}):
            merged_settings['context_tokens'] = req_ctx
        chosen_settings = merged_settings
        if isinstance(saved_dsh_policy, str) and saved_dsh_policy:
            chosen_policy = saved_dsh_policy

    caregiver = None
    branch_id = args.branch_id or branch_for_output(work / 'body')
    if not args.no_mother:
        dsh = DshMother(
            endpoint_url=args.mother_harness_url,
            branch_id=branch_id,
            policy=chosen_policy,
            settings=chosen_settings,
            max_retries=3,
            provenance=dict(
                source_checkpoint=str(source),
                source_sha256=source_sha,
                source_cycle=int(cycle),
                mode='mengli_umwelt_v2',
                context_tokens=int(chosen_settings['context_tokens']),
            ),
        )
        caregiver = FamilyCaregiver(dsh)

    port = UnifiedWorldPort(
        base_world,
        caregiver=caregiver,
        mother_cooldown=args.mother_cooldown,
        mother_max_gap=args.mother_max_gap,
        mirror_ratio=args.mirror_ratio,
        auto_commit=False,
        inbox_path=args.inbox_file,
        sidecar_corpus=args.chinese_corpus,
        sidecar_mix_ratio=(args.chinese_mix_ratio if args.chinese_corpus is not None else 0.0),
        user_plan_text=args.care_plan,
        respond_on_child_speech=(not args.book_floor_default),
    )
    restored_prior = _restore_unified_state_if_present(
        port, base_world, skip_dsh_session=args.fresh_mother_session,
    )
    port.attach_engine(engine)
    engine.world = port
    config['mother_mode'] = 'real'
    if not args.no_mother:
        config['mother_context_harness'] = dict(
            schema=2,
            kind='dsh',
            branch_id=branch_id,
            endpoint_url=args.mother_harness_url,
            settings=dict(caregiver.automatic.settings),
        )
    config['unified_world_port'] = dict(
        schema=1,
        branch_id=branch_id,
        mother_enabled=(not args.no_mother),
        mother_harness_url=(None if args.no_mother else args.mother_harness_url),
        mother_cooldown=int(args.mother_cooldown),
        mother_max_gap=int(args.mother_max_gap),
        mirror_ratio=float(args.mirror_ratio),
        chinese_corpus=(None if args.chinese_corpus is None else str(args.chinese_corpus.resolve())),
        chinese_mix_ratio=float(port.sidecar_mix_ratio),
        restored_prior_port_state=restored_prior,
    )

    # Activation storage hooks identical to champion runner (bounded GPU & host commit for 4 GiB Windows WDDM).
    activation_options = dict(
        mode='tiered',
        gpu_budget_bytes=int(args.gpu_act_mb) * 1024 * 1024,
        host_budget_bytes=512 * 1024 * 1024,
    )
    if getattr(engine, 'native_replay_batch_size', 0) > 4:
        engine.native_replay_batch_size = 4
    world_act_dir = temp_dir / 'world'
    policy_act_dir = temp_dir / 'policy'
    original_policy = engine._policy_grad
    had_policy_attr = '_policy_grad' in vars(engine)
    original_objective = engine.model.objective
    had_obj_attr = 'objective' in vars(engine.model)
    hooks = [None]

    def wrapped_policy(*a, **kw):
        with policy_storage(engine.params, policy_act_dir, **activation_options):
            return original_policy(*a, **kw)

    def wrapped_objective(*a, **kw):
        if hooks[0] is not None:
            with hooks[0]():
                return original_objective(*a, **kw)
        return original_objective(*a, **kw)

    engine._policy_grad = wrapped_policy
    engine.model.objective = wrapped_objective

    start_cycle = int(cycle)
    start_ext_bytes = int(engine.alif_executive_state['total_external_bytes'])
    start_spikes = int(engine.alif_executive_state['executive_spikes'])
    start_calms = int(engine.alif_executive_state['calm_pieces'])
    start_mother_turns = int(port.mother_turns)

    def _run_flux_probe(life_s: float) -> dict:
        from scripts.ib_local.wood_policy_measurement import segments, score
        from scripts.run_native_causal_use_value import evaluate as eval_facts
        hnet = getattr(engine.model.body, 'hnet_chunker', None)
        saved_hnet = None
        if hnet is not None:
            saved_hnet = (
                int(hnet.calls), float(hnet.last_boundary_rate),
                float(hnet.last_compression_ratio), float(hnet.last_ratio_loss),
                float(hnet.mean_boundary_rate_ema), bool(hnet.calibrated),
            )
        fresh_segs = segments(FRESH_TEXT_SPEC)
        classic_segs = segments(CLASSIC_TEXT_SPEC)
        cn_res = None
        cn_segs = None
        if args.chinese_val_file is not None and args.chinese_val_file.exists():
            cn_path = args.chinese_val_file.resolve()
            cn_size = cn_path.stat().st_size
            step = max(2048, (cn_size - 2049) // 4)
            cn_spec = dict(
                path=str(cn_path),
                sha256=sha256_file(cn_path),
                starts=[0, step, 2 * step, 3 * step],
                bytes_per_segment=2048,
                total_scored_bytes=8192,
            )
            cn_segs = segments(cn_spec)
        facts_payload = torch.load(FRESH_FACTS, map_location=engine.device, weights_only=False)
        try:
            with torch.no_grad():
                fresh_res = score(engine, fresh_segs)
                classic_res = score(engine, classic_segs)
                if cn_segs is not None:
                    cn_res = score(engine, cn_segs)
                facts_res = eval_facts(engine, facts_payload)
        finally:
            del facts_payload
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
        if hnet is not None and saved_hnet is not None:
            hnet.calls.fill_(saved_hnet[0])
            hnet.last_boundary_rate.fill_(saved_hnet[1])
            hnet.last_compression_ratio.fill_(saved_hnet[2])
            hnet.last_ratio_loss.fill_(saved_hnet[3])
            hnet.mean_boundary_rate_ema.fill_(saved_hnet[4])
            hnet.calibrated.fill_(saved_hnet[5])
        comb = 0.5 * (fresh_res['mean_nll'] + classic_res['mean_nll'])
        pt = dict(
            life_seconds=round(life_s, 2),
            run_external_bytes=int(engine.alif_executive_state['total_external_bytes']) - start_ext_bytes,
            cycle=int(cycle),
            combined_16384_nll=float(comb),
            fresh_8192_nll=float(fresh_res['mean_nll']),
            classic_8192_nll=float(classic_res['mean_nll']),
            chinese_8192_nll=(None if cn_res is None else float(cn_res['mean_nll'])),
            fresh_64_facts_score=float(facts_res['mean_score']),
            fresh_64_facts_mse=float(facts_res['mean_mse']),
            mother_turns=int(port.mother_turns),
            provider_hit_rate=port.telemetry_summary().get('provider_hit_rate'),
        )
        cn_str = '' if cn_res is None else f' cn_8KB={cn_res["mean_nll"]:.4f}'
        print(f'  [FLUX @ {life_s:6.1f}s | +{pt["run_external_bytes"]:6d}B | mother={port.mother_turns}] '
              f'16KB_NLL={comb:.6f}{cn_str} 64facts={pt["fresh_64_facts_score"]:.6f} '
              f'mse={pt["fresh_64_facts_mse"]:.4f} hit={pt["provider_hit_rate"]}', flush=True)
        return pt

    flux_trajectory = []
    if args.eval_every_seconds > 0.0:
        flux_trajectory.append(_run_flux_probe(0.0))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    from scripts.ib_local.life_card import card_config_for, export_live, identity_for
    body_dir = work / 'body'
    body_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = body_dir / 'metrics.jsonl'
    guard_path = work / 'guard_status.json'
    life_card_path = None

    def _write_guard(status: str) -> None:
        if not args.bind_dashboard:
            return
        payload = dict(
            schema=1,
            status=status,
            pid=os.getpid(),
            child_pid=os.getpid(),
            updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        )
        tmp_g = guard_path.with_suffix('.tmp.json')
        tmp_g.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(tmp_g, guard_path)

    def _safe_json_value(v):
        if v is None or isinstance(v, (bool, int, str)):
            return v
        if isinstance(v, float):
            return v if math.isfinite(v) else None
        if isinstance(v, np.generic):
            return _safe_json_value(v.item())
        if isinstance(v, np.ndarray):
            return v.tolist() if v.size <= 64 else None
        if torch.is_tensor(v):
            return v.detach().cpu().tolist() if v.numel() <= 64 else None
        if isinstance(v, dict):
            return {str(k): _safe_json_value(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_safe_json_value(val) for val in v]
        return str(v)

    def _publish_live_checkpoint(last_row: dict | None = None) -> str:
        # Temporarily unwrap local closures so torch.save never tries to pickle them.
        cur_policy = getattr(engine, '_policy_grad', None)
        cur_obj = getattr(engine.model, 'objective', None)
        restore_policy = (cur_policy is wrapped_policy)
        restore_obj = (cur_obj is wrapped_objective)
        if restore_policy:
            if had_policy_attr:
                engine._policy_grad = original_policy
            elif '_policy_grad' in vars(engine):
                delattr(engine, '_policy_grad')
        if restore_obj:
            if had_obj_attr:
                engine.model.objective = original_objective
            elif 'objective' in vars(engine.model):
                delattr(engine.model, 'objective')
        try:
            ckpt_sha = save_ckpt(engine, body_dir, cycle, config)
        finally:
            if restore_policy:
                engine._policy_grad = wrapped_policy
            if restore_obj:
                engine.model.objective = wrapped_objective
        if args.bind_dashboard and life_card_path is not None:
            write_json(body_dir / 'config.json', _safe_json_value(config))
            export_live(
                engine,
                checkpoint=body_dir / 'last.pt.gz',
                config=config,
                cycle=cycle,
                row=last_row,
                output=life_card_path,
                branch_id=branch_id,
                committed_sha=ckpt_sha,
            )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ckpt_sha

    if args.bind_dashboard:
        ident = identity_for(config, source_sha)
        config['life_identity'] = ident
        lc_cfg = card_config_for(config, branch_id)
        config['life_card'] = lc_cfg
        life_card_path = Path(lc_cfg['path'])
        _write_guard('running')
        _publish_live_checkpoint(None)
        obs_path = PROJECT / 'data/harness/workbench/training-observer.json'
        obs_payload = dict(
            schema=1,
            run_id=branch_id,
            individual_id=ident['individual_id'],
            branch_id=branch_id,
            body=str(body_dir),
            life_card=str(life_card_path),
            guard_status=str(guard_path),
            segments=[str(body_dir)],
            storage_roots=[str(work)],
            inbox_file=(None if args.inbox_file is None else str(args.inbox_file.resolve())),
            start_cycle=int(start_cycle),
            target_cycle=None,
        )
        tmp_obs = obs_path.with_suffix('.tmp.json')
        tmp_obs.write_text(json.dumps(obs_payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(tmp_obs, obs_path)
        print(f'[UNIFIED] Bound live HTML dashboard -> {obs_path} (card={life_card_path.name})', flush=True)

    events_path = work / 'events.jsonl'
    dialogue_log = []
    t_start = time.perf_counter()
    pause_seconds = 0.0
    next_eval_at = float(args.eval_every_seconds) if args.eval_every_seconds > 0.0 else float('inf')
    next_save_at = float(args.save_every_seconds) if args.save_every_seconds > 0.0 else float('inf')
    calm_times, exec_times = [], []
    last_exec_row = None

    def _life_elapsed() -> float:
        return (time.perf_counter() - t_start) - pause_seconds

    try:
        with events_path.open('w', encoding='utf8') as ev_f, metrics_path.open('a', encoding='utf8') as met_f:
            while _life_elapsed() < float(args.budget_seconds):
                if args.max_mother_turns > 0 and (port.mother_turns - start_mother_turns) >= args.max_mother_turns:
                    break

                # Optional natural accompany / solo day-rhythm (clean `[在场]` Yoneda presence switching).
                if (not args.no_mother) and args.accompany_minutes > 0.0 and args.solo_minutes > 0.0:
                    acc_s = float(args.accompany_minutes) * 60.0
                    sol_s = float(args.solo_minutes) * 60.0
                    phase_s = _life_elapsed() % (acc_s + sol_s)
                    want_present = (phase_s < acc_s)
                    if want_present and not port.mother_present:
                        port.join_mother()
                        print(f'  [UMWELT @ {_life_elapsed():6.1f}s | cyc={cycle}] Mother joined co-reading ([在场] 书页, 妈妈)', flush=True)
                    elif (not want_present) and port.mother_present and port.awaiting_reply_from != 'mother':
                        port.leave_scene('mother')
                        print(f'  [UMWELT @ {_life_elapsed():6.1f}s | cyc={cycle}] Mother stepped away -> solo book navigation ([在场] 书页)', flush=True)

                t_step = time.perf_counter()
                if not bool(engine.alif_executive_state['pending_spike']) and port.has_pending_interrupt():
                    engine.alif_executive_state['pending_spike'] = True
                    engine.alif_executive_state['external_interrupt'] = True
                is_spike = bool(engine.alif_executive_state['pending_spike'])
                if is_spike:
                    port.set_cycle_hint(cycle + 1)

                hooks[0] = lambda: activation_storage(engine.params, world_act_dir, **activation_options)
                try:
                    sub_events, row = alif.step_body(engine)
                finally:
                    hooks[0] = None
                step_dt = time.perf_counter() - t_step

                commit_info = None
                if row.get('kind') == 'calm_sensory_piece':
                    calm_times.append(step_dt)
                else:
                    cycle += 1
                    exec_times.append(step_dt)
                    commit_info = port.commit_pending_turn(
                        metal_g=row.get('metal_g'),
                        evidence_available=row.get('evidence_available', True),
                    )
                    if args.reflect_every_turns > 0 and commit_info and commit_info.get('committed') == 'mother':
                        try:
                            refl = port.maybe_reflect_and_adapt(
                                reflect_every_turns=args.reflect_every_turns,
                                cycle=cycle,
                                recent_flux=(flux_trajectory[-1] if flux_trajectory else None),
                                notes_path=args.mother_notes,
                            )
                            if refl is not None:
                                print(
                                    f'  [MOTHER REFLECTION #{refl["reflection_index"]} | cyc={cycle} | '
                                    f'plan_updated={refl["plan_updated"]} (v{refl["plan_version"]})]\n'
                                    f'    model: {refl["mengli_model"]}\n'
                                    f'    note : {refl["progress_note"]}',
                                    flush=True,
                                )
                        except Exception as refl_err:
                            print(f'  [MOTHER REFLECTION SKIPPED] {refl_err}', flush=True)
                    metric_row = _safe_json_value(dict(
                        row,
                        cycle=int(cycle),
                        elapsed_seconds=round(_life_elapsed(), 3),
                        spoke=bool(row.get('self_bytes', 0)),
                    ))
                    met_f.write(json.dumps(metric_row, ensure_ascii=False) + '\n')
                    met_f.flush()
                    last_exec_row = metric_row

                world_info = row.get('world') or {}
                channel = world_info.get('channel', 'unknown')
                if channel in ('dsh_mother', 'live_participant', 'tool'):
                    t_sum = port.telemetry_summary()
                    hr = t_sum.get('provider_hit_rate')
                    hr_str = 'n/a' if hr is None else f'{100.0 * hr:.1f}%'
                    t_hit = world_info.get('caregiver_cache_hit_tokens')
                    t_miss = world_info.get('caregiver_cache_miss_tokens')
                    turn_hr = (
                        float(t_hit) / float(t_hit + t_miss)
                        if isinstance(t_hit, (int, float)) and isinstance(t_miss, (int, float)) and (t_hit + t_miss) > 0
                        else None
                    )
                    turn_hr_str = 'n/a' if turn_hr is None else f'{100.0 * turn_hr:.1f}%'
                    sample = dict(
                        cycle=int(cycle),
                        life_seconds=round(_life_elapsed(), 2),
                        channel=channel,
                        speaker=world_info.get('speaker'),
                        perspective_mode=world_info.get('perspective_mode'),
                        care_stage=world_info.get('care_stage'),
                        care_plan_version=world_info.get('care_plan_version'),
                        utterance_kind=world_info.get('utterance_kind'),
                        mirror_excerpt=world_info.get('mirror_excerpt'),
                        child_text=world_info.get('child_text'),
                        reply_text=world_info.get('mother_reply'),
                        metal_g=(None if row.get('metal_g') is None else float(row['metal_g'])),
                        cache_hit_tokens=t_hit,
                        cache_miss_tokens=t_miss,
                        turn_hit_rate=turn_hr,
                        provider_hit_rate=hr,
                        caregiver_source=world_info.get('caregiver_source'),
                        summary_version=t_sum.get('summary_version'),
                        stage_changed=(bool(commit_info.get('stage_changed')) if commit_info else False),
                    )
                    dialogue_log.append(sample)
                    print(
                        f'  [{channel} | cyc={cycle} | t={sample["life_seconds"]:5.1f}s | '
                        f'mode={sample["perspective_mode"] or "-"} | v={sample["care_plan_version"] or "-"} | '
                        f'src={sample["caregiver_source"] or "-"} | turn_hit={turn_hr_str} | cum_hit={hr_str} | '
                        f'comp={t_sum.get("summary_version")}]\n'
                        f'    stumble: {sample["mirror_excerpt"] or "-"}\n'
                        f'    mengli : {sample["child_text"] or "(silent)"}\n'
                        f'    reply  : {sample["reply_text"]}',
                        flush=True,
                    )

                rec = dict(
                    cycle=int(cycle),
                    life_seconds=round(_life_elapsed(), 3),
                    step_seconds=round(step_dt, 4),
                    kind=row.get('kind', 'executive_spike'),
                    external_bytes=int(row.get('external_bytes', 0)),
                    response_nll=float(row.get('response_nll', 0.0)),
                    metal_g=(None if row.get('metal_g') is None else float(row['metal_g'])),
                    channel=channel,
                    perspective_mode=world_info.get('perspective_mode'),
                    care_stage=world_info.get('care_stage'),
                    care_plan_version=world_info.get('care_plan_version'),
                    mother_bytes=int(world_info.get('mother_bytes', 0)),
                    sub_events=len(sub_events),
                )
                ev_f.write(json.dumps(rec, ensure_ascii=False) + '\n')

                cur_life = _life_elapsed()
                if cur_life >= next_eval_at:
                    t_p0 = time.perf_counter()
                    flux_trajectory.append(_run_flux_probe(cur_life))
                    pause_seconds += time.perf_counter() - t_p0
                    next_eval_at += float(args.eval_every_seconds)

                should_save_now = (
                    (cur_life >= next_save_at)
                    or (args.bind_dashboard and channel in ('live_participant', 'reply_to_parent', 'call_parent'))
                    or (args.save_on_mother_turn and channel == 'dsh_mother')
                )
                if should_save_now:
                    t_s0 = time.perf_counter()
                    _publish_live_checkpoint(last_exec_row)
                    pause_seconds += time.perf_counter() - t_s0
                    if cur_life >= next_save_at:
                        next_save_at += float(args.save_every_seconds)
                        print(f'  [SAVE @ {cur_life:6.1f}s | cyc={cycle}] saved periodic checkpoint.', flush=True)
    finally:
        if had_policy_attr:
            engine._policy_grad = original_policy
        elif '_policy_grad' in vars(engine):
            delattr(engine, '_policy_grad')
        if had_obj_attr:
            engine.model.objective = original_objective
        elif 'objective' in vars(engine.model):
            delattr(engine.model, 'objective')

    life_seconds = _life_elapsed()
    peak_vram_gib = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0

    t_save0 = time.perf_counter()
    ckpt_sha = _publish_live_checkpoint(last_exec_row)
    _write_guard('completed')
    save_seconds = time.perf_counter() - t_save0
    ckpt_path = body_dir / 'last.pt.gz'

    run_ext_bytes = int(engine.alif_executive_state['total_external_bytes']) - start_ext_bytes
    run_spikes = int(engine.alif_executive_state['executive_spikes']) - start_spikes
    run_calms = int(engine.alif_executive_state['calm_pieces']) - start_calms
    port_summary = port.telemetry_summary()
    alif_diag = alif.diagnostics(engine)
    hnet_diag = hnet_mod.diagnostics(engine)

    # Free live body before independent restore eval.
    del engine, loaded, port
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_out = None
    restore_exact = None
    if not args.skip_final_eval:
        eval_out = evaluate_checkpoint(ckpt_path, work)
        alif_r = eval_out.get('alif_restored') or {}
        restore_exact = (
            int(eval_out['cycle']) == int(cycle)
            and int(alif_r.get('executive_spikes', -1)) == int(alif_diag['executive_spikes'])
            and int(alif_r.get('calm_pieces', -1)) == int(alif_diag['calm_pieces'])
            and int(alif_r.get('total_external_bytes', -1)) == int(alif_diag['total_external_bytes'])
        )

    report = dict(
        schema=1,
        source=dict(path=str(source), sha256=source_sha, cycle=start_cycle),
        output=dict(checkpoint=str(ckpt_path), sha256=ckpt_sha, cycle=int(cycle)),
        config=config.get('unified_world_port'),
        timing=dict(
            life_seconds=round(life_seconds, 3),
            save_seconds=round(save_seconds, 3),
            peak_vram_gib=round(peak_vram_gib, 4),
            run_external_bytes=int(run_ext_bytes),
            bytes_per_second=round(run_ext_bytes / max(life_seconds, 1e-6), 2),
            run_calm_pieces=int(run_calms),
            run_executive_spikes=int(run_spikes),
            mean_calm_seconds=(round(float(np.mean(calm_times)), 4) if calm_times else None),
            mean_exec_seconds=(round(float(np.mean(exec_times)), 4) if exec_times else None),
        ),
        unified_port=port_summary,
        hnet=hnet_diag,
        flux_trajectory=flux_trajectory,
        dialogue_samples=dialogue_log[-32:],
        evaluation=eval_out,
        restore_exact=restore_exact,
    )
    write_json(args.report, report)
    shutil.rmtree(temp_dir, ignore_errors=True)

    print('\n=== UNIFIED MENGLI LIFE SUMMARY ===', flush=True)
    print(f'  life={life_seconds:.2f}s | +{run_ext_bytes}B ({report["timing"]["bytes_per_second"]:.2f} B/s) | '
          f'VRAM={peak_vram_gib:.3f} GiB', flush=True)
    print(f'  mother_turns={port_summary["mother_turns"]} '
          f'(mirror={port_summary["mirror_visible_turns"]}, book_only={port_summary["book_only_turns"]}) | '
          f'stage={port_summary["care_stage"]} (v{port_summary["care_plan_version"]}) | '
          f'hit_rate={port_summary["provider_hit_rate"]} | compactions={port_summary["summary_version"]}', flush=True)
    gp_sum = port_summary.get('gaze_pointer')
    if isinstance(gp_sum, dict):
        print(f'  gaze_pointer: params={gp_sum["param_count"]} | steps={gp_sum["step_count"]} | '
              f'entropy={gp_sum["mean_entropy"]} | counts={gp_sum["target_counts"]} | g_ema={gp_sum["target_g_ema"]}', flush=True)
    if eval_out is not None:
        print(f'  16KB_NLL={eval_out["combined_16384_mean_nll"]:.6f} | '
              f'64facts={eval_out["fresh_64_facts"]["mean_score"]:.6f} '
              f'(mse={eval_out["fresh_64_facts"]["mean_mse"]:.4f}) | restore_exact={restore_exact}', flush=True)
    print(f'  report={args.report}', flush=True)


if __name__ == '__main__':
    main()
