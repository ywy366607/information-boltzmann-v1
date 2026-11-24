"""Equal-wall-clock battle for ALIF Surprise-Gated Executive from cycle 2897.

Uses the EXACT same source checkpoint (strict_10min cycle 2897), exact same
strict acceleration stack, and exact same locked 16,384B FineWeb + 64 Old Facts
evaluation protocol as `results/published/cgb_bounded_reading_battle_20261123.json`,
so Strict vs GPT Bounded-Reading (K=4) vs ALIF Executive can be compared
100% apples-to-apples.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

PROJECT = Path('D:/information-boltzmann-v1').resolve()
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

DEFAULT_SOURCE = Path('E:/mengli-wallclock-10min-20261123/strict_10min/body/last.pt.gz').resolve()
FRESH_FACTS = Path('E:/mengli-ridge-lr-acceptance-20261123/fresh_old_facts_64.pt').resolve()
VAL_PATH = (PROJECT / 'data/cgb_natural_text_20261115/validation.bin').resolve()
SPOOL = Path('E:/mengli-runtime-staging-20261123').resolve()
GPT_BATTLE_PATH = PROJECT / 'results/published/cgb_bounded_reading_battle_20261123.json'

FRESH_TEXT_SPEC = dict(
    path=str(VAL_PATH),
    sha256='b09c3879960cab5b5bbe3d3282059d472d7ee140925c58d7550b7fedebfbee93',
    starts=[4096, 98304, 155648, 221184],
    bytes_per_segment=2048,
    total_scored_bytes=8192,
)
CLASSIC_TEXT_SPEC = dict(
    path=str(VAL_PATH),
    sha256='b09c3879960cab5b5bbe3d3282059d472d7ee140925c58d7550b7fedebfbee93',
    starts=[0, 95629, 191258, 286887],
    bytes_per_segment=2048,
    total_scored_bytes=8192,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def load_strict_engine(ckpt: Path, work: Path, *, temp_dir: Path, warm_runtime: bool = True):
    from scripts.ib_local import run_mother_continuation as continuation
    from scripts.ib_local.mother_checkpoint import configure_compression
    from scripts.ib_local.native_replay_batch import configure as configure_replay
    from scripts.ib_local.scoped_decision_graph import configure_fused_step
    from scripts.ib_local.epistemic_value_reference import configure as configure_epistemic_value
    from scripts.ib_local.native_amortized_decision import configure as configure_amortized
    from scripts.ib_local.persistent_rhythm import configure as configure_rhythm
    from scripts.ib_local.optimizer_policy import configure_wiener_gate
    from scripts.ib_local.learnable_precision_scales import configure as configure_scales

    continuation.ROOT = PROJECT
    body_dir = work / 'body'
    causal_dir = work / 'causal'
    causal_dir.mkdir(parents=True, exist_ok=True)

    opts = continuation.continuation_parser().parse_args([
        '--checkpoint', str(ckpt),
        '--output', str(body_dir),
        '--cycles', '500',
        '--world-source', 'text',
        '--autonomous-events',
        '--optimizer-policy', 'budgeted',
        '--table-storage', str(temp_dir / 'table'),
        '--causal-storage', str(causal_dir),
    ])

    saved_save, saved_export = continuation.save, continuation.export_live
    continuation.save = lambda *a, **kw: None
    continuation.export_live = lambda *a, **kw: None
    try:
        loaded = continuation.prepare_continuation(opts)
    finally:
        continuation.save, continuation.export_live = saved_save, saved_export

    engine, config = loaded.engine, loaded.config
    configure_epistemic_value(engine, schema=5, word_execution='native_anchor')
    engine.native_word_runtime = 'batch'
    config['native_word_runtime'] = dict(schema=2, mode='batch', batch_size=4)
    engine.model._preview_batch_enabled = True
    amort_spec = dict(schema=1, seed=201120, planner='action_batch',
                      correction='decision_budget', root_native_correction='observe')
    configure_amortized(engine, amort_spec)
    config['native_amortization'] = amort_spec
    configure_rhythm(engine, interleave=True)
    config['persistent_rhythm'] = dict(schema=1, interleave=True)
    configure_wiener_gate(engine, config, True)
    configure_scales(engine, config.get('learnable_precision_scales', {'schema': 1, 'ridge': True, 'temperature': True}),
                     initialize=False)
    configure_compression(config, backend='isal', workers=2)
    engine.native_direct_question = True
    engine.native_factorized_endpoints = True
    engine.native_shared_utter_preview = True
    configure_replay(engine, config, batch_size=8, reuse_actual=True)
    engine.policy_proposal_batch_size = 4
    configure_fused_step(engine, config)

    if warm_runtime:
        from scripts.ib_local.table_branch import table_branch
        from scripts.ib_local.language_action import peek
        from scripts.ib_local.navigation_live_witness import frozen_table_logp
        readonly = np.frombuffer(bytes((int(engine.pending), int(engine.pending))), dtype=np.uint8)
        for raw in (readonly, readonly.copy()):
            with table_branch(engine.table) as branch:
                branch.run_credit(raw, max_window=1)
                peek(branch, int(raw[-1]))
            frozen_table_logp(engine.table, raw)

    return loaded


def evaluate_checkpoint(ckpt: Path, work: Path) -> dict:
    """Evaluate one saved checkpoint on locked 16,384B FineWeb + 64 Old Facts + restore check."""
    from scripts.ib_local import run_mother_continuation as continuation
    from scripts.ib_local.wood_policy_measurement import segments, score
    from scripts.ib_local.learnable_precision_scales import diagnostics as scale_diag
    from scripts.run_native_causal_use_value import evaluate as eval_facts

    tmp = work / 'eval_tmp'
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    t0 = time.perf_counter()
    continuation.ROOT = PROJECT
    opts = continuation.continuation_parser().parse_args([
        '--checkpoint', str(ckpt),
        '--output', str(tmp),
        '--cycles', '1',
        '--world-source', 'text',
        '--autonomous-events',
        '--optimizer-policy', 'budgeted',
        '--table-storage', str(tmp / 'table'),
        '--causal-storage', str(tmp / 'causal'),
    ])
    saved_save, saved_export = continuation.save, continuation.export_live
    continuation.save = lambda *a, **kw: None
    continuation.export_live = lambda *a, **kw: None
    try:
        loaded = continuation.prepare_continuation(opts)
    finally:
        continuation.save, continuation.export_live = saved_save, saved_export
    load_s = time.perf_counter() - t0
    engine = loaded.engine

    fresh_segs = segments(FRESH_TEXT_SPEC)
    classic_segs = segments(CLASSIC_TEXT_SPEC)
    facts_payload = torch.load(FRESH_FACTS, map_location=engine.device, weights_only=False)
    with torch.no_grad():
        fresh_res = score(engine, fresh_segs)
        classic_res = score(engine, classic_segs)
        facts_res = eval_facts(engine, facts_payload)
    combined_nll = 0.5 * (fresh_res['mean_nll'] + classic_res['mean_nll'])

    # Verify optimizer gates & ALIF state restored cleanly.
    gates = []
    alive = 0
    total_el = 0
    for st in engine.optimizer.state.values():
        g = st.get('budget_gate')
        if torch.is_tensor(g):
            gf = g.detach().float()
            gates.append(float(gf.mean()))
            alive += int((gf > 0.05).sum())
            total_el += int(gf.numel())

    alif_state = getattr(engine, 'alif_executive_state', None)
    alif_restored = None if alif_state is None else dict(
        v=float(alif_state['v']),
        a=float(alif_state['a']),
        executive_spikes=int(alif_state['executive_spikes']),
        calm_pieces=int(alif_state['calm_pieces']),
        total_external_bytes=int(alif_state['total_external_bytes']),
        total_learned_bytes=int(alif_state['total_learned_bytes']),
        pending_spike=bool(alif_state['pending_spike']),
    )

    out = dict(
        checkpoint=str(ckpt),
        cycle=int(loaded.cycle),
        load_seconds=round(load_s, 3),
        fresh_8192_nll=fresh_res,
        classic_8192_nll=classic_res,
        combined_16384_mean_nll=combined_nll,
        fresh_64_facts=dict(mean_score=float(facts_res['mean_score']), mean_mse=float(facts_res['mean_mse'])),
        scale_diagnostics=scale_diag(engine),
        optimizer_mean_gate=float(np.mean(gates)) if gates else None,
        optimizer_alive_frac=(alive / total_el) if total_el else None,
        alif_restored=alif_restored,
    )
    del loaded, engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--work', type=Path, default=Path('E:/mengli-alif-executive-20261124'))
    parser.add_argument('--temp-root', type=Path, default=None)
    parser.add_argument('--budget-seconds', type=float, default=600.0)
    parser.add_argument('--report', type=Path, default=PROJECT / 'results/published/cgb_alif_executive_battle_20261124.json')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--lam-v', type=float, default=0.84)
    parser.add_argument('--lam-a', type=float, default=0.78)
    parser.add_argument('--theta0', type=float, default=1.00)
    parser.add_argument('--beta', type=float, default=0.95)
    parser.add_argument('--delta0', type=float, default=0.20)
    parser.add_argument('--i-max', type=float, default=1.85)
    parser.add_argument('--hippocampal-anchor-scale', type=float, default=0.5)
    parser.add_argument('--piece-bytes', type=int, default=480)
    parser.add_argument('--calm-lr-scale', type=float, default=1.0,
                        help='Calm-piece LR scale (>0 explicit, <0 auto sqrt(piece_bytes/640))')
    parser.add_argument('--extra-hippocampal-steps', type=int, default=0,
                        help='Extra ~10ms hippocampal micro-steps per calm piece (>=0 explicit, <0 auto round(piece_bytes/640)-1)')
    parser.add_argument('--eval-every-seconds', type=float, default=0.0,
                        help='If >0, run read-only locked 16KB + 64-fact flux probes at t=0 and every N life seconds')
    parser.add_argument('--no-activation-storage', action='store_true')
    parser.add_argument('--encoder-gram-damping', type=float, default=0.0)
    parser.add_argument('--encoder-drift-budget', type=float, default=0.0)
    parser.add_argument('--encoder-pre-adam', action='store_true')
    parser.add_argument('--emeans-alif-reservoir', action='store_true')
    parser.add_argument('--emeans-clusters', type=int, default=64)
    parser.add_argument('--emeans-top-clusters', type=int, default=6)
    parser.add_argument('--learnable-alif-rhythm', action='store_true')
    parser.add_argument('--learnable-alif-lr', type=float, default=0.03)
    parser.add_argument('--learnable-alif-tau', type=float, default=0.25)
    parser.add_argument('--learnable-alif-time-cost', type=float, default=0.012)
    parser.add_argument('--phase-space-boltzmann', action='store_true')
    parser.add_argument('--psb-lr', type=float, default=0.01)
    parser.add_argument('--psb-alpha-init', type=float, default=0.08)
    parser.add_argument('--psb-eta', type=float, default=0.12)
    parser.add_argument('--psb-gamma', type=float, default=0.16)
    parser.add_argument('--root-latent-planning', action='store_true')
    parser.add_argument('--rlp-full-search-every', type=int, default=8)
    parser.add_argument('--rlp-teacher-cal-every', type=int, default=4)
    parser.add_argument('--rlp-novelty-threshold', type=float, default=0.75)
    parser.add_argument('--rlp-margin-threshold', type=float, default=0.001)
    parser.add_argument('--rlp-steer-scale', type=float, default=0.25)
    parser.add_argument('--hnet-chunking', action='store_true')
    parser.add_argument('--hnet-target-ratio', type=float, default=4.0)
    parser.add_argument('--hnet-enc-layers', type=int, default=4)
    parser.add_argument('--hnet-dec-layers', type=int, default=4)
    parser.add_argument('--hnet-ratio-weight', type=float, default=0.03)
    parser.add_argument('--hnet-lr', type=float, default=None)
    args = parser.parse_args()

    spool_dir = (args.temp_root / 'spool') if args.temp_root is not None else SPOOL
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
    print(f'[ALIF] Loading source {source.name} (sha={source_sha[:12]})...', flush=True)
    loaded = load_strict_engine(source, work, temp_dir=temp_dir, warm_runtime=True)
    engine, config = loaded.engine, loaded.config
    cycle = int(loaded.cycle)

    from scripts.ib_local import alif_executive as alif
    from scripts.ib_local.mother_checkpoint import save as save_ckpt
    from scripts.native_policy_activation_storage import activation_storage, policy_storage

    import math
    calm_lr_scale = math.sqrt(args.piece_bytes / 640.0) if args.calm_lr_scale < 0 else float(args.calm_lr_scale)
    extra_h_steps = max(0, int(round(args.piece_bytes / 640.0)) - 1) if args.extra_hippocampal_steps < 0 else int(args.extra_hippocampal_steps)
    alif.configure(
        engine,
        config,
        piece_bytes=args.piece_bytes,
        lam_v=args.lam_v,
        lam_a=args.lam_a,
        theta0=args.theta0,
        beta=args.beta,
        delta0=args.delta0,
        i_max=args.i_max,
        hippocampal_anchor_scale=args.hippocampal_anchor_scale,
        calm_lr_scale=calm_lr_scale,
        extra_hippocampal_steps=extra_h_steps,
    )
    if args.learnable_alif_rhythm:
        from scripts.ib_local import learnable_alif_rhythm as lar
        lar.configure(
            engine,
            config,
            lr=args.learnable_alif_lr,
            tau=args.learnable_alif_tau,
            time_cost_weight=args.learnable_alif_time_cost,
        )
    if args.encoder_gram_damping > 0.0 or args.encoder_drift_budget > 0.0:
        from scripts.ib_local import encoder_subspace_metric as esm
        esm.configure(
            engine,
            config,
            damping=(args.encoder_gram_damping if args.encoder_gram_damping > 0.0 else 0.25),
            drift_budget=(args.encoder_drift_budget if args.encoder_drift_budget > 0.0 else None),
            post_step=(not args.encoder_pre_adam),
        )
    if args.emeans_alif_reservoir:
        from scripts.ib_local import emeans_alif_reservoir as ear
        ear.configure(
            engine,
            config,
            n_clusters=args.emeans_clusters,
            top_clusters=args.emeans_top_clusters,
            k_max=int(engine.model.body.coevolution.slots),
        )
    if args.phase_space_boltzmann:
        from scripts.ib_local import phase_space_boltzmann as psb
        psb.configure(
            engine,
            config,
            lr=args.psb_lr,
            alpha_init=args.psb_alpha_init,
            eta_free_energy=args.psb_eta,
            gamma_diss=args.psb_gamma,
        )
    if args.root_latent_planning:
        from scripts.ib_local import root_latent_planning as rlp
        rlp.configure(
            engine,
            config,
            full_search_every=args.rlp_full_search_every,
            teacher_cal_every=args.rlp_teacher_cal_every,
            novelty_threshold=args.rlp_novelty_threshold,
            margin_threshold=args.rlp_margin_threshold,
            intent_steer_scale=args.rlp_steer_scale,
        )
    if args.hnet_chunking:
        from scripts.ib_local import hnet_chunking as hnet_mod
        hnet_mod.configure(
            engine,
            config,
            enc_layers=args.hnet_enc_layers,
            dec_layers=args.hnet_dec_layers,
            target_ratio=args.hnet_target_ratio,
            ratio_loss_weight=args.hnet_ratio_weight,
            lr=args.hnet_lr,
        )

    activation_options = dict(
        mode='tiered',
        gpu_budget_bytes=1536 * 1024 * 1024,
        host_budget_bytes=1024 * 1024 * 1024,
    )
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

    use_act_storage = not bool(args.no_activation_storage)
    if use_act_storage:
        engine._policy_grad = wrapped_policy
        engine.model.objective = wrapped_objective

    start_cycle = int(cycle)
    start_ext_bytes = int(engine.alif_executive_state['total_external_bytes'])
    start_spikes = int(engine.alif_executive_state['executive_spikes'])
    start_calms = int(engine.alif_executive_state['calm_pieces'])

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
        facts_payload = torch.load(FRESH_FACTS, map_location=engine.device, weights_only=False)
        with torch.no_grad():
            fresh_res = score(engine, fresh_segs)
            classic_res = score(engine, classic_segs)
            facts_res = eval_facts(engine, facts_payload)
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
            total_external_bytes=int(engine.alif_executive_state['total_external_bytes']),
            cycle=int(cycle),
            combined_16384_nll=float(comb),
            fresh_8192_nll=float(fresh_res['mean_nll']),
            classic_8192_nll=float(classic_res['mean_nll']),
            fresh_64_facts_score=float(facts_res['mean_score']),
            fresh_64_facts_mse=float(facts_res['mean_mse']),
        )
        print(f'  [FLUX @ {life_s:6.1f}s | +{pt["run_external_bytes"]:6d}B] '
              f'16KB_NLL={comb:.6f} (fresh={fresh_res["mean_nll"]:.4f} classic={classic_res["mean_nll"]:.4f}) '
              f'64facts={pt["fresh_64_facts_score"]:.6f} mse={pt["fresh_64_facts_mse"]:.4f}', flush=True)
        return pt

    flux_trajectory = []
    if args.eval_every_seconds > 0.0:
        flux_trajectory.append(_run_flux_probe(0.0))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    events_path = work / 'events.jsonl'
    t_start = time.perf_counter()
    eval_pause_seconds = 0.0
    next_eval_at = float(args.eval_every_seconds) if args.eval_every_seconds > 0.0 else float('inf')
    calm_times, exec_times = [], []
    metal_gs, pre_gain_mses = [], []
    total_updates = 0
    step_idx = 0

    print(f'[ALIF] Starting continuous life (budget={args.budget_seconds:.1f}s, smoke={args.smoke}, '
          f'piece_bytes={args.piece_bytes}, calm_lr_scale={calm_lr_scale:.3f}, extra_h={extra_h_steps}, '
          f'act_storage={use_act_storage}, learnable_rhythm={args.learnable_alif_rhythm})...', flush=True)
    try:
        with events_path.open('w', encoding='utf8') as evf:
            while True:
                t_ev = time.perf_counter()
                was_spike = bool(engine.alif_executive_state['pending_spike'])
                if use_act_storage:
                    with activation_storage(engine.params, world_act_dir, **activation_options) as (_, ctx):
                        hooks[0] = ctx
                        sub_events, main_event = alif.step_body(engine)
                    hooks[0] = None
                else:
                    sub_events, main_event = alif.step_body(engine)
                dt = time.perf_counter() - t_ev
                step_idx += 1

                for sub in sub_events:
                    evf.write(json.dumps(sub, ensure_ascii=False) + '\n')

                if was_spike:
                    cycle += 1
                    main_event['cycle'] = cycle
                    exec_times.append(dt)
                    if main_event.get('metal_g') is not None:
                        metal_gs.append(float(main_event['metal_g']))
                    na = main_event.get('causal_use', {}).get('native_amortization') or {}
                    if na.get('real_prediction') is not None and na.get('real_target') is not None:
                        pre_gain_mses.append((float(na['real_prediction']) - float(na['real_target'])) ** 2)
                    print(f'  [SPIKE #{engine.alif_executive_state["executive_spikes"]:2d}] cycle={cycle} '
                          f'choice={main_event.get("choice")} nll={main_event.get("response_nll", 0):.4f} '
                          f'metal_g={main_event.get("metal_g", 0):+.4f} dt={dt:.2f}s '
                          f'total_B={engine.alif_executive_state["total_external_bytes"]}', flush=True)
                else:
                    calm_times.append(dt)
                    ae = main_event['alif_executive']
                    if step_idx <= 12 or ae['spiked'] or (engine.alif_executive_state['calm_pieces'] % 10 == 0):
                        print(f'  [CALM  #{engine.alif_executive_state["calm_pieces"]:3d}] '
                              f'nll={ae["piece_nll"]:.4f} burst={ae["burst_nll"]:.4f} '
                              f'V={ae["v_plus"]:.3f}/{ae["threshold"]:.3f} '
                              f'anchor=({ae["wood_replay_samples"]},{ae["internal_transition_samples"]},{ae["causal_program_replay"]}) '
                              f'spiked={ae["spiked"]}({ae["cause"]}) dt={dt:.2f}s '
                              f'total_B={engine.alif_executive_state["total_external_bytes"]}', flush=True)

                total_updates += int(main_event.get('updates', 0))
                evf.write(json.dumps(main_event, ensure_ascii=False) + '\n')
                evf.flush()

                elapsed = time.perf_counter() - t_start - eval_pause_seconds
                if (not args.smoke and args.eval_every_seconds > 0.0 and
                        elapsed >= next_eval_at and elapsed < args.budget_seconds and
                        not engine.alif_executive_state['pending_spike']):
                    t_p = time.perf_counter()
                    flux_trajectory.append(_run_flux_probe(elapsed))
                    eval_pause_seconds += time.perf_counter() - t_p
                    next_eval_at += float(args.eval_every_seconds)
                if args.smoke:
                    if ((engine.alif_executive_state['executive_spikes'] - start_spikes) >= 2 and
                            (engine.alif_executive_state['calm_pieces'] - start_calms) >= 4 and
                            not engine.alif_executive_state['pending_spike']):
                        break
                    if elapsed >= args.budget_seconds:
                        break
                else:
                    if elapsed >= args.budget_seconds and not engine.alif_executive_state['pending_spike']:
                        break
    finally:
        if use_act_storage:
            if had_policy_attr:
                engine._policy_grad = original_policy
            else:
                del engine._policy_grad
            if had_obj_attr:
                engine.model.objective = original_objective
            else:
                del engine.model.objective

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    life_seconds = time.perf_counter() - t_start - eval_pause_seconds
    peak_vram_gib = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0
    run_ext_bytes = int(engine.alif_executive_state['total_external_bytes']) - start_ext_bytes
    if args.eval_every_seconds > 0.0:
        flux_trajectory.append(_run_flux_probe(life_seconds))

    diag = alif.diagnostics(engine)
    from scripts.ib_local import encoder_subspace_metric as esm
    from scripts.ib_local import emeans_alif_reservoir as ear
    from scripts.ib_local import learnable_alif_rhythm as lar
    from scripts.ib_local import phase_space_boltzmann as psb
    from scripts.ib_local import root_latent_planning as rlp
    from scripts.ib_local import hnet_chunking as hnet_mod
    gram_diag = esm.diagnostics(engine)
    emeans_diag = ear.diagnostics(engine)
    rhythm_diag = lar.diagnostics(engine)
    psb_diag = psb.diagnostics(engine)
    rlp_diag = rlp.diagnostics(engine)
    hnet_diag = hnet_mod.diagnostics(engine)
    t_save = time.perf_counter()
    ckpt_sha = save_ckpt(engine, work / 'body', cycle, config)
    save_seconds = time.perf_counter() - t_save
    ckpt_path = work / 'body/last.pt.gz'

    # Release GPU memory & temporary staging before independent restore evaluation.
    del loaded, engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f'[ALIF] Life done: +{run_ext_bytes}B (cum {diag["total_external_bytes"]}B) in {life_seconds:.2f}s '
          f'({run_ext_bytes/life_seconds:.2f} B/s), '
          f'spikes=+{diag["executive_spikes"]-start_spikes}, calm_pieces=+{diag["calm_pieces"]-start_calms}, '
          f'peak_vram={peak_vram_gib:.3f} GiB, save={save_seconds:.2f}s (sha={ckpt_sha[:12]})', flush=True)
    if gram_diag is not None:
        print(f'[ALIF] Subspace metric: post_step={gram_diag["post_step"]} budget={gram_diag["drift_budget"]} '
              f'apps={gram_diag["applications"]} clamps={gram_diag["trust_region_clamps"]} '
              f'pass={gram_diag["pass_throughs"]} vecs={gram_diag["last_vector_count"]} '
              f'last_drift={gram_diag["last_raw_drift"]:.2e}->{gram_diag["last_proj_drift"]:.2e}', flush=True)
    if emeans_diag is not None:
        print(f'[ALIF] e-means Population ALIF: clusters={emeans_diag["non_empty_clusters"]}/{emeans_diag["active_clusters"]} '
              f'queries={emeans_diag["total_queries"]} mean_woken={emeans_diag["mean_woken"]:.1f} '
              f'mean_spiked={emeans_diag["mean_spiked"]:.2f}/{emeans_diag["k_max"]} '
              f'revivals={emeans_diag["revivals"]}', flush=True)
    if rhythm_diag is not None:
        rp = rhythm_diag['params']
        print(f'[ALIF] Learnable rhythm: updates={rhythm_diag["credit_updates"]} '
              f'theta0={rp["theta0"]:.3f} beta={rp["beta"]:.3f} lam_v={rp["lam_v"]:.3f} '
              f'lam_a={rp["lam_a"]:.3f} delta0={rp["delta0"]:.3f} w_burst={rp["w_burst"]:.3f} '
              f'wiener_gate={rhythm_diag["wiener_mean_gate"]}', flush=True)
    if psb_diag is not None:
        print(f'[ALIF] PhaseSpaceBoltzmann: alpha_p={psb_diag["alpha_p"]:.4f} '
              f'kappa=({psb_diag["kappa_lang"]:.4f},{psb_diag["kappa_mem"]:.4f},{psb_diag["kappa_act"]:.4f}) '
              f'z_kin={psb_diag["live_z_kin"]:+.3f} scale={psb_diag["last_thermostat_scale"]:.4f} '
              f'H={psb_diag["last_entropy"]:.3f} steps={psb_diag["thermostat_steps"]} '
              f'wiener_gate={psb_diag["wiener_mean_gate"]}', flush=True)
    if rlp_diag is not None:
        print(f'[ALIF] RootLatentPlanning: single={rlp_diag["latent_single_count"]}/{rlp_diag["decisions"]} '
              f'({100*rlp_diag["latent_single_rate"]:.1f}%) '
              f'teacher_b=(0:{rlp_diag["teacher_budget_0"]},1:{rlp_diag["teacher_budget_1"]},2:{rlp_diag["teacher_budget_2"]}) '
              f'err_ema={rlp_diag["teacher_abs_error_ema"]:.4f}', flush=True)
    if hnet_diag is not None:
        print(f'[ALIF] H-Net ({hnet_diag["enc_layers"]}+{32-hnet_diag["enc_layers"]-hnet_diag["dec_layers"]}+{hnet_diag["dec_layers"]}): '
              f'target_R={hnet_diag["target_ratio"]:.1f} eff_R={hnet_diag["effective_compression_ema"]:.2f}x '
              f'b_rate={hnet_diag["mean_boundary_rate_ema"]:.3f} thresh={hnet_diag["router_thresh"]:.4f} '
              f'res_rms={hnet_diag["residual_weight_rms"]:.4f} wiener={hnet_diag["wiener_mean_gate"]}', flush=True)

    print('[ALIF] Running independent restore & locked 16,384B FineWeb + 64 Old Facts eval...', flush=True)
    eval_work = (args.temp_root.resolve() / 'eval_tmp_root') if args.temp_root is not None else work
    eval_work.mkdir(parents=True, exist_ok=True)
    eval_res = evaluate_checkpoint(ckpt_path, eval_work)
    if args.temp_root is not None:
        shutil.rmtree(args.temp_root.resolve(), ignore_errors=True)

    # Verify restored ALIF state matches pre-save state identically.
    restored = eval_res['alif_restored']
    restore_exact = (
        restored is not None and
        restored['executive_spikes'] == diag['executive_spikes'] and
        restored['calm_pieces'] == diag['calm_pieces'] and
        restored['total_external_bytes'] == diag['total_external_bytes'] and
        restored['total_learned_bytes'] == diag['total_learned_bytes'] and
        abs(restored['v'] - diag['v']) < 1e-12 and
        abs(restored['a'] - diag['a']) < 1e-12
    )
    if not restore_exact:
        raise RuntimeError(f'ALIF restore mismatch: pre={diag} vs post={restored}')

    # Load reference Strict & GPT K=4 numbers from cgb_bounded_reading_battle_20261123.json if present.
    ref_summary = {}
    if GPT_BATTLE_PATH.exists():
        b = json.loads(GPT_BATTLE_PATH.read_text(encoding='utf8'))
        ref_summary['origin_2897'] = dict(
            combined_16384_nll=b['origin']['combined_16384_mean_nll'],
            fresh_64_facts_score=b['origin']['fresh_64_facts']['mean_score'],
            fresh_64_facts_mse=b['origin']['fresh_64_facts']['mean_mse'],
        )
        for name, arm in b.get('arms', {}).items():
            s = arm['summary']
            ev = arm.get('audit') or arm.get('final') or arm.get('evaluation') or {}
            ref_summary[name] = dict(
                life_seconds=s['life_seconds'],
                external_bytes=s['external_bytes'],
                bytes_per_second=s['bytes_per_second'],
                combined_16384_nll=ev.get('combined_16384_mean_nll'),
                fresh_8192_nll=(ev.get('fresh_8192_nll') or {}).get('mean_nll'),
                classic_8192_nll=(ev.get('classic_8192_nll') or {}).get('mean_nll'),
                fresh_64_facts_score=(ev.get('fresh_64_facts') or {}).get('mean_score'),
                fresh_64_facts_mse=(ev.get('fresh_64_facts') or {}).get('mean_mse'),
            )

    report = dict(
        schema=1,
        mode='smoke' if args.smoke else 'equal_wallclock_600s',
        source=str(source),
        source_sha256=source_sha,
        source_cycle=start_cycle,
        end_cycle=cycle,
        budget_seconds=args.budget_seconds,
        life_seconds=life_seconds,
        save_seconds=save_seconds,
        checkpoint=str(ckpt_path),
        checkpoint_sha256=ckpt_sha,
        restore_exact=restore_exact,
        peak_vram_gib=peak_vram_gib,
        total_updates=total_updates,
        run_external_bytes=run_ext_bytes,
        bytes_per_second=run_ext_bytes / life_seconds,
        learning_flux_trajectory=flux_trajectory,
        mean_calm_piece_seconds=float(np.mean(calm_times)) if calm_times else None,
        mean_exec_spike_seconds=float(np.mean(exec_times)) if exec_times else None,
        mean_metal_g=float(np.mean(metal_gs)) if metal_gs else None,
        pre_outcome_gain_mse=float(np.mean(pre_gain_mses)) if pre_gain_mses else None,
        alif_diagnostics=diag,
        encoder_subspace_metric=gram_diag,
        emeans_alif_reservoir=emeans_diag,
        learnable_alif_rhythm=rhythm_diag,
        phase_space_boltzmann=psb_diag,
        root_latent_planning=rlp_diag,
        hnet_chunking=hnet_diag,
        evaluation=eval_res,
        references_from_same_2897=ref_summary,
    )
    out_report = (work / 'smoke_report.json') if args.smoke else args.report
    write_json(out_report, report)
    print(f'[ALIF] Report saved to {out_report}', flush=True)
    print(json.dumps(dict(
        bytes_per_second=round(report['bytes_per_second'], 2),
        total_external_bytes=diag['total_external_bytes'],
        executive_spikes=diag['executive_spikes'],
        calm_pieces=diag['calm_pieces'],
        mean_calm_s=round(report['mean_calm_piece_seconds'] or 0, 3),
        mean_exec_s=round(report['mean_exec_spike_seconds'] or 0, 3),
        combined_16384_nll=round(eval_res['combined_16384_mean_nll'], 6),
        fresh_8192_nll=round(eval_res['fresh_8192_nll']['mean_nll'], 6),
        classic_8192_nll=round(eval_res['classic_8192_nll']['mean_nll'], 6),
        fresh_64_facts_score=round(eval_res['fresh_64_facts']['mean_score'], 6),
        fresh_64_facts_mse=round(eval_res['fresh_64_facts']['mean_mse'], 6),
        restore_exact=restore_exact,
        peak_vram_gib=round(peak_vram_gib, 3),
    ), indent=2), flush=True)


if __name__ == '__main__':
    main()
