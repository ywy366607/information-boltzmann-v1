"""Fair 600-second battle for GPT Bounded Reading (K=4) repaired with 2:1 Hippocampal Anchor.

Uses the EXACT same cycle 2897 checkpoint, strict acceleration stack, 600s wall-clock
budget, and locked 16,384B FineWeb + 64 Old Facts evaluation as:
  - strict_600s
  - option4_600s (original unanchored GPT K=4)
  - alif_600s (our Rheobase-ALIF + 480B interrupt + 2:1 hippocampal anchor)
"""
from __future__ import annotations

import argparse
import gc
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

from scripts.run_alif_executive_battle import (
    DEFAULT_SOURCE,
    GPT_BATTLE_PATH,
    evaluate_checkpoint,
    load_strict_engine,
    sha256_file,
    write_json,
)

ALIF_BATTLE_PATH = PROJECT / 'results/published/cgb_alif_executive_battle_20261124.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--work', type=Path, default=Path('E:/mengli-option4-anchored-600s-20261124'))
    parser.add_argument('--temp-root', type=Path, default=Path('C:/mengli-tmp-option4-anchored'))
    parser.add_argument('--budget-seconds', type=float, default=600.0)
    parser.add_argument('--max-blocks', type=int, default=4)
    parser.add_argument('--surprise-excess', type=float, default=0.5)
    parser.add_argument('--hippocampal-anchor-scale', type=float, default=0.5)
    parser.add_argument('--report', type=Path, default=PROJECT / 'results/published/cgb_four_way_fair_battle_20261124.json')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    spool = args.temp_root / 'spool'
    spool.mkdir(parents=True, exist_ok=True)
    os.environ['CGB_CHECKPOINT_SPOOL'] = str(spool)
    torch.set_num_threads(1)

    work = args.work.resolve()
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    temp_dir = args.temp_root.resolve() / 'run_tmp'
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    source = args.source.resolve()
    source_sha = sha256_file(source)
    print(f'[OPT4+ANCHOR] Loading source {source.name} (sha={source_sha[:12]})...', flush=True)
    loaded = load_strict_engine(source, work, temp_dir=temp_dir, warm_runtime=True)
    engine, config = loaded.engine, loaded.config
    cycle = int(loaded.cycle)

    from scripts.ib_local import autonomous_cycle
    from scripts.ib_local import bounded_reading_option
    from scripts.ib_local.mother_checkpoint import save as save_ckpt
    from scripts.native_policy_activation_storage import activation_storage, policy_storage

    bounded_reading_option.configure(
        engine,
        config,
        max_blocks=args.max_blocks,
        surprise_excess=args.surprise_excess,
        hippocampal_anchor_scale=args.hippocampal_anchor_scale,
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

    engine._policy_grad = wrapped_policy
    engine.model.objective = wrapped_objective

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    events_path = work / 'events.jsonl'
    t_start = time.perf_counter()
    opt_times, metal_gs, pre_gain_mses = [], [], []
    total_ext_bytes = 0
    total_updates = 0
    options_completed = 0
    anchor_piece_count = 0

    print(f'[OPT4+ANCHOR] Starting continuous life (K={args.max_blocks}, anchor={args.hippocampal_anchor_scale}, '
          f'budget={args.budget_seconds:.1f}s, smoke={args.smoke})...', flush=True)
    try:
        with events_path.open('w', encoding='utf8') as evf:
            while True:
                t_opt = time.perf_counter()
                while True:
                    with activation_storage(engine.params, world_act_dir, **activation_options) as (_, ctx):
                        hooks[0] = ctx
                        ev = autonomous_cycle.dispatch(engine)
                    hooks[0] = None
                    if ev.get('autonomous_event', {}).get('kind') == 'external':
                        cycle += 1
                        ev['cycle'] = cycle
                        evf.write(json.dumps(ev, ensure_ascii=False) + '\n')
                        break
                    evf.write(json.dumps(ev, ensure_ascii=False) + '\n')
                evf.flush()
                dt = time.perf_counter() - t_opt
                opt_times.append(dt)
                options_completed += 1
                total_ext_bytes += int(ev.get('external_bytes', 0))
                total_updates += int(ev.get('updates', 0))
                if ev.get('metal_g') is not None:
                    metal_gs.append(float(ev['metal_g']))
                na = ev.get('causal_use', {}).get('native_amortization') or {}
                if na.get('real_prediction') is not None and na.get('real_target') is not None:
                    pre_gain_mses.append((float(na['real_prediction']) - float(na['real_target'])) ** 2)
                opt_rec = ev.get('bounded_reading_option') or {}
                anchors = opt_rec.get('hippocampal_anchors') or []
                anchor_piece_count += len(anchors)
                a0 = anchors[0] if anchors else {}
                print(f'  [OPT #{options_completed:2d}] cycle={cycle} blocks={opt_rec.get("blocks")} '
                      f'stop={opt_rec.get("stop")} anchors={len(anchors)} '
                      f'a0=({a0.get("wood_replay_samples",0)},{a0.get("internal_transition_samples",0)},{a0.get("causal_program_replay",0)}) '
                      f'choice={ev.get("choice")} nll={ev.get("response_nll", 0):.4f} '
                      f'metal_g={ev.get("metal_g", 0):+.4f} dt={dt:.2f}s total_B={total_ext_bytes}', flush=True)

                elapsed = time.perf_counter() - t_start
                if args.smoke:
                    if options_completed >= 2 or elapsed >= args.budget_seconds:
                        break
                else:
                    if elapsed >= args.budget_seconds:
                        break
    finally:
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
    life_seconds = time.perf_counter() - t_start
    peak_vram_gib = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0

    ro_state = dict(engine.reading_option_state)
    ro_spec = dict(engine.reading_option_spec)
    t_save = time.perf_counter()
    ckpt_sha = save_ckpt(engine, work / 'body', cycle, config)
    save_seconds = time.perf_counter() - t_save
    ckpt_path = work / 'body/last.pt.gz'

    del loaded, engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f'[OPT4+ANCHOR] Life done: {total_ext_bytes}B in {life_seconds:.2f}s '
          f'({total_ext_bytes/life_seconds:.2f} B/s), options={options_completed}, '
          f'blocks={ro_state["blocks"]}, anchored_pieces={anchor_piece_count}, '
          f'peak_vram={peak_vram_gib:.3f} GiB, save={save_seconds:.2f}s (sha={ckpt_sha[:12]})', flush=True)

    eval_work = args.temp_root.resolve() / 'eval_work'
    eval_work.mkdir(parents=True, exist_ok=True)
    eval_res = evaluate_checkpoint(ckpt_path, eval_work)
    shutil.rmtree(args.temp_root.resolve(), ignore_errors=True)

    # Assemble four-way fair comparison table
    four_way = {}
    if GPT_BATTLE_PATH.exists():
        b = json.loads(GPT_BATTLE_PATH.read_text(encoding='utf8'))
        four_way['origin_2897'] = dict(
            combined_16384_nll=b['origin']['combined_16384_mean_nll'],
            fresh_8192_nll=b['origin']['fresh_8192_nll']['mean_nll'],
            classic_8192_nll=b['origin']['classic_8192_nll']['mean_nll'],
            fresh_64_facts_score=b['origin']['fresh_64_facts']['mean_score'],
            fresh_64_facts_mse=b['origin']['fresh_64_facts']['mean_mse'],
        )
        for name, arm in b.get('arms', {}).items():
            s = arm['summary']
            ev = arm.get('audit') or arm.get('final') or {}
            four_way[name] = dict(
                life_seconds=s['life_seconds'],
                external_bytes=s['external_bytes'],
                updates=s['updates'],
                bytes_per_second=s['bytes_per_second'],
                combined_16384_nll=ev.get('combined_16384_mean_nll'),
                fresh_8192_nll=(ev.get('fresh_8192_nll') or {}).get('mean_nll'),
                classic_8192_nll=(ev.get('classic_8192_nll') or {}).get('mean_nll'),
                fresh_64_facts_score=(ev.get('fresh_64_facts') or {}).get('mean_score'),
                fresh_64_facts_mse=(ev.get('fresh_64_facts') or {}).get('mean_mse'),
                mean_metal_g=s.get('mean_metal_g'),
                pre_outcome_gain_mse=s.get('pre_outcome_gain_mse'),
            )
    if ALIF_BATTLE_PATH.exists():
        a = json.loads(ALIF_BATTLE_PATH.read_text(encoding='utf8'))
        ev = a['evaluation']
        four_way['alif_600s'] = dict(
            life_seconds=a['life_seconds'],
            external_bytes=a['total_external_bytes'],
            updates=a['total_updates'],
            bytes_per_second=a['bytes_per_second'],
            combined_16384_nll=ev['combined_16384_mean_nll'],
            fresh_8192_nll=ev['fresh_8192_nll']['mean_nll'],
            classic_8192_nll=ev['classic_8192_nll']['mean_nll'],
            fresh_64_facts_score=ev['fresh_64_facts']['mean_score'],
            fresh_64_facts_mse=ev['fresh_64_facts']['mean_mse'],
            mean_metal_g=a.get('mean_metal_g'),
            pre_outcome_gain_mse=a.get('pre_outcome_gain_mse'),
        )

    four_way['option4_anchored_600s'] = dict(
        life_seconds=life_seconds,
        external_bytes=total_ext_bytes,
        updates=total_updates,
        bytes_per_second=total_ext_bytes / life_seconds,
        options_completed=options_completed,
        total_blocks=int(ro_state['blocks']),
        terminations=dict(ro_state['terminations']),
        anchored_continuation_pieces=anchor_piece_count,
        combined_16384_nll=eval_res['combined_16384_mean_nll'],
        fresh_8192_nll=eval_res['fresh_8192_nll']['mean_nll'],
        classic_8192_nll=eval_res['classic_8192_nll']['mean_nll'],
        fresh_64_facts_score=eval_res['fresh_64_facts']['mean_score'],
        fresh_64_facts_mse=eval_res['fresh_64_facts']['mean_mse'],
        mean_metal_g=float(np.mean(metal_gs)) if metal_gs else None,
        pre_outcome_gain_mse=float(np.mean(pre_gain_mses)) if pre_gain_mses else None,
        peak_vram_gib=peak_vram_gib,
    )

    report = dict(
        schema=1,
        mode='smoke' if args.smoke else 'four_way_equal_wallclock_600s',
        source=str(source),
        source_sha256=source_sha,
        source_cycle=2897,
        end_cycle=cycle,
        checkpoint=str(ckpt_path),
        checkpoint_sha256=ckpt_sha,
        reading_option_spec=ro_spec,
        reading_option_state=ro_state,
        evaluation=eval_res,
        four_way_comparison=four_way,
    )
    out_path = (work / 'smoke_report.json') if args.smoke else args.report
    write_json(out_path, report)
    print(f'[OPT4+ANCHOR] Report saved to {out_path}', flush=True)
    print(json.dumps(four_way, indent=2), flush=True)


if __name__ == '__main__':
    main()
