"""Equal wall-clock (600s per arm) comparison from cycle 2861: strict Batch 2 vs +phase-coupled-rhythm.

Compares real learning yield per unit wall-clock time on the same 10-minute
budget (stopping cleanly at the first external boundary >= 600s, without
mid-run evaluation overhead). Evaluates both final checkpoints on 16,384
independent FineWeb bytes (fresh 8192B + classic 8192B), fresh 64 old facts,
online calibration/metal_g, and full restore verification.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path('D:/information-boltzmann-v1').resolve()
SOURCE = Path('E:/mengli-ridge-lr-acceptance-20261123/ref_ridge_3e4_16c/body/last.pt.gz').resolve()
FRESH_FACTS = Path('E:/mengli-ridge-lr-acceptance-20261123/fresh_old_facts_64.pt').resolve()
OUT_ROOT = Path('E:/mengli-wallclock-10min-20261123').resolve()
SPOOL = Path('E:/mengli-runtime-staging-20261123').resolve()
REPORT_PATH = PROJECT / 'results/published/cgb_equal_wallclock_10min_strict_vs_phase_20261123.json'
VAL_PATH = (PROJECT / 'data/cgb_natural_text_20261115/validation.bin').resolve()

BUDGET_SECONDS = 600.0
FRESH_TEXT_SPEC = dict(
    path=str(VAL_PATH),
    sha256='b09c3879960cab5b5bbe3d3282059d472d7ee140925c58d7550b7fedebfbee93',
    starts=[49152, 147456, 212992, 274432],
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


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def reclaim_temp(arm_name: str):
    arm_dir = OUT_ROOT / arm_name
    ckpt = arm_dir / 'body/last.pt.gz'
    meta = json.loads(ckpt.with_name('last.metadata.json').read_text(encoding='utf8'))
    digest = sha256_file(ckpt)
    if digest != meta['checkpoint_sha256']:
        raise RuntimeError(f'{arm_name}: checkpoint sha256 mismatch')
    temp_dir = OUT_ROOT / 'temporary' / arm_name
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    return digest


def run_arm(arm_name: str, extra_flags: list[str], status_path: Path):
    arm_dir = OUT_ROOT / arm_name
    if (arm_dir / 'result.json').exists() and (arm_dir / 'body/last.pt.gz').exists():
        print(f'[{arm_name}] already completed, reusing', flush=True)
        return json.loads((arm_dir / 'result.json').read_text(encoding='utf8'))
    if arm_dir.exists():
        shutil.rmtree(arm_dir)
    temp_dir = OUT_ROOT / 'temporary' / arm_name
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    SPOOL.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONIOENCODING='utf-8', CGB_CHECKPOINT_SPOOL=str(SPOOL),
               PYTHONPATH=str(PROJECT))
    cmd = [
        sys.executable, '-m', 'scripts.run_native_throughput_worker',
        '--profile-core',
        '--checkpoint', str(SOURCE),
        '--work', str(arm_dir),
        '--cycles', '500',
        '--max-life-seconds', str(BUDGET_SECONDS),
        '--evaluate-every', '500',
        '--native-decision', '--native-word-execution', '--successor-distribution',
        '--joint-successor', '--batch-native-words', '--batch-preview-scores',
        '--amortized-decision', '--latent-action-planner', '--persistent-rhythm', 'coupled',
        '--root-native-correction', 'observe', '--retention-scope-prior', 'off',
        '--profile-runtime',
        '--activation-mode', 'tiered',
        '--activation-gpu-mib', '1536',
        '--activation-host-mib', '1024',
        '--direct-typed-question', '--factorized-endpoints', '--shared-utter-preview',
        '--fused-decode-step', '--native-replay-batch-size', '8',
        '--policy-proposal-batch-size', '4', '--reuse-objective-logits', '--warm-runtime',
        '--checkpoint-compressor', 'isal', '--checkpoint-compression-workers', '2',
        '--wiener-gate', '--split-hamiltonian-theta', '--learnable-precision-scales',
        '--table-storage', str(temp_dir / 'table'),
        '--world-activations', str(temp_dir / 'world'),
        '--policy-activations', str(temp_dir / 'policy'),
        *extra_flags,
    ]
    log_path = OUT_ROOT / f'{arm_name}.log'
    t0 = time.perf_counter()
    write_json(status_path, dict(state=f'running_{arm_name}', started_at=time.strftime('%Y-%m-%d %H:%M:%S')))
    print(f'[{arm_name}] launching {BUDGET_SECONDS:.0f}s equal-wall-clock run from 2861...', flush=True)
    with log_path.open('w', encoding='utf8') as logf:
        proc = subprocess.run(cmd, cwd=PROJECT, env=env, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        write_json(status_path, dict(state=f'failed_{arm_name}', returncode=proc.returncode, seconds=elapsed))
        raise RuntimeError(f'{arm_name} failed (code={proc.returncode}); see {log_path}')
    digest = reclaim_temp(arm_name)
    res = json.loads((arm_dir / 'result.json').read_text(encoding='utf8'))
    print(f'[{arm_name}] done: {res.get("cycles")} cycles, {res.get("updates")} updates, '
          f'{res.get("external_bytes")}B in life={res.get("life_seconds", 0):.1f}s (sha={digest[:12]})', flush=True)
    return res


def eval_ckpt(label: str, ckpt: Path, work_dir: Path):
    eval_out = work_dir / f'{label}_eval.json'
    code = r'''
import json, os, shutil, sys, time
from pathlib import Path
import torch
sys.path.insert(0, r"D:\information-boltzmann-v1")
from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
from scripts.ib_local.wood_policy_measurement import segments, score
from scripts.ib_local.learnable_precision_scales import diagnostics as scale_diag
from scripts.run_native_causal_use_value import evaluate

ckpt = Path(sys.argv[1])
work = Path(sys.argv[2])
out_path = Path(sys.argv[3])
fresh_spec = json.loads(sys.argv[4])
classic_spec = json.loads(sys.argv[5])
fresh_facts_path = Path(sys.argv[6])

torch.set_num_threads(1)
tmp_out = work / 'eval_tmp'
if tmp_out.exists():
    shutil.rmtree(tmp_out, ignore_errors=True)
t0 = time.perf_counter()
opts = continuation_parser().parse_args([
    '--checkpoint', str(ckpt), '--output', str(tmp_out), '--cycles', '1',
    '--world-source', 'text', '--autonomous-events', '--optimizer-policy', 'budgeted',
    '--table-storage', str(tmp_out / 'table'), '--causal-storage', str(tmp_out / 'causal'),
])
from scripts.ib_local import run_mother_continuation as continuation
saved_save, saved_export = continuation.save, continuation.export_live
continuation.save = lambda *a, **kw: None
continuation.export_live = lambda *a, **kw: None
try:
    loaded = prepare_continuation(opts, resume_output=False)
finally:
    continuation.save, continuation.export_live = saved_save, saved_export
load_s = time.perf_counter() - t0
engine = loaded.engine
model = engine.model
opt = engine.optimizer

mat_fresh = segments(fresh_spec)
mat_classic = segments(classic_spec)
fresh_facts = torch.load(fresh_facts_path, map_location=engine.device, weights_only=False)
with torch.no_grad():
    sc_fresh = score(engine, mat_fresh)
    sc_classic = score(engine, mat_classic)
    sc_facts = evaluate(engine, fresh_facts)

# Param-weighted gate summary
tot_n, w_gate, w_alive = 0, 0.0, 0.0
for p in model.parameters():
    st = opt.state.get(p, {})
    if 'budget_gate' not in st:
        continue
    g = st['budget_gate'].float()
    n = p.numel()
    tot_n += n
    w_gate += n * float(g.mean())
    w_alive += n * float((g > 0).float().mean())

pr_state = getattr(engine, '_phase_rhythm_state', None)
audit = dict(
    checkpoint=str(ckpt),
    cycle=loaded.cycle,
    load_seconds=round(load_s, 3),
    fresh_8192_nll=sc_fresh,
    classic_8192_nll=sc_classic,
    combined_16384_mean_nll=0.5 * (sc_fresh['mean_nll'] + sc_classic['mean_nll']),
    fresh_64_facts=dict(mean_score=sc_facts['mean_score'], mean_mse=sc_facts['mean_mse']),
    scale_diagnostics=scale_diag(engine),
    optimizer_mean_gate=(w_gate / tot_n) if tot_n else None,
    optimizer_alive_frac=(w_alive / tot_n) if tot_n else None,
    wiener_gate_all_groups=all(bool(g.get('wiener_gate', False)) for g in opt.param_groups),
    split_theta=bool(getattr(model.head_coupling, '_split_theta', False)),
    trainable_missing_from_optimizer=loaded.config.get('mother_parameter_audit', {}).get('trainable_missing_from_optimizer', []),
    has_phase_coupled_rhythm=loaded.config.get('phase_coupled_rhythm') is not None,
    reading_option_state=getattr(engine, 'reading_option_state', None),
    reading_option_spec=getattr(engine, 'reading_option_spec', None),
    pending_duration_blocks=(engine.prev or {}).get('duration_blocks', 1),
    active_reading_option=getattr(engine, '_reading_option_session', None) is not None,
    restored_phase_rhythm_cycles=None if pr_state is None else pr_state.get('cycles'),
)
out_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
if hasattr(model, '_causal_archive') and hasattr(model._causal_archive, 'db'):
    model._causal_archive.db.close()
shutil.rmtree(tmp_out, ignore_errors=True)
'''
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONIOENCODING='utf-8', CGB_CHECKPOINT_SPOOL=str(SPOOL),
               PYTHONPATH=str(PROJECT))
    proc = subprocess.run([sys.executable, '-c', code, str(ckpt), str(work_dir), str(eval_out),
                           json.dumps(FRESH_TEXT_SPEC), json.dumps(CLASSIC_TEXT_SPEC), str(FRESH_FACTS)],
                          cwd=PROJECT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'eval_ckpt({label}) failed:\n{proc.stdout}\n{proc.stderr}')
    return json.loads(eval_out.read_text(encoding='utf8'))


def summarize_arm(arm_name: str):
    arm_dir = OUT_ROOT / arm_name
    rows = [json.loads(l) for l in (arm_dir / 'events.jsonl').read_text(encoding='utf8').splitlines() if l.strip()]
    ext = [r for r in rows if r.get('autonomous_event', {}).get('kind') == 'external']
    res = json.loads((arm_dir / 'result.json').read_text(encoding='utf8'))
    evals = json.loads((arm_dir / 'evaluations.json').read_text(encoding='utf8'))

    metal_gs, pre_errs, read_scores = [], [], []
    regimes = {}
    for r in ext:
        if 'metal_g' in r:
            metal_gs.append(float(r['metal_g']))
        pog = r.get('pre_outcome_gain') or {}
        if 'expected' in pog and 'target' in pog:
            pre_errs.append((float(pog['expected']) - float(pog['target'])) ** 2)
        for ru in (r.get('read_use') or []):
            if 'log_score' in ru:
                read_scores.append(float(ru['log_score']))
        pr = r.get('phase_coupled_rhythm') or {}
        if 'regime' in pr:
            rg = pr['regime']
            regimes[rg] = regimes.get(rg, 0) + 1

    life_s = float(res.get('life_seconds', 0.0))
    ext_b = int(res.get('external_bytes', 0))
    return dict(
        external_cycles=len(ext),
        total_events=len(rows),
        updates=int(res.get('updates', 0)),
        external_bytes=ext_b,
        life_seconds=life_s,
        seconds_per_cycle=life_s / max(len(ext), 1),
        bytes_per_second=ext_b / max(life_s, 1e-9),
        checkpoint_commit_seconds=float(res.get('checkpoint_commit_seconds', 0.0)),
        checkpoint_sha256=res.get('checkpoint_sha256'),
        mean_metal_g=(sum(metal_gs) / len(metal_gs)) if metal_gs else None,
        pre_outcome_gain_mse=(sum(pre_errs) / len(pre_errs)) if pre_errs else None,
        online_read_use_mean_log_score=(sum(read_scores) / len(read_scores)) if read_scores else None,
        phase_regimes=regimes,
        own_64_old_facts=[dict(external=e['external'], cycle=e['cycle'],
                               mean_score=e['mean_score'], mean_mse=e['mean_mse']) for e in evals],
    )


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = OUT_ROOT / 'campaign_status.json'
    t_all = time.perf_counter()

    write_json(status_path, dict(state='eval_origin_2861'))
    print('[origin] scoring cycle 2861 baseline on 16384B FineWeb + fresh 64 old facts...', flush=True)
    origin_eval = eval_ckpt('origin_2861', SOURCE, OUT_ROOT)

    run_arm('strict_10min', [], status_path)
    run_arm('phase_rhythm_10min', ['--phase-coupled-rhythm'], status_path)

    write_json(status_path, dict(state='evaluating'))
    print('[eval] scoring both 10-minute endpoints...', flush=True)
    strict_eval = eval_ckpt('strict_10min', OUT_ROOT / 'strict_10min/body/last.pt.gz', OUT_ROOT / 'strict_10min')
    phase_eval = eval_ckpt('phase_rhythm_10min', OUT_ROOT / 'phase_rhythm_10min/body/last.pt.gz', OUT_ROOT / 'phase_rhythm_10min')
    strict_sum = summarize_arm('strict_10min')
    phase_sum = summarize_arm('phase_rhythm_10min')

    def rates(ev, sm):
        mins = sm['life_seconds'] / 60.0
        fresh_imp = origin_eval['fresh_8192_nll']['mean_nll'] - ev['fresh_8192_nll']['mean_nll']
        classic_imp = origin_eval['classic_8192_nll']['mean_nll'] - ev['classic_8192_nll']['mean_nll']
        comb_imp = origin_eval['combined_16384_mean_nll'] - ev['combined_16384_mean_nll']
        facts_imp = ev['fresh_64_facts']['mean_score'] - origin_eval['fresh_64_facts']['mean_score']
        facts_mse_d = ev['fresh_64_facts']['mean_mse'] - origin_eval['fresh_64_facts']['mean_mse']
        return dict(
            cycles=sm['external_cycles'],
            updates=sm['updates'],
            external_bytes=sm['external_bytes'],
            life_seconds=round(sm['life_seconds'], 2),
            seconds_per_cycle=round(sm['seconds_per_cycle'], 3),
            bytes_per_second=round(sm['bytes_per_second'], 2),
            fresh_8192_nll=ev['fresh_8192_nll']['mean_nll'],
            fresh_8192_nll_gain=fresh_imp,
            classic_8192_nll=ev['classic_8192_nll']['mean_nll'],
            classic_8192_nll_gain=classic_imp,
            combined_16384_nll=ev['combined_16384_mean_nll'],
            combined_16384_nll_gain=comb_imp,
            combined_nll_gain_per_min=comb_imp / max(mins, 1e-9),
            fresh_64_facts_score=ev['fresh_64_facts']['mean_score'],
            fresh_64_facts_gain=facts_imp,
            fresh_64_facts_gain_per_min=facts_imp / max(mins, 1e-9),
            fresh_64_facts_mse_delta=facts_mse_d,
            mean_metal_g=sm['mean_metal_g'],
            pre_outcome_gain_mse=sm['pre_outcome_gain_mse'],
            restore_verified=(ev['wiener_gate_all_groups'] and ev['split_theta'] and
                              ev['trainable_missing_from_optimizer'] == []),
        )

    report = dict(
        schema=1,
        source=str(SOURCE),
        source_sha256=sha256_file(SOURCE),
        source_cycle=2861,
        budget_seconds_per_arm=BUDGET_SECONDS,
        origin_2861=origin_eval,
        comparison=dict(
            strict_10min=rates(strict_eval, strict_sum),
            phase_rhythm_10min=rates(phase_eval, phase_sum),
        ),
        arms=dict(
            strict_10min=dict(summary=strict_sum, audit=strict_eval),
            phase_rhythm_10min=dict(summary=phase_sum, audit=phase_eval),
        ),
        total_wall_seconds=round(time.perf_counter() - t_all, 2),
    )
    write_json(REPORT_PATH, report)
    write_json(status_path, dict(state='completed', report=str(REPORT_PATH), total_wall_seconds=report['total_wall_seconds']))
    print(json.dumps(report['comparison'], indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
