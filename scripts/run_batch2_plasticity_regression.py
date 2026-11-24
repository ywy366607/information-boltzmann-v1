"""Serial 32-cycle A/B regression from original 2813: baseline vs Batch 2 full fix.

Batch 2 opt-in bundle:
  1. --wiener-gate (momentum Wiener cooling gate in BudgetedAdamW, CPU + Triton)
  2. --split-hamiltonian-theta (4 role-specific parameters in AdaptiveHamiltonian)
  3. --learnable-precision-scales (log-space relation_ridge + memory.temperature)

Runs both arms serially on the strict-1:1 fast stack, reclaims temporary tables,
evaluates independent FineWeb 8192B NLL + 64 old facts + full internal readouts
(gate distributions, coupling trajectory, ridge/temperature trajectories,
PastPresentField workset_retained stats, finite_use_value.expected call counts),
and verifies complete restore of the Batch 2 checkpoint.
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
SOURCE = Path('E:/mengli-native-supply-completed-20261122/original/last.pt.gz').resolve()
OUT_ROOT = Path('E:/mengli-batch2-regression-20261123').resolve()
SPOOL = Path('E:/mengli-runtime-staging-20261123').resolve()
REPORT_PATH = PROJECT / 'results/published/cgb_batch2_plasticity_scales_32cycles_20261123.json'
VALIDATION_SPEC = dict(
    path=str((PROJECT / 'data/cgb_natural_text_20261115/validation.bin').resolve()),
    sha256='b09c3879960cab5b5bbe3d3282059d472d7ee140925c58d7550b7fedebfbee93',
    starts=[0, 95629, 191258, 286887],
    bytes_per_segment=2048,
    total_scored_bytes=8192,
)
ORIGIN_2813_NLL = 1.5830430808561182
CYCLES = 32


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
        '--cycles', str(CYCLES),
        '--native-decision', '--native-word-execution', '--successor-distribution',
        '--joint-successor', '--batch-native-words', '--batch-preview-scores',
        '--amortized-decision', '--latent-action-planner', '--persistent-rhythm', 'coupled',
        '--root-native-correction', 'observe', '--retention-scope-prior', 'off',
        '--profile-runtime', '--evaluate-every', '8',
        '--activation-mode', 'tiered',
        '--activation-gpu-mib', '1536',
        '--activation-host-mib', '1024',
        '--direct-typed-question', '--factorized-endpoints', '--shared-utter-preview',
        '--fused-decode-step', '--native-replay-batch-size', '8',
        '--policy-proposal-batch-size', '4', '--reuse-objective-logits', '--warm-runtime',
        '--checkpoint-compressor', 'isal', '--checkpoint-compression-workers', '2',
        '--table-storage', str(temp_dir / 'table'),
        '--world-activations', str(temp_dir / 'world'),
        '--policy-activations', str(temp_dir / 'policy'),
        *extra_flags,
    ]
    log_path = OUT_ROOT / f'{arm_name}.log'
    t0 = time.perf_counter()
    write_json(status_path, dict(state=f'running_{arm_name}', started_at=time.strftime('%Y-%m-%d %H:%M:%S')))
    print(f'[{arm_name}] launching {CYCLES} cycles...', flush=True)
    with log_path.open('w', encoding='utf8') as logf:
        proc = subprocess.run(cmd, cwd=PROJECT, env=env, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        write_json(status_path, dict(state=f'failed_{arm_name}', returncode=proc.returncode, seconds=elapsed))
        raise RuntimeError(f'{arm_name} failed with exit code {proc.returncode}; see {log_path}')
    digest = reclaim_temp(arm_name)
    res = json.loads((arm_dir / 'result.json').read_text(encoding='utf8'))
    print(f'[{arm_name}] done in {elapsed:.1f}s (life={res.get("life_seconds", 0):.1f}s, sha={digest[:12]})', flush=True)
    return res


def eval_arm_subprocess(arm_name: str):
    """Score independent FineWeb 8192B NLL and inspect restored model/optimizer state."""
    arm_dir = OUT_ROOT / arm_name
    ckpt = arm_dir / 'body/last.pt.gz'
    eval_out = arm_dir / 'fineweb_and_state_audit.json'
    code = r'''
import json, os, sys, time
from pathlib import Path
import torch
sys.path.insert(0, r"D:\information-boltzmann-v1")
from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
from scripts.ib_local.wood_policy_measurement import segments, score
from scripts.ib_local.learnable_precision_scales import effective_ridge, effective_temperature

ckpt = Path(sys.argv[1])
work = Path(sys.argv[2])
out_path = Path(sys.argv[3])
val_spec = json.loads(sys.argv[4])

torch.set_num_threads(1)
tmp_out = work / 'eval_restore_tmp'
if tmp_out.exists():
    import shutil
    shutil.rmtree(tmp_out, ignore_errors=True)
t0 = time.perf_counter()
opts = continuation_parser().parse_args([
    '--checkpoint', str(ckpt),
    '--output', str(tmp_out),
    '--cycles', '1',
    '--world-source', 'text',
    '--autonomous-events',
    '--optimizer-policy', 'budgeted',
    '--table-storage', str(tmp_out / 'table'),
    '--causal-storage', str(tmp_out / 'causal'),
])
loaded = prepare_continuation(opts, resume_output=False)
load_s = time.perf_counter() - t0
engine = loaded.engine
model = engine.model
opt = engine.optimizer

# Score independent FineWeb 8192 bytes (read-only)
mat = segments(val_spec)
with torch.no_grad():
    sc = score(engine, mat)

# Inspect restored optimizer gates (param-weighted)
tot_n = 0
w_gate = 0.0
w_alive = 0.0
zero_tensors = 0
budget_tensors = 0
for p in model.parameters():
    st = opt.state.get(p, {})
    if 'budget_gate' not in st:
        continue
    budget_tensors += 1
    g = st['budget_gate'].float()
    n = p.numel()
    tot_n += n
    w_gate += n * float(g.mean())
    w_alive += n * float((g > 0).float().mean())
    if bool((g == 0).all()):
        zero_tensors += 1

hc = model.head_coupling
with torch.no_grad():
    coupling = torch.nn.functional.softplus(hc.theta[:, hc.heads + 1]).cpu().tolist()
    role_gates = {}
    for rn, rp in zip(hc.theta_part_names(), hc.theta_parts()):
        g = opt.state.get(rp, {}).get('budget_gate')
        role_gates[rn] = None if g is None else float(g.detach().mean())

wiener_groups = [bool(gr.get('wiener_gate', False)) for gr in opt.param_groups]
_r = effective_ridge(model, getattr(engine, 'relation_ridge', 0.1))
_t = effective_temperature(model.memory)
audit = dict(
    checkpoint=str(ckpt),
    cycle=loaded.cycle,
    load_seconds=round(load_s, 3),
    fineweb_8192=sc,
    trainable_missing_from_optimizer=loaded.config.get('mother_parameter_audit', {}).get('trainable_missing_from_optimizer', []),
    wiener_gate_all_groups=all(wiener_groups) and len(wiener_groups) > 0,
    wiener_gate_any_group=any(wiener_groups),
    split_theta=bool(getattr(hc, '_split_theta', False)),
    coupling_mean=sum(coupling) / len(coupling),
    coupling_min=min(coupling),
    coupling_max=max(coupling),
    coupling_per_layer=coupling,
    hamiltonian_role_gates=role_gates,
    effective_ridge=float(_r.detach() if torch.is_tensor(_r) else _r),
    effective_temperature=float(_t.detach() if torch.is_tensor(_t) else _t),
    has_log_relation_ridge=hasattr(model, 'log_relation_ridge'),
    has_log_temperature=hasattr(model.memory, 'log_temperature'),
    optimizer_param_weighted_mean_gate=(w_gate / tot_n) if tot_n else None,
    optimizer_param_weighted_alive_frac=(w_alive / tot_n) if tot_n else None,
    optimizer_budget_tensors=budget_tensors,
    optimizer_tensors_all_zero=zero_tensors,
)
out_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
if hasattr(model, '_causal_archive') and hasattr(model._causal_archive, 'db'):
    model._causal_archive.db.close()
import shutil
shutil.rmtree(tmp_out, ignore_errors=True)
'''
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONIOENCODING='utf-8', CGB_CHECKPOINT_SPOOL=str(SPOOL),
               PYTHONPATH=str(PROJECT))
    proc = subprocess.run([sys.executable, '-c', code, str(ckpt), str(arm_dir),
                           str(eval_out), json.dumps(VALIDATION_SPEC)],
                          cwd=PROJECT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'eval_arm_subprocess({arm_name}) failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}')
    return json.loads(eval_out.read_text(encoding='utf8'))


def summarize_events(arm_name: str):
    arm_dir = OUT_ROOT / arm_name
    rows = [json.loads(line) for line in (arm_dir / 'events.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    ext = [r for r in rows if r.get('autonomous_event', {}).get('kind') == 'external']
    retained_hist = {}
    coupling_traj = []
    ridge_traj = []
    temp_traj = []
    metal_gs = []
    opt_write_means = []
    for r in ext:
        pp = r.get('past_present') or {}
        if 'retained' in pp:
            k = str(pp['retained'])
            retained_hist[k] = retained_hist.get(k, 0) + 1
        ad = r.get('adaptive_dynamics') or {}
        if 'coupling' in ad:
            cs = ad['coupling']
            coupling_traj.append(dict(cycle=r['cycle'], mean=sum(cs)/len(cs), min=min(cs), max=max(cs),
                                      gate=ad.get('budget_write_fraction')))
        ls = r.get('learnable_precision_scales') or {}
        if 'relation_ridge' in ls:
            ridge_traj.append(dict(cycle=r['cycle'], ridge=ls['relation_ridge'], gate=ls.get('relation_ridge_gate')))
        if 'memory_temperature' in ls:
            temp_traj.append(dict(cycle=r['cycle'], temperature=ls['memory_temperature'], gate=ls.get('memory_temperature_gate')))
        if 'metal_g' in r:
            metal_gs.append(float(r['metal_g']))
        op = r.get('optimizer_plasticity') or {}
        if 'layers' in op and op['layers']:
            ms = [v['mean_write_fraction'] for v in op['layers'].values()]
            opt_write_means.append(dict(cycle=r['cycle'], layer_mean_write=sum(ms)/len(ms)))
    prof = json.loads((arm_dir / 'runtime_profile.json').read_text(encoding='utf8')) if (arm_dir / 'runtime_profile.json').exists() else {}
    evals = json.loads((arm_dir / 'evaluations.json').read_text(encoding='utf8')) if (arm_dir / 'evaluations.json').exists() else []
    evals_compact = [dict(external=e['external'], cycle=e['cycle'],
                          mean_score=e['mean_score'], mean_mse=e['mean_mse']) for e in evals]
    res = json.loads((arm_dir / 'result.json').read_text(encoding='utf8'))
    fuv_calls = None
    for k, v in prof.get('functions', {}).items():
        if 'finite_use_value' in k and 'expected' in k:
            fuv_calls = dict(name=k, calls=v.get('calls'), inclusive_seconds=v.get('inclusive_seconds'))
    return dict(
        external_cycles=len(ext),
        total_events=len(rows),
        updates=res.get('updates'),
        external_bytes=res.get('external_bytes'),
        life_seconds=res.get('life_seconds'),
        seconds_per_cycle=(res.get('life_seconds', 0) / max(len(ext), 1)),
        checkpoint_commit_seconds=res.get('checkpoint_commit_seconds'),
        mean_metal_g=(sum(metal_gs) / len(metal_gs)) if metal_gs else None,
        workset_retained_histogram=retained_hist,
        old_facts_evaluations=evals_compact,
        coupling_start=coupling_traj[0] if coupling_traj else None,
        coupling_mid=coupling_traj[len(coupling_traj)//2] if coupling_traj else None,
        coupling_end=coupling_traj[-1] if coupling_traj else None,
        ridge_trajectory=ridge_traj[::4] + ([ridge_traj[-1]] if ridge_traj else []),
        temperature_trajectory=temp_traj[::4] + ([temp_traj[-1]] if temp_traj else []),
        layer_mean_write_start=opt_write_means[0] if opt_write_means else None,
        layer_mean_write_end=opt_write_means[-1] if opt_write_means else None,
        finite_use_expected_profile=fuv_calls,
    )


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = OUT_ROOT / 'campaign_status.json'
    t_all = time.perf_counter()

    run_arm('baseline_32c', [], status_path)
    run_arm('batch2_full_32c', ['--wiener-gate', '--split-hamiltonian-theta', '--learnable-precision-scales'], status_path)

    write_json(status_path, dict(state='evaluating'))
    print('[eval] scoring FineWeb 8192B + restore audits...', flush=True)
    base_eval = eval_arm_subprocess('baseline_32c')
    cand_eval = eval_arm_subprocess('batch2_full_32c')
    base_sum = summarize_events('baseline_32c')
    cand_sum = summarize_events('batch2_full_32c')

    base_nll = base_eval['fineweb_8192']['mean_nll']
    cand_nll = cand_eval['fineweb_8192']['mean_nll']
    base_facts = base_sum['old_facts_evaluations']
    cand_facts = cand_sum['old_facts_evaluations']
    report = dict(
        schema=1,
        source=str(SOURCE),
        source_cycle=2813,
        cycles=CYCLES,
        validation=VALIDATION_SPEC,
        origin_2813_fineweb_mean_nll=ORIGIN_2813_NLL,
        comparison=dict(
            baseline_fineweb_nll=base_nll,
            batch2_fineweb_nll=cand_nll,
            baseline_nll_improvement=ORIGIN_2813_NLL - base_nll,
            batch2_nll_improvement=ORIGIN_2813_NLL - cand_nll,
            batch2_vs_baseline_nll_advantage=base_nll - cand_nll,
            baseline_old_facts_delta=(base_facts[-1]['mean_score'] - base_facts[0]['mean_score']) if len(base_facts) >= 2 else None,
            batch2_old_facts_delta=(cand_facts[-1]['mean_score'] - cand_facts[0]['mean_score']) if len(cand_facts) >= 2 else None,
            baseline_old_facts_mse_delta=(base_facts[-1]['mean_mse'] - base_facts[0]['mean_mse']) if len(base_facts) >= 2 else None,
            batch2_old_facts_mse_delta=(cand_facts[-1]['mean_mse'] - cand_facts[0]['mean_mse']) if len(cand_facts) >= 2 else None,
            baseline_mean_gate=base_eval['optimizer_param_weighted_mean_gate'],
            batch2_mean_gate=cand_eval['optimizer_param_weighted_mean_gate'],
            baseline_coupling_mean=base_eval['coupling_mean'],
            batch2_coupling_mean=cand_eval['coupling_mean'],
            batch2_final_ridge=cand_eval['effective_ridge'],
            batch2_final_temperature=cand_eval['effective_temperature'],
            batch2_restore_verified=(
                cand_eval['split_theta'] and cand_eval['wiener_gate_all_groups']
                and cand_eval['has_log_relation_ridge'] and cand_eval['has_log_temperature']
                and cand_eval['trainable_missing_from_optimizer'] == []
            ),
        ),
        arms=dict(
            baseline_32c=dict(summary=base_sum, audit=base_eval),
            batch2_full_32c=dict(summary=cand_sum, audit=cand_eval),
        ),
        total_wall_seconds=round(time.perf_counter() - t_all, 2),
    )
    write_json(REPORT_PATH, report)
    write_json(status_path, dict(state='completed', report=str(REPORT_PATH), total_wall_seconds=report['total_wall_seconds']))
    print(json.dumps(report['comparison'], indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
