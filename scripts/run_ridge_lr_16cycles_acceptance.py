"""16-cycle A/B acceptance from cycle 2845 (batch2_full_32c): ridge lr=3e-4 vs lr=1e-2.

Strictly locks fresh held-out acceptance materials (disjoint FineWeb 8192B
segments and disjoint 64 old-fact queries never scanned in Stage 1) BEFORE
launching either arm. Keeps PastPresentField, memory.log_temperature (lr=3e-4),
Wiener gate, split Hamiltonian theta, and strict acceleration stack identical;
phase-coupled-rhythm remains OFF.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path('D:/information-boltzmann-v1').resolve()
SOURCE = Path('E:/mengli-batch2-regression-20261123/batch2_full_32c/body/last.pt.gz').resolve()
DEV_MATERIAL = Path('E:/mengli-batch2-regression-20261123/batch2_full_32c/material.pt').resolve()
OUT_ROOT = Path('E:/mengli-ridge-lr-acceptance-20261123').resolve()
SPOOL = Path('E:/mengli-runtime-staging-20261123').resolve()
REPORT_PATH = PROJECT / 'results/published/cgb_ridge_lr_16cycles_acceptance_20261123.json'
VAL_PATH = (PROJECT / 'data/cgb_natural_text_20261115/validation.bin').resolve()

HISTORICAL_STARTS = [
    0, 16384, 32768, 40960, 65536, 81920, 95629, 114688,
    131072, 163840, 191258, 196608, 229376, 245760, 262144, 286887,
]
FRESH_STARTS = [49152, 147456, 212992, 274432]
BYTES_PER_SEG = 2048
CYCLES = 16


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def lock_acceptance_materials():
    manifest_path = OUT_ROOT / 'locked_acceptance_manifest.json'
    fresh_mat_path = OUT_ROOT / 'fresh_old_facts_64.pt'
    if manifest_path.exists() and fresh_mat_path.exists():
        print('[lock] reusing already locked acceptance manifest', flush=True)
        return json.loads(manifest_path.read_text(encoding='utf8'))

    # Verify 100% disjointness of fresh text segments from ALL historical/dev starts
    val_size = VAL_PATH.stat().st_size
    for s in FRESH_STARTS:
        assert s + BYTES_PER_SEG + 1 <= val_size
        for old in HISTORICAL_STARTS:
            assert s + BYTES_PER_SEG + 1 <= old or old + BYTES_PER_SEG + 1 <= s, (s, old)
    raw_val = VAL_PATH.read_bytes()
    seg_hashes = [hashlib.sha256(raw_val[s:s + BYTES_PER_SEG + 1]).hexdigest() for s in FRESH_STARTS]
    val_spec = dict(
        path=str(VAL_PATH),
        sha256=sha256_file(VAL_PATH),
        starts=FRESH_STARTS,
        bytes_per_segment=BYTES_PER_SEG,
        total_scored_bytes=BYTES_PER_SEG * len(FRESH_STARTS),
        segment_sha256s=seg_hashes,
        disjoint_from_historical_starts=HISTORICAL_STARTS,
    )

    code = r'''
import hashlib, json, os, shutil, sys, time
from dataclasses import replace
from pathlib import Path
import torch
sys.path.insert(0, r"D:\information-boltzmann-v1")
from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
from scripts.ib_local.wood_policy_measurement import segments, score
from scripts.ib_local.causal_use_value import queries
from scripts.ib_local.learnable_precision_scales import set_scale_learning_rates, diagnostics as scale_diag
from scripts.run_native_causal_use_value import evaluate

ckpt = Path(sys.argv[1])
dev_mat_path = Path(sys.argv[2])
fresh_mat_path = Path(sys.argv[3])
out_json = Path(sys.argv[4])
val_spec = json.loads(sys.argv[5])
tmp_dir = fresh_mat_path.parent / 'lock_tmp'
if tmp_dir.exists():
    shutil.rmtree(tmp_dir, ignore_errors=True)

torch.set_num_threads(1)
opts = continuation_parser().parse_args([
    '--checkpoint', str(ckpt), '--output', str(tmp_dir), '--cycles', '1',
    '--world-source', 'text', '--autonomous-events', '--optimizer-policy', 'budgeted',
    '--table-storage', str(tmp_dir / 'table'), '--causal-storage', str(tmp_dir / 'causal'),
])
loaded = prepare_continuation(opts, resume_output=False)
engine = loaded.engine
model = engine.model

# Identify dev relation IDs to exclude completely
dev_mat = torch.load(dev_mat_path, map_location='cpu', weights_only=False)
dev_rel_ids = {int(st.identity) for st, _ in dev_mat}
dev_fact_ids = {str(f.fact_id) for _, f in dev_mat}

bank = model.causal_relations
all_indices = sorted(
    (i for i in range(bank.capacity) if int(bank.identity[i]) > 0 and int(bank.identity[i]) not in dev_rel_ids),
    key=lambda i: int(bank.identity[i])
)
fresh = []
picked_rels = []
for i in all_indices:
    st = bank.state(i)
    facts = tuple(queries(engine, st))
    if not facts:
        continue
    st = replace(st, constraints=facts, evidence_bound=True, extra=(), organization=None, variable_uses=())
    picked_rels.append(int(st.identity))
    for f in facts:
        fresh.append((st, f))
        if len(fresh) == 64:
            break
    if len(fresh) == 64:
        break
assert len(fresh) == 64, f'expected 64 fresh old facts, got {len(fresh)}'
fresh_rel_ids = {int(st.identity) for st, _ in fresh}
assert not (fresh_rel_ids & dev_rel_ids), 'overlap with Stage 1 dev relations!'
torch.save(fresh, fresh_mat_path)

# Evaluate 2845 origin on fresh 8192B text and fresh 64 old facts
mat_text = segments(val_spec)
with torch.no_grad():
    origin_text = score(engine, mat_text)
    origin_facts = evaluate(engine, fresh)

# Verify migration-moment invariance when applying ridge_lr=1e-2 on actual 2845 body
p_r = model.log_relation_ridge
p_t = model.memory.log_temperature
r_before, t_before = p_r.detach().clone(), p_t.detach().clone()
st_r_ids = {k: (id(v) if torch.is_tensor(v) else v) for k, v in engine.optimizer.state[p_r].items()}
st_t_ids = {k: (id(v) if torch.is_tensor(v) else v) for k, v in engine.optimizer.state[p_t].items()}
cfg_copy = dict(loaded.config)
set_scale_learning_rates(engine, cfg_copy, ridge_lr=1e-2)
with torch.no_grad():
    post_mig_facts = evaluate(engine, fresh)
mig_diag = scale_diag(engine)
mig_verified = (
    torch.equal(p_r.detach(), r_before) and
    torch.equal(p_t.detach(), t_before) and
    all((id(engine.optimizer.state[p_r][k]) if torch.is_tensor(engine.optimizer.state[p_r][k]) else engine.optimizer.state[p_r][k]) == v for k, v in st_r_ids.items()) and
    all((id(engine.optimizer.state[p_t][k]) if torch.is_tensor(engine.optimizer.state[p_t][k]) else engine.optimizer.state[p_t][k]) == v for k, v in st_t_ids.items()) and
    abs(post_mig_facts['mean_score'] - origin_facts['mean_score']) == 0.0 and
    mig_diag['relation_ridge_lr'] == 1e-2 and
    mig_diag['memory_temperature_lr'] == engine.optimizer.param_groups[0]['lr']
)
assert mig_verified, 'migration invariance verification failed on 2845 body!'

info = dict(
    origin_checkpoint=str(ckpt),
    origin_cycle=loaded.cycle,
    fresh_old_facts_path=str(fresh_mat_path),
    fresh_old_facts_count=len(fresh),
    fresh_relation_ids=sorted(fresh_rel_ids),
    dev_relation_ids_excluded=sorted(dev_rel_ids),
    fresh_query_pairs=[[int(st.identity), str(f.fact_id)] for st, f in fresh],
    origin_2845_fresh_8192_nll=origin_text,
    origin_2845_fresh_64_facts=dict(mean_score=origin_facts['mean_score'], mean_mse=origin_facts['mean_mse']),
    migration_invariance_verified_on_2845=mig_verified,
    migration_verified_lrs=dict(ridge_lr=mig_diag['relation_ridge_lr'], temperature_lr=mig_diag['memory_temperature_lr']),
)
out_json.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
if hasattr(model, '_causal_archive') and hasattr(model._causal_archive, 'db'):
    model._causal_archive.db.close()
shutil.rmtree(tmp_dir, ignore_errors=True)
'''
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SPOOL.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONIOENCODING='utf-8', CGB_CHECKPOINT_SPOOL=str(SPOOL),
               PYTHONPATH=str(PROJECT))
    tmp_json = OUT_ROOT / '_lock_info.json'
    print('[lock] locking fresh held-out materials + verifying 2845 migration invariance...', flush=True)
    proc = subprocess.run([sys.executable, '-c', code, str(SOURCE), str(DEV_MATERIAL),
                           str(fresh_mat_path), str(tmp_json), json.dumps(val_spec)],
                          cwd=PROJECT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'lock_acceptance_materials failed:\n{proc.stdout}\n{proc.stderr}')
    info = json.loads(tmp_json.read_text(encoding='utf8'))
    tmp_json.unlink(missing_ok=True)
    manifest = dict(
        schema=1,
        locked_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        source_sha256=sha256_file(SOURCE),
        fresh_validation_text=val_spec,
        fresh_old_facts_sha256=sha256_file(fresh_mat_path),
        **info,
    )
    write_json(manifest_path, manifest)
    print(f'[lock] locked! 2845 fresh NLL={manifest["origin_2845_fresh_8192_nll"]["mean_nll"]:.6f}, '
          f'fresh 64 facts={manifest["origin_2845_fresh_64_facts"]["mean_score"]:.6f}, '
          f'migration_invariance={manifest["migration_invariance_verified_on_2845"]}', flush=True)
    return manifest


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
        '--wiener-gate', '--split-hamiltonian-theta', '--learnable-precision-scales',
        '--table-storage', str(temp_dir / 'table'),
        '--world-activations', str(temp_dir / 'world'),
        '--policy-activations', str(temp_dir / 'policy'),
        *extra_flags,
    ]
    log_path = OUT_ROOT / f'{arm_name}.log'
    t0 = time.perf_counter()
    write_json(status_path, dict(state=f'running_{arm_name}', started_at=time.strftime('%Y-%m-%d %H:%M:%S')))
    print(f'[{arm_name}] launching {CYCLES} cycles from 2845...', flush=True)
    with log_path.open('w', encoding='utf8') as logf:
        proc = subprocess.run(cmd, cwd=PROJECT, env=env, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        write_json(status_path, dict(state=f'failed_{arm_name}', returncode=proc.returncode, seconds=elapsed))
        raise RuntimeError(f'{arm_name} failed (code={proc.returncode}); see {log_path}')
    digest = reclaim_temp(arm_name)
    res = json.loads((arm_dir / 'result.json').read_text(encoding='utf8'))
    print(f'[{arm_name}] done in {elapsed:.1f}s (life={res.get("life_seconds", 0):.1f}s, sha={digest[:12]})', flush=True)
    return res


def eval_arm_subprocess(arm_name: str, manifest: dict):
    arm_dir = OUT_ROOT / arm_name
    ckpt = arm_dir / 'body/last.pt.gz'
    eval_out = arm_dir / 'acceptance_audit.json'
    code = r'''
import json, math, os, shutil, sys, time
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
val_spec = json.loads(sys.argv[4])
fresh_mat_path = Path(sys.argv[5])

torch.set_num_threads(1)
tmp_out = work / 'eval_restore_tmp'
if tmp_out.exists():
    shutil.rmtree(tmp_out, ignore_errors=True)
t0 = time.perf_counter()
opts = continuation_parser().parse_args([
    '--checkpoint', str(ckpt), '--output', str(tmp_out), '--cycles', '1',
    '--world-source', 'text', '--autonomous-events', '--optimizer-policy', 'budgeted',
    '--table-storage', str(tmp_out / 'table'), '--causal-storage', str(tmp_out / 'causal'),
])
loaded = prepare_continuation(opts, resume_output=False)
load_s = time.perf_counter() - t0
engine = loaded.engine
model = engine.model
opt = engine.optimizer

mat_text = segments(val_spec)
fresh_facts = torch.load(fresh_mat_path, map_location=engine.device, weights_only=False)
with torch.no_grad():
    sc_text = score(engine, mat_text)
    sc_facts = evaluate(engine, fresh_facts)

sdiag = scale_diag(engine)
p_r = model.log_relation_ridge
p_t = model.memory.log_temperature
st_r = opt.state[p_r]
st_t = opt.state[p_t]
gr_idx = next(i for i, g in enumerate(opt.param_groups) if any(p is p_r for p in g['params']))
gt_idx = next(i for i, g in enumerate(opt.param_groups) if any(p is p_t for p in g['params']))
b1, b2 = opt.param_groups[gr_idx]['betas']
eps = opt.param_groups[gr_idx]['eps']
sr = int(st_r['step'])
mr = float(st_r['exp_avg']) / (1 - b1 ** sr)
vr = float(st_r['exp_avg_sq']) / (1 - b2 ** sr)

audit = dict(
    checkpoint=str(ckpt),
    cycle=loaded.cycle,
    load_seconds=round(load_s, 3),
    fresh_8192_nll=sc_text,
    fresh_64_facts=dict(mean_score=sc_facts['mean_score'], mean_mse=sc_facts['mean_mse']),
    scale_diagnostics=sdiag,
    ridge_group_index=gr_idx,
    ridge_group_lr=float(opt.param_groups[gr_idx]['lr']),
    ridge_group_wd=float(opt.param_groups[gr_idx]['weight_decay']),
    ridge_step=sr,
    ridge_adam_dir=mr / (math.sqrt(vr) + eps),
    temp_group_index=gt_idx,
    temp_group_lr=float(opt.param_groups[gt_idx]['lr']),
    temp_group_wd=float(opt.param_groups[gt_idx]['weight_decay']),
    temp_step=int(st_t['step']),
    param_group_count=len(opt.param_groups),
    wiener_gate_all_groups=all(bool(g.get('wiener_gate', False)) for g in opt.param_groups),
    split_theta=bool(getattr(model.head_coupling, '_split_theta', False)),
    trainable_missing_from_optimizer=loaded.config.get('mother_parameter_audit', {}).get('trainable_missing_from_optimizer', []),
    saved_scale_config=loaded.config.get('learnable_precision_scales'),
    peak_eval_vram_mib=round(torch.cuda.max_memory_reserved() / (1024 ** 2), 1) if torch.cuda.is_available() else 0.0,
)
out_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
if hasattr(model, '_causal_archive') and hasattr(model._causal_archive, 'db'):
    model._causal_archive.db.close()
shutil.rmtree(tmp_out, ignore_errors=True)
'''
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONIOENCODING='utf-8', CGB_CHECKPOINT_SPOOL=str(SPOOL),
               PYTHONPATH=str(PROJECT))
    proc = subprocess.run([sys.executable, '-c', code, str(ckpt), str(arm_dir), str(eval_out),
                           json.dumps(manifest['fresh_validation_text']),
                           manifest['fresh_old_facts_path']],
                          cwd=PROJECT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'eval_arm_subprocess({arm_name}) failed:\n{proc.stdout}\n{proc.stderr}')
    return json.loads(eval_out.read_text(encoding='utf8'))


def summarize_events(arm_name: str):
    arm_dir = OUT_ROOT / arm_name
    rows = [json.loads(l) for l in (arm_dir / 'events.jsonl').read_text(encoding='utf8').splitlines() if l.strip()]
    ext = [r for r in rows if r.get('autonomous_event', {}).get('kind') == 'external']
    res = json.loads((arm_dir / 'result.json').read_text(encoding='utf8'))
    evals = json.loads((arm_dir / 'evaluations.json').read_text(encoding='utf8'))

    ridge_traj = []
    temp_traj = []
    read_use_scores = []
    pre_errs = []
    metal_gs = []
    revisions_committed = 0
    for r in ext:
        ls = r.get('learnable_precision_scales') or {}
        if 'relation_ridge' in ls:
            ridge_traj.append(dict(cycle=r['cycle'], ridge=ls['relation_ridge'],
                                   log_ridge=math.log(ls['relation_ridge']),
                                   gate=ls.get('relation_ridge_gate'), lr=ls.get('relation_ridge_lr')))
        if 'memory_temperature' in ls:
            temp_traj.append(dict(cycle=r['cycle'], temperature=ls['memory_temperature'],
                                  gate=ls.get('memory_temperature_gate'), lr=ls.get('memory_temperature_lr')))
        for ru in (r.get('read_use') or []):
            if 'log_score' in ru:
                read_use_scores.append(float(ru['log_score']))
        pog = r.get('pre_outcome_gain') or {}
        if 'expected' in pog and 'target' in pog:
            pre_errs.append((float(pog['expected']) - float(pog['target'])) ** 2)
        if 'metal_g' in r:
            metal_gs.append(float(r['metal_g']))
        su = r.get('shared_unknown') or {}
        if (su.get('revision') or {}).get('committed'):
            revisions_committed += 1

    log_steps = [ridge_traj[i]['log_ridge'] - ridge_traj[i - 1]['log_ridge'] for i in range(1, len(ridge_traj))]
    peak_vram_bytes = max((r.get('world_activation_storage', {}).get('peak_reserved_bytes', 0) for r in ext), default=0)
    return dict(
        external_cycles=len(ext),
        total_events=len(rows),
        updates=res.get('updates'),
        external_bytes=res.get('external_bytes'),
        life_seconds=res.get('life_seconds'),
        seconds_per_cycle=res.get('life_seconds', 0) / max(len(ext), 1),
        checkpoint_commit_seconds=res.get('checkpoint_commit_seconds'),
        checkpoint_sha256=res.get('checkpoint_sha256'),
        peak_vram_mib=round(peak_vram_bytes / (1024 ** 2), 1) if peak_vram_bytes else None,
        ridge_trajectory=ridge_traj,
        ridge_log_step_stats=dict(
            mean=sum(log_steps) / len(log_steps) if log_steps else 0.0,
            max_abs=max((abs(x) for x in log_steps), default=0.0),
            total_log_delta=(ridge_traj[-1]['log_ridge'] - ridge_traj[0]['log_ridge']) if len(ridge_traj) >= 2 else 0.0,
        ),
        temperature_start=temp_traj[0] if temp_traj else None,
        temperature_end=temp_traj[-1] if temp_traj else None,
        online_read_use_mean_log_score=(sum(read_use_scores) / len(read_use_scores)) if read_use_scores else None,
        online_read_use_count=len(read_use_scores),
        pre_outcome_gain_mse=(sum(pre_errs) / len(pre_errs)) if pre_errs else None,
        mean_metal_g=(sum(metal_gs) / len(metal_gs)) if metal_gs else None,
        revisions_committed=revisions_committed,
        dev_64_old_facts_evals=[dict(external=e['external'], cycle=e['cycle'],
                                     mean_score=e['mean_score'], mean_mse=e['mean_mse']) for e in evals],
    )


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = OUT_ROOT / 'acceptance_status.json'
    t_all = time.perf_counter()

    manifest = lock_acceptance_materials()
    run_arm('ref_ridge_3e4_16c', [], status_path)
    run_arm('cand_ridge_1e2_16c', ['--ridge-lr', '0.01'], status_path)

    write_json(status_path, dict(state='evaluating'))
    print('[eval] scoring both 2861 checkpoints on locked fresh materials + restore audit...', flush=True)
    ref_eval = eval_arm_subprocess('ref_ridge_3e4_16c', manifest)
    cand_eval = eval_arm_subprocess('cand_ridge_1e2_16c', manifest)
    ref_sum = summarize_events('ref_ridge_3e4_16c')
    cand_sum = summarize_events('cand_ridge_1e2_16c')

    o_nll = manifest['origin_2845_fresh_8192_nll']['mean_nll']
    o_fsc = manifest['origin_2845_fresh_64_facts']['mean_score']
    o_fmse = manifest['origin_2845_fresh_64_facts']['mean_mse']

    cand_restore_ok = (
        cand_eval['ridge_group_lr'] == 0.01 and
        cand_eval['ridge_group_wd'] == 0.0 and
        cand_eval['temp_group_lr'] == 0.0003 and
        cand_eval['temp_group_wd'] == 0.0 and
        cand_eval['ridge_group_index'] != cand_eval['temp_group_index'] and
        cand_eval['wiener_gate_all_groups'] and
        cand_eval['split_theta'] and
        cand_eval['trainable_missing_from_optimizer'] == []
    )
    report = dict(
        schema=1,
        source=str(SOURCE),
        source_cycle=2845,
        cycles=CYCLES,
        locked_manifest=manifest,
        comparison=dict(
            origin_2845_fresh_nll=o_nll,
            ref_fresh_nll=ref_eval['fresh_8192_nll']['mean_nll'],
            cand_fresh_nll=cand_eval['fresh_8192_nll']['mean_nll'],
            ref_fresh_nll_improvement=o_nll - ref_eval['fresh_8192_nll']['mean_nll'],
            cand_fresh_nll_improvement=o_nll - cand_eval['fresh_8192_nll']['mean_nll'],
            cand_vs_ref_nll_advantage=ref_eval['fresh_8192_nll']['mean_nll'] - cand_eval['fresh_8192_nll']['mean_nll'],
            origin_2845_fresh_64_facts_score=o_fsc,
            ref_fresh_64_facts_score=ref_eval['fresh_64_facts']['mean_score'],
            cand_fresh_64_facts_score=cand_eval['fresh_64_facts']['mean_score'],
            ref_fresh_64_facts_delta=ref_eval['fresh_64_facts']['mean_score'] - o_fsc,
            cand_fresh_64_facts_delta=cand_eval['fresh_64_facts']['mean_score'] - o_fsc,
            ref_fresh_64_facts_mse_delta=ref_eval['fresh_64_facts']['mean_mse'] - o_fmse,
            cand_fresh_64_facts_mse_delta=cand_eval['fresh_64_facts']['mean_mse'] - o_fmse,
            ref_final_ridge=ref_eval['scale_diagnostics']['relation_ridge'],
            cand_final_ridge=cand_eval['scale_diagnostics']['relation_ridge'],
            ref_final_temperature=ref_eval['scale_diagnostics']['memory_temperature'],
            cand_final_temperature=cand_eval['scale_diagnostics']['memory_temperature'],
            ref_online_read_use_score=ref_sum['online_read_use_mean_log_score'],
            cand_online_read_use_score=cand_sum['online_read_use_mean_log_score'],
            ref_pre_outcome_gain_mse=ref_sum['pre_outcome_gain_mse'],
            cand_pre_outcome_gain_mse=cand_sum['pre_outcome_gain_mse'],
            ref_mean_metal_g=ref_sum['mean_metal_g'],
            cand_mean_metal_g=cand_sum['mean_metal_g'],
            cand_restore_verified=cand_restore_ok,
        ),
        arms=dict(
            ref_ridge_3e4_16c=dict(summary=ref_sum, audit=ref_eval),
            cand_ridge_1e2_16c=dict(summary=cand_sum, audit=cand_eval),
        ),
        total_wall_seconds=round(time.perf_counter() - t_all, 2),
    )
    write_json(REPORT_PATH, report)
    write_json(status_path, dict(state='completed', report=str(REPORT_PATH), total_wall_seconds=report['total_wall_seconds']))
    print(json.dumps(report['comparison'], indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
