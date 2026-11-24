"""Reachable H1-response directions of the ACTUAL old pending relation9.

The carrier and shared-fact response map were already measured and checked.
Here a low-rank input lift selects directions in the eight encoded fact
residuals. It changes the original posterior input, not a fabricated edge state.
The matched intervention has the same residual Frobenius norm and lies in the
kernel of that H1 response map. Both are fixed activation offsets (identity
derivative at the intervention port), followed by unchanged native learning.
"""
from __future__ import annotations

import functools
import gzip
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def prepare(output):
    from scripts.analyze_native_path_composition import load
    from scripts.measure_native_relation_percolation import sha
    report_path = ROOT/'results/published/cgb_native_shared_response_scoped_2401_20261117.json'
    report = json.loads(report_path.read_text(encoding='utf8'))
    array_path = ROOT/report['arrays']['path']
    assert sha(array_path) == report['arrays']['sha256']
    harmonic = np.load(array_path)['harmonic_basis']
    record = Path(report['source']['operations'])
    assert sha(record) == report['source']['operations_sha256']
    assert sha(record.with_name('tensors.bin')) == report['source']['tensors_sha256']
    with gzip.open(record,'rt',encoding='utf8') as f:
        calls = {r['data']['call_id']:r['data'] for r in map(json.loads,f)
                 if r['kind'] == 'relation_signal'}
    modes, arrays = [], {}
    with record.with_name('tensors.bin').open('rb') as handle:
        for cid in (947,950):
            measured = next(c for c in report['calls'] if c['call_id'] == cid)
            parent = calls[cid]
            posterior = next(c for c in calls.values() if c['parent_id'] == cid and c['operation']=='posterior')
            x,y,q = [load(handle,posterior[k]) for k in ('design','residual','query')]
            alpha = q @ x.T @ np.linalg.inv(x @ x.T + posterior['ridge']*np.eye(len(x)))
            mask = np.asarray(measured['mask'],dtype=float)
            edge_basis = harmonic[measured['edge_ids']]
            coefficient = edge_basis.T @ (mask * alpha)
            u,s,v = np.linalg.svd(coefficient,full_matrices=False)
            rank = int((s > s[0]*1e-10).sum())
            projection = v[:rank].T @ v[:rank]
            complement = np.eye(len(x))-projection
            assert np.linalg.norm(coefficient @ complement) < 1e-10
            assert np.linalg.norm(projection @ projection-projection) < 1e-10
            target = -projection @ y
            matched = -complement @ y
            matched *= np.linalg.norm(target)/np.linalg.norm(matched)
            assert abs(np.linalg.norm(target)-np.linalg.norm(matched)) < 1e-10
            modes.append(dict(call_id=cid,rank=rank,transport=measured['transport'],
                relation=measured['relation'],fact_keys=measured['fact_keys'],
                variable_uses=parent['state']['variable_uses'],
                measured_relative_harmonic_after=float(np.linalg.norm(coefficient@(y+target))/np.linalg.norm(coefficient@y)),
                measured_relative_matched_harmonic_change=float(np.linalg.norm(coefficient@matched)/np.linalg.norm(coefficient@y)),
                residual_delta_norm=float(np.linalg.norm(target)),
                target_total_mean_change=float(np.linalg.norm(alpha@target)),
                matched_total_mean_change=float(np.linalg.norm(alpha@matched))))
            for key,value in dict(projection=projection,coefficient=coefficient,mask=mask,
                                  edge_basis=edge_basis).items():
                arrays[f'{key}_{cid}'] = value
    arrays_path = output.with_suffix('.npz')
    np.savez_compressed(arrays_path,**arrays)
    proof = ROOT/'lean/CGBAxioms/reviews/NativePendingModeUse.lean'
    proof.write_text(LEAN,encoding='utf8')
    result = subprocess.run(['lake','env','lean',str(proof.relative_to(ROOT/'lean/CGBAxioms'))],
        cwd=ROOT/'lean/CGBAxioms',capture_output=True,text=True,encoding='utf8',errors='replace')
    output.with_suffix('.lean.log').write_text(result.stdout+result.stderr,encoding='utf8')
    if result.returncode:
        raise RuntimeError('native residual input-lift proof failed')
    result = dict(status='actual_pending_input_lift_checked',modes=modes,
        input='eight original encoded old-fact residual rows; no current target in selection',
        intervention='Y + stop_gradient(delta); identity derivative at residual input',
        arrays=dict(path=str(arrays_path),sha256=sha(arrays_path)),
        proof=dict(path=str(proof),sha256=sha(proof),exit_code=0),
        source=dict(path=str(report_path),sha256=sha(report_path)))
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    return result


LEAN = '''import Mathlib.Tactic
import Mathlib.Analysis.Calculus.FDeriv.Add

namespace CGB.NativePendingModeUse
open Matrix

/-- C is the actual H^T M diag(alpha) map from the eight native fact rows.
    CP=C says P covers the reachable measured mode. It does not assert that
    the rest of the full edge field can be held fixed by this input lift. -/
theorem targeted_residual_erases (C : Matrix (Fin 64) (Fin 8) ℝ)
    (P : Matrix (Fin 8) (Fin 8) ℝ) (Y : Matrix (Fin 8) (Fin 128) ℝ)
    (cover : C * P = C) : C * (Y-P*Y) = 0 := by
  rw [Matrix.mul_sub, ← Matrix.mul_assoc, cover, sub_self]

theorem matched_residual_preserves (C : Matrix (Fin 64) (Fin 8) ℝ)
    (Q : Matrix (Fin 8) (Fin 8) ℝ) (Y : Matrix (Fin 8) (Fin 128) ℝ)
    (zero : C * Q = 0) (a : ℝ) : C * (Y+a • (Q*Y)) = C*Y := by
  simp only [Matrix.mul_add, Matrix.mul_smul, ← Matrix.mul_assoc, zero,
    Matrix.zero_mul, smul_zero, add_zero]

theorem matched_norm {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
    (target control : V) (positive : 0 < ‖control‖) :
    ‖(‖target‖/‖control‖) • control‖ = ‖target‖ := by
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (div_nonneg (norm_nonneg _) (norm_nonneg _))]
  exact div_mul_cancel₀ _ (ne_of_gt positive)

/-- Runtime uses a detached offset; gradients still flow through the old
    fact encoder/other original inputs with the original input Jacobian. -/
theorem fixed_offset_derivative (delta x : (Fin 8 → Fin 128 → ℝ)) :
    HasFDerivAt (fun y => y+delta) (ContinuousLinearMap.id ℝ _) x := by
  simpa using (hasFDerivAt_id x).add_const delta
end CGB.NativePendingModeUse
'''


def install(observer, family, specification):
    import torch
    from scripts.ib_local import causal_relation_program as cp
    arrays = np.load(specification['arrays']['path'])
    contexts, receipts, undo = [], [], []
    old_mechanism,old_posterior = cp.mechanism,cp.posterior
    modes = {m['transport']:m for m in specification['modes']}
    completed = set()

    @functools.wraps(old_mechanism)
    def mechanism(*args,**kwargs):
        state = args[1]
        uses = [[v.identity,v.port.value] for v,version,seen in state.variable_uses]
        mode = modes[kwargs.get('mother') is not None]
        expected = [[v['identity'],v['port']] for v,version,seen in mode['variable_uses']]
        active = ((state.identity,state.generation)==(9,1) and uses==expected and
                  kwargs.get('kind',cp.Port.MOTHER)==cp.Port.DECISION and mode['call_id'] not in completed)
        contexts.append(mode if active else None)
        try:
            return old_mechanism(*args,**kwargs)
        finally:
            contexts.pop()

    @functools.wraps(old_posterior)
    def posterior(x,y,query,ridge,**kwargs):
        mode = contexts[-1] if contexts else None
        if mode is None:
            return old_posterior(x,y,query,ridge,**kwargs)
        cid = mode['call_id']
        assert y.shape == (8,128) and query.shape[0] == 1
        # Recompute only the intervention's measurement coordinates from the
        # actual current pre-intervention query. No world/target is inspected.
        xd,yd,qd = [a.detach().cpu().double().numpy() for a in (x,y,query)]
        alpha = qd @ xd.T @ np.linalg.inv(xd @ xd.T+ridge*np.eye(len(xd)))
        c = arrays[f'edge_basis_{cid}'].T @ (arrays[f'mask_{cid}']*alpha)
        u,s,v = np.linalg.svd(c,full_matrices=False)
        rank = int((s>s[0]*1e-10).sum())
        p = v[:rank].T @ v[:rank]
        q = np.eye(len(xd))-p
        target = -p @ yd
        matched = -q @ yd
        assert np.linalg.norm(matched) > 0
        matched *= np.linalg.norm(target)/np.linalg.norm(matched)
        delta = target if family == 'pending_mode_cut' else matched
        offset = torch.as_tensor(delta,device=y.device,dtype=y.dtype)
        revised = y + offset
        # Forward values change; this input's original backpropagation route
        # remains enabled. The original posterior also recomputes uncertainty.
        result = old_posterior(x,revised,query,ridge,**kwargs)
        original_h = c @ yd
        actual_y = revised.detach().cpu().double().numpy()
        receipt = dict(family=family,original_call_id=cid,event=observer.clock,
            external_cycle=observer.external_cycle,update_clock=observer.update_clock,
            relation=[9,1],transport=mode['transport'],reachable_rank=rank,
            original_harmonic_norm=float(np.linalg.norm(original_h)),
            relative_harmonic_after=float(np.linalg.norm(c@actual_y)/np.linalg.norm(original_h)),
            relative_harmonic_change=float(np.linalg.norm(c@(actual_y-yd))/np.linalg.norm(original_h)),
            delta_frobenius=float(np.linalg.norm(actual_y-yd)),
            expected_matched_delta_frobenius=float(np.linalg.norm(target)),
            total_posterior_mean_change=float(np.linalg.norm(alpha@(actual_y-yd))),
            original_residual_norm=float(np.linalg.norm(yd)),
            all_native_gradients_enabled=bool(revised.requires_grad==y.requires_grad),
            fact_keys=mode['fact_keys'])
        completed.add(cid)
        receipts.append(receipt)
        observer.record('identified_pending_mode_intervention',receipt)
        return result

    for module in list(sys.modules.values()):
        if module is None or not getattr(module,'__name__','').startswith('scripts.ib_local'):
            continue
        for name,old,new in [('mechanism',old_mechanism,mechanism),('posterior',old_posterior,posterior)]:
            if getattr(module,name,None) is old:
                undo.append((module,name,old))
                setattr(module,name,new)
    def close():
        for module,name,old in reversed(undo):
            setattr(module,name,old)
    return receipts,close
